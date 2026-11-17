"""Wake remote embedding workers that sleep when idle, with a Wake-on-LAN magic packet.

A magic packet is six ``0xff`` bytes followed by the target MAC sixteen times, sent
as a UDP broadcast on the target's LAN. It cannot cross Tailscale or a router, so a
workstation off the worker's ``network`` sends it from ``relay``, an SSH host on that
LAN, by running a short Python script there. Wake-on-LAN wakes a suspended host, not
one that was shut down.

The pool wakes a worker whose connection times out (a refused connection means the
host is awake), resends every ``RESEND_INTERVAL`` while it stays down, and waits up
to the endpoint's ``wake.timeout`` before following ``remote_embedding.unavailable``.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

from arcaneum.embeddings.remote.config import EndpointConfig, WakeConfig

logger = logging.getLogger(__name__)

LIMITED_BROADCAST = "255.255.255.255"
# Discard (9) is the usual Wake-on-LAN port; some NICs and firmware listen on echo (7).
WAKE_PORTS = (9, 7)
# UDP can drop a packet; each send repeats it.
REPEATS = 3
RESEND_INTERVAL = 30.0
RELAY_TIMEOUT = 20.0
RELAY_SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]

Target = tuple[str, int]


class WakeError(Exception):
    """A wake packet could not be sent."""


def magic_packet(mac: str) -> bytes:
    return b"\xff" * 6 + bytes.fromhex(mac.replace(":", "")) * 16


def wake_targets(wake: WakeConfig) -> list[Target]:
    """The worker network's broadcast address, if known, then the limited broadcast."""
    addresses = []
    if wake.network:
        addresses.append(str(ipaddress.IPv4Network(wake.network).broadcast_address))
    addresses.append(LIMITED_BROADCAST)
    return [(address, port) for address in addresses for port in WAKE_PORTS]


def send_magic_packet(mac: str, targets: Sequence[Target]) -> None:
    packet = magic_packet(mac)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        for _ in range(REPEATS):
            for target in targets:
                sock.sendto(packet, target)


# Runs on the relay as `python3 - MAC HOST:PORT...`; stdlib only, any Python 3.
RELAY_SCRIPT = """\
import socket, sys
mac = bytes.fromhex(sys.argv[1].replace(":", ""))
packet = b"\\xff" * 6 + mac * 16
targets = [(t.rsplit(":", 1)[0], int(t.rsplit(":", 1)[1])) for t in sys.argv[2:]]
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
for _ in range(%d):
    for t in targets:
        s.sendto(packet, t)
""" % REPEATS


def relay_command(relay: str, mac: str, targets: Sequence[Target]) -> list[str]:
    return [
        "ssh",
        *RELAY_SSH_OPTIONS,
        relay,
        "python3",
        "-",
        mac,
        *(f"{host}:{port}" for host, port in targets),
    ]


def relay_magic_packet(
    relay: str,
    mac: str,
    targets: Sequence[Target],
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    command = relay_command(relay, mac, targets)
    try:
        completed = run(
            command, input=RELAY_SCRIPT, capture_output=True, text=True, timeout=RELAY_TIMEOUT
        )
    except subprocess.TimeoutExpired as exc:
        raise WakeError(f"relay {relay} timed out after {RELAY_TIMEOUT:.0f}s") from exc
    except OSError as exc:
        raise WakeError(f"could not run ssh to relay {relay} ({exc})") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-500:]
        raise WakeError(f"relay {relay} failed (exit {completed.returncode}): {detail}")


def on_network(network: str) -> bool:
    """Whether this machine routes to ``network`` from an address on it (sends nothing)."""
    net = ipaddress.IPv4Network(network)
    probe = next(net.hosts(), net.network_address)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((str(probe), 9))
            source = sock.getsockname()[0]
    except OSError:
        return False
    return ipaddress.ip_address(source) in net


def send_wake(
    wake: WakeConfig,
    *,
    on_lan: Callable[[str], bool] = on_network,
    direct: Callable[[str, Sequence[Target]], None] = send_magic_packet,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> str:
    """Send one wake, directly or through the relay; returns what was done.

    Raises WakeError (or OSError from a direct send) when it could not be sent.
    """
    targets = wake_targets(wake)
    broadcast = targets[0][0]
    local = on_lan(wake.network) if wake.network else None
    if wake.relay and not local:
        relay_magic_packet(wake.relay, wake.mac, targets, run=run)
        return f"sent a magic packet to {wake.mac} via {wake.relay}"
    direct(wake.mac, targets)
    detail = f"sent a magic packet to {wake.mac} on {broadcast}"
    if local is False:
        detail += f" (this machine is not on {wake.network}; set wake.relay if it does not wake)"
    return detail


def host_state(host: str, port: int, timeout: float) -> str:
    """``listening``, ``refused`` (host awake, nothing on the port) or ``silent``."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "listening"
    except ConnectionRefusedError:
        return "refused"
    except OSError:
        return "silent"


# --- Pool side -------------------------------------------------------------------------


@dataclass
class _HostWake:
    sent_at: Optional[float] = None
    deadline: Optional[float] = None
    note: str = "waking it"


class Waker:
    """Rate-limited wakes per host, and whether a woken host is still expected up.

    Keyed by MAC address, so endpoints on one host share a wake.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        send: Callable[[WakeConfig], str] = send_wake,
        resend_interval: float = RESEND_INTERVAL,
    ) -> None:
        self.clock = clock
        self.send = send
        self.resend_interval = resend_interval
        self._hosts: dict[str, _HostWake] = {}
        self._lock = threading.Lock()

    def wake(self, endpoint: EndpointConfig) -> str:
        """Wake the endpoint's host unless a wake went out recently; returns a note."""
        wake = endpoint.wake
        if wake is None:
            return ""
        now = self.clock()
        with self._lock:
            host = self._hosts.setdefault(wake.mac, _HostWake())
            if host.sent_at is not None and now - host.sent_at < self.resend_interval:
                return host.note
            host.sent_at = now
        try:
            detail = self.send(wake)
        except (OSError, WakeError) as exc:
            note = f"wake failed ({exc})"
            logger.warning("Could not wake remote embedding endpoint %s: %s", endpoint.name, exc)
        else:
            note = f"waking it ({detail})"
            logger.info("Waking remote embedding endpoint %s: %s", endpoint.name, detail)
            with self._lock:
                if host.deadline is None:
                    host.deadline = now + wake.timeout
        with self._lock:
            host.note = note
        return note

    def waking(self, endpoint: EndpointConfig) -> bool:
        """Whether the endpoint's host was woken and its wait has not run out."""
        if endpoint.wake is None:
            return False
        with self._lock:
            host = self._hosts.get(endpoint.wake.mac)
            return bool(host and host.deadline is not None and self.clock() < host.deadline)

    def awake(self, endpoint: EndpointConfig) -> None:
        """The endpoint's host answered; a later sleep starts a new wake."""
        if endpoint.wake is not None:
            with self._lock:
                self._hosts.pop(endpoint.wake.mac, None)


# --- arc models wake -------------------------------------------------------------------


def wake_endpoints(
    endpoints: Sequence[EndpointConfig],
    *,
    explicit: bool,
    wait: bool = True,
    dry_run: bool = False,
    timeout: Optional[float] = None,
    connect_timeout: float = 2.0,
    out: Callable[[str], None] = print,
    send: Callable[[WakeConfig], str] = send_wake,
    state: Callable[[str, int, float], str] = host_state,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll: float = 2.0,
) -> bool:
    """Wake each sleeping endpoint and, unless ``wait`` is false, wait until it answers.

    ``explicit`` means the endpoints were named, so one without wake settings fails
    rather than being skipped. Returns False if any endpoint failed.
    """
    ok = True
    pending: dict[str, tuple[EndpointConfig, float, float]] = {}  # name -> (ep, start, sent)

    def try_send(endpoint: EndpointConfig) -> Optional[str]:
        try:
            return send(endpoint.wake)
        except (OSError, WakeError) as exc:
            out(f"✗ {endpoint.name}: could not send a wake packet: {exc}")
            return None

    for endpoint in endpoints:
        if endpoint.wake is None:
            if explicit:
                ok = False
                out(
                    f"✗ {endpoint.name}: no wake settings; add wake.mac to this endpoint "
                    "under remote_embedding.endpoints"
                )
            else:
                out(f"- {endpoint.name}: no wake settings; skipped")
            continue
        current = state(endpoint.host, endpoint.port, connect_timeout)
        if current == "listening":
            out(f"✓ {endpoint.name}: already awake")
            continue
        if current == "refused":
            out(_awake_without_worker(endpoint))
            continue
        if dry_run:
            out(f"{endpoint.name}: would send a magic packet to {endpoint.wake.mac}")
            continue
        detail = try_send(endpoint)
        if detail is None:
            ok = False
            continue
        if not wait:
            out(f"✓ {endpoint.name}: {detail}")
            continue
        out(f"  {endpoint.name}: {detail}; waiting for it to answer")
        now = clock()
        pending[endpoint.name] = (endpoint, now, now)

    while pending:
        sleep(poll)
        now = clock()
        for name, (endpoint, started, sent) in list(pending.items()):
            current = state(endpoint.host, endpoint.port, connect_timeout)
            if current != "silent":
                del pending[name]
                if current == "listening":
                    out(f"✓ {name}: awake after {now - started:.0f}s")
                else:
                    out(_awake_without_worker(endpoint))
                continue
            limit = timeout if timeout is not None else endpoint.wake.timeout
            if now - started >= limit:
                del pending[name]
                ok = False
                out(
                    f"✗ {name}: did not answer within {limit:.0f}s; Wake-on-LAN wakes a "
                    "suspended host, not one that is shut down or unplugged"
                )
            elif now - sent >= RESEND_INTERVAL and try_send(endpoint) is not None:
                pending[name] = (endpoint, started, now)
    return ok


def _awake_without_worker(endpoint: EndpointConfig) -> str:
    return (
        f"✓ {endpoint.name}: awake, but no worker listens on port {endpoint.port} "
        "(`arc models start`, or `arc models deploy` if it has none)"
    )

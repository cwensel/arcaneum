"""Client pool that spreads document embedding across remote workers (RDR-025).

A pool opens one connection per endpoint for each model it routes there, with one
request in flight per connection. An ``embed()`` call is split into sub-batches no
larger than each endpoint's encode batch size, spread across the endpoints that
hold the model or have no work, and reassembled in order. An endpoint that errors,
disconnects, stops answering heartbeats or exceeds a deadline is marked down, its
sub-batch is retried elsewhere, and it is probed in the background until it rejoins.
"""

from __future__ import annotations

import ipaddress
import logging
import math
import queue
import socket
import struct
import sys
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, answer_challenge, deliver_challenge
from typing import Any, Optional

import numpy as np

from arcaneum.embeddings.client import EMBEDDING_MODELS
from arcaneum.embeddings.remote.config import EndpointConfig, RemoteEmbeddingConfig
from arcaneum.embeddings.remote.errors import (
    RemoteEmbeddingBatchError,
    RemoteEmbeddingFallback,
    RemoteEmbeddingUnavailable,
)
from arcaneum.embeddings.remote.identity import identity_mismatches, local_identity
from arcaneum.embeddings.remote.server import TAILSCALE_NETWORK, default_encode_batch_size
from arcaneum.embeddings.worker_protocol import (
    ConnectionChannel,
    MessageType,
    WorkerProtocolError,
    decode_embeddings,
    make_message,
)

logger = logging.getLogger(__name__)

NOTIFY_INTERVAL = 60.0

__all__ = ["RemoteEmbeddingFallback", "RemoteEmbeddingUnavailable", "RemoteEmbeddingPool"]


class EndpointFailure(Exception):
    """One endpoint could not complete a request; the pool retries elsewhere."""


class BatchFailure(Exception):
    """The endpoint answered, but this batch failed (encode error or invalid vectors)."""


# Attempts per sub-batch for batch failures before the embed call fails; a
# deterministic error must not take endpoints down or loop under `wait`.
BATCH_ATTEMPTS = 2


# --- Connections -------------------------------------------------------------------


def _set_receive_timeout(fd: int, seconds: float) -> None:
    """Set SO_RCVTIMEO on a socket descriptor without taking ownership of it."""
    whole = int(seconds)
    value = struct.pack("ll", whole, int((seconds - whole) * 1_000_000))
    sock = socket.socket(fileno=fd)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, value)
    finally:
        sock.detach()


def connect(host: str, port: int, authkey: Optional[bytes], timeout: float) -> ConnectionChannel:
    """Open an authenticated protocol connection with bounded connect and auth time."""
    sock = socket.create_connection((host, port), timeout=timeout)
    try:
        sock.setblocking(True)
        # Bound the authentication exchange: a wedged server can accept at the
        # kernel level and then never answer.
        _set_receive_timeout(sock.fileno(), timeout)
        connection = Connection(sock.detach())
    except BaseException:
        sock.close()
        raise
    try:
        if authkey:
            try:
                answer_challenge(connection, authkey)
                deliver_challenge(connection, authkey)
            except Exception as exc:
                raise ConnectionError(
                    f"authentication failed ({exc or type(exc).__name__}); check that "
                    "both ends use the same authkey"
                ) from exc
    except BaseException:
        connection.close()
        raise
    # Reads poll before receiving; the receive timeout stays set so a reply that
    # stalls partway through raises instead of blocking past every deadline.
    return ConnectionChannel(connection)


def tailscale_resolution_problem(endpoint: EndpointConfig) -> Optional[str]:
    """Explain a ``bind: tailscale`` endpoint whose name resolves outside the tailnet."""
    if endpoint.bind != "tailscale":
        return None
    try:
        infos = socket.getaddrinfo(endpoint.host, endpoint.port, type=socket.SOCK_STREAM)
    except OSError as exc:
        return (
            f"{endpoint.name} binds its Tailscale address, but {endpoint.host} does not "
            f"resolve on this workstation ({exc}); check Tailscale and MagicDNS here"
        )
    addresses = [info[4][0] for info in infos if info[0] == socket.AF_INET]
    outside = [a for a in addresses if ipaddress.ip_address(a) not in TAILSCALE_NETWORK]
    if addresses and not outside:
        return None
    resolved = ", ".join(outside or [info[4][0] for info in infos]) or "nothing"
    return (
        f"{endpoint.name} binds its Tailscale address, but {endpoint.host} resolves to "
        f"{resolved} on this workstation, outside the tailnet (100.64.0.0/10); check that "
        f"Tailscale is running here and MagicDNS resolves {endpoint.host}"
    )


def _unreachable_reason(endpoint: EndpointConfig, exc: BaseException) -> str:
    return tailscale_resolution_problem(endpoint) or f"unreachable ({exc})"


def _short(value: Any) -> str:
    text = str(value)
    return f"{text[:12]}…" if len(text) > 16 else text


def describe_mismatch(local: dict[str, Any], remote: dict[str, Any], fields: list[str]) -> str:
    parts = [
        f"{name} local {_short(local.get(name))} remote {_short(remote.get(name))}"
        for name in fields
    ]
    if "arcaneum_version" not in fields and "code_fingerprint" in fields:
        parts.insert(
            0,
            f"versions local {local.get('arcaneum_version')} "
            f"remote {remote.get('arcaneum_version')}",
        )
    return "version mismatch: " + "; ".join(parts)


@dataclass
class LinkStatus:
    endpoint: str
    model: str
    state: str  # up | down | excluded | untried
    reason: Optional[str] = None


class _Link:
    """One connection from this client to one endpoint for one model."""

    def __init__(self, pool: "RemoteEmbeddingPool", endpoint: EndpointConfig, model: str) -> None:
        self.pool = pool
        self.endpoint = endpoint
        self.model = model
        self.channel: Optional[ConnectionChannel] = None
        self.state = "untried"
        self.reason: Optional[str] = None
        self.batch_size = default_encode_batch_size(model)
        self.next_probe = 0.0
        self._probing = False
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.endpoint.name

    @property
    def up(self) -> bool:
        return self.state == "up"

    def status(self) -> LinkStatus:
        return LinkStatus(self.name, self.model, self.state, self.reason)

    # Connection management -------------------------------------------------------

    def connect(self) -> bool:
        """Connect and handshake; return whether the link is usable."""
        config = self.pool.config
        self.close()
        if self.pool.closed:
            return False
        try:
            channel = self.pool.connector(
                self.endpoint.host,
                self.endpoint.port,
                config.authkey_bytes,
                config.deadlines.connect_timeout,
            )
        except Exception as exc:
            self._mark("down", _unreachable_reason(self.endpoint, exc))
            return False
        local = local_identity(self.model)
        try:
            request_id = uuid.uuid4().hex
            channel.send(make_message(MessageType.INITIALIZE, request_id, **local))
            reply = channel.receive(timeout=config.deadlines.connect_timeout)
        except (EOFError, OSError, WorkerProtocolError) as exc:
            channel.close()
            self._mark("down", f"handshake failed ({exc})")
            return False
        if reply is None or reply["type"] != MessageType.INITIALIZED:
            channel.close()
            self._mark("down", "handshake failed (no initialized reply)")
            return False
        payload = reply["payload"]
        mismatches = identity_mismatches(local, payload)
        if mismatches:
            channel.close()
            self._mark("excluded", describe_mismatch(local, payload, mismatches))
            return False
        if not payload.get("serves"):
            channel.close()
            self._mark("excluded", f"does not serve {self.model}")
            return False
        self.batch_size = max(1, int(payload.get("encode_batch_size") or self.batch_size))
        configured = self.endpoint.batch_size.get(self.model)
        if configured:
            self.batch_size = min(self.batch_size, configured)
        self.channel = channel
        self.state, self.reason = "up", None
        return True

    def _mark(self, state: str, reason: str) -> None:
        self.state, self.reason = state, reason
        self.next_probe = self.pool.clock() + self.pool.config.probe_interval

    def fail(self, reason: str) -> None:
        logger.warning("Remote embedding endpoint %s (%s): %s", self.name, self.model, reason)
        self.close()
        self._mark("down", reason)

    def close(self) -> None:
        if self.channel is not None:
            self.channel.close()
            self.channel = None

    def probe_if_due(self) -> None:
        """Reconnect a down or excluded link in the background once its probe is due."""
        if self.up or self.state == "untried":
            return
        with self._lock:
            if self._probing or self.pool.clock() < self.next_probe:
                return
            self._probing = True

        def probe() -> None:
            try:
                if self.connect():
                    logger.info(
                        "Remote embedding endpoint %s rejoined for %s", self.name, self.model
                    )
            finally:
                with self._lock:
                    self._probing = False

        threading.Thread(target=probe, name=f"arc-embed-probe-{self.name}", daemon=True).start()

    # Requests ------------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Heartbeat on an idle link; raises EndpointFailure."""
        channel = self._channel()
        request_id = uuid.uuid4().hex
        deadline = self.pool.clock() + self.pool.config.deadlines.heartbeat_timeout
        try:
            channel.send(make_message(MessageType.HEARTBEAT, request_id))
            while True:
                remaining = deadline - self.pool.clock()
                if remaining <= 0:
                    raise EndpointFailure("stopped answering heartbeats")
                reply = channel.receive(timeout=remaining)
                if reply is not None and reply["request_id"] == request_id:
                    return reply["payload"]
        except (EOFError, OSError, WorkerProtocolError) as exc:
            raise EndpointFailure(f"disconnected ({exc})") from exc

    def _channel(self) -> ConnectionChannel:
        channel = self.channel
        if channel is None or self.pool.closed:
            raise EndpointFailure("not connected")
        return channel

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode on this endpoint, enforcing per-state deadlines; raises EndpointFailure."""
        channel = self._channel()
        pool = self.pool
        deadlines = pool.config.deadlines
        clock = pool.clock
        allowances = {
            "encoding": deadlines.encode_allowance(self.model, len(texts)),
            "loading": deadlines.loading,
            "queued": pool.queued_allowance(),
        }
        # Until a heartbeat reports the request's state, allow at least one heartbeat.
        allowances["submitted"] = max(allowances["encoding"], deadlines.heartbeat_timeout)
        state, state_since = "submitted", clock()
        heartbeat_id: Optional[str] = None
        heartbeat_sent = last_heartbeat = state_since
        request_id = uuid.uuid4().hex
        try:
            channel.send(make_message(MessageType.ENCODE, request_id, texts=texts))
            while True:
                now = clock()
                if heartbeat_id is None and now - last_heartbeat >= deadlines.heartbeat_interval:
                    heartbeat_id = uuid.uuid4().hex
                    channel.send(make_message(MessageType.HEARTBEAT, heartbeat_id))
                    heartbeat_sent = last_heartbeat = now
                if heartbeat_id is not None and now - heartbeat_sent > deadlines.heartbeat_timeout:
                    raise EndpointFailure("stopped answering heartbeats")
                allowance = allowances[state]
                if now - state_since > allowance:
                    raise EndpointFailure(f"exceeded its {state} deadline ({allowance:.1f}s)")
                wait = min(
                    deadlines.heartbeat_interval,
                    max(0.0, state_since + allowance - now),
                )
                reply = channel.receive(timeout=max(wait, 0.001))
                if reply is None:
                    continue
                if reply["request_id"] == heartbeat_id:
                    heartbeat_id = None
                    reported = reply["payload"].get("state")
                    if reported in allowances and reported != state:
                        state, state_since = reported, clock()
                elif reply["request_id"] == request_id:
                    return self._result(reply, len(texts))
        except (EOFError, OSError) as exc:
            raise EndpointFailure(f"disconnected ({exc or type(exc).__name__})") from exc
        except WorkerProtocolError as exc:
            raise EndpointFailure(f"protocol error ({exc})") from exc

    def _result(self, reply: dict[str, Any], count: int) -> np.ndarray:
        payload = reply["payload"]
        if reply["type"] == MessageType.ERROR:
            message = f"{payload.get('code')}: {payload.get('message')}"
            if payload.get("code") == "backend_error":
                raise BatchFailure(message)
            raise EndpointFailure(message)
        if reply["type"] != MessageType.ENCODED:
            raise EndpointFailure(f"unexpected reply {reply['type']!r}")
        result = decode_embeddings(payload)
        dimensions = EMBEDDING_MODELS[self.model]["dimensions"]
        if result.shape != (count, dimensions):
            raise EndpointFailure(f"returned shape {result.shape}, expected {(count, dimensions)}")
        return result


# --- Pool --------------------------------------------------------------------------


def _stderr_notify(message: str) -> None:
    print(f"   {message}", file=sys.stderr, flush=True)


class RemoteEmbeddingPool:
    def __init__(
        self,
        config: RemoteEmbeddingConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        connector: Callable[..., ConnectionChannel] = connect,
        notify: Callable[[str], None] = _stderr_notify,
    ) -> None:
        self.config = config
        self.clock = clock
        self.sleep = sleep
        self.connector = connector
        self.notify = notify
        self._links: dict[tuple[str, str], _Link] = {}
        self._lock = threading.Lock()
        self._last_notice = -math.inf
        self.closed = False

    def queued_allowance(self) -> float:
        """Bound the turn rules guarantee: one turn plus one encode and one load."""
        deadlines = self.config.deadlines
        encodes = [
            deadlines.encode_allowance(model, default_encode_batch_size(model))
            for model in self.config.routed_models()
        ]
        return (
            self.config.turn_length
            + self.config.idle_grace
            + deadlines.loading
            + max(encodes, default=0.0)
        )

    def _links_for(self, model: str) -> list[_Link]:
        links = []
        for endpoint in self.config.endpoints:
            if model not in self.config.endpoint_models(endpoint):
                continue
            key = (endpoint.name, model)
            if key not in self._links:
                self._links[key] = _Link(self, endpoint, model)
            links.append(self._links[key])
        return links

    def link_status(self, model: str) -> list[LinkStatus]:
        with self._lock:
            return [link.status() for link in self._links_for(model)]

    def embed(
        self,
        texts: list[str],
        model: str,
        validate: Optional[Callable[[np.ndarray, int], bool]] = None,
    ) -> np.ndarray:
        """Embed document texts remotely; rows are returned in input order."""
        dimensions = EMBEDDING_MODELS[model]["dimensions"]
        if not texts:
            return np.zeros((0, dimensions), dtype=np.float32)
        # One embed at a time: each connection carries one request in flight.
        with self._lock:
            links = self._links_for(model)
            results: list[Optional[np.ndarray]] = [None] * len(texts)
            pending = list(range(len(texts)))
            while pending:
                if self.closed:
                    raise RemoteEmbeddingUnavailable("remote embedding pool is closed")
                usable = self._usable(links)
                if not usable:
                    self._unavailable(model, links)
                    continue
                chosen = self._prefer(usable, model)
                if not chosen:
                    continue
                pending = self._dispatch(chosen, texts, pending, results, validate)
            return np.ascontiguousarray(np.vstack(results), dtype=np.float32)

    def _usable(self, links: list[_Link]) -> list[_Link]:
        untried = [link for link in links if link.state == "untried"]
        if untried:
            # First use: connect to every endpoint in parallel before choosing.
            with ThreadPoolExecutor(max_workers=len(untried)) as executor:
                list(executor.map(lambda link: link.connect(), untried))
        for link in links:
            link.probe_if_due()
        return [link for link in links if link.up]

    def _unavailable(self, model: str, links: list[_Link]) -> None:
        details = "; ".join(f"{link.name} {link.reason or link.state}" for link in links) or (
            "no endpoint is configured to serve it"
        )
        policy = self.config.unavailable
        if policy == "local":
            logger.warning(
                "No remote endpoint can serve %s (%s); embedding locally", model, details
            )
            raise RemoteEmbeddingFallback(details)
        hint = "Run `arc models deploy` to update the workers"
        local = (
            "rerun with --local-embed (sync, repair) or with ARC_EMBED_ENDPOINTS= " "(any command)"
        )
        if policy == "fail":
            raise RemoteEmbeddingUnavailable(
                f"No remote embedding endpoint can serve {model}: {details}. {hint}, "
                f"or {local} to embed locally."
            )
        now = self.clock()
        if now - self._last_notice >= NOTIFY_INTERVAL:
            self._last_notice = now
            self.notify(
                f"Waiting for remote embedding endpoints for {model}: {details}. {hint}, "
                f"or {local} to proceed without them."
            )
        self.sleep(min(self.config.probe_interval, 1.0))

    def _prefer(self, usable: list[_Link], model: str) -> list[_Link]:
        """Endpoints holding the model or without work; else one endpoint to queue at.

        Eligible endpoints are ordered so a round that needs fewer endpoints than are
        eligible uses those holding the model first, then empty ones, and switches an
        idle endpoint away from another model only when it needs it.
        """
        holders, empty, idle_other, busy_other = [], [], [], []

        def check(link: _Link) -> Optional[dict[str, Any]]:
            try:
                return link.health()
            except EndpointFailure as exc:
                link.fail(str(exc))
                return None

        with ThreadPoolExecutor(max_workers=max(1, len(usable))) as executor:
            healths = list(executor.map(check, usable))
        for link, health in zip(usable, healths):
            if health is None:
                continue
            resident = health.get("resident_model") or health.get("loading_model")
            if resident == model:
                holders.append(link)
            elif health.get("busy"):
                busy_other.append(link)
            elif resident is None:
                empty.append(link)
            else:
                idle_other.append(link)
        return holders + empty + idle_other or busy_other[:1]

    def _dispatch(
        self,
        chosen: list[_Link],
        texts: list[str],
        pending: list[int],
        results: list[Optional[np.ndarray]],
        validate: Optional[Callable[[np.ndarray, int], bool]],
    ) -> list[int]:
        """Run one round over the chosen links; return the indices still pending."""
        size = min(math.ceil(len(pending) / len(chosen)), min(link.batch_size for link in chosen))
        work: queue.Queue[list[int]] = queue.Queue()
        for start in range(0, len(pending), size):
            work.put(pending[start : start + size])
        # Fewer sub-batches than endpoints: use the most preferred endpoints only.
        chosen = chosen[: work.qsize()]
        attempts: dict[tuple[int, ...], int] = {}
        failed: list[str] = []
        lock = threading.Lock()

        def run(link: _Link) -> None:
            while not failed:
                try:
                    batch = work.get_nowait()
                except queue.Empty:
                    return
                try:
                    rows = link.encode([texts[i] for i in batch])
                    if validate is not None and not validate(rows, len(batch)):
                        raise BatchFailure("returned invalid embeddings")
                except EndpointFailure as exc:
                    link.fail(str(exc))
                    work.put(batch)
                    return
                except BatchFailure as exc:
                    # The endpoint is healthy; retry the batch, then give up on it.
                    with lock:
                        key = tuple(batch)
                        attempts[key] = attempts.get(key, 0) + 1
                        if attempts[key] >= BATCH_ATTEMPTS:
                            failed.append(f"{link.name}: {exc}")
                            return
                    logger.warning("Retrying a batch that failed on %s: %s", link.name, exc)
                    work.put(batch)
                    continue
                for offset, index in enumerate(batch):
                    results[index] = rows[offset : offset + 1]

        with ThreadPoolExecutor(max_workers=len(chosen)) as executor:
            list(executor.map(run, chosen))
        if failed:
            raise RemoteEmbeddingBatchError(
                f"Remote embedding of {len(pending)} texts failed after "
                f"{BATCH_ATTEMPTS} attempts ({failed[0]})"
            )
        return [index for index in pending if results[index] is None]

    def close(self) -> None:
        """Close every connection; an embed in progress ends with RemoteEmbeddingUnavailable."""
        self.closed = True
        for link in list(self._links.values()):
            link.close()


# --- Status probes -----------------------------------------------------------------


@dataclass
class EndpointStatus:
    endpoint: str
    state: str  # up | mismatch | unreachable
    reason: Optional[str] = None
    arcaneum_version: Optional[str] = None
    code_fingerprint: Optional[str] = None
    resident_model: Optional[str] = None
    loading_model: Optional[str] = None
    busy: Optional[bool] = None
    dependencies: dict[str, Optional[str]] = field(default_factory=dict)
    mismatches: list[str] = field(default_factory=list)


def probe_endpoint(
    endpoint: EndpointConfig,
    config: RemoteEmbeddingConfig,
    *,
    connector: Callable[..., ConnectionChannel] = connect,
) -> EndpointStatus:
    """Open a status-only connection and report the endpoint's health; never encodes."""
    timeout = config.deadlines.connect_timeout
    try:
        channel = connector(endpoint.host, endpoint.port, config.authkey_bytes, timeout)
    except Exception as exc:
        return EndpointStatus(endpoint.name, "unreachable", _unreachable_reason(endpoint, exc))
    try:
        local = local_identity(None)
        channel.send(make_message(MessageType.INITIALIZE, uuid.uuid4().hex, **local))
        initialized = channel.receive(timeout=timeout)
        heartbeat_id = uuid.uuid4().hex
        channel.send(make_message(MessageType.HEARTBEAT, heartbeat_id))
        health = None
        deadline = time.monotonic() + config.deadlines.heartbeat_timeout
        while health is None and time.monotonic() < deadline:
            reply = channel.receive(timeout=max(0.0, deadline - time.monotonic()))
            if reply is not None and reply["request_id"] == heartbeat_id:
                health = reply["payload"]
    except (EOFError, OSError, WorkerProtocolError) as exc:
        return EndpointStatus(endpoint.name, "unreachable", f"handshake failed ({exc})")
    finally:
        channel.close()
    if initialized is None or health is None:
        return EndpointStatus(endpoint.name, "unreachable", "not answering")
    remote = initialized["payload"]
    mismatches = identity_mismatches(local, remote)
    return EndpointStatus(
        endpoint.name,
        "mismatch" if mismatches else "up",
        describe_mismatch(local, remote, mismatches) if mismatches else None,
        arcaneum_version=health.get("arcaneum_version"),
        code_fingerprint=health.get("code_fingerprint"),
        resident_model=health.get("resident_model"),
        loading_model=health.get("loading_model"),
        busy=health.get("busy"),
        dependencies=health.get("dependencies") or {},
        mismatches=mismatches,
    )

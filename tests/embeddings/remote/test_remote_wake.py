"""Waking sleeping remote embedding workers with a Wake-on-LAN magic packet."""

from __future__ import annotations

import socket
import subprocess
import sys

import pytest

from arcaneum.embeddings.remote.config import (
    EndpointConfig,
    RemoteEmbeddingConfigError,
    WakeConfig,
    load_remote_embedding_config,
)
from arcaneum.embeddings.remote.wake import (
    RELAY_SCRIPT,
    WakeError,
    Waker,
    host_state,
    magic_packet,
    on_network,
    send_magic_packet,
    send_wake,
    wake_endpoints,
    wake_targets,
)

MAC = "10:dd:b1:c9:7d:22"
MAC_BYTES = bytes.fromhex("10ddb1c97d22")


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def udp_receiver() -> socket.socket:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(5)
    return receiver


# --- Configuration -----------------------------------------------------------------


def test_endpoints_have_no_wake_settings_by_default():
    assert EndpointConfig(host="dev-0").wake is None


def test_wake_settings_are_loaded_from_the_endpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_EMBED_ENDPOINTS", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(
        "remote_embedding:\n"
        "  endpoints:\n"
        "    - host: dev-0\n"
        "      wake:\n"
        "        mac: 10-DD-B1-C9-7D-22\n"
        "        network: 10.0.1.0/24\n"
        "        relay: laptop-1\n"
    )

    wake = load_remote_embedding_config(path).endpoints[0].wake

    assert wake.mac == MAC
    assert wake.network == "10.0.1.0/24"
    assert wake.relay == "laptop-1"
    assert wake.timeout == 120.0


@pytest.mark.parametrize("mac", ["10:dd:b1:c9:7d", "10:dd:b1:c9:7d:zz", "not a mac", ""])
def test_invalid_mac_is_a_configuration_error(tmp_path, mac):
    path = tmp_path / "config.yaml"
    path.write_text(
        f"remote_embedding:\n  endpoints:\n    - host: dev-0\n      wake: {{mac: '{mac}'}}\n"
    )

    with pytest.raises(RemoteEmbeddingConfigError, match="mac"):
        load_remote_embedding_config(path)


def test_invalid_network_is_rejected():
    with pytest.raises(ValueError, match="network"):
        WakeConfig(mac=MAC, network="10.0.1.0/33")


def test_network_is_normalized_to_its_prefix():
    assert WakeConfig(mac=MAC, network="10.0.1.7/24").network == "10.0.1.0/24"


# --- Packets -----------------------------------------------------------------------


def test_magic_packet_is_six_ff_bytes_then_the_mac_sixteen_times():
    packet = magic_packet(MAC)

    assert packet == b"\xff" * 6 + MAC_BYTES * 16
    assert len(packet) == 102


def test_targets_are_the_limited_broadcast_on_ports_9_and_7_without_a_network():
    assert wake_targets(WakeConfig(mac=MAC)) == [("255.255.255.255", 9), ("255.255.255.255", 7)]


def test_targets_put_the_network_broadcast_first():
    assert wake_targets(WakeConfig(mac=MAC, network="10.0.1.0/24")) == [
        ("10.0.1.255", 9),
        ("10.0.1.255", 7),
        ("255.255.255.255", 9),
        ("255.255.255.255", 7),
    ]


def test_send_magic_packet_delivers_the_packet_to_every_target():
    receiver = udp_receiver()
    try:
        send_magic_packet(MAC, [receiver.getsockname()])
        packets = [receiver.recv(200) for _ in range(3)]
    finally:
        receiver.close()

    assert packets == [magic_packet(MAC)] * 3


def test_relay_script_sends_the_packet_from_the_relay_host():
    receiver = udp_receiver()
    try:
        host, port = receiver.getsockname()
        subprocess.run(
            [sys.executable, "-I", "-", MAC, f"{host}:{port}"],
            input=RELAY_SCRIPT,
            text=True,
            check=True,
            timeout=30,
        )
        packet = receiver.recv(200)
    finally:
        receiver.close()

    assert packet == magic_packet(MAC)


# --- Direct or relayed -------------------------------------------------------------


def test_on_network_is_true_for_a_network_this_machine_routes_from():
    assert on_network("127.0.0.0/8")


def test_on_network_is_false_for_a_network_this_machine_is_not_on():
    assert not on_network("192.0.2.0/24")


def test_send_wake_broadcasts_directly_on_the_workers_network():
    sent = []
    wake = WakeConfig(mac=MAC, network="10.0.1.0/24", relay="laptop-1")

    detail = send_wake(
        wake,
        on_lan=lambda network: True,
        direct=lambda mac, targets: sent.append((mac, targets)),
        run=lambda *a, **k: pytest.fail("must not relay"),
    )

    assert sent == [(MAC, wake_targets(wake))]
    assert MAC in detail and "10.0.1.255" in detail


def test_send_wake_relays_over_ssh_off_the_workers_network():
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    wake = WakeConfig(mac=MAC, network="10.0.1.0/24", relay="laptop-1")
    detail = send_wake(wake, on_lan=lambda network: False, direct=pytest.fail, run=run)

    command, kwargs = calls[0]
    assert command[0] == "ssh" and "BatchMode=yes" in command
    assert command[command.index("laptop-1") :] == [
        "laptop-1",
        "python3",
        "-",
        MAC,
        "10.0.1.255:9",
        "10.0.1.255:7",
        "255.255.255.255:9",
        "255.255.255.255:7",
    ]
    assert kwargs["input"] == RELAY_SCRIPT
    assert "via laptop-1" in detail


def test_send_wake_relays_when_no_network_says_where_the_worker_is():
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    send_wake(
        WakeConfig(mac=MAC, relay="laptop-1"),
        on_lan=lambda network: pytest.fail("no network to check"),
        direct=pytest.fail,
        run=run,
    )

    assert calls and "laptop-1" in calls[0]


def test_send_wake_without_a_relay_broadcasts_directly_even_off_the_network():
    sent = []

    detail = send_wake(
        WakeConfig(mac=MAC, network="10.0.1.0/24"),
        on_lan=lambda network: False,
        direct=lambda mac, targets: sent.append(mac),
    )

    assert sent == [MAC]
    assert "not on 10.0.1.0/24" in detail


def test_a_failed_relay_raises_wake_error_with_its_output():
    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 255, "", "ssh: connect: no route to host")

    with pytest.raises(WakeError, match="no route to host"):
        send_wake(WakeConfig(mac=MAC, relay="laptop-1"), run=run)


def test_a_relay_that_hangs_raises_wake_error():
    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))

    with pytest.raises(WakeError, match="timed out"):
        send_wake(WakeConfig(mac=MAC, relay="laptop-1"), run=run)


# --- Host state --------------------------------------------------------------------


def test_host_state_reports_a_listening_port():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        assert host_state("127.0.0.1", server.getsockname()[1], 1.0) == "listening"


def test_host_state_reports_a_refused_port_as_awake_without_a_worker():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    assert host_state("127.0.0.1", port, 1.0) == "refused"


def test_host_state_reports_an_unanswering_host_as_silent():
    assert host_state("192.0.2.1", 7300, 0.2) == "silent"


# --- Waker (pool side) -------------------------------------------------------------


def endpoint(host="dev-0", mac=MAC, timeout=120.0):
    return EndpointConfig(host=host, wake=WakeConfig(mac=mac, timeout=timeout))


def test_waker_sends_once_per_resend_interval_and_waits_out_the_timeout():
    clock, sent = FakeClock(), []
    waker = Waker(clock=clock, send=lambda wake: sent.append(wake.mac) or "sent")
    worker = endpoint(timeout=60.0)

    assert not waker.waking(worker)
    assert "sent" in waker.wake(worker)
    assert waker.waking(worker)
    clock.now += 10
    waker.wake(worker)
    assert sent == [MAC]
    clock.now += 25
    waker.wake(worker)
    assert sent == [MAC, MAC]
    # The wait runs from the first packet; later packets don't extend it.
    clock.now += 30
    assert not waker.waking(worker)


def test_waker_shares_a_host_between_its_endpoints():
    sent = []
    waker = Waker(clock=FakeClock(), send=lambda wake: sent.append(wake.mac) or "sent")

    waker.wake(EndpointConfig(host="dev-0", port=7300, wake=WakeConfig(mac=MAC)))
    waker.wake(EndpointConfig(host="dev-0", port=7301, wake=WakeConfig(mac=MAC)))

    assert sent == [MAC]


def test_awake_ends_the_wait_and_a_later_sleep_is_woken_again():
    clock, sent = FakeClock(), []
    waker = Waker(clock=clock, send=lambda wake: sent.append(wake.mac) or "sent")
    worker = endpoint()

    waker.wake(worker)
    waker.awake(worker)
    assert not waker.waking(worker)
    clock.now += 1
    waker.wake(worker)

    assert sent == [MAC, MAC]
    assert waker.waking(worker)


def test_a_failed_wake_is_reported_and_does_not_start_a_wait():
    def fail(wake):
        raise WakeError("relay laptop-1 unreachable")

    waker = Waker(clock=FakeClock(), send=fail)
    worker = endpoint()

    note = waker.wake(worker)

    assert "wake failed" in note and "laptop-1" in note
    assert not waker.waking(worker)


# --- wake_endpoints (arc models wake) ----------------------------------------------


class Hosts:
    """Fake host states: each host answers after a number of polls."""

    def __init__(self, **states) -> None:
        self.states = {host: list(seq) for host, seq in states.items()}

    def __call__(self, host, port, timeout):
        sequence = self.states[host]
        return sequence.pop(0) if len(sequence) > 1 else sequence[0]


def run_wake(endpoints, hosts, **kwargs):
    clock, sent, lines = FakeClock(), [], []

    def sleep(seconds):
        clock.now += seconds

    ok = wake_endpoints(
        endpoints,
        explicit=kwargs.pop("explicit", False),
        out=lines.append,
        send=lambda wake: sent.append(wake.mac) or f"sent a magic packet to {wake.mac}",
        state=hosts,
        clock=clock,
        sleep=sleep,
        **kwargs,
    )
    return ok, sent, lines


def test_wake_leaves_an_awake_worker_alone():
    ok, sent, lines = run_wake([endpoint()], Hosts(**{"dev-0": ["listening"]}))

    assert ok and sent == []
    assert "already awake" in lines[0]


def test_wake_sends_and_waits_until_the_worker_answers():
    hosts = Hosts(**{"dev-0": ["silent", "silent", "silent", "listening"]})

    ok, sent, lines = run_wake([endpoint()], hosts)

    assert ok and sent == [MAC]
    assert any("awake after" in line for line in lines)


def test_wake_resends_while_waiting():
    hosts = Hosts(**{"dev-0": ["silent"] * 20 + ["listening"]})

    ok, sent, _ = run_wake([endpoint()], hosts, poll=2.0)

    assert ok and len(sent) == 2


def test_wake_reports_a_host_that_never_answers():
    ok, _, lines = run_wake([endpoint(timeout=10.0)], Hosts(**{"dev-0": ["silent"]}))

    assert not ok
    assert "did not answer within 10s" in lines[-1]
    assert "suspended" in lines[-1]


def test_wake_reports_an_awake_host_without_a_worker():
    ok, _, lines = run_wake([endpoint()], Hosts(**{"dev-0": ["silent", "refused"]}))

    assert ok
    assert "arc models start" in lines[-1]


def test_wake_without_waiting_only_sends():
    ok, sent, lines = run_wake([endpoint()], Hosts(**{"dev-0": ["silent"]}), wait=False)

    assert ok and sent == [MAC]
    assert "sent a magic packet" in lines[-1]


def test_wake_dry_run_sends_nothing():
    ok, sent, lines = run_wake([endpoint()], Hosts(**{"dev-0": ["silent"]}), dry_run=True)

    assert ok and sent == []
    assert "would send" in lines[0]


def test_wake_skips_endpoints_without_wake_settings_unless_named():
    hosts = Hosts(**{"dev-0": ["listening"], "dev-1": ["silent"]})
    workers = [endpoint(), EndpointConfig(host="dev-1")]

    ok, _, lines = run_wake(workers, hosts)
    assert ok and any("dev-1:7300: no wake settings; skipped" in line for line in lines)

    ok, _, lines = run_wake([EndpointConfig(host="dev-1")], hosts, explicit=True)
    assert not ok and "no wake settings" in lines[0]


def test_wake_waits_for_several_hosts_together():
    hosts = Hosts(**{"dev-0": ["silent", "silent", "listening"], "dev-1": ["silent", "listening"]})
    workers = [endpoint("dev-0"), endpoint("dev-1", mac="0c:4d:e9:99:4e:97")]

    ok, sent, lines = run_wake(workers, hosts)

    assert ok and sent == [MAC, "0c:4d:e9:99:4e:97"]
    assert sum("awake after" in line for line in lines) == 2


def test_wake_reports_a_failed_send():
    lines = []

    def fail(wake):
        raise WakeError("relay laptop-1 unreachable")

    ok = wake_endpoints(
        [endpoint()],
        explicit=False,
        out=lines.append,
        send=fail,
        state=Hosts(**{"dev-0": ["silent"]}),
        clock=FakeClock(),
        sleep=lambda s: None,
    )

    assert not ok
    assert "laptop-1 unreachable" in lines[-1]

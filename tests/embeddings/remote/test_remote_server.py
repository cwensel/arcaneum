"""`arc models serve` embedding server (RDR-025 Step 3)."""

from __future__ import annotations

import logging
import socket
import threading
import time

import numpy as np
import pytest
from remote_fakes import FakeClock, FakeHost, RawClient, wait_for

from arcaneum.embeddings.remote.identity import dependency_versions, local_identity
from arcaneum.embeddings.remote.server import (
    EmbeddingServer,
    ServerSettings,
    default_encode_batch_size,
    find_tailscale_address,
    resolve_bind_address,
)
from arcaneum.embeddings.worker_protocol import (
    AcceleratorWorkerSession,
    MessageType,
    WorkerConfig,
)

FACTORY = "arcaneum.embeddings.worker_protocol:create_deterministic_fake_backend"


@pytest.fixture
def host():
    return FakeHost()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def server_factory(host, clock):
    servers = []

    def make(**settings) -> EmbeddingServer:
        options = {"port": 0, "bind": "127.0.0.1", **settings}
        server = EmbeddingServer(ServerSettings(**options), host=host, clock=clock)
        server.start()
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.stop()


def test_socket_transport_returns_the_queue_transports_results(server_factory):
    server = server_factory()
    client = RawClient(server.address, "arctic-m")
    texts = ["alpha", "bb", "gamma ray"]

    reply = client.wait(client.encode(texts))
    with AcceleratorWorkerSession(WorkerConfig(FACTORY, {}), startup_timeout=3) as worker:
        expected = worker.encode(texts, timeout=2)

    assert reply["type"] == MessageType.ENCODED
    payload = reply["payload"]
    result = np.array(payload["embeddings"], dtype=np.dtype(payload["dtype"]))
    assert payload["shape"] == [3, 3]
    np.testing.assert_array_equal(result, expected)
    client.close()


def test_socket_transport_reports_the_queue_transports_error_envelopes(server_factory):
    server = server_factory()
    client = RawClient(server.address, "arctic-m")

    reply = client.wait(client.send(MessageType.ENCODE, texts="not a list"))

    assert reply["type"] == MessageType.ERROR
    assert reply["payload"]["code"] == "backend_error"
    assert reply["payload"]["message"] == "encode texts must be a list of strings"
    assert reply["payload"]["exception_type"] == "ValueError"
    client.close()


def test_handshake_answers_from_code_without_loading(server_factory, host):
    server = server_factory()
    client = RawClient(server.address, "qwen3-embed")

    payload = client.initialized["payload"]
    assert client.initialized["type"] == MessageType.INITIALIZED
    assert {k: payload[k] for k in local_identity("qwen3-embed")} == local_identity("qwen3-embed")
    assert payload["serves"] is True
    assert payload["encode_batch_size"] == 8
    assert host.loads == []
    client.close()


def test_first_encode_reports_loading_then_completes(server_factory, host):
    server = server_factory()
    host.load_gate = threading.Event()
    client = RawClient(server.address, "arctic-m")
    request = client.encode(["alpha"])

    wait_for(lambda: host.loads == ["arctic-m"])
    assert client.heartbeat()["state"] == "loading"

    host.load_gate.set()
    assert client.wait(request)["type"] == MessageType.ENCODED
    assert client.heartbeat()["state"] == "idle"
    assert client.heartbeat()["resident_model"] == "arctic-m"
    client.close()


def test_load_failure_fails_the_request_and_later_encodes_retry_the_load(server_factory, host):
    server = server_factory()
    calls = []

    def failing_load(model):
        calls.append(model)
        if len(calls) == 1:
            raise RuntimeError("download failed")

    host.load = failing_load
    client = RawClient(server.address, "arctic-m")

    failed = client.wait(client.encode(["alpha"]))
    assert failed["type"] == MessageType.ERROR
    assert failed["payload"]["code"] == "load_failed"
    assert client.wait(client.encode(["alpha"]))["type"] == MessageType.ENCODED
    assert calls == ["arctic-m", "arctic-m"]
    client.close()


def test_status_only_connection_reports_health_and_cannot_encode(server_factory, host):
    server = server_factory()
    client = RawClient(server.address)

    health = client.heartbeat()
    assert (
        client.initialized["payload"]["code_fingerprint"]
        == local_identity(None)["code_fingerprint"]
    )
    assert health["arcaneum_version"] == local_identity(None)["arcaneum_version"]
    assert health["code_fingerprint"] == local_identity(None)["code_fingerprint"]
    assert health["dependencies"] == dependency_versions()
    assert health["resident_model"] is None
    assert health["busy"] is False
    assert health["state"] == "idle"

    reply = client.wait(client.encode(["alpha"]))
    assert reply["type"] == MessageType.ERROR
    assert reply["payload"]["code"] == "not_serving"
    assert host.loads == []
    client.close()


def test_model_outside_the_served_list_is_not_served(server_factory, host):
    server = server_factory(models=("arctic-m",))
    client = RawClient(server.address, "qwen3-embed")

    assert client.initialized["payload"]["serves"] is False
    assert client.wait(client.encode(["alpha"]))["payload"]["code"] == "not_serving"
    assert host.loads == []
    client.close()


def test_unpinned_models_are_never_served(server_factory):
    server = server_factory()
    client = RawClient(server.address, "bge-small")

    assert client.initialized["payload"]["serves"] is False
    client.close()


def test_mismatched_client_is_answered_but_refused_encodes(server_factory, host):
    server = server_factory()
    identity = {**local_identity("arctic-m"), "code_fingerprint": "0" * 64}
    client = RawClient(server.address, identity=identity)

    payload = client.initialized["payload"]
    assert payload["code_fingerprint"] == local_identity(None)["code_fingerprint"]
    reply = client.wait(client.encode(["alpha"]))
    assert reply["payload"]["code"] == "incompatible"
    assert host.loads == []
    client.close()


def test_configured_encode_batch_size_is_reported(server_factory):
    server = server_factory(batch_sizes={"arctic-m": 16})
    client = RawClient(server.address, "arctic-m")

    assert client.initialized["payload"]["encode_batch_size"] == 16
    client.close()


def test_default_encode_batch_sizes_follow_cpu_batching():
    assert default_encode_batch_size("qwen3-embed") == 8
    assert default_encode_batch_size("arctic-m") == 100


def test_authkey_is_required_when_configured(server_factory):
    server = server_factory(authkey=b"secret")

    client = RawClient(server.address, "arctic-m", authkey=b"secret")
    assert client.initialized["type"] == MessageType.INITIALIZED
    client.close()
    with pytest.raises(Exception):
        RawClient(server.address, "arctic-m", authkey=b"wrong")


def test_one_request_in_flight_per_connection(server_factory, host):
    server = server_factory()
    permits = host.encode_permits["arctic-m"] = threading.Semaphore(0)
    client = RawClient(server.address, "arctic-m")

    first = client.encode(["alpha"])
    second = client.wait(client.encode(["beta"]))

    assert second["payload"]["code"] == "busy"
    permits.release()
    assert client.wait(first)["type"] == MessageType.ENCODED
    client.close()


def test_same_model_requests_run_in_arrival_order_without_overlap(server_factory, host):
    server = server_factory()
    permits = host.encode_permits["arctic-m"] = threading.Semaphore(0)
    clients = [RawClient(server.address, "arctic-m") for _ in range(3)]

    requests = []
    for index, client in enumerate(clients):
        requests.append(client.encode([f"text-{index}"]))
        wait_for(lambda index=index: server.queued_count() + host.active == index + 1)
    permits.release(3)
    for client, request in zip(clients, requests):
        assert client.wait(request)["type"] == MessageType.ENCODED

    assert [texts for _, texts in host.encodes] == [["text-0"], ["text-1"], ["text-2"]]
    assert host.max_active == 1
    for client in clients:
        client.close()


def test_other_model_waits_for_a_busy_resident_until_its_turn_ends(server_factory, host, clock):
    server = server_factory(idle_grace=60, turn_length=600)
    first, second = RawClient(server.address, "arctic-m"), RawClient(server.address, "arctic-m")
    qwen = RawClient(server.address, "qwen3-embed")
    assert first.wait(first.encode(["warm"]))["type"] == MessageType.ENCODED

    # Two arctic-m clients keep one request running and one queued at all times.
    permits = host.encode_permits["arctic-m"] = threading.Semaphore(0)
    pending = {first: first.encode(["first-0"])}
    wait_for(lambda: host.active == 1)
    waiting = qwen.encode(["other"])
    pending[second] = second.encode(["second-0"])
    wait_for(lambda: server.queued_count() == 2)

    for step in range(5):
        clock.now += 100
        done = (first, second)[step % 2]
        permits.release()
        assert done.wait(pending[done])["type"] == MessageType.ENCODED
        wait_for(lambda: host.active == 1)
        pending[done] = done.encode([f"next-{step}"])
        wait_for(lambda: server.queued_count() == 2)
        assert qwen.heartbeat()["state"] == "queued"

    # The turn has now lasted 600 s: the resident yields after its current encode
    # even though another of its requests is queued.
    clock.now += 100
    permits.release()
    assert second.wait(pending[second])["type"] == MessageType.ENCODED
    assert qwen.wait(waiting)["type"] == MessageType.ENCODED
    assert first.heartbeat()["state"] == "queued"

    # qwen3-embed now holds the endpoint until it has been idle for the grace period.
    clock.now += 61
    server.wake()
    permits.release()
    assert first.wait(pending[first])["type"] == MessageType.ENCODED

    assert host.loads == ["arctic-m", "qwen3-embed", "arctic-m"]
    assert host.unloads == ["arctic-m", "qwen3-embed"]
    assert host.max_active == 1
    for client in (first, second, qwen):
        client.close()


def test_other_model_takes_over_after_the_idle_grace_period(server_factory, host, clock):
    server = server_factory(idle_grace=60, turn_length=600)
    arctic = RawClient(server.address, "arctic-m")
    qwen = RawClient(server.address, "qwen3-embed")
    assert arctic.wait(arctic.encode(["warm"]))["type"] == MessageType.ENCODED

    waiting = qwen.encode(["other"])
    wait_for(lambda: server.queued_count() == 1)
    clock.now += 59
    server.wake()
    time.sleep(0.05)
    assert qwen.heartbeat()["state"] == "queued"
    assert host.loads == ["arctic-m"]

    clock.now += 2
    server.wake()
    assert qwen.wait(waiting)["type"] == MessageType.ENCODED
    assert host.loads == ["arctic-m", "qwen3-embed"]
    arctic.close()
    qwen.close()


def test_disconnected_clients_queued_requests_are_dropped(server_factory, host):
    server = server_factory()
    permits = host.encode_permits["arctic-m"] = threading.Semaphore(0)
    first = RawClient(server.address, "arctic-m")
    second = RawClient(server.address, "arctic-m")
    first.encode(["running"])
    assert host.encode_started.wait(5)
    second.encode(["queued"])
    wait_for(lambda: server.queued_count() == 1)

    second.close()
    wait_for(lambda: server.queued_count() == 0)
    permits.release()
    wait_for(lambda: host.active == 0 and len(host.encodes) == 1)
    first.close()


def test_server_binds_a_literal_address(server_factory):
    server = server_factory(bind="127.0.0.1")

    assert server.address[0] == "127.0.0.1"
    with socket.create_connection(server.address, timeout=2):
        pass


class FakeInterfaces:
    def __init__(self, tables):
        self.tables = list(tables)

    def __call__(self):
        return self.tables[0]

    def advance(self):
        if len(self.tables) > 1:
            self.tables.pop(0)


def addr(ip: str):
    return type("snicaddr", (), {"family": socket.AF_INET, "address": ip})()


def test_bind_all_listens_on_every_interface():
    assert resolve_bind_address("all") == "0.0.0.0"


def test_tailscale_address_prefers_tailscale0():
    table = {
        "eth0": [addr("192.168.1.5")],
        "utun3": [addr("100.70.1.1")],
        "tailscale0": [addr("100.84.12.108")],
    }
    assert find_tailscale_address(table) == "100.84.12.108"


def test_tailscale_address_falls_back_to_any_interface_without_tailscale0():
    table = {"eth0": [addr("192.168.1.5")], "utun3": [addr("100.70.1.1")]}
    assert find_tailscale_address(table) == "100.70.1.1"


def test_tailscale0_without_a_tailnet_address_is_not_ready():
    table = {"tailscale0": [], "utun3": [addr("100.70.1.1")]}
    assert find_tailscale_address(table) is None


def test_bind_tailscale_waits_for_the_address_and_logs_once_a_minute(caplog):
    clock = FakeClock()
    pending = [{"eth0": [addr("192.168.1.5")]}] * 70
    interfaces = FakeInterfaces([*pending, {"tailscale0": [addr("100.84.12.108")]}])
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds
        interfaces.advance()

    with caplog.at_level(logging.INFO, logger="arcaneum.embeddings.remote.server"):
        address = resolve_bind_address("tailscale", interfaces=interfaces, sleep=sleep, clock=clock)

    assert address == "100.84.12.108"
    assert set(sleeps) == {2.0}
    waiting = [r for r in caplog.records if "Waiting for" in r.getMessage()]
    # 140 s of waiting logs at 0 s, 60 s and 120 s.
    assert len(waiting) == 3
    assert any("Binding" in r.getMessage() for r in caplog.records)


def test_bind_literal_waits_until_the_address_exists():
    clock = FakeClock()
    interfaces = FakeInterfaces([{"lo": [addr("127.0.0.1")]}, {"eth0": [addr("10.0.0.7")]}])

    def sleep(seconds):
        clock.now += seconds
        interfaces.advance()

    assert (
        resolve_bind_address("10.0.0.7", interfaces=interfaces, sleep=sleep, clock=clock)
        == "10.0.0.7"
    )


def test_bind_rejects_values_that_are_not_modes_or_addresses():
    with pytest.raises(ValueError, match="bind"):
        resolve_bind_address("dev-0")


@pytest.mark.slow
def test_real_server_vectors_match_local_cpu_embedding():
    """arctic-m through the socket equals the workstation's in-process CPU path."""
    from arcaneum.embeddings.client import EmbeddingClient
    from arcaneum.paths import get_models_dir

    local = EmbeddingClient(cache_dir=str(get_models_dir()))
    if not local.is_model_cached("arctic-m"):
        pytest.skip("arctic-m pinned snapshot is not cached locally")
    server = EmbeddingServer(
        ServerSettings(port=0, bind="127.0.0.1", cache_dir=str(get_models_dir()))
    )
    server.start()
    try:
        client = RawClient(server.address, "arctic-m")
        texts = ["Remote workers embed documents.", "A chunk ending in **emphasis**", "x" * 3000]
        reply = client.wait(client.encode(texts), timeout=120)
        client.close()
    finally:
        server.stop()

    assert reply["type"] == MessageType.ENCODED, reply["payload"]
    remote = np.array(reply["payload"]["embeddings"], dtype=np.float32)
    expected = np.asarray(local.embed(texts, "arctic-m"), dtype=np.float32)
    np.testing.assert_allclose(remote, expected, atol=1e-6)


def test_open_listener_without_an_authkey_warns(host, clock, caplog):
    server = EmbeddingServer(ServerSettings(port=0, bind="all"), host=host, clock=clock)
    with caplog.at_level(logging.WARNING, logger="arcaneum.embeddings.remote.server"):
        server.start()
    server.stop()

    assert any("authkey" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


def test_authkey_or_specific_address_does_not_warn(host, clock, caplog):
    with caplog.at_level(logging.WARNING, logger="arcaneum.embeddings.remote.server"):
        for settings in (
            ServerSettings(port=0, bind="all", authkey=b"k"),
            ServerSettings(port=0, bind="127.0.0.1"),
        ):
            server = EmbeddingServer(settings, host=host, clock=clock)
            server.start()
            server.stop()

    assert not [r for r in caplog.records if r.levelno == logging.WARNING]

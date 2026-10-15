"""Remote embedding pool (RDR-025 Step 4)."""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest
from remote_fakes import FakeClock, FakeHost, text_vector, wait_for

from arcaneum.embeddings.client import EMBEDDING_MODELS
from arcaneum.embeddings.remote.config import (
    DeadlineConfig,
    EndpointConfig,
    RemoteEmbeddingConfig,
)
from arcaneum.embeddings.remote.errors import RemoteEmbeddingBatchError
from arcaneum.embeddings.remote.identity import local_identity
from arcaneum.embeddings.remote.pool import (
    RemoteEmbeddingFallback,
    RemoteEmbeddingPool,
    RemoteEmbeddingUnavailable,
    probe_endpoint,
    tailscale_resolution_problem,
)
from arcaneum.embeddings.remote.server import EmbeddingServer, ServerSettings

MODEL = "arctic-m"


def expected(texts, model=MODEL):
    dimension = EMBEDDING_MODELS[model]["dimensions"]
    return np.array([text_vector(t, dimension) for t in texts], dtype=np.float32)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Cluster:
    """In-process servers, each with its own fake host, sharing one fake clock."""

    def __init__(self) -> None:
        self.clock = FakeClock()
        self.servers: list[EmbeddingServer] = []
        self.hosts: list[FakeHost] = []
        self.pools: list[RemoteEmbeddingPool] = []

    def server(self, *, port=0, identity=None, **settings) -> EmbeddingServer:
        host = FakeHost(per_text=True)
        options = {"port": port, "bind": "127.0.0.1", **settings}
        kwargs = {"identity": identity} if identity else {}
        server = EmbeddingServer(ServerSettings(**options), host=host, clock=self.clock, **kwargs)
        server.start()
        self.servers.append(server)
        self.hosts.append(host)
        return server

    def pool(self, ports, *, unavailable="fail", notify=None, **overrides) -> RemoteEmbeddingPool:
        deadlines = DeadlineConfig(
            heartbeat_interval=0.02,
            heartbeat_timeout=1.0,
            connect_timeout=1.0,
            loading=overrides.pop("loading", 5.0),
            encode_per_chunk=overrides.pop("encode_per_chunk", {MODEL: 5.0, "qwen3-embed": 5.0}),
        )
        config = RemoteEmbeddingConfig(
            endpoints=[
                EndpointConfig(host="127.0.0.1", port=port, **overrides.pop("endpoint", {}))
                for port in ports
            ],
            unavailable=unavailable,
            probe_interval=0.05,
            deadlines=deadlines,
            **overrides,
        )
        pool = RemoteEmbeddingPool(config, notify=notify or (lambda message: None))
        self.pools.append(pool)
        return pool

    def close(self) -> None:
        for pool in self.pools:
            pool.close()
        for host in self.hosts:
            if host.load_gate is not None:
                host.load_gate.set()
            for permits in host.encode_permits.values():
                permits.release(100)
        for server in self.servers:
            server.stop()


@pytest.fixture
def cluster():
    value = Cluster()
    yield value
    value.close()


def port_of(server):
    return server.address[1]


def test_embed_spreads_sub_batches_and_reassembles_in_order(cluster):
    first = cluster.server(batch_sizes={MODEL: 2})
    second = cluster.server(batch_sizes={MODEL: 2})
    pool = cluster.pool([port_of(first), port_of(second)])
    texts = [f"text number {i}" for i in range(7)]

    result = pool.embed(texts, MODEL)

    np.testing.assert_array_equal(result, expected(texts))
    assert result.dtype == np.float32 and result.flags.c_contiguous and result.flags.owndata
    batches = [batch for host in cluster.hosts for _, batch in host.encodes]
    assert all(len(batch) <= 2 for batch in batches)
    assert all(host.encodes for host in cluster.hosts)
    assert sorted(t for batch in batches for t in batch) == sorted(texts)


def test_empty_input_sends_nothing(cluster):
    server = cluster.server()
    pool = cluster.pool([port_of(server)])

    assert pool.embed([], MODEL).shape == (0, 768)
    assert cluster.hosts[0].encodes == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_version", 99),
        ("arcaneum_version", "0.0.1"),
        ("code_fingerprint", "0" * 64),
        ("revision", "f" * 40),
        ("prompt_policy_version", 2),
        ("dimensions", 1024),
    ],
)
def test_handshake_mismatch_excludes_the_endpoint_without_encoding(cluster, field, value):
    def mismatched(model):
        identity = local_identity(model)
        if model is not None:
            identity[field] = value
        return identity

    bad = cluster.server(identity=mismatched)
    good = cluster.server()
    pool = cluster.pool([port_of(bad), port_of(good)])

    texts = ["alpha", "beta", "gamma"]
    np.testing.assert_array_equal(pool.embed(texts, MODEL), expected(texts))
    assert cluster.hosts[0].encodes == []
    assert cluster.hosts[0].loads == []


def test_handshake_mismatch_under_fail_names_both_sides(cluster):
    def mismatched(model):
        return {**local_identity(model), "code_fingerprint": "0" * 64}

    bad = cluster.server(identity=mismatched)
    pool = cluster.pool([port_of(bad)])

    with pytest.raises(RemoteEmbeddingUnavailable) as raised:
        pool.embed(["alpha"], MODEL)
    message = str(raised.value)
    assert "code_fingerprint" in message
    assert local_identity(None)["code_fingerprint"][:12] in message
    assert "000000000000" in message
    assert "arc models deploy" in message


def test_first_encode_waits_through_a_slow_load(cluster):
    server = cluster.server()
    cluster.hosts[0].load_gate = threading.Event()
    pool = cluster.pool([port_of(server)], encode_per_chunk={MODEL: 0.1})
    result = {}

    worker = threading.Thread(target=lambda: result.setdefault("v", pool.embed(["a"], MODEL)))
    worker.start()
    wait_for(lambda: cluster.hosts[0].loads == [MODEL])
    time.sleep(0.4)  # longer than the encode allowance; loading has its own deadline
    cluster.hosts[0].load_gate.set()
    worker.join(timeout=5)

    np.testing.assert_array_equal(result["v"], expected(["a"]))


def test_overlong_load_marks_the_endpoint_down_and_retries_elsewhere(cluster):
    slow = cluster.server()
    cluster.hosts[0].load_gate = threading.Event()
    fast = cluster.server()
    pool = cluster.pool([port_of(slow), port_of(fast)], loading=0.3)
    texts = [f"t{i}" for i in range(6)]

    np.testing.assert_array_equal(pool.embed(texts, MODEL), expected(texts))
    status = {h.endpoint: h for h in pool.link_status(MODEL)}
    assert status[f"127.0.0.1:{port_of(slow)}"].state == "down"
    assert "loading deadline" in status[f"127.0.0.1:{port_of(slow)}"].reason


def test_killed_endpoint_mid_batch_is_retried_and_results_stay_in_order(cluster):
    doomed = cluster.server(batch_sizes={MODEL: 3})
    survivor = cluster.server(batch_sizes={MODEL: 3})
    cluster.hosts[0].encode_permits[MODEL] = threading.Semaphore(0)
    pool = cluster.pool([port_of(doomed), port_of(survivor)])
    texts = [f"chunk {i}" for i in range(6)]
    result = {}

    worker = threading.Thread(target=lambda: result.setdefault("v", pool.embed(texts, MODEL)))
    worker.start()
    assert cluster.hosts[0].encode_started.wait(5)
    doomed.stop()
    worker.join(timeout=10)

    np.testing.assert_array_equal(result["v"], expected(texts))


def test_encode_deadline_marks_the_endpoint_down_and_retries(cluster):
    stuck = cluster.server(batch_sizes={MODEL: 2})
    healthy = cluster.server(batch_sizes={MODEL: 2})
    cluster.hosts[0].encode_permits[MODEL] = threading.Semaphore(0)
    pool = cluster.pool([port_of(stuck), port_of(healthy)], encode_per_chunk={MODEL: 0.1})
    texts = [f"chunk {i}" for i in range(4)]

    np.testing.assert_array_equal(pool.embed(texts, MODEL), expected(texts))
    status = {h.endpoint: h for h in pool.link_status(MODEL)}
    assert "encoding deadline" in status[f"127.0.0.1:{port_of(stuck)}"].reason


def test_invalid_results_are_retried_once_without_taking_the_endpoint_down(cluster):
    first = cluster.server()
    second = cluster.server()
    pool = cluster.pool([port_of(first), port_of(second)])
    seen = []

    def validate(array, count):
        seen.append(count)
        return len(seen) > 1

    texts = ["only"]
    np.testing.assert_array_equal(pool.embed(texts, MODEL, validate=validate), expected(texts))
    assert len(seen) == 2
    assert {status.state for status in pool.link_status(MODEL)} == {"up"}


@pytest.mark.parametrize("unavailable", ["wait", "fail"])
def test_a_batch_that_always_fails_fails_the_call_not_the_endpoints(cluster, unavailable):
    """A deterministic per-batch error must not loop forever under `wait`."""
    servers = [cluster.server(), cluster.server()]
    for host in cluster.hosts:
        original = host.encode

        def encode(model, texts, original=original):
            if "poison" in texts:
                raise ValueError("tokenizer exploded")
            return original(model, texts)

        host.encode = encode
    pool = cluster.pool([port_of(s) for s in servers], unavailable=unavailable)

    with pytest.raises(RemoteEmbeddingBatchError, match="tokenizer exploded"):
        pool.embed(["poison"], MODEL)

    assert {status.state for status in pool.link_status(MODEL)} == {"up"}
    np.testing.assert_array_equal(pool.embed(["fine"], MODEL), expected(["fine"]))


def test_always_invalid_results_fail_the_call(cluster):
    server = cluster.server()
    pool = cluster.pool([port_of(server)], unavailable="wait")

    with pytest.raises(RemoteEmbeddingBatchError, match="invalid"):
        pool.embed(["x"], MODEL, validate=lambda rows, count: False)


def test_unavailable_messages_name_the_environment_override(cluster):
    pool = cluster.pool([free_port()])

    with pytest.raises(RemoteEmbeddingUnavailable) as raised:
        pool.embed(["alpha"], MODEL)
    assert "ARC_EMBED_ENDPOINTS=" in str(raised.value)
    assert "--local-embed" in str(raised.value)


def test_health_checks_run_in_parallel():
    pool = RemoteEmbeddingPool(RemoteEmbeddingConfig())

    class SlowLink:
        def health(self):
            time.sleep(0.3)
            return {"resident_model": MODEL, "busy": False}

    links = [SlowLink() for _ in range(4)]
    started = time.monotonic()
    chosen = pool._prefer(links, MODEL)

    assert chosen == links
    assert time.monotonic() - started < 0.9


def test_all_endpoints_down_under_fail_raises(cluster):
    pool = cluster.pool([free_port()])

    with pytest.raises(RemoteEmbeddingUnavailable, match="127.0.0.1"):
        pool.embed(["alpha"], MODEL)


def test_all_endpoints_down_under_local_asks_for_local_embedding(cluster):
    pool = cluster.pool([free_port()], unavailable="local")

    with pytest.raises(RemoteEmbeddingFallback):
        pool.embed(["alpha"], MODEL)


def test_all_endpoints_down_under_wait_blocks_and_recovers(cluster):
    port = free_port()
    messages = []
    pool = cluster.pool([port], unavailable="wait", notify=messages.append)
    result = {}

    worker = threading.Thread(target=lambda: result.setdefault("v", pool.embed(["a"], MODEL)))
    worker.start()
    wait_for(lambda: messages)
    assert worker.is_alive()
    assert f"127.0.0.1:{port}" in messages[0]
    cluster.server(port=port)
    worker.join(timeout=10)

    np.testing.assert_array_equal(result["v"], expected(["a"]))


def test_mismatched_endpoints_under_wait_resume_after_redeploy(cluster):
    port = free_port()

    def old_release(model):
        identity = local_identity(model)
        identity.update(arcaneum_version="0.9.0", code_fingerprint="0" * 64)
        return identity

    stale = cluster.server(port=port, identity=old_release)
    messages = []
    pool = cluster.pool([port], unavailable="wait", notify=messages.append)
    result = {}

    worker = threading.Thread(target=lambda: result.setdefault("v", pool.embed(["a"], MODEL)))
    worker.start()
    wait_for(lambda: messages)
    message = messages[0]
    assert "0.9.0" in message and local_identity(None)["arcaneum_version"] in message
    assert "arc models deploy" in message and "--local-embed" in message
    assert cluster.hosts[0].encodes == []

    stale.stop()
    cluster.server(port=port)
    worker.join(timeout=10)
    np.testing.assert_array_equal(result["v"], expected(["a"]))


def test_bind_tailscale_endpoint_resolving_outside_the_tailnet_is_named(cluster):
    port = free_port()
    pool = cluster.pool([port], endpoint={"bind": "tailscale"})

    with pytest.raises(RemoteEmbeddingUnavailable) as raised:
        pool.embed(["alpha"], MODEL)
    message = str(raised.value)
    assert f"127.0.0.1:{port}" in message
    assert "outside the tailnet" in message
    assert "MagicDNS" in message


def test_tailscale_resolution_problem_is_none_inside_the_tailnet(monkeypatch):
    endpoint = EndpointConfig(host="dev-0", bind="tailscale")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("100.84.12.108", 7300))],
    )

    assert tailscale_resolution_problem(endpoint) is None
    assert tailscale_resolution_problem(EndpointConfig(host="dev-0")) is None


def test_second_model_queues_at_one_endpoint_and_takes_it_over(cluster):
    """Testing Strategy 8: B queues at one endpoint; afterwards A and B split the pair."""
    first = cluster.server(idle_grace=60, turn_length=600, batch_sizes={MODEL: 1})
    second = cluster.server(idle_grace=60, turn_length=600, batch_sizes={MODEL: 1})
    ports = [port_of(first), port_of(second)]
    for host in cluster.hosts:
        host.encode_permits[MODEL] = threading.Semaphore(0)
    pool_a = cluster.pool(ports)
    pool_b = cluster.pool(ports)

    # A keeps both endpoints busy.
    a_result = {}
    a_texts = ["a0", "a1"]
    a_worker = threading.Thread(
        target=lambda: a_result.setdefault("v", pool_a.embed(a_texts, MODEL))
    )
    a_worker.start()
    wait_for(lambda: all(host.active == 1 for host in cluster.hosts))

    b_result = {}
    b_worker = threading.Thread(
        target=lambda: b_result.setdefault("v", pool_b.embed(["b0"], "qwen3-embed"))
    )
    b_worker.start()
    wait_for(lambda: sum(server.queued_count() for server in cluster.servers) == 1)
    queued_at = [i for i, server in enumerate(cluster.servers) if server.queued_count() == 1]

    # A's turn ends while it is busy: the endpoint holding B's request switches.
    cluster.clock.now += 600
    for host in cluster.hosts:
        host.encode_permits[MODEL].release()
    a_worker.join(timeout=10)
    b_worker.join(timeout=10)
    np.testing.assert_array_equal(a_result["v"], expected(a_texts))
    np.testing.assert_array_equal(b_result["v"], expected(["b0"], "qwen3-embed"))

    switched, kept = queued_at[0], 1 - queued_at[0]
    assert cluster.hosts[switched].loads == [MODEL, "qwen3-embed"]
    assert cluster.hosts[kept].loads == [MODEL]

    # While B works on the switched endpoint, A uses only the one still holding A.
    qwen_permits = cluster.hosts[switched].encode_permits["qwen3-embed"] = threading.Semaphore(0)
    b_busy = threading.Thread(target=lambda: pool_b.embed(["b1"], "qwen3-embed"))
    b_busy.start()
    wait_for(lambda: cluster.hosts[switched].active == 1)
    cluster.hosts[kept].encode_permits[MODEL].release(10)
    before = len(cluster.hosts[switched].encodes)
    np.testing.assert_array_equal(pool_a.embed(["a2", "a3"], MODEL), expected(["a2", "a3"]))
    qwen_permits.release()
    b_busy.join(timeout=10)
    assert [m for m, _ in cluster.hosts[switched].encodes[before:]] == ["qwen3-embed"]

    # While A works on its endpoint, B's next run stays on the switched one.
    a_permits = cluster.hosts[kept].encode_permits[MODEL] = threading.Semaphore(0)
    a_busy = threading.Thread(target=lambda: pool_a.embed(["a4"], MODEL))
    a_busy.start()
    wait_for(lambda: cluster.hosts[kept].active == 1)
    qwen_permits.release(10)
    before = len(cluster.hosts[kept].encodes)
    texts = ["b2", "b3"]
    np.testing.assert_array_equal(
        pool_b.embed(texts, "qwen3-embed"), expected(texts, "qwen3-embed")
    )
    a_permits.release()
    a_busy.join(timeout=10)
    assert [m for m, _ in cluster.hosts[kept].encodes[before:]] == [MODEL]
    assert cluster.hosts[kept].loads == [MODEL]


def test_probe_endpoint_reports_versions_and_resident_model(cluster):
    server = cluster.server()
    pool = cluster.pool([port_of(server)])
    pool.embed(["warm"], MODEL)

    status = probe_endpoint(pool.config.endpoints[0], pool.config)

    assert status.state == "up"
    assert status.resident_model == MODEL
    assert status.code_fingerprint == local_identity(None)["code_fingerprint"]
    assert status.dependencies["fastembed"]
    assert status.mismatches == []


def test_probe_endpoint_reports_unreachable_endpoints(cluster):
    pool = cluster.pool([free_port()])

    status = probe_endpoint(pool.config.endpoints[0], pool.config)

    assert status.state == "unreachable"
    assert status.reason


@pytest.mark.parametrize(("server_key", "client_key"), [(b"secret", None), (None, "secret")])
def test_authkey_mismatch_is_reported_not_raised(cluster, server_key, client_key):
    server = cluster.server(authkey=server_key)
    pool = cluster.pool([port_of(server)], authkey=client_key)

    with pytest.raises(RemoteEmbeddingUnavailable, match="authkey"):
        pool.embed(["alpha"], MODEL)
    status = probe_endpoint(pool.config.endpoints[0], pool.config)
    assert status.state == "unreachable"
    assert "authkey" in status.reason

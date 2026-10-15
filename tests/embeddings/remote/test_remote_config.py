"""Remote embedding configuration and routing (RDR-025 Step 4)."""

import pytest

from arcaneum.embeddings.remote.config import (
    EndpointConfig,
    RemoteEmbeddingConfig,
    RemoteEmbeddingConfigError,
    load_remote_embedding_config,
    parse_endpoint_list,
)
from arcaneum.embeddings.remote.routing import RemoteRouter


def write_config(tmp_path, text: str):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_missing_config_file_has_no_endpoints(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_EMBED_ENDPOINTS", raising=False)
    config = load_remote_embedding_config(tmp_path / "absent.yaml")

    assert config.endpoints == []
    assert config.unavailable == "wait"


def test_config_section_is_loaded_with_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_EMBED_ENDPOINTS", raising=False)
    path = write_config(
        tmp_path,
        """
qdrant:
  url: http://localhost:6333
remote_embedding:
  authkey: secret
  unavailable: fail
  endpoints:
    - host: dev-0
      bind: tailscale
      batch_size: {qwen3-embed: 4}
    - host: dev-1
      port: 7301
      ssh: admin@dev-1
      models: [arctic-m]
""",
    )
    config = load_remote_embedding_config(path)

    first, second = config.endpoints
    assert (first.host, first.port, first.ssh_target, first.bind) == (
        "dev-0",
        7300,
        "dev-0",
        "tailscale",
    )
    assert first.batch_size == {"qwen3-embed": 4}
    assert (second.port, second.ssh_target, second.models) == (7301, "admin@dev-1", ["arctic-m"])
    assert config.authkey == "secret"
    assert config.unavailable == "fail"


@pytest.mark.parametrize(
    "priority, shares", [(None, 128), ("idle", 2), ("low", 128), ("normal", 1024)]
)
def test_endpoint_cpu_priority_maps_to_docker_cpu_shares(priority, shares):
    endpoint = (
        EndpointConfig(host="dev-0")
        if priority is None
        else EndpointConfig(host="dev-0", cpu_priority=priority)
    )
    assert endpoint.cpu_shares == shares


def test_unknown_cpu_priority_is_rejected():
    with pytest.raises(ValueError):
        EndpointConfig(host="dev-0", cpu_priority="high")


def test_default_routing_sends_arctic_m_remote_and_qwen3_on_request():
    config = RemoteEmbeddingConfig()

    assert config.model_routing("arctic-m") == "remote"
    assert config.model_routing("qwen3-embed") == "on-request"
    assert config.model_routing("jina-code") == "local"


def test_configured_routing_overrides_defaults():
    config = RemoteEmbeddingConfig(routing={"arctic-m": "local", "qwen3-embed": "remote"})

    assert config.model_routing("arctic-m") == "local"
    assert config.model_routing("qwen3-embed") == "remote"


@pytest.mark.parametrize("mode", ["remote", "on-request"])
def test_unpinned_fastembed_model_cannot_be_routed_remotely(mode):
    with pytest.raises(RemoteEmbeddingConfigError, match="bge-small.*revision"):
        RemoteEmbeddingConfig(routing={"bge-small": mode})


def test_unknown_routed_model_is_a_configuration_error():
    with pytest.raises(RemoteEmbeddingConfigError, match="nope"):
        RemoteEmbeddingConfig(routing={"nope": "remote"})


def test_endpoint_models_default_to_every_routed_model():
    config = RemoteEmbeddingConfig(endpoints=[EndpointConfig(host="dev-0")])

    assert config.endpoint_models(config.endpoints[0]) == ["qwen3-embed", "arctic-m"]


def test_environment_endpoint_list_replaces_configured_endpoints(tmp_path, monkeypatch):
    path = write_config(
        tmp_path,
        """
remote_embedding:
  endpoints:
    - host: dev-0
      bind: tailscale
""",
    )
    monkeypatch.setenv("ARC_EMBED_ENDPOINTS", "box-a:7400, box-b")
    config = load_remote_embedding_config(path)

    assert [(e.host, e.port, e.bind) for e in config.endpoints] == [
        ("box-a", 7400, "all"),
        ("box-b", 7300, "all"),
    ]


def test_empty_environment_endpoint_list_disables_the_pool(tmp_path, monkeypatch):
    path = write_config(tmp_path, "remote_embedding:\n  endpoints:\n    - host: dev-0\n")
    monkeypatch.setenv("ARC_EMBED_ENDPOINTS", "")

    assert load_remote_embedding_config(path).endpoints == []


def test_parse_endpoint_list_rejects_bad_ports():
    with pytest.raises(RemoteEmbeddingConfigError, match="port"):
        parse_endpoint_list("dev-0:http")


def test_environment_authkey_overrides_configuration(tmp_path, monkeypatch):
    path = write_config(tmp_path, "remote_embedding:\n  authkey: from-file\n")
    monkeypatch.setenv("ARC_EMBED_AUTHKEY", "from-env")

    assert load_remote_embedding_config(path).authkey == "from-env"


def test_invalid_section_is_a_configuration_error(tmp_path):
    path = write_config(tmp_path, "remote_embedding:\n  unavailable: sometimes\n")

    with pytest.raises(RemoteEmbeddingConfigError, match="unavailable"):
        load_remote_embedding_config(path)


# --- Routing -------------------------------------------------------------------------


def router(mode: str, endpoints=True, **config) -> RemoteRouter:
    hosts = [EndpointConfig(host="dev-0")] if endpoints else []
    return RemoteRouter(RemoteEmbeddingConfig(endpoints=hosts, **config), mode=mode)


@pytest.mark.parametrize(
    ("mode", "arctic", "qwen", "jina"),
    [
        ("default", True, False, False),
        ("remote", True, True, False),
        ("local", False, False, False),
    ],
)
def test_routing_modes(mode, arctic, qwen, jina):
    value = router(mode)

    assert value.routes("arctic-m", "document") is arctic
    assert value.routes("qwen3-embed", "document") is qwen
    assert value.routes("jina-code", "document") is jina


@pytest.mark.parametrize("mode", ["default", "remote"])
def test_queries_are_never_routed(mode):
    assert router(mode).routes("arctic-m", "query") is False


def test_nothing_is_routed_without_endpoints():
    assert router("remote", endpoints=False).routes("arctic-m", "document") is False


def test_unknown_routing_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        router("sometimes")

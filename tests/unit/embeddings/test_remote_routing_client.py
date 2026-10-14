"""EmbeddingClient routing of document embedding to the remote pool (RDR-025)."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from arcaneum.embeddings import memory_probe
from arcaneum.embeddings.client import EmbeddingClient
from arcaneum.embeddings.remote.pool import RemoteEmbeddingFallback, RemoteEmbeddingUnavailable
from arcaneum.embeddings.remote.routing import install_router


class StubRouter:
    """Routes the given models' documents; returns unit-ish vectors."""

    def __init__(self, models=("arctic-m",), error=None):
        self.models = set(models)
        self.error = error
        self.calls = []

    def routes(self, model, prompt_type):
        return model in self.models and prompt_type == "document"

    def embed(self, texts, model, validate=None):
        self.calls.append((list(texts), model))
        if self.error is not None:
            raise self.error
        rows = np.random.default_rng(len(texts)).normal(size=(len(texts), 768)).astype(np.float32)
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        if validate is not None:
            assert validate(rows, len(texts))
        return rows

    def close(self):
        return None


@pytest.fixture
def router():
    value = StubRouter()
    install_router(value)
    yield value
    install_router(None)


def local_model_forbidden(*args, **kwargs):
    raise AssertionError("a local model was constructed for remote-routed work")


def test_remote_routed_embed_constructs_no_local_model(tmp_path, router):
    client = EmbeddingClient(cache_dir=str(tmp_path))

    with patch.object(client, "get_model", side_effect=local_model_forbidden):
        result = client.embed(["alpha", "beta"], "arctic-m")

    assert result.shape == (2, 768)
    assert router.calls == [(["alpha", "beta"], "arctic-m")]
    assert client.local_models == []


def test_remote_routed_embed_parallel_constructs_no_local_model(tmp_path, router):
    client = EmbeddingClient(cache_dir=str(tmp_path))
    texts = [f"chunk {i}" for i in range(5)]

    with patch.object(client, "get_model", side_effect=local_model_forbidden):
        result = client.embed_parallel(texts, "arctic-m", batch_size=2)

    assert len(result) == 5
    assert [len(call[0]) for call in router.calls] == [2, 2, 1]
    assert client.local_models == []


def test_query_embedding_stays_local(tmp_path, router):
    client = EmbeddingClient(cache_dir=str(tmp_path))
    local = np.eye(1, 768, dtype=np.float32)

    with (
        patch.object(client, "get_model") as get_model,
        patch.object(client, "_embed_impl", return_value=local) as embed_impl,
    ):
        client.embed(["what is arctic?"], "arctic-m", prompt_type="query")

    embed_impl.assert_called_once()
    assert router.calls == []
    get_model.assert_not_called()


def test_unrouted_models_embed_locally(tmp_path, router):
    client = EmbeddingClient(cache_dir=str(tmp_path))

    with patch.object(client, "_embed_impl", return_value=np.eye(1, 768)) as embed_impl:
        client.embed(["x"], "bge-base")

    embed_impl.assert_called_once()
    assert router.calls == []


def test_local_policy_falls_back_to_local_embedding(tmp_path):
    install_router(StubRouter(error=RemoteEmbeddingFallback("all down")))
    client = EmbeddingClient(cache_dir=str(tmp_path))
    try:
        with patch.object(client, "_embed_impl", return_value=np.eye(1, 768)) as embed_impl:
            client.embed(["x"], "arctic-m")
    finally:
        install_router(None)

    embed_impl.assert_called_once()


def test_fail_policy_propagates(tmp_path):
    install_router(StubRouter(error=RemoteEmbeddingUnavailable("all down")))
    client = EmbeddingClient(cache_dir=str(tmp_path))
    try:
        with pytest.raises(RemoteEmbeddingUnavailable):
            client.embed(["x"], "arctic-m")
    finally:
        install_router(None)


def test_local_model_loads_are_recorded(tmp_path):
    client = EmbeddingClient(cache_dir=str(tmp_path))

    with (
        patch("arcaneum.embeddings.client._snapshot_download", return_value=str(tmp_path)),
        patch("arcaneum.embeddings.client.os.path.isfile", return_value=True),
        patch("arcaneum.embeddings.client.TextEmbedding"),
    ):
        client.get_model("arctic-m")

    assert client.local_models == ["arctic-m"]


def test_memory_probe_reports_local_models(tmp_path, router):
    client = EmbeddingClient(cache_dir=str(tmp_path))
    client.embed(["alpha"], "arctic-m")

    snap = memory_probe.snapshot(client)
    line = json.loads(memory_probe.format_snapshot_jsonl(snap, phase="encoding"))

    assert snap.local_models == ()
    assert line["local_models"] == []

    client._local_models.add("qwen3-embed")
    line = json.loads(memory_probe.format_snapshot_jsonl(memory_probe.snapshot(client)))
    assert line["local_models"] == ["qwen3-embed"]

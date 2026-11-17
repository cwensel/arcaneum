"""EmbeddingClient(allow_download=False) never reaches the network (RDR-026 Q9).

The MCP server searches with downloads off. A model directory can exist yet lack
its weights, so the pre-load cache test (``is_model_cached``) can pass for a
cache the loader cannot use. With downloads off, every loader refuses a failed
local load with ``ModelNotCachedError`` instead of retrying with network access.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from arcaneum.embeddings.client import EmbeddingClient, ModelNotCachedError


def _fake_sentence_transformers(monkeypatch):
    """A SentenceTransformer whose local-only load fails, as with missing weights."""
    calls = []

    def construct(_name, **kwargs):
        calls.append(kwargs["local_files_only"])
        if kwargs["local_files_only"]:
            raise OSError("model.safetensors not found in the local cache")
        return MagicMock()

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = construct
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return calls


def _incomplete_cache(tmp_path, repo):
    """A model directory that exists but holds no weights."""
    model_dir = tmp_path / f"models--{repo.replace('/', '--')}"
    (model_dir / "snapshots" / "abc").mkdir(parents=True)
    return model_dir


class TestSentenceTransformersLoader:
    def test_incomplete_cache_is_model_not_cached_without_a_download(self, tmp_path, monkeypatch):
        calls = _fake_sentence_transformers(monkeypatch)
        _incomplete_cache(tmp_path, "Qwen/Qwen3-Embedding-0.6B")
        client = EmbeddingClient(cache_dir=str(tmp_path), allow_download=False)
        assert client.is_model_cached("qwen3-embed")  # the directory alone passes the guard

        with pytest.raises(ModelNotCachedError) as exc_info:
            client.get_model("qwen3-embed")

        assert exc_info.value.model_key == "qwen3-embed"
        assert calls == [True]

    def test_uncached_model_is_model_not_cached_without_a_load(self, tmp_path, monkeypatch):
        calls = _fake_sentence_transformers(monkeypatch)
        client = EmbeddingClient(cache_dir=str(tmp_path), allow_download=False)

        with pytest.raises(ModelNotCachedError):
            client.get_model("qwen3-embed")

        assert calls == []

    def test_downloads_allowed_by_default_retry_with_network(self, tmp_path, monkeypatch):
        calls = _fake_sentence_transformers(monkeypatch)
        _incomplete_cache(tmp_path, "Qwen/Qwen3-Embedding-0.6B")
        client = EmbeddingClient(cache_dir=str(tmp_path))

        client.get_model("qwen3-embed")

        assert calls == [True, False]

    def test_cpu_fallback_does_not_download(self, tmp_path, monkeypatch):
        calls = _fake_sentence_transformers(monkeypatch)
        client = EmbeddingClient(cache_dir=str(tmp_path), allow_download=False)

        with pytest.raises(ModelNotCachedError):
            client._get_cpu_fallback_model("qwen3-embed")

        assert calls == [True]

    def test_accelerator_worker_loads_strictly_local(self, tmp_path):
        client = EmbeddingClient(cache_dir=str(tmp_path), allow_download=False)
        started = []

        class Session:
            def __init__(self, config, **_kwargs):
                started.append(config.backend_config)

            def start(self):
                return self

        with patch("arcaneum.embeddings.client.AcceleratorWorkerSession", Session):
            client._get_accelerator_worker("qwen3-embed")

        assert started[0]["local_files_only"] is True
        assert started[0]["strict_local_files_only"] is True


class TestFastembedLoader:
    def test_uncached_model_is_model_not_cached_without_a_load(self, tmp_path):
        client = EmbeddingClient(cache_dir=str(tmp_path), allow_download=False)

        with patch("fastembed.TextEmbedding") as text_embedding:
            with pytest.raises(ModelNotCachedError):
                client.get_model("bge-small")

        text_embedding.assert_not_called()

    def test_failed_local_load_is_model_not_cached_without_a_download(self, tmp_path):
        model_dir = _incomplete_cache(tmp_path, "qdrant/bge-small-en-v1.5-onnx-q")
        client = EmbeddingClient(cache_dir=str(tmp_path), allow_download=False)
        missing = RuntimeError("[ONNXRuntimeError] : 3 : NO_SUCHFILE : model_optimized.onnx")

        with (
            patch.object(EmbeddingClient, "is_model_cached", return_value=True),
            patch("fastembed.TextEmbedding", side_effect=missing) as text_embedding,
        ):
            with pytest.raises(ModelNotCachedError):
                client.get_model("bge-small")

        assert [call.kwargs["local_files_only"] for call in text_embedding.call_args_list] == [True]
        assert model_dir.is_dir()  # nothing purged for a redownload

    def test_pinned_snapshot_not_cached_is_model_not_cached_without_a_download(self, tmp_path):
        client = EmbeddingClient(cache_dir=str(tmp_path), allow_download=False)
        calls = []

        def download(**kwargs):
            calls.append(kwargs["local_files_only"])
            raise FileNotFoundError("not cached")

        with (
            patch("arcaneum.embeddings.client._snapshot_download", side_effect=download),
            patch("fastembed.TextEmbedding") as text_embedding,
        ):
            with pytest.raises(ModelNotCachedError):
                client.get_model("arctic-m")

        assert calls == [True]
        text_embedding.assert_not_called()

    def test_pinned_snapshot_with_missing_artifacts_is_not_purged(self, tmp_path):
        snapshot = tmp_path / "models--Snowflake--snowflake-arctic-embed-m" / "snapshots" / "x"
        (snapshot / "onnx").mkdir(parents=True)
        (snapshot / "onnx" / "model.onnx").write_bytes(b"onnx")
        client = EmbeddingClient(cache_dir=str(tmp_path), allow_download=False)
        missing = RuntimeError("[ONNXRuntimeError] : 3 : NO_SUCHFILE : onnx/model.onnx")
        calls = []

        def download(**kwargs):
            calls.append(kwargs["local_files_only"])
            return str(snapshot)

        with (
            patch("arcaneum.embeddings.client._snapshot_download", side_effect=download),
            patch("fastembed.TextEmbedding", side_effect=missing),
        ):
            with pytest.raises(ModelNotCachedError):
                client.get_model("arctic-m")

        assert calls == [True]
        assert snapshot.is_dir()

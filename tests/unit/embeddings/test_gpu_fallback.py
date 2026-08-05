"""Unit tests for GPU fallback stability (RDR-020)."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _install_fake_sentence_transformers(monkeypatch, side_effect=None, return_value=None):
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = MagicMock(side_effect=side_effect, return_value=return_value)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module.SentenceTransformer


@pytest.fixture
def embedding_client():
    """Create an EmbeddingClient with GPU enabled but mocked internals."""
    with patch("arcaneum.embeddings.client.get_models_dir", return_value="/tmp/models"):
        from arcaneum.embeddings.client import EmbeddingClient

        client = EmbeddingClient(cache_dir="/tmp/models", use_gpu=True)
        # Override device detection for tests
        client._device = "mps"
        yield client
        client.close()


class TestGetModelReturnsCPUWhenPoisoned:
    """get_model() returns CPU fallback for sentence-transformers when poisoned."""

    def test_returns_cpu_model_when_poisoned(self, embedding_client):
        cpu_model = MagicMock()
        cpu_model.device = "cpu"
        embedding_client._cpu_fallback_models["jina-code-st"] = cpu_model
        embedding_client._gpu_poisoned = True

        result = embedding_client.get_model("jina-code-st")

        assert result is cpu_model

    def test_loads_cpu_model_lazily_when_poisoned(self, embedding_client):
        embedding_client._gpu_poisoned = True

        mock_model = MagicMock()
        with patch(
            "arcaneum.embeddings.client.EmbeddingClient._get_cpu_fallback_model",
            return_value=mock_model,
        ) as mock_get_fallback:
            result = embedding_client.get_model("jina-code-st")

        mock_get_fallback.assert_called_once_with("jina-code-st")
        assert result is mock_model

    def test_cpu_fallback_tries_local_model_before_download(self, embedding_client, monkeypatch):
        embedding_client._gpu_poisoned = True
        mock_model = MagicMock()
        st = _install_fake_sentence_transformers(monkeypatch, return_value=mock_model)

        with patch.object(embedding_client, "is_model_cached", return_value=False):
            result = embedding_client.get_model("jina-code-st")

        assert result is mock_model
        assert st.call_count == 1
        assert st.call_args.kwargs["local_files_only"] is True
        assert st.call_args.kwargs["device"] == "cpu"

    def test_cpu_fallback_downloads_after_local_load_failure(self, embedding_client, monkeypatch):
        embedding_client._gpu_poisoned = True
        mock_model = MagicMock()
        st = _install_fake_sentence_transformers(
            monkeypatch,
            side_effect=[OSError("incomplete cache"), mock_model],
        )

        with patch.object(embedding_client, "is_model_cached", return_value=True):
            result = embedding_client.get_model("jina-code-st")

        assert result is mock_model
        assert st.call_count == 2
        assert st.call_args_list[0].kwargs["local_files_only"] is True
        assert st.call_args_list[1].kwargs["local_files_only"] is False
        assert st.call_args_list[1].kwargs["device"] == "cpu"

    def test_does_not_load_gpu_model_when_poisoned(self, embedding_client):
        """Poisoned client should not attempt to create a new GPU model."""
        embedding_client._gpu_poisoned = True
        embedding_client._cpu_fallback_models["jina-code-st"] = MagicMock()

        embedding_client.get_model("jina-code-st")

        # Should NOT have added to _models (GPU model dict)
        assert "jina-code-st" not in embedding_client._models

    def test_fastembed_model_not_affected_by_poison(self, embedding_client):
        """FastEmbed models (non-sentence-transformers) should load normally even when poisoned."""
        embedding_client._gpu_poisoned = True

        # jina-code is a FastEmbed default — get_model should try to load it normally
        # We mock the actual loading to avoid needing the model files
        mock_model = MagicMock()
        with patch("arcaneum.embeddings.client.TextEmbedding", return_value=mock_model):
            with patch.object(embedding_client, "is_model_cached", return_value=True):
                result = embedding_client.get_model("jina-code")

        assert result is mock_model

    def test_fastembed_downloads_after_local_load_failure(self, embedding_client):
        """FastEmbed cache may exist without backend-specific ONNX artifacts."""
        mock_model = MagicMock()
        with patch(
            "arcaneum.embeddings.client.TextEmbedding",
            side_effect=[OSError("incomplete onnx cache"), mock_model],
        ) as mock_text_embedding:
            with patch.object(embedding_client, "is_model_cached", return_value=True):
                result = embedding_client.get_model("jina-code")

        assert result is mock_model
        assert mock_text_embedding.call_count == 2
        assert mock_text_embedding.call_args_list[0].kwargs["local_files_only"] is True
        assert mock_text_embedding.call_args_list[1].kwargs["local_files_only"] is False

    def test_fastembed_purges_corrupt_cache_and_retries_download(self, embedding_client, tmp_path):
        """FastEmbed can reuse incomplete HF snapshots unless the cache is purged."""
        cache_dir = tmp_path / "models"
        model_dir = cache_dir / "models--jinaai--jina-embeddings-v2-base-code"
        (model_dir / "snapshots" / "bad" / "onnx").mkdir(parents=True)
        embedding_client.cache_dir = str(cache_dir)

        missing_model_error = OSError(
            "[ONNXRuntimeError] : 3 : NO_SUCHFILE : Load model from "
            f"{model_dir}/snapshots/bad/onnx/model.onnx failed. File doesn't exist"
        )
        mock_model = MagicMock()

        with patch(
            "arcaneum.embeddings.client.TextEmbedding",
            side_effect=[missing_model_error, missing_model_error, mock_model],
        ) as mock_text_embedding:
            with patch.object(embedding_client, "is_model_cached", return_value=True):
                result = embedding_client.get_model("jina-code")

        assert result is mock_model
        assert mock_text_embedding.call_count == 3
        assert not model_dir.exists()
        assert mock_text_embedding.call_args_list[0].kwargs["local_files_only"] is True
        assert mock_text_embedding.call_args_list[1].kwargs["local_files_only"] is False
        assert mock_text_embedding.call_args_list[2].kwargs["local_files_only"] is False


class TestFastEmbedCoreMLPolicy:
    """FastEmbed CoreML remains opt-in on Apple Silicon."""

    def test_fastembed_uses_cpu_provider_by_default_on_apple_silicon(
        self, embedding_client, monkeypatch, capsys
    ):
        monkeypatch.delenv("ARC_EXPERIMENTAL_COREML", raising=False)

        with patch("arcaneum.embeddings.client.sys.platform", "darwin"):
            with patch("arcaneum.embeddings.client.platform.machine", return_value="arm64"):
                providers = embedding_client._resolve_fastembed_providers("bge-large")

        assert providers == ["CPUExecutionProvider"]
        captured = capsys.readouterr()
        assert "GPU requested, but FastEmbed/CoreML is experimental" in captured.err
        assert "ARC_EXPERIMENTAL_COREML=1" in captured.err

    def test_fastembed_uses_coreml_when_gpu_flag_authorizes_it(self, monkeypatch):
        """--gpu is an explicit opt-in; no additional env var should be needed."""
        monkeypatch.delenv("ARC_EXPERIMENTAL_COREML", raising=False)

        with patch("arcaneum.embeddings.client.get_models_dir", return_value="/tmp/models"):
            from arcaneum.embeddings.client import EmbeddingClient

            client = EmbeddingClient(
                cache_dir="/tmp/models", use_gpu=True, allow_experimental_coreml=True
            )
            client._device = "mps"

        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]

        with patch("arcaneum.embeddings.client.sys.platform", "darwin"):
            with patch("arcaneum.embeddings.client.platform.machine", return_value="arm64"):
                with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
                    providers = client._resolve_fastembed_providers("arctic-m")

        assert providers[0][0] == "CoreMLExecutionProvider"
        assert providers[0][1]["ModelFormat"] == "MLProgram"
        assert providers[0][1]["RequireStaticInputShapes"] == "1"
        assert providers[1] == "CPUExecutionProvider"

    def test_fastembed_uses_coreml_when_explicitly_enabled(self, embedding_client, monkeypatch):
        monkeypatch.setenv("ARC_EXPERIMENTAL_COREML", "1")

        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]

        with patch("arcaneum.embeddings.client.sys.platform", "darwin"):
            with patch("arcaneum.embeddings.client.platform.machine", return_value="arm64"):
                with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
                    providers = embedding_client._resolve_fastembed_providers("bge-large")

        assert providers[0][0] == "CoreMLExecutionProvider"
        assert providers[0][1]["SpecializationStrategy"] == "FastPrediction"
        assert providers[1] == "CPUExecutionProvider"

    def test_get_model_passes_cpu_provider_for_fastembed_by_default(
        self, embedding_client, monkeypatch
    ):
        monkeypatch.delenv("ARC_EXPERIMENTAL_COREML", raising=False)
        mock_model = MagicMock()

        with patch("arcaneum.embeddings.client.sys.platform", "darwin"):
            with patch("arcaneum.embeddings.client.platform.machine", return_value="arm64"):
                with patch(
                    "arcaneum.embeddings.client.TextEmbedding",
                    return_value=mock_model,
                ) as mock_text_embedding:
                    with patch.object(embedding_client, "is_model_cached", return_value=True):
                        result = embedding_client.get_model("bge-large")

        assert result is mock_model
        assert mock_text_embedding.call_args.kwargs["providers"] == ["CPUExecutionProvider"]


class TestCoreMLCrashSentinel:
    """CoreML sessions leave a sentinel so an OS kill is reported on the next run."""

    def _client_with_coreml(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ARC_EXPERIMENTAL_COREML", raising=False)
        sentinel = tmp_path / "coreml-session.json"
        with patch("arcaneum.embeddings.client._coreml_sentinel_path", return_value=sentinel):
            with patch("arcaneum.embeddings.client.get_models_dir", return_value="/tmp/models"):
                from arcaneum.embeddings.client import EmbeddingClient

                client = EmbeddingClient(
                    cache_dir="/tmp/models", use_gpu=True, allow_experimental_coreml=True
                )
                client._device = "mps"
        return client, sentinel

    def test_enabling_coreml_warns_and_writes_sentinel(self, tmp_path, monkeypatch, capsys):
        client, sentinel = self._client_with_coreml(tmp_path, monkeypatch)

        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]

        with patch("arcaneum.embeddings.client._coreml_sentinel_path", return_value=sentinel):
            with patch("arcaneum.embeddings.client.sys.platform", "darwin"):
                with patch("arcaneum.embeddings.client.platform.machine", return_value="arm64"):
                    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
                        providers = client._resolve_fastembed_providers("arctic-m")

        assert providers[0][0] == "CoreMLExecutionProvider"
        assert providers[0][1]["ProfileComputePlan"] == "1"
        assert providers[1] == "CPUExecutionProvider"
        captured = capsys.readouterr()
        assert "experimental CoreML" in captured.err
        assert "killed" in captured.err

        import json
        import os

        data = json.loads(sentinel.read_text())
        assert data["pid"] == os.getpid()
        assert data["model"] == "arctic-m"

    def test_stale_sentinel_warns_about_killed_run_and_clears(self, tmp_path, monkeypatch, capsys):
        import json

        sentinel = tmp_path / "coreml-session.json"
        sentinel.write_text(json.dumps({"pid": 999999, "model": "arctic-m", "started": "x"}))

        monkeypatch.delenv("ARC_EXPERIMENTAL_COREML", raising=False)
        with patch("arcaneum.embeddings.client._coreml_sentinel_path", return_value=sentinel):
            with patch("arcaneum.embeddings.client._pid_is_alive", return_value=False):
                with patch("arcaneum.embeddings.client.get_models_dir", return_value="/tmp/models"):
                    from arcaneum.embeddings.client import EmbeddingClient

                    EmbeddingClient(cache_dir="/tmp/models")

        captured = capsys.readouterr()
        assert "did not exit cleanly" in captured.err
        assert "arctic-m" in captured.err
        assert not sentinel.exists()

    def test_sentinel_for_live_process_is_left_alone(self, tmp_path, monkeypatch, capsys):
        import json

        sentinel = tmp_path / "coreml-session.json"
        sentinel.write_text(json.dumps({"pid": 12345, "model": "arctic-m", "started": "x"}))

        monkeypatch.delenv("ARC_EXPERIMENTAL_COREML", raising=False)
        with patch("arcaneum.embeddings.client._coreml_sentinel_path", return_value=sentinel):
            with patch("arcaneum.embeddings.client._pid_is_alive", return_value=True):
                with patch("arcaneum.embeddings.client.get_models_dir", return_value="/tmp/models"):
                    from arcaneum.embeddings.client import EmbeddingClient

                    EmbeddingClient(cache_dir="/tmp/models")

        captured = capsys.readouterr()
        assert "did not exit cleanly" not in captured.err
        assert sentinel.exists()

    def test_clean_exit_removes_own_sentinel_only(self, tmp_path):
        import json
        import os

        from arcaneum.embeddings import client as client_module

        sentinel = tmp_path / "coreml-session.json"

        sentinel.write_text(json.dumps({"pid": os.getpid(), "model": "arctic-m"}))
        with patch("arcaneum.embeddings.client._coreml_sentinel_path", return_value=sentinel):
            client_module._remove_coreml_sentinel()
        assert not sentinel.exists()

        sentinel.write_text(json.dumps({"pid": 999999, "model": "arctic-m"}))
        with patch("arcaneum.embeddings.client._coreml_sentinel_path", return_value=sentinel):
            client_module._remove_coreml_sentinel()
        assert sentinel.exists()


class TestFastEmbedCacheCompleteness:
    """FastEmbed cache detection validates backend-specific ONNX artifacts."""

    def test_fastembed_cache_missing_model_file_is_not_cached(self, embedding_client, tmp_path):
        cache_dir = tmp_path / "models"
        model_dir = (
            cache_dir
            / "models--jinaai--jina-embeddings-v2-base-code"
            / "snapshots"
            / "516f4baf13dec4ddddda8631e019b5737c8bc250"
        )
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_text("", encoding="utf-8")
        embedding_client.cache_dir = str(cache_dir)

        with patch.object(
            embedding_client, "_fastembed_required_model_file", return_value="onnx/model.onnx"
        ):
            assert embedding_client.is_model_cached("jina-code") is False

    def test_fastembed_cache_with_model_file_is_cached(self, embedding_client, tmp_path):
        cache_dir = tmp_path / "models"
        model_dir = (
            cache_dir
            / "models--jinaai--jina-embeddings-v2-base-code"
            / "snapshots"
            / "516f4baf13dec4ddddda8631e019b5737c8bc250"
            / "onnx"
        )
        model_dir.mkdir(parents=True)
        (model_dir / "model.onnx").write_text("", encoding="utf-8")
        embedding_client.cache_dir = str(cache_dir)

        with patch.object(
            embedding_client, "_fastembed_required_model_file", return_value="onnx/model.onnx"
        ):
            assert embedding_client.is_model_cached("jina-code") is True

    def test_fastembed_cache_unknown_model_file_fails_open(self, embedding_client, tmp_path):
        cache_dir = tmp_path / "models"
        model_dir = (
            cache_dir
            / "models--jinaai--jina-embeddings-v2-base-code"
            / "snapshots"
            / "516f4baf13dec4ddddda8631e019b5737c8bc250"
        )
        model_dir.mkdir(parents=True)
        embedding_client.cache_dir = str(cache_dir)

        # Model absent from FastEmbed's registry: completeness cannot be
        # judged, so an existing directory is trusted (fail-open).
        with patch.object(embedding_client, "_fastembed_required_model_file", return_value=None):
            assert embedding_client.is_model_cached("jina-code") is True


class TestFastEmbedArtifactErrorDetection:
    """Only genuine missing-.onnx-artifact errors trigger cache self-healing."""

    def test_missing_model_artifact_is_detected(self, embedding_client):
        error = RuntimeError(
            "[ONNXRuntimeError] : 3 : NO_SUCHFILE : Load model from "
            "/cache/snapshots/abc/onnx/model.onnx failed: file doesn't exist"
        )
        assert embedding_client._is_missing_fastembed_artifact_error(error) is True

    def test_missing_optimized_artifact_is_detected(self, embedding_client):
        error = RuntimeError("NO_SUCHFILE: /cache/snapshots/abc/model_optimized.onnx")
        assert embedding_client._is_missing_fastembed_artifact_error(error) is True

    def test_missing_runtime_library_is_not_detected(self, embedding_client):
        error = RuntimeError(
            "ONNX Runtime provider load failed: file doesn't exist "
            "/usr/lib/libonnxruntime_providers_shared.so"
        )
        assert embedding_client._is_missing_fastembed_artifact_error(error) is False

    def test_none_error_is_not_detected(self, embedding_client):
        assert embedding_client._is_missing_fastembed_artifact_error(None) is False


class TestFastEmbedCachePurge:
    """Purging an incomplete cache never removes sibling models' caches."""

    def test_purge_removes_exact_and_wrapped_dirs_only(self, embedding_client, tmp_path):
        cache_dir = tmp_path / "models"
        exact = cache_dir / "models--BAAI--bge-large-en-v1.5"
        wrapped = cache_dir / "models--qdrant--bge-large-en-v1.5-onnx"
        for d in (exact, wrapped):
            d.mkdir(parents=True)
        embedding_client.cache_dir = str(cache_dir)

        assert embedding_client._purge_fastembed_model_cache("bge-large") is True
        assert not exact.exists()
        assert not wrapped.exists()

    def test_purge_leaves_same_family_sibling_models(self, embedding_client, tmp_path):
        cache_dir = tmp_path / "models"
        target = cache_dir / "models--jinaai--jina-embeddings-v2-base-code"
        sibling = cache_dir / "models--jinaai--jina-embeddings-v2-base-en"
        for d in (target, sibling):
            d.mkdir(parents=True)
        embedding_client.cache_dir = str(cache_dir)

        assert embedding_client._purge_fastembed_model_cache("jina-code") is True
        assert not target.exists()
        assert sibling.exists()


class TestSystemMemoryPressureGuard:
    """Low system memory disables accelerator work before starting a batch."""

    def test_low_available_memory_poisons_gpu_and_drops_model(self, embedding_client, monkeypatch):
        monkeypatch.setenv("ARC_MIN_SYSTEM_AVAILABLE_GB", "4")
        embedding_client._models["jina-code-st"] = MagicMock()

        with patch.object(embedding_client, "_system_memory_available_gb", return_value=2.5):
            disabled = embedding_client._maybe_disable_gpu_for_memory_pressure("jina-code-st")

        assert disabled is True
        assert embedding_client._gpu_poisoned is True
        assert "jina-code-st" not in embedding_client._models

    def test_healthy_available_memory_keeps_gpu_enabled(self, embedding_client, monkeypatch):
        monkeypatch.setenv("ARC_MIN_SYSTEM_AVAILABLE_GB", "4")
        embedding_client._models["jina-code-st"] = MagicMock()

        with patch.object(embedding_client, "_system_memory_available_gb", return_value=8.0):
            disabled = embedding_client._maybe_disable_gpu_for_memory_pressure("jina-code-st")

        assert disabled is False
        assert embedding_client._gpu_poisoned is False
        assert "jina-code-st" in embedding_client._models

    def test_invalid_memory_floor_uses_default(self, embedding_client, monkeypatch):
        monkeypatch.setenv("ARC_MIN_SYSTEM_AVAILABLE_GB", "invalid")

        assert embedding_client._min_system_available_gb() == 4.0

    def test_embed_falls_back_before_direct_encode_under_low_memory(
        self, embedding_client, monkeypatch
    ):
        monkeypatch.setenv("ARC_MIN_SYSTEM_AVAILABLE_GB", "4")
        embedding_client._models["jina-code-st"] = MagicMock()

        cpu_model = SimpleNamespace(_backend="sentence-transformers")

        with patch.object(embedding_client, "_system_memory_available_gb", return_value=2.5):
            with patch.object(embedding_client, "_get_cpu_fallback_model", return_value=cpu_model):
                with patch.object(embedding_client, "_encode_on_cpu_fallback") as mock_encode:
                    mock_encode.return_value = np.random.rand(1, 768).astype(np.float32)
                    with patch.object(embedding_client, "_validate_embeddings", return_value=True):
                        with patch("arcaneum.utils.memory.get_gpu_memory_info") as mock_gpu_mem:
                            embeddings = embedding_client.embed(["text"], "jina-code-st")

        assert embeddings.shape == (1, 768)
        assert embedding_client._gpu_poisoned is True
        assert "jina-code-st" not in embedding_client._models
        assert mock_encode.call_args.args[0] is cpu_model
        mock_gpu_mem.assert_not_called()

    def test_cpu_only_direct_embed_does_not_poison_gpu(self, embedding_client, monkeypatch):
        monkeypatch.setenv("ARC_MIN_SYSTEM_AVAILABLE_GB", "4")
        embedding_client.use_gpu = False

        mock_model = MagicMock()
        mock_model._backend = "sentence-transformers"
        embedding_client._models["jina-code-st"] = mock_model

        with patch.object(embedding_client, "_system_memory_available_gb", return_value=2.5):
            with patch.object(embedding_client, "_encode_with_oom_recovery") as mock_encode:
                mock_encode.return_value = np.random.rand(1, 768).astype(np.float32)
                with patch.object(embedding_client, "_validate_embeddings", return_value=True):
                    embeddings = embedding_client.embed(["text"], "jina-code-st")

        assert embeddings.shape == (1, 768)
        assert embedding_client._gpu_poisoned is False
        assert embedding_client._models["jina-code-st"] is mock_model


class TestEmbedImplCpuBatchSizingWhenPoisoned:
    """_embed_impl uses CPU batch sizing when poisoned (not GPU memory probing)."""

    def test_cpu_batch_sizing_when_poisoned(self, embedding_client):
        """When poisoned, should NOT call estimate_safe_batch_size_v2."""
        embedding_client._gpu_poisoned = True

        mock_model = MagicMock()
        mock_model._backend = "sentence-transformers"
        mock_model.encode.return_value = np.random.rand(2, 768).astype(np.float32)

        # Put the mock model in the client so get_model returns it
        embedding_client._cpu_fallback_models["jina-code-st"] = mock_model

        # Mock _encode_with_oom_recovery to capture that CPU path is used
        with patch.object(embedding_client, "_encode_with_oom_recovery") as mock_encode:
            mock_encode.return_value = np.random.rand(2, 768).astype(np.float32)
            with patch.object(embedding_client, "_validate_embeddings", return_value=True):
                with patch("arcaneum.utils.memory.get_gpu_memory_info") as mock_gpu_mem:
                    embedding_client._embed_impl(
                        ["text1", "text2"],
                        model_name="jina-code-st",
                        batch_size=32,
                    )

                    # GPU memory probing should NOT be called when poisoned
                    mock_gpu_mem.assert_not_called()


class TestCpuFallbackBounded:
    """_encode_on_cpu_fallback chunks work + caps thread counts."""

    def test_small_input_single_encode_call(self, embedding_client):
        cpu_model = MagicMock()
        cpu_model.encode.return_value = np.random.rand(2, 768).astype(np.float32)

        embedding_client._encode_on_cpu_fallback(cpu_model, ["a", "b"], "jina-code-st", "document")

        assert cpu_model.encode.call_count == 1
        kwargs = cpu_model.encode.call_args.kwargs
        assert kwargs["batch_size"] == embedding_client._CPU_FALLBACK_INNER_BATCH

    def test_large_input_split_into_outer_batches(self, embedding_client):
        outer = embedding_client._CPU_FALLBACK_OUTER_BATCH
        n = outer * 2 + 5  # forces 3 outer batches

        cpu_model = MagicMock()
        cpu_model.encode.side_effect = [
            np.random.rand(outer, 768).astype(np.float32),
            np.random.rand(outer, 768).astype(np.float32),
            np.random.rand(5, 768).astype(np.float32),
        ]

        result = embedding_client._encode_on_cpu_fallback(
            cpu_model, [f"t{i}" for i in range(n)], "jina-code-st", "document"
        )

        assert cpu_model.encode.call_count == 3
        assert result.shape == (n, 768)
        # Each call uses the reduced inner batch size, not the historical 32
        for call in cpu_model.encode.call_args_list:
            assert call.kwargs["batch_size"] == embedding_client._CPU_FALLBACK_INNER_BATCH

    def test_cpu_threading_configured_on_fallback(self, embedding_client):
        """Client started in GPU mode has no thread caps; fallback must set them."""
        cpu_model = MagicMock()
        cpu_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)

        with patch.object(embedding_client, "_configure_cpu_threading") as mock_configure:
            embedding_client._encode_on_cpu_fallback(cpu_model, ["a"], "jina-code-st", "document")

        mock_configure.assert_called_once()


class TestCpuShortCircuit:
    """_encode_with_oom_recovery on CPU device runs inline, no daemon thread."""

    def test_cpu_device_skips_daemon_thread(self, embedding_client):
        """On CPU, the encode must not spawn a timeout thread — legitimate slow
        encodes trip the 120s timeout and spawn a second competing CPU encode."""
        embedding_client._device = "cpu"

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(3, 768).astype(np.float32)

        with patch("threading.Thread") as mock_thread_ctor:
            result = embedding_client._encode_with_oom_recovery(
                mock_model,
                ["a", "b", "c"],
                internal_batch_size=256,
                model_name="jina-code-st",
            )

        mock_thread_ctor.assert_not_called()
        assert result.shape == (3, 768)
        # Inner batch size must be the bounded CPU value, not the 256 passed in
        assert (
            mock_model.encode.call_args.kwargs["batch_size"]
            == embedding_client._CPU_FALLBACK_INNER_BATCH
        )

    def test_cpu_device_does_not_poison(self, embedding_client):
        """Even if the underlying encode takes a long time, CPU path must never
        set _gpu_poisoned — there is no GPU to poison."""
        embedding_client._device = "cpu"

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)

        embedding_client._encode_with_oom_recovery(
            mock_model,
            ["a"],
            internal_batch_size=256,
            model_name="jina-code-st",
        )

        assert embedding_client._gpu_poisoned is False


class TestEmbedSortsByLength:
    """embed() sorts texts by length internally then unsorts results.

    Length-sorted batches let SentenceTransformer's per-batch padding pad to
    the longest sequence *in that batch* rather than the longest sequence in
    the whole file. On MPS this is the difference between a 7GB driver
    allocation spike on a mixed-length file and no spike at all.
    """

    def _stub_st_model(self, embedding_client):
        """Wire up a SentenceTransformers-style mock model that the
        SentenceTransformers branch in _embed_impl will use."""
        from arcaneum.embeddings.client import EMBEDDING_MODELS  # noqa: F401

        mock_model = MagicMock()
        mock_model._backend = "sentence-transformers"
        embedding_client._models["jina-code-st"] = mock_model
        # CPU-only CI resolves SentenceTransformers through the fallback cache
        # before consulting the general model cache.
        embedding_client._cpu_fallback_models["jina-code-st"] = mock_model
        return mock_model

    def test_sorted_input_to_encode(self, embedding_client):
        """The texts handed to model.encode() are sorted shortest→longest."""
        mock_model = self._stub_st_model(embedding_client)

        # encode echoes the *input order* it was given as a 1-D float per item
        def encode_side_effect(input_texts, **kwargs):
            return np.array([[float(len(t))] * 768 for t in input_texts], dtype=np.float32)

        mock_model.encode.side_effect = encode_side_effect

        # Mixed lengths in a deliberately scrambled order
        texts = ["xxxxxxxxxx", "x", "xxxxx", "xx", "xxxxxxxx"]
        # _encode_with_oom_recovery on MPS would spawn a daemon thread; bypass it
        with patch.object(
            embedding_client,
            "_encode_with_oom_recovery",
            side_effect=lambda model, t, ibs, mn, pt: encode_side_effect(t),
        ) as mock_recover:
            result = embedding_client.embed(texts, "jina-code-st")

        # _encode_with_oom_recovery must have been handed length-sorted texts
        sent_texts = mock_recover.call_args.args[1]
        assert sent_texts == sorted(texts, key=len)

        # And the result must be in the *original* (caller) order
        assert result.shape == (5, 768)
        for i, t in enumerate(texts):
            # Each row's first value equals len(original_text_at_that_index)
            assert result[i][0] == float(len(t)), (
                f"Row {i} (text len {len(t)}) got value {result[i][0]} — unsort failed"
            )

    def test_unsort_preserves_unique_embeddings(self, embedding_client):
        """No two texts of the same length must get crossed-up rows."""
        self._stub_st_model(embedding_client)

        def encode_side_effect(input_texts, **kwargs):
            # Encode = one-hot of the input order so we can detect any mix-up
            n = len(input_texts)
            arr = np.zeros((n, 768), dtype=np.float32)
            for i in range(n):
                arr[i][i % 768] = 1.0
            return arr

        # 3 texts, all length 4 (sort order is unstable on length alone — must
        # rely on stable sort to keep ties in original order, then unsort)
        texts = ["aaaa", "bbbb", "cccc"]

        with patch.object(
            embedding_client,
            "_encode_with_oom_recovery",
            side_effect=lambda model, t, ibs, mn, pt: encode_side_effect(t),
        ):
            result = embedding_client.embed(texts, "jina-code-st")

        # Each row's argmax should equal its original index, not the sorted index
        for i in range(3):
            assert int(np.argmax(result[i])) == i

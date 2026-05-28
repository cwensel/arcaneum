"""Unit tests for GPU fallback stability (RDR-020)."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def embedding_client():
    """Create an EmbeddingClient with GPU enabled but mocked internals."""
    with patch('arcaneum.embeddings.client.get_models_dir', return_value='/tmp/models'):
        from arcaneum.embeddings.client import EmbeddingClient
        client = EmbeddingClient(cache_dir='/tmp/models', use_gpu=True)
        # Override device detection for tests
        client._device = "mps"
        yield client
        # Clear any mock threads left in _pending_gpu_cleanup so the atexit
        # handler doesn't log "did not finish within 300s" warnings for the
        # MagicMock threads these tests inject.
        client._pending_gpu_cleanup.clear()


class TestGetModelReturnsCPUWhenPoisoned:
    """get_model() returns CPU fallback for sentence-transformers when poisoned."""

    def test_returns_cpu_model_when_poisoned(self, embedding_client):
        cpu_model = MagicMock()
        cpu_model.device = "cpu"
        embedding_client._cpu_fallback_models["jina-code"] = cpu_model
        embedding_client._gpu_poisoned = True

        result = embedding_client.get_model("jina-code")

        assert result is cpu_model

    def test_loads_cpu_model_lazily_when_poisoned(self, embedding_client):
        embedding_client._gpu_poisoned = True

        mock_model = MagicMock()
        with patch(
            'arcaneum.embeddings.client.EmbeddingClient._get_cpu_fallback_model',
            return_value=mock_model
        ) as mock_get_fallback:
            result = embedding_client.get_model("jina-code")

        mock_get_fallback.assert_called_once_with("jina-code")
        assert result is mock_model

    def test_cpu_fallback_tries_local_model_before_download(self, embedding_client):
        embedding_client._gpu_poisoned = True
        mock_model = MagicMock()

        with patch.object(embedding_client, "is_model_cached", return_value=False):
            with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as st:
                result = embedding_client.get_model("jina-code")

        assert result is mock_model
        assert st.call_count == 1
        assert st.call_args.kwargs["local_files_only"] is True
        assert st.call_args.kwargs["device"] == "cpu"

    def test_cpu_fallback_downloads_after_local_load_failure(self, embedding_client):
        embedding_client._gpu_poisoned = True
        mock_model = MagicMock()

        with patch.object(embedding_client, "is_model_cached", return_value=True):
            with patch(
                "sentence_transformers.SentenceTransformer",
                side_effect=[OSError("incomplete cache"), mock_model],
            ) as st:
                result = embedding_client.get_model("jina-code")

        assert result is mock_model
        assert st.call_count == 2
        assert st.call_args_list[0].kwargs["local_files_only"] is True
        assert st.call_args_list[1].kwargs["local_files_only"] is False
        assert st.call_args_list[1].kwargs["device"] == "cpu"

    def test_does_not_load_gpu_model_when_poisoned(self, embedding_client):
        """Poisoned client should not attempt to create a new GPU model."""
        embedding_client._gpu_poisoned = True
        embedding_client._cpu_fallback_models["jina-code"] = MagicMock()

        embedding_client.get_model("jina-code")

        # Should NOT have added to _models (GPU model dict)
        assert "jina-code" not in embedding_client._models

    def test_fastembed_model_not_affected_by_poison(self, embedding_client):
        """FastEmbed models (non-sentence-transformers) should load normally even when poisoned."""
        embedding_client._gpu_poisoned = True

        # bge-small is fastembed backend — get_model should try to load it normally
        # We mock the actual loading to avoid needing the model files
        mock_model = MagicMock()
        with patch('arcaneum.embeddings.client.TextEmbedding', return_value=mock_model):
            with patch.object(embedding_client, 'is_model_cached', return_value=True):
                result = embedding_client.get_model("bge-small")

        assert result is mock_model


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

        assert providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]

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


class TestSystemMemoryPressureGuard:
    """Low system memory disables accelerator work before starting a batch."""

    def test_low_available_memory_poisons_gpu_and_drops_model(self, embedding_client, monkeypatch):
        monkeypatch.setenv("ARC_MIN_SYSTEM_AVAILABLE_GB", "4")
        embedding_client._models["jina-code"] = MagicMock()

        with patch.object(embedding_client, "_system_memory_available_gb", return_value=2.5):
            disabled = embedding_client._maybe_disable_gpu_for_memory_pressure("jina-code")

        assert disabled is True
        assert embedding_client._gpu_poisoned is True
        assert "jina-code" not in embedding_client._models

    def test_healthy_available_memory_keeps_gpu_enabled(self, embedding_client, monkeypatch):
        monkeypatch.setenv("ARC_MIN_SYSTEM_AVAILABLE_GB", "4")
        embedding_client._models["jina-code"] = MagicMock()

        with patch.object(embedding_client, "_system_memory_available_gb", return_value=8.0):
            disabled = embedding_client._maybe_disable_gpu_for_memory_pressure("jina-code")

        assert disabled is False
        assert embedding_client._gpu_poisoned is False
        assert "jina-code" in embedding_client._models

    def test_invalid_memory_floor_uses_default(self, embedding_client, monkeypatch):
        monkeypatch.setenv("ARC_MIN_SYSTEM_AVAILABLE_GB", "invalid")

        assert embedding_client._min_system_available_gb() == 4.0

    def test_embed_falls_back_before_direct_encode_under_low_memory(
        self, embedding_client, monkeypatch
    ):
        monkeypatch.setenv("ARC_MIN_SYSTEM_AVAILABLE_GB", "4")
        embedding_client._models["jina-code"] = MagicMock()

        cpu_model = SimpleNamespace(_backend="sentence-transformers")

        with patch.object(embedding_client, "_system_memory_available_gb", return_value=2.5):
            with patch.object(embedding_client, "_get_cpu_fallback_model", return_value=cpu_model):
                with patch.object(embedding_client, "_encode_on_cpu_fallback") as mock_encode:
                    mock_encode.return_value = np.random.rand(1, 768).astype(np.float32)
                    with patch.object(embedding_client, "_validate_embeddings", return_value=True):
                        with patch("arcaneum.utils.memory.get_gpu_memory_info") as mock_gpu_mem:
                            embeddings = embedding_client.embed(["text"], "jina-code")

        assert embeddings.shape == (1, 768)
        assert embedding_client._gpu_poisoned is True
        assert "jina-code" not in embedding_client._models
        assert mock_encode.call_args.args[0] is cpu_model
        mock_gpu_mem.assert_not_called()

    def test_cpu_only_direct_embed_does_not_poison_gpu(self, embedding_client, monkeypatch):
        monkeypatch.setenv("ARC_MIN_SYSTEM_AVAILABLE_GB", "4")
        embedding_client.use_gpu = False

        mock_model = MagicMock()
        mock_model._backend = "sentence-transformers"
        embedding_client._models["jina-code"] = mock_model

        with patch.object(embedding_client, "_system_memory_available_gb", return_value=2.5):
            with patch.object(embedding_client, "_encode_with_oom_recovery") as mock_encode:
                mock_encode.return_value = np.random.rand(1, 768).astype(np.float32)
                with patch.object(embedding_client, "_validate_embeddings", return_value=True):
                    embeddings = embedding_client.embed(["text"], "jina-code")

        assert embeddings.shape == (1, 768)
        assert embedding_client._gpu_poisoned is False
        assert embedding_client._models["jina-code"] is mock_model


class TestDeferredGpuCleanup:
    """_try_deferred_gpu_cleanup() handles dead/alive daemon threads."""

    def test_cleanup_dead_thread(self, embedding_client):
        dead_thread = MagicMock(spec=threading.Thread)
        dead_thread.is_alive.return_value = False
        model_ref = MagicMock()

        embedding_client._pending_gpu_cleanup["jina-code"] = (dead_thread, model_ref)

        with patch.object(embedding_client, '_clear_gpu_cache'):
            result = embedding_client._try_deferred_gpu_cleanup()

        assert result is True
        assert "jina-code" not in embedding_client._pending_gpu_cleanup

    def test_no_cleanup_alive_thread(self, embedding_client):
        alive_thread = MagicMock(spec=threading.Thread)
        alive_thread.is_alive.return_value = True
        model_ref = MagicMock()

        embedding_client._pending_gpu_cleanup["jina-code"] = (alive_thread, model_ref)

        result = embedding_client._try_deferred_gpu_cleanup()

        assert result is False
        assert "jina-code" in embedding_client._pending_gpu_cleanup

    def test_no_pending_cleanup(self, embedding_client):
        result = embedding_client._try_deferred_gpu_cleanup()

        assert result is False

    def test_mixed_threads(self, embedding_client):
        dead_thread = MagicMock(spec=threading.Thread)
        dead_thread.is_alive.return_value = False
        alive_thread = MagicMock(spec=threading.Thread)
        alive_thread.is_alive.return_value = True

        embedding_client._pending_gpu_cleanup["model-a"] = (dead_thread, MagicMock())
        embedding_client._pending_gpu_cleanup["model-b"] = (alive_thread, MagicMock())

        with patch.object(embedding_client, '_clear_gpu_cache'):
            result = embedding_client._try_deferred_gpu_cleanup()

        assert result is True
        assert "model-a" not in embedding_client._pending_gpu_cleanup
        assert "model-b" in embedding_client._pending_gpu_cleanup


class TestEmbedImplCpuBatchSizingWhenPoisoned:
    """_embed_impl uses CPU batch sizing when poisoned (not GPU memory probing)."""

    def test_cpu_batch_sizing_when_poisoned(self, embedding_client):
        """When poisoned, should NOT call estimate_safe_batch_size_v2."""
        embedding_client._gpu_poisoned = True

        mock_model = MagicMock()
        mock_model._backend = "sentence-transformers"
        mock_model.encode.return_value = np.random.rand(2, 768).astype(np.float32)

        # Put the mock model in the client so get_model returns it
        embedding_client._cpu_fallback_models["jina-code"] = mock_model

        # Mock _encode_with_oom_recovery to capture that CPU path is used
        with patch.object(embedding_client, '_encode_with_oom_recovery') as mock_encode:
            mock_encode.return_value = np.random.rand(2, 768).astype(np.float32)
            with patch.object(embedding_client, '_validate_embeddings', return_value=True):
                with patch(
                    'arcaneum.utils.memory.get_gpu_memory_info'
                ) as mock_gpu_mem:
                    embedding_client._embed_impl(
                        ["text1", "text2"],
                        model_name="jina-code",
                        batch_size=32,
                    )

                    # GPU memory probing should NOT be called when poisoned
                    mock_gpu_mem.assert_not_called()


class TestGpuPoisonStaysSticky:
    """Completed cleanup does not re-enable GPU after timeout poisoning."""

    def test_deferred_cleanup_keeps_cpu_fallback(self, embedding_client):
        embedding_client._gpu_poisoned = True
        finished_thread = MagicMock(spec=threading.Thread)
        finished_thread.is_alive.return_value = False
        embedding_client._pending_gpu_cleanup["jina-code"] = (finished_thread, MagicMock())
        cpu_model = MagicMock()

        with patch.object(
            embedding_client,
            "_get_cpu_fallback_model",
            return_value=cpu_model,
        ) as get_cpu:
            with patch.object(embedding_client, "_encode_on_cpu_fallback") as encode_cpu:
                encode_cpu.return_value = np.random.rand(2, 768).astype(np.float32)

                result = embedding_client._encode_with_oom_recovery(
                    MagicMock(),
                    ["text1", "text2"],
                    internal_batch_size=8,
                    model_name="jina-code",
                )

        assert result is encode_cpu.return_value
        assert embedding_client._gpu_poisoned is True
        assert embedding_client._pending_gpu_cleanup == {}
        get_cpu.assert_called_once_with("jina-code")
        encode_cpu.assert_called_once_with(cpu_model, ["text1", "text2"])

    def test_poison_stays_set_with_alive_thread(self, embedding_client):
        embedding_client._gpu_poisoned = True
        alive_thread = MagicMock(spec=threading.Thread)
        alive_thread.is_alive.return_value = True
        embedding_client._pending_gpu_cleanup["jina-code"] = (alive_thread, MagicMock())
        cpu_model = MagicMock()

        with patch.object(embedding_client, "_get_cpu_fallback_model", return_value=cpu_model):
            with patch.object(embedding_client, "_encode_on_cpu_fallback") as encode_cpu:
                encode_cpu.return_value = np.random.rand(2, 768).astype(np.float32)

                embedding_client._encode_with_oom_recovery(
                    MagicMock(),
                    ["text1", "text2"],
                    internal_batch_size=8,
                    model_name="jina-code",
                )

        assert embedding_client._gpu_poisoned is True
        assert "jina-code" in embedding_client._pending_gpu_cleanup


class TestTimeoutHandlerReleasesModel:
    """try_encode() timeout handler pops GPU model and stores for deferred cleanup."""

    def test_model_popped_on_timeout(self, embedding_client):
        """When encode times out, GPU model is removed from _models and stored in _pending_gpu_cleanup."""
        mock_model = MagicMock()
        mock_model._backend = "sentence-transformers"
        embedding_client._models["jina-code"] = mock_model

        # Make the encode thread hang (never complete within timeout)
        original_thread_init = threading.Thread.__init__

        def mock_thread_join(self_thread, timeout=None):
            """Simulate thread not completing within timeout."""
            pass  # Don't actually wait

        mock_cpu_model = MagicMock()
        mock_cpu_model.encode.return_value = np.random.rand(2, 768).astype(np.float32)

        with patch.object(threading.Thread, 'join', mock_thread_join):
            with patch.object(threading.Thread, 'is_alive', return_value=True):
                with patch.object(
                    embedding_client, '_get_cpu_fallback_model',
                    return_value=mock_cpu_model
                ):
                    embedding_client._encode_with_oom_recovery(
                        mock_model, ["text1", "text2"],
                        internal_batch_size=8,
                        model_name="jina-code",
                        encode_timeout=0  # Immediate timeout
                    )

        # Model should be removed from _models
        assert "jina-code" not in embedding_client._models

        # Model should be in pending cleanup
        assert "jina-code" in embedding_client._pending_gpu_cleanup
        thread, model_ref = embedding_client._pending_gpu_cleanup["jina-code"]
        assert model_ref is mock_model

        # GPU should be poisoned
        assert embedding_client._gpu_poisoned is True


class TestAtexitJoinGpuThreads:
    """_atexit_join_gpu_threads() joins pending daemon threads before exit."""

    def test_joins_alive_thread(self, embedding_client):
        thread = MagicMock(spec=threading.Thread)
        thread.is_alive.return_value = True
        embedding_client._pending_gpu_cleanup["jina-code"] = (thread, MagicMock())

        embedding_client._atexit_join_gpu_threads()

        thread.join.assert_called_once_with(timeout=300)

    def test_skips_dead_thread(self, embedding_client):
        thread = MagicMock(spec=threading.Thread)
        thread.is_alive.return_value = False
        embedding_client._pending_gpu_cleanup["jina-code"] = (thread, MagicMock())

        embedding_client._atexit_join_gpu_threads()

        thread.join.assert_not_called()

    def test_noop_when_no_pending(self, embedding_client):
        """Should not raise when no pending cleanup, and leave state untouched."""
        embedding_client._pending_gpu_cleanup = {}

        embedding_client._atexit_join_gpu_threads()

        # Nothing to join, pending set remains empty
        assert embedding_client._pending_gpu_cleanup == {}


class TestCpuFallbackBounded:
    """_encode_on_cpu_fallback chunks work + caps thread counts."""

    def test_small_input_single_encode_call(self, embedding_client):
        cpu_model = MagicMock()
        cpu_model.encode.return_value = np.random.rand(2, 768).astype(np.float32)

        embedding_client._encode_on_cpu_fallback(cpu_model, ["a", "b"])

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
            cpu_model, [f"t{i}" for i in range(n)]
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

        with patch.object(
            embedding_client, '_configure_cpu_threading'
        ) as mock_configure:
            embedding_client._encode_on_cpu_fallback(cpu_model, ["a"])

        mock_configure.assert_called_once()


class TestCpuShortCircuit:
    """_encode_with_oom_recovery on CPU device runs inline, no daemon thread."""

    def test_cpu_device_skips_daemon_thread(self, embedding_client):
        """On CPU, the encode must not spawn a timeout thread — legitimate slow
        encodes trip the 120s timeout and spawn a second competing CPU encode."""
        embedding_client._device = "cpu"

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(3, 768).astype(np.float32)

        with patch('threading.Thread') as mock_thread_ctor:
            result = embedding_client._encode_with_oom_recovery(
                mock_model,
                ["a", "b", "c"],
                internal_batch_size=256,
                model_name="jina-code",
            )

        mock_thread_ctor.assert_not_called()
        assert result.shape == (3, 768)
        # Inner batch size must be the bounded CPU value, not the 256 passed in
        assert mock_model.encode.call_args.kwargs["batch_size"] == \
            embedding_client._CPU_FALLBACK_INNER_BATCH

    def test_cpu_device_does_not_poison(self, embedding_client):
        """Even if the underlying encode takes a long time, CPU path must never
        set _gpu_poisoned — there is no GPU to poison."""
        embedding_client._device = "cpu"

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)

        embedding_client._encode_with_oom_recovery(
            mock_model, ["a"],
            internal_batch_size=256,
            model_name="jina-code",
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
        embedding_client._models["jina-code"] = mock_model
        return mock_model

    def test_sorted_input_to_encode(self, embedding_client):
        """The texts handed to model.encode() are sorted shortest→longest."""
        mock_model = self._stub_st_model(embedding_client)

        # encode echoes the *input order* it was given as a 1-D float per item
        def encode_side_effect(input_texts, **kwargs):
            return np.array(
                [[float(len(t))] * 768 for t in input_texts], dtype=np.float32
            )

        mock_model.encode.side_effect = encode_side_effect

        # Mixed lengths in a deliberately scrambled order
        texts = ["xxxxxxxxxx", "x", "xxxxx", "xx", "xxxxxxxx"]
        # _encode_with_oom_recovery on MPS would spawn a daemon thread; bypass it
        with patch.object(
            embedding_client, '_encode_with_oom_recovery',
            side_effect=lambda model, t, ibs, mn: encode_side_effect(t),
        ) as mock_recover:
            result = embedding_client.embed(texts, "jina-code")

        # _encode_with_oom_recovery must have been handed length-sorted texts
        sent_texts = mock_recover.call_args.args[1]
        assert sent_texts == sorted(texts, key=len)

        # And the result must be in the *original* (caller) order
        assert result.shape == (5, 768)
        for i, t in enumerate(texts):
            # Each row's first value equals len(original_text_at_that_index)
            assert result[i][0] == float(len(t)), (
                f"Row {i} (text len {len(t)}) got value {result[i][0]} — "
                f"unsort failed"
            )

    def test_unsort_preserves_unique_embeddings(self, embedding_client):
        """No two texts of the same length must get crossed-up rows."""
        mock_model = self._stub_st_model(embedding_client)

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
            embedding_client, '_encode_with_oom_recovery',
            side_effect=lambda model, t, ibs, mn: encode_side_effect(t),
        ):
            result = embedding_client.embed(texts, "jina-code")

        # Each row's argmax should equal its original index, not the sorted index
        for i in range(3):
            assert int(np.argmax(result[i])) == i

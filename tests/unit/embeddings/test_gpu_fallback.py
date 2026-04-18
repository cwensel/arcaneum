"""Unit tests for GPU fallback stability (RDR-020)."""

import threading
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


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


class TestGpuRecovery:
    """_try_gpu_recovery() clears poison and releases CPU model."""

    def test_recovery_clears_poison(self, embedding_client):
        embedding_client._gpu_poisoned = True
        embedding_client._cpu_fallback_models["jina-code"] = MagicMock()

        result = embedding_client._try_gpu_recovery("jina-code")

        assert result is True
        assert embedding_client._gpu_poisoned is False
        assert "jina-code" not in embedding_client._cpu_fallback_models

    def test_recovery_not_poisoned(self, embedding_client):
        result = embedding_client._try_gpu_recovery("jina-code")

        assert result is False

    def test_recovery_increments_attempt_counter(self, embedding_client):
        embedding_client._gpu_poisoned = True

        embedding_client._try_gpu_recovery("jina-code")

        assert embedding_client._gpu_recovery_attempts == 1

    def test_recovery_blocked_by_alive_thread(self, embedding_client):
        embedding_client._gpu_poisoned = True
        alive_thread = MagicMock(spec=threading.Thread)
        alive_thread.is_alive.return_value = True
        embedding_client._pending_gpu_cleanup["jina-code"] = (alive_thread, MagicMock())

        result = embedding_client._try_gpu_recovery("jina-code")

        assert result is False
        assert embedding_client._gpu_poisoned is True


class TestGpuRecoveryLimit:
    """Second poisoning prevents further recovery attempts."""

    def test_second_recovery_rejected(self, embedding_client):
        embedding_client._gpu_poisoned = True
        embedding_client._gpu_recovery_attempts = 1  # Already used one attempt
        embedding_client._max_gpu_recovery_attempts = 1

        result = embedding_client._try_gpu_recovery("jina-code")

        assert result is False
        assert embedding_client._gpu_poisoned is True

    def test_first_recovery_allowed_second_rejected(self, embedding_client):
        # First recovery
        embedding_client._gpu_poisoned = True
        result1 = embedding_client._try_gpu_recovery("jina-code")
        assert result1 is True

        # Simulate GPU failing again
        embedding_client._gpu_poisoned = True

        # Second recovery should be rejected
        result2 = embedding_client._try_gpu_recovery("jina-code")
        assert result2 is False
        assert embedding_client._gpu_poisoned is True


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

"""Opt-in smoke tests for real accelerator embedding paths.

These tests intentionally use small public models and are skipped unless
ARC_RUN_GPU_SMOKE=1 is set. They are meant for self-hosted CUDA/MPS runners,
not the default pull-request matrix.
"""

import os
import threading
import time

import numpy as np
import pytest

pytestmark = pytest.mark.gpu_smoke


def _gpu_smoke_enabled() -> bool:
    return os.environ.get("ARC_RUN_GPU_SMOKE", "").lower() in {"1", "true", "yes", "on"}


def _require_gpu_smoke_accelerator() -> bool:
    return os.environ.get("ARC_REQUIRE_GPU_SMOKE_ACCELERATOR", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _sentence_transformer_model() -> str:
    return os.environ.get("ARC_GPU_SMOKE_MODEL", "minilm")


def _fastembed_model() -> str:
    return os.environ.get("ARC_GPU_SMOKE_FASTEMBED_MODEL", "bge-small")


def _assert_embeddings(embeddings, *, rows: int, dims: int) -> None:
    array = np.asarray(embeddings)
    assert array.shape == (rows, dims)
    assert np.isfinite(array).all()
    assert np.linalg.norm(array, axis=1).min() > 0


@pytest.fixture
def gpu_client():
    if not _gpu_smoke_enabled():
        pytest.skip("set ARC_RUN_GPU_SMOKE=1 to run accelerator smoke tests")

    from arcaneum.embeddings.client import EmbeddingClient
    from arcaneum.paths import get_models_dir

    cache_dir = os.path.expanduser(os.environ.get("ARC_GPU_SMOKE_CACHE_DIR", get_models_dir()))
    client = EmbeddingClient(cache_dir=cache_dir, use_gpu=True)
    if client.get_device_info()["device"] == "cpu":
        message = "no CUDA or MPS accelerator detected"
        if _require_gpu_smoke_accelerator():
            pytest.fail(message)
        pytest.skip(message)

    yield client

    deadline = time.monotonic() + 30
    while client._pending_gpu_cleanup and time.monotonic() < deadline:
        client._try_deferred_gpu_cleanup()
        if not client._pending_gpu_cleanup:
            break
        time.sleep(0.25)

    client._try_deferred_gpu_cleanup()
    live_cleanup = [
        name
        for name, (thread, _model_ref) in client._pending_gpu_cleanup.items()
        if thread.is_alive()
    ]
    if live_cleanup:
        pytest.fail(
            "GPU cleanup thread(s) still alive after smoke-test drain: "
            + ", ".join(sorted(live_cleanup))
        )
    assert not client._pending_gpu_cleanup


def test_sentence_transformer_small_encode_on_accelerator(gpu_client):
    from arcaneum.embeddings.client import EMBEDDING_MODELS

    model_name = _sentence_transformer_model()
    config = EMBEDDING_MODELS[model_name]
    assert config["backend"] == "sentence-transformers"

    texts = [
        "Arcaneum indexes code and documents for semantic search.",
        "This smoke test exercises a real accelerator encode path.",
    ]
    embeddings = gpu_client.embed(texts, model_name, batch_size=2, max_internal_batch=2)

    assert gpu_client.get_device_info()["device"] in {"mps", "cuda"}
    assert gpu_client._gpu_poisoned is False
    _assert_embeddings(embeddings, rows=len(texts), dims=config["dimensions"])


def test_low_memory_guard_falls_back_to_cached_cpu_model(gpu_client, monkeypatch):
    from arcaneum.embeddings.client import EMBEDDING_MODELS

    model_name = _sentence_transformer_model()
    config = EMBEDDING_MODELS[model_name]

    gpu_client.get_model(model_name)
    assert model_name in gpu_client._models

    monkeypatch.setenv("ARC_MIN_SYSTEM_AVAILABLE_GB", "999999")
    monkeypatch.setattr(gpu_client, "_system_memory_available_gb", lambda: 0.01)

    assert gpu_client._maybe_disable_gpu_for_memory_pressure(model_name) is True
    assert gpu_client._gpu_poisoned is True
    assert model_name not in gpu_client._models

    embeddings = gpu_client.embed(["CPU fallback after simulated memory pressure"], model_name)

    assert model_name in gpu_client._cpu_fallback_models
    _assert_embeddings(embeddings, rows=1, dims=config["dimensions"])


def test_fastembed_provider_selection_and_encode_smoke(gpu_client, monkeypatch):
    from arcaneum.embeddings.client import EMBEDDING_MODELS

    model_name = _fastembed_model()
    config = EMBEDDING_MODELS[model_name]
    assert config["backend"] == "fastembed"

    device = gpu_client.get_device_info()["device"]
    if device == "mps":
        monkeypatch.delenv("ARC_EXPERIMENTAL_COREML", raising=False)
        assert gpu_client._resolve_fastembed_providers(model_name) == ["CPUExecutionProvider"]
    else:
        assert gpu_client._resolve_fastembed_providers(model_name) is None

    embeddings = gpu_client.embed(["FastEmbed provider smoke test"], model_name)

    _assert_embeddings(embeddings, rows=1, dims=config["dimensions"])


def test_timeout_poison_path_falls_back_after_real_accelerator_thread(gpu_client, monkeypatch):
    from arcaneum.embeddings.client import EMBEDDING_MODELS

    model_name = _sentence_transformer_model()
    config = EMBEDDING_MODELS[model_name]
    model = gpu_client.get_model(model_name)
    original_encode = model.encode

    original_encode(
        ["Warm accelerator kernels before timeout smoke."],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    encode_completed = threading.Event()
    release_encode = threading.Event()

    def encode_then_block(*args, **kwargs):
        result = original_encode(*args, **kwargs)
        encode_completed.set()
        release_encode.wait(timeout=30)
        return result

    monkeypatch.setattr(model, "encode", encode_then_block)

    try:
        embeddings = gpu_client._encode_with_oom_recovery(
            model,
            ["Timeout poison smoke test"],
            internal_batch_size=1,
            model_name=model_name,
            encode_timeout=5,
        )

        assert encode_completed.wait(timeout=30)
        assert any(
            thread.is_alive() for thread, _model_ref in gpu_client._pending_gpu_cleanup.values()
        )
    finally:
        release_encode.set()

    assert gpu_client._gpu_poisoned is True
    assert model_name not in gpu_client._models
    assert model_name in gpu_client._cpu_fallback_models
    _assert_embeddings(embeddings, rows=1, dims=config["dimensions"])

    deadline = time.monotonic() + 30
    while gpu_client._pending_gpu_cleanup and time.monotonic() < deadline:
        gpu_client._try_deferred_gpu_cleanup()
        if not gpu_client._pending_gpu_cleanup:
            break
        time.sleep(0.25)

    assert not any(
        thread.is_alive() for thread, _model_ref in gpu_client._pending_gpu_cleanup.values()
    )

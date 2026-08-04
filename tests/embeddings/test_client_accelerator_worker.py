from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from arcaneum.embeddings.client import EmbeddingClient
from arcaneum.embeddings.worker_protocol import WorkerTimeoutError


def client():
    with patch("arcaneum.embeddings.client.get_models_dir", return_value="/tmp/models"):
        value = EmbeddingClient(cache_dir="/tmp/models", use_gpu=False)
    value.use_gpu = True
    value._device = "mps"
    return value


def test_accelerator_model_is_only_a_parent_marker():
    value = client()
    model = value.get_model("jina-code-st")

    assert model._backend == "sentence-transformers"
    assert not hasattr(model, "encode")
    assert "jina-code-st" not in value._models


def test_persistent_worker_is_reused_and_prompt_is_applied_in_parent():
    value = client()
    worker = MagicMock(is_alive=True, pid=1234)
    worker.encode.side_effect = [
        np.ones((1, 768), dtype=np.float32),
        np.ones((1, 768), dtype=np.float32) * 2,
    ]
    value._accelerator_workers["jina-code-st"] = worker

    first = value.embed(["alpha"], "jina-code-st", prompt_type="document")
    second = value.embed(["beta"], "jina-code-st", prompt_type="query")

    assert first.shape == second.shape == (1, 768)
    assert worker.encode.call_count == 2
    assert value._accelerator_workers["jina-code-st"] is worker
    document_text = worker.encode.call_args_list[0].args[0][0]
    query_text = worker.encode.call_args_list[1].args[0][0]
    assert document_text != query_text
    assert "alpha" in document_text
    assert "beta" in query_text


def test_timeout_reaps_worker_before_cpu_fallback():
    value = client()
    events = []
    worker = MagicMock(is_alive=True)
    worker.encode.side_effect = WorkerTimeoutError("hung")
    worker.shutdown.side_effect = lambda: events.append("reaped")
    value._accelerator_workers["jina-code-st"] = worker
    value._models["jina-code-st"] = object()
    cpu_model = SimpleNamespace(_backend="sentence-transformers")

    with patch.object(value, "_get_cpu_fallback_model", return_value=cpu_model):
        with patch.object(value, "_encode_on_cpu_fallback") as fallback:
            fallback.side_effect = lambda *args: events.append("cpu") or np.ones(
                (1, 768), dtype=np.float32
            )
            result = value._encode_with_oom_recovery(
                value.get_model("jina-code-st"),
                ["alpha"],
                8,
                "jina-code-st",
                encode_timeout=0.01,
            )

    assert result.shape == (1, 768)
    assert events == ["reaped", "cpu"]
    assert value._gpu_poisoned
    assert "jina-code-st" not in value._accelerator_workers
    assert "jina-code-st" not in value._models


def test_timeout_poison_is_sticky_and_next_encode_stays_on_cpu():
    value = client()
    worker = MagicMock(is_alive=True)
    worker.encode.side_effect = WorkerTimeoutError("hung")
    value._accelerator_workers["jina-code-st"] = worker
    cpu_model = SimpleNamespace(_backend="sentence-transformers")

    with patch.object(value, "_get_cpu_fallback_model", return_value=cpu_model):
        with patch.object(value, "_encode_on_cpu_fallback") as fallback:
            fallback.return_value = np.ones((1, 768), dtype=np.float32)
            value._encode_with_oom_recovery(
                value.get_model("jina-code-st"), ["first"], 8, "jina-code-st"
            )
            with patch.object(value, "_get_accelerator_worker") as get_worker:
                value._encode_with_oom_recovery(
                    value.get_model("jina-code-st"), ["second"], 8, "jina-code-st"
                )

    assert value._gpu_poisoned is True
    get_worker.assert_not_called()
    assert fallback.call_count == 2


def test_close_reaps_all_workers_and_is_idempotent():
    value = client()
    worker = MagicMock(is_alive=True)
    value._accelerator_workers["jina-code-st"] = worker

    value.close()
    value.close()

    worker.shutdown.assert_called_once_with()
    assert value._accelerator_workers == {}

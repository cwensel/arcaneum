import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from arcaneum.benchmarks.cuda import (
    SOAK_TARGET_SECONDS,
    SOAK_TARGET_TEXTS,
    _empty_result,
    _encode_scheduled,
)
from arcaneum.embeddings.sentence_transformer_worker import (
    SentenceTransformerAcceleratorBackend,
)

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "benchmarks" / "fixtures" / "accelerator-v1" / "manifest.json"
SCHEMA = ROOT / "benchmarks" / "schema" / "accelerator-result-v1.schema.json"


def test_inconclusive_cuda_result_has_zero_measurements_and_exact_reason():
    reason = "PyTorch CUDA is not available; qualification cannot run"
    result = _empty_result(MANIFEST, "jina-code-st", reason, token_budget=4096, batch_size=4)
    schema = json.loads(SCHEMA.read_text())

    assert set(schema["required"]) <= result.keys()
    assert result["schema_version"] == schema["properties"]["schema_version"]["const"]
    assert set(schema["properties"]["qualification"]["required"]) <= result["qualification"].keys()
    assert result["run"]["status"] == "inconclusive"
    assert result["qualification"]["decision"] == "experimental"
    assert result["qualification"]["reason"] == reason
    assert result["performance"]["throughput_texts_per_second"] == 0.0
    assert result["performance"]["cuda_peak_allocated_bytes"] is None
    assert result["qualification"]["soak_target_texts"] == SOAK_TARGET_TEXTS
    assert result["qualification"]["soak_target_seconds"] == SOAK_TARGET_SECONDS
    assert result["qualification"]["token_budget"]["max_actual_tokens"] == 4096


def test_token_budgeted_encode_restores_original_order():
    class Worker:
        def encode(self, texts, **_options):
            return np.array([[float(len(text))] for text in texts], dtype=np.float32)

    texts = ["longer text here", "x", "medium"]
    result, batch_count = _encode_scheduled(
        Worker(), texts, timeout=1, token_budget=8, batch_size=2
    )

    assert result[:, 0].tolist() == [16.0, 1.0, 6.0]
    assert batch_count >= 2


def test_cuda_oom_retry_is_bounded_to_two_retries():
    backend = SentenceTransformerAcceleratorBackend.__new__(SentenceTransformerAcceleratorBackend)
    backend._encodes = 0
    backend._oom_retries = 0
    backend._clear = lambda: None
    backend.model = MagicMock(max_seq_length=512)
    backend.model.tokenizer.encode.return_value = [1, 2]
    attempts = []

    def fail(_texts, batch_size):
        attempts.append(batch_size)
        raise RuntimeError("CUDA out of memory")

    backend._encode_once = fail

    try:
        backend.encode(["text"], batch_size=8)
    except RuntimeError as exc:
        assert "memory exhausted" in str(exc)
    else:
        raise AssertionError("bounded OOM retry must fail after its retry budget")

    assert attempts == [8, 1, 1]
    assert backend._oom_retries == 2

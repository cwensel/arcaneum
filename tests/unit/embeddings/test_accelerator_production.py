import sys
from unittest.mock import MagicMock

import numpy as np

from arcaneum.embeddings.client import EmbeddingClient
from arcaneum.embeddings.sentence_transformer_worker import SentenceTransformerAcceleratorBackend


def test_worker_forwards_task_and_prompt_name_policy():
    backend = SentenceTransformerAcceleratorBackend.__new__(SentenceTransformerAcceleratorBackend)
    backend._encodes = backend._oom_retries = 0
    captured = {}

    def encode_once(texts, batch_size, **policy):
        captured.update(policy)
        return np.ones((len(texts), 2), dtype=np.float32)

    backend._encode_once = encode_once
    backend.encode(["query"], batch_size=1, task="retrieval.query", prompt_name="query")
    assert captured == {"task": "retrieval.query", "prompt_name": "query"}


def test_production_scheduler_buckets_mixed_lengths_and_restores_order(monkeypatch):
    client = EmbeddingClient.__new__(EmbeddingClient)
    client._gpu_poisoned = False
    client._device = "mps"
    client._worker_failure_count = 0
    client._backend_fallback_reasons = {}
    client._models = {}
    worker = MagicMock()
    worker.encode.side_effect = lambda texts, **kwargs: np.asarray(
        [[float(len(text)), float(ord(text[0]))] for text in texts], dtype=np.float32
    )
    monkeypatch.setattr(client, "_get_accelerator_worker", lambda model: worker)
    monkeypatch.setattr(client, "_validate_embeddings", lambda *args: True)
    texts = ["z" * 100, "a", "middle"]
    actual = client._encode_with_oom_recovery(
        object(), texts, 2, "jina-code-st", prompt_type="document"
    )
    assert actual[:, 0].tolist() == [100, 1, 6]
    assert worker.encode.call_count >= 2
    assert all(call.kwargs["task"] == "retrieval.passage" for call in worker.encode.call_args_list)


def test_memory_probe_does_not_import_torch(monkeypatch):
    sys.modules.pop("torch", None)
    from arcaneum.embeddings import memory_probe

    memory_probe.snapshot()
    assert "torch" not in sys.modules


def test_cpu_fastembed_never_uses_auto_provider(tmp_path):
    client = EmbeddingClient(cache_dir=str(tmp_path), use_gpu=False)
    try:
        assert client._resolve_fastembed_providers("bge-small") == ["CPUExecutionProvider"]
    finally:
        client.close()


def test_fastembed_production_scheduler_restores_mixed_length_order(monkeypatch):
    client = EmbeddingClient.__new__(EmbeddingClient)
    client._device = "cpu"
    client.use_gpu = False
    client._gpu_poisoned = False
    model = MagicMock()
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return [np.asarray([len(text), ord(text[0])], dtype=np.float32) for text in texts]

    model.embed.side_effect = embed
    monkeypatch.setattr(client, "get_model", lambda name: model)
    monkeypatch.setattr(client, "_validate_embeddings", lambda *args: True)
    actual = client._embed_impl(["z" * 100, "a", "middle"], "bge-small", batch_size=2)
    assert actual[:, 0].tolist() == [100, 1, 6]
    assert len(calls) >= 2

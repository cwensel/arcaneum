import sys
from unittest.mock import MagicMock

import numpy as np

from arcaneum.embeddings.client import EmbeddingClient
from arcaneum.embeddings.sentence_transformer_worker import SentenceTransformerAcceleratorBackend


class Encoding:
    def __init__(self, text):
        self.ids = [ord(char) for char in text]
        self.attention_mask = [1] * len(self.ids)


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        ids = [ord(char) for char in text]
        return [101, *ids, 102] if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(value) for value in ids if value not in {101, 102})


def test_worker_forwards_task_and_prompt_name_policy():
    backend = SentenceTransformerAcceleratorBackend.__new__(SentenceTransformerAcceleratorBackend)
    backend._encodes = backend._oom_retries = 0
    backend.model = MagicMock(max_seq_length=512, tokenizer=Tokenizer())
    captured = {}

    def encode_once(texts, batch_size, **policy):
        captured.update(policy)
        return np.ones((len(texts), 2), dtype=np.float32)

    backend._encode_once = encode_once
    backend.encode(["query"], batch_size=1, task="retrieval.query", prompt_name="query")
    assert captured == {"task": "retrieval.query", "prompt_name": "query"}


def test_parent_passes_true_token_budget_to_child_without_tokenizing(monkeypatch):
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
    worker.encode.assert_called_once()
    assert worker.encode.call_args.kwargs["task"] == "retrieval.passage"
    assert worker.encode.call_args.kwargs["max_sequence_tokens"] == 2048
    assert worker.encode.call_args.kwargs["token_budget"] == 4096


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
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text, add_special_tokens=True: Encoding(text)
    tokenizer.decode.side_effect = lambda ids, skip_special_tokens=True: "".join(
        chr(value) for value in ids
    )
    model.model.tokenizer = tokenizer
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


def test_child_tokenizer_schedules_high_density_and_multibyte_in_true_tokens():
    backend = SentenceTransformerAcceleratorBackend.__new__(SentenceTransformerAcceleratorBackend)
    backend._encodes = backend._oom_retries = 0
    backend.model = MagicMock(max_seq_length=8, tokenizer=Tokenizer())
    calls = []

    def encode_once(texts, batch_size, **policy):
        calls.append(list(texts))
        return np.asarray([[len(text)] for text in texts], dtype=np.float32)

    backend._encode_once = encode_once
    actual = backend.encode(
        ["界界界界", "abcdef", "x"], batch_size=2, max_sequence_tokens=8, token_budget=10
    )
    assert actual[:, 0].tolist() == [4, 6, 1]
    assert len(calls) >= 2

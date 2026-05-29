"""Regression tests for query/document prompt policy enforcement."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from arcaneum.embeddings.client import (
    EMBEDDING_MODELS,
    EmbeddingClient,
    get_embedding_prompt_policy,
)
from arcaneum.search.embedder import SearchEmbedder


def _client(metadata):
    client = MagicMock()
    state = {"metadata": dict(metadata)}
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(vectors={"e5-base": SimpleNamespace(size=768)})
        )
    )

    def retrieve(collection_name, ids, with_payload, with_vectors):
        return [SimpleNamespace(payload=state["metadata"])]

    def upsert(collection_name, points):
        state["metadata"] = points[0].payload

    client.retrieve.side_effect = retrieve
    client.upsert.side_effect = upsert
    return client


def test_sentence_transformer_query_embedding_uses_query_prompt_type():
    embedding_client = MagicMock()
    embedding_client.get_model.return_value = object()
    embedding_client.embed.return_value = [[0.1, 0.2]]

    embedder = SearchEmbedder.__new__(SearchEmbedder)
    embedder._embedding_client = embedding_client

    model_key, vector = embedder.generate_query_embedding(
        "auth checks",
        "docs",
        _client({"embedding_prompt_policy": {"e5-base": get_embedding_prompt_policy("e5-base")}}),
    )

    assert model_key == "e5-base"
    assert vector == [0.1, 0.2]
    embedding_client.embed.assert_called_once_with(
        ["auth checks"],
        "e5-base",
        prompt_type="query",
    )


def test_query_embedding_backfills_collections_without_prompt_policy():
    embedding_client = MagicMock()
    embedding_client.get_model.return_value = object()
    embedding_client.embed.return_value = [[0.1, 0.2]]

    embedder = SearchEmbedder.__new__(SearchEmbedder)
    embedder._embedding_client = embedding_client
    client = _client({})

    model_key, vector = embedder.generate_query_embedding("auth checks", "docs", client)

    assert model_key == "e5-base"
    assert vector == [0.1, 0.2]
    client.upsert.assert_called_once()


def test_query_embedding_rejects_changed_prompt_policy():
    embedder = SearchEmbedder.__new__(SearchEmbedder)
    embedder._embedding_client = MagicMock()
    stale_policy = {
        **get_embedding_prompt_policy("e5-base"),
        "document": {"method": "encode"},
    }

    with pytest.raises(ValueError, match="differs"):
        embedder.generate_query_embedding(
            "auth checks",
            "docs",
            _client({"embedding_prompt_policy": {"e5-base": stale_policy}}),
        )


def test_prompt_prefix_length_is_reserved_when_clipping(monkeypatch):
    config = {**EMBEDDING_MODELS["e5-base"], "max_seq_length": 8}
    monkeypatch.setitem(EMBEDDING_MODELS, "e5-base", config)

    client = EmbeddingClient.__new__(EmbeddingClient)
    client.use_gpu = False
    client._device = "cpu"
    client._gpu_poisoned = False
    client.get_model = MagicMock(return_value=SimpleNamespace(_backend="sentence-transformers"))
    captured = {}

    def fake_encode(_model, texts, _model_name, _prompt_type):
        captured["texts"] = texts
        return [[0.1, 0.2]]

    client._encode_on_cpu_fallback = fake_encode
    client._validate_embeddings = MagicMock(return_value=True)

    result = client.embed(["x" * 20], "e5-base", prompt_type="document")

    assert result.tolist() == [[0.1, 0.2]]
    assert captured["texts"] == ["passage: " + ("x" * 7)]
    assert len(captured["texts"][0]) == 16


def test_near_limit_chunks_log_preserved_not_warning(monkeypatch, caplog):
    config = {**EMBEDDING_MODELS["stella"], "max_seq_length": 10}
    monkeypatch.setitem(EMBEDDING_MODELS, "stella", config)

    client = EmbeddingClient.__new__(EmbeddingClient)
    client.use_gpu = False
    client._device = "cpu"
    client._gpu_poisoned = False
    client._maybe_disable_gpu_for_memory_pressure = MagicMock()
    client.get_model = MagicMock(return_value=SimpleNamespace(_backend="sentence-transformers"))
    client._encode_on_cpu_fallback = MagicMock(return_value=np.array([[0.1, 0.2]]))
    client._validate_embeddings = MagicMock(return_value=True)

    with caplog.at_level(logging.INFO, logger="arcaneum.embeddings.client"):
        result = client.embed(["x" * 17], "stella", prompt_type="document")

    assert result.tolist() == [[0.1, 0.2]]
    assert "no clipping was needed" in caplog.text
    assert "Upstream chunking/windowing preserved the full text" in caplog.text
    assert "Large chunks detected" not in caplog.text


def test_oversized_chunks_warn_when_embedding_clips(monkeypatch, caplog):
    config = {**EMBEDDING_MODELS["stella"], "max_seq_length": 10}
    monkeypatch.setitem(EMBEDDING_MODELS, "stella", config)

    client = EmbeddingClient.__new__(EmbeddingClient)
    client.use_gpu = False
    client._device = "cpu"
    client._gpu_poisoned = False
    client._maybe_disable_gpu_for_memory_pressure = MagicMock()
    client.get_model = MagicMock(return_value=SimpleNamespace(_backend="sentence-transformers"))
    captured = {}

    def fake_encode(_model, texts, _model_name, _prompt_type):
        captured["texts"] = texts
        return np.array([[0.1, 0.2]])

    client._encode_on_cpu_fallback = fake_encode
    client._validate_embeddings = MagicMock(return_value=True)

    with caplog.at_level(logging.WARNING, logger="arcaneum.embeddings.client"):
        result = client.embed(["x" * 25], "stella", prompt_type="document")

    assert result.tolist() == [[0.1, 0.2]]
    assert captured["texts"] == ["x" * 20]
    assert "Embedding safety clipped 1/1 oversized texts" in caplog.text
    assert "content beyond 20 chars is not represented in vectors" in caplog.text

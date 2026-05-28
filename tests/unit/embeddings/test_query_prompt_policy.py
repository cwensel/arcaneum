"""Regression tests for query/document prompt policy enforcement."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from arcaneum.embeddings.client import get_embedding_prompt_policy
from arcaneum.search.embedder import SearchEmbedder


def _client(metadata):
    client = MagicMock()
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(vectors={"e5-base": SimpleNamespace(size=768)})
        )
    )
    client.retrieve.return_value = [SimpleNamespace(payload=metadata)]
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


def test_query_embedding_rejects_collections_without_prompt_policy():
    embedder = SearchEmbedder.__new__(SearchEmbedder)
    embedder._embedding_client = MagicMock()

    with pytest.raises(ValueError, match="missing embedding_prompt_policy"):
        embedder.generate_query_embedding("auth checks", "docs", _client({}))

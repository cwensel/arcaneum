"""A corpus records its text backend on the collection metadata point (RDR-027 Q2)."""

from __future__ import annotations

import pytest

from arcaneum.indexing.collection_metadata import (
    get_collection_metadata,
    set_collection_metadata,
    text_backend_of,
    update_collection_metadata,
)


@pytest.fixture
def client():
    from qdrant_client import QdrantClient
    from qdrant_client.http import models

    client = QdrantClient(location=":memory:")
    client.create_collection(
        "Docs",
        vectors_config={"stub": models.VectorParams(size=2, distance=models.Distance.COSINE)},
    )
    return client


def test_an_absent_key_means_meili():
    assert text_backend_of({}) == "meili"
    assert text_backend_of({"text_backend": None}) == "meili"


@pytest.mark.parametrize("backend", ["meili", "qdrant"])
def test_create_records_the_backend(client, backend):
    set_collection_metadata(client, "Docs", "pdf", "stella", text_backend=backend)

    metadata = get_collection_metadata(client, "Docs")
    assert metadata["text_backend"] == backend
    assert text_backend_of(metadata) == backend


def test_an_unknown_backend_is_refused_before_any_write(client):
    with pytest.raises(ValueError, match="text backend"):
        set_collection_metadata(client, "Docs", "pdf", "stella", text_backend="sqlite")
    assert get_collection_metadata(client, "Docs") == {}


def test_an_update_validates_the_backend(client):
    set_collection_metadata(client, "Docs", "pdf", "stella")

    with pytest.raises(ValueError, match="text backend"):
        update_collection_metadata(client, "Docs", text_backend="elastic")
    assert update_collection_metadata(client, "Docs", text_backend="qdrant")["text_backend"] == (
        "qdrant"
    )


def test_a_stored_unknown_value_is_an_error():
    with pytest.raises(ValueError, match="sqlite"):
        text_backend_of({"text_backend": "sqlite"})

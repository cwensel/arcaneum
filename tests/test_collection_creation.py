"""Integration tests for collection creation (RDR-003)."""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance


QDRANT_URL = "http://localhost:6333"


@pytest.fixture
def qdrant_client():
    """Provide Qdrant client connected to test server."""
    client = QdrantClient(url=QDRANT_URL)
    yield client
    # Cleanup: delete test collections
    try:
        collections = client.get_collections()
        for col in collections.collections:
            if col.name.startswith("test_"):
                client.delete_collection(col.name)
    except Exception:
        pass


def test_qdrant_connection(qdrant_client):
    """Test that Qdrant server is accessible."""
    collections = qdrant_client.get_collections()
    assert collections is not None


def test_create_single_vector_collection(qdrant_client):
    """Test creating a collection with a single vector."""
    from qdrant_client.models import VectorParams, HnswConfigDiff

    collection_name = "test_single_vector"

    # Create collection
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "stella": VectorParams(size=1024, distance=Distance.COSINE)
        },
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        on_disk_payload=True,
    )

    # Verify collection exists
    info = qdrant_client.get_collection(collection_name)
    assert info.points_count == 0
    assert hasattr(info.config.params, 'vectors')
    assert 'stella' in info.config.params.vectors
    assert info.config.params.vectors['stella'].size == 1024
    assert info.config.params.vectors['stella'].distance == Distance.COSINE


def test_create_multi_vector_collection(qdrant_client):
    """Test creating a collection with multiple named vectors."""
    from qdrant_client.models import VectorParams, HnswConfigDiff

    collection_name = "test_multi_vector"

    # Create collection with multiple vectors
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "stella": VectorParams(size=1024, distance=Distance.COSINE),
            "jina": VectorParams(size=768, distance=Distance.COSINE),
            "bge": VectorParams(size=1024, distance=Distance.COSINE),
        },
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        on_disk_payload=True,
    )

    # Verify collection structure
    info = qdrant_client.get_collection(collection_name)
    assert hasattr(info.config.params, 'vectors')
    assert len(info.config.params.vectors) == 3
    assert 'stella' in info.config.params.vectors
    assert 'jina' in info.config.params.vectors
    assert 'bge' in info.config.params.vectors

    # Verify dimensions
    assert info.config.params.vectors['stella'].size == 1024
    assert info.config.params.vectors['jina'].size == 768
    assert info.config.params.vectors['bge'].size == 1024


def test_create_collection_with_indexes(qdrant_client):
    """Test creating a collection with payload indexes."""
    from qdrant_client.models import VectorParams, HnswConfigDiff, PayloadSchemaType

    collection_name = "test_with_indexes"

    # Create collection
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "stella": VectorParams(size=1024, distance=Distance.COSINE)
        },
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        on_disk_payload=True,
    )

    # Create payload indexes
    qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="programming_language",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="file_extension",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    # Verify collection exists
    info = qdrant_client.get_collection(collection_name)
    assert info is not None


def test_hnsw_configuration(qdrant_client):
    """Test HNSW index configuration."""
    from qdrant_client.models import VectorParams, HnswConfigDiff

    collection_name = "test_hnsw_config"

    # Create collection with custom HNSW settings
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "stella": VectorParams(size=1024, distance=Distance.COSINE)
        },
        hnsw_config=HnswConfigDiff(m=32, ef_construct=200),
        on_disk_payload=False,
    )

    # Verify HNSW configuration
    info = qdrant_client.get_collection(collection_name)
    assert info.config.hnsw_config.m == 32
    assert info.config.hnsw_config.ef_construct == 200


def test_list_collections(qdrant_client):
    """Test listing all collections."""
    from qdrant_client.models import VectorParams, HnswConfigDiff

    # Create test collections
    for i in range(3):
        collection_name = f"test_list_{i}"
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "stella": VectorParams(size=1024, distance=Distance.COSINE)
            },
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )

    # List collections
    collections = qdrant_client.get_collections()
    collection_names = [col.name for col in collections.collections]

    # Verify test collections are present
    assert "test_list_0" in collection_names
    assert "test_list_1" in collection_names
    assert "test_list_2" in collection_names


def test_delete_collection(qdrant_client):
    """Test deleting a collection."""
    from qdrant_client.models import VectorParams, HnswConfigDiff

    collection_name = "test_delete"

    # Create collection
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "stella": VectorParams(size=1024, distance=Distance.COSINE)
        },
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
    )

    # Verify it exists
    info = qdrant_client.get_collection(collection_name)
    assert info is not None

    # Delete collection
    qdrant_client.delete_collection(collection_name)

    # Verify it's gone
    collections = qdrant_client.get_collections()
    collection_names = [col.name for col in collections.collections]
    assert collection_name not in collection_names

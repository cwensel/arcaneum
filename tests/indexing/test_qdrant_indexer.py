"""Tests for Qdrant indexer module."""

from unittest.mock import Mock, MagicMock, patch
import pytest

from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from arcaneum.indexing.qdrant_indexer import QdrantIndexer, create_qdrant_client
from arcaneum.indexing.types import CodeChunk, CodeChunkMetadata


@pytest.fixture
def mock_qdrant():
    """Create a mock Qdrant client."""
    return Mock(spec=QdrantClient)


@pytest.fixture
def indexer(mock_qdrant):
    """Create a QdrantIndexer with mocked client."""
    return QdrantIndexer(mock_qdrant)


def create_test_chunk(identifier="project#main", commit="a" * 40):
    """Helper to create a test CodeChunk with embedding."""
    metadata = CodeChunkMetadata(
        git_project_identifier=identifier,
        file_path="/repo/test.py",
        filename="test.py",
        file_extension=".py",
        programming_language="python",
        file_size=100,
        line_count=10,
        chunk_index=0,
        chunk_count=1,
        text_extraction_method="ast_python",
        git_project_root="/repo",
        git_project_name=identifier.split("#")[0],
        git_branch=identifier.split("#")[1],
        git_commit_hash=commit
    )

    chunk = CodeChunk(
        content="def test(): pass",
        metadata=metadata,
        embedding=[0.1] * 768  # Mock 768D embedding
    )

    return chunk


class TestQdrantIndexer:
    """Tests for QdrantIndexer class."""

    def test_initialization(self, mock_qdrant):
        """Test basic initialization."""
        indexer = QdrantIndexer(mock_qdrant, batch_size=200)

        assert indexer.client == mock_qdrant
        assert indexer.batch_size == 200

    def test_initialization_default_batch_size(self, mock_qdrant):
        """Test initialization with default batch size."""
        indexer = QdrantIndexer(mock_qdrant)

        assert indexer.batch_size == 150

    def test_delete_branch_chunks(self, indexer, mock_qdrant):
        """Test filter-based branch deletion."""
        mock_result = Mock()
        mock_result.operation_id = "test-op-123"
        mock_qdrant.delete.return_value = mock_result

        result = indexer.delete_branch_chunks("test-collection", "project#main")

        # Should call delete with filter
        mock_qdrant.delete.assert_called_once()
        call_args = mock_qdrant.delete.call_args

        assert call_args.kwargs["collection_name"] == "test-collection"
        assert call_args.kwargs["points_selector"] is not None

    def test_upload_chunks_batch_empty(self, indexer, mock_qdrant):
        """Test uploading empty batch."""
        result = indexer.upload_chunks_batch("test-collection", [])

        assert result == 0
        mock_qdrant.upsert.assert_not_called()

    def test_upload_chunks_batch_single(self, indexer, mock_qdrant):
        """Test uploading single chunk."""
        chunk = create_test_chunk()

        result = indexer.upload_chunks_batch("test-collection", [chunk])

        assert result == 1
        mock_qdrant.upsert.assert_called_once()

        # Verify upsert was called with correct args
        call_args = mock_qdrant.upsert.call_args
        assert call_args.kwargs["collection_name"] == "test-collection"
        assert len(call_args.kwargs["points"]) == 1

    def test_upload_chunks_batch_multiple(self, indexer, mock_qdrant):
        """Test uploading multiple chunks."""
        chunks = [create_test_chunk(f"project{i}#main") for i in range(5)]

        result = indexer.upload_chunks_batch("test-collection", chunks)

        assert result == 5
        mock_qdrant.upsert.assert_called_once()

        call_args = mock_qdrant.upsert.call_args
        assert len(call_args.kwargs["points"]) == 5

    def test_upload_chunks_batch_missing_embedding(self, indexer, mock_qdrant):
        """Test that missing embeddings raise error."""
        chunk = create_test_chunk()
        chunk.embedding = None  # Remove embedding

        with pytest.raises(ValueError, match="missing embedding"):
            indexer.upload_chunks_batch("test-collection", [chunk])

    def test_upload_chunks_batch_with_wait(self, indexer, mock_qdrant):
        """Test upload with wait parameter."""
        chunk = create_test_chunk()

        indexer.upload_chunks_batch("test-collection", [chunk], wait=False)

        call_args = mock_qdrant.upsert.call_args
        assert call_args.kwargs["wait"] is False

    def test_upload_chunks_single_batch(self, indexer, mock_qdrant):
        """Test upload_chunks with single batch."""
        chunks = [create_test_chunk() for _ in range(100)]

        result = indexer.upload_chunks("test-collection", chunks)

        assert result == 100
        # Should fit in one batch (default 150)
        assert mock_qdrant.upsert.call_count == 1

    def test_upload_chunks_multiple_batches(self, indexer, mock_qdrant):
        """Test upload_chunks with multiple batches."""
        # Create 300 chunks (should be 2 batches with batch_size=150)
        chunks = [create_test_chunk() for _ in range(300)]

        result = indexer.upload_chunks("test-collection", chunks)

        assert result == 300
        # Should be 2 batches
        assert mock_qdrant.upsert.call_count == 2

    def test_upload_chunks_empty(self, indexer, mock_qdrant):
        """Test upload_chunks with empty list."""
        result = indexer.upload_chunks("test-collection", [])

        assert result == 0
        mock_qdrant.upsert.assert_not_called()

    def test_create_collection(self, indexer, mock_qdrant):
        """Test creating a collection."""
        indexer.create_collection("test-collection", vector_size=768)

        mock_qdrant.create_collection.assert_called_once()
        call_args = mock_qdrant.create_collection.call_args

        assert call_args.kwargs["collection_name"] == "test-collection"
        assert call_args.kwargs["vectors_config"].size == 768
        assert call_args.kwargs["vectors_config"].distance == Distance.COSINE

    def test_create_collection_custom_distance(self, indexer, mock_qdrant):
        """Test creating collection with custom distance metric."""
        indexer.create_collection(
            "test-collection",
            vector_size=1536,
            distance=Distance.DOT
        )

        call_args = mock_qdrant.create_collection.call_args
        assert call_args.kwargs["vectors_config"].distance == Distance.DOT

    def test_collection_exists_true(self, indexer, mock_qdrant):
        """Test checking if collection exists (true)."""
        mock_collection = Mock()
        mock_collection.name = "test-collection"
        mock_collections = Mock()
        mock_collections.collections = [mock_collection]
        mock_qdrant.get_collections.return_value = mock_collections

        exists = indexer.collection_exists("test-collection")

        assert exists is True

    def test_collection_exists_false(self, indexer, mock_qdrant):
        """Test checking if collection exists (false)."""
        mock_collection = Mock()
        mock_collection.name = "other-collection"
        mock_collections = Mock()
        mock_collections.collections = [mock_collection]
        mock_qdrant.get_collections.return_value = mock_collections

        exists = indexer.collection_exists("test-collection")

        assert exists is False

    def test_get_collection_info(self, indexer, mock_qdrant):
        """Test getting collection info."""
        mock_info = Mock()
        mock_info.config.params.vectors.size = 768
        mock_info.config.params.vectors.distance = Distance.COSINE
        mock_info.points_count = 1000
        mock_info.indexed_vectors_count = 1000
        mock_qdrant.get_collection.return_value = mock_info

        info = indexer.get_collection_info("test-collection")

        assert info["name"] == "test-collection"
        assert info["vector_size"] == 768
        assert info["distance"] == Distance.COSINE
        assert info["points_count"] == 1000

    def test_count_chunks_for_project(self, indexer, mock_qdrant):
        """Test counting chunks for a project."""
        mock_result = Mock()
        mock_result.count = 42
        mock_qdrant.count.return_value = mock_result

        count = indexer.count_chunks_for_project("test-collection", "project#main")

        assert count == 42
        mock_qdrant.count.assert_called_once()

    def test_count_chunks_for_project_error(self, indexer, mock_qdrant):
        """Test error handling when counting fails."""
        mock_qdrant.count.side_effect = Exception("Connection error")

        count = indexer.count_chunks_for_project("test-collection", "project#main")

        # Should return 0 on error
        assert count == 0

    def test_delete_collection(self, indexer, mock_qdrant):
        """Test deleting a collection."""
        indexer.delete_collection("test-collection")

        mock_qdrant.delete_collection.assert_called_once_with("test-collection")

    def test_retry_on_failure(self, indexer, mock_qdrant):
        """Test that upload retries on failure."""
        chunk = create_test_chunk()

        # Fail twice, then succeed
        mock_qdrant.upsert.side_effect = [
            Exception("Network error"),
            Exception("Network error"),
            None  # Success on third try
        ]

        result = indexer.upload_chunks_batch("test-collection", [chunk])

        assert result == 1
        # Should have been called 3 times
        assert mock_qdrant.upsert.call_count == 3

    def test_retry_exhausted(self, indexer, mock_qdrant):
        """Test that retries are exhausted after max attempts."""
        chunk = create_test_chunk()

        # Always fail
        mock_qdrant.upsert.side_effect = Exception("Network error")

        with pytest.raises(Exception, match="Network error"):
            indexer.upload_chunks_batch("test-collection", [chunk])

        # Should have been called MAX_RETRIES times (3)
        assert mock_qdrant.upsert.call_count == 3


class TestCreateQdrantClient:
    """Tests for create_qdrant_client helper function."""

    @patch('arcaneum.indexing.qdrant_indexer.QdrantClient')
    def test_create_client_with_grpc(self, mock_client_class):
        """Test creating client with gRPC enabled."""
        create_qdrant_client(
            url="localhost",
            port=6333,
            grpc_port=6334,
            prefer_grpc=True
        )

        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args.kwargs

        assert call_kwargs["host"] == "localhost"
        assert call_kwargs["port"] == 6333
        assert call_kwargs["grpc_port"] == 6334
        assert call_kwargs["prefer_grpc"] is True

    @patch('arcaneum.indexing.qdrant_indexer.QdrantClient')
    def test_create_client_without_grpc(self, mock_client_class):
        """Test creating client with gRPC disabled."""
        create_qdrant_client(
            url="localhost",
            port=6333,
            prefer_grpc=False
        )

        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args.kwargs

        assert call_kwargs["host"] == "localhost"
        assert call_kwargs["port"] == 6333
        assert "grpc_port" not in call_kwargs
        assert "prefer_grpc" not in call_kwargs

    @patch('arcaneum.indexing.qdrant_indexer.QdrantClient')
    def test_create_client_with_api_key(self, mock_client_class):
        """Test creating client with API key."""
        create_qdrant_client(
            url="cloud.qdrant.io",
            api_key="test-key-123"
        )

        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["api_key"] == "test-key-123"


class TestCodeChunkConversion:
    """Tests for CodeChunk to PointStruct conversion."""

    def test_chunk_to_point(self):
        """Test converting CodeChunk to PointStruct."""
        chunk = create_test_chunk("project#main", "a" * 40)

        point = chunk.to_point()

        assert point.id == chunk.id
        assert point.vector == chunk.embedding
        assert point.payload["git_project_identifier"] == "project#main"
        assert point.payload["git_commit_hash"] == "a" * 40
        assert point.payload["file_path"] == "/repo/test.py"

    def test_multiple_chunks_unique_ids(self):
        """Test that multiple chunks have unique IDs."""
        chunks = [create_test_chunk() for _ in range(10)]

        ids = [chunk.id for chunk in chunks]

        # All IDs should be unique
        assert len(ids) == len(set(ids))

"""Unit tests for DualIndexer (RDR-009)."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call

from arcaneum.schema.document import DualIndexDocument
from arcaneum.indexing.dual_indexer import DualIndexer


class TestDualIndexer:
    """Tests for DualIndexer class."""

    @pytest.fixture
    def mock_qdrant(self):
        """Create mock Qdrant client."""
        client = Mock()
        client.upsert = Mock()
        client.delete = Mock()
        client.get_collection = Mock(return_value=Mock(
            points_count=100,
            indexed_vectors_count=100,
            status="green",
        ))
        return client

    @pytest.fixture
    def mock_meili(self):
        """Create mock MeiliSearch client."""
        client = Mock()
        client.add_documents = Mock()
        client.add_documents_sync = Mock()
        client.get_index_stats = Mock(return_value={"numberOfDocuments": 100})
        return client

    @pytest.fixture
    def dual_indexer(self, mock_qdrant, mock_meili):
        """Create DualIndexer with mocked clients."""
        return DualIndexer(
            qdrant_client=mock_qdrant,
            meili_client=mock_meili,
            collection_name="test-corpus",
            index_name="test-corpus",
            batch_size=50,
        )

    def test_init(self, mock_qdrant, mock_meili):
        """Test DualIndexer initialization."""
        indexer = DualIndexer(
            qdrant_client=mock_qdrant,
            meili_client=mock_meili,
            collection_name="my-corpus",
            index_name="my-corpus",
            batch_size=100,
        )

        assert indexer.collection_name == "my-corpus"
        assert indexer.index_name == "my-corpus"
        assert indexer.batch_size == 100

    def test_index_batch_empty(self, dual_indexer):
        """Test indexing empty batch returns zeros."""
        qdrant_count, meili_count = dual_indexer.index_batch([])

        assert qdrant_count == 0
        assert meili_count == 0
        dual_indexer.qdrant.upsert.assert_not_called()
        dual_indexer.meili.add_documents_sync.assert_not_called()

    def test_index_batch_single_document(self, dual_indexer):
        """Test indexing a single document."""
        doc = DualIndexDocument(
            id="doc-1",
            content="Test content",
            file_path="/path/to/file.py",
            filename="file.py",
            file_extension=".py",
            chunk_index=0,
            chunk_count=1,
            vectors={"stella": [0.1, 0.2, 0.3]},
        )

        qdrant_count, meili_count = dual_indexer.index_batch([doc])

        assert qdrant_count == 1
        assert meili_count == 1
        dual_indexer.qdrant.upsert.assert_called_once()
        dual_indexer.meili.add_documents_sync.assert_called_once()

    def test_index_batch_multiple_documents(self, dual_indexer):
        """Test indexing multiple documents."""
        docs = [
            DualIndexDocument(
                id=f"doc-{i}",
                content=f"Content {i}",
                file_path=f"/path/file{i}.py",
                filename=f"file{i}.py",
                file_extension=".py",
                chunk_index=0,
                chunk_count=1,
                vectors={"stella": [0.1, 0.2, 0.3]},
            )
            for i in range(5)
        ]

        qdrant_count, meili_count = dual_indexer.index_batch(docs)

        assert qdrant_count == 5
        assert meili_count == 5

    def test_index_batch_raises_on_missing_vectors(self, dual_indexer):
        """Test that indexing fails if document has no vectors."""
        doc = DualIndexDocument(
            id="doc-no-vectors",
            content="Test content",
            file_path="/path/to/file.py",
            filename="file.py",
            file_extension=".py",
            chunk_index=0,
            chunk_count=1,
            # No vectors!
        )

        with pytest.raises(ValueError, match="missing vectors"):
            dual_indexer.index_batch([doc])

    def test_index_batch_batches_meili_uploads(self, dual_indexer):
        """Test that MeiliSearch uploads are batched."""
        dual_indexer.batch_size = 2  # Small batch for testing

        docs = [
            DualIndexDocument(
                id=f"doc-{i}",
                content=f"Content {i}",
                file_path=f"/path/file{i}.py",
                filename=f"file{i}.py",
                file_extension=".py",
                chunk_index=0,
                chunk_count=1,
                vectors={"model": [0.1]},
            )
            for i in range(5)
        ]

        qdrant_count, meili_count = dual_indexer.index_batch(docs)

        # Should be 3 batches: 2 + 2 + 1
        assert dual_indexer.meili.add_documents_sync.call_count == 3
        assert qdrant_count == 5
        assert meili_count == 5

    def test_index_batch_generates_ids_if_missing(self, dual_indexer):
        """Test that UUIDs are generated for documents without IDs."""
        doc = DualIndexDocument(
            id="",  # Empty ID
            content="Test content",
            file_path="/path/to/file.py",
            filename="file.py",
            file_extension=".py",
            chunk_index=0,
            chunk_count=1,
            vectors={"model": [0.1]},
        )

        dual_indexer.index_batch([doc])

        # Document should have ID now
        assert doc.id != ""

    def test_index_single(self, dual_indexer):
        """Test index_single convenience method."""
        doc = DualIndexDocument(
            id="single-doc",
            content="Single document",
            file_path="/path/file.py",
            filename="file.py",
            file_extension=".py",
            chunk_index=0,
            chunk_count=1,
            vectors={"model": [0.1, 0.2]},
        )

        qdrant_count, meili_count = dual_indexer.index_single(doc)

        assert qdrant_count == 1
        assert meili_count == 1

    def test_index_batch_async_mode(self, dual_indexer):
        """Test indexing with wait=False uses async API."""
        doc = DualIndexDocument(
            id="async-doc",
            content="Async content",
            file_path="/path/file.py",
            filename="file.py",
            file_extension=".py",
            chunk_index=0,
            chunk_count=1,
            vectors={"model": [0.1]},
        )

        dual_indexer.index_batch([doc], wait=False)

        # Should use add_documents (async) instead of add_documents_sync
        dual_indexer.meili.add_documents.assert_called()
        dual_indexer.meili.add_documents_sync.assert_not_called()

    def test_get_stats(self, dual_indexer):
        """Test getting stats from both systems."""
        stats = dual_indexer.get_stats()

        assert "qdrant" in stats
        assert "meilisearch" in stats
        assert stats["qdrant"]["points_count"] == 100
        assert stats["meilisearch"]["numberOfDocuments"] == 100

    def test_delete_by_file_path(self, dual_indexer):
        """Test deletion by file path."""
        dual_indexer.delete_by_file_path("/path/to/file.py")

        dual_indexer.qdrant.delete.assert_called_once()

    def test_delete_by_project_identifier(self, dual_indexer):
        """Test deletion by project identifier."""
        dual_indexer.delete_by_project_identifier("myproject#main")

        dual_indexer.qdrant.delete.assert_called_once()


class TestDualIndexerWithMultipleModels:
    """Tests for DualIndexer with multiple embedding models."""

    @pytest.fixture
    def mock_clients(self):
        """Create mock clients."""
        qdrant = Mock()
        meili = Mock()
        meili.add_documents_sync = Mock()
        return qdrant, meili

    def test_index_document_with_multiple_vectors(self, mock_clients):
        """Test indexing document with multiple embedding models."""
        qdrant, meili = mock_clients
        indexer = DualIndexer(qdrant, meili, "corpus", "corpus")

        doc = DualIndexDocument(
            id="multi-vec-doc",
            content="Multi-vector content",
            file_path="/path/file.py",
            filename="file.py",
            file_extension=".py",
            chunk_index=0,
            chunk_count=1,
            vectors={
                "stella": [0.1, 0.2, 0.3],
                "jina": [0.4, 0.5, 0.6, 0.7],
                "bge": [0.8, 0.9],
            },
        )

        indexer.index_batch([doc])

        # Verify Qdrant was called with all vectors
        upsert_call = qdrant.upsert.call_args
        points = upsert_call.kwargs.get('points') or upsert_call[1].get('points')
        assert len(points) == 1
        assert "stella" in points[0].vector
        assert "jina" in points[0].vector
        assert "bge" in points[0].vector

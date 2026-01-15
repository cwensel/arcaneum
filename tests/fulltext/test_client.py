"""Integration tests for MeiliSearch client (RDR-008).

These tests require a running MeiliSearch server.
Run with: pytest tests/fulltext/test_client.py -v
"""

import pytest
import os
from arcaneum.fulltext.client import FullTextClient
from arcaneum.fulltext.indexes import SOURCE_CODE_SETTINGS


MEILISEARCH_URL = os.environ.get("MEILISEARCH_URL", "http://localhost:7700")
MEILISEARCH_API_KEY = os.environ.get("MEILISEARCH_API_KEY")


@pytest.fixture
def client():
    """Provide MeiliSearch client connected to test server."""
    client = FullTextClient(url=MEILISEARCH_URL, api_key=MEILISEARCH_API_KEY)

    # Skip tests if server not available
    if not client.health_check():
        pytest.skip("MeiliSearch server not available")

    yield client

    # Cleanup: delete test indexes
    try:
        indexes = client.list_indexes()
        for idx in indexes:
            if idx['uid'].startswith("test_"):
                client.delete_index(idx['uid'])
    except Exception:
        pass


class TestClientConnection:
    """Tests for client connection and health check."""

    def test_health_check(self, client):
        """Test health check returns True for healthy server."""
        assert client.health_check() is True

    def test_get_version(self, client):
        """Test getting server version."""
        version = client.get_version()
        assert "pkgVersion" in version
        assert version["pkgVersion"].startswith("1.")

    def test_get_stats(self, client):
        """Test getting global stats."""
        stats = client.get_stats()
        assert "databaseSize" in stats
        assert "indexes" in stats


class TestIndexOperations:
    """Tests for index creation and management."""

    def test_create_index(self, client):
        """Test creating a basic index."""
        index_name = "test_create_basic"
        index = client.create_index(index_name)

        assert index is not None
        assert client.index_exists(index_name)

    def test_create_index_with_settings(self, client):
        """Test creating an index with settings."""
        index_name = "test_create_with_settings"
        index = client.create_index(
            index_name,
            primary_key="id",
            settings=SOURCE_CODE_SETTINGS
        )

        assert index is not None

        # Verify settings were applied
        settings = client.get_index_settings(index_name)
        assert settings["searchableAttributes"] == SOURCE_CODE_SETTINGS["searchableAttributes"]

    def test_index_exists_true(self, client):
        """Test index_exists returns True for existing index."""
        index_name = "test_exists_true"
        client.create_index(index_name)

        assert client.index_exists(index_name) is True

    def test_index_exists_false(self, client):
        """Test index_exists returns False for non-existing index."""
        assert client.index_exists("nonexistent_index_xyz") is False

    def test_list_indexes(self, client):
        """Test listing indexes."""
        # Create test indexes
        client.create_index("test_list_1")
        client.create_index("test_list_2")

        indexes = client.list_indexes()
        index_names = [idx['uid'] for idx in indexes]

        assert "test_list_1" in index_names
        assert "test_list_2" in index_names

    def test_delete_index(self, client):
        """Test deleting an index."""
        index_name = "test_delete"
        client.create_index(index_name)

        assert client.index_exists(index_name) is True

        client.delete_index(index_name)

        assert client.index_exists(index_name) is False

    def test_update_index_settings(self, client):
        """Test updating index settings."""
        index_name = "test_update_settings"
        client.create_index(index_name)

        # Update settings
        new_settings = {
            "filterableAttributes": ["language", "project"],
            "searchableAttributes": ["content", "title"]
        }
        client.update_index_settings(index_name, new_settings)

        # Verify settings were updated
        settings = client.get_index_settings(index_name)
        assert "language" in settings["filterableAttributes"]
        assert "project" in settings["filterableAttributes"]


class TestDocumentOperations:
    """Tests for document indexing and retrieval."""

    def test_add_documents(self, client):
        """Test adding documents to an index."""
        index_name = "test_add_docs"
        client.create_index(index_name)

        documents = [
            {"id": 1, "content": "Hello world", "language": "python"},
            {"id": 2, "content": "Goodbye world", "language": "javascript"},
        ]

        result = client.add_documents(index_name, documents)
        assert "task_uid" in result

    def test_add_documents_sync(self, client):
        """Test adding documents synchronously."""
        index_name = "test_add_docs_sync"
        client.create_index(index_name, settings={
            "filterableAttributes": ["language"]
        })

        documents = [
            {"id": 1, "content": "Python code example", "language": "python"},
            {"id": 2, "content": "JavaScript code example", "language": "javascript"},
        ]

        result = client.add_documents_sync(index_name, documents)
        assert result["status"] == "succeeded"

        # Verify documents were added
        stats = client.get_index_stats(index_name)
        assert stats["numberOfDocuments"] == 2


class TestSearchOperations:
    """Tests for search functionality."""

    @pytest.fixture
    def indexed_data(self, client):
        """Create index with test data."""
        index_name = "test_search_data"

        # Create index with settings
        client.create_index(index_name, settings={
            "searchableAttributes": ["content", "filename"],
            "filterableAttributes": ["language", "project"]
        })

        # Add test documents
        documents = [
            {"id": 1, "content": "def authenticate(user, password):", "filename": "auth.py", "language": "python", "project": "myapp"},
            {"id": 2, "content": "function authenticate(user, pass) {", "filename": "auth.js", "language": "javascript", "project": "myapp"},
            {"id": 3, "content": "class UserAuthentication:", "filename": "user.py", "language": "python", "project": "myapp"},
            {"id": 4, "content": "def calculate_total(items):", "filename": "utils.py", "language": "python", "project": "utils"},
        ]

        client.add_documents_sync(index_name, documents)

        yield index_name

    def test_basic_search(self, client, indexed_data):
        """Test basic search query."""
        results = client.search(indexed_data, "authenticate")

        assert "hits" in results
        assert len(results["hits"]) > 0
        assert "processingTimeMs" in results

    def test_search_with_filter(self, client, indexed_data):
        """Test search with filter."""
        results = client.search(
            indexed_data,
            "authenticate",
            filter="language = python"
        )

        assert len(results["hits"]) > 0
        # All results should be Python
        for hit in results["hits"]:
            assert hit["language"] == "python"

    def test_search_with_limit(self, client, indexed_data):
        """Test search with limit."""
        results = client.search(indexed_data, "authenticate", limit=1)

        assert len(results["hits"]) <= 1

    def test_search_with_offset(self, client, indexed_data):
        """Test search with pagination offset."""
        # First get all results
        all_results = client.search(indexed_data, "authenticate", limit=10)
        total = len(all_results["hits"])

        if total > 1:
            # Get results with offset
            offset_results = client.search(indexed_data, "authenticate", limit=10, offset=1)
            assert len(offset_results["hits"]) == total - 1

    def test_search_with_highlight(self, client, indexed_data):
        """Test search with highlighting."""
        results = client.search(
            indexed_data,
            "authenticate",
            attributes_to_highlight=["content"]
        )

        assert len(results["hits"]) > 0
        # Check that highlighted content is present
        hit = results["hits"][0]
        assert "_formatted" in hit
        assert "content" in hit["_formatted"]

    def test_exact_phrase_search(self, client, indexed_data):
        """Test exact phrase search with quotes."""
        results = client.search(indexed_data, '"def authenticate"')

        assert len(results["hits"]) > 0
        # Should find the Python function definition
        found_python = any(hit["language"] == "python" for hit in results["hits"])
        assert found_python

    def test_search_no_results(self, client, indexed_data):
        """Test search with no results."""
        results = client.search(indexed_data, "xyznonexistent123")

        assert results["hits"] == []
        assert results["estimatedTotalHits"] == 0

"""Integration tests for source code indexing pipeline."""

import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch

import pytest
import git

from arcaneum.indexing.source_code_pipeline import SourceCodeIndexer
from arcaneum.indexing.qdrant_indexer import QdrantIndexer


@pytest.fixture
def temp_git_repo():
    """Create a temporary git repository with some Python code."""
    temp_dir = tempfile.mkdtemp()

    # Initialize git repo
    repo = git.Repo.init(temp_dir)
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@example.com").release()

    # Add some Python files
    files = {
        "main.py": '''
def hello():
    """Say hello."""
    print("Hello, World!")

def add(a, b):
    """Add two numbers."""
    return a + b
''',
        "utils.py": '''
class Calculator:
    """Simple calculator."""

    def multiply(self, x, y):
        return x * y

    def divide(self, x, y):
        if y == 0:
            raise ValueError("Division by zero")
        return x / y
''',
        "tests.py": '''
def test_add():
    assert add(2, 3) == 5

def test_multiply():
    calc = Calculator()
    assert calc.multiply(4, 5) == 20
'''
    }

    for filename, content in files.items():
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        repo.index.add([filename])

    repo.index.commit("Initial commit")
    repo.create_remote("origin", "https://github.com/user/test-repo.git")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    client = Mock()
    client.scroll.return_value = ([], None)  # Empty collection
    client.upsert.return_value = None
    client.delete.return_value = Mock(operation_id="test-op")
    return client


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    with patch('arcaneum.indexing.source_code_pipeline.TextEmbedding') as mock:
        embedder_instance = Mock()
        # Mock embed method to return fake embeddings
        def mock_embed(texts):
            for _ in texts:
                embedding = Mock()
                embedding.tolist.return_value = [0.1] * 384
                yield embedding
        embedder_instance.embed = mock_embed
        mock.return_value = embedder_instance
        yield mock


class TestSourceCodeIndexer:
    """Integration tests for SourceCodeIndexer."""

    def test_initialization(self, mock_qdrant_client):
        """Test basic initialization."""
        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_model="test-model",
            chunk_size=400
        )

        assert indexer.qdrant_indexer == indexer_obj
        assert indexer.chunker.chunk_size == 400
        assert indexer.stats["projects_discovered"] == 0

    def test_index_directory_empty(self, mock_qdrant_client, mock_embedder):
        """Test indexing empty directory."""
        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_model="test-model"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            stats = indexer.index_directory(
                input_path=temp_dir,
                collection_name="test-collection",
                show_progress=False
            )

            assert stats["projects_discovered"] == 0
            assert stats["projects_indexed"] == 0

    def test_index_directory_single_repo(self, temp_git_repo, mock_qdrant_client, mock_embedder):
        """Test indexing directory with single git repository."""
        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_model="test-model"
        )

        parent_dir = os.path.dirname(temp_git_repo)

        stats = indexer.index_directory(
            input_path=parent_dir,
            collection_name="test-collection",
            show_progress=False
        )

        # Should discover and index the repository
        assert stats["projects_discovered"] == 1
        assert stats["projects_indexed"] == 1
        assert stats["files_processed"] >= 3  # main.py, utils.py, tests.py
        assert stats["chunks_created"] > 0

        # Should have called upsert
        assert mock_qdrant_client.upsert.called

    def test_index_directory_force_mode(self, temp_git_repo, mock_qdrant_client, mock_embedder):
        """Test force mode bypasses incremental sync."""
        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_model="test-model"
        )

        parent_dir = os.path.dirname(temp_git_repo)

        stats = indexer.index_directory(
            input_path=parent_dir,
            collection_name="test-collection",
            force=True,
            show_progress=False
        )

        # Should index even if already indexed
        assert stats["projects_indexed"] == 1

        # Should not have queried Qdrant for indexed projects
        assert not mock_qdrant_client.scroll.called

    def test_index_directory_with_depth(self, mock_qdrant_client, mock_embedder):
        """Test indexing with depth limit."""
        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_model="test-model"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested structure with repos at different depths
            repo1 = os.path.join(temp_dir, "repo1")
            os.makedirs(repo1)
            git_repo1 = git.Repo.init(repo1)
            git_repo1.config_writer().set_value("user", "name", "Test").release()
            git_repo1.config_writer().set_value("user", "email", "test@example.com").release()

            # Add a file and commit
            file1 = os.path.join(repo1, "test.py")
            with open(file1, 'w') as f:
                f.write("print('test')")
            git_repo1.index.add(["test.py"])
            git_repo1.index.commit("Initial")

            # Index with depth=0
            stats = indexer.index_directory(
                input_path=temp_dir,
                collection_name="test-collection",
                depth=0,
                show_progress=False
            )

            # Should find repo at depth 0
            assert stats["projects_discovered"] == 1

    def test_incremental_sync_skips_unchanged(self, temp_git_repo, mock_qdrant_client, mock_embedder):
        """Test that unchanged projects are skipped."""
        # Mock that project is already indexed with same commit
        repo = git.Repo(temp_git_repo)
        current_commit = repo.head.commit.hexsha

        mock_point = Mock()
        mock_point.payload = {
            "git_project_identifier": "test-repo#master",
            "git_commit_hash": current_commit
        }
        mock_qdrant_client.scroll.return_value = ([mock_point], None)

        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_model="test-model"
        )

        parent_dir = os.path.dirname(temp_git_repo)

        stats = indexer.index_directory(
            input_path=parent_dir,
            collection_name="test-collection",
            show_progress=False
        )

        # Should discover but skip (unchanged)
        assert stats["projects_discovered"] == 1
        assert stats["projects_skipped"] == 1
        assert stats["projects_indexed"] == 0

    def test_incremental_sync_reindexes_changed(self, temp_git_repo, mock_qdrant_client, mock_embedder):
        """Test that changed projects are re-indexed."""
        # Mock that project is indexed with different commit
        mock_point = Mock()
        mock_point.payload = {
            "git_project_identifier": "test-repo#master",
            "git_commit_hash": "a" * 40  # Different commit
        }
        mock_qdrant_client.scroll.return_value = ([mock_point], None)

        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_model="test-model"
        )

        parent_dir = os.path.dirname(temp_git_repo)

        stats = indexer.index_directory(
            input_path=parent_dir,
            collection_name="test-collection",
            show_progress=False
        )

        # Should discover and re-index (commit changed)
        assert stats["projects_discovered"] == 1
        assert stats["projects_indexed"] == 1
        assert stats["projects_skipped"] == 0

        # Should have deleted old chunks
        assert mock_qdrant_client.delete.called

    def test_reset_stats(self, mock_qdrant_client):
        """Test resetting statistics."""
        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_model="test-model"
        )

        # Modify stats
        indexer.stats["projects_indexed"] = 5
        indexer.stats["files_processed"] = 10

        # Reset
        indexer.reset_stats()

        # Should be zero
        assert indexer.stats["projects_indexed"] == 0
        assert indexer.stats["files_processed"] == 0

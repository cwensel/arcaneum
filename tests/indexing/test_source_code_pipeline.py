"""Integration tests for source code indexing pipeline."""

import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch

import pytest
import git

from arcaneum.indexing.source_code_pipeline import (
    SourceCodeIndexer,
    _process_file_worker,
)
from arcaneum.indexing.qdrant_indexer import QdrantIndexer
from arcaneum.indexing.types import GitMetadata


@pytest.fixture
def temp_git_repo():
    """Create a temporary git repository with some Python code.

    Yields the repo directory. The repo lives inside a dedicated parent
    directory (parent/test-repo/) so that tests using os.path.dirname()
    get a controlled search root rather than /tmp (which is unreliable
    on Linux CI runners due to find scanning the entire /tmp tree).
    """
    parent_dir = tempfile.mkdtemp()
    temp_dir = os.path.join(parent_dir, "test-repo")
    os.makedirs(temp_dir)

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
        "tests.py": """
def test_add():
    assert add(2, 3) == 5

def test_multiply():
    calc = Calculator()
    assert calc.multiply(4, 5) == 20
""",
    }

    for filename, content in files.items():
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)
        repo.index.add([filename])

    repo.index.commit("Initial commit")
    repo.create_remote("origin", "https://github.com/user/test-repo.git")

    yield temp_dir

    # Cleanup
    shutil.rmtree(parent_dir)


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
    """Create a mock EmbeddingClient.

    Mocks embed_parallel to invoke on_batch_complete with fake embeddings,
    matching the streaming (accumulate=False) pattern used by SourceCodeIndexer.
    """
    client = Mock()
    client.use_gpu = False
    client.get_device_info = Mock(
        return_value={
            "gpu_enabled": False,
            "gpu_available": False,
            "device": "cpu",
        }
    )

    def fake_embed_parallel(texts, model_name, on_batch_complete=None, batch_size=None, **kwargs):
        embeddings = [[0.1] * 384] * len(texts)
        if on_batch_complete is not None:
            # on_batch_complete(batch_idx, start_idx, batch_embeddings)
            on_batch_complete(0, 0, embeddings)
        return embeddings

    client.embed_parallel = Mock(side_effect=fake_embed_parallel)
    return client


class TestSourceCodeIndexer:
    """Integration tests for SourceCodeIndexer."""

    def test_initialization(self, mock_qdrant_client, mock_embedder):
        """Test basic initialization."""
        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_client=mock_embedder,
            embedding_model_id="test-model",
            chunk_size=400,
        )

        assert indexer.qdrant_indexer == indexer_obj
        assert indexer.chunker.chunk_size == 400
        assert indexer.stats["projects_discovered"] == 0

    def test_index_directory_empty(self, mock_qdrant_client, mock_embedder):
        """Test indexing empty directory."""
        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_client=mock_embedder,
            embedding_model_id="test-model",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            stats = indexer.index_directory(
                input_path=temp_dir, collection_name="test-collection", show_progress=False
            )

            assert stats["projects_discovered"] == 0
            assert stats["projects_indexed"] == 0

    def test_index_directory_single_repo(self, temp_git_repo, mock_qdrant_client, mock_embedder):
        """Test indexing directory with single git repository."""
        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_client=mock_embedder,
            embedding_model_id="test-model",
        )

        parent_dir = os.path.dirname(temp_git_repo)

        stats = indexer.index_directory(
            input_path=parent_dir, collection_name="test-collection", show_progress=False
        )

        # Should discover and index the repository
        assert stats["projects_discovered"] == 1
        assert stats["projects_indexed"] == 1
        assert stats["files_processed"] >= 3  # main.py, utils.py, tests.py
        assert stats["chunks_created"] > 0
        assert set(stats["covered_paths"]) >= {
            os.path.join(temp_git_repo, "main.py"),
            os.path.join(temp_git_repo, "utils.py"),
            os.path.join(temp_git_repo, "tests.py"),
        }

        # Should have called upsert
        assert mock_qdrant_client.upsert.called

    def test_index_directory_force_mode(self, temp_git_repo, mock_qdrant_client, mock_embedder):
        """Test force mode bypasses incremental sync."""
        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_client=mock_embedder,
            embedding_model_id="test-model",
        )

        parent_dir = os.path.dirname(temp_git_repo)

        stats = indexer.index_directory(
            input_path=parent_dir,
            collection_name="test-collection",
            force=True,
            show_progress=False,
        )

        # Should index even if already indexed (force bypasses skip logic)
        assert stats["projects_indexed"] == 1
        assert stats["projects_skipped"] == 0

    def test_index_directory_force_deletes_branch_chunks_before_upload(
        self, temp_git_repo, mock_qdrant_client, mock_embedder
    ):
        """Force reindex must delete the project's existing branch chunks
        before uploading new ones, so no stale vectors survive.

        Regression for job-1921: on plain force (no repair_targets) the
        delete_branch_chunks call was gated on `identifier in indexed_projects`,
        but force sets indexed_projects = {} so the delete never fired. Source
        chunks use random uuid4 point IDs, so re-upload appends new chunks
        beside stale ones rather than replacing them.
        """
        repo = git.Repo(temp_git_repo)
        current_branch = repo.active_branch.name
        identifier = f"test-repo#{current_branch}"

        # Record call order so we can assert delete precedes upload.
        call_order = []
        mock_qdrant_client.delete.side_effect = lambda *a, **k: (
            call_order.append("delete") or Mock(operation_id="op")
        )
        mock_qdrant_client.upsert.side_effect = lambda *a, **k: call_order.append("upsert")

        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_client=mock_embedder,
            embedding_model_id="test-model",
        )

        parent_dir = os.path.dirname(temp_git_repo)

        stats = indexer.index_directory(
            input_path=parent_dir,
            collection_name="test-collection",
            force=True,
            show_progress=False,
        )

        assert stats["projects_indexed"] == 1

        # Old branch chunks must have been deleted on force...
        assert mock_qdrant_client.delete.called, (
            "force reindex did not delete existing branch chunks"
        )
        # ...by the project's composite identifier...
        delete_filter = mock_qdrant_client.delete.call_args.kwargs["points_selector"]
        assert delete_filter.must[0].match.value == identifier

        # ...and before any new chunks were uploaded (no stale vectors survive).
        assert "delete" in call_order and "upsert" in call_order
        assert call_order.index("delete") < call_order.index("upsert"), (
            f"delete must precede upload, got order: {call_order}"
        )

    def test_per_file_failure_increments_error_count(
        self, temp_git_repo, mock_qdrant_client, mock_embedder
    ):
        """A file whose worker raises must be counted in stats['errors'].

        Regression for job-1921 Fix C: the source stamp gate previously
        hardcoded errors=0, so a reindex with per-file failures could still
        stamp. The pipeline must track real failures so the gate can withhold
        the stamp. We force every submitted future to raise on .result() and
        assert stats['errors'] reflects the failed files (and nothing was
        indexed/uploaded).
        """
        from concurrent.futures import Future

        class _FakeExecutor:
            def submit(self, fn, *args, **kwargs):
                fut = Future()
                fut.set_exception(RuntimeError("worker boom"))
                return fut

            def shutdown(self, *a, **k):
                pass

        indexer_obj = QdrantIndexer(mock_qdrant_client)
        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_client=mock_embedder,
            embedding_model_id="test-model",
        )

        parent_dir = os.path.dirname(temp_git_repo)

        with patch(
            "arcaneum.indexing.source_code_pipeline.create_process_pool",
            return_value=_FakeExecutor(),
        ):
            stats = indexer.index_directory(
                input_path=parent_dir,
                collection_name="test-collection",
                force=True,
                show_progress=False,
            )

        # Every tracked file failed -> errors counted, nothing uploaded.
        assert stats["errors"] >= 1
        assert stats["files_processed"] == 0
        assert stats["covered_paths"] == []
        assert stats["chunks_uploaded"] == 0

    def test_process_file_worker_propagates_read_failures(self, tmp_path):
        """Worker read/chunk failures must count as indexing errors upstream."""
        missing_file = tmp_path / "missing.py"
        metadata = GitMetadata(
            project_root=str(tmp_path),
            commit_hash="a" * 40,
            branch="main",
            project_name="repo",
            remote_url=None,
        )

        with pytest.raises(FileNotFoundError):
            _process_file_worker(
                str(missing_file),
                "repo#main",
                metadata,
                "test-model",
                400,
                20,
            )

    def test_index_directory_with_depth(self, mock_qdrant_client, mock_embedder):
        """Test indexing with depth limit."""
        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_client=mock_embedder,
            embedding_model_id="test-model",
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
            with open(file1, "w") as f:
                f.write("print('test')")
            git_repo1.index.add(["test.py"])
            git_repo1.index.commit("Initial")

            # Index with depth=0
            stats = indexer.index_directory(
                input_path=temp_dir, collection_name="test-collection", depth=0, show_progress=False
            )

            # Should find repo at depth 0
            assert stats["projects_discovered"] == 1

    def test_incremental_sync_skips_unchanged(
        self, temp_git_repo, mock_qdrant_client, mock_embedder
    ):
        """Test that unchanged projects are skipped."""
        # Mock that project is already indexed with same commit
        repo = git.Repo(temp_git_repo)
        current_commit = repo.head.commit.hexsha
        current_branch = repo.active_branch.name

        mock_point = Mock()
        mock_point.payload = {
            "git_project_identifier": f"test-repo#{current_branch}",
            "git_commit_hash": current_commit,
        }
        mock_qdrant_client.scroll.return_value = ([mock_point], None)

        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_client=mock_embedder,
            embedding_model_id="test-model",
        )

        parent_dir = os.path.dirname(temp_git_repo)

        stats = indexer.index_directory(
            input_path=parent_dir, collection_name="test-collection", show_progress=False
        )

        # Should discover but skip (unchanged)
        assert stats["projects_discovered"] == 1
        assert stats["projects_skipped"] == 1
        assert stats["projects_indexed"] == 0

    def test_incremental_sync_reindexes_changed(
        self, temp_git_repo, mock_qdrant_client, mock_embedder
    ):
        """Test that changed projects are re-indexed."""
        # Mock that project is indexed with different commit
        repo = git.Repo(temp_git_repo)
        current_branch = repo.active_branch.name

        mock_point = Mock()
        mock_point.payload = {
            "git_project_identifier": f"test-repo#{current_branch}",
            "git_commit_hash": "a" * 40,  # Different commit
        }
        mock_qdrant_client.scroll.return_value = ([mock_point], None)

        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_client=mock_embedder,
            embedding_model_id="test-model",
        )

        parent_dir = os.path.dirname(temp_git_repo)

        stats = indexer.index_directory(
            input_path=parent_dir, collection_name="test-collection", show_progress=False
        )

        # Should discover and re-index (commit changed)
        assert stats["projects_discovered"] == 1
        assert stats["projects_indexed"] == 1
        assert stats["projects_skipped"] == 0

        # Should have deleted old chunks
        assert mock_qdrant_client.delete.called

    def test_reset_stats(self, mock_qdrant_client, mock_embedder):
        """Test resetting statistics."""
        indexer_obj = QdrantIndexer(mock_qdrant_client)

        indexer = SourceCodeIndexer(
            qdrant_indexer=indexer_obj,
            embedding_client=mock_embedder,
            embedding_model_id="test-model",
        )

        # Modify stats
        indexer.stats["projects_indexed"] = 5
        indexer.stats["files_processed"] = 10
        indexer.stats["covered_paths"] = ["/repo/main.py"]

        # Reset
        indexer.reset_stats()

        # Should be zero
        assert indexer.stats["projects_indexed"] == 0
        assert indexer.stats["files_processed"] == 0
        assert indexer.stats["covered_paths"] == []

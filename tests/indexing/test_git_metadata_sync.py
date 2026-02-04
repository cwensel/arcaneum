"""Tests for git metadata sync module."""

from unittest.mock import Mock, MagicMock
import pytest

from qdrant_client import QdrantClient
from qdrant_client.models import Record, ScoredPoint

from arcaneum.indexing.git_metadata_sync import GitMetadataSync, IndexedProject


@pytest.fixture
def mock_qdrant():
    """Create a mock Qdrant client."""
    return Mock(spec=QdrantClient)


@pytest.fixture
def metadata_sync(mock_qdrant):
    """Create a GitMetadataSync instance with mocked client."""
    return GitMetadataSync(mock_qdrant)


def create_mock_point(point_id, identifier, commit_hash):
    """Helper to create a mock Qdrant point."""
    point = Mock(spec=Record)
    point.id = point_id
    point.payload = {
        "git_project_identifier": identifier,
        "git_commit_hash": commit_hash
    }
    return point


class TestGitMetadataSync:
    """Tests for GitMetadataSync class."""

    def test_initialization(self, mock_qdrant):
        """Test basic initialization."""
        sync = GitMetadataSync(mock_qdrant)

        assert sync.qdrant == mock_qdrant
        assert sync._cache == {}

    def test_get_indexed_projects_empty(self, metadata_sync, mock_qdrant):
        """Test getting indexed projects from empty collection."""
        # Mock scroll returning empty results
        mock_qdrant.scroll.return_value = ([], None)

        indexed = metadata_sync.get_indexed_projects("test-collection")

        assert indexed == {}
        mock_qdrant.scroll.assert_called_once()

    def test_get_indexed_projects_single_project(self, metadata_sync, mock_qdrant):
        """Test getting indexed projects with single project."""
        # Mock scroll returning one project with multiple chunks
        points = [
            create_mock_point(1, "project#main", "a" * 40),
            create_mock_point(2, "project#main", "a" * 40),
            create_mock_point(3, "project#main", "a" * 40),
        ]
        mock_qdrant.scroll.return_value = (points, None)

        indexed = metadata_sync.get_indexed_projects("test-collection")

        assert len(indexed) == 1
        assert "project#main" in indexed

        project = indexed["project#main"]
        assert project.identifier == "project#main"
        assert project.commit_hash == "a" * 40
        assert project.point_count == 3

    def test_get_indexed_projects_multiple_projects(self, metadata_sync, mock_qdrant):
        """Test getting indexed projects with multiple projects."""
        # Mock scroll returning multiple projects
        points = [
            create_mock_point(1, "project-a#main", "a" * 40),
            create_mock_point(2, "project-a#main", "a" * 40),
            create_mock_point(3, "project-b#feature", "b" * 40),
            create_mock_point(4, "project-c#develop", "c" * 40),
        ]
        mock_qdrant.scroll.return_value = (points, None)

        indexed = metadata_sync.get_indexed_projects("test-collection")

        assert len(indexed) == 3
        assert "project-a#main" in indexed
        assert "project-b#feature" in indexed
        assert "project-c#develop" in indexed

        assert indexed["project-a#main"].point_count == 2
        assert indexed["project-b#feature"].point_count == 1
        assert indexed["project-c#develop"].point_count == 1

    def test_get_indexed_projects_multiple_batches(self, metadata_sync, mock_qdrant):
        """Test pagination through multiple batches."""
        # Mock scroll returning multiple batches
        batch1 = [create_mock_point(i, f"proj{i}#main", "a" * 40) for i in range(100)]
        batch2 = [create_mock_point(i + 100, f"proj{i+100}#main", "b" * 40) for i in range(50)]

        mock_qdrant.scroll.side_effect = [
            (batch1, "offset1"),  # First batch with offset
            (batch2, None)        # Second batch, no more
        ]

        indexed = metadata_sync.get_indexed_projects("test-collection")

        assert len(indexed) == 150
        assert mock_qdrant.scroll.call_count == 2

    def test_get_indexed_projects_caching(self, metadata_sync, mock_qdrant):
        """Test that results are cached."""
        points = [create_mock_point(1, "project#main", "a" * 40)]
        mock_qdrant.scroll.return_value = (points, None)

        # First call
        indexed1 = metadata_sync.get_indexed_projects("test-collection")

        # Second call should use cache
        indexed2 = metadata_sync.get_indexed_projects("test-collection")

        assert indexed1 == indexed2
        # Should only call scroll once (cached second time)
        mock_qdrant.scroll.assert_called_once()

    def test_get_indexed_projects_force_refresh(self, metadata_sync, mock_qdrant):
        """Test forcing cache refresh."""
        points = [create_mock_point(1, "project#main", "a" * 40)]
        mock_qdrant.scroll.return_value = (points, None)

        # First call
        metadata_sync.get_indexed_projects("test-collection")

        # Force refresh should query again
        metadata_sync.get_indexed_projects("test-collection", force_refresh=True)

        # Should call scroll twice
        assert mock_qdrant.scroll.call_count == 2

    def test_should_reindex_project_new_project(self, metadata_sync, mock_qdrant):
        """Test that new projects should be indexed."""
        # Empty collection
        mock_qdrant.scroll.return_value = ([], None)

        should_index = metadata_sync.should_reindex_project(
            "test-collection",
            "new-project#main",
            "a" * 40
        )

        assert should_index is True

    def test_should_reindex_project_commit_changed(self, metadata_sync, mock_qdrant):
        """Test that projects with changed commits should be re-indexed."""
        # Project indexed with old commit
        points = [create_mock_point(1, "project#main", "a" * 40)]
        mock_qdrant.scroll.return_value = (points, None)

        # Check with new commit
        should_index = metadata_sync.should_reindex_project(
            "test-collection",
            "project#main",
            "b" * 40  # Different commit
        )

        assert should_index is True

    def test_should_reindex_project_unchanged(self, metadata_sync, mock_qdrant):
        """Test that unchanged projects should not be re-indexed."""
        # Project indexed with current commit
        commit = "a" * 40
        points = [create_mock_point(1, "project#main", commit)]
        mock_qdrant.scroll.return_value = (points, None)

        # Check with same commit
        should_index = metadata_sync.should_reindex_project(
            "test-collection",
            "project#main",
            commit  # Same commit
        )

        assert should_index is False

    def test_get_project_stats(self, metadata_sync, mock_qdrant):
        """Test getting statistics for a specific project."""
        points = [
            create_mock_point(1, "project#main", "a" * 40),
            create_mock_point(2, "project#main", "a" * 40),
            create_mock_point(3, "project#main", "a" * 40),
        ]
        mock_qdrant.scroll.return_value = (points, None)

        stats = metadata_sync.get_project_stats("test-collection", "project#main")

        assert stats is not None
        assert stats.identifier == "project#main"
        assert stats.point_count == 3
        assert stats.commit_hash == "a" * 40

    def test_get_project_stats_not_found(self, metadata_sync, mock_qdrant):
        """Test getting stats for non-existent project."""
        mock_qdrant.scroll.return_value = ([], None)

        stats = metadata_sync.get_project_stats("test-collection", "nonexistent#main")

        assert stats is None

    def test_get_all_branches(self, metadata_sync, mock_qdrant):
        """Test getting all branches for a project."""
        points = [
            create_mock_point(1, "myproject#main", "a" * 40),
            create_mock_point(2, "myproject#develop", "b" * 40),
            create_mock_point(3, "myproject#feature-x", "c" * 40),
            create_mock_point(4, "otherproject#main", "d" * 40),
        ]
        mock_qdrant.scroll.return_value = (points, None)

        branches = metadata_sync.get_all_branches("test-collection", "myproject")

        assert branches == {"main", "develop", "feature-x"}

    def test_get_all_branches_no_matches(self, metadata_sync, mock_qdrant):
        """Test getting branches when project has none."""
        points = [create_mock_point(1, "otherproject#main", "a" * 40)]
        mock_qdrant.scroll.return_value = (points, None)

        branches = metadata_sync.get_all_branches("test-collection", "myproject")

        assert branches == set()

    def test_count_total_chunks(self, metadata_sync, mock_qdrant):
        """Test counting total chunks across all projects."""
        points = [
            create_mock_point(1, "project-a#main", "a" * 40),
            create_mock_point(2, "project-a#main", "a" * 40),
            create_mock_point(3, "project-a#main", "a" * 40),
            create_mock_point(4, "project-b#feature", "b" * 40),
            create_mock_point(5, "project-b#feature", "b" * 40),
        ]
        mock_qdrant.scroll.return_value = (points, None)

        total = metadata_sync.count_total_chunks("test-collection")

        assert total == 5

    def test_clear_cache_specific_collection(self, metadata_sync, mock_qdrant):
        """Test clearing cache for specific collection."""
        # Populate cache
        mock_qdrant.scroll.return_value = ([], None)
        metadata_sync.get_indexed_projects("collection-1")
        metadata_sync.get_indexed_projects("collection-2")

        assert "collection-1" in metadata_sync._cache
        assert "collection-2" in metadata_sync._cache

        # Clear one collection
        metadata_sync.clear_cache("collection-1")

        assert "collection-1" not in metadata_sync._cache
        assert "collection-2" in metadata_sync._cache

    def test_clear_cache_all(self, metadata_sync, mock_qdrant):
        """Test clearing all cache."""
        # Populate cache
        mock_qdrant.scroll.return_value = ([], None)
        metadata_sync.get_indexed_projects("collection-1")
        metadata_sync.get_indexed_projects("collection-2")

        assert len(metadata_sync._cache) == 2

        # Clear all
        metadata_sync.clear_cache()

        assert len(metadata_sync._cache) == 0

    def test_get_stale_identifiers(self, metadata_sync, mock_qdrant):
        """Test finding stale identifiers."""
        # Mock indexed projects
        points = [
            create_mock_point(1, "project#main", "a" * 40),
            create_mock_point(2, "project#old-branch", "b" * 40),
            create_mock_point(3, "project#feature", "c" * 40),
        ]
        mock_qdrant.scroll.return_value = (points, None)

        # Current scan only finds main and feature
        current_identifiers = {"project#main", "project#feature"}

        stale = metadata_sync.get_stale_identifiers("test-collection", current_identifiers)

        assert stale == {"project#old-branch"}

    def test_get_stale_identifiers_none(self, metadata_sync, mock_qdrant):
        """Test when no stale identifiers exist."""
        points = [
            create_mock_point(1, "project#main", "a" * 40),
            create_mock_point(2, "project#feature", "b" * 40),
        ]
        mock_qdrant.scroll.return_value = (points, None)

        current_identifiers = {"project#main", "project#feature"}

        stale = metadata_sync.get_stale_identifiers("test-collection", current_identifiers)

        assert stale == set()

    def test_verify_consistency_empty(self, metadata_sync, mock_qdrant):
        """Test consistency verification on empty collection."""
        mock_qdrant.scroll.return_value = ([], None)

        is_consistent, message = metadata_sync.verify_consistency("test-collection")

        assert is_consistent is True
        assert "empty" in message.lower()

    def test_verify_consistency_low_count(self, metadata_sync, mock_qdrant):
        """Test consistency verification detects low chunk counts."""
        # Project with only 2 chunks (suspicious)
        points = [
            create_mock_point(1, "project#main", "a" * 40),
            create_mock_point(2, "project#main", "a" * 40),
        ]
        mock_qdrant.scroll.return_value = (points, None)

        is_consistent, message = metadata_sync.verify_consistency("test-collection")

        assert is_consistent is False
        assert "<5 chunks" in message

    def test_verify_consistency_normal(self, metadata_sync, mock_qdrant):
        """Test consistency verification passes for normal projects."""
        # Project with sufficient chunks
        points = [
            create_mock_point(i, "project#main", "a" * 40)
            for i in range(10)
        ]
        mock_qdrant.scroll.return_value = (points, None)

        is_consistent, message = metadata_sync.verify_consistency("test-collection")

        assert is_consistent is True
        assert "consistent" in message.lower()

    def test_inconsistent_commits_warning(self, metadata_sync, mock_qdrant, caplog):
        """Test that inconsistent commits are warned about."""
        import logging
        caplog.set_level(logging.WARNING)

        # Same identifier with different commits (shouldn't happen normally)
        points = [
            create_mock_point(1, "project#main", "a" * 40),
            create_mock_point(2, "project#main", "b" * 40),  # Different commit!
        ]
        mock_qdrant.scroll.return_value = (points, None)

        metadata_sync.get_indexed_projects("test-collection")

        # Should log warning
        assert "Inconsistent commits" in caplog.text

    def test_error_handling_scroll_failure(self, metadata_sync, mock_qdrant):
        """Test error handling when scroll fails."""
        # Mock scroll raising exception
        mock_qdrant.scroll.side_effect = Exception("Connection error")

        indexed = metadata_sync.get_indexed_projects("test-collection")

        # Should return empty dict on error (fail-safe)
        assert indexed == {}

    def test_is_version_indexed_found(self, metadata_sync, mock_qdrant):
        """Test is_version_indexed returns True when version exists."""
        mock_qdrant.scroll.return_value = ([create_mock_point(1, "project#main", "a" * 40)], None)

        result = metadata_sync.is_version_indexed("test-collection", "project#main@abc1234")

        assert result is True
        # Verify the scroll was called with the correct filter
        mock_qdrant.scroll.assert_called()
        call_args = mock_qdrant.scroll.call_args
        assert call_args.kwargs['collection_name'] == "test-collection"
        assert call_args.kwargs['limit'] == 1

    def test_is_version_indexed_not_found(self, metadata_sync, mock_qdrant):
        """Test is_version_indexed returns False when version doesn't exist."""
        mock_qdrant.scroll.return_value = ([], None)

        result = metadata_sync.is_version_indexed("test-collection", "project#main@xyz9999")

        assert result is False

    def test_is_version_indexed_error_handling(self, metadata_sync, mock_qdrant):
        """Test is_version_indexed returns False on error (fail-safe)."""
        mock_qdrant.scroll.side_effect = Exception("Connection error")

        result = metadata_sync.is_version_indexed("test-collection", "project#main@abc1234")

        # Should return False on error (fail-safe: will allow indexing)
        assert result is False


class TestIndexedProject:
    """Tests for IndexedProject dataclass."""

    def test_creation(self):
        """Test creating an IndexedProject."""
        project = IndexedProject(
            identifier="myproject#main",
            commit_hash="a" * 40,
            point_count=100
        )

        assert project.identifier == "myproject#main"
        assert project.commit_hash == "a" * 40
        assert project.point_count == 100

    def test_default_point_count(self):
        """Test default point count is 0."""
        project = IndexedProject(
            identifier="project#branch",
            commit_hash="a" * 40
        )

        assert project.point_count == 0

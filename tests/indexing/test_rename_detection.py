"""Unit tests for rename/move detection in dual-index corpus sync."""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

from arcaneum.cli.sync import (
    _detect_renames,
    _handle_renames_meili,
    _detect_stale_paths,
)


class TestDetectRenames:
    """Tests for _detect_renames function."""

    @pytest.fixture
    def mock_sync_manager(self):
        """Create mock MetadataBasedSync instance."""
        manager = Mock()
        manager.find_file_by_content_hash = Mock(return_value=[])
        return manager

    def test_detect_renames_finds_moved_files(self, mock_sync_manager, tmp_path):
        """Hash match + old path gone on disk = rename detected."""
        # Create a file at the new path
        new_file = tmp_path / "new_dir" / "doc.pdf"
        new_file.parent.mkdir()
        new_file.write_text("content")

        new_path = str(new_file.absolute())
        old_path = str(tmp_path / "old_dir" / "doc.pdf")  # Does not exist on disk

        # Mock: content hash matches an old path that no longer exists
        mock_sync_manager.find_file_by_content_hash.return_value = [old_path]

        with patch("arcaneum.cli.sync.compute_file_hash", return_value="abc123"):
            renames = _detect_renames(
                {new_path}, mock_sync_manager, "test-corpus",
            )

        assert len(renames) == 1
        assert renames[0] == (old_path, new_path)

    def test_detect_renames_ignores_duplicates(self, mock_sync_manager, tmp_path):
        """Hash match + old path still exists = not a rename (duplicate)."""
        # Create both files
        old_file = tmp_path / "old_dir" / "doc.pdf"
        new_file = tmp_path / "new_dir" / "doc.pdf"
        old_file.parent.mkdir()
        new_file.parent.mkdir()
        old_file.write_text("content")
        new_file.write_text("content")

        new_path = str(new_file.absolute())
        old_path = str(old_file.absolute())

        mock_sync_manager.find_file_by_content_hash.return_value = [old_path]

        with patch("arcaneum.cli.sync.compute_file_hash", return_value="abc123"):
            renames = _detect_renames(
                {new_path}, mock_sync_manager, "test-corpus",
            )

        assert len(renames) == 0

    def test_detect_renames_no_match(self, mock_sync_manager, tmp_path):
        """No hash match = truly new file, not a rename."""
        new_file = tmp_path / "brand_new.pdf"
        new_file.write_text("new content")

        new_path = str(new_file.absolute())

        # No matching content hash in the index
        mock_sync_manager.find_file_by_content_hash.return_value = []

        with patch("arcaneum.cli.sync.compute_file_hash", return_value="xyz789"):
            renames = _detect_renames(
                {new_path}, mock_sync_manager, "test-corpus",
            )

        assert len(renames) == 0

    def test_detect_renames_modified_and_moved(self, mock_sync_manager, tmp_path):
        """Different hash = content changed, should re-index not rename."""
        new_file = tmp_path / "moved_and_edited.pdf"
        new_file.write_text("modified content")

        new_path = str(new_file.absolute())

        # No match because the hash is different from anything in the index
        mock_sync_manager.find_file_by_content_hash.return_value = []

        with patch("arcaneum.cli.sync.compute_file_hash", return_value="newhash"):
            renames = _detect_renames(
                {new_path}, mock_sync_manager, "test-corpus",
            )

        assert len(renames) == 0


class TestHandleRenamesMeili:
    """Tests for _handle_renames_meili function."""

    def test_handle_renames_meili_updates_docs(self):
        """Verifies MeiliSearch update_documents call with correct data."""
        mock_qdrant = Mock()
        mock_meili = Mock()
        mock_index = Mock()
        mock_meili.get_index.return_value = mock_index

        mock_task = Mock()
        mock_task.task_uid = 42
        mock_index.update_documents.return_value = mock_task

        # Simulate two Qdrant points with the old path
        mock_point_1 = Mock()
        mock_point_1.id = "point-1"
        mock_point_2 = Mock()
        mock_point_2.id = "point-2"

        # First scroll returns points, second returns empty
        mock_qdrant.scroll.side_effect = [
            ([mock_point_1, mock_point_2], None),
        ]

        renames = [("/old/dir/doc.pdf", "/new/dir/doc.pdf")]

        updated = _handle_renames_meili(
            renames, mock_qdrant, mock_meili, "test-corpus",
        )

        assert updated == 2
        mock_index.update_documents.assert_called_once()
        update_call_docs = mock_index.update_documents.call_args[0][0]
        assert len(update_call_docs) == 2
        assert update_call_docs[0]["file_path"] == "/new/dir/doc.pdf"
        assert update_call_docs[0]["filename"] == "doc.pdf"
        assert update_call_docs[1]["id"] == "point-2"

    def test_handle_renames_meili_empty_renames(self):
        """No renames = no work done."""
        mock_qdrant = Mock()
        mock_meili = Mock()

        updated = _handle_renames_meili([], mock_qdrant, mock_meili, "test-corpus")

        assert updated == 0
        mock_meili.get_index.assert_not_called()


class TestDetectStalePaths:
    """Tests for _detect_stale_paths function."""

    def test_detect_stale_paths_scoped_to_synced_dirs(self, tmp_path):
        """Only cleans indexed paths within the synced directories."""
        synced_dir = tmp_path / "synced"
        synced_dir.mkdir()

        # This path is in scope but doesn't exist on disk
        stale_in_scope = str(synced_dir / "deleted.pdf")

        # This path is out of scope (different directory)
        stale_out_of_scope = "/other/dir/deleted.pdf"

        # This path exists on disk — not stale
        existing_file = synced_dir / "exists.pdf"
        existing_file.write_text("content")
        existing_path = str(existing_file.absolute())

        all_indexed = {stale_in_scope, stale_out_of_scope, existing_path}

        stale = _detect_stale_paths(all_indexed, [synced_dir], set())

        assert stale_in_scope in stale
        assert stale_out_of_scope not in stale
        assert existing_path not in stale

    def test_detect_stale_paths_excludes_renamed(self, tmp_path):
        """Doesn't clean paths already handled by rename detection."""
        synced_dir = tmp_path / "synced"
        synced_dir.mkdir()

        old_path = str(synced_dir / "old_name.pdf")  # Doesn't exist on disk
        renamed_old_paths = {old_path}

        all_indexed = {old_path}

        stale = _detect_stale_paths(all_indexed, [synced_dir], renamed_old_paths)

        assert len(stale) == 0

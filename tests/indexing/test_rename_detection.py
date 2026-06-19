"""Unit tests for rename/move detection in dual-index corpus sync."""

from unittest.mock import Mock, patch

import pytest

from arcaneum.cli.sync import (
    _detect_renames,
    _detect_stale_paths,
    _filter_rename_candidate_paths,
    _handle_renames_meili,
    _rename_tuples_with_metadata,
)
from arcaneum.indexing.common.sync import MetadataBasedSync


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
                {new_path},
                mock_sync_manager,
                "test-corpus",
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
                {new_path},
                mock_sync_manager,
                "test-corpus",
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
                {new_path},
                mock_sync_manager,
                "test-corpus",
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
                {new_path},
                mock_sync_manager,
                "test-corpus",
            )

        assert len(renames) == 0

    def test_detect_renames_aborts_on_first_content_hash_backend_failure(
        self, mock_sync_manager, tmp_path, caplog
    ):
        """A Qdrant disconnect should not emit one warning per new file."""
        files = []
        for name in ("a.pdf", "b.pdf", "c.pdf"):
            file_path = tmp_path / name
            file_path.write_text(name)
            files.append(str(file_path.absolute()))

        mock_sync_manager.find_file_by_content_hash.side_effect = RuntimeError(
            "Server disconnected without sending a response."
        )

        with patch("arcaneum.cli.sync.compute_file_hash", return_value="abc123"):
            renames = _detect_renames(set(files), mock_sync_manager, "test-corpus")

        assert renames == []
        assert mock_sync_manager.find_file_by_content_hash.call_count == 1
        assert "Skipping rename detection after content-hash lookup failed" in caplog.text

    def test_filter_rename_candidates_excludes_existing_paths(self, mock_sync_manager):
        """Same-path quick-hash misses are modified files, not rename candidates."""
        mock_sync_manager._get_indexed_file_paths_set.return_value = {
            "/repo/edited.md",
            "/repo/unchanged.md",
        }

        candidates = _filter_rename_candidate_paths(
            {"/repo/edited.md", "/repo/new-location.md"},
            mock_sync_manager,
            "test-corpus",
        )

        assert candidates == {"/repo/new-location.md"}

    def test_find_file_by_content_hash_can_raise_backend_errors(self):
        """Strict callers can distinguish lookup failure from no hash match."""
        qdrant = Mock()
        qdrant.scroll.side_effect = RuntimeError("disconnected")
        sync = MetadataBasedSync(qdrant)

        with pytest.raises(RuntimeError, match="disconnected"):
            sync.find_file_by_content_hash("test-corpus", "abc123", raise_on_error=True)

        assert sync.find_file_by_content_hash("test-corpus", "abc123") == []


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

        updated, confirmed_renames = _handle_renames_meili(
            renames,
            mock_qdrant,
            mock_meili,
            "test-corpus",
        )

        assert updated == 2
        assert confirmed_renames == renames
        mock_index.update_documents.assert_called_once()
        update_call_docs = mock_index.update_documents.call_args[0][0]
        assert len(update_call_docs) == 2
        assert update_call_docs[0]["file_path"] == "/new/dir/doc.pdf"
        assert update_call_docs[0]["filename"] == "doc.pdf"
        assert update_call_docs[1]["id"] == "point-2"
        mock_meili.client.wait_for_task.assert_called_once_with(42, timeout_in_ms=300000)

    def test_handle_renames_meili_returns_no_confirmed_renames_on_timeout(self):
        """Timed-out MeiliSearch tasks should not advance Qdrant rename metadata."""
        mock_qdrant = Mock()
        mock_meili = Mock()
        mock_index = Mock()
        mock_meili.get_index.return_value = mock_index

        mock_task = Mock()
        mock_task.task_uid = 42
        mock_index.update_documents.return_value = mock_task
        mock_meili.client.wait_for_task.side_effect = TimeoutError("timed out")

        mock_point = Mock()
        mock_point.id = "point-1"
        mock_qdrant.scroll.return_value = ([mock_point], None)

        updated, confirmed_renames = _handle_renames_meili(
            [("/old/dir/doc.pdf", "/new/dir/doc.pdf")],
            mock_qdrant,
            mock_meili,
            "test-corpus",
        )

        assert updated == 0
        assert confirmed_renames == []

    def test_handle_renames_meili_empty_renames(self):
        """No renames = no work done."""
        mock_qdrant = Mock()
        mock_meili = Mock()

        updated, confirmed_renames = _handle_renames_meili(
            [], mock_qdrant, mock_meili, "test-corpus"
        )

        assert updated == 0
        assert confirmed_renames == []
        mock_meili.get_index.assert_not_called()


class TestHandleRenamesQdrant:
    """Tests for MetadataBasedSync.handle_renames metadata updates."""

    def test_handle_renames_updates_fast_path_metadata(self):
        """Renames update file_paths and file_quick_hashes so fast sync can skip later."""
        qdrant = Mock()
        point = Mock()
        point.payload = {
            "file_paths": ["/old/doc.pdf"],
            "file_quick_hashes": {"/old/doc.pdf": "oldquick"},
            "quick_hash": "oldquick",
        }
        qdrant.scroll.return_value = ([point], None)

        sync = MetadataBasedSync(qdrant)
        renamed = sync.handle_renames(
            "test-corpus",
            [("/old/doc.pdf", "/new/doc.pdf", {"filename": "doc.pdf", "quick_hash": "newquick"})],
        )

        assert renamed == 1
        payload = qdrant.set_payload.call_args.kwargs["payload"]
        assert payload["file_path"] == "/new/doc.pdf"
        assert payload["filename"] == "doc.pdf"
        assert payload["quick_hash"] == "newquick"
        assert payload["file_paths"] == ["/new/doc.pdf"]
        assert payload["file_quick_hashes"] == {"/new/doc.pdf": "newquick"}

    def test_handle_renames_continues_when_existing_payload_is_missing(self):
        """Base rename payload is still applied when optional multi-path fields are absent."""
        qdrant = Mock()
        point = Mock()
        point.payload = {"quick_hash": "oldquick"}
        qdrant.scroll.return_value = ([point], None)

        sync = MetadataBasedSync(qdrant)
        renamed = sync.handle_renames(
            "test-corpus",
            [("/old/doc.pdf", "/new/doc.pdf", {"filename": "doc.pdf"})],
        )

        assert renamed == 1
        payload = qdrant.set_payload.call_args.kwargs["payload"]
        assert payload["file_path"] == "/new/doc.pdf"
        assert payload["filename"] == "doc.pdf"
        assert "file_paths" not in payload
        assert "file_quick_hashes" not in payload

    def test_handle_renames_continues_when_existing_point_is_not_found(self):
        """Missing lookup data should not block the simple file_path rename update."""
        qdrant = Mock()
        qdrant.scroll.return_value = ([], None)

        sync = MetadataBasedSync(qdrant)
        renamed = sync.handle_renames(
            "test-corpus",
            [("/old/doc.pdf", "/new/doc.pdf", {"filename": "doc.pdf"})],
        )

        assert renamed == 1
        payload = qdrant.set_payload.call_args.kwargs["payload"]
        assert payload == {"file_path": "/new/doc.pdf", "filename": "doc.pdf"}

    def test_handle_renames_falls_back_to_point_quick_hash_for_missing_old_key(self):
        qdrant = Mock()
        point = Mock()
        point.payload = {
            "file_paths": ["/old/doc.pdf"],
            "file_quick_hashes": {"/other/doc.pdf": "otherquick"},
            "quick_hash": "pointquick",
        }
        qdrant.scroll.return_value = ([point], None)

        sync = MetadataBasedSync(qdrant)
        renamed = sync.handle_renames(
            "test-corpus",
            [("/old/doc.pdf", "/new/doc.pdf", {"filename": "doc.pdf"})],
        )

        assert renamed == 1
        payload = qdrant.set_payload.call_args.kwargs["payload"]
        assert payload["file_quick_hashes"] == {
            "/other/doc.pdf": "otherquick",
            "/new/doc.pdf": "pointquick",
        }


class TestRenameTuplesWithMetadata:
    """Tests for building Qdrant rename metadata."""

    def test_rename_tuples_include_new_path_quick_hash(self, tmp_path):
        new_file = tmp_path / "new.pdf"
        new_file.write_text("content")

        with patch("arcaneum.cli.sync.compute_quick_hash", return_value="quick123"):
            tuples = _rename_tuples_with_metadata([("/old/doc.pdf", str(new_file.absolute()))])

        assert tuples == [
            (
                "/old/doc.pdf",
                str(new_file.absolute()),
                {"filename": "new.pdf", "quick_hash": "quick123"},
            )
        ]

    def test_rename_tuples_omit_quick_hash_when_new_path_is_missing(self):
        tuples = _rename_tuples_with_metadata([("/old/doc.pdf", "/missing/doc.pdf")])

        assert tuples == [
            (
                "/old/doc.pdf",
                "/missing/doc.pdf",
                {"filename": "doc.pdf"},
            )
        ]


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

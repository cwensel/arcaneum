"""Focused CLI orchestration tests for file manifests."""

from pathlib import Path
from unittest.mock import Mock, call

from arcaneum.cli import sync as sync_module


def test_code_manifest_migration_reports_verbose_progress(monkeypatch):
    manager = Mock()
    manager.qdrant = Mock()
    messages = []
    monkeypatch.setattr(sync_module, "file_manifests_ready", lambda *_: False)
    monkeypatch.setattr(sync_module, "print_info", messages.append)

    def backfill(corpus, progress_callback):
        assert corpus == "Code"
        progress_callback(100)
        progress_callback(2500)  # Throttled from human output.
        progress_callback(10000)
        return 42

    manager.backfill_file_manifests.side_effect = backfill

    migrated = sync_module._ensure_file_manifests(
        manager,
        "Code",
        "code",
        dry_run=False,
        verbose=True,
        output_json=False,
    )

    assert migrated == 42
    assert messages == [
        "Migrating legacy code metadata to file manifests...",
        "  Manifest migration scanned 100 legacy chunks...",
        "  Manifest migration scanned 10,000 legacy chunks...",
        "File manifest migration complete: 42 files",
    ]


def test_manifest_migration_supports_all_corpus_types_but_not_dry_run(monkeypatch):
    manager = Mock()
    manager.qdrant = Mock()
    ready = Mock(return_value=False)
    monkeypatch.setattr(sync_module, "file_manifests_ready", ready)

    manager.backfill_file_manifests.return_value = 3
    assert (
        sync_module._ensure_file_manifests(
            manager, "Docs", "markdown", dry_run=False, verbose=True, output_json=False
        )
        == 3
    )
    assert (
        sync_module._ensure_file_manifests(
            manager, "Code", "code", dry_run=True, verbose=True, output_json=False
        )
        == 0
    )
    manager.backfill_file_manifests.assert_called_once()


def test_successful_code_index_publishes_complete_manifest(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("value = 1\n")
    manager = Mock()

    sync_module._upsert_file_manifest(
        manager,
        "Code",
        "code",
        source,
        "quick",
        file_hash="content",
        chunk_count=3,
    )

    manager.upsert_file_manifest.assert_called_once_with(
        "Code",
        str(source.absolute()),
        "quick",
        file_hash="content",
        chunk_count=3,
        file_size=source.stat().st_size,
        store_type="code",
    )


def test_repair_and_qdrant_backfill_share_manifest_publication(tmp_path):
    source = tmp_path / "repair.py"
    source.write_text("pass\n")
    manager = Mock()

    for chunk_count in (2, 4):
        sync_module._upsert_file_manifest(
            manager,
            "Code",
            "code",
            source,
            f"quick-{chunk_count}",
            file_hash=f"hash-{chunk_count}",
            chunk_count=chunk_count,
        )

    assert manager.upsert_file_manifest.call_count == 2
    assert manager.upsert_file_manifest.call_args_list[1].kwargs["chunk_count"] == 4


def test_code_rename_replaces_manifest_and_stale_cleanup_deletes(tmp_path):
    old_path = str(tmp_path / "old.py")
    new_file = tmp_path / "new.py"
    new_file.write_text("pass\n")
    stale_path = str(tmp_path / "stale.py")
    manager = Mock()

    sync_module._rename_file_manifests(manager, "Code", "code", [(old_path, str(new_file))])
    sync_module._delete_file_manifests(manager, "Code", "code", [stale_path])

    assert manager.method_calls[0] == call.copy_file_manifest(
        "Code",
        old_path,
        str(new_file),
        sync_module.compute_quick_hash(new_file),
        delete_source=True,
        file_size=new_file.stat().st_size,
        store_type="code",
    )
    assert manager.method_calls[1:] == [call.delete_file_manifest("Code", stale_path)]


def test_manifest_lifecycle_supports_markdown_and_pdf(tmp_path):
    source = Path(tmp_path / "document.md")
    source.write_text("# Title\n")
    manager = Mock()

    for corpus_type in ("markdown", "pdf"):
        sync_module._upsert_file_manifest(
            manager,
            "Docs",
            corpus_type,
            source,
            "quick",
            file_hash="content",
            chunk_count=1,
        )
        sync_module._delete_file_manifests(manager, "Docs", corpus_type, [str(source)])

    assert manager.upsert_file_manifest.call_count == 2
    assert manager.delete_file_manifest.call_count == 2


def test_zero_chunk_file_publishes_manifest(tmp_path):
    source = tmp_path / "empty.md"
    source.write_text("")
    manager = Mock()

    sync_module._upsert_file_manifest(
        manager,
        "Docs",
        "markdown",
        source,
        "quick",
        file_hash="content",
        chunk_count=0,
    )

    manager.upsert_file_manifest.assert_called_once_with(
        "Docs",
        str(source),
        "quick",
        file_hash="content",
        chunk_count=0,
        file_size=0,
        store_type="markdown",
    )

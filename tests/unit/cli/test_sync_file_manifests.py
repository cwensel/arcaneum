"""Focused CLI orchestration tests for file manifests."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from arcaneum.cli import sync as sync_module
from arcaneum.indexing.policy import build_indexing_policy


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
        indexing_policy=build_indexing_policy("code"),
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
        indexing_policy=build_indexing_policy("markdown"),
    )


def _pdf_alias_manifests(canonical_path, alias_path):
    shared = {
        "file_hash": "content",
        "chunk_count": 3,
        "store_type": "pdf",
        "canonical_path": canonical_path,
    }
    return {
        canonical_path: {**shared, "file_path": canonical_path, "quick_hash": "canonical"},
        alias_path: {**shared, "file_path": alias_path, "quick_hash": "alias"},
    }


def test_pdf_stale_canonical_promotes_live_alias_after_readback(tmp_path, monkeypatch):
    canonical_path = str(tmp_path / "missing.pdf")
    alias_file = tmp_path / "surviving.pdf"
    alias_file.write_bytes(b"same pdf")
    alias_path = str(alias_file)
    manager = Mock()
    manager.get_file_manifest_snapshot.return_value = _pdf_alias_manifests(
        canonical_path, alias_path
    )
    manager.handle_renames.return_value = 1
    manager.has_chunks_for_file_path.side_effect = lambda _, path: path == alias_path
    manager.get_file_manifest_snapshot.side_effect = [
        _pdf_alias_manifests(canonical_path, alias_path),
        {
            alias_path: {
                **_pdf_alias_manifests(canonical_path, alias_path)[alias_path],
                "canonical_path": alias_path,
            }
        },
        {
            alias_path: {
                **_pdf_alias_manifests(canonical_path, alias_path)[alias_path],
                "canonical_path": alias_path,
            }
        },
    ]
    meili = Mock()
    qdrant = Mock()
    monkeypatch.setattr(
        sync_module,
        "_handle_renames_meili",
        Mock(return_value=(3, [(canonical_path, alias_path)])),
    )
    monkeypatch.setattr(
        sync_module,
        "_meili_has_documents_for_path",
        lambda _meili, _corpus, path: path == alias_path,
    )

    removed = sync_module._remove_indexed_paths(
        qdrant, meili, manager, "Papers", "pdf", [canonical_path]
    )

    assert removed == 1
    manager.handle_renames.assert_called_once()
    manager.set_file_manifest_canonical_path.assert_called_once_with(
        "Papers", alias_path, alias_path
    )
    manager.delete_file_manifest.assert_called_once_with("Papers", canonical_path)
    manager.remove_alternate_path.assert_not_called()
    meili.delete_documents_by_file_paths.assert_not_called()
    qdrant.delete.assert_not_called()


def test_pdf_canonical_promotion_rolls_back_before_manifest_delete(tmp_path, monkeypatch):
    canonical_path = str(tmp_path / "missing.pdf")
    alias_file = tmp_path / "surviving.pdf"
    alias_file.write_bytes(b"same pdf")
    alias_path = str(alias_file)
    manager = Mock()
    manager.get_file_manifest_snapshot.return_value = _pdf_alias_manifests(
        canonical_path, alias_path
    )
    manager.handle_renames.side_effect = [1, 1]
    manager.has_chunks_for_file_path.return_value = False
    meili = Mock()
    qdrant = Mock()
    rename_meili = Mock(
        side_effect=[
            (3, [(canonical_path, alias_path)]),
            (3, [(canonical_path, canonical_path)]),
        ]
    )
    monkeypatch.setattr(sync_module, "_handle_renames_meili", rename_meili)
    monkeypatch.setattr(sync_module, "_meili_has_documents_for_path", lambda *_args: True)

    with pytest.raises(RuntimeError, match="read-back verification failed"):
        sync_module._remove_indexed_paths(qdrant, meili, manager, "Papers", "pdf", [canonical_path])

    assert manager.handle_renames.call_count == 2
    reverse = manager.handle_renames.call_args_list[-1].args[1][0]
    assert reverse[:2] == (alias_path, canonical_path)
    manager.delete_file_manifest.assert_not_called()
    manager.set_file_manifest_canonical_path.assert_not_called()
    assert rename_meili.call_args_list[-1].args[0] == [(canonical_path, canonical_path)]


def test_pdf_stale_alias_only_removes_alias_metadata(tmp_path):
    canonical_file = tmp_path / "canonical.pdf"
    canonical_file.write_bytes(b"same pdf")
    canonical_path = str(canonical_file)
    alias_path = str(tmp_path / "missing-alias.pdf")
    manager = Mock()
    manager.get_file_manifest_snapshot.return_value = _pdf_alias_manifests(
        canonical_path, alias_path
    )
    manager.remove_alternate_path.return_value = 1
    meili = Mock()
    qdrant = Mock()

    removed = sync_module._remove_indexed_paths(
        qdrant, meili, manager, "Papers", "pdf", [alias_path]
    )

    assert removed == 1
    manager.remove_alternate_path.assert_called_once_with("Papers", "content", alias_path)
    manager.delete_file_manifest.assert_called_once_with("Papers", alias_path)
    meili.delete_documents_by_file_paths.assert_not_called()
    qdrant.delete.assert_not_called()


def test_pdf_last_stale_alias_deletes_source_less_canonical_chunks(tmp_path):
    canonical_path = str(tmp_path / "missing-canonical.pdf")
    alias_path = str(tmp_path / "missing-alias.pdf")
    manager = Mock()
    manager.get_file_manifest_snapshot.return_value = _pdf_alias_manifests(
        canonical_path, alias_path
    )
    manager.remove_alternate_path.return_value = 0
    meili = Mock()
    qdrant = Mock()

    removed = sync_module._remove_indexed_paths(
        qdrant, meili, manager, "Papers", "pdf", [alias_path]
    )

    assert removed == 1
    meili.delete_documents_by_file_paths.assert_called_once_with("Papers", [canonical_path])
    assert manager.delete_file_manifest.call_args_list == [
        call("Papers", alias_path),
        call("Papers", canonical_path),
    ]
    qdrant.delete.assert_called_once()


def test_pdf_stale_alias_without_chunks_uses_ordinary_cleanup(tmp_path):
    canonical_path = str(tmp_path / "missing-canonical.pdf")
    alias_path = str(tmp_path / "missing-alias.pdf")
    manager = Mock()
    manager.get_file_manifest_snapshot.return_value = _pdf_alias_manifests(
        canonical_path, alias_path
    )
    manager.remove_alternate_path.return_value = None
    meili = Mock()
    qdrant = Mock()

    removed = sync_module._remove_indexed_paths(
        qdrant, meili, manager, "Papers", "pdf", [alias_path]
    )

    assert removed == 1
    meili.delete_documents_by_file_paths.assert_called_once_with("Papers", [alias_path])
    manager.delete_file_manifest.assert_called_once_with("Papers", alias_path)


def test_pdf_stale_alias_removes_provenance_when_canonical_is_already_missing(tmp_path):
    canonical_path = str(tmp_path / "missing-canonical.pdf")
    alias_path = str(tmp_path / "missing-alias.pdf")
    manager = Mock()
    manager.get_file_manifest_snapshot.return_value = _pdf_alias_manifests(
        canonical_path, alias_path
    )
    manager.remove_alternate_path.return_value = 1
    meili = Mock()
    qdrant = Mock()

    removed = sync_module._remove_indexed_paths(
        qdrant, meili, manager, "Papers", "pdf", [alias_path]
    )

    assert removed == 1
    manager.remove_alternate_path.assert_called_once_with("Papers", "content", alias_path)
    manager.delete_file_manifest.assert_called_once_with("Papers", alias_path)
    qdrant.delete.assert_not_called()


def test_pdf_promotion_repoints_only_surviving_aliases(tmp_path, monkeypatch):
    canonical_path = str(tmp_path / "missing-canonical.pdf")
    stale_alias = str(tmp_path / "missing-alias.pdf")
    live_file = tmp_path / "live.pdf"
    live_file.write_bytes(b"same pdf")
    live_alias = str(live_file)
    shared = {
        "file_hash": "content",
        "chunk_count": 3,
        "store_type": "pdf",
        "canonical_path": canonical_path,
    }
    manager = Mock()
    manager.get_file_manifest_snapshot.return_value = {
        path: {**shared, "file_path": path, "quick_hash": path}
        for path in (canonical_path, stale_alias, live_alias)
    }
    promote = Mock()
    monkeypatch.setattr(sync_module, "_promote_pdf_canonical", promote)

    removed, ordinary = sync_module._remove_pdf_alias_paths(
        Mock(), Mock(), manager, "Papers", [canonical_path, stale_alias]
    )

    assert removed == 2
    assert ordinary == []
    assert promote.call_args.args[-1] == [live_alias]


def test_meili_rename_confirms_pdf_alias_without_moving_canonical_documents():
    qdrant = Mock()
    qdrant.scroll.side_effect = [([], None), ([SimpleNamespace(id="chunk")], None)]
    meili = Mock()

    updated, confirmed = sync_module._handle_renames_meili(
        [("/old-alias.pdf", "/new-alias.pdf")], qdrant, meili, "Papers"
    )

    assert updated == 0
    assert confirmed == [("/old-alias.pdf", "/new-alias.pdf")]
    meili.get_index.return_value.update_documents.assert_not_called()


def test_pdf_canonical_rename_repoints_other_alias_manifests(tmp_path):
    old_path = str(tmp_path / "old.pdf")
    new_file = tmp_path / "new.pdf"
    new_file.write_bytes(b"pdf")
    alias_path = str(tmp_path / "alias.pdf")
    manager = Mock()
    manager.get_file_manifest_snapshot.return_value = _pdf_alias_manifests(old_path, alias_path)

    sync_module._rename_file_manifests(manager, "Papers", "pdf", [(old_path, str(new_file))])

    manager.set_file_manifest_canonical_path.assert_called_once_with(
        "Papers", alias_path, str(new_file)
    )

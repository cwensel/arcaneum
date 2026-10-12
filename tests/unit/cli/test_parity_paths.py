"""How parity handles paths that differ between Qdrant and MeiliSearch.

Pruning (kata 96r3): parity can only re-embed MeiliSearch-only files that
still exist on disk. When the source is gone too, `--prune-orphans` deletes
the MeiliSearch documents, but only when nothing in Qdrant still points at
the path and the file's directory exists (a missing directory is more likely
an unmounted volume than a deletion; `--force-prune` overrides that guard).

Reconciliation (kata yn5w): Qdrant's side comes from file manifests, which
can disagree with the stored chunks. A manifest with zero chunks (an empty
file) is not drift, and chunks whose manifest is missing are not absent from
Qdrant: their manifest is rebuilt instead of re-embedding the file.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from arcaneum.cli import sync as sync_module
from arcaneum.cli.main import cli
from arcaneum.cli.output import capture_json


class FakeQdrant:
    def __init__(self, chunks_by_path):
        self.chunks_by_path = chunks_by_path
        self.count_calls = 0

    def get_collection(self, _name):
        return object()

    def count(self, collection_name, count_filter, exact):
        self.count_calls += 1
        (path,) = [c.match.value for c in count_filter.must if c.key == "file_path"]
        return SimpleNamespace(count=self.chunks_by_path.get(path, 0))


class FakeMeili:
    def __init__(self, paths):
        self.paths = set(paths)
        self.deleted = []

    def health_check(self):
        return True

    def index_exists(self, _name):
        return True

    def get_all_file_paths(self, _name):
        return set(self.paths)

    def delete_documents_by_file_paths(self, _name, paths):
        self.deleted.append(list(paths))
        return {"deleted_count": len(paths), "task_uids": []}


class FakeSyncManager:
    def __init__(self, indexed_paths, manifests, alias_paths=()):
        self.indexed_paths = set(indexed_paths)
        self.manifests = manifests
        self.alias_paths = set(alias_paths)
        self.rebuilt = []

    def _get_indexed_file_paths_set(self, _corpus):
        return set(self.indexed_paths)

    def get_alias_file_paths(self, _corpus):
        return set(self.alias_paths)

    def get_file_manifest_snapshot(self, _corpus):
        return self.manifests

    def rebuild_file_manifest(self, _corpus, path, *, store_type):
        self.rebuilt.append((path, store_type))
        return True


@pytest.fixture
def layout(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    return SimpleNamespace(
        # Deleted from a directory that still exists: a true orphan.
        gone=str(docs / "gone.pdf"),
        # Directory is missing too: maybe an unmounted volume.
        unmounted=str(tmp_path / "Volumes" / "Ext" / "paper.pdf"),
        # No manifest, but Qdrant still holds chunks under this path.
        has_chunks=str(docs / "has-chunks.pdf"),
        # Canonical copy of a duplicate PDF whose own manifest is gone.
        canonical=str(docs / "canonical.pdf"),
        alias=str(docs / "alias.pdf"),
    )


def _classify(layout, force_prune=False):
    qdrant = FakeQdrant({layout.has_chunks: 3})
    manifests = {layout.alias: {"canonical_path": layout.canonical}}
    with patch.object(sync_module, "file_manifests_ready", lambda *_args: True):
        return sync_module._classify_meili_orphans(
            qdrant,
            FakeSyncManager([layout.alias], manifests),
            "Docs",
            [layout.gone, layout.unmounted, layout.has_chunks, layout.canonical],
            force_prune=force_prune,
        )


def test_classify_keeps_paths_still_referenced_by_qdrant(layout):
    result = _classify(layout)

    assert result == {
        "prunable": [layout.gone],
        "dir_missing": [layout.unmounted],
        "referenced": sorted([layout.has_chunks, layout.canonical]),
    }


def test_classify_force_prune_overrides_missing_directory_guard(layout):
    result = _classify(layout, force_prune=True)

    assert result["prunable"] == sorted([layout.gone, layout.unmounted])
    assert result["dir_missing"] == []
    # Referenced paths are never pruned, even when forced.
    assert result["referenced"] == sorted([layout.has_chunks, layout.canonical])


def _run_parity(monkeypatch, layout, **options):
    return _parity(
        monkeypatch,
        points={layout.has_chunks: 3},
        meili_paths=[layout.gone, layout.unmounted, layout.has_chunks, layout.canonical],
        sync_manager=FakeSyncManager(
            [layout.alias],
            {layout.alias: {"canonical_path": layout.canonical}},
            alias_paths=[layout.alias],
        ),
        **options,
    )[:3]


def _parity(monkeypatch, *, points, meili_paths, sync_manager, corpus_type="pdf", **options):
    qdrant = FakeQdrant(points)
    meili = FakeMeili(meili_paths)
    monkeypatch.setattr(sync_module, "create_qdrant_client", lambda **_kwargs: qdrant)
    monkeypatch.setattr(sync_module, "get_meili_client", lambda: meili)
    monkeypatch.setattr(sync_module, "MetadataBasedSync", lambda _qdrant: sync_manager)
    monkeypatch.setattr(sync_module, "file_manifests_ready", lambda *_args: True)
    monkeypatch.setattr(sync_module, "get_collection_type", lambda *_args: corpus_type)
    monkeypatch.setattr(sync_module, "get_collection_metadata", lambda *_args: {"model": "x"})
    monkeypatch.setattr(
        sync_module,
        "_maybe_backfill_legacy_prompt_policy",
        lambda _q, _c, _t, _m, metadata, _o: (metadata, False),
    )

    output_json = options.pop("output_json", True)
    docs = []
    with capture_json(docs):
        sync_module._parity_single_corpus(
            corpus="Docs",
            dry_run=options.pop("dry_run", False),
            verify=False,
            repair_metadata=False,
            effective_text_workers=1,
            qdrant_timeout=120,
            max_embedding_batch=None,
            create_missing=False,
            verbose=False,
            output_json=output_json,
            all_corpora_mode=True,
            **options,
        )
    return qdrant, meili, docs, sync_manager


def test_parity_prunes_only_safe_orphans(monkeypatch, layout):
    _qdrant, meili, docs = _run_parity(monkeypatch, layout, prune_orphans=True)

    assert meili.deleted == [[layout.gone]]
    data = docs[-1]["data"]
    assert data["pruned_orphans"] == [layout.gone]
    assert data["orphans_kept_dir_missing"] == [layout.unmounted]


def test_parity_dry_run_previews_pruning(monkeypatch, layout):
    _qdrant, meili, docs = _run_parity(monkeypatch, layout, prune_orphans=True, dry_run=True)

    assert meili.deleted == []
    data = docs[-1]["data"]
    assert data["would_prune_orphans"] == [layout.gone]
    assert data["orphans_kept_dir_missing"] == [layout.unmounted]


def test_parity_without_flag_deletes_nothing_and_hints(monkeypatch, layout, capsys):
    def no_classification(*_args, **_kwargs):
        raise AssertionError("orphans are only classified with --prune-orphans")

    monkeypatch.setattr(sync_module, "_classify_meili_orphans", no_classification)
    _qdrant, meili, _docs = _run_parity(monkeypatch, layout, output_json=False)

    assert meili.deleted == []
    assert "--prune-orphans" in capsys.readouterr().out


def test_reconcile_splits_manifest_and_chunk_disagreements():
    # Qdrant's side: a real file, an empty file (manifest, zero chunks), and an alias.
    # MeiliSearch's side: a file whose chunks lost their manifest, and a true gap.
    sync_manager = FakeSyncManager(
        ["/d/a.md", "/d/empty.md", "/d/alias.md"], {}, alias_paths=["/d/alias.md"]
    )
    qdrant = FakeQdrant({"/d/a.md": 4, "/d/no-manifest.md": 96})

    result = sync_module._reconcile_parity_paths(
        qdrant,
        sync_manager,
        "Docs",
        qdrant_file_paths={"/d/a.md", "/d/empty.md", "/d/alias.md"},
        meili_file_paths={"/d/no-manifest.md", "/d/only-meili.md"},
    )

    assert result == {
        "missing_from_meili": {"/d/a.md"},
        "missing_from_qdrant": {"/d/only-meili.md"},
        "empty_files": {"/d/empty.md"},
        "manifest_missing": {"/d/no-manifest.md"},
    }


def _manifest_drift(monkeypatch, tmp_path, **options):
    """StateMachineRes/Claude shape: an empty file and chunks without a manifest."""
    empty = tmp_path / "_index.md"
    empty.write_text("---\ntitle: Docs\n---\n")
    unrecorded = tmp_path / "session.md"
    unrecorded.write_text("indexed long ago")
    return (
        _parity(
            monkeypatch,
            points={str(unrecorded): 96},
            meili_paths=[str(unrecorded)],
            sync_manager=FakeSyncManager([str(empty)], {str(empty): {"chunk_count": 0}}),
            corpus_type="markdown",
            **options,
        ),
        str(empty),
        str(unrecorded),
    )


def test_parity_dry_run_reports_no_backfill_for_manifest_drift(monkeypatch, tmp_path):
    (_q, _m, docs, sync_manager), _empty, unrecorded = _manifest_drift(
        monkeypatch, tmp_path, dry_run=True
    )

    data = docs[-1]["data"]
    assert data["would_backfill_to_meili"] == 0
    assert data["would_backfill_to_qdrant"] == 0
    assert data["would_rebuild_manifests"] == [unrecorded]
    assert sync_manager.rebuilt == []


def test_parity_rebuilds_missing_manifests_instead_of_reembedding(monkeypatch, tmp_path):
    def no_reembedding(*_args, **_kwargs):
        raise AssertionError("files with stored chunks must not be re-embedded")

    monkeypatch.setattr(sync_module, "_backfill_meili_to_qdrant", no_reembedding)
    (_q, _m, docs, sync_manager), _empty, unrecorded = _manifest_drift(monkeypatch, tmp_path)

    assert sync_manager.rebuilt == [(unrecorded, "markdown")]
    data = docs[-1]["data"]
    assert data["manifests_rebuilt"] == [unrecorded]
    assert data["qdrant_backfilled"] == 0
    assert data["meili_backfilled"] == 0


def test_parity_reports_already_in_parity_for_empty_files_only(monkeypatch, tmp_path):
    empty = tmp_path / "_index.md"
    empty.write_text("---\n---\n")
    _q, _m, docs, _sm = _parity(
        monkeypatch,
        points={},
        meili_paths=[],
        sync_manager=FakeSyncManager([str(empty)], {str(empty): {"chunk_count": 0}}),
        corpus_type="markdown",
    )

    assert docs[-1]["message"] == "Indexes are already in parity"


@pytest.fixture
def isolated_locks(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], (False, False)),
        (["--prune-orphans"], (True, False)),
        # --force-prune implies --prune-orphans.
        (["--force-prune"], (True, True)),
    ],
)
def test_parity_cli_passes_prune_flags(isolated_locks, flags, expected):
    seen = {}

    def single(**kwargs):
        seen["flags"] = (kwargs["prune_orphans"], kwargs["force_prune"])

    with patch("arcaneum.cli.sync._parity_single_corpus", single):
        result = CliRunner().invoke(
            cli, ["corpus", "parity", "Docs", *flags], catch_exceptions=False
        )

    assert result.exit_code == 0, result.output
    assert seen["flags"] == expected


def test_repair_cli_passes_prune_flags_to_parity(isolated_locks):
    seen = {}

    def single(**kwargs):
        seen["flags"] = (kwargs["prune_orphans"], kwargs["force_prune"])

    with (
        patch("arcaneum.cli.sync._sync_directory_locked", lambda *_a, **_k: None),
        patch("arcaneum.cli.sync._parity_single_corpus", single),
        patch("arcaneum.cli.utils.create_qdrant_client", side_effect=RuntimeError("no server")),
    ):
        result = CliRunner().invoke(
            cli, ["corpus", "repair", "Docs", "--prune-orphans"], catch_exceptions=False
        )

    assert result.exit_code == 0, result.output
    assert seen["flags"] == (True, False)


@pytest.mark.parametrize("dry_run", [False, True])
def test_sync_parity_plan_skips_empty_files_and_rebuilds_missing_manifests(tmp_path, dry_run):
    """`arc corpus sync --parity` shares the reconciliation."""
    real = tmp_path / "real.md"
    real.write_text("content")
    empty = tmp_path / "_index.md"
    empty.write_text("---\n---\n")
    unrecorded = tmp_path / "session.md"
    unrecorded.write_text("indexed long ago")
    only_meili = tmp_path / "only-meili.md"
    only_meili.write_text("needs embedding")
    sync_manager = FakeSyncManager([str(real), str(empty)], {})
    qdrant = FakeQdrant({str(real): 2, str(unrecorded): 96})

    meili_backfill, qdrant_backfill = sync_module._plan_parity_backfill(
        qdrant,
        sync_manager,
        "Docs",
        "markdown",
        qdrant_file_paths={str(real), str(empty)},
        meili_file_paths={str(unrecorded), str(only_meili)},
        dry_run=dry_run,
        output_json=True,
    )

    assert meili_backfill == [str(real)]
    assert qdrant_backfill == [str(only_meili)]
    assert sync_manager.rebuilt == ([] if dry_run else [(str(unrecorded), "markdown")])

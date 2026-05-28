"""Integration tests for the shared prune+stamp orphan flow (C3/C4/C5).

These exercise the single helper all force paths call so the stamp contract is
enforced identically: stamp is skipped when orphans remain and --prune is off,
applied after --prune removes orphans, and never applied for partial
--file-list force runs.
"""

from types import SimpleNamespace

import pytest

from arcaneum.embeddings.client import get_embedding_prompt_policies
from arcaneum.indexing.collection_metadata import (
    MultiRootPruneError,
    prune_orphans_and_stamp,
)


class FakeSync:
    """Records delete_chunks_by_file_path calls and reports indexed paths."""

    def __init__(self, indexed_paths):
        self._indexed = set(indexed_paths)
        self.deleted = []

    def _get_indexed_file_paths_set(self, collection_name):
        return set(self._indexed)

    def delete_chunks_by_file_path(self, collection_name, file_path):
        self.deleted.append(file_path)
        self._indexed.discard(file_path)
        return 1

    def has_chunks_for_file_path(self, collection_name, file_path):
        # Models actual collection state: a path still "has chunks" until its
        # delete removed it from the indexed set.
        return file_path in self._indexed


class NoOpDeleteSync(FakeSync):
    """delete removes nothing (returns 0) and the chunks REMAIN on disk.

    Models a genuine no-op/failed prune where stale vectors survive: the path
    is never discarded, so has_chunks_for_file_path stays True and the orphan
    must remain (stamp withheld).
    """

    def delete_chunks_by_file_path(self, collection_name, file_path):
        self.deleted.append(file_path)
        return 0


class AlreadyClearedSync(FakeSync):
    """delete returns 0 because the path's chunks were ALREADY cleared.

    Models the source pipeline: branch-delete removed the orphan's chunks
    before the per-file prune ran, so the delete matches nothing (0) but the
    path genuinely has no chunks remaining -> orphan IS resolved.
    """

    def delete_chunks_by_file_path(self, collection_name, file_path):
        self.deleted.append(file_path)
        self._indexed.discard(file_path)
        return 0


class RaisingDeleteSync(FakeSync):
    """delete_chunks_by_file_path raises (e.g. transient Qdrant error)."""

    def delete_chunks_by_file_path(self, collection_name, file_path):
        self.deleted.append(file_path)
        raise RuntimeError("delete failed")


class StampQdrant:
    def __init__(self, metadata):
        self.metadata = dict(metadata)
        self.upserted = None

    def get_collection(self, _name):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=2))
            )
        )

    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        return [SimpleNamespace(payload={**self.metadata, "is_metadata": True})]

    def upsert(self, collection_name, points):
        self.upserted = points[0].payload


def _meta():
    return {"collection_type": "markdown", "model": "stella", "created_by": "arcaneum"}


def _captured():
    warnings = []
    return warnings, (lambda msg: warnings.append(msg))


def test_stamp_skipped_when_orphans_remain_and_prune_off():
    # Two indexed paths; only one still on disk -> one orphan.
    sync = FakeSync(["/disk/a.md", "/disk/gone.md"])
    qdrant = StampQdrant(_meta())
    warnings, warn = _captured()

    result = prune_orphans_and_stamp(
        qdrant=qdrant,
        sync=sync,
        collection_name="md",
        collection_type="markdown",
        model="stella",
        force=True,
        file_list=None,
        stats={"files": 1, "errors": 0},
        on_disk_paths={"/disk/a.md"},
        pre_run_paths={"/disk/a.md", "/disk/gone.md"},
        prune=False,
        warn=warn,
    )

    assert result["stamped"] is False
    assert result["orphans_remaining"] == 1
    assert sync.deleted == []  # nothing pruned
    assert qdrant.upserted is None  # not stamped
    assert any("--prune" in w for w in warnings)


def test_stamp_applied_after_prune_removes_orphans():
    sync = FakeSync(["/disk/a.md", "/disk/gone.md"])
    qdrant = StampQdrant(_meta())
    warnings, warn = _captured()

    result = prune_orphans_and_stamp(
        qdrant=qdrant,
        sync=sync,
        collection_name="md",
        collection_type="markdown",
        model="stella",
        force=True,
        file_list=None,
        stats={"files": 1, "errors": 0},
        on_disk_paths={"/disk/a.md"},
        pre_run_paths={"/disk/a.md", "/disk/gone.md"},
        prune=True,
        warn=warn,
    )

    assert sync.deleted == ["/disk/gone.md"]  # orphan pruned by path
    assert result["orphans_remaining"] == 0
    assert result["stamped"] is True
    assert qdrant.upserted is not None
    assert qdrant.upserted["embedding_prompt_policy"] == get_embedding_prompt_policies("stella")


def test_partial_file_list_force_never_stamps():
    # No orphans, clean run, but a partial --file-list run must not stamp.
    sync = FakeSync(["/disk/a.md"])
    qdrant = StampQdrant(_meta())
    warnings, warn = _captured()

    result = prune_orphans_and_stamp(
        qdrant=qdrant,
        sync=sync,
        collection_name="md",
        collection_type="markdown",
        model="stella",
        force=True,
        file_list=["/disk/a.md"],
        stats={"files": 1, "errors": 0},
        on_disk_paths={"/disk/a.md"},
        pre_run_paths={"/disk/a.md"},
        prune=False,
        warn=warn,
    )

    assert result["stamped"] is False
    assert qdrant.upserted is None


# --- Fix B: honest orphans_remaining when a delete removes nothing/fails ---


def test_noop_delete_keeps_orphan_remaining_and_does_not_stamp():
    # Two indexed paths, one orphan; --prune set, but the delete removes 0
    # points (no-op/failed prune). orphans_remaining must stay > 0 and the
    # collection must NOT be stamped (job-1921 Fix B).
    sync = NoOpDeleteSync(["/disk/a.md", "/disk/gone.md"])
    qdrant = StampQdrant(_meta())
    warnings, warn = _captured()

    result = prune_orphans_and_stamp(
        qdrant=qdrant,
        sync=sync,
        collection_name="md",
        collection_type="markdown",
        model="stella",
        force=True,
        file_list=None,
        stats={"files": 1, "errors": 0},
        on_disk_paths={"/disk/a.md"},
        pre_run_paths={"/disk/a.md", "/disk/gone.md"},
        prune=True,
        warn=warn,
    )

    assert sync.deleted == ["/disk/gone.md"]  # attempted
    assert result["orphans_pruned"] == 0  # nothing actually removed
    assert result["orphans_remaining"] == 1  # honest: still an orphan
    assert result["stamped"] is False
    assert qdrant.upserted is None


def test_already_cleared_orphan_counts_pruned_and_stamps():
    # Source pipeline reconciliation: the orphan's chunks were already removed
    # earlier in the run (branch-delete), so the per-file delete returns 0 yet
    # no chunks remain. The orphan must count as resolved so a clean force
    # --prune source reindex can stamp (job-1921 review finding #2).
    sync = AlreadyClearedSync(["/disk/a.py", "/disk/gone.py"])
    qdrant = StampQdrant(_meta())
    warnings, warn = _captured()

    result = prune_orphans_and_stamp(
        qdrant=qdrant,
        sync=sync,
        collection_name="code",
        collection_type="code",
        model="stella",
        force=True,
        file_list=None,
        stats={"files": 1, "errors": 0},
        on_disk_paths={"/disk/a.py"},
        pre_run_paths={"/disk/a.py", "/disk/gone.py"},
        prune=True,
        warn=warn,
    )

    assert sync.deleted == ["/disk/gone.py"]
    assert result["orphans_pruned"] == 1  # resolved via state check
    assert result["orphans_remaining"] == 0
    assert result["stamped"] is True
    assert qdrant.upserted is not None


def test_no_prune_already_cleared_orphan_stamps():
    # Source `arc index code DIR --force` (no --prune): a removed repo file's
    # chunks were already cleared by the branch-delete, so the orphan has no
    # remaining chunks. It must NOT withhold the stamp on a clean reindex
    # (job-1921 review finding #2, re-review iteration 2).
    sync = AlreadyClearedSync(["/disk/a.py"])  # gone.py already absent
    sync._indexed = {"/disk/a.py"}  # branch-delete already removed gone.py chunks
    qdrant = StampQdrant(_meta())
    warnings, warn = _captured()

    result = prune_orphans_and_stamp(
        qdrant=qdrant,
        sync=sync,
        collection_name="code",
        collection_type="code",
        model="stella",
        force=True,
        file_list=None,
        stats={"files": 1, "errors": 0},
        on_disk_paths={"/disk/a.py"},
        pre_run_paths={"/disk/a.py", "/disk/gone.py"},
        prune=False,  # no prune, yet the cleared orphan must not block stamping
        warn=warn,
    )

    assert result["orphans"] == ["/disk/gone.py"]
    assert result["orphans_remaining"] == 0  # cleared orphan has no chunks
    assert result["stamped"] is True
    assert qdrant.upserted is not None


def test_no_prune_orphan_with_chunks_still_withholds_stamp():
    # An orphan whose chunks DO remain (markdown/PDF: file deleted, chunks not
    # yet pruned) must still withhold the stamp and warn under bare force.
    sync = FakeSync(["/disk/a.md", "/disk/gone.md"])  # both still have chunks
    qdrant = StampQdrant(_meta())
    warnings, warn = _captured()

    result = prune_orphans_and_stamp(
        qdrant=qdrant,
        sync=sync,
        collection_name="md",
        collection_type="markdown",
        model="stella",
        force=True,
        file_list=None,
        stats={"files": 1, "errors": 0},
        on_disk_paths={"/disk/a.md"},
        pre_run_paths={"/disk/a.md", "/disk/gone.md"},
        prune=False,
        warn=warn,
    )

    assert result["orphans_remaining"] == 1
    assert result["stamped"] is False
    assert qdrant.upserted is None
    assert any("--prune" in w for w in warnings)


def test_failed_delete_leaves_orphan_remaining_and_does_not_stamp():
    # A raised delete must also leave that orphan counted as remaining.
    sync = RaisingDeleteSync(["/disk/a.md", "/disk/gone.md"])
    qdrant = StampQdrant(_meta())
    warnings, warn = _captured()

    result = prune_orphans_and_stamp(
        qdrant=qdrant,
        sync=sync,
        collection_name="md",
        collection_type="markdown",
        model="stella",
        force=True,
        file_list=None,
        stats={"files": 1, "errors": 0},
        on_disk_paths={"/disk/a.md"},
        pre_run_paths={"/disk/a.md", "/disk/gone.md"},
        prune=True,
        warn=warn,
    )

    assert result["orphans_pruned"] == 0
    assert result["orphans_remaining"] == 1
    assert result["stamped"] is False
    assert qdrant.upserted is None


# --- Fix A: multi-root guard for markdown + PDF full-directory force ---


def test_multi_root_prune_refuses_and_deletes_nothing():
    # Collection spans two trees (/dirA and /dirB); we index only /dirA with
    # --prune. /dirB files are NOT orphans — refuse, delete nothing, no stamp.
    sync = FakeSync(["/dirA/a.md", "/dirB/keep.md"])
    qdrant = StampQdrant(_meta())
    warnings, warn = _captured()

    with pytest.raises(MultiRootPruneError):
        prune_orphans_and_stamp(
            qdrant=qdrant,
            sync=sync,
            collection_name="md",
            collection_type="markdown",
            model="stella",
            force=True,
            file_list=None,
            stats={"files": 1, "errors": 0},
            on_disk_paths={"/dirA/a.md"},
            pre_run_paths={"/dirA/a.md", "/dirB/keep.md"},
            prune=True,
            indexed_dir="/dirA",
            warn=warn,
        )

    assert sync.deleted == []  # data loss prevented
    assert qdrant.upserted is None  # not stamped


def test_multi_root_bare_force_skips_stamp_and_warns():
    # Bare force (no prune) on a multi-root collection: skip the stamp, warn,
    # and do NOT classify/count the other-directory files as orphans.
    sync = FakeSync(["/dirA/a.md", "/dirB/keep.md"])
    qdrant = StampQdrant(_meta())
    warnings, warn = _captured()

    result = prune_orphans_and_stamp(
        qdrant=qdrant,
        sync=sync,
        collection_name="md",
        collection_type="markdown",
        model="stella",
        force=True,
        file_list=None,
        stats={"files": 1, "errors": 0},
        on_disk_paths={"/dirA/a.md"},
        pre_run_paths={"/dirA/a.md", "/dirB/keep.md"},
        prune=False,
        indexed_dir="/dirA",
        warn=warn,
    )

    assert result["stamped"] is False
    assert result["orphans"] == []  # other-dir files not classified as orphans
    assert result["orphans_pruned"] == 0
    assert result["orphans_remaining"] == 0
    assert sync.deleted == []
    assert qdrant.upserted is None
    assert any("multiple directories" in w for w in warnings)


def test_scope_limited_existing_files_are_not_pruned():
    # HIGH regression (re-review iteration 2): a scope-limited run (e.g.
    # --no-recursive) must not delete still-existing indexed files that the
    # current scan did not cover. The CLI now builds on_disk_paths from
    # existence (pre_run filtered by Path.exists), so an existing subdirectory
    # file appears in on_disk and is NOT classified as an orphan even though it
    # was outside this run's discovery scope. Only the genuinely-missing file
    # is pruned.
    sync = FakeSync(["/dir/top.md", "/dir/sub/keep.md", "/dir/gone.md"])
    qdrant = StampQdrant(_meta())
    warnings, warn = _captured()

    # on_disk = pre_run filtered by existence: top + sub/keep still exist,
    # gone.md does not. (Discovery scope is irrelevant to this set.)
    on_disk = {"/dir/top.md", "/dir/sub/keep.md"}

    result = prune_orphans_and_stamp(
        qdrant=qdrant,
        sync=sync,
        collection_name="md",
        collection_type="markdown",
        model="stella",
        force=True,
        file_list=None,
        stats={"files": 1, "errors": 0},
        on_disk_paths=on_disk,
        pre_run_paths={"/dir/top.md", "/dir/sub/keep.md", "/dir/gone.md"},
        prune=True,
        indexed_dir="/dir",
        warn=warn,
    )

    assert result["orphans"] == ["/dir/gone.md"]  # only the missing file
    assert sync.deleted == ["/dir/gone.md"]  # subdir file NOT deleted
    assert "/dir/sub/keep.md" not in sync.deleted
    assert result["orphans_remaining"] == 0
    assert result["stamped"] is True


def test_scope_limited_run_does_not_certify_uncovered_files():
    # MEDIUM regression (re-review iteration 4): a scope-limited force (e.g.
    # markdown --no-recursive) re-embeds only part of the collection. Files that
    # still exist but were NOT covered by this run retain potentially-stale
    # vectors, so the collection must NOT be certified even though there are no
    # orphans to prune.
    sync = FakeSync(["/dir/top.md", "/dir/sub/deep.md"])
    qdrant = StampQdrant(_meta())
    warnings, warn = _captured()

    result = prune_orphans_and_stamp(
        qdrant=qdrant,
        sync=sync,
        collection_name="md",
        collection_type="markdown",
        model="stella",
        force=True,
        file_list=None,
        stats={"files": 1, "errors": 0},
        on_disk_paths={"/dir/top.md", "/dir/sub/deep.md"},  # both still on disk
        pre_run_paths={"/dir/top.md", "/dir/sub/deep.md"},
        prune=False,
        indexed_dir="/dir",
        covered_paths={"/dir/top.md"},  # --no-recursive: only top level covered
        warn=warn,
    )

    assert result["orphans"] == []  # nothing missing -> no orphans/no deletes
    assert sync.deleted == []
    assert result["stamped"] is False  # uncovered file bars certification
    assert qdrant.upserted is None
    assert any("not re-indexed" in w for w in warnings)


def test_multi_root_source_reindex_does_not_certify_other_root():
    # MEDIUM regression (re-review iteration 5): a code collection spanning two
    # source roots, reindexed under only /repos/A. covered_paths is scoped to
    # the run's root, so /repos/B's still-existing files are uncovered and the
    # collection must NOT be certified (their vectors may be stale). Mirrors how
    # index_source now scopes covered_paths to source_dir via _is_under.
    sync = FakeSync(["/repos/A/x.py", "/repos/B/y.py"])
    qdrant = StampQdrant(_meta())
    warnings, warn = _captured()

    result = prune_orphans_and_stamp(
        qdrant=qdrant,
        sync=sync,
        collection_name="code",
        collection_type="code",
        model="stella",
        force=True,
        file_list=None,
        stats={"files": 1, "errors": 0},
        on_disk_paths={"/repos/A/x.py", "/repos/B/y.py"},  # both still on disk
        pre_run_paths={"/repos/A/x.py", "/repos/B/y.py"},
        prune=False,
        indexed_dir=None,  # source keeps collection-wide orphan semantics
        covered_paths={"/repos/A/x.py"},  # only the reindexed root is covered
        warn=warn,
    )

    assert result["orphans"] == []  # /repos/B/y.py still exists -> not an orphan
    assert sync.deleted == []  # and is never deleted
    assert result["stamped"] is False  # but bars certification
    assert qdrant.upserted is None
    assert any("not re-indexed" in w for w in warnings)


def test_full_coverage_run_certifies():
    # When covered_paths includes every still-existing indexed file, the
    # coverage gate allows the stamp (full reindex certifies the collection).
    sync = FakeSync(["/dir/top.md", "/dir/sub/deep.md"])
    qdrant = StampQdrant(_meta())
    warnings, warn = _captured()

    result = prune_orphans_and_stamp(
        qdrant=qdrant,
        sync=sync,
        collection_name="md",
        collection_type="markdown",
        model="stella",
        force=True,
        file_list=None,
        stats={"files": 2, "errors": 0},
        on_disk_paths={"/dir/top.md", "/dir/sub/deep.md"},
        pre_run_paths={"/dir/top.md", "/dir/sub/deep.md"},
        prune=False,
        indexed_dir="/dir",
        covered_paths={"/dir/top.md", "/dir/sub/deep.md"},  # recursive: all covered
        warn=warn,
    )

    assert result["stamped"] is True
    assert qdrant.upserted is not None


def test_single_root_force_prune_prunes_orphans_and_stamps():
    # Regression: single-root collection + force --prune still prunes genuine
    # orphans and stamps (good behavior survives the multi-root guard).
    sync = FakeSync(["/dirA/a.md", "/dirA/gone.md"])
    qdrant = StampQdrant(_meta())
    warnings, warn = _captured()

    result = prune_orphans_and_stamp(
        qdrant=qdrant,
        sync=sync,
        collection_name="md",
        collection_type="markdown",
        model="stella",
        force=True,
        file_list=None,
        stats={"files": 1, "errors": 0},
        on_disk_paths={"/dirA/a.md"},
        pre_run_paths={"/dirA/a.md", "/dirA/gone.md"},
        prune=True,
        indexed_dir="/dirA",
        warn=warn,
    )

    assert sync.deleted == ["/dirA/gone.md"]
    assert result["orphans_pruned"] == 1
    assert result["orphans_remaining"] == 0
    assert result["stamped"] is True
    assert qdrant.upserted is not None

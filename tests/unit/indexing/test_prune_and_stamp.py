"""Integration tests for the shared prune+stamp orphan flow (C3/C4/C5).

These exercise the single helper all force paths call so the stamp contract is
enforced identically: stamp is skipped when orphans remain and --prune is off,
applied after --prune removes orphans, and never applied for partial
--file-list force runs.
"""

from types import SimpleNamespace

from arcaneum.embeddings.client import get_embedding_prompt_policies
from arcaneum.indexing.collection_metadata import (
    METADATA_POINT_ID,
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

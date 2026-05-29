"""Regression tests for the orphan-aware prompt-policy stamp gate (C4).

The stamp certifies that every vector in a collection was produced under the
collection's recorded embedding prompt policy. It may be written ONLY when a
force, full-directory run completed with no errors, indexed at least one file,
and left no orphan vectors (indexed files no longer on disk). Partial
--file-list force runs never stamp.
"""

from types import SimpleNamespace

import pytest

from arcaneum.embeddings.client import get_embedding_prompt_policies
from arcaneum.indexing.collection_metadata import (
    backfill_embedding_prompt_policy,
    should_stamp_prompt_policy,
    stamp_embedding_prompt_policy,
)


class StampQdrant:
    """Qdrant stand-in supporting metadata read/upsert."""

    def __init__(self, metadata=None, retrieve_error=None):
        self.metadata = dict(metadata) if metadata else None
        self.retrieve_error = retrieve_error
        self.upserted = None

    def get_collection(self, _name):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=2))
            )
        )

    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        if self.retrieve_error is not None:
            raise self.retrieve_error
        if self.metadata is not None:
            return [SimpleNamespace(payload={**self.metadata, "is_metadata": True})]
        return []

    def upsert(self, collection_name, points):
        self.upserted = points[0].payload
        self.metadata = dict(self.upserted)


# ---- should_stamp_prompt_policy gate -----------------------------------------

def _stats(files=1, errors=0):
    return {"files": files, "errors": errors}


def test_gate_true_when_force_full_clean_no_orphans():
    assert should_stamp_prompt_policy(
        force=True, file_list=None, stats=_stats(), orphans_remaining=0
    ) is True


def test_gate_false_when_not_force():
    assert should_stamp_prompt_policy(
        force=False, file_list=None, stats=_stats(), orphans_remaining=0
    ) is False


def test_gate_false_on_partial_file_list_force():
    assert should_stamp_prompt_policy(
        force=True, file_list=["/a.md"], stats=_stats(), orphans_remaining=0
    ) is False


def test_gate_false_when_errors_present():
    assert should_stamp_prompt_policy(
        force=True, file_list=None, stats=_stats(errors=2), orphans_remaining=0
    ) is False


def test_gate_false_when_no_files_indexed():
    assert should_stamp_prompt_policy(
        force=True, file_list=None, stats=_stats(files=0), orphans_remaining=0
    ) is False


def test_gate_false_when_orphans_remain():
    assert should_stamp_prompt_policy(
        force=True, file_list=None, stats=_stats(), orphans_remaining=3
    ) is False


# ---- stamp_embedding_prompt_policy writer ------------------------------------

def test_stamp_writes_prompt_policy_into_metadata():
    qdrant = StampQdrant(metadata={
        "collection_type": "markdown",
        "model": "stella",
        "created_by": "arcaneum",
    })

    stamp_embedding_prompt_policy(qdrant, "md", "markdown", "stella")

    assert qdrant.upserted is not None
    assert qdrant.upserted["embedding_prompt_policy"] == get_embedding_prompt_policies("stella")
    # Preserves existing metadata.
    assert qdrant.upserted["collection_type"] == "markdown"


def test_stamp_initializes_legacy_collection_without_metadata_point():
    qdrant = StampQdrant(metadata=None)

    metadata = stamp_embedding_prompt_policy(qdrant, "md", "markdown", "stella")

    assert qdrant.upserted is not None
    assert qdrant.upserted["collection_type"] == "markdown"
    assert qdrant.upserted["model"] == "stella"
    assert qdrant.upserted["embedding_prompt_policy"] == get_embedding_prompt_policies("stella")
    assert metadata["embedding_prompt_policy"] == get_embedding_prompt_policies("stella")


def test_stamp_does_not_initialize_when_metadata_retrieve_fails():
    qdrant = StampQdrant(metadata=None, retrieve_error=RuntimeError("qdrant unavailable"))

    try:
        stamp_embedding_prompt_policy(qdrant, "md", "markdown", "stella")
    except RuntimeError as exc:
        assert str(exc) == "qdrant unavailable"
    else:
        raise AssertionError("expected metadata retrieve failure to propagate")

    assert qdrant.upserted is None


@pytest.mark.parametrize("collection_type", ["pdf", "markdown", "code"])
def test_backfill_prompt_policy_supports_all_corpus_types(collection_type):
    qdrant = StampQdrant(metadata={
        "collection_type": collection_type,
        "model": "stella",
        "created_by": "arcaneum",
    })

    metadata = backfill_embedding_prompt_policy(
        qdrant,
        "docs",
        collection_type,
        "stella",
    )

    assert metadata["collection_type"] == collection_type
    assert metadata["embedding_prompt_policy"] == get_embedding_prompt_policies("stella")

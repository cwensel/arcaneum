"""Tests for duplicate-chunk detection in CollectionVerifier."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from arcaneum.indexing import verify as verify_mod
from arcaneum.indexing.verify import CollectionVerifier


def _point(payload):
    """Build a scroll-result point with the given payload."""
    return SimpleNamespace(id=1, payload=payload, vector=None)


def _scroll_once(points):
    """qdrant.scroll returns (points, next_offset). Emit one batch then stop."""

    def _impl(**kwargs):
        if not hasattr(_impl, "_called"):
            _impl._called = True
            return points, None
        return [], None

    return _impl


@pytest.fixture
def qdrant_client():
    client = MagicMock()
    client.get_collection.return_value = SimpleNamespace(points_count=1)
    return client


def _chunk(index, path="/tmp/dupe.pdf", chunk_count=2):
    return _point(
        {
            "file_path": path,
            "chunk_index": index,
            "chunk_count": chunk_count,
            "text": "some readable text",
        }
    )


def test_duplicate_chunk_indices_are_detected(qdrant_client):
    """A file indexed twice keeps both copies of every chunk.

    Coverage is tracked as a set of chunk_index values, so duplicates collapse
    and the file looks complete. Counting points per file is what exposes it.
    """
    qdrant_client.scroll.side_effect = _scroll_once([_chunk(0), _chunk(1), _chunk(0), _chunk(1)])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=4,
        )

    assert result.duplicate_items == 1
    duped = [f for f in result.files if f.has_duplicate_chunks]
    assert [f.file_path for f in duped] == ["/tmp/dupe.pdf"]
    assert duped[0].duplicate_chunk_count == 2
    assert not result.is_healthy


def test_duplicate_files_are_offered_for_repair(qdrant_client):
    """Duplicated files must reach the repair list, alongside incomplete ones."""
    qdrant_client.scroll.side_effect = _scroll_once([_chunk(0), _chunk(1), _chunk(0), _chunk(1)])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=4,
        )

    assert result.get_items_needing_repair() == ["/tmp/dupe.pdf"]


def test_clean_file_reports_no_duplicates(qdrant_client):
    """A correctly indexed file must not be flagged."""
    qdrant_client.scroll.side_effect = _scroll_once([_chunk(0), _chunk(1)])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=2,
        )

    assert result.duplicate_items == 0
    assert result.get_items_needing_repair() == []

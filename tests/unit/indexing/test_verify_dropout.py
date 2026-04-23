"""Tests for extraction-dropout detection in CollectionVerifier."""

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
        # Emit the batch on first call, then empty on the second
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


def test_dropout_detected_from_payload_page_count(qdrant_client):
    # 500 chars on an 8-page PDF → dropout
    qdrant_client.scroll.side_effect = _scroll_once([
        _point({
            "file_path": "/tmp/fake.pdf",
            "chunk_index": 0,
            "chunk_count": 1,
            "page_count": 8,
            "text": "x" * 500,
        }),
    ])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
        )

    assert result.dropout_items == 1
    assert result.dropout_at_floor == 0
    assert result.files[0].suspected_dropout is True
    assert result.files[0].page_count == 8
    assert result.files[0].total_text_chars == 500
    assert result.get_items_needing_repair() == ["/tmp/fake.pdf"]


def test_extraction_floor_skips_repair(qdrant_client):
    # Same dropout signal, but already marked extraction_floor → don't re-index
    qdrant_client.scroll.side_effect = _scroll_once([
        _point({
            "file_path": "/tmp/floor.pdf",
            "chunk_index": 0,
            "chunk_count": 1,
            "page_count": 8,
            "text": "x" * 500,
            "extraction_floor": True,
        }),
    ])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
        )

    assert result.dropout_items == 0
    assert result.dropout_at_floor == 1
    assert result.files[0].suspected_dropout is False
    # The floor-marked file must NOT show up in the repair list, even though
    # its text density still looks dropout-shaped.
    assert result.get_items_needing_repair() == []


def test_healthy_pdf_not_flagged(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once([
        _point({
            "file_path": "/tmp/good.pdf",
            "chunk_index": 0,
            "chunk_count": 1,
            "page_count": 8,
            "text": "x" * 39000,
        }),
    ])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
        )

    assert result.is_healthy is True
    assert result.dropout_items == 0


def test_dropout_falls_back_to_disk_page_count(qdrant_client, tmp_path):
    # No page_count in payload (simulates older indexed data); verify should
    # read it from disk via _page_count_from_disk.
    qdrant_client.scroll.side_effect = _scroll_once([
        _point({
            "file_path": "/tmp/legacy.pdf",
            "chunk_index": 0,
            "chunk_count": 1,
            "text": "x" * 500,
        }),
    ])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"), \
         patch.object(verify_mod, "_page_count_from_disk", return_value=8):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
        )

    assert result.dropout_items == 1
    assert result.files[0].page_count == 8

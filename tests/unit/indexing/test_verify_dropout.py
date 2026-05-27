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
    assert "suspected_dropout" in result.files[0].quality_manifest["quality_warnings"]
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
    assert result.files[0].quality_manifest["quality_warnings"] == []


def test_chunk_count_detects_missing_tail_chunk(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once([
        _point({
            "file_path": "/tmp/incomplete.pdf",
            "chunk_index": 0,
            "chunk_count": 3,
        }),
        _point({
            "file_path": "/tmp/incomplete.pdf",
            "chunk_index": 1,
            "chunk_count": 3,
        }),
    ])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=2,
        )

    assert result.is_healthy is False
    assert result.files[0].expected_chunks == 3
    assert result.files[0].actual_chunks == 2
    assert result.files[0].missing_indices == [2]
    assert result.files[0].quality_manifest["chunk_count"] == 3


def test_quality_manifest_marks_stale_source(qdrant_client, tmp_path):
    stale = tmp_path / "stale.pdf"
    stale.write_bytes(b"changed")
    qdrant_client.scroll.side_effect = _scroll_once([
        _point({
            "file_path": str(stale),
            "chunk_index": 0,
            "chunk_count": 1,
            "source_hash": "0" * 64,
            "page_count": 1,
            "text": "x" * 5000,
        }),
    ])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
        )

    assert result.is_healthy is False
    assert result.files[0].is_complete is False
    assert "stale_source" in result.files[0].quality_manifest["quality_warnings"]


def test_stale_source_accepts_sync_short_hash(qdrant_client, tmp_path):
    current = tmp_path / "current.pdf"
    current.write_bytes(b"current")
    import hashlib
    short_hash = hashlib.sha256(b"current").hexdigest()[:16]
    qdrant_client.scroll.side_effect = _scroll_once([
        _point({
            "file_path": str(current),
            "chunk_index": 0,
            "chunk_count": 1,
            "source_hash": short_hash,
            "page_count": 1,
            "text": "x" * 5000,
        }),
    ])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
        )

    assert result.is_healthy is True
    assert "stale_source" not in result.files[0].quality_manifest["quality_warnings"]


def test_stale_source_accepts_xxhash_file_hash(qdrant_client, tmp_path):
    import xxhash
    current = tmp_path / "current.pdf"
    current.write_bytes(b"current")
    qdrant_client.scroll.side_effect = _scroll_once([
        _point({
            "file_path": str(current),
            "chunk_index": 0,
            "chunk_count": 1,
            "source_hash": xxhash.xxh64(b"current").hexdigest(),
            "page_count": 1,
            "text": "x" * 5000,
        }),
    ])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
        )

    assert result.is_healthy is True
    assert "stale_source" not in result.files[0].quality_manifest["quality_warnings"]


def test_stale_source_accepts_normalized_text_hash(qdrant_client, tmp_path):
    from arcaneum.indexing.common.sync import compute_text_file_hash
    current = tmp_path / "current.md"
    current.write_bytes(b"# Title\r\nBody\r\n")
    qdrant_client.scroll.side_effect = _scroll_once([
        _point({
            "file_path": str(current),
            "chunk_index": 0,
            "chunk_count": 1,
            "source_hash": compute_text_file_hash(current),
            "text": "# Title\nBody\n",
        }),
    ])

    with patch.object(verify_mod, "get_collection_type", return_value="markdown"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="markdown",
            total_points=1,
        )

    assert result.is_healthy is True
    assert "stale_source" not in result.files[0].quality_manifest["quality_warnings"]


def test_quality_manifest_preserves_ocr_fallback(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once([
        _point({
            "file_path": "/tmp/ocr.pdf",
            "chunk_index": 0,
            "chunk_count": 1,
            "page_count": 1,
            "text": "x" * 5000,
            "quality_manifest": {
                "schema_version": 1,
                "file_path": "/tmp/ocr.pdf",
                "source_hash": "abc",
                "extractor": "pdf",
                "extractor_version": "arcaneum.quality_manifest.v1",
                "extraction_method": "pymupdf4llm_ocr",
                "fallback_method": None,
                "chunk_count": 1,
                "page_coverage": {
                    "page_count": 1,
                    "covered_pages": [1],
                    "empty_pages": [],
                    "low_text_pages": [],
                },
                "ocr": {
                    "triggered": True,
                    "reason": "quality",
                    "pages_processed": 1,
                    "confidence": 72.0,
                    "failures": 0,
                },
                "quality_warnings": [],
                "repair_command": "arc corpus sync <corpus> /tmp/ocr.pdf --repair",
                "verify_command": "arc corpus verify <corpus> --json",
            },
        }),
    ])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
        )

    manifest = result.files[0].quality_manifest
    assert manifest["ocr"]["triggered"] is True
    assert manifest["ocr"]["reason"] == "quality"
    assert result.files[0].is_complete is True


def test_quality_manifest_marks_garbled_text(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once([
        _point({
            "file_path": "/tmp/garbled.pdf",
            "chunk_index": 0,
            "chunk_count": 1,
            "page_count": 1,
            "text": "\ufffd" * 1000,
        }),
    ])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
            check_quality=True,
        )

    assert result.garbled_items == 1
    assert "garbled_text" in result.files[0].quality_manifest["quality_warnings"]


def test_chunk_count_rejects_sparse_out_of_range_indices(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once([
        _point({
            "file_path": "/tmp/sparse.pdf",
            "chunk_index": 0,
            "chunk_count": 3,
        }),
        _point({
            "file_path": "/tmp/sparse.pdf",
            "chunk_index": 2,
            "chunk_count": 3,
        }),
        _point({
            "file_path": "/tmp/sparse.pdf",
            "chunk_index": 3,
            "chunk_count": 3,
        }),
    ])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=3,
        )

    assert result.is_healthy is False
    assert result.files[0].missing_indices == [1]
    assert result.files[0].is_complete is False


def test_inconsistent_chunk_counts_are_incomplete(qdrant_client, caplog):
    qdrant_client.scroll.side_effect = _scroll_once([
        _point({
            "file_path": "/tmp/inconsistent.pdf",
            "chunk_index": 0,
            "chunk_count": 3,
        }),
        _point({
            "file_path": "/tmp/inconsistent.pdf",
            "chunk_index": 1,
            "chunk_count": 2,
        }),
    ])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=2,
        )

    assert result.is_healthy is False
    assert result.files[0].expected_chunks == 3
    assert result.files[0].missing_indices == [2]
    assert "Inconsistent chunk_count metadata" in caplog.text


def test_explicit_chunk_count_overrides_legacy_inference(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once([
        _point({
            "file_path": "/tmp/mixed.pdf",
            "chunk_index": 0,
            "chunk_count": 3,
        }),
        _point({
            "file_path": "/tmp/mixed.pdf",
            "chunk_index": 1,
            "chunk_count": 3,
        }),
        _point({
            "file_path": "/tmp/mixed.pdf",
            "chunk_index": 2,
            "chunk_count": 3,
        }),
        _point({
            "file_path": "/tmp/mixed.pdf",
            "chunk_index": 3,
        }),
    ])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=4,
        )

    assert result.is_healthy is False
    assert result.files[0].expected_chunks == 3
    assert result.files[0].missing_indices == []
    assert result.files[0].is_complete is False


def test_legacy_missing_chunk_count_logs_warning(qdrant_client, caplog):
    qdrant_client.scroll.side_effect = _scroll_once([
        _point({
            "file_path": "/tmp/legacy.pdf",
            "chunk_index": 0,
        }),
    ])

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
        )

    assert result.is_healthy is True
    assert "Legacy chunk metadata missing chunk_count" in caplog.text


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

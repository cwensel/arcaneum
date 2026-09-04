"""Tests for extraction-dropout detection in CollectionVerifier."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from arcaneum.indexing import verify as verify_mod
from arcaneum.indexing.pdf.chunker import PDFChunker
from arcaneum.indexing.policy import build_indexing_policy
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
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/fake.pdf",
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "page_count": 8,
                    "text": "x" * 500,
                }
            ),
        ]
    )

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


def test_duplicate_pdf_sources_are_repairable_as_one_content_group(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/a.pdf",
                    "source_hash": "same-content",
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "page_count": 1,
                    "text": "A complete page of body text.",
                }
            ),
            _point(
                {
                    "file_path": "/tmp/b.pdf",
                    "source_hash": "same-content",
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "page_count": 1,
                    "text": "A complete page of body text.",
                }
            ),
        ]
    )

    with patch.object(verify_mod, "file_manifests_ready", return_value=False):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=2,
        )

    assert result.duplicate_source_groups == 1
    assert result.is_healthy is False
    assert result.get_items_needing_repair() == ["/tmp/a.pdf", "/tmp/b.pdf"]
    assert all(file.has_duplicate_source for file in result.files)
    assert all(file.duplicate_source_paths == ["/tmp/a.pdf", "/tmp/b.pdf"] for file in result.files)


def test_duplicate_markdown_sources_are_not_flagged_without_alias_repair(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/a.md",
                    "source_hash": "same-content",
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "text": "same text",
                }
            ),
            _point(
                {
                    "file_path": "/tmp/b.md",
                    "source_hash": "same-content",
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "text": "same text",
                }
            ),
        ]
    )

    with patch.object(verify_mod, "file_manifests_ready", return_value=False):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="markdown",
            total_points=2,
        )

    assert result.is_healthy is True
    assert result.duplicate_source_groups == 0
    assert result.get_items_needing_repair() == []


def test_alias_manifest_does_not_look_like_an_incomplete_second_document(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/a.pdf",
                    "source_hash": "same-content",
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "page_count": 1,
                    "text": "A complete page of body text.",
                }
            )
        ]
    )
    manifests = {
        "/tmp/a.pdf": {
            "file_hash": "same-content",
            "chunk_count": 1,
            "canonical_path": "/tmp/a.pdf",
            "store_type": "pdf",
            "indexing_policy": build_indexing_policy("pdf"),
            "quality_manifest": {"quality_warnings": []},
        },
        "/tmp/b.pdf": {
            "file_hash": "same-content",
            "chunk_count": 1,
            "canonical_path": "/tmp/a.pdf",
            "store_type": "pdf",
            "indexing_policy": build_indexing_policy("pdf"),
            "quality_manifest": {"quality_warnings": []},
        },
    }

    with (
        patch.object(verify_mod, "file_manifests_ready", return_value=True),
        patch.object(
            verify_mod.MetadataBasedSync,
            "get_file_manifest_snapshot",
            return_value=manifests,
        ),
    ):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
        )

    assert result.total_items == 1
    assert result.complete_items == 1
    assert result.duplicate_source_groups == 0


def test_stale_policy_manifest_is_unhealthy_and_repairable(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/a.pdf",
                    "source_hash": "content",
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "page_count": 1,
                    "text": "A complete page of body text.",
                }
            )
        ]
    )
    stale_policy = build_indexing_policy("pdf")
    stale_policy["chunking"]["id"] = "pdf-chunking:v2"
    manifests = {
        "/tmp/a.pdf": {
            "file_hash": "content",
            "chunk_count": 1,
            "canonical_path": "/tmp/a.pdf",
            "store_type": "pdf",
            "indexing_policy": stale_policy,
        }
    }

    with (
        patch.object(verify_mod, "file_manifests_ready", return_value=True),
        patch.object(
            verify_mod.MetadataBasedSync,
            "get_file_manifest_snapshot",
            return_value=manifests,
        ),
    ):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
        )

    assert result.is_healthy is False
    assert result.stale_policy_items == 1
    assert result.files[0].stale_policy is True
    assert result.get_items_needing_repair() == ["/tmp/a.pdf"]
    assert "stale_indexing_policy" in result.files[0].quality_manifest["quality_warnings"]


def test_pdf_quality_counts_dropped_and_sub_floor_chunks(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/a.pdf",
                    "source_hash": "content",
                    "chunk_index": 0,
                    "chunk_count": 2,
                    "page_count": 1,
                    "text": "short",
                    "quality_manifest": {
                        "dropped_chunk_count": 2,
                        "quality_warnings": [],
                    },
                }
            ),
            _point(
                {
                    "file_path": "/tmp/a.pdf",
                    "source_hash": "content",
                    "chunk_index": 1,
                    "chunk_count": 2,
                    "page_count": 1,
                    "text": "x" * 500,
                }
            ),
        ]
    )
    manifests = {
        "/tmp/a.pdf": {
            "file_hash": "content",
            "chunk_count": 2,
            "canonical_path": "/tmp/a.pdf",
            "store_type": "pdf",
            "indexing_policy": build_indexing_policy("pdf", {"min_chunk_chars": 200}),
            "quality_manifest": {
                "dropped_chunk_count": 2,
                "quality_warnings": [],
            },
        }
    }

    with (
        patch.object(verify_mod, "file_manifests_ready", return_value=True),
        patch.object(
            verify_mod.MetadataBasedSync,
            "get_file_manifest_snapshot",
            return_value=manifests,
        ),
    ):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=2,
            check_quality=True,
        )

    assert result.is_healthy is False
    assert result.dropped_chunks == 2
    assert result.sub_floor_chunks == 1
    assert result.files[0].dropped_chunk_count == 2
    assert result.files[0].sub_floor_chunk_count == 1


def test_authoritative_pdf_without_quality_manifest_reports_gap(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/a.pdf",
                    "source_hash": "content",
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "page_count": 1,
                    "text": "x" * 500,
                }
            )
        ]
    )
    manifests = {
        "/tmp/a.pdf": {
            "file_hash": "content",
            "chunk_count": 1,
            "canonical_path": "/tmp/a.pdf",
            "store_type": "pdf",
            "indexing_policy": build_indexing_policy("pdf"),
        }
    }

    with (
        patch.object(verify_mod, "file_manifests_ready", return_value=True),
        patch.object(
            verify_mod.MetadataBasedSync,
            "get_file_manifest_snapshot",
            return_value=manifests,
        ),
    ):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
            check_quality=True,
        )

    assert result.is_healthy is False
    assert result.quality_manifest_gaps == 1
    assert result.files[0].quality_manifest_missing is True
    assert result.get_items_needing_repair() == ["/tmp/a.pdf"]


def test_short_single_chunk_section_is_not_a_sub_floor_defect(qdrant_client):
    text = (
        "Introduction\n"
        + ("Detailed body evidence and analysis. " * 20)
        + "\nAcknowledgements\nThanks to the team."
    )
    chunks = PDFChunker(
        {
            "chunk_size": 120,
            "chunk_overlap": 0,
            "char_to_token_ratio": 1,
            "min_chunk_chars": 200,
        },
        overlap_percent=0,
        late_chunking_enabled=False,
    ).chunk(text, {})
    assert any(
        chunk.metadata["section_type"] == "acknowledgements" and len(chunk.text) < 200
        for chunk in chunks
    )
    points = [
        _point(
            {
                "file_path": "/tmp/paper.pdf",
                "chunk_index": chunk.chunk_index,
                "chunk_count": len(chunks),
                "text": chunk.text,
                "section_type": chunk.metadata["section_type"],
                "section_title": chunk.metadata["section_title"],
                "section_offset": chunk.metadata["section_offset"],
                "chunk_start_char": chunk.metadata["chunk_start_char"],
            }
        )
        for chunk in chunks
    ]
    qdrant_client.scroll.side_effect = _scroll_once(points)

    with patch.object(verify_mod, "file_manifests_ready", return_value=False):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=len(points),
            check_quality=True,
        )

    assert result.sub_floor_chunks == 0
    assert all(file.sub_floor_chunk_count == 0 for file in result.files)


def test_repeated_short_headings_are_distinct_section_occurrences(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/paper.pdf",
                    "chunk_index": index,
                    "chunk_count": 2,
                    "text": f"1. Introduction\nShort section {index}.",
                    "section_type": "body",
                    "section_title": "1. Introduction",
                    "section_offset": index * 100,
                    "chunk_start_char": index * 100,
                }
            )
            for index in range(2)
        ]
    )

    with patch.object(verify_mod, "file_manifests_ready", return_value=False):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=2,
            check_quality=True,
        )

    assert result.sub_floor_chunks == 0


def test_body_text_starting_with_title_does_not_hide_short_fragment(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/paper.pdf",
                    "chunk_index": 0,
                    "chunk_count": 2,
                    "text": "Results\n" + "A" * 300,
                    "section_type": "body",
                    "section_title": "Results",
                    "section_offset": 25,
                    "chunk_start_char": 25,
                }
            ),
            _point(
                {
                    "file_path": "/tmp/paper.pdf",
                    "chunk_index": 1,
                    "chunk_count": 2,
                    "text": "Results remain preliminary.",
                    "section_type": "body",
                    "section_title": "Results",
                    "section_offset": 25,
                    "chunk_start_char": 330,
                }
            ),
        ]
    )

    with patch.object(verify_mod, "file_manifests_ready", return_value=False):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=2,
            check_quality=True,
        )

    assert result.sub_floor_chunks == 1


def test_legacy_section_chunks_without_offsets_still_detect_short_fragment(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/legacy.pdf",
                    "chunk_index": 0,
                    "chunk_count": 2,
                    "text": "A" * 300,
                    "section_type": "body",
                    "section_title": "Results",
                    "chunk_start_char": 25,
                }
            ),
            _point(
                {
                    "file_path": "/tmp/legacy.pdf",
                    "chunk_index": 1,
                    "chunk_count": 2,
                    "text": "Short legacy fragment.",
                    "section_type": "body",
                    "section_title": "Results",
                    "chunk_start_char": 330,
                }
            ),
        ]
    )

    with patch.object(verify_mod, "file_manifests_ready", return_value=False):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=2,
            check_quality=True,
        )

    assert result.sub_floor_chunks == 1


def test_standard_pdf_verification_does_not_read_source_files(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/a.pdf",
                    "source_hash": "content",
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "text": "x" * 500,
                }
            )
        ]
    )

    with (
        patch.object(verify_mod, "_source_hash_matches_disk") as source_check,
        patch.object(verify_mod, "_page_count_from_disk") as page_check,
    ):
        CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
            check_quality=True,
        )

    source_check.assert_not_called()
    page_check.assert_not_called()


def test_verify_collection_scores_pdf_quality_by_default(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/garbled.pdf",
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "page_count": 1,
                    "text": "\ufffd" * 300,
                }
            )
        ]
    )

    with (
        patch.object(
            verify_mod,
            "get_collection_metadata",
            return_value={"collection_type": "pdf", "schema_version": 1},
        ),
        patch.object(verify_mod, "persisted_schema_issues", return_value=[]),
        patch.object(verify_mod, "user_point_count", return_value=1),
        patch.object(verify_mod, "file_manifests_ready", return_value=False),
    ):
        result = CollectionVerifier(qdrant_client).verify_collection("Dummy")

    assert result.garbled_items == 1
    assert result.is_healthy is False
    assert "text" in qdrant_client.scroll.call_args.kwargs["with_payload"]


def test_verify_maps_legacy_model_alias_before_policy_comparison(qdrant_client):
    from arcaneum.config import DEFAULT_MODELS

    verifier = CollectionVerifier(qdrant_client)
    verifier._verify_file_collection = MagicMock(
        return_value=SimpleNamespace(errors=[], is_healthy=True)
    )

    with (
        patch.object(
            verify_mod,
            "get_collection_metadata",
            return_value={
                "collection_type": "pdf",
                "model": "BAAI/bge-large-en-v1.5",
                "schema_version": 1,
            },
        ),
        patch.object(verify_mod, "persisted_schema_issues", return_value=[]),
        patch.object(verify_mod, "user_point_count", return_value=0),
    ):
        verifier.verify_collection("Dummy")

    active_policy = verifier._verify_file_collection.call_args.kwargs["active_policy"]
    assert (
        active_policy["chunking"]["config"]
        == build_indexing_policy("pdf", DEFAULT_MODELS["bge-large"].__dict__)["chunking"]["config"]
    )


def test_extraction_floor_skips_repair(qdrant_client):
    # Same dropout signal, but already marked extraction_floor → don't re-index
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/floor.pdf",
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "page_count": 8,
                    "text": "x" * 500,
                    "extraction_floor": True,
                }
            ),
        ]
    )

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
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/good.pdf",
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "page_count": 8,
                    "text": "x" * 39000,
                }
            ),
        ]
    )

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
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/incomplete.pdf",
                    "chunk_index": 0,
                    "chunk_count": 3,
                }
            ),
            _point(
                {
                    "file_path": "/tmp/incomplete.pdf",
                    "chunk_index": 1,
                    "chunk_count": 3,
                }
            ),
        ]
    )

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
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": str(stale),
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "source_hash": "0" * 64,
                    "page_count": 1,
                    "text": "x" * 5000,
                }
            ),
        ]
    )

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
            deep=True,
        )

    assert result.is_healthy is False
    assert result.files[0].is_complete is False
    assert "stale_source" in result.files[0].quality_manifest["quality_warnings"]


def test_stale_source_accepts_sync_short_hash(qdrant_client, tmp_path):
    current = tmp_path / "current.pdf"
    current.write_bytes(b"current")
    import hashlib

    short_hash = hashlib.sha256(b"current").hexdigest()[:16]
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": str(current),
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "source_hash": short_hash,
                    "page_count": 1,
                    "text": "x" * 5000,
                }
            ),
        ]
    )

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
            deep=True,
        )

    assert result.is_healthy is True
    assert "stale_source" not in result.files[0].quality_manifest["quality_warnings"]


def test_stale_source_accepts_xxhash_file_hash(qdrant_client, tmp_path):
    import xxhash

    current = tmp_path / "current.pdf"
    current.write_bytes(b"current")
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": str(current),
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "source_hash": xxhash.xxh64(b"current").hexdigest(),
                    "page_count": 1,
                    "text": "x" * 5000,
                }
            ),
        ]
    )

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
            deep=True,
        )

    assert result.is_healthy is True
    assert "stale_source" not in result.files[0].quality_manifest["quality_warnings"]


def test_stale_source_accepts_normalized_text_hash(qdrant_client, tmp_path):
    from arcaneum.indexing.common.sync import compute_text_file_hash

    current = tmp_path / "current.md"
    current.write_bytes(b"# Title\r\nBody\r\n")
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": str(current),
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "source_hash": compute_text_file_hash(current),
                    "text": "# Title\nBody\n",
                }
            ),
        ]
    )

    with patch.object(verify_mod, "get_collection_type", return_value="markdown"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="markdown",
            total_points=1,
            deep=True,
        )

    assert result.is_healthy is True
    assert "stale_source" not in result.files[0].quality_manifest["quality_warnings"]


def test_quality_manifest_preserves_ocr_fallback(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
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
                }
            ),
        ]
    )

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
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/garbled.pdf",
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "page_count": 1,
                    "text": "\ufffd" * 1000,
                }
            ),
        ]
    )

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
            check_quality=True,
        )

    assert result.garbled_items == 1
    assert "garbled_text" in result.files[0].quality_manifest["quality_warnings"]


def test_all_dropped_file_manifest_is_exposed_by_verify_json_model(qdrant_client):
    manifest = {
        "schema_version": 1,
        "file_path": "/tmp/all-garbage.pdf",
        "source_hash": None,
        "extractor": "pdf",
        "extractor_version": "arcaneum.quality_manifest.v1",
        "extraction_method": "pymupdf4llm_markdown",
        "fallback_method": None,
        "chunk_count": 0,
        "page_coverage": {
            "page_count": 1,
            "covered_pages": [1],
            "empty_pages": [],
            "low_text_pages": [],
        },
        "ocr": {
            "triggered": False,
            "reason": None,
            "pages_processed": None,
            "confidence": None,
            "failures": None,
        },
        "dropped_chunk_count": 2,
        "dropped_chunk_reason": "replacement_character_ratio_gt_0.05",
        "quality_warnings": ["replacement_heavy_chunks_dropped"],
        "repair_command": "arc corpus sync <corpus> /tmp/all-garbage.pdf --repair",
        "verify_command": "arc corpus verify <corpus> --json",
    }
    qdrant_client.scroll.return_value = ([], None)

    with (
        patch.object(verify_mod, "file_manifests_ready", return_value=True),
        patch.object(
            verify_mod.MetadataBasedSync,
            "get_file_manifest_snapshot",
            return_value={
                "/tmp/all-garbage.pdf": {
                    "file_path": "/tmp/all-garbage.pdf",
                    "chunk_count": 0,
                    "quality_manifest": manifest,
                }
            },
        ),
    ):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=0,
        )

    assert result.total_items == 1
    assert result.files[0].actual_chunks == 0
    assert result.files[0].quality_manifest == manifest
    assert result.files[0].has_omitted_text is True
    assert result.files[0].fidelity_degraded is True
    assert result.files[0].recovery_exhausted is False
    assert result.files[0].repair_recommended is True
    assert result.files[0].is_complete is False
    assert result.is_healthy is False
    assert result.get_items_needing_repair() == ["/tmp/all-garbage.pdf"]


def test_current_partial_replacement_omissions_are_observable_but_healthy(
    qdrant_client,
):
    file_path = "/tmp/normalized.pdf"
    manifest = {
        "chunk_count": 1,
        "replacement_omissions": {
            "source_character_count": 100,
            "retained_character_count": 95,
            "character_count": 5,
            "replacement_ratio": 0.05,
            "degraded": False,
            "singleton_count": 5,
            "multi_character_run_count": 0,
            "multi_character_run_character_count": 0,
            "hard_boundary_count": 0,
            "hard_boundary_character_count": 0,
            "normalized_run_count": 0,
            "normalized_run_character_count": 0,
        },
        "quality_warnings": ["replacement_characters_omitted"],
    }
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": file_path,
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "quality_manifest": manifest,
                }
            )
        ]
    )

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
        )

    file_result = result.files[0]
    assert file_result.expected_chunks == 1
    assert file_result.actual_chunks == 1
    assert file_result.missing_indices == []
    assert file_result.has_omitted_text is True
    assert file_result.fidelity_degraded is False
    assert file_result.recovery_exhausted is False
    assert file_result.repair_recommended is False
    assert file_result.is_complete is True
    assert result.is_healthy is True
    assert result.get_items_needing_repair() == []


def test_over_five_percent_replacement_loss_is_degraded_and_repairable(qdrant_client):
    file_path = "/tmp/degraded.pdf"
    manifest = {
        "chunk_count": 1,
        "dropped_chunk_count": 0,
        "dropped_chunk_reason": None,
        "replacement_omissions": {
            "source_character_count": 100,
            "retained_character_count": 94,
            "character_count": 6,
            "replacement_ratio": 0.06,
            "degraded": True,
        },
        "ocr": {"triggered": False},
        "quality_warnings": ["replacement_characters_omitted"],
    }
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": file_path,
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "quality_manifest": manifest,
                }
            )
        ]
    )

    with patch.object(verify_mod, "get_collection_type", return_value="pdf"):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
        )

    file_result = result.files[0]
    assert file_result.has_omitted_text is True
    assert file_result.fidelity_degraded is True
    assert file_result.recovery_exhausted is False
    assert file_result.repair_recommended is True
    assert file_result.is_complete is False
    assert "replacement_fidelity_degraded" in file_result.quality_manifest["quality_warnings"]
    assert result.is_healthy is False
    assert result.get_items_needing_repair() == [file_path]


def test_ocr_exhausted_degradation_is_unhealthy_but_not_requeued(qdrant_client):
    file_path = "/tmp/ocr-exhausted.pdf"
    manifest = {
        "chunk_count": 1,
        "dropped_chunk_count": 0,
        "dropped_chunk_reason": None,
        "replacement_omissions": {
            "source_character_count": 100,
            "retained_character_count": 10,
            "character_count": 90,
            "replacement_ratio": 0.9,
            "degraded": True,
        },
        "ocr": {"triggered": True, "reason": "garbled"},
        "quality_warnings": ["replacement_characters_omitted"],
    }
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": file_path,
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "text": "still unreadable after OCR",
                    "quality_manifest": manifest,
                }
            )
        ]
    )

    with (
        patch.object(verify_mod, "get_collection_type", return_value="pdf"),
        patch("arcaneum.indexing.pdf.quality.score_text", return_value=0.1),
    ):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
            check_quality=True,
        )

    file_result = result.files[0]
    assert file_result.fidelity_degraded is True
    assert file_result.recovery_exhausted is True
    assert file_result.has_garbled_text is True
    assert file_result.repair_recommended is False
    assert file_result.is_complete is False
    assert "replacement_recovery_exhausted" in file_result.quality_manifest["quality_warnings"]
    assert result.is_healthy is False
    assert result.get_items_needing_repair() == []


def test_all_omitted_current_manifest_keeps_zero_chunk_structure_coherent(qdrant_client):
    file_path = "/tmp/all-omitted.pdf"
    manifest = {
        "chunk_count": 0,
        "replacement_omissions": {
            "source_character_count": 30,
            "retained_character_count": 0,
            "character_count": 30,
            "replacement_ratio": 1.0,
            "degraded": True,
            "singleton_count": 0,
            "multi_character_run_count": 1,
            "multi_character_run_character_count": 30,
            "hard_boundary_count": 0,
            "hard_boundary_character_count": 0,
            "normalized_run_count": 1,
            "normalized_run_character_count": 30,
        },
        "quality_warnings": ["replacement_characters_omitted"],
    }
    qdrant_client.scroll.return_value = ([], None)

    with (
        patch.object(verify_mod, "file_manifests_ready", return_value=True),
        patch.object(
            verify_mod.MetadataBasedSync,
            "get_file_manifest_snapshot",
            return_value={
                file_path: {
                    "file_path": file_path,
                    "chunk_count": 0,
                    "quality_manifest": manifest,
                }
            },
        ),
    ):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=0,
        )

    file_result = result.files[0]
    assert file_result.expected_chunks == 0
    assert file_result.actual_chunks == 0
    assert file_result.missing_indices == []
    assert file_result.completion_percentage == 100.0
    assert file_result.has_omitted_text is True
    assert file_result.fidelity_degraded is True
    assert file_result.recovery_exhausted is False
    assert file_result.repair_recommended is True
    assert file_result.is_complete is False
    assert result.is_healthy is False
    assert result.get_items_needing_repair() == [file_path]


def test_chunk_count_rejects_sparse_out_of_range_indices(qdrant_client):
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/sparse.pdf",
                    "chunk_index": 0,
                    "chunk_count": 3,
                }
            ),
            _point(
                {
                    "file_path": "/tmp/sparse.pdf",
                    "chunk_index": 2,
                    "chunk_count": 3,
                }
            ),
            _point(
                {
                    "file_path": "/tmp/sparse.pdf",
                    "chunk_index": 3,
                    "chunk_count": 3,
                }
            ),
        ]
    )

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
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/inconsistent.pdf",
                    "chunk_index": 0,
                    "chunk_count": 3,
                }
            ),
            _point(
                {
                    "file_path": "/tmp/inconsistent.pdf",
                    "chunk_index": 1,
                    "chunk_count": 2,
                }
            ),
        ]
    )

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
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/mixed.pdf",
                    "chunk_index": 0,
                    "chunk_count": 3,
                }
            ),
            _point(
                {
                    "file_path": "/tmp/mixed.pdf",
                    "chunk_index": 1,
                    "chunk_count": 3,
                }
            ),
            _point(
                {
                    "file_path": "/tmp/mixed.pdf",
                    "chunk_index": 2,
                    "chunk_count": 3,
                }
            ),
            _point(
                {
                    "file_path": "/tmp/mixed.pdf",
                    "chunk_index": 3,
                }
            ),
        ]
    )

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
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/legacy.pdf",
                    "chunk_index": 0,
                }
            ),
        ]
    )

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
    qdrant_client.scroll.side_effect = _scroll_once(
        [
            _point(
                {
                    "file_path": "/tmp/legacy.pdf",
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "text": "x" * 500,
                }
            ),
        ]
    )

    with (
        patch.object(verify_mod, "get_collection_type", return_value="pdf"),
        patch.object(verify_mod, "_page_count_from_disk", return_value=8),
    ):
        result = CollectionVerifier(qdrant_client)._verify_file_collection(
            collection_name="Dummy",
            collection_type="pdf",
            total_points=1,
            deep=True,
        )

    assert result.dropout_items == 1
    assert result.files[0].page_count == 8

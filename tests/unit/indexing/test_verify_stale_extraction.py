"""Verify flags PDFs extracted by the normalized extractor before the column fix.

Before fallback_reason existed, normalized extraction sorted text by vertical
position and merged side-by-side columns. Those manifests are recognizable:
they name pymupdf_normalized and lack the fallback_reason key every newer
manifest records.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from arcaneum.indexing.index_selector import select_indexed_paths
from arcaneum.indexing.verify import STALE_EXTRACTION_SELECTOR, CollectionVerifier


def _payload(path, method, chunk_index=0, chunk_count=1, **manifest):
    return {
        "file_path": path,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "page_count": 1,
        "text": "x" * 5000,
        "quality_manifest": {
            "schema_version": 1,
            "file_path": path,
            "extraction_method": method,
            "fallback_method": None,
            "chunk_count": chunk_count,
            "quality_warnings": [],
            **manifest,
        },
    }


def _verify(payloads):
    client = MagicMock()
    client.get_collection.return_value = SimpleNamespace(points_count=len(payloads))
    points = [SimpleNamespace(id=i, payload=p, vector=None) for i, p in enumerate(payloads)]
    client.scroll.side_effect = [(points, None)]
    return CollectionVerifier(client)._verify_file_collection(
        collection_name="Papers",
        collection_type="pdf",
        total_points=len(payloads),
    )


def test_pre_fix_normalized_extraction_is_unhealthy_and_repairable():
    result = _verify([_payload("/p/old.pdf", "pymupdf_normalized")])

    file = result.files[0]
    assert result.is_healthy is False
    assert result.stale_extraction_items == 1
    assert file.stale_extraction is True
    assert file.stale_extraction_only is True
    assert "stale_extraction" in file.quality_manifest["quality_warnings"]
    assert result.get_items_needing_repair(include_stale_policy=False) == ["/p/old.pdf"]


@pytest.mark.parametrize(
    "method",
    ["pymupdf4llm_markdown+pymupdf_normalized", "pymupdf_normalized+pymupdf_normalized"],
)
def test_combined_methods_with_a_normalized_part_are_stale(method):
    result = _verify([_payload("/p/merged.pdf", method)])

    assert result.files[0].stale_extraction is True


def test_normalized_extraction_recorded_after_the_fix_is_healthy():
    result = _verify(
        [_payload("/p/new.pdf", "pymupdf_normalized", fallback_reason="LayoutWorkerTimeout")]
    )

    assert result.is_healthy is True
    assert result.stale_extraction_items == 0
    assert result.files[0].stale_extraction is False


def test_markdown_extraction_without_fallback_reason_is_healthy():
    result = _verify([_payload("/p/clean.pdf", "pymupdf4llm_markdown")])

    assert result.is_healthy is True
    assert result.files[0].stale_extraction is False


def test_stale_extraction_with_another_defect_is_not_extraction_only():
    # Chunk 1 of 2 is missing, so the file is also incomplete.
    result = _verify([_payload("/p/broken.pdf", "pymupdf_normalized", chunk_count=2)])

    file = result.files[0]
    assert file.stale_extraction is True
    assert file.stale_extraction_only is False


def test_selector_selects_exactly_what_verify_flags():
    payloads = [
        _payload("/p/old.pdf", "pymupdf_normalized"),
        _payload("/p/merged.pdf", "pymupdf4llm_markdown+pymupdf_normalized"),
        _payload("/p/new.pdf", "pymupdf_normalized", fallback_reason="LayoutWorkerCrashed"),
        _payload("/p/clean.pdf", "pymupdf4llm_markdown", fallback_reason=None),
        _payload("/p/legacy-clean.pdf", "pymupdf4llm_markdown"),
    ]
    flagged = sorted(f.file_path for f in _verify(payloads).files if f.stale_extraction)

    client = MagicMock()
    client.scroll.return_value = ([SimpleNamespace(payload=p) for p in payloads], None)

    assert select_indexed_paths(client, "Papers", STALE_EXTRACTION_SELECTOR) == flagged
    assert flagged == ["/p/merged.pdf", "/p/old.pdf"]

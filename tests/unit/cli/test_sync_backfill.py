"""Regression tests for sync backfill helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from arcaneum.cli.sync import _fetch_chunks_for_files_bulk


def test_fetch_chunks_for_files_bulk_preserves_pdf_ocr_metadata():
    qdrant = MagicMock()
    qdrant.scroll.return_value = (
        [
            SimpleNamespace(
                id="point-1",
                payload={
                    "text": "Recovered OCR text",
                    "file_path": "/docs/report.pdf",
                    "filename": "report.pdf",
                    "file_extension": ".pdf",
                    "chunk_index": 3,
                    "document_type": "pdf",
                    "page_number": 2,
                    "ocr_confidence": 64.5,
                    "ocr_language": "eng",
                    "ocr_pages_processed": 4,
                    "ocr_pages_failed": 1,
                    "ocr_low_confidence_word_count": 7,
                    "ocr_merge_strategy": "append_missing_pages",
                    "ocr_triggered_by": "quality_gate",
                },
            )
        ],
        None,
    )

    chunks_by_file, error = _fetch_chunks_for_files_bulk(
        qdrant,
        "test-corpus",
        {"/docs/report.pdf"},
        verbose=False,
        output_json=True,
        console=MagicMock(),
    )

    assert error is None
    doc = chunks_by_file["/docs/report.pdf"][0]
    assert doc["ocr_confidence"] == 64.5
    assert doc["ocr_language"] == "eng"
    assert doc["ocr_pages_processed"] == 4
    assert doc["ocr_pages_failed"] == 1
    assert doc["ocr_low_confidence_word_count"] == 7
    assert doc["ocr_merge_strategy"] == "append_missing_pages"
    assert doc["ocr_triggered_by"] == "quality_gate"


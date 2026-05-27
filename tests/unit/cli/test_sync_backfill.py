"""Regression tests for sync backfill helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from arcaneum.cli.sync import _backfill_meili_to_qdrant, _fetch_chunks_for_files_bulk


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
                    "quality_manifest": {
                        "schema_version": 1,
                        "source_hash": "abc123",
                        "quality_warnings": ["low_text_pages"],
                    },
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
    assert doc["quality_manifest"]["source_hash"] == "abc123"


def test_backfill_meili_to_qdrant_builds_code_quality_manifest(tmp_path):
    source = tmp_path / "example.py"
    source.write_text("def hello():\n    return 'world'\n")
    qdrant = MagicMock()
    embedding_client = MagicMock()
    embedding_client.embed.return_value = [[0.1, 0.2]]
    progress = MagicMock()

    files_success, chunks_success, files_failed, skipped = _backfill_meili_to_qdrant(
        qdrant=qdrant,
        embedding_client=embedding_client,
        corpus="code-corpus",
        corpus_type="code",
        model_list=["test-model"],
        model_config={"chunk_size": 8000, "chunk_overlap": 20},
        file_paths=[str(source)],
        verbose=False,
        output_json=True,
        progress=progress,
        backfill_task=1,
        text_workers=1,
    )

    assert files_success == 1
    assert chunks_success >= 1
    assert files_failed == 0
    assert skipped == []
    points = qdrant.upsert.call_args.kwargs["points"]
    manifest = points[0].payload["quality_manifest"]
    assert manifest["extractor"] == "code"
    assert manifest["source_hash"] == points[0].payload["source_hash"]

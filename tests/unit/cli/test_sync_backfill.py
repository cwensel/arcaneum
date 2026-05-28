"""Regression tests for sync backfill helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from arcaneum.cli.sync import (
    _backfill_meili_to_qdrant,
    _fetch_chunks_for_files_bulk,
    _repair_meili_metadata,
)


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


def test_repair_meili_version_identifier_stamps_persisted_schema():
    index = MagicMock()
    index.get_documents.side_effect = [
        {
            "results": [
                {
                    "id": "doc-1",
                    "file_path": "/repo/example.py",
                    "chunk_index": 0,
                    "git_project_identifier": "proj#main",
                    "git_project_name": "proj",
                    "git_branch": "main",
                    "git_commit_hash": "abcdef123456",
                }
            ]
        },
        {"results": []},
    ]
    index.update_documents.return_value = SimpleNamespace(task_uid=123)
    meili = MagicMock()
    meili.get_index.return_value = index

    updated, failed = _repair_meili_metadata(
        qdrant=MagicMock(),
        meili=meili,
        corpus="code-corpus",
        output_json=True,
        console=MagicMock(),
    )

    assert updated == 1
    assert failed == 0
    update_doc = index.update_documents.call_args.args[0][0]
    assert update_doc["schema_version"] == 1
    assert update_doc["app_version"]
    assert update_doc["git_version_identifier"] == "proj#main@abcdef1"

"""Tests for PDF full-text indexing CLI JSON output."""

import json
from unittest.mock import MagicMock

from arcaneum.cli import index_text
from arcaneum.indexing.fulltext import pdf_indexer


def test_index_text_pdf_json_includes_ocr_stats(monkeypatch, tmp_path, capsys):
    meili_client = MagicMock()
    meili_client.health_check.return_value = True
    meili_client.index_exists.return_value = True
    meili_client.get_index_settings.return_value = {
        "filterableAttributes": ["file_path", "file_hash"],
    }

    class FakePDFIndexer:
        def __init__(self, **kwargs):
            pass

        def index_directory(self, **kwargs):
            return {
                "total_pdfs": 1,
                "indexed_pdfs": 1,
                "skipped_pdfs": 0,
                "failed_pdfs": 0,
                "total_pages": 2,
                "ocr_pages_processed": 2,
                "ocr_pages_failed": 1,
                "ocr_confidence": 63.5,
                "errors": [],
            }

    monkeypatch.setattr(index_text, "set_process_priority", lambda priority: None)
    monkeypatch.setattr(index_text, "setup_logging_default", lambda: None)
    monkeypatch.setattr(index_text.interaction_logger, "start", lambda *args, **kwargs: None)
    monkeypatch.setattr(index_text.interaction_logger, "finish", lambda *args, **kwargs: None)
    monkeypatch.setattr(index_text, "get_meili_client", lambda: meili_client)
    monkeypatch.setattr(pdf_indexer, "PDFFullTextIndexer", FakePDFIndexer)

    index_text.index_text_pdf_command(
        path=str(tmp_path),
        from_file=None,
        index_name="pdfs",
        recursive=False,
        ocr_enabled=True,
        ocr_language="eng",
        ocr_workers=1,
        normalize_only=False,
        batch_size=10,
        force=False,
        process_priority="normal",
        verbose=False,
        debug=False,
        output_json=True,
    )

    output = json.loads(capsys.readouterr().out)

    assert output["stats"]["ocr_pages_processed"] == 2
    assert output["stats"]["ocr_pages_failed"] == 1
    assert output["stats"]["ocr_confidence"] == 63.5

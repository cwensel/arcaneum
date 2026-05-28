"""Unit tests for page-preserving OCR metadata."""

from io import BytesIO
from unittest.mock import MagicMock

from PIL import Image

from arcaneum.indexing.common.sync import MetadataBasedSync
from arcaneum.indexing.pdf.ocr import (
    OCREngine,
    _ocr_single_page_worker,
    merge_extracted_text_with_ocr,
)
from arcaneum.indexing.uploader import PDFBatchUploader


def _png_bytes() -> bytes:
    image = Image.new("RGB", (10, 10), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_ocr_worker_keeps_low_confidence_words_and_lines(monkeypatch):
    def fake_image_to_data(*args, **kwargs):
        return {
            "conf": [-1, 94.0, 31.0, 88.0],
            "text": ["", "Clean", "smudged", "Next"],
            "block_num": [0, 1, 1, 1],
            "par_num": [0, 1, 1, 1],
            "line_num": [0, 1, 1, 2],
            "word_num": [0, 1, 2, 1],
        }

    monkeypatch.setattr("arcaneum.indexing.pdf.ocr.pytesseract.image_to_data", fake_image_to_data)

    page_num, text, confidence, page_meta = _ocr_single_page_worker(
        _png_bytes(),
        page_num=7,
        language="eng",
        confidence_threshold=60.0,
        image_scale=1.0,
    )

    assert page_num == 7
    assert text == "Clean smudged\nNext"
    assert round(confidence, 2) == 71.0
    assert page_meta["page_number"] == 7
    assert page_meta["word_count"] == 3
    assert page_meta["low_confidence_word_count"] == 1
    assert page_meta["low_confidence_words"] == [{"text": "smudged", "confidence": 31.0}]


def test_process_pdf_reports_page_boundaries_and_ocr_page_stats(monkeypatch, tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr("arcaneum.indexing.pdf.ocr.pdfinfo_from_path", lambda path: {"Pages": 2})

    engine = OCREngine(ocr_workers=1, page_batch_size=1)

    def fake_process_batch(path, first_page, last_page, total_pages, verbose):
        if first_page == 1:
            return {
                1: (
                    "Page one low",
                    80.0,
                    {
                        "page_number": 1,
                        "confidence": 80.0,
                        "word_count": 3,
                        "low_confidence_word_count": 1,
                        "low_confidence_words": [{"text": "low", "confidence": 42.0}],
                        "failed": False,
                    },
                )
            }
        return {
            2: (
                "Page two\nkeeps lines",
                64.0,
                {
                    "page_number": 2,
                    "confidence": 64.0,
                    "word_count": 4,
                    "low_confidence_word_count": 0,
                    "low_confidence_words": [],
                    "failed": False,
                },
            )
        }

    monkeypatch.setattr(engine, "_process_batch", fake_process_batch)

    text, metadata = engine.process_pdf(pdf_path)

    assert text == "Page one low\nPage two\nkeeps lines"
    assert metadata["ocr_pages_processed"] == 2
    assert metadata["ocr_pages_failed"] == 0
    assert metadata["ocr_confidence"] == 72.0
    assert metadata["ocr_low_confidence_word_count"] == 1
    assert metadata["page_boundaries"] == [
        {"page_number": 1, "start_char": 0, "page_text_length": 12},
        {"page_number": 2, "start_char": 13, "page_text_length": 20},
    ]
    assert metadata["ocr_pages"][0]["low_confidence_words"][0]["text"] == "low"


def test_merge_extracted_text_with_ocr_preserves_original_markdown_and_offsets_boundaries():
    extracted_text = "# Table\n\n| A | B |\n| - | - |"
    extracted_metadata = {
        "extraction_method": "pymupdf4llm_markdown",
        "page_count": 1,
        "page_boundaries": [
            {"page_number": 1, "start_char": 0, "page_text_length": len(extracted_text)}
        ],
    }
    ocr_text = "faint low confidence words"
    ocr_metadata = {
        "extraction_method": "ocr_tesseract",
        "ocr_confidence": 51.0,
        "ocr_pages_processed": 1,
        "ocr_pages_failed": 0,
        "page_boundaries": [{"page_number": 1, "start_char": 0, "page_text_length": len(ocr_text)}],
    }

    merged_text, merged_metadata = merge_extracted_text_with_ocr(
        extracted_text,
        extracted_metadata,
        ocr_text,
        ocr_metadata,
    )

    assert merged_text.startswith(extracted_text)
    assert "faint low confidence words" in merged_text
    assert merged_metadata["original_extraction_method"] == "pymupdf4llm_markdown"
    assert merged_metadata["extraction_method"] == "pymupdf4llm_markdown+ocr_tesseract"
    assert merged_metadata["ocr_merge_strategy"] == "append_ocr_to_extracted_text"
    assert merged_metadata["page_boundaries"][0] == extracted_metadata["page_boundaries"][0]
    assert merged_metadata["page_boundaries"][1]["page_number"] == 1
    assert merged_metadata["page_boundaries"][1]["start_char"] > len(extracted_text)


def test_pdf_batch_uploader_duplicate_path_preserves_return_contract(tmp_path):
    pdf_path = tmp_path / "duplicate.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    uploader = object.__new__(PDFBatchUploader)
    uploader.sync = MagicMock()
    uploader.sync.find_file_by_content_hash.return_value = [str(tmp_path / "original.pdf")]
    uploader.sync.filter_existing_paths.return_value = [str(tmp_path / "original.pdf")]
    uploader.sync.add_alternate_path.return_value = 1

    result = uploader._process_single_pdf(
        pdf_path=pdf_path,
        collection_name="docs",
        model_name="test-model",
        chunker=MagicMock(),
        point_id_start=1,
        verbose=False,
        pdf_idx=1,
        total_pdfs=1,
    )

    assert result == (
        [],
        0,
        None,
        {
            "ocr_pages_processed": 0,
            "ocr_pages_failed": 0,
            "ocr_confidence": None,
        },
    )


def test_pdf_force_delete_failure_counts_file_error(tmp_path):
    pdf_path = tmp_path / "broken-delete.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class FailingDeleteQdrant:
        def get_collection(self, _name):
            return MagicMock(points_count=0)

        def scroll(self, **kwargs):
            raise RuntimeError("delete unavailable")

    uploader = object.__new__(PDFBatchUploader)
    uploader.qdrant = FailingDeleteQdrant()
    uploader.sync = MetadataBasedSync(uploader.qdrant)
    uploader.file_workers = 1
    uploader.pdf_timeout = 1
    uploader.embedding_batch_size = 128

    stats = uploader.index_directory(
        pdf_dir=pdf_path.parent,
        collection_name="docs",
        model_name="stella",
        model_config={"chunk_size": 512, "chunk_overlap": 50},
        force_reindex=True,
        file_list=[pdf_path],
    )

    assert stats["files"] == 0
    assert stats["chunks"] == 0
    assert stats["errors"] == 1

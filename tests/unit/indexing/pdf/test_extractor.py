"""Unit tests for PDF extraction metadata and worker fallback."""

import pymupdf
import pytest

from arcaneum.indexing.pdf.extractor import PDFExtractor
from arcaneum.indexing.pdf.layout_worker import (
    LayoutConversionError,
    LayoutWorkerContainmentError,
    LayoutWorkerCrashed,
)


class _FakeDoc:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __len__(self):
        return 1


class _FakeWorker:
    def __init__(self, pages=None, *, error=None, page_count=1):
        self.pages = pages or [{"text": "Worker text", "metadata": {"page_number": 1}}]
        self.error = error
        self.page_count = page_count
        self.requests = []
        self.closed = False

    def convert(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return {"pages": self.pages, "page_count": self.page_count, "worker_pid": 123}

    def close(self):
        self.closed = True


def test_markdown_extraction_marks_auto_ocr_method(monkeypatch, tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr("arcaneum.indexing.pdf.extractor.pymupdf.open", lambda path: _FakeDoc())
    worker = _FakeWorker(pages=[{"text": "OCR text", "metadata": {"page_number": 1}}])

    _, metadata = PDFExtractor(use_ocr=True, layout_worker=worker).extract(pdf_path)

    assert metadata["extraction_method"] == "pymupdf4llm_ocr"
    assert worker.requests[0].use_ocr is True


def test_type3_fonts_still_use_markdown_conversion(monkeypatch, tmp_path):
    """Type3 fonts no longer bypass the layout worker.

    LaTeX papers routinely embed a Type3 font for a figure or symbol, and the
    old guard sent all of them to normalized extraction. The spawned worker's
    timeout now contains any pymupdf4llm hang, so the guard only cost quality.
    """
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _Type3Page:
        def get_fonts(self):
            return [(7, "n/a", "Type3", "T3Font_0", "F1", "")]

    class _Type3Doc(_FakeDoc):
        def __iter__(self):
            return iter([_Type3Page()])

    monkeypatch.setattr("arcaneum.indexing.pdf.extractor.pymupdf.open", lambda path: _Type3Doc())
    worker = _FakeWorker(pages=[{"text": "Markdown text", "metadata": {"page_number": 1}}])

    text, metadata = PDFExtractor(layout_worker=worker).extract(pdf_path)

    assert text == "Markdown text"
    assert metadata["extraction_method"] == "pymupdf4llm_markdown"
    assert len(worker.requests) == 1


def test_normalized_extraction_keeps_columns_separate(tmp_path):
    """Normalized extraction must not merge side-by-side columns into one line.

    get_text(sort=True) orders text by vertical position, so lines from the
    left and right columns that share a baseline were joined, leaving
    two-column papers unquotable.
    """
    pdf_path = tmp_path / "two-column.pdf"
    left = ["alpha one", "alpha two", "alpha three"]
    right = ["omega one", "omega two", "omega three"]
    with pymupdf.open() as doc:
        page = doc.new_page()
        for column, x in ((left, 50), (right, 320)):
            for row, line in enumerate(column):
                page.insert_text((x, 100 + row * 14), line)
        doc.save(pdf_path)

    text, metadata = PDFExtractor(markdown_conversion=False).extract(pdf_path)

    lines = [line for line in text.splitlines() if line.strip()]
    assert metadata["extraction_method"] == "pymupdf_normalized"
    assert lines == left + right


def test_markdown_extraction_parses_document_once_with_page_chunks(monkeypatch, tmp_path):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _TwoPageDoc(_FakeDoc):
        def __len__(self):
            return 2

    worker = _FakeWorker(
        pages=[
            {"text": "First page", "metadata": {"page_number": 1}},
            {"text": "Second page", "metadata": {"page_number": 2}},
        ],
        page_count=2,
    )


    text, metadata = PDFExtractor(layout_worker=worker).extract(pdf_path)

    assert text == "First page\nSecond page"
    assert len(worker.requests) == 1
    assert worker.requests[0].pdf_path == str(pdf_path.resolve())
    assert metadata["page_count"] == 2
    assert metadata["page_boundaries"] == [
        {"page_number": 1, "start_char": 0, "page_text_length": 10},
        {"page_number": 2, "start_char": 11, "page_text_length": 11},
    ]


def test_markdown_extraction_disables_layout_for_conversion(monkeypatch, tmp_path):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    worker = _FakeWorker(pages=[{"text": "Plain text", "metadata": {"page_number": 1}}])


    _, metadata = PDFExtractor(use_layout_analysis=False, layout_worker=worker).extract(pdf_path)

    assert worker.requests[0].layout is False
    assert metadata["layout_analyzed"] is False


def test_font_conversion_error_preserves_normalized_fallback(monkeypatch, tmp_path):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    worker = _FakeWorker(error=LayoutConversionError("font code=4"))

    monkeypatch.setattr(
        PDFExtractor,
        "_extract_with_pymupdf_normalized",
        lambda self, path: ("normalized", {"extraction_method": "pymupdf_normalized"}),
    )

    text, metadata = PDFExtractor(layout_worker=worker).extract(pdf_path)

    assert text == "normalized"
    assert metadata["extraction_method"] == "pymupdf_normalized"
    assert metadata["layout_fallback_reason"] == "LayoutConversionError"
    assert "font code=4" in metadata["layout_fallback_detail"]
    assert worker.closed is True


def test_non_font_conversion_error_still_falls_back_to_normalized(monkeypatch, tmp_path):
    """A malformed-PDF conversion error must not strand an extractable file.

    Regression: the fallback used to be gated on a substring match for "font" in
    the error text, so a malformed xref entry ("invalid literal for int() with
    base 10: '6842 1 R'") re-raised and failed the file even though normalized
    extraction reads it fine. Fallback viability does not depend on the cause.
    """
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    worker = _FakeWorker(
        error=LayoutConversionError("invalid literal for int() with base 10: '6842 1 R'")
    )

    monkeypatch.setattr(
        PDFExtractor,
        "_extract_with_pymupdf_normalized",
        lambda self, path: ("normalized", {"extraction_method": "pymupdf_normalized"}),
    )

    text, metadata = PDFExtractor(layout_worker=worker).extract(pdf_path)

    assert text == "normalized"
    assert metadata["extraction_method"] == "pymupdf_normalized"
    assert metadata["layout_fallback_reason"] == "LayoutConversionError"
    assert "6842 1 R" in metadata["layout_fallback_detail"]
    assert worker.closed is True


def test_crashed_worker_is_contained_before_truthful_fallback(monkeypatch, tmp_path):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    worker = _FakeWorker(error=LayoutWorkerCrashed("exit code 17"))


    def normalized(self, path):
        # The production worker raises only after terminating and joining its child.
        assert worker.error is not None
        return "normalized", {"extraction_method": "pymupdf_normalized"}

    monkeypatch.setattr(PDFExtractor, "_extract_with_pymupdf_normalized", normalized)

    text, metadata = PDFExtractor(layout_worker=worker).extract(pdf_path)

    assert text == "normalized"
    assert metadata["extraction_method"] == "pymupdf_normalized"
    assert metadata["layout_fallback_reason"] == "LayoutWorkerCrashed"
    assert "exit code 17" in metadata["layout_fallback_detail"]


def test_uncontained_worker_blocks_in_process_fallback(monkeypatch, tmp_path):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    worker = _FakeWorker(error=LayoutWorkerContainmentError("pid 123 remained alive"))
    fallback_calls = []


    def normalized(self, path):
        fallback_calls.append(path)
        return "normalized", {"extraction_method": "pymupdf_normalized"}

    monkeypatch.setattr(PDFExtractor, "_extract_with_pymupdf_normalized", normalized)

    with pytest.raises(LayoutWorkerContainmentError, match="remained alive"):
        PDFExtractor(layout_worker=worker).extract(pdf_path)

    assert fallback_calls == []

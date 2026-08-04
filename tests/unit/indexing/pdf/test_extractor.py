"""Unit tests for PDF extraction metadata and worker fallback."""

from arcaneum.indexing.pdf.extractor import PDFExtractor
from arcaneum.indexing.pdf.layout_worker import (
    LayoutConversionError,
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

    monkeypatch.setattr(PDFExtractor, "_has_type3_fonts", lambda self, path: False)
    monkeypatch.setattr("arcaneum.indexing.pdf.extractor.pymupdf.open", lambda path: _FakeDoc())
    worker = _FakeWorker(pages=[{"text": "OCR text", "metadata": {"page_number": 1}}])

    _, metadata = PDFExtractor(use_ocr=True, layout_worker=worker).extract(pdf_path)

    assert metadata["extraction_method"] == "pymupdf4llm_ocr"
    assert worker.requests[0].use_ocr is True


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

    monkeypatch.setattr(PDFExtractor, "_has_type3_fonts", lambda self, path: False)

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

    monkeypatch.setattr(PDFExtractor, "_has_type3_fonts", lambda self, path: False)

    _, metadata = PDFExtractor(use_layout_analysis=False, layout_worker=worker).extract(pdf_path)

    assert worker.requests[0].layout is False
    assert metadata["layout_analyzed"] is False


def test_font_conversion_error_preserves_normalized_fallback(monkeypatch, tmp_path):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    worker = _FakeWorker(error=LayoutConversionError("font code=4", font_error=True))
    fallback = ("normalized", {"extraction_method": "pymupdf_normalized"})

    monkeypatch.setattr(PDFExtractor, "_has_type3_fonts", lambda self, path: False)
    monkeypatch.setattr(
        PDFExtractor, "_extract_with_pymupdf_normalized", lambda self, path: fallback
    )

    assert PDFExtractor(layout_worker=worker).extract(pdf_path) == fallback
    assert worker.closed is True


def test_crashed_worker_is_contained_before_truthful_fallback(monkeypatch, tmp_path):
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    worker = _FakeWorker(error=LayoutWorkerCrashed("exit code 17"))

    monkeypatch.setattr(PDFExtractor, "_has_type3_fonts", lambda self, path: False)

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

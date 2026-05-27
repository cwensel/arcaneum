"""Unit tests for PDF extraction metadata."""

from arcaneum.indexing.pdf.extractor import PDFExtractor


class _FakeDoc:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __len__(self):
        return 1


def test_markdown_extraction_marks_auto_ocr_method(monkeypatch, tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(PDFExtractor, "_has_type3_fonts", lambda self, path: False)
    monkeypatch.setattr("arcaneum.indexing.pdf.extractor.pymupdf.open", lambda path: _FakeDoc())
    monkeypatch.setattr(
        "arcaneum.indexing.pdf.extractor.pymupdf4llm.to_markdown",
        lambda *args, **kwargs: "OCR text",
    )

    _, metadata = PDFExtractor(use_ocr=True).extract(pdf_path)

    assert metadata["extraction_method"] == "pymupdf4llm_ocr"


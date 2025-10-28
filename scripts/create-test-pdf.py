#!/usr/bin/env python3
"""Create a simple test PDF for testing PDF indexing."""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path

def create_test_pdf(output_path: Path):
    """Create a simple test PDF with multiple pages."""
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    # Page 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Test Document for PDF Indexing")

    c.setFont("Helvetica", 12)
    c.drawString(100, height - 150, "This is a test document to verify the PDF indexing system.")
    c.drawString(100, height - 180, "It contains multiple sections and pages to test chunking.")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 230, "Section 1: Introduction")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 260, "The PDF indexing system uses PyMuPDF for fast text extraction.")
    c.drawString(100, height - 280, "It is approximately 95x faster than alternative libraries.")
    c.drawString(100, height - 300, "The system also supports OCR for scanned documents using Tesseract.")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 350, "Section 2: Features")
    c.setFont("Helvetica", 12)
    features = [
        "- Text extraction from machine-generated PDFs",
        "- OCR support for scanned documents",
        "- Intelligent chunking with 15% overlap",
        "- Late chunking for long documents (2K-8K tokens)",
        "- Incremental indexing (only new/modified files)",
        "- Batch upload with retry logic",
    ]
    y = height - 380
    for feature in features:
        c.drawString(100, y, feature)
        y -= 20

    c.showPage()

    # Page 2
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 100, "Section 3: Technical Details")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 130, "The system supports multiple embedding models:")

    models = [
        "- stella: Best for long documents, 1024 dimensions",
        "- bge: Best for precision, 1024 dimensions",
        "- modernbert: Best for recent content, 768 dimensions",
        "- jina: Best for code + text, 768 dimensions",
    ]
    y = height - 160
    for model in models:
        c.drawString(100, y, model)
        y -= 20

    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 280, "Section 4: Performance")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 310, "Performance metrics:")
    c.drawString(100, height - 330, "- Text PDFs: ~100 files/minute")
    c.drawString(100, height - 350, "- Scanned PDFs: ~30 pages/minute with OCR")
    c.drawString(100, height - 370, "- Upload: ~333 chunks/second (sequential)")
    c.drawString(100, height - 390, "- Upload: ~1,111 chunks/second (4 parallel workers)")

    c.showPage()

    # Page 3
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 100, "Section 5: Usage Examples")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 130, "Basic indexing command:")
    c.drawString(100, height - 150, "  bin/arc index-pdfs ./pdfs --collection docs --model stella")

    c.drawString(100, height - 190, "Index PDFs (OCR enabled by default):")
    c.drawString(100, height - 210, "  bin/arc index-pdfs ./pdfs --collection docs")

    c.drawString(100, height - 250, "Force reindex:")
    c.drawString(100, height - 270, "  bin/arc index-pdfs ./pdfs --collection docs --force")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 320, "Conclusion")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 350, "This test document verifies that multi-page PDFs are properly")
    c.drawString(100, height - 370, "extracted, chunked, and indexed. Each section should become")
    c.drawString(100, height - 390, "one or more chunks depending on the model's chunk size.")

    c.showPage()
    c.save()

    print(f"✓ Created test PDF: {output_path}")

if __name__ == "__main__":
    test_dir = Path("test_pdfs")
    test_dir.mkdir(exist_ok=True)

    output_file = test_dir / "sample-document.pdf"
    create_test_pdf(output_file)

    print(f"✓ Test PDF ready for indexing")
    print(f"  Run: bin/arc index-pdfs ./test_pdfs --collection pdf-test --model stella")

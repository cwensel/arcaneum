#!/usr/bin/env bash
# Test script for PDF indexing (RDR-004)
set -e

echo "=== PDF Indexing Test Script ==="
echo ""

# Check if Qdrant is running
echo "Checking Qdrant server..."
if curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo "✓ Qdrant is running"
else
    echo "✗ Qdrant is not running. Start with: docker compose -f deploy/docker-compose.yml up -d"
    exit 1
fi

# Check Python dependencies
echo ""
echo "Checking Python dependencies..."
python -c "import pymupdf; import pdfplumber; import pytesseract; print('✓ PDF dependencies installed')" 2>/dev/null || {
    echo "✗ PDF dependencies missing. Install with:"
    echo "  pip install -e ."
    exit 1
}

# Create test collection if it doesn't exist
echo ""
echo "Creating test collection..."
bin/arc create-collection pdf-test --model stella --hnsw-m 16 --hnsw-ef 100 || {
    echo "Collection might already exist, continuing..."
}

# Run PDF indexing
echo ""
echo "Testing PDF indexing..."
if [ -d "./test_pdfs" ]; then
    bin/arc index-pdfs ./test_pdfs \
        --collection pdf-test \
        --model stella \
        --workers 4 \
        --verbose
else
    echo "⚠ No test_pdfs directory found. Create one with sample PDFs to test."
fi

echo ""
echo "=== Test Complete ==="

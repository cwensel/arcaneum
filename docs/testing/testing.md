# Testing Instructions for RDR-004 PDF Indexing

Simple terminal commands to test the PDF indexing implementation.

**Note on recommended workflow:** For normal use, prefer `arc corpus sync`
(dual-indexing to Qdrant + MeiliSearch) — see the
[Corpus Workflow (Recommended)](#corpus-workflow-recommended) section below.
The rest of this doc tests the single-system `arc index pdf` implementation
from RDR-004; those sections are marked "Single-System (Advanced)".

## Prerequisites Check

### 1. Check Python Version

```bash
python --version
# Should be >= 3.12
```

### 2. Check Qdrant is Running

```bash
curl http://localhost:6333/health
# Should return: {"title":"qdrant - vector search engine","version":"..."}
```

If not running:
```bash
arc container start
```

### 3. Install Dependencies

```bash
pip install -e .
```

### 4. Check System Dependencies (for OCR)

**macOS:**
```bash
# Install if needed
brew install tesseract poppler

# Verify
tesseract --version
pdfinfo -v
```

**Ubuntu/Debian:**
```bash
# Install if needed
sudo apt-get install tesseract-ocr poppler-utils

# Verify
tesseract --version
pdfinfo -v
```

## Corpus Workflow (Recommended)

For normal use, the dual-indexing corpus workflow is what you should test.
It indexes to both Qdrant and MeiliSearch in a single command.

```bash
# Create a corpus (both Qdrant collection and MeiliSearch index)
arc corpus create pdf-test --type pdf

# Sync PDFs to both systems
arc corpus sync pdf-test ./test_pdfs

# Re-run to verify incremental sync (nothing reindexed if files unchanged)
arc corpus sync pdf-test ./test_pdfs

# Detect renames and remove files no longer on disk
arc corpus sync pdf-test ./test_pdfs --parity

# Preview rename/remove changes without writing
arc corpus sync pdf-test ./test_pdfs --parity --dry-run

# Search both systems
arc search semantic "machine learning" --corpus pdf-test
arc search text '"neural network"' --corpus pdf-test

# List indexed items
arc corpus items pdf-test
arc corpus info pdf-test

# Cleanup
arc corpus delete pdf-test --confirm
```

## Quick Test (No PDFs Required) — Single-System (Advanced)

The tests below validate the `arc index pdf` single-system path (Qdrant only).
For normal use, prefer the corpus workflow above.

### Test 1: Verify CLI Works

```bash
bin/arc --help
```

**Expected Output:**
```
Usage: arc [OPTIONS] COMMAND [ARGS]...
  Arcaneum: Semantic and full-text search tools for Qdrant and MeiliSearch
...
```

### Test 2: Check Index PDFs Command

```bash
bin/arc index pdf --help
```

See `arc index pdf --help` output for the current option list. Key options:
`--collection` (required), `--force`, `--no-ocr`, `--ocr-language`,
`--ocr-workers`, `--embedding-batch-size`, `--no-gpu`, `--verbose`, `--json`.

### Test 3: List Collections

```bash
bin/arc collection list
```

**Expected Output:**
```
Collections (0 total)
No collections found
```

## Full Test with PDFs

### Test 4: Create Test Collection

```bash
bin/arc collection create pdf-test --type pdf
```

**Expected Output:**
```
✓ Collection created: pdf-test
  Model: stella (BAAI/bge-large-en-v1.5)
  Dimensions: 1024
  Distance: cosine
```

### Test 5: Create Test PDFs Directory

```bash
mkdir -p test_pdfs
```

**Option A: Use Sample PDFs**
- Add some PDF files to `test_pdfs/` directory
- Any PDFs work (documentation, papers, books, etc.)

**Option B: Create a Test PDF (macOS/Linux with LaTeX)**
```bash
cat > test_pdfs/sample.txt << 'EOF'
This is a test document for PDF indexing.
It contains some sample text to verify the extraction works.

Section 1: Introduction
The PDF indexing system supports text extraction using PyMuPDF.

Section 2: Features
- Fast extraction (95x faster than alternatives)
- OCR support for scanned documents
- Intelligent chunking with overlap
EOF

# Convert to PDF (if you have tools installed)
# Otherwise, just add your own PDFs to test_pdfs/
```

### Test 6: Index PDFs (Basic)

```bash
bin/arc index pdf ./test_pdfs --collection pdf-test --verbose
```

**Expected Output:**
```
Indexing PDFs from: /path/to/test_pdfs
Collection: pdf-test
Model: stella

Found 1 total PDF files
Incremental sync: 1 new/modified, 0 already indexed

PDFs: 100%|████████████████| 1/1 [00:02<00:00]
Chunks: 5it [00:02,  2.13it/s]

============================================================
INDEXING COMPLETE
Files processed: 1
Chunks uploaded: 5
Errors: 0
============================================================

✓ Indexing Complete

Indexing Results
┌─────────────────┬────────┐
│ Metric          │ Value  │
├─────────────────┼────────┤
│ Files Processed │ 1      │
│ Chunks Uploaded │ 5      │
│ Errors          │ 0      │
└─────────────────┴────────┘
```

### Test 7: Verify Collection Info

```bash
bin/arc collection info pdf-test
```

**Expected Output:**
```
Collection: pdf-test

Vectors Configuration
┌──────────┬────────────┬────────┬─────────┐
│ Vector   │ Size       │ Dist.  │ Count   │
├──────────┼────────────┼────────┼─────────┤
│ stella   │ 1024       │ cosine │ 5       │
└──────────┴────────────┴────────┴─────────┘

Configuration
  Points: 5
  Segments: 1
  Status: green
```

### Test 8: Incremental Indexing (Run Again)

```bash
bin/arc index pdf ./test_pdfs --collection pdf-test
```

**Expected Output:**
```
...
Incremental sync: 0 new/modified, 1 already indexed
No PDFs to index
...
```

This proves incremental indexing works!

### Test 9: Force Reindex

```bash
bin/arc index pdf ./test_pdfs --collection pdf-test --force
```

**Expected Output:**
```
...
Force reindex: processing all 1 PDFs
...
Files processed: 1
```

### Test 10: JSON Output (for Automation)

```bash
bin/arc index pdf ./test_pdfs --collection pdf-test --json
```

**Expected Output:**
```json
{
  "success": true,
  "collection": "pdf-test",
  "model": "stella",
  "stats": {
    "files": 0,
    "chunks": 0,
    "errors": 0
  }
}
```

## Test OCR (If You Have Scanned PDFs)

### Test 11: Index with OCR (enabled by default)

```bash
# Add a scanned PDF to test_pdfs/ first
# OCR is enabled by default, so no flag needed
bin/arc index pdf ./test_pdfs \
  --collection pdf-test \
  --ocr-language eng \
  --verbose
```

**Expected Output:**
```
...
Triggering OCR for scanned.pdf (text: 50 chars)
OCR completed: 5000 chars, avg confidence: 85.0%
...
```

## Automated Test Script

### Test 12: Run Full Test Suite

```bash
./scripts/test-pdf-indexing.sh
```

**Expected Output:**
```
=== PDF Indexing Test Script ===

Checking Qdrant server...
✓ Qdrant is running

Checking Python dependencies...
✓ PDF dependencies installed

Creating test collection...
✓ Collection created: pdf-test

Testing PDF indexing...
[progress bars and results]

=== Test Complete ===
```

## Cleanup After Testing

```bash
# Delete test collection
bin/arc collection delete pdf-test --confirm

# Remove test PDFs
rm -rf test_pdfs
```

## Common Issues

### Issue: "Command not found"

```bash
chmod +x bin/arc
```

### Issue: "Qdrant connection refused"

```bash
arc container start
sleep 5  # Wait for startup
curl http://localhost:6333/health
```

### Issue: "tesseract not found"

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

### Issue: "Module not found"

```bash
pip install -e .
```

## Success Criteria

✅ All commands run without errors
✅ PDFs are indexed and chunks created
✅ Collection shows correct point count
✅ Incremental indexing skips already-indexed files
✅ JSON output is valid
✅ OCR processes scanned PDFs (if tested)

## Next Steps

Once testing is complete, you can:

1. **Index your real documents** (recommended, dual-indexing):
   ```bash
   bin/arc corpus create my-docs --type pdf
   bin/arc corpus sync my-docs /path/to/your/pdfs
   ```

   Or single-system (Qdrant only):
   ```bash
   bin/arc collection create my-docs --type pdf
   bin/arc index pdf /path/to/your/pdfs --collection my-docs
   ```

2. **Search your documents:**
   ```bash
   bin/arc search semantic "your query" --corpus my-docs
   bin/arc search text "your query" --corpus my-docs
   ```

3. **Commit the implementation:**
   ```bash
   git add .
   git commit -m "Implement RDR-004: PDF Bulk Indexing"
   ```

# Quick Test Commands

Copy/paste these commands to test the PDF indexing implementation.

## Prerequisites

```bash
# 1. Start Qdrant
docker compose -f deploy/docker-compose.yml up -d

# 2. Install dependencies
pip install -e .

# 3. Verify CLI works
bin/arc --help

# 4. (If behind corporate proxy) Pre-download models and use --offline
#    See CORPORATE-PROXY.md for details
```

## Basic Test (5 commands)

```bash
# 1. Create collection
bin/arc create-collection pdf-test --model stella

# 2. Create test directory
mkdir -p test_pdfs
# (Add some PDFs to test_pdfs/ or use the generator below)

# 3. Index PDFs
bin/arc index-pdfs ./test_pdfs --collection pdf-test --model stella --verbose

# 4. Check results
bin/arc collection-info pdf-test

# 5. Test incremental indexing (should skip already-indexed files)
bin/arc index-pdfs ./test_pdfs --collection pdf-test --model stella
```

## Generate Test PDF (Optional)

```bash
# Install reportlab first
pip install reportlab

# Create test PDF
python scripts/create-test-pdf.py
```

## Full Test Script

```bash
./scripts/test-pdf-indexing.sh
```

## Cleanup

```bash
bin/arc delete-collection pdf-test --confirm
rm -rf test_pdfs
```

## Expected Results

✅ Collection created successfully
✅ PDFs indexed with chunks count
✅ Collection shows correct point count
✅ Incremental indexing skips already-indexed files
✅ No errors in output

## If Something Fails

See **TESTING.md** for detailed troubleshooting and step-by-step instructions.

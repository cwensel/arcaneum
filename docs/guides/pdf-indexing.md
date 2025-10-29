# PDF Indexing Guide (RDR-004)

This guide covers the bulk PDF indexing system with OCR support.

## Overview

The PDF indexing pipeline supports:

- **Text PDFs**: Machine-generated documents with embedded text (PyMuPDF, ~95x faster)
- **Image PDFs**: Scanned documents requiring OCR (Tesseract)
- **Mixed PDFs**: Documents with both text and scanned images
- **Incremental indexing**: Only new/modified files are processed
- **Late chunking**: Improved retrieval quality for long documents

## Prerequisites

### System Dependencies

**macOS:**
```bash
brew install tesseract poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils
```

### Python Dependencies

Already included in `pyproject.toml`:
```bash
pip install -e .
```

### Start Qdrant Server

```bash
docker compose up -d
```

## Quick Start

### 1. Create Collection

```bash
arc collection create pdf-docs --model stella --hnsw-m 16 --hnsw-ef 100
```

### 2. Index PDFs

```bash
arc index pdfs /path/to/pdfs \
  --collection pdf-docs \
  --model stella \
  --workers 4
```

### 3. Index PDFs with OCR disabled (if all PDFs are machine-generated text)

```bash
arc index pdfs /path/to/text-pdfs \
  --collection pdf-docs \
  --model stella \
  --no-ocr \
  --workers 4
```

**Note**: OCR is enabled by default to handle scanned PDFs automatically.

## Usage

### Basic Command

```bash
arc index pdfs <directory> --collection <name> --model <model>
```

### Options

- `--collection`: Target Qdrant collection (required)
- `--model`: Embedding model (stella, bge, modernbert, jina) [default: stella]
- `--workers`: Parallel upload workers [default: 4]
- `--no-ocr`: Disable OCR (enabled by default for scanned PDFs)
- `--ocr-language`: OCR language code (eng, fra, spa, deu, etc.) [default: eng]
- `--force`: Force reindex all files (bypass incremental sync)
- `--verbose`: Verbose output
- `--json`: Output JSON format

### Examples

**Index technical documentation:**
```bash
arc index pdfs ./docs \
  --collection tech-docs \
  --model stella \
  --workers 8
```

**Index scanned books (OCR enabled by default):**
```bash
arc index pdfs ./books \
  --collection book-archive \
  --model stella \
  --ocr-language eng \
  --workers 4
```

**Force reindex all PDFs:**
```bash
arc index pdfs ./pdfs \
  --collection pdf-docs \
  --model stella \
  --force
```

**JSON output for scripting:**
```bash
arc index pdfs ./pdfs \
  --collection pdf-docs \
  --model stella \
  --json > results.json
```

## Simplified CLI Scripts

For convenience, use the `bin/arc` wrapper:

```bash
# Direct CLI usage (development mode)
bin/arc index pdfs /path/to/pdfs --collection pdf-docs --model stella

# After pip install
arc index pdfs /path/to/pdfs --collection pdf-docs --model stella

# Full test script
./scripts/test-pdf-indexing.sh
```

## Incremental Indexing

The system automatically tracks indexed files using metadata queries:

- **First run**: All PDFs are indexed
- **Subsequent runs**: Only new or modified PDFs are processed
- **Detection**: Based on file path and content hash (SHA256)

To bypass incremental sync and reindex everything, use `--force`.

## Model Selection

Choose the embedding model based on your use case:

| Model | Best For | Chunk Size | Late Chunking |
|-------|----------|------------|---------------|
| **stella** | Long documents, general purpose | 768 tokens | ✅ Yes |
| **bge** | Precision, short documents | 460 tokens | ❌ No |
| **modernbert** | Long context, recent content | 1536 tokens | ✅ Yes |
| **jina** | Code + text, multilingual | 1536 tokens | ✅ Yes |

## OCR Configuration

### Tesseract (Default)

- **Speed**: 2s per page (CPU)
- **Accuracy**: 99%+ on clean printed text at 300 DPI
- **Languages**: 100+ supported
- **Best for**: High-quality scans, CPU-only environments

### Trigger Logic

OCR is enabled by default and automatically triggered when:
- Extracted text < 100 characters (scanned PDFs)

To disable OCR completely (if all PDFs are machine-generated text):
- Use `--no-ocr` flag

### Supported Languages

Common language codes:
- `eng` - English
- `fra` - French
- `spa` - Spanish
- `deu` - German
- `ara` - Arabic
- `chi_sim` - Chinese (Simplified)
- `jpn` - Japanese

Install additional languages:
```bash
# macOS
brew install tesseract-lang

# Ubuntu/Debian
sudo apt-get install tesseract-ocr-fra tesseract-ocr-spa
```

## Performance

### Throughput

- **Text PDFs**: ~100 files/minute (PyMuPDF)
- **Scanned PDFs**: ~30 pages/minute (Tesseract OCR)
- **Upload**: ~333 chunks/second (sequential), ~1,111 chunks/second (4 workers)

### Optimization

- **Batch size**: 100-200 chunks per batch (default: 100)
- **Parallel workers**: 4 recommended (default: 4)
- **HNSW indexing**: Disable during bulk upload (`m=0`), re-enable after

## Troubleshooting

### Tesseract Not Found

```
Error: Tesseract not installed
```

**Solution:**
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

### Poppler Not Found (pdf2image)

```
Error: Unable to get page count. Is poppler installed?
```

**Solution:**
```bash
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils
```

### Out of Memory

For large PDFs or many parallel workers:

1. Reduce workers: `--workers 2`
2. Use on-disk vectors in collection config
3. Disable HNSW indexing during upload

### Slow OCR

OCR is CPU-intensive (2s per page). To speed up:

1. Disable OCR if all PDFs are machine-generated text (`--no-ocr`)
2. Use GPU with EasyOCR (future enhancement)
3. Process in smaller batches

## Architecture

### Pipeline Phases

```
Phase 1: PDF Extraction (PyMuPDF + pdfplumber fallback)
    ↓
Phase 2: OCR Processing (if needed, Tesseract)
    ↓
Phase 3: Chunking (Traditional or Late Chunking)
    ↓
Phase 4: Embedding Generation (FastEmbed)
    ↓
Phase 5: Batch Upload (100-200 chunks, 4 workers)
```

### Modules

- `indexing/pdf/extractor.py` - PDF text extraction
- `indexing/pdf/ocr.py` - OCR integration
- `indexing/pdf/chunker.py` - Chunking strategies
- `indexing/common/sync.py` - Incremental indexing
- `indexing/uploader.py` - Batch upload orchestrator
- `cli/index_pdfs.py` - CLI command

## Configuration

Model configs are defined in `config.py`:

```python
DEFAULT_MODELS = {
    "stella": ModelConfig(
        chunk_size=768,  # Conservative for PDF
        chunk_overlap=115,  # 15% overlap
        late_chunking=True,
        char_to_token_ratio=3.3,
    ),
    # ... other models
}
```

## Related Documentation

- [RDR-004: PDF Bulk Indexing](../docs/rdr/RDR-004-pdf-bulk-indexing.md) - Full specification
- [RDR-002: Qdrant Integration](../docs/rdr/RDR-002-qdrant-docker-compose.md) - Server setup
- [RDR-003: Collection Management](../docs/rdr/RDR-003-qdrant-collection-creation-cli.md) - Collection creation

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
arc index pdf /path/to/pdfs \
  --collection pdf-docs \
  --model stella \
  --workers 4
```

### 3. Index PDFs with OCR disabled (if all PDFs are machine-generated text)

```bash
arc index pdf /path/to/text-pdfs \
  --collection pdf-docs \
  --model stella \
  --no-ocr \
  --workers 4
```

**Note**: OCR is enabled by default to handle scanned PDFs automatically.

## Usage

### Basic Command

```bash
arc index pdf <directory> --collection <name> --model <model>
```

### Options

**Basic Options:**

- `--collection`: Target Qdrant collection (required)
- `--model`: Embedding model (stella, bge, modernbert, jina) [default: stella]
- `--force`: Force reindex all files (bypass incremental sync)
- `--no-gpu`: Disable GPU acceleration (GPU enabled by default for 1.5-3x speedup)
- `--streaming`: Stream embeddings to Qdrant immediately (lower memory usage)
- `--verbose`: Verbose output (show progress, suppress library warnings)
- `--debug`: Debug mode (show all library warnings including transformers)
- `--json`: Output JSON format

**Performance Tuning:**

- `--embedding-batch-size`: Batch size for embedding generation [default: 200]
- `--process-priority`: Process scheduling priority (low/normal/high) [default: normal]

**OCR Options:**

- `--no-ocr`: Disable OCR (enabled by default for scanned PDFs)
- `--ocr-language`: OCR language code (eng, fra, spa, deu, etc.) [default: eng]
- `--ocr-workers`: Number of parallel OCR workers for page processing [default: cpu_count]

### Examples

**Index technical documentation:**

```bash
arc index pdf ./docs \
  --collection tech-docs \
  --model stella \
  --workers 8
```

**Index scanned books (OCR enabled by default with parallel page processing):**

```bash
arc index pdf ./books \
  --collection book-archive \
  --model stella \
  --ocr-language eng \
  --ocr-workers 8 \
  --workers 4
```

**Force reindex all PDFs:**

```bash
arc index pdf ./pdfs \
  --collection pdf-docs \
  --model stella \
  --force
```

**JSON output for scripting:**

```bash
arc index pdf ./pdfs \
  --collection pdf-docs \
  --model stella \
  --json > results.json
```

**Disable GPU for CPU-only mode:**

```bash
arc index pdf ./pdfs \
  --collection pdf-docs \
  --model stella \
  --no-gpu
```

**Debug mode (show all warnings):**

```bash
arc index pdf ./pdfs \
  --collection pdf-docs \
  --model stella \
  --debug
```

**Maximum performance (use all CPU cores):**

```bash
arc index pdf ./pdfs \
  --collection pdf-docs \
  --model stella \
  --embedding-batch-size 500 \
  --process-priority low
```

**Streaming mode (lower memory for large collections):**

```bash
arc index pdf ./pdfs \
  --collection pdf-docs \
  --model stella \
  --streaming
```

**Conservative (25% of CPU, good for background processing):**

```bash
arc index pdf ./pdfs \
  --collection pdf-docs \
  --model stella \
  --process-priority low
```

## Simplified CLI Scripts

For convenience, use the `bin/arc` wrapper:

```bash
# Direct CLI usage (development mode)
bin/arc index pdf /path/to/pdfs --collection pdf-docs --model stella

# After pip install
arc index pdf /path/to/pdfs --collection pdf-docs --model stella

# Full test script
./scripts/test-pdf-indexing.sh
```

## Incremental Indexing

The system automatically tracks indexed files using metadata queries:

- **First run**: All PDFs are indexed
- **Subsequent runs**: Only new or modified PDFs are processed
- **Detection**: Based on file path and content hash (SHA256)

To bypass incremental sync and reindex everything, use `--force`.

## GPU Acceleration

GPU acceleration is **enabled by default** for 1.5-3x faster embedding generation:

- **Apple Silicon**: Uses MPS (Metal Performance Shaders) backend
- **NVIDIA GPUs**: Uses CUDA backend
- **CPU Fallback**: Automatic when GPU unavailable
- **Disable**: Use `--no-gpu` flag for CPU-only mode

**Compatible Models** (verified with GPU support):

- **stella** (recommended) - Full MPS support on Apple Silicon
- **jina-code** - Full MPS support on Apple Silicon
- **bge-small**, **bge-base** - CoreML support

**Performance**: 1.5-3x speedup with GPU compared to CPU-only mode.

**When to disable GPU**:

- Thermal concerns (laptop getting too hot)
- Battery life (running on battery power)
- GPU busy with other tasks

```bash
# Force CPU-only mode
arc index pdf ./pdfs --collection docs --model stella --no-gpu
```

## Model Selection

Choose the embedding model based on your use case:

| Model | Best For | Chunk Size | Late Chunking | GPU Support |
|-------|----------|------------|---------------|-------------|
| **stella** | Long documents, general purpose | 768 tokens | ✅ Yes | ✅ MPS |
| **bge** | Precision, short documents | 460 tokens | ❌ No | ⚠️ CoreML |
| **modernbert** | Long context, recent content | 1536 tokens | ✅ Yes | ✅ MPS |
| **jina** | Code + text, multilingual | 1536 tokens | ✅ Yes | ✅ MPS |

## OCR Configuration

### Tesseract (Default)

- **Speed**: 2s per page (CPU), ~0.5s per page with parallel processing (8 workers)
- **Accuracy**: 99%+ on clean printed text at 300 DPI
- **Languages**: 100+ supported
- **Parallelization**: ProcessPoolExecutor for concurrent page processing (default: cpu_count workers)
- **Best for**: High-quality scans, CPU-only environments, multi-page documents

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
- **Scanned PDFs**: ~30 pages/minute (single-threaded Tesseract), ~120 pages/minute (parallel, 8 workers)
- **Upload**: ~333 chunks/second (sequential), ~1,111 chunks/second (4 workers)

### Optimization

- **Batch size**: 200-300 chunks per batch (default: 200 for embeddings, 300 for uploads)
- **Parallel workers**: 4 recommended (default: 4)
- **HNSW indexing**: Disable during bulk upload (`m=0`), re-enable after (RDR-013 bulk mode)

## Troubleshooting

### Tesseract Not Found

```text
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

```text
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

1. Use streaming mode: `--streaming` (uploads embeddings immediately, reduces memory)
2. Reduce workers: `--workers 2`
3. Use on-disk vectors in collection config
4. Disable HNSW indexing during upload

### GPU Memory Errors (MPS/CUDA)

If you see errors like `MPS backend out of memory` when using large models like `stella` (1.5B params):

```text
RuntimeError: MPS backend out of memory (MPS allocated: 12.25 GiB...)
```

The system uses adaptive batch sizes based on model size, but if you still hit memory limits:

1. **Use a smaller model**: Try `bge` or `minilm` instead of `stella`
2. **Disable GPU**: `--no-gpu` forces CPU-only mode
3. **Close other apps**: Free up GPU memory used by other applications
4. **Reduce batch size**: `--embedding-batch-size 100` (lower = less memory)

**Model memory requirements (approximate on MPS):**

| Model | Size | Memory Usage |
|-------|------|--------------|
| `stella` | 1.5B params | ~12-15 GB |
| `jina-code-1.5b` | 1.5B params | ~12-15 GB |
| `nomic-code` | 7B params | ~20+ GB |
| `jina-code-0.5b` | 500M params | ~4-6 GB |
| `jina-code` | 137M params | ~2-3 GB |
| `minilm` | 22M params | <1 GB |

### Slow OCR

OCR is CPU-intensive (2s per page). To speed up:

1. Enable parallel processing: `--ocr-workers 8` (default: cpu_count, provides 4x speedup)
2. Disable OCR if all PDFs are machine-generated text (`--no-ocr`)
3. Use GPU with EasyOCR (future enhancement)
4. Process in smaller batches

## Architecture

### Pipeline Phases

```text
Phase 1: PDF Extraction (PyMuPDF + pdfplumber fallback)
    ↓
Phase 2: OCR Processing (if needed, Tesseract with parallel page processing)
    ↓
Phase 3: Chunking (Traditional or Late Chunking)
    ↓
Phase 4: Embedding Generation (FastEmbed, 200 chunks/batch, parallel)
    ↓
Phase 5: Batch Upload (300 chunks/batch, 4 workers, bulk mode)
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

## Full-Text Indexing (RDR-010)

In addition to semantic search via Qdrant, PDFs can be indexed to MeiliSearch for
exact phrase and keyword search. This complements semantic search by providing:

- **Exact phrase matching**: Find specific quotes or terminology
- **Typo-tolerant search**: Find content despite spelling variations
- **Page-level granularity**: Results point to specific pages

### Quick Start

```bash
# Create MeiliSearch index (if not already created)
arc indexes create pdf-docs --type pdf

# Index PDFs to MeiliSearch for full-text search
arc index text pdf /path/to/pdfs --index pdf-docs
```

### Command Options

```bash
arc index text pdf <directory> --index <name> [options]
```

**Required:**

- `--index`: Target MeiliSearch index name

**Optional:**

- `--recursive / --no-recursive`: Search subdirectories (default: recursive)
- `--force`: Force reindex all files (skip change detection)
- `--ocr / --no-ocr`: Enable/disable OCR for scanned PDFs (default: enabled)
- `--ocr-language`: OCR language code (default: eng)
- `--batch-size`: Documents per batch upload (default: 1000)
- `--verbose`: Show detailed progress
- `--json`: JSON output for scripting

### Examples

**Index technical documentation:**

```bash
arc index text pdf ./docs --index pdf-docs
```

**Index with OCR for scanned documents:**

```bash
arc index text pdf ./scanned-books --index pdf-docs --ocr-language eng
```

**Force reindex all PDFs:**

```bash
arc index text pdf ./pdfs --index pdf-docs --force
```

**JSON output for scripting:**

```bash
arc index text pdf ./pdfs --index pdf-docs --json > results.json
```

### Dual Indexing Strategy

For comprehensive search, index PDFs to both Qdrant (semantic) and MeiliSearch (full-text).

#### Using Corpus Commands (Recommended)

A "corpus" is a paired Qdrant collection and MeiliSearch index that share the same name.
The `corpus` commands provide a unified workflow for dual indexing:

```bash
# Create both collection and index in one command
arc corpus create my-papers --type pdf --models stella

# Index to both systems in one command
arc corpus sync /path/to/pdfs --corpus my-papers

# Search both systems
arc search semantic "machine learning" --collection my-papers  # Qdrant
arc search text '"neural network"' --index my-papers           # MeiliSearch
```

**Using Existing Collection/Index as a Corpus:**

If you already have a Qdrant collection and MeiliSearch index with the same name,
you can use `corpus sync` directly without running `corpus create`:

```bash
# If 'Papers' collection and 'Papers' index already exist:
arc corpus sync /path/to/pdfs --corpus Papers
```

The only requirement is that both the collection and index exist with the same name.
The `corpus create` command is just a convenience that creates both in one step.

#### Using Separate Commands

Alternatively, you can manage Qdrant and MeiliSearch separately:

```bash
# Step 1: Create collections/indexes
arc collection create pdf-docs --type pdf      # Qdrant collection
arc indexes create pdf-docs --type pdf         # MeiliSearch index

# Step 2: Index to Qdrant (semantic search)
arc index pdf /path/to/pdfs --collection pdf-docs

# Step 3: Index to MeiliSearch (full-text search)
arc index text pdf /path/to/pdfs --index pdf-docs

# Search semantically (conceptual matches)
arc search semantic "machine learning techniques" --collection pdf-docs

# Search exact phrases (keyword matches)
arc search text '"neural network architecture"' --index pdf-docs
```

### Change Detection

The full-text indexer tracks indexed files using SHA-256 file hashes:

- **First run**: All PDFs are indexed
- **Subsequent runs**: Only new or modified PDFs are processed
- **Detection**: Based on file path and content hash

To bypass change detection and reindex everything, use `--force`.

### Document Schema

Each page is indexed as a separate document with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique document ID (filename + path hash + page) |
| `content` | string | Page text content (searchable) |
| `filename` | string | PDF filename (searchable) |
| `file_path` | string | Absolute file path (filterable) |
| `page_number` | int | Page number (filterable, sortable) |
| `file_hash` | string | SHA-256 hash for change detection (filterable) |
| `extraction_method` | string | How text was extracted (filterable) |
| `is_image_pdf` | bool | Whether OCR was used (filterable) |

### Filtering Examples

```bash
# Search specific page range
arc search text "results" --index pdf-docs --filter "page_number > 10 AND page_number < 20"

# Search only OCR'd documents
arc search text "scanned content" --index pdf-docs --filter "is_image_pdf = true"

# Search by filename pattern (requires exact match)
arc search text "findings" --index pdf-docs --filter 'filename = "report.pdf"'
```

## Related Documentation

- [RDR-004: PDF Bulk Indexing](../rdr/RDR-004-pdf-bulk-indexing.md) - Semantic indexing specification
- [RDR-010: PDF Full-Text Indexing](../rdr/RDR-010-pdf-fulltext-indexing.md) - Full-text indexing specification
- [RDR-008: Full-Text Search Server Setup](../rdr/RDR-008-fulltext-search-server-setup.md) - MeiliSearch setup
- [RDR-009: Dual Indexing Strategy](../rdr/RDR-009-dual-indexing-strategy.md) - Dual indexing architecture

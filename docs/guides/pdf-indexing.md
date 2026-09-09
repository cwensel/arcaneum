# PDF Indexing Guide (RDR-004)

This guide covers the bulk PDF indexing system with OCR support.

## Overview

The PDF indexing pipeline supports:

- **Text PDFs**: Machine-generated documents with embedded text (PyMuPDF, ~95x faster)
- **Image PDFs**: Scanned documents requiring OCR (Tesseract)
- **Mixed PDFs**: Page-level selection between embedded text and raster OCR
- **Incremental indexing**: Only new/modified files are processed
- **PDF text normalization**: Markdown conversion, safe fragment merging, and
  omission of unrecoverable replacement-character spans
- **Quality tracking**: Persisted extraction candidates, page coverage, OCR
  provenance, warnings, and indexing-policy identity
- **Search hygiene**: Content-identical source deduplication and reference
  sections excluded from semantic search by default

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

### Start Search Services

```bash
arc container start
```

## Quick Start

**Use `arc corpus sync` for PDF indexing unless you explicitly need Qdrant-only
(no MeiliSearch).** The `arc index pdf` command is a single-system advanced
alternative documented further below.

```bash
# Create corpus (creates both Qdrant collection and MeiliSearch index)
arc corpus create pdf-docs --type pdf

# Sync PDFs to both systems
arc corpus sync pdf-docs /path/to/pdfs

# Also detect renames and remove indexed PDFs no longer on disk
arc corpus sync pdf-docs /path/to/pdfs --parity

# Search with semantic or full-text
arc search semantic "machine learning concepts" --corpus pdf-docs
arc search text '"specific phrase"' --corpus pdf-docs
```

Detected reference sections remain indexed and available for citation research,
but ordinary semantic search omits them so bibliography fragments do not crowd
out body evidence:

```bash
arc search semantic "machine learning concepts" --corpus pdf-docs --include-references
```

### PDFs with OCR disabled (if all PDFs are machine-generated text)

OCR is not tunable from `arc corpus sync` today — the corpus sync pipeline
auto-detects and applies OCR as needed. If you need explicit OCR control
(`--no-ocr`, `--ocr-language`, `--ocr-workers`), use the single-system
`arc index pdf` command documented in
[Single-System PDF Indexing](#single-system-pdf-indexing-advanced) below.

### Common `arc corpus sync` options for PDFs

```bash
# Force reindex all PDFs
arc corpus sync pdf-docs /path/to/pdfs --force

# Large batch size for throughput (cap if OOM)
arc corpus sync pdf-docs /path/to/pdfs --max-embedding-batch 500

# Opt into accelerator embedding
arc corpus sync pdf-docs /path/to/pdfs --gpu

# Verbose progress
arc corpus sync pdf-docs /path/to/pdfs --verbose

# JSON output for scripting
arc corpus sync pdf-docs /path/to/pdfs --json > results.json
```

See [CLI Reference: Corpus Sync Options](cli-reference.md#corpus-sync-options)
for the full option list.

## Single-System PDF Indexing (Advanced)

`arc index pdf` indexes PDFs to a Qdrant collection only (no MeiliSearch).
Prefer `arc corpus sync` unless you deliberately want Qdrant-only indexing,
or you need fine-grained OCR control (`--no-ocr`, `--ocr-language`,
`--ocr-workers`, `--normalize-only`).

### Basic Command

```bash
arc index pdf <directory> --collection <name>
```

The embedding model is set at collection creation time via
`arc collection create <name> --type pdf --model <model>`; the old `--model`
flag on `arc index pdf` is deprecated.

### Options

**Basic Options:**

- `--collection`: Target Qdrant collection (required)
- `--force`: Force reindex all files (bypass incremental sync)
- `--gpu`: Opt into accelerator embedding (CPU is the stable default)
- `--no-streaming`: Disable streaming mode (accumulate all embeddings before upload)
- `--verbose`: Verbose output (show progress, suppress library warnings)
- `--debug`: Debug mode (show all library warnings including transformers)
- `--json`: Output JSON format

**Performance Tuning:**

- `--embedding-batch-size`: Batch size for embedding generation (default: auto-tuned)
- `--process-priority`: Process scheduling priority (low/normal/high) [default: normal]

**OCR Options:**

- `--no-ocr`: Disable OCR (enabled by default for scanned PDFs)
- `--ocr-language`: OCR language code (eng, fra, spa, deu, etc.) [default: eng]
- `--ocr-workers`: Number of parallel OCR workers for page processing [default: cpu_count]
- `--normalize-only`: Skip markdown conversion, only normalize whitespace

### Examples

**Index technical documentation:**

```bash
arc index pdf ./docs --collection tech-docs
```

**Index scanned books (OCR enabled by default with parallel page processing):**

```bash
arc index pdf ./books \
  --collection book-archive \
  --ocr-language eng \
  --ocr-workers 8
```

**Force reindex all PDFs:**

```bash
arc index pdf ./pdfs --collection pdf-docs --force
```

**JSON output for scripting:**

```bash
arc index pdf ./pdfs --collection pdf-docs --json > results.json
```

**Opt into accelerator embedding:**

```bash
arc index pdf ./pdfs --collection pdf-docs --gpu
```

**Debug mode (show all warnings):**

```bash
arc index pdf ./pdfs --collection pdf-docs --debug
```

**Maximum performance (large batch, low priority):**

```bash
arc index pdf ./pdfs \
  --collection pdf-docs \
  --embedding-batch-size 500 \
  --process-priority low
```

**Non-streaming mode (accumulate all embeddings before upload):**

```bash
arc index pdf ./pdfs --collection pdf-docs --no-streaming
```

**Conservative (low priority, good for background processing):**

```bash
arc index pdf ./pdfs --collection pdf-docs --process-priority low
```

## Simplified CLI Scripts

For convenience, use the `bin/arc` wrapper during development:

```bash
# Development mode (from repository root)
bin/arc corpus sync pdf-docs /path/to/pdfs

# After pip install
arc corpus sync pdf-docs /path/to/pdfs

# Full test script
./scripts/test-pdf-indexing.sh
```

## Incremental Indexing

The system automatically tracks indexed files using metadata queries:

- **First run**: All PDFs are indexed
- **Subsequent runs**: Only new or modified PDFs are processed
- **Detection**: Based on file path and content hash (SHA256)
- **Rename / removal**: Add `--parity` to `arc corpus sync` to detect
  renamed/moved files and remove indexed entries for files no longer on disk
- **Duplicate content**: Byte-identical PDFs share one canonical searchable
  chunk set. Every physical path remains in the manifest as an alias.
- **Alias lifecycle**: Renaming or removing an alias preserves the shared
  content. If the canonical path is removed, parity sync deterministically
  promotes the lexicographically first live alias and verifies both indexes
  before deleting the old manifest.

To bypass incremental sync and reindex everything, use `--force`.

Incremental sync reports stale or missing indexing policy metadata without
selecting unchanged PDFs. Opt into that one-time policy migration explicitly:

```bash
arc corpus sync pdf-docs ./pdfs --include-stale-policy
```

Source changes are always re-indexed; this option only controls whether policy
staleness by itself selects an otherwise unchanged file.

To repair verifier-selected unhealthy PDFs—including corrupt extraction,
incomplete/duplicate chunks, and duplicate sources—run:

```bash
arc corpus repair pdf-docs --dry-run --json  # Preview exact files
arc corpus repair pdf-docs                   # Repair that selection
```

Files whose only finding is stale or missing indexing policy are reported but
not re-indexed by default. Include them when intentionally migrating policies:

```bash
arc corpus repair pdf-docs --include-stale-policy
```

PDFs that are both policy-stale and corrupt remain in the default repair set.
Healthy PDFs are not re-indexed; no separate `--only-corrupt` option is needed.

## Extraction Quality and Provenance

Arcaneum evaluates PDF extraction per page. When raster OCR clearly scores
better than embedded text, only the OCR candidate enters the searchable body for
that page; healthy embedded Markdown remains selected elsewhere. Rejected text
is represented by candidate scores and provenance instead of being appended to
the indexed text.

Each indexed PDF has a quality manifest containing:

- covered, empty, and low-text pages
- OCR trigger, confidence, failures, and selected extraction candidates
- omitted replacement-character counts and fidelity warnings
- extraction and chunking policy identities

Content-identical PDFs retain every physical source path as provenance for one
canonical searchable document; verification reports those aliases alongside the
quality evidence.

Standard verification reads this persisted evidence without re-extracting every
source file:

```bash
arc corpus verify pdf-docs
arc corpus verify pdf-docs --json
```

Verification reports incomplete or duplicated chunks, extraction dropout,
quality-manifest gaps, duplicate sources, and stale policies. Use `repair
--dry-run` to inspect the actionable selection before re-indexing.

## GPU Acceleration

CPU embedding is the stable default. `--gpu` requests an eligible experimental
PyTorch MPS/CUDA or FastEmbed/CoreML backend; it does not guarantee placement or
improved throughput. MLX is unavailable. PDF layout analysis itself runs in a
separate spawned process and is not controlled by the embedding backend.

Use acceleration only when you can observe the run and accept a worker reap plus
CPU fallback. See [Embedding acceleration and PDF layout workers](accelerators.md)
for current model/platform states, verbose diagnostics, safety limits, and
checked-in benchmark evidence.

```bash
# Opt into GPU for corpus sync
arc corpus sync docs ./pdfs --gpu

# Same for single-system (Qdrant-only) indexing
arc index pdf ./pdfs --collection docs --gpu
```

## Model Selection

Choose the embedding model based on your use case:

| Model           | Best For                     | Chunk Size | Late Chunking | GPU Support          |
| --------------- | ---------------------------- | ---------- | ------------- | -------------------- |
| **arctic-m**    | Stable default for PDFs/docs | 460 tokens | No            | CPU/FastEmbed        |
| **qwen3-embed** | High-quality documents       | 768 tokens | Yes           | MPS                  |
| **mxbai-large** | High-quality FastEmbed docs  | 460 tokens | No            | CPU/FastEmbed        |
| **bge**         | Legacy BGE documents         | 460 tokens | No            | Experimental CoreML  |

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

If you see errors like `MPS backend out of memory` when using large models like `jina-code-1.5b` (1.5B params):

```text
RuntimeError: MPS backend out of memory (MPS allocated: 12.25 GiB...)
```

The system uses adaptive batch sizes based on model size, but if you still hit memory limits:

1. **Use the stable default**: `arctic-m` is the default for document corpora
2. **Stay on CPU**: CPU mode is the default; omit `--gpu`
3. **Close other apps**: Free up GPU memory used by other applications
4. **Reduce GPU batch size**: add `--gpu --embedding-batch-size 100`

**Model memory requirements (approximate on MPS):**

| Model            | Size        | Memory Usage |
| ---------------- | ----------- | ------------ |
| `nomic-code`     | 7B params   | ~20+ GB      |
| `jina-code-1.5b` | 1.5B params | ~12-15 GB    |
| `qwen3-embed`    | 0.6B params | ~4-6 GB      |
| `jina-code-0.5b` | 500M params | ~4-6 GB      |
| `jina-code`      | 137M params | ~2-3 GB      |
| `minilm`         | 22M params  | <1 GB        |

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
    "qwen3-embed": ModelConfig(
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

### Quick Start (Corpus - Recommended)

The easiest way to get both semantic and full-text search is to use corpus:

```bash
# Create corpus and sync (indexes to both systems)
arc corpus create pdf-docs --type pdf
arc corpus sync pdf-docs /path/to/pdfs

# Both search types work
arc search semantic "machine learning concepts" --corpus pdf-docs
arc search text '"neural network"' --corpus pdf-docs
```

### Quick Start (MeiliSearch Only, Advanced)

**Prefer `arc corpus sync` for normal use** (see above) — it gives you both
semantic and full-text search. Use the MeiliSearch-only path below only if
you explicitly don't want Qdrant:

```bash
# Create MeiliSearch index
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
arc corpus create my-papers --type pdf

# Index to both systems in one command
arc corpus sync my-papers /path/to/pdfs

# Sync multiple directories at once
arc corpus sync my-papers /path/to/pdfs /path/to/more/pdfs

# Search both systems using --corpus flag
arc search semantic "machine learning" --corpus my-papers    # Qdrant
arc search text '"neural network"' --corpus my-papers        # MeiliSearch
```

**Using Existing Collection/Index as a Corpus:**

If you already have a Qdrant collection and MeiliSearch index with the same name,
you can use `corpus sync` directly without running `corpus create`:

```bash
# If 'Papers' collection and 'Papers' index already exist:
arc corpus sync Papers /path/to/pdfs
```

The only requirement is that both the collection and index exist with the same name.
The `corpus create` command is just a convenience that creates both in one step.

#### Using Separate Commands (Advanced)

If you need fine-grained control, you can manage Qdrant and MeiliSearch separately:

```bash
# Step 1: Create collections/indexes
arc collection create pdf-docs --type pdf      # Qdrant collection
arc indexes create pdf-docs --type pdf         # MeiliSearch index

# Step 2: Index to Qdrant (semantic search)
arc index pdf /path/to/pdfs --collection pdf-docs

# Step 3: Index to MeiliSearch (full-text search)
arc index text pdf /path/to/pdfs --index pdf-docs

# Search semantically (conceptual matches)
arc search semantic "machine learning techniques" --corpus pdf-docs

# Search exact phrases (keyword matches)
arc search text '"neural network architecture"' --corpus pdf-docs
```

### Change Detection

The full-text indexer tracks indexed files using SHA-256 file hashes:

- **First run**: All PDFs are indexed
- **Subsequent runs**: Only new or modified PDFs are processed
- **Detection**: Based on file path and content hash

To bypass change detection and reindex everything, use `--force`.

### Document Schema

Each page is indexed as a separate document with the following fields:

| Field               | Type   | Description                                      |
| ------------------- | ------ | ------------------------------------------------ |
| `id`                | string | Unique document ID (filename + path hash + page) |
| `content`           | string | Page text content (searchable)                   |
| `filename`          | string | PDF filename (searchable)                        |
| `file_path`         | string | Absolute file path (filterable)                  |
| `page_number`       | int    | Page number (filterable, sortable)               |
| `file_hash`         | string | SHA-256 hash for change detection (filterable)   |
| `extraction_method` | string | How text was extracted (filterable)              |
| `is_image_pdf`      | bool   | Whether OCR was used (filterable)                |

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

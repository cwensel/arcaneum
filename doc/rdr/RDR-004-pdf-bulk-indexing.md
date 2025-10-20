# Recommendation 004: Bulk PDF Indexing with OCR Support

## Metadata
- **Date**: 2025-10-19
- **Status**: Recommendation
- **Type**: Feature
- **Priority**: High
- **Related Issues**: arcaneum-4
- **Related Tests**: PDF extraction tests, OCR accuracy tests, batch upload integration tests

## Problem Statement

Create a production-ready bulk PDF indexing system for Qdrant that handles text PDFs, image PDFs (scanned documents), and mixed PDFs with optimal chunking for multiple embedding models. The system must:

1. **Extract text efficiently** from machine-generated PDFs (PyMuPDF, ~95x faster than alternatives)
2. **Handle scanned documents** with OCR (Tesseract for printed text, EasyOCR for mixed content)
3. **Chunk intelligently** with model-specific token limits and semantic awareness
4. **Upload efficiently** with batching, retry logic, and resumability
5. **Integrate seamlessly** with RDR-002 (Qdrant server) and RDR-003 (collection management)

This addresses the need to index large PDF repositories (technical documentation, research papers, books, scanned archives) for semantic search across the Arcaneum ecosystem.

## Context

### Background

Arcaneum requires a robust PDF indexing pipeline adapted from the proven `chroma-embedded/upload.sh` implementation. The system must handle diverse PDF types:

- **Text PDFs**: Machine-generated documents with embedded text layers (80% of use cases)
- **Image PDFs**: Scanned documents requiring OCR (15% of use cases)
- **Mixed PDFs**: Documents with both text and scanned images (5% of use cases)
- **Complex layouts**: Multi-column formats, tables, headers/footers

**Design Goals** (from arcaneum-4):
- PyMuPDF vs pdfplumber for text extraction?
- Tesseract vs EasyOCR for image PDFs?
- When to trigger OCR (threshold for "no text")?
- Chunking strategy - token-aware with model-specific sizing?
- Batch upload size for Qdrant (100-200 chunks)?
- Error handling for corrupt PDFs?

**Reference Implementation**: The `chroma-embedded/upload.sh` script (lines 1372-1522 for PDF extraction, lines 269-324 for token-optimized chunking) provides battle-tested patterns for ChromaDB that we adapt for Qdrant.

### Technical Environment

- **Python**: >= 3.12
- **Qdrant**: v1.15.4+ (from RDR-002)
- **PDF Libraries**:
  - PyMuPDF (fitz) >= 1.23.0 - Primary text extraction
  - pdfplumber >= 0.10.0 - Table extraction fallback
- **OCR Engines**:
  - Tesseract 5.x with pytesseract - Default OCR
  - EasyOCR >= 1.7.0 - Alternative for mixed content
- **Embedding**:
  - FastEmbed >= 0.3.0 (from RDR-003)
  - qdrant-client[fastembed] >= 1.15.0
- **Supporting Libraries**:
  - pdf2image (for OCR preprocessing)
  - opencv-python-headless (image preprocessing)
  - tenacity (retry logic)
  - tqdm/rich (progress tracking)

**Target Embedding Models** (from RDR-003):
- stella_en_1.5B_v5: 1024D, 512-1024 token chunks
- modernbert-base: 768D, 1024-2048 token chunks
- bge-large-en-v1.5: 1024D, 256-512 token chunks (hard limit)
- jina-embeddings-v3: 1024D, 1024-2048 token chunks

## Research Findings

### Investigation Process

**Nine parallel research tracks** were completed to inform this RDR:

1. **Prior RDR Analysis**: Reviewed RDR-001 (project structure), RDR-002 (Qdrant setup), RDR-003 (collection creation) for consistency and integration patterns
2. **ChromaDB Pattern Analysis**: Deep analysis of `chroma-embedded/upload.sh` PDF extraction (lines 1372-1522) and chunking logic (lines 269-324)
3. **Requirements Review**: Analyzed `outstar-rag-requirements.md` (lines 136-167) for PDF specifications
4. **PyMuPDF Research**: Opensource code explorer agent analyzed text extraction capabilities, performance, table handling
5. **pdfplumber Research**: Opensource code explorer agent evaluated table extraction, when to use vs PyMuPDF
6. **Tesseract OCR Research**: Comprehensive study of installation, language support, accuracy, confidence scoring, integration patterns
7. **EasyOCR Research**: Detailed comparison with Tesseract, GPU requirements, use case recommendations
8. **Embedding Model Research**: Token limits, optimal chunk sizes, character-to-token ratios, late chunking technique
9. **Qdrant Batch Upload Research**: Optimal batch sizes, error handling, retry strategies, performance characteristics

### Key Discoveries

#### 1. PDF Extraction Libraries

**PyMuPDF (Primary)**:
- **Speed**: 95x faster than pdfplumber (0.003s vs 0.1s per page)
- **Memory**: Low footprint (~100-300MB for moderate workloads)
- **Accuracy**: Excellent for machine-generated PDFs
- **Table support**: Basic table extraction, adequate for simple layouts
- **Use case**: 95% of PDF extraction tasks

**pdfplumber (Fallback)**:
- **Speed**: Slower but comprehensive (0.1s per page)
- **Table extraction**: Superior algorithm for complex tables
- **Layout preservation**: Better for structured data
- **Use case**: Complex tables, forms, invoices (5% of tasks)

**Decision**: Use PyMuPDF as primary extractor for speed, pdfplumber as fallback for documents where PyMuPDF fails table extraction validation.

#### 2. OCR Engines

**Tesseract 5.x (Default)**:
- **Accuracy**: 99%+ on clean printed text at 300 DPI
- **Speed**: 0.5 pages/second (CPU), 2s per page typical
- **Languages**: 100+ supported (eng, fra, spa, deu, ara, etc.)
- **Dependencies**: System binary + pytesseract wrapper
- **Installation**:
  - Linux: `apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`
  - Windows: Binary installer
- **Confidence scoring**: 0-100 range, 60+ threshold recommended
- **Best for**: High-quality scanned documents, CPU-only environments

**EasyOCR (Alternative)**:
- **Accuracy**: Better on handwritten/noisy images, 95%+ on varied content
- **Speed**: 4x faster than Tesseract on GPU, slower on CPU (20-30s per operation)
- **Languages**: 80+ supported
- **Dependencies**: PyTorch (~7GB with dependencies)
- **GPU requirement**: NVIDIA GPU with CUDA for performance benefit
- **Confidence scoring**: 0.0-1.0 range, 0.5-0.6 threshold typical
- **Best for**: Mixed content (handwritten + printed), noisy scans, GPU available

**Decision**: Tesseract as default (CPU-friendly, proven accuracy), EasyOCR as optional alternative for mixed-content PDFs with GPU acceleration.

#### 3. OCR Trigger Logic

From ChromaDB analysis (lines 1404-1433):
```python
# Trigger OCR if extracted text is empty or minimal
if store_type == 'pdf' and not text.strip():
    # < 100 characters threshold from chroma-embedded
    if ocr_enabled.lower() == 'true':
        # Proceed with OCR
```

**Recommendation**: Trigger OCR when extracted text < 100 characters. This threshold balances false negatives (missing text on sparse pages) with unnecessary OCR overhead.

#### 4. Embedding Model Token Limits (Validated from RDR-003)

| Model | Token Limit | Optimal Chunk Size | Overlap (15%) | Char-to-Token Ratio | Validation |
|-------|-------------|-------------------|---------------|---------------------|------------|
| **stella_en_1.5B_v5** | 512 (train), 8192 (capable) | 512-1024 | 77-154 | 3.2-3.5 | ✅ Accurate |
| **modernbert-base** | 8192 (native) | 1024-2048 | 154-307 | 3.4-3.5 | ✅ Accurate |
| **bge-large-en-v1.5** | 512 (hard limit) | 256-512 (460 safe) | 38-77 | 3.3-3.5 | ✅ Accurate |
| **jina-embeddings-v3** | 8192 | 1024-2048 | 154-307 | 3.2-3.8 | ✅ Accurate |

**Key Update from RDR-003**: Overlap should be **15% (not 10%)** based on NVIDIA 2025 research showing 15% overlap performed best on FinanceBench benchmark with 1,024 token chunks.

#### 5. Late Chunking: 2025 Breakthrough Technique

**What is Late Chunking**:
- Apply transformer layer to **entire document** (or as much as fits in context window)
- Apply mean pooling to **chunks of the token sequence** after embedding
- Creates **contextual chunk embeddings** that consider surrounding content

**Performance**:
- **nDCG improvements**: Significant gains on long documents (>2,000 tokens)
- **Correlation**: Longer documents = greater improvement
- **Requirement**: Document must fit in model's context window (8,192 tokens for stella/modernbert/jina-v3)

**Model Support**:
- ✅ **stella_en_1.5B_v5**: Compatible (8K context, custom implementation needed)
- ✅ **modernbert-base**: Compatible (8K context, custom implementation needed)
- ❌ **bge-large-en-v1.5**: Not suitable (512 token limit)
- ✅ **jina-embeddings-v3**: Native support (`late_chunking=True` API parameter)

**Recommendation**: Implement late chunking for documents between 2,000-8,000 tokens when using stella, modernbert, or jina-v3. For longer documents (>8K tokens), split into ~6,000 token sections and apply late chunking to each section.

#### 6. PDF-Specific Chunking Considerations

**Challenge**: PDFs often have poor logical structure:
- Multi-column layouts
- Headers/footers/page numbers mixed with content
- Tables and embedded images
- Formatting artifacts

**Best Practice (2025)**:
1. **Convert PDF to Markdown first** using tools like pymupdf4llm
2. **Apply semantic chunking** with heading-awareness
3. **Use RecursiveCharacterTextSplitter** or MarkdownTextSplitter
4. **Preserve document structure** (sections, paragraphs)
5. **Remove artifacts** (headers, footers, page numbers) before chunking

**Alternative**: If Markdown conversion fails or introduces errors, fall back to token-aware chunking with 15% overlap directly on extracted text.

#### 7. Qdrant Batch Upload (Validated from RDR-003)

**Optimal Batch Size**: 100-200 chunks per batch
- **Validation**: ✅ Falls within Qdrant's recommended 100-1,000 range
- **Comparison**: 2-4x larger than ChromaDB's 50-250 limit
- **Performance**: ~333 chunks/second sequential, ~1,111 chunks/second with 4 parallel workers

**Error Handling**:
- **Built-in retry**: Qdrant client has `max_retries=3` with exponential backoff
- **Rate limiting**: Server provides `retry_after_s` value for adaptive delays
- **External retry**: Use Tenacity library for transient network errors

**Resumability**:
- **SQLite checkpoint DB**: Track completed batches for crash recovery
- **Batch-level granularity**: Resume from last completed batch
- **Pattern**: Same as ChromaDB (proven in production)

**Memory Optimization**:
- Disable HNSW indexing during bulk upload (`m=0`)
- Re-enable after upload completes (`m=16`)
- Use `on_disk=True` for vectors to reduce RAM 4x

#### 8. Character-to-Token Ratio Validation

From RDR-003 and new research:

| Model | RDR-003 Value | Research Finding | Assessment |
|-------|--------------|------------------|------------|
| stella | 3.2 | 3.5-4.0 | ✅ Conservative (safe) |
| modernbert | 3.4 | 3.5-4.5 | ✅ Reasonable |
| bge-large | 3.3 | 3.5-4.0 | ✅ Accurate |
| jina-v3 | 3.2 | 3.2-3.8 | ✅ Accurate |

**Recommendation**: Keep RDR-003 values (conservative estimates prevent token limit overruns). Optionally increase to 3.5-4.0 after empirical validation with actual PDFs.

#### 9. ChromaDB to Qdrant Migration Insights

From `chroma-embedded/upload.sh` analysis:

**What Transfers Directly**:
- ✅ Same embedding models (stella, modernbert, bge-large)
- ✅ Same chunk sizes with 10% safety margins (update to 15%)
- ✅ Same store-specific adjustments (source-code, markdown, PDF)
- ✅ Same metadata schema patterns
- ✅ Same AST-aware chunking (for source code, not PDFs)

**What Changes for Qdrant**:
- 🔄 Batch size: Increase from 50 to 100-200
- 🔄 Embedding generation: Move to client-side (FastEmbed)
- 🔄 Point IDs: Use integers (not string hashes)
- 🔄 Filtering: Use Qdrant Filter API (different syntax)
- 🔄 Retry logic: Use Qdrant's built-in retry + Tenacity

**Performance Improvement**: Qdrant with parallelization is ~30x faster than original ChromaDB, ~7x faster than 2025 ChromaDB Rust-core rewrite.

## Proposed Solution

### Approach

**Five-Phase PDF Indexing Pipeline**:

```
Phase 1: PDF Extraction
├─ PyMuPDF primary extraction
├─ pdfplumber fallback for complex tables
└─ OCR trigger if text < 100 chars

Phase 2: OCR Processing (if needed)
├─ Tesseract (default, CPU-friendly)
├─ EasyOCR (alternative, GPU-accelerated)
├─ Confidence filtering (≥60 for Tesseract, ≥0.5 for EasyOCR)
└─ 2x image scaling for accuracy

Phase 3: Preprocessing & Chunking
├─ Convert to Markdown (preserve structure)
├─ Remove artifacts (headers, footers, page numbers)
├─ Semantic chunking with heading-awareness
├─ Late chunking for long documents (2K-8K tokens)
└─ 15% overlap between chunks

Phase 4: Embedding Generation
├─ FastEmbed with model-specific configs
├─ Batch embed 100-200 texts at once
├─ Model selection based on document type
└─ Named vectors for multi-model support

Phase 5: Batch Upload
├─ Upload 100-200 chunks per batch
├─ Parallel workers (4 recommended)
├─ Exponential backoff retry
├─ SQLite checkpoint for resumability
└─ Progress tracking (tqdm/Rich)
```

### Technical Design

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   PDF Indexing Pipeline                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: PDF Discovery & Extraction                        │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │  Glob PDFs  │──▶│  PyMuPDF    │──▶│ Validate    │       │
│  │  Recursively│   │  Extract    │   │ Text Quality│       │
│  └─────────────┘   └─────────────┘   └──────┬──────┘       │
│                           │                  │               │
│                           │ < 100 chars?     │ >= 100 chars │
│                           ▼                  ▼               │
│                    ┌─────────────┐   ┌─────────────┐       │
│                    │ pdfplumber  │   │  Continue   │       │
│                    │ Fallback?   │   │  to Phase 2 │       │
│                    └──────┬──────┘   └─────────────┘       │
│                           │ Still < 100?                    │
│                           ▼                                 │
│                    ┌─────────────┐                         │
│                    │ Trigger OCR │                         │
│                    └──────┬──────┘                         │
└───────────────────────────┼─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: OCR Processing (Conditional)                      │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │ pdf2image   │──▶│ Tesseract/  │──▶│ Confidence  │       │
│  │ Convert@300 │   │ EasyOCR     │   │ Filter ≥60% │       │
│  │ DPI         │   │ Recognition │   │             │       │
│  └─────────────┘   └─────────────┘   └──────┬──────┘       │
│        │                                     │               │
│        │ 2x image scaling                    │ Filtered text│
│        ▼                                     ▼               │
│  ┌─────────────┐                     ┌─────────────┐       │
│  │ OpenCV      │                     │  Merge with │       │
│  │ Preprocess  │                     │  Extracted  │       │
│  └─────────────┘                     └──────┬──────┘       │
└───────────────────────────────────────────────┼─────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: Preprocessing & Chunking                          │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │ PDF → MD    │──▶│ Remove      │──▶│ Select      │       │
│  │ (pymupdf4   │   │ Artifacts   │   │ Chunking    │       │
│  │  llm)       │   │             │   │ Strategy    │       │
│  └─────────────┘   └─────────────┘   └──────┬──────┘       │
│                                              │               │
│                           ┌──────────────────┴────────┐     │
│                           │                           │     │
│                           ▼                           ▼     │
│                    ┌─────────────┐           ┌─────────────┐│
│                    │ Traditional │           │ Late        ││
│                    │ Chunking    │           │ Chunking    ││
│                    │ 15% overlap │           │ (2K-8K tok) ││
│                    └──────┬──────┘           └──────┬──────┘│
│                           └──────────┬───────────────┘      │
│                                      ▼                       │
└──────────────────────────────────────┼───────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 4: Embedding Generation                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │ FastEmbed   │──▶│ Batch Embed │──▶│ Create      │       │
│  │ Model Load  │   │ 100-200     │   │ Points with │       │
│  │ (cached)    │   │ chunks      │   │ Metadata    │       │
│  └─────────────┘   └─────────────┘   └──────┬──────┘       │
│                                              │               │
│  Model Selection:                            │               │
│  • stella (long docs)                        │               │
│  • bge-large (precision)                     ▼               │
│  • jina-v3 (multilingual)           ┌─────────────┐        │
│                                      │ Named       │        │
│                                      │ Vectors     │        │
│                                      │ (multi-     │        │
│                                      │  model)     │        │
│                                      └──────┬──────┘        │
└──────────────────────────────────────────────┼──────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 5: Batch Upload to Qdrant                            │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │ Batch       │──▶│ Upload with │──▶│ Checkpoint  │       │
│  │ 100-200     │   │ Retry       │   │ Progress    │       │
│  │ points      │   │ (Tenacity)  │   │ (SQLite)    │       │
│  └─────────────┘   └─────────────┘   └──────┬──────┘       │
│        │                                     │               │
│        │ parallel=4                          │               │
│        ▼                                     ▼               │
│  ┌─────────────┐                     ┌─────────────┐       │
│  │ 4 Workers   │                     │ Progress    │       │
│  │ 3-4x faster │                     │ Tracking    │       │
│  └─────────────┘                     │ (tqdm/Rich) │       │
│                                      └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

#### Configuration Schema (arcaneum.yaml)

```yaml
# PDF-specific collection configuration
collections:
  pdf-docs:
    models: [stella, bge]  # stella for long docs, bge for precision
    hnsw_m: 16
    hnsw_ef_construct: 100
    on_disk_payload: true
    indexes: [filename, file_path, page_count, extraction_method]

# PDF processing settings
pdf_processing:
  extraction:
    primary: pymupdf        # Primary extraction library
    fallback: pdfplumber    # Fallback for complex tables
    table_validation: true  # Validate table extraction quality

  ocr:
    enabled: true
    engine: tesseract       # tesseract | easyocr
    language: eng           # Language code (eng, fra, spa, etc.)
    trigger_threshold: 100  # Trigger OCR if text < N chars
    confidence_threshold: 60 # Minimum confidence (Tesseract: 0-100, EasyOCR: 0.0-1.0)
    image_dpi: 300         # DPI for PDF to image conversion
    image_scale: 2.0       # Scale factor for OCR accuracy (2x recommended)
    preprocessing:
      grayscale: true
      threshold: otsu      # otsu | adaptive
      denoise: true
      deskew: true

  preprocessing:
    convert_to_markdown: true  # Convert PDF to Markdown first
    remove_artifacts: true     # Remove headers/footers/page numbers
    preserve_structure: true   # Preserve headings and sections

  chunking:
    strategy: semantic_with_late_chunking  # traditional | semantic | semantic_with_late_chunking
    overlap_percent: 15       # 15% overlap (NVIDIA recommendation)
    late_chunking:
      enabled: true
      min_doc_tokens: 2000    # Minimum document length for late chunking
      max_doc_tokens: 8000    # Maximum (model context limit)
      section_size: 6000      # Split very long docs into sections

  upload:
    batch_size: 100           # Chunks per batch
    parallel_workers: 4       # Parallel upload workers
    max_retries: 5            # Maximum retry attempts
    checkpoint_enabled: true  # Enable SQLite checkpointing for crash recovery
    checkpoint_db: ./upload_checkpoint.db

# Model-specific settings (from RDR-003, updated with 15% overlap)
models:
  stella:
    name: BAAI/bge-large-en-v1.5
    dimensions: 1024
    chunk_size: 768          # Conservative for PDF (allows room for structure)
    chunk_overlap: 115       # 15% overlap
    distance: cosine
    late_chunking: true

  bge:
    name: BAAI/bge-large-en-v1.5
    dimensions: 1024
    chunk_size: 460          # Safe margin from 512 limit
    chunk_overlap: 69        # 15% overlap
    distance: cosine
    late_chunking: false     # Not supported (512 token limit)

  modernbert:
    name: answerdotai/ModernBERT-base
    dimensions: 768
    chunk_size: 1536         # Conservative for long context
    chunk_overlap: 230       # 15% overlap
    distance: cosine
    late_chunking: true

  jina:
    name: jinaai/jina-embeddings-v3
    dimensions: 1024
    chunk_size: 1536
    chunk_overlap: 230       # 15% overlap
    distance: cosine
    late_chunking: true      # Native API support
```

#### Module Structure

```
src/arcaneum/
├── indexing/
│   ├── __init__.py
│   ├── pdf/
│   │   ├── __init__.py
│   │   ├── extractor.py         # PDF text extraction (PyMuPDF + pdfplumber)
│   │   ├── ocr.py                # OCR integration (Tesseract + EasyOCR)
│   │   ├── preprocessor.py      # PDF to Markdown, artifact removal
│   │   └── chunker.py            # Chunking strategies (traditional + late)
│   ├── common/
│   │   ├── __init__.py
│   │   ├── checkpoint.py         # SQLite checkpoint management
│   │   ├── sync.py               # Metadata-based sync for incremental indexing
│   │   ├── progress.py           # Progress tracking (tqdm/Rich)
│   │   └── retry.py              # Retry logic with Tenacity
│   └── uploader.py               # Batch upload orchestrator
├── embeddings/
│   ├── __init__.py
│   ├── client.py                 # EmbeddingClient (from RDR-003)
│   └── late_chunking.py          # Late chunking implementation
└── cli/
    └── index_pdfs.py             # CLI command: arcaneum index pdfs
```

### Implementation Example

#### PDF Extraction Module

```python
# src/arcaneum/indexing/pdf/extractor.py
import pymupdf  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PDFExtractor:
    """Extract text from PDFs using PyMuPDF with pdfplumber fallback."""

    def __init__(self, fallback_enabled: bool = True, table_validation: bool = True):
        self.fallback_enabled = fallback_enabled
        self.table_validation = table_validation

    def extract(self, pdf_path: Path) -> Tuple[str, dict]:
        """
        Extract text from PDF.

        Returns:
            Tuple of (text, metadata)
            metadata includes: extraction_method, is_image_pdf, page_count
        """
        try:
            # Primary: PyMuPDF (95x faster)
            text, metadata = self._extract_with_pymupdf(pdf_path)

            # Validate extraction quality
            if self.table_validation and self._has_complex_tables(pdf_path):
                # Fallback to pdfplumber for table-heavy documents
                logger.info(f"Complex tables detected in {pdf_path.name}, using pdfplumber")
                text, metadata = self._extract_with_pdfplumber(pdf_path)

            return text, metadata

        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            raise

    def _extract_with_pymupdf(self, pdf_path: Path) -> Tuple[str, dict]:
        """Extract text using PyMuPDF (fast, general-purpose)."""
        text_parts = []

        with pymupdf.open(pdf_path) as doc:
            page_count = len(doc)

            for page_num, page in enumerate(doc):
                page_text = page.get_text(sort=True)  # Sort for reading order

                if page_text.strip():
                    text_parts.append(page_text)

        text = '\n'.join(text_parts)

        metadata = {
            'extraction_method': 'pymupdf',
            'is_image_pdf': False,
            'page_count': page_count,
            'file_size': pdf_path.stat().st_size,
        }

        return text, metadata

    def _extract_with_pdfplumber(self, pdf_path: Path) -> Tuple[str, dict]:
        """Extract text using pdfplumber (slower, better table handling)."""
        text_parts = []

        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)

            for page in pdf.pages:
                # Extract tables first
                tables = page.extract_tables()

                # Extract regular text
                page_text = page.extract_text(layout=True)  # Preserve layout

                # Combine text and tables
                if tables:
                    table_texts = [self._format_table(table) for table in tables]
                    page_text = page_text + '\n\n' + '\n\n'.join(table_texts)

                if page_text and page_text.strip():
                    text_parts.append(page_text)

        text = '\n'.join(text_parts)

        metadata = {
            'extraction_method': 'pdfplumber',
            'is_image_pdf': False,
            'page_count': page_count,
            'file_size': pdf_path.stat().st_size,
        }

        return text, metadata

    def _has_complex_tables(self, pdf_path: Path) -> bool:
        """Quick check if PDF has complex tables (heuristic)."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Check first page only (performance)
                if pdf.pages:
                    tables = pdf.pages[0].find_tables()
                    return len(tables) > 0
            return False
        except:
            return False

    def _format_table(self, table: list) -> str:
        """Format extracted table as Markdown."""
        if not table:
            return ""

        # Simple Markdown table formatting
        lines = []
        for row in table:
            lines.append("| " + " | ".join(str(cell) if cell else "" for cell in row) + " |")

        # Add header separator after first row
        if len(lines) > 1:
            lines.insert(1, "|" + "|".join([" --- " for _ in table[0]]) + "|")

        return '\n'.join(lines)
```

#### OCR Module

```python
# src/arcaneum/indexing/pdf/ocr.py
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class OCREngine:
    """OCR engine supporting Tesseract and EasyOCR."""

    def __init__(
        self,
        engine: str = 'tesseract',
        language: str = 'eng',
        confidence_threshold: float = 60.0,
        image_dpi: int = 300,
        image_scale: float = 2.0
    ):
        self.engine = engine
        self.language = language
        self.confidence_threshold = confidence_threshold
        self.image_dpi = image_dpi
        self.image_scale = image_scale

        if engine == 'easyocr':
            try:
                import easyocr
                self.reader = easyocr.Reader([language])
            except ImportError:
                logger.error("EasyOCR not installed. Install with: pip install easyocr")
                raise

    def process_pdf(self, pdf_path: Path) -> Tuple[str, dict]:
        """
        Perform OCR on PDF.

        Returns:
            Tuple of (text, metadata)
        """
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=self.image_dpi)

        text_parts = []
        confidence_scores = []

        for page_num, image in enumerate(images, 1):
            # Preprocess image
            processed_image = self._preprocess_image(image)

            # Perform OCR
            if self.engine == 'tesseract':
                page_text, page_conf = self._ocr_tesseract(processed_image)
            else:  # easyocr
                page_text, page_conf = self._ocr_easyocr(processed_image)

            if page_text.strip():
                text_parts.append(page_text)
                confidence_scores.append(page_conf)
                logger.debug(f"Page {page_num}: {len(page_text)} chars, confidence: {page_conf:.1f}")

        text = '\n'.join(text_parts)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        metadata = {
            'extraction_method': f'ocr_{self.engine}',
            'is_image_pdf': True,
            'ocr_confidence': avg_confidence,
            'ocr_language': self.language,
            'page_count': len(images),
        }

        logger.info(f"OCR completed: {len(text)} chars, avg confidence: {avg_confidence:.1f}%")

        return text, metadata

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for better OCR accuracy."""
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Scale image (2x recommended for accuracy)
        if self.image_scale != 1.0:
            width = int(img_array.shape[1] * self.image_scale)
            height = int(img_array.shape[0] * self.image_scale)
            img_array = cv2.resize(img_array, (width, height), interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply thresholding (Otsu's method)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Denoise
        denoised = cv2.medianBlur(thresh, 3)

        # Deskew (optional, can add if needed)
        # denoised = self._deskew(denoised)

        return denoised

    def _ocr_tesseract(self, image: np.ndarray) -> Tuple[str, float]:
        """Perform OCR with Tesseract."""
        try:
            # Get detailed data with confidence
            data = pytesseract.image_to_data(
                image,
                lang=self.language,
                config='--psm 3 --oem 1',  # Auto segmentation, LSTM engine
                output_type=pytesseract.Output.DICT
            )

            # Extract text with confidence filtering
            filtered_text = []
            confidences = []

            for i, conf in enumerate(data['conf']):
                if conf == -1:  # No text detected
                    continue
                if conf >= self.confidence_threshold:
                    text = data['text'][i]
                    if text.strip():
                        filtered_text.append(text)
                        confidences.append(conf)

            text = ' '.join(filtered_text)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0

            return text, avg_conf

        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            raise

    def _ocr_easyocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Perform OCR with EasyOCR."""
        try:
            # EasyOCR returns [(bbox, text, confidence), ...]
            results = self.reader.readtext(image, detail=1)

            # Filter by confidence and extract text
            filtered_text = []
            confidences = []

            for bbox, text, conf in results:
                if conf >= (self.confidence_threshold / 100.0):  # EasyOCR uses 0-1 scale
                    filtered_text.append(text)
                    confidences.append(conf * 100)  # Convert to percentage

            text = ' '.join(filtered_text)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0

            return text, avg_conf

        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            raise
```

#### Chunking Module with Late Chunking

```python
# src/arcaneum/indexing/pdf/chunker.py
from typing import List, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_index: int
    token_count: int
    metadata: Dict

class PDFChunker:
    """Chunk PDF text with semantic awareness and late chunking support."""

    def __init__(
        self,
        model_config: Dict,
        overlap_percent: float = 0.15,
        late_chunking_enabled: bool = True,
        min_doc_tokens: int = 2000,
        max_doc_tokens: int = 8000
    ):
        self.model_config = model_config
        self.overlap_percent = overlap_percent
        self.late_chunking_enabled = late_chunking_enabled
        self.min_doc_tokens = min_doc_tokens
        self.max_doc_tokens = max_doc_tokens

        self.chunk_size = model_config['chunk_size']
        self.chunk_overlap = int(self.chunk_size * overlap_percent)

    def chunk(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Chunk text using appropriate strategy.

        Strategies:
        1. Late chunking: For documents 2K-8K tokens (if supported by model)
        2. Semantic chunking: Preserve paragraph/section boundaries
        3. Traditional chunking: Token-aware splitting with overlap
        """
        # Estimate token count (rough approximation)
        char_to_token = self.model_config.get('char_to_token_ratio', 3.3)
        estimated_tokens = len(text) / char_to_token

        # Select chunking strategy
        if (self.late_chunking_enabled and
            self.model_config.get('late_chunking', False) and
            self.min_doc_tokens < estimated_tokens < self.max_doc_tokens):

            logger.info(f"Using late chunking (doc tokens: {estimated_tokens:.0f})")
            return self._late_chunking(text, metadata)

        else:
            logger.info(f"Using traditional chunking (doc tokens: {estimated_tokens:.0f})")
            return self._traditional_chunking(text, metadata)

    def _late_chunking(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Implement late chunking strategy.

        Note: This is a simplified example. Production implementation would:
        1. Embed entire document first
        2. Apply mean pooling to chunk-sized windows of token embeddings
        3. Return contextual chunk embeddings

        For jina-v3, use API parameter: late_chunking=True
        For stella/modernbert, implement custom mean pooling after embedding.
        """
        # For now, return traditional chunks with metadata flag
        # Actual late chunking happens in embedding phase
        chunks = self._traditional_chunking(text, metadata)

        # Mark chunks for late chunking processing
        for chunk in chunks:
            chunk.metadata['late_chunking'] = True

        return chunks

    def _traditional_chunking(self, text: str, metadata: Dict) -> List[Chunk]:
        """Traditional token-aware chunking with overlap."""
        chunks = []
        char_to_token = self.model_config.get('char_to_token_ratio', 3.3)

        # Calculate character limits
        chunk_chars = int(self.chunk_size * char_to_token)
        overlap_chars = int(self.chunk_overlap * char_to_token)

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_chars

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence boundary in last 20% of chunk
                search_start = end - int(chunk_chars * 0.2)
                sentence_end = text.rfind('. ', search_start, end)

                if sentence_end != -1:
                    end = sentence_end + 1  # Include the period

            chunk_text = text[start:end].strip()

            if chunk_text:
                # Estimate token count
                token_count = int(len(chunk_text) / char_to_token)

                chunk = Chunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    metadata={
                        **metadata,
                        'chunk_index': chunk_index,
                        'chunk_start_char': start,
                        'chunk_end_char': end,
                        'late_chunking': False,
                    }
                )

                chunks.append(chunk)
                chunk_index += 1

            # Move start position (with overlap)
            start = end - overlap_chars

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def _semantic_chunking(self, text: str, metadata: Dict) -> List[Chunk]:
        """
        Semantic chunking preserving paragraph/section boundaries.

        TODO: Implement with RecursiveCharacterTextSplitter or similar
        """
        # Placeholder: use traditional chunking for now
        return self._traditional_chunking(text, metadata)
```

#### Batch Upload Orchestrator

```python
# src/arcaneum/indexing/uploader.py
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

from ..embeddings.client import EmbeddingClient
from .common.checkpoint import BatchCheckpoint
from .common.sync import MetadataBasedSync, compute_file_hash
from .pdf.extractor import PDFExtractor
from .pdf.ocr import OCREngine
from .pdf.chunker import PDFChunker

logger = logging.getLogger(__name__)

class PDFBatchUploader:
    """Orchestrate bulk PDF indexing with batching and error recovery."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_client: EmbeddingClient,
        config: Dict
    ):
        self.qdrant = qdrant_client
        self.embeddings = embedding_client
        self.config = config

        # Initialize components
        self.extractor = PDFExtractor(
            fallback_enabled=config['pdf_processing']['extraction']['fallback'] == 'pdfplumber',
            table_validation=config['pdf_processing']['extraction'].get('table_validation', True)
        )

        self.ocr = OCREngine(
            engine=config['pdf_processing']['ocr']['engine'],
            language=config['pdf_processing']['ocr']['language'],
            confidence_threshold=config['pdf_processing']['ocr']['confidence_threshold'],
            image_dpi=config['pdf_processing']['ocr']['image_dpi'],
            image_scale=config['pdf_processing']['ocr']['image_scale']
        )

        self.checkpoint = BatchCheckpoint(
            config['pdf_processing']['upload']['checkpoint_db']
        ) if config['pdf_processing']['upload']['checkpoint_enabled'] else None

        # Metadata-based sync (queries Qdrant directly, no separate DB)
        self.sync = MetadataBasedSync(qdrant_client)

        self.batch_size = config['pdf_processing']['upload']['batch_size']
        self.parallel_workers = config['pdf_processing']['upload']['parallel_workers']
        self.max_retries = config['pdf_processing']['upload']['max_retries']

    def index_directory(
        self,
        pdf_dir: Path,
        collection_name: str,
        model_name: str,
        resume: bool = True,
        force_reindex: bool = False
    ):
        """
        Index PDFs in directory with incremental sync.

        Args:
            pdf_dir: Directory containing PDFs
            collection_name: Qdrant collection name
            model_name: Embedding model to use
            resume: Resume from checkpoint if crash occurred
            force_reindex: Bypass metadata sync and reindex all files
        """

        # Get model config
        model_config = self.config['models'][model_name]

        # Initialize chunker
        chunker = PDFChunker(
            model_config=model_config,
            overlap_percent=self.config['pdf_processing']['chunking']['overlap_percent'],
            late_chunking_enabled=self.config['pdf_processing']['chunking'].get('late_chunking', {}).get('enabled', True),
            min_doc_tokens=self.config['pdf_processing']['chunking'].get('late_chunking', {}).get('min_doc_tokens', 2000),
            max_doc_tokens=self.config['pdf_processing']['chunking'].get('late_chunking', {}).get('max_doc_tokens', 8000)
        )

        # Discover all PDFs
        all_pdf_files = sorted(pdf_dir.rglob("*.pdf"))
        logger.info(f"Found {len(all_pdf_files)} total PDF files")

        # Filter to unindexed files via metadata queries (unless force_reindex)
        if force_reindex:
            pdf_files = all_pdf_files
            logger.info(f"Force reindex: processing all {len(pdf_files)} PDFs")
        else:
            pdf_files = self.sync.get_unindexed_files(collection_name, all_pdf_files)
            skipped = len(all_pdf_files) - len(pdf_files)
            logger.info(f"Incremental sync: {len(pdf_files)} new/modified, {skipped} already indexed")

        # Resume from checkpoint (for crash recovery)
        last_batch_id = 0
        if resume and self.checkpoint:
            last_batch_id = self.checkpoint.get_last_completed_batch()
            logger.info(f"Resuming from batch {last_batch_id + 1}")

        # Process PDFs
        batch = []
        batch_id = last_batch_id + 1
        point_id = 0
        stats = {"files": 0, "chunks": 0, "errors": 0}

        with tqdm(total=len(pdf_files), desc="PDFs", unit="file") as file_pbar:
            with tqdm(desc="Chunks", unit="chunk") as chunk_pbar:

                for pdf_path in pdf_files:
                    try:
                        # Compute file hash for incremental indexing
                        file_hash = compute_file_hash(pdf_path)

                        # Extract text
                        text, extract_meta = self.extractor.extract(pdf_path)

                        # Check if OCR needed
                        if (self.config['pdf_processing']['ocr']['enabled'] and
                            len(text) < self.config['pdf_processing']['ocr']['trigger_threshold']):

                            logger.info(f"Triggering OCR for {pdf_path.name} (text: {len(text)} chars)")
                            text, ocr_meta = self.ocr.process_pdf(pdf_path)
                            extract_meta.update(ocr_meta)

                        # Chunk text with file metadata (including hash for sync)
                        base_metadata = {
                            'filename': pdf_path.name,
                            'file_path': str(pdf_path),
                            'file_hash': file_hash,  # For incremental sync
                            'file_size': pdf_path.stat().st_size,
                            'store_type': 'pdf',
                            **extract_meta
                        }

                        chunks = chunker.chunk(text, base_metadata)

                        # Generate embeddings
                        texts = [chunk.text for chunk in chunks]
                        embeddings = self.embeddings.embed(texts, model_name)

                        # Create points
                        for chunk, embedding in zip(chunks, embeddings):
                            point = PointStruct(
                                id=point_id,
                                vector=embedding,
                                payload={
                                    'text': chunk.text,
                                    **chunk.metadata
                                }
                            )
                            batch.append(point)
                            point_id += 1
                            chunk_pbar.update(1)

                            # Upload when batch full
                            if len(batch) >= self.batch_size:
                                self._upload_batch(collection_name, batch, batch_id)

                                if self.checkpoint:
                                    self.checkpoint.mark_completed(
                                        batch_id,
                                        str(pdf_path),
                                        (batch[0].id, batch[-1].id)
                                    )

                                stats["chunks"] += len(batch)
                                batch = []
                                batch_id += 1

                        stats["files"] += 1
                        file_pbar.update(1)

                    except Exception as e:
                        logger.error(f"Failed to process {pdf_path}: {e}")
                        stats["errors"] += 1
                        file_pbar.update(1)
                        continue

                # Upload remaining batch
                if batch:
                    self._upload_batch(collection_name, batch, batch_id)
                    if self.checkpoint:
                        self.checkpoint.mark_completed(
                            batch_id,
                            "final_batch",
                            (batch[0].id, batch[-1].id)
                        )
                    stats["chunks"] += len(batch)

        # Final report
        logger.info("=" * 60)
        logger.info("INDEXING COMPLETE")
        logger.info(f"Files processed: {stats['files']}")
        logger.info(f"Chunks uploaded: {stats['chunks']}")
        logger.info(f"Errors: {stats['errors']}")
        logger.info("=" * 60)

        return stats

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True
    )
    def _upload_batch(self, collection_name: str, points: List[PointStruct], batch_id: int):
        """Upload batch with exponential backoff retry."""
        try:
            result = self.qdrant.upload_points(
                collection_name=collection_name,
                points=points,
                batch_size=self.batch_size,
                parallel=self.parallel_workers,
                max_retries=3,  # Inner retry for rate limiting
                wait=True
            )

            logger.debug(f"Batch {batch_id} uploaded: {len(points)} points")
            return result

        except Exception as e:
            logger.error(f"Batch {batch_id} upload failed: {e}")
            raise
```

## Alternatives Considered

### Alternative 1: PyMuPDF Only (No Fallback)

**Description**: Use PyMuPDF exclusively for all PDF extraction, no pdfplumber fallback.

**Pros**:
- Simpler implementation
- Faster (no fallback overhead)
- Fewer dependencies

**Cons**:
- Poor table extraction quality on complex documents
- No recovery for PyMuPDF edge cases
- May miss structured data in forms/invoices

**Reason for rejection**: Table extraction is critical for many PDF types (research papers, financial reports, technical documentation). The 5% fallback overhead is acceptable for better quality.

### Alternative 2: Always Use OCR

**Description**: Apply OCR to all PDFs regardless of embedded text quality.

**Pros**:
- Consistent processing pipeline
- May catch missed text in hybrid PDFs
- Validates embedded text accuracy

**Cons**:
- 20-30s overhead per page (massive slowdown)
- Unnecessary for 80% of PDFs with good text layers
- Higher computational cost (CPU/GPU)
- Lower accuracy than embedded text

**Reason for rejection**: The 80/20 rule applies - most PDFs have excellent embedded text. OCR should be conditional on quality threshold.

### Alternative 3: Fixed Chunking (No Late Chunking)

**Description**: Use only traditional token-aware chunking with 10-15% overlap.

**Pros**:
- Simpler implementation
- No model context length concerns
- Well-understood technique

**Cons**:
- Misses contextual information across chunk boundaries
- Lower retrieval quality on long documents
- Doesn't leverage 2025 breakthrough research
- No benefit from long-context models

**Reason for rejection**: Late chunking shows significant nDCG improvements (5-8%) on long documents. The implementation complexity is manageable (conditional strategy selection).

### Alternative 4: ChromaDB Patterns Unchanged

**Description**: Keep ChromaDB patterns exactly as-is (50 batch size, manual retry, no parallelization).

**Pros**:
- Minimal adaptation work
- Proven in production
- Low risk

**Cons**:
- Misses Qdrant-specific optimizations
- 2-4x slower uploads (50 vs 100-200 batch size)
- No built-in retry benefit
- No parallel upload benefit

**Reason for rejection**: Qdrant handles larger batches better, has built-in retry, and supports parallelization. Adapting patterns yields 3-4x performance improvement.

### Alternative 5: Markdown-Only Chunking

**Description**: Always convert PDF to Markdown before chunking, never fall back to raw text chunking.

**Pros**:
- Cleanest semantic structure
- Best heading-aware chunking
- Removes all formatting artifacts

**Cons**:
- Markdown conversion may fail on complex layouts
- Additional processing overhead
- May lose information in conversion
- Single point of failure

**Reason for rejection**: Markdown conversion is valuable but shouldn't be mandatory. Provide it as primary strategy with fallback to raw text chunking.

## Trade-offs and Consequences

### Positive Consequences

1. **Extraction Speed**: PyMuPDF provides 95x faster extraction than pdfplumber (0.003s vs 0.1s per page), enabling high-throughput indexing
2. **OCR Flexibility**: Tesseract (CPU) and EasyOCR (GPU) options cover different deployment scenarios and document types
3. **Quality-Driven OCR**: Conditional OCR triggering (< 100 chars threshold) avoids unnecessary overhead on 80% of PDFs
4. **Optimal Chunking**: 15% overlap (vs 10%) and late chunking improve retrieval quality by 5-8% on long documents
5. **Batch Efficiency**: 100-200 batch size with 4 parallel workers delivers 3-4x faster uploads than ChromaDB patterns
6. **Resumability**: SQLite checkpointing enables crash recovery for long-running jobs
7. **Multi-Model Support**: Named vectors architecture (from RDR-002/003) allows different embedding models per use case
8. **Production-Ready**: Exponential backoff retry, progress tracking, and error handling patterns from battle-tested implementations

### Negative Consequences

1. **Complexity**: Multiple libraries (PyMuPDF, pdfplumber, Tesseract, EasyOCR) increase surface area
2. **OCR Overhead**: Scanned PDFs require 20-30s per page (vs 0.003s for text PDFs)
3. **Memory Requirements**: FastEmbed models (~3-4GB) + Qdrant RAM (~1-6GB depending on vector count)
4. **CPU-Bound**: Tesseract OCR and FastEmbed embedding are CPU-intensive (EasyOCR requires GPU for speed)
5. **Late Chunking Complexity**: Conditional strategy selection adds logic complexity
6. **Dependency Count**: 10+ Python packages plus system dependencies (Tesseract, Poppler)

### Risks and Mitigations

- **Risk**: PyMuPDF fails on corrupted PDFs
  **Mitigation**: Wrap extraction in try-except, log errors, skip corrupted files with error report at end

- **Risk**: OCR misreads scanned text (< 60% confidence)
  **Mitigation**: Confidence filtering removes low-quality text, manual review queue for borderline cases

- **Risk**: Late chunking implementation bugs
  **Mitigation**: Make late chunking optional (config flag), fall back to traditional chunking on errors

- **Risk**: Batch upload exhausts memory with large batches
  **Mitigation**: Disable HNSW indexing during upload (`m=0`), use `on_disk=True` for vectors

- **Risk**: Character-to-token ratio inaccuracy causes chunk overflow
  **Mitigation**: Use conservative RDR-003 values (3.2-3.4), validate empirically, add 10% safety buffer

- **Risk**: Checkpoint database corruption
  **Mitigation**: SQLite is robust; add checkpoint backup before each run, validate checkpoint integrity on load

## Implementation Plan

### Prerequisites

- [x] RDR-001: Project structure established
- [x] RDR-002: Qdrant server running (http://localhost:6333)
- [x] RDR-003: Collection creation CLI tool (`arcaneum collection create`)
- [ ] Python 3.12+ installed
- [ ] System dependencies:
  - Tesseract 5.x (`apt-get install tesseract-ocr` or `brew install tesseract`)
  - Poppler (`apt-get install poppler-utils` or `brew install poppler`)
- [ ] Python packages:
  - pymupdf >= 1.23.0
  - pdfplumber >= 0.10.0
  - pytesseract (+ optional: easyocr)
  - pdf2image
  - opencv-python-headless
  - tenacity
  - tqdm

### Step-by-Step Implementation

#### Step 1: Install Dependencies

```bash
# System dependencies (Ubuntu/Debian)
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils

# Python dependencies
pip install pymupdf pdfplumber pytesseract pdf2image opencv-python-headless tenacity tqdm

# Optional: EasyOCR (if GPU available)
pip install easyocr
```

#### Step 2: Create PDF Indexing Module Structure

```bash
mkdir -p src/arcaneum/indexing/pdf
mkdir -p src/arcaneum/indexing/common
touch src/arcaneum/indexing/__init__.py
touch src/arcaneum/indexing/pdf/__init__.py
touch src/arcaneum/indexing/pdf/extractor.py
touch src/arcaneum/indexing/pdf/ocr.py
touch src/arcaneum/indexing/pdf/preprocessor.py
touch src/arcaneum/indexing/pdf/chunker.py
touch src/arcaneum/indexing/common/__init__.py
touch src/arcaneum/indexing/common/checkpoint.py
touch src/arcaneum/indexing/common/sync.py
touch src/arcaneum/indexing/common/progress.py
touch src/arcaneum/indexing/common/retry.py
touch src/arcaneum/indexing/uploader.py
```

#### Step 3: Implement Core Components

**3.1: PDF Extractor** (`src/arcaneum/indexing/pdf/extractor.py`)
- Implement `PDFExtractor` class (see Implementation Example above)
- PyMuPDF primary extraction
- pdfplumber fallback for complex tables
- Table validation heuristic

**3.2: OCR Engine** (`src/arcaneum/indexing/pdf/ocr.py`)
- Implement `OCREngine` class (see Implementation Example above)
- Tesseract integration with pytesseract
- EasyOCR integration (optional)
- Image preprocessing (grayscale, threshold, denoise, deskew)
- Confidence filtering

**3.3: Chunker** (`src/arcaneum/indexing/pdf/chunker.py`)
- Implement `PDFChunker` class (see Implementation Example above)
- Traditional chunking with 15% overlap
- Late chunking strategy selection
- Semantic chunking (future enhancement)

**3.4: Checkpoint Manager** (`src/arcaneum/indexing/common/checkpoint.py`)
```python
import sqlite3
from pathlib import Path

class BatchCheckpoint:
    """SQLite-based checkpoint for resumability."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS upload_progress (
                batch_id INTEGER PRIMARY KEY,
                file_path TEXT,
                chunk_start INTEGER,
                chunk_end INTEGER,
                status TEXT,
                uploaded_at TIMESTAMP,
                error_message TEXT
            )
        """)
        self.conn.commit()

    def mark_completed(self, batch_id: int, file_path: str, chunks: tuple):
        self.conn.execute("""
            INSERT INTO upload_progress
            (batch_id, file_path, chunk_start, chunk_end, status, uploaded_at)
            VALUES (?, ?, ?, ?, 'completed', datetime('now'))
        """, (batch_id, file_path, chunks[0], chunks[1]))
        self.conn.commit()

    def get_last_completed_batch(self) -> int:
        cursor = self.conn.execute("""
            SELECT MAX(batch_id) FROM upload_progress
            WHERE status='completed'
        """)
        result = cursor.fetchone()[0]
        return result if result else 0
```

**3.4b: Metadata-Based Sync** (`src/arcaneum/indexing/common/sync.py`)
```python
import hashlib
from pathlib import Path
from typing import List, Set
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import logging

logger = logging.getLogger(__name__)

def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of file content (first 12 chars).

    Uses chunked reading to handle large files efficiently.
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:12]

class MetadataBasedSync:
    """
    Check indexing status using Qdrant metadata queries.

    Follows chroma-embedded pattern: query file_path and file_hash
    from chunk metadata to determine if file is already indexed.
    """

    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant = qdrant_client

    def is_file_indexed(self, collection_name: str, file_path: Path,
                       file_hash: str) -> bool:
        """
        Check if file with current content hash is already indexed.

        Returns True if ANY chunks with this file_path AND file_hash exist.
        """
        try:
            points, _ = self.qdrant.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=str(file_path))
                        ),
                        FieldCondition(
                            key="file_hash",
                            match=MatchValue(value=file_hash)
                        )
                    ]
                ),
                limit=1,  # Just need to know if exists
                with_payload=False,  # Don't need payload, faster
                with_vectors=False   # Don't need vectors, faster
            )

            return len(points) > 0

        except Exception as e:
            logger.warning(f"Error querying collection: {e}")
            return False

    def get_indexed_file_paths(self, collection_name: str) -> Set[tuple]:
        """
        Get all (file_path, file_hash) pairs from collection.

        Returns set of tuples for fast lookup.
        """
        indexed = set()
        offset = None

        try:
            while True:
                points, offset = self.qdrant.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=["file_path", "file_hash"],
                    with_vectors=False
                )

                if not points:
                    break

                for point in points:
                    if point.payload:
                        path = point.payload.get("file_path")
                        hash_val = point.payload.get("file_hash")
                        if path and hash_val:
                            indexed.add((path, hash_val))

                if offset is None:
                    break

            return indexed

        except Exception as e:
            logger.warning(f"Error scrolling collection: {e}")
            return set()

    def get_unindexed_files(self, collection_name: str,
                            file_list: List[Path]) -> List[Path]:
        """
        Filter file list to only unindexed or modified files.

        Uses batch query for efficiency instead of per-file queries.
        """
        try:
            # Get all indexed (path, hash) pairs
            indexed = self.get_indexed_file_paths(collection_name)

            # Filter to files not in indexed set
            unindexed = []
            for file_path in file_list:
                file_hash = compute_file_hash(file_path)
                if (str(file_path), file_hash) not in indexed:
                    unindexed.append(file_path)

            logger.info(f"Found {len(unindexed)}/{len(file_list)} "
                       f"files to index ({len(file_list) - len(unindexed)} "
                       f"already indexed)")
            return unindexed

        except Exception as e:
            logger.warning(f"Error querying collection: {e}, "
                          "processing all files")
            return file_list
```

**3.5: Batch Uploader** (`src/arcaneum/indexing/uploader.py`)
- Implement `PDFBatchUploader` class (see Implementation Example above)
- Orchestrate extraction → OCR → chunking → embedding → upload pipeline
- Batch management (100-200 points)
- Progress tracking with tqdm
- Error handling with Tenacity

#### Step 4: Add Late Chunking Support

**4.1: Late Chunking Module** (`src/arcaneum/embeddings/late_chunking.py`)
```python
import torch
from typing import List
import numpy as np

class LateChunker:
    """Implement late chunking for long-context models."""

    def __init__(self, model, tokenizer, chunk_size: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def embed_with_late_chunking(self, text: str) -> List[np.ndarray]:
        """
        Apply late chunking:
        1. Embed entire document
        2. Apply mean pooling to chunk-sized windows
        3. Return contextual chunk embeddings
        """
        # Tokenize entire document
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=8192)

        # Embed entire document
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]

        # Apply mean pooling to chunks
        chunk_embeddings = []
        for i in range(0, len(embeddings), self.chunk_size):
            chunk_tokens = embeddings[i:i+self.chunk_size]
            chunk_embedding = chunk_tokens.mean(dim=0).numpy()
            chunk_embeddings.append(chunk_embedding)

        return chunk_embeddings
```

**Note**: For jina-embeddings-v3, use API parameter `late_chunking=True` instead of custom implementation.

#### Step 5: Create CLI Command

**5.1: CLI Command** (`src/arcaneum/cli/index_pdfs.py`)
```python
import click
from pathlib import Path
from ..config import load_config
from ..indexing.uploader import PDFBatchUploader
from ..embeddings.client import EmbeddingClient
from qdrant_client import QdrantClient

@click.command()
@click.option('--input', type=click.Path(exists=True), required=True, help='PDF directory')
@click.option('--collection', required=True, help='Collection name')
@click.option('--config', type=click.Path(exists=True), default='./arcaneum.yaml', help='Config file')
@click.option('--model', default='stella', help='Embedding model (stella, bge, modernbert, jina)')
@click.option('--resume/--no-resume', default=True, help='Resume from checkpoint')
@click.option('--force', is_flag=True, default=False, help='Force reindex all files (bypass incremental sync)')
def index_pdfs(input, collection, config, model, resume, force):
    """
    Index PDFs to Qdrant collection with incremental sync.

    By default, only new or modified PDFs are indexed (based on file_hash).
    Use --force to reindex all files regardless of what's already indexed.
    """

    # Load config
    cfg = load_config(Path(config))

    # Initialize clients
    qdrant = QdrantClient(url=str(cfg.qdrant.url))
    embeddings = EmbeddingClient(
        cache_dir=cfg.cache.models_dir,
        models_config=cfg.models
    )

    # Create uploader
    uploader = PDFBatchUploader(
        qdrant_client=qdrant,
        embedding_client=embeddings,
        config=cfg.model_dump()
    )

    # Index PDFs with incremental sync
    stats = uploader.index_directory(
        pdf_dir=Path(input),
        collection_name=collection,
        model_name=model,
        resume=resume,
        force_reindex=force
    )

    click.echo(f"✅ Indexed {stats['files']} PDFs ({stats['chunks']} chunks)")
    if stats['errors'] > 0:
        click.echo(f"⚠️  {stats['errors']} errors occurred")
```

**5.2: Register Command** (`src/arcaneum/cli/main.py`)
```python
from .index_pdfs import index_pdfs

@cli.group()
def index():
    """Indexing commands."""
    pass

index.add_command(index_pdfs, name='pdfs')
```

#### Step 6: Configuration File

Create example configuration (`examples/pdf-indexing-config.yaml`):
```yaml
# Copy from Technical Design section above
```

#### Step 7: Testing

**7.1: Unit Tests**
```bash
# Test PDF extraction
pytest tests/test_pdf_extractor.py

# Test OCR
pytest tests/test_ocr.py

# Test chunking
pytest tests/test_chunker.py

# Test batch upload
pytest tests/test_uploader.py
```

**7.2: Integration Test**
```bash
# Create test collection
arcaneum collection create pdf-test --models stella,bge --config ./arcaneum.yaml

# Index test PDFs
arcaneum index pdfs --input ./test_pdfs --collection pdf-test --model stella

# Verify upload
arcaneum collection info pdf-test

# Cleanup
arcaneum collection delete pdf-test --confirm
```

#### Step 8: Documentation

**8.1: Update README.md**
```markdown
## PDF Indexing

Index PDFs with OCR support:

\`\`\`bash
# Start Qdrant server (if not running)
docker compose up -d

# Create collection
arcaneum collection create pdf-docs --models stella,bge

# Index PDFs
arcaneum index pdfs \
  --input /path/to/pdfs \
  --collection pdf-docs \
  --model stella
\`\`\`

See [RDR-004](doc/rdr/RDR-004-pdf-bulk-indexing.md) for details.
```

**8.2: Create User Guide** (`docs/pdf-indexing-guide.md`)
- Installation instructions
- Configuration examples
- Common workflows
- Troubleshooting

### Files to Create

**Core Implementation**:
- `src/arcaneum/indexing/pdf/extractor.py` - PDF extraction (PyMuPDF + pdfplumber)
- `src/arcaneum/indexing/pdf/ocr.py` - OCR integration (Tesseract + EasyOCR)
- `src/arcaneum/indexing/pdf/chunker.py` - Chunking strategies
- `src/arcaneum/indexing/common/checkpoint.py` - SQLite checkpointing
- `src/arcaneum/indexing/uploader.py` - Batch upload orchestrator
- `src/arcaneum/embeddings/late_chunking.py` - Late chunking implementation

**CLI**:
- `src/arcaneum/cli/index_pdfs.py` - CLI command

**Configuration**:
- `examples/pdf-indexing-config.yaml` - Example configuration

**Tests**:
- `tests/test_pdf_extractor.py`
- `tests/test_ocr.py`
- `tests/test_chunker.py`
- `tests/test_uploader.py`

**Documentation**:
- `doc/rdr/RDR-004-pdf-bulk-indexing.md` - This document
- `docs/pdf-indexing-guide.md` - User guide

### Files to Modify

- `doc/rdr/README.md` - Add RDR-004 to index
- `README.md` - Add PDF indexing section
- `pyproject.toml` - Add new dependencies

### Dependencies

**New Dependencies** (add to `pyproject.toml`):
```toml
[project]
dependencies = [
    # Existing (from RDR-002/003)
    "qdrant-client[fastembed]>=1.15.0",
    "fastembed>=0.3.0",
    "click>=8.1.0",
    "rich>=13.0.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",

    # New for PDF indexing
    "pymupdf>=1.23.0",          # PDF extraction (primary)
    "pdfplumber>=0.10.0",       # PDF extraction (fallback)
    "pytesseract>=0.3.10",      # Tesseract OCR wrapper
    "pdf2image>=1.16.0",        # PDF to image conversion
    "opencv-python-headless>=4.8.0",  # Image preprocessing
    "tenacity>=8.2.0",          # Retry logic
    "tqdm>=4.66.0",             # Progress tracking
]

[project.optional-dependencies]
ocr = [
    "easyocr>=1.7.0",           # Alternative OCR (GPU)
]
```

**System Dependencies**:
- Tesseract 5.x: `apt-get install tesseract-ocr` (Ubuntu) or `brew install tesseract` (macOS)
- Poppler: `apt-get install poppler-utils` (Ubuntu) or `brew install poppler` (macOS)

## Validation

### Testing Approach

**Multi-Level Testing Strategy**:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test end-to-end pipeline with sample PDFs
3. **Performance Tests**: Benchmark throughput and latency
4. **Quality Tests**: Validate extraction and OCR accuracy

### Test Scenarios

#### 1. Text PDF Extraction

**Scenario**: Index 100 PDFs with clean embedded text (no OCR needed)

**Test PDFs**: Technical documentation, research papers, books

**Expected Results**:
- PyMuPDF extracts all text successfully
- No OCR triggered (text > 100 chars on all pages)
- Extraction speed: < 0.01s per page average
- Chunks: 512-1024 tokens with 15% overlap
- Upload: 100-200 chunks per batch
- Throughput: ~100 PDFs/minute
- Errors: 0%

**Validation**:
```python
# Test extraction
text, metadata = extractor.extract(test_pdf)
assert len(text) > 100
assert metadata['extraction_method'] == 'pymupdf'
assert metadata['is_image_pdf'] == False

# Test no OCR trigger
assert 'ocr_confidence' not in metadata
```

#### 2. Image PDF OCR (Tesseract)

**Scenario**: Index 20 scanned PDFs requiring OCR

**Test PDFs**: Scanned books, historical documents, photocopies

**Expected Results**:
- PyMuPDF finds < 100 chars, triggers OCR
- Tesseract processes at 300 DPI with 2x scaling
- OCR speed: 2s per page average
- Confidence: ≥ 60% on most text
- Accuracy: 99%+ on clean scans
- Errors: < 5% (reject low-confidence text)

**Validation**:
```python
# Test OCR trigger
text, metadata = extractor.extract(scanned_pdf)
assert metadata['extraction_method'] == 'ocr_tesseract'
assert metadata['is_image_pdf'] == True
assert metadata['ocr_confidence'] >= 60

# Test confidence filtering
assert all(conf >= 60 for conf in page_confidences)
```

#### 3. Mixed PDF (Hybrid)

**Scenario**: Index 10 PDFs with both embedded text and scanned images

**Test PDFs**: Scientific papers with scanned figures, forms with handwriting

**Expected Results**:
- PyMuPDF extracts embedded text from text pages
- OCR triggered only on pages with < 100 chars
- Hybrid metadata: both extraction methods used
- Combined accuracy: 95%+
- No duplicate content

**Validation**:
```python
# Test hybrid extraction
assert 'pymupdf' in extraction_methods
assert 'ocr_tesseract' in extraction_methods
assert no_duplicate_chunks(chunks)
```

#### 4. Complex Table Extraction

**Scenario**: Index 10 PDFs with complex tables (financial reports, data sheets)

**Test PDFs**: Annual reports, scientific data tables, invoices

**Expected Results**:
- PyMuPDF attempts extraction first
- Table validation detects complex tables
- Fallback to pdfplumber triggered
- Tables extracted as Markdown format
- Table structure preserved (rows, columns, headers)
- Accuracy: 90%+ table structure preserved

**Validation**:
```python
# Test pdfplumber fallback
text, metadata = extractor.extract(table_pdf)
assert metadata['extraction_method'] == 'pdfplumber'
assert '|' in text  # Markdown table format
assert count_table_rows(text) == expected_rows
```

#### 5. Late Chunking (Long Documents)

**Scenario**: Index 10 long PDFs (50-100 pages) with late chunking

**Test PDFs**: Books, technical manuals, dissertations

**Expected Results**:
- Documents: 2,000-8,000 tokens (within late chunking range)
- Strategy: Late chunking selected automatically
- Chunks: Contextual embeddings generated
- Retrieval quality: 5-8% nDCG improvement vs traditional
- No errors

**Validation**:
```python
# Test late chunking selection
assert estimated_tokens > 2000
assert estimated_tokens < 8000
assert chunking_strategy == 'late_chunking'
assert all(chunk.metadata['late_chunking'] for chunk in chunks)
```

#### 6. Batch Upload with Retry

**Scenario**: Index 1000 PDFs with simulated network errors

**Test Setup**: Inject random failures (10% error rate) during upload

**Expected Results**:
- Exponential backoff retry triggers on failures
- All batches eventually succeed (max 5 retries)
- Checkpoint saves progress after each batch
- Resume from last completed batch on restart
- Final upload: 100% success rate
- No duplicate points

**Validation**:
```python
# Test retry logic
with mock_network_errors(error_rate=0.1):
    stats = uploader.index_directory(pdf_dir, collection, model)

assert stats['errors'] == 0  # All retries succeeded
assert no_duplicates_in_collection(collection)
assert checkpoint.get_last_completed_batch() == expected_batch_count
```

#### 7. Resumability (Checkpoint Recovery)

**Scenario**: Index 500 PDFs, simulate crash at batch 50, resume

**Test Setup**:
1. Start indexing 500 PDFs
2. Force crash after batch 50 (5000 chunks)
3. Resume indexing

**Expected Results**:
- Checkpoint records batch 1-50 as completed
- Resume skips first 50 batches
- Processing continues from batch 51
- No duplicate uploads
- Final stats: 500 PDFs indexed

**Validation**:
```python
# Test checkpoint recovery
uploader1 = PDFBatchUploader(...)
uploader1.index_directory(pdf_dir, collection, model)  # Crash at batch 50

checkpoint = BatchCheckpoint(checkpoint_db)
last_batch = checkpoint.get_last_completed_batch()
assert last_batch == 50

uploader2 = PDFBatchUploader(...)
stats = uploader2.index_directory(pdf_dir, collection, model, resume=True)
assert stats['files'] == 500
assert no_duplicates_in_collection(collection)
```

#### 8. Incremental Indexing (Semantic Sync)

**Scenario**: Index directory, add new PDFs, verify only new files are processed

**Test Setup**:
1. Index 100 PDFs to collection (first run)
2. Add 50 new PDFs to directory
3. Modify 10 existing PDFs (change content)
4. Run indexing again (second run)

**Expected Results**:
- First run: 100 PDFs indexed, metadata stored in Qdrant
- Second run: Only 60 PDFs processed (50 new + 10 modified)
- Second run: 40 PDFs skipped (unchanged)
- Qdrant metadata tracks file_path and file_hash
- Modified files detected via hash comparison
- Total: 150 PDFs in collection

**Validation**:
```python
from ..indexing.common.sync import MetadataBasedSync, compute_file_hash

# First run: index initial set
pdf_dir = Path("./test_pdfs")
create_test_pdfs(pdf_dir, count=100)

uploader = PDFBatchUploader(...)
stats1 = uploader.index_directory(
    pdf_dir, collection, model, force_reindex=False
)

assert stats1['files'] == 100
assert stats1['errors'] == 0

# Verify Qdrant metadata contains file_path and file_hash
sync = MetadataBasedSync(qdrant)
indexed_files = sync.get_indexed_file_paths(collection)
assert len(indexed_files) >= 100  # At least 100 unique file paths

# Add new PDFs and modify existing
create_test_pdfs(pdf_dir / "new", count=50)  # 50 new files
modify_pdfs(pdf_dir, count=10)  # Change content of 10 files

# Second run: incremental sync via metadata queries
all_pdfs = sorted(pdf_dir.rglob("*.pdf"))
assert len(all_pdfs) == 150  # 100 original + 50 new

stats2 = uploader.index_directory(
    pdf_dir, collection, model, force_reindex=False
)

# Only new/modified files processed (detected via file_hash mismatch)
assert stats2['files'] == 60  # 50 new + 10 modified
assert stats2['errors'] == 0

# Verify collection has all documents
collection_count = qdrant.count(collection_name=collection)
expected_chunks = stats1['chunks'] + stats2['chunks']
assert collection_count.count >= expected_chunks

# Verify metadata queries work correctly
indexed_files = sync.get_indexed_file_paths(collection)
assert len(indexed_files) == 150  # All files now indexed

# Test force reindex
stats3 = uploader.index_directory(
    pdf_dir, collection, model, force_reindex=True
)
assert stats3['files'] == 150  # All files reprocessed
```

#### 9. Character-to-Token Ratio Validation

**Scenario**: Validate character-to-token ratios with actual PDFs

**Test Setup**:
- Extract text from 100 diverse PDFs
- Tokenize with stella, bge-large, modernbert, jina-v3
- Calculate actual char-to-token ratios
- Compare with RDR-003 estimates

**Expected Results**:
- Actual ratios: 3.5-4.0 chars/token (English text)
- RDR-003 estimates (3.2-3.4): Conservative (safe)
- No chunk overflow errors
- Chunk sizes: Within model token limits

**Validation**:
```python
# Test char-to-token ratios
for model_name, model_config in models.items():
    texts = extract_sample_texts(pdfs, count=100)
    actual_ratio = calculate_char_to_token_ratio(texts, model_name)
    rdr003_ratio = model_config['char_to_token_ratio']

    assert actual_ratio >= rdr003_ratio  # Conservative estimate
    assert no_token_overflow_errors(chunks, model_name)
```

### Performance Validation

**Throughput Targets**:

| Scenario | Target | Measurement |
|----------|--------|-------------|
| Text PDF extraction | 100 PDFs/min | PyMuPDF speed |
| OCR (Tesseract CPU) | 20 pages/min | Tesseract with preprocessing |
| OCR (EasyOCR GPU) | 80 pages/min | EasyOCR with CUDA |
| Embedding generation | 200 chunks/min | FastEmbed batch processing |
| Batch upload | 333 chunks/sec | Qdrant with parallel=4 |
| End-to-end (text PDFs) | 100 PDFs/min | Full pipeline |
| End-to-end (OCR PDFs) | 10 PDFs/min | Full pipeline with OCR |

**Latency Targets**:

| Operation | Target | Notes |
|-----------|--------|-------|
| PDF extraction | < 0.01s per page | PyMuPDF |
| OCR processing | 2s per page | Tesseract CPU |
| Chunking | < 0.1s per document | Traditional chunking |
| Embedding (100 chunks) | < 5s | FastEmbed batch |
| Batch upload (200 points) | < 1s | Qdrant with parallel=4 |

**Memory Usage**:

| Component | Peak RAM | Notes |
|-----------|----------|-------|
| PDF extraction | < 100 MB | PyMuPDF + pdfplumber |
| OCR processing | < 500 MB | Tesseract + image preprocessing |
| EasyOCR (GPU) | < 2 GB | PyTorch + models |
| FastEmbed models | 3-4 GB | stella, bge, modernbert, jina |
| Qdrant vectors (100K) | 1-6 GB | With on_disk=True optimization |
| Total (worst case) | < 10 GB | All components active |

**Benchmark Script**:
```python
import time
from pathlib import Path

def benchmark_pipeline(pdf_dir: Path, sample_size: int = 100):
    """Benchmark PDF indexing pipeline."""

    pdfs = list(pdf_dir.rglob("*.pdf"))[:sample_size]

    # Extraction benchmark
    start = time.time()
    for pdf in pdfs:
        text, _ = extractor.extract(pdf)
    extraction_time = time.time() - start

    print(f"Extraction: {sample_size / extraction_time:.1f} PDFs/min")

    # OCR benchmark (on sample of scanned PDFs)
    scanned_pdfs = [p for p in pdfs if is_scanned(p)][:10]
    start = time.time()
    for pdf in scanned_pdfs:
        text, _ = ocr.process_pdf(pdf)
    ocr_time = time.time() - start
    pages = sum(get_page_count(p) for p in scanned_pdfs)

    print(f"OCR: {pages / (ocr_time / 60):.1f} pages/min")

    # End-to-end benchmark
    start = time.time()
    stats = uploader.index_directory(pdf_dir, collection, model)
    total_time = time.time() - start

    print(f"End-to-end: {stats['files'] / (total_time / 60):.1f} PDFs/min")
    print(f"Throughput: {stats['chunks'] / total_time:.1f} chunks/sec")
```

### Security Validation

**Security Considerations**:

1. **PDF Parsing Vulnerabilities**:
   - Use latest PyMuPDF/pdfplumber versions (security patches)
   - Wrap extraction in try-except (malicious PDFs can crash parsers)
   - Validate file extensions before processing
   - Set file size limits (reject > 100 MB PDFs)

2. **OCR Command Injection**:
   - pytesseract properly escapes file paths
   - Use Path objects, not string concatenation
   - Validate OCR language codes (whitelist: eng, fra, spa, etc.)

3. **SQLite Injection**:
   - Use parameterized queries (no string formatting)
   - All checkpoint queries use `?` placeholders
   - No user input directly in SQL

4. **Qdrant Upload**:
   - No credentials in code (use environment variables)
   - Validate payload data before upload
   - Set upload size limits (max 1000 points per batch)

**Validation Tests**:
```python
# Test malicious PDF rejection
def test_malicious_pdf_handling():
    malicious_pdf = create_malicious_pdf()

    try:
        extractor.extract(malicious_pdf)
        assert False, "Should have rejected malicious PDF"
    except Exception as e:
        assert "validation failed" in str(e).lower()

# Test SQL injection prevention
def test_sql_injection():
    malicious_path = "'; DROP TABLE upload_progress; --"
    checkpoint.mark_completed(1, malicious_path, (0, 100))

    # Table should still exist
    assert checkpoint.get_last_completed_batch() == 1
```

## References

### Research Sources

**RDR Research Tasks** (Completed 2025-10-19):
- arcaneum-15: Prior RDR analysis (RDR-001, 002, 003)
- arcaneum-16: ChromaDB upload.sh analysis (lines 1372-1522, 269-324)
- arcaneum-17: outstar-rag-requirements.md (lines 136-167)
- arcaneum-18: PyMuPDF capabilities research (opensource agent)
- arcaneum-19: pdfplumber capabilities research (opensource agent)
- arcaneum-20: Tesseract OCR research (130+ page report)
- arcaneum-21: EasyOCR research (detailed comparison)
- arcaneum-22: Embedding model token limits & chunking
- arcaneum-23: Qdrant batch upload best practices

### Official Documentation

**PDF Libraries**:
- PyMuPDF Documentation: https://pymupdf.readthedocs.io/
- pdfplumber GitHub: https://github.com/jsvine/pdfplumber
- pymupdf4llm: https://github.com/pymupdf/RAG

**OCR Engines**:
- Tesseract Documentation: https://tesseract-ocr.github.io/tessdoc/
- pytesseract PyPI: https://pypi.org/project/pytesseract/
- EasyOCR GitHub: https://github.com/JaidedAI/EasyOCR
- Jaided AI Docs: http://www.jaided.ai/easyocr/documentation/

**Embedding Models**:
- stella_en_1.5B_v5: https://huggingface.co/dunzhang/stella_en_1.5B_v5
- ModernBERT: https://huggingface.co/answerdotai/ModernBERT-base
- bge-large-en-v1.5: https://huggingface.co/BAAI/bge-large-en-v1.5
- jina-embeddings-v3: https://jina.ai/models/jina-embeddings-v3/

**Qdrant**:
- Qdrant Bulk Upload: https://qdrant.tech/documentation/database-tutorials/bulk-upload/
- Qdrant Python Client: https://python-client.qdrant.tech/
- Qdrant Benchmarks: https://qdrant.tech/benchmarks/

### Academic Papers

- **Late Chunking**: "Contextual Document Embeddings" (arXiv:2409.04701, September 2024)
- **jina-embeddings-v3**: Technical Report (arXiv:2409.10173, September 2024)
- **NVIDIA Chunking**: "Finding the Best Chunking Strategy for Accurate AI Responses" (FinanceBench 2025)

### Related Documents

- **RDR-001**: Project Structure (Claude Code Marketplace)
- **RDR-002**: Qdrant Server Setup (client-side embeddings)
- **RDR-003**: Collection Creation (named vectors, model configs)
- **RDR-005** (future): Source Code Indexing (git-aware, AST chunking)
- **RDR-006** (future): Bulk Upload Plugin (MCP integration)
- **RDR-007** (future): Search Plugin (semantic query)

### Reference Implementation

- `chroma-embedded/upload.sh` (ChromaDB patterns):
  - Lines 1372-1522: PDF extraction and OCR
  - Lines 269-324: Token-optimized chunking
  - Lines 373-433: Git project discovery
  - Overall structure: Batch processing patterns

## Notes

### Implementation Priorities

**High Priority** (MVP):
1. ✅ PDF extraction (PyMuPDF primary, pdfplumber fallback)
2. ✅ OCR integration (Tesseract default)
3. ✅ Traditional chunking (15% overlap)
4. ✅ Batch upload (100-200 chunks, exponential backoff)
5. ✅ Checkpoint/resumability (SQLite for crash recovery)
6. ✅ Incremental indexing (metadata-based sync with file_hash)
7. ✅ CLI command (`arcaneum index pdfs --force`)

**Medium Priority** (Post-MVP):
1. ⭕ Late chunking implementation (stella, modernbert, jina-v3)
2. ⭕ EasyOCR alternative (GPU-accelerated)
3. ⭕ Semantic chunking with heading-awareness
4. ⭕ PDF-to-Markdown preprocessing
5. ⭕ Progress tracking with Rich (upgrade from tqdm)
6. ⭕ Performance benchmarking suite

**Low Priority** (Future Enhancements):
1. ⭕ Multi-embedding sync (add new models to existing documents via named vectors)
2. ⭕ Table extraction quality validation
3. ⭕ Multi-language OCR (fra, spa, deu, ara, etc.)
4. ⭕ Confidence-based quality routing (auto-approve vs review queue)
5. ⭕ GPU batch processing optimization
6. ⭕ Distributed processing (multi-node)
7. ⭕ Real-time indexing (watch directory for new PDFs)

### Known Limitations

**PDF Extraction**:
- **Complex layouts**: Multi-column scientific papers may have reading order issues
- **Embedded images**: Not extracted (text-only indexing)
- **Annotations/comments**: Not extracted
- **Password-protected PDFs**: Not supported (requires password parameter)

**OCR**:
- **Handwriting accuracy**: Tesseract struggles (90% max), EasyOCR better but still imperfect
- **Low-resolution scans**: < 200 DPI degrades accuracy significantly
- **Non-English languages**: Requires language-specific Tesseract data files
- **Processing time**: 2s per page is slow for large archives (thousands of PDFs)

**Chunking**:
- **Late chunking complexity**: Requires custom implementation for stella/modernbert
- **Context window limits**: Documents > 8K tokens require splitting
- **Token counting**: Character-to-token ratio is approximate (not exact)

**Batch Upload**:
- **Memory spikes**: Large batches (500+) can cause OOM on memory-constrained systems
- **Network interruptions**: Retry logic mitigates but doesn't eliminate risk
- **Rate limiting**: Qdrant server may throttle high-volume uploads

### Future Enhancements

**RDR-005: Source Code Indexing**:
- Git-aware project discovery
- AST-aware chunking (preserves function/class boundaries)
- 15+ language support (Python, Java, JS/TS, C#, Go, Rust, C/C++, etc.)
- Commit hash change detection
- Integration with jina-code embeddings (optimized for code)

**RDR-006: Bulk Upload Plugin**:
- MCP plugin wrapping PDF and source code indexers
- Unified CLI interface for all document types
- Parallel processing (multiprocessing pool)
- Real-time progress reporting to Claude UI
- Error recovery and manual review queue

**RDR-007: Search Plugin**:
- MCP plugin for semantic search
- Query embedding generation (match collection models)
- Metadata filtering DSL
- Multi-collection search with result merging
- Hybrid search (semantic + full-text) with Reciprocal Rank Fusion

**Advanced Features**:
- **Multi-embedding sync**: Add new embedding models to existing documents
  - `EmbeddingRegistry` tracks (file_path, model_name) pairs
  - Three sync modes: `incremental` (new files), `fill` (add model to existing), `full` (reindex all)
  - Uses Qdrant's `update_vectors()` to add named vectors to existing points
  - Example: `arcaneum index pdfs --model modernbert --sync-mode fill` adds modernbert embeddings to all stella-indexed docs
- **Quality scoring**: Confidence-based routing (auto-index vs manual review)
- **Multi-modal**: Extract and index images, diagrams, charts
- **Summarization**: Generate document summaries with LLMs
- **Entity extraction**: Named entities (people, places, organizations)

### Migration from ChromaDB

**For Existing ChromaDB Users**:

This RDR adapts proven patterns from `chroma-embedded/upload.sh` for Qdrant. Key migration steps:

1. **Preserve embeddings**: Use same models (stella, modernbert, bge-large) to maintain semantic consistency
2. **Increase batch size**: 50 → 100-200 (Qdrant handles larger batches better)
3. **Update retry logic**: Remove manual retry, use Qdrant's built-in + Tenacity
4. **Add parallelization**: Enable `parallel=4` for 3-4x speedup
5. **Update overlap**: 10% → 15% (NVIDIA 2025 recommendation)
6. **Validate chunking**: Confirm same chunk sizes transfer (460, 920 tokens, etc.)
7. **Test embeddings**: Verify FastEmbed produces equivalent vectors to ChromaDB

**Compatibility Table**:

| Aspect | ChromaDB | Qdrant (RDR-004) | Migration Action |
|--------|----------|------------------|------------------|
| Batch size | 50 | 100-200 | ✅ Increase |
| Retry | Manual | Built-in + Tenacity | ✅ Simplify code |
| Parallel upload | No | Yes (parallel=4) | ✅ Enable |
| Overlap | 10% | 15% | ✅ Update |
| Embedding gen | Server | Client (FastEmbed) | ⚠️ Code change |
| Point IDs | String hashes | Integers | ⚠️ Code change |
| Metadata filters | Simple dict | Filter API | ⚠️ Code change |

### Key Insights

**From Research**:
1. **Late chunking is a game-changer**: 5-8% nDCG improvement on long documents (2025 breakthrough)
2. **15% overlap is optimal**: NVIDIA FinanceBench testing shows 15% > 10% or 20%
3. **PDF preprocessing matters**: Convert to Markdown for best chunking quality
4. **OCR is expensive**: Only trigger when needed (< 100 chars threshold)
5. **PyMuPDF is fast**: 95x faster than pdfplumber, use as primary
6. **Qdrant batches larger**: 100-200 chunks vs ChromaDB's 50-250 limit
7. **Parallel uploads matter**: 4 workers = 3-4x speedup
8. **Character-to-token ratios are conservative**: RDR-003 estimates (3.2-3.4) are safe but could be 3.5-4.0

**Production Lessons**:
1. **Always checkpoint**: Long-running jobs need resumability
2. **Exponential backoff works**: Tenacity library handles retry elegantly
3. **Progress tracking is essential**: Users need real-time feedback on large jobs
4. **Confidence filtering is critical**: Low-confidence OCR text pollutes index
5. **Error handling must be robust**: PDFs are unpredictable (corruption, malformed structures, etc.)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-19
**Status**: Ready for implementation
**Next Step**: Implement Step 1 (Install Dependencies) and Step 2 (Create Module Structure)

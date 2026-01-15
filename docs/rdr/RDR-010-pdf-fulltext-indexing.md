# Recommendation 010: Bulk PDF Indexing to Full-Text Search (MeiliSearch)

## Metadata

- **Date**: 2025-10-21
- **Updated**: 2026-01-14
- **Status**: Recommendation
- **Type**: Feature
- **Priority**: High
- **Related Issues**: arcaneum-69, arcaneum-5ysl, arcaneum-h6bo
- **Related Tests**: PDF full-text indexing tests, dual indexing integration tests

## Problem Statement

Create a PDF indexing pipeline for MeiliSearch that enables exact phrase search and
keyword matching across PDF documents. The system must:

1. **Reuse RDR-004 extraction pipeline** - PyMuPDF + OCR for text extraction
2. **Index to MeiliSearch** - Text-only (no vectors), page-level granularity
3. **Enable exact search** - Quote syntax, regex, line-number precision
4. **Share metadata with Qdrant** - Complementary to semantic search (RDR-004)
5. **Support dual indexing** - Same PDFs indexed to both systems (RDR-009)

This addresses the need to search PDFs for exact phrases, keywords, and regex patterns
complementary to the semantic search provided by Qdrant (RDR-004).

## Context

### Background

Arcaneum provides two complementary search systems:

- **Qdrant (RDR-004)**: Semantic similarity search on PDF embeddings
- **MeiliSearch (RDR-008)**: Full-text exact phrase and keyword search

**Use Case Examples:**

1. **Semantic discovery** (Qdrant): "Find documents about authentication patterns"
2. **Exact verification** (MeiliSearch): "Find exact phrase 'SHA-256 encryption'"
3. **Combined workflow**: Semantic search finds relevant PDFs → Exact search locates specific content

**Key Requirements (from arcaneum-69):**

- Reuse RDR-004 PDF extraction pipeline (PyMuPDF + OCR)
- Page-level vs document-level indexing decision
- Metadata synchronization with Qdrant
- Duplicate detection and updates
- Change detection using file hash strategy

### CLI Naming Convention (Updated 2026-01-14)

This RDR follows the symmetric CLI naming convention established for semantic/text
operations (see arcaneum-h6bo):

```bash
# Search commands (existing)
arc search semantic "query" --collection X    # Vector search (Qdrant)
arc search text "query" --index X             # Full-text search (MeiliSearch)

# Index commands (symmetric naming)
arc index semantic pdf /path --collection X   # Semantic indexing (Qdrant)
arc index text pdf /path --index X            # Full-text indexing (MeiliSearch) - THIS RDR

# Management commands
arc collection list/create/delete             # Qdrant collection management
arc indexes list/create/delete                # MeiliSearch index management
```

**Note**: The existing `arc index pdf` command will be renamed to `arc index semantic pdf`
as part of arcaneum-h6bo. This RDR implements the complementary `arc index text pdf` command.

### Relationship to RDR-009 Dual Indexing

RDR-009 implements dual indexing via `arc corpus sync`, which indexes to BOTH Qdrant
and MeiliSearch simultaneously. This RDR provides a **standalone MeiliSearch-only**
pathway that complements (not replaces) the dual-indexing workflow:

| Command                  | Target           | Use Case                            |
|--------------------------|------------------|-------------------------------------|
| `arc index semantic pdf` | Qdrant only      | Semantic search without full-text   |
| `arc index text pdf`     | MeiliSearch only | Full-text search without embeddings |
| `arc corpus sync`        | Both             | Unified dual-index corpus           |

**Key difference**: `arc index text pdf` uses **page-level** documents for precise
citations, while `arc corpus sync` uses **chunk-level** documents for consistency
with Qdrant vector chunks. See "Page-Level vs Chunk-Level" section below.

### Technical Environment

- **Python**: >= 3.12
- **MeiliSearch**: v1.32.0 (from RDR-008, updated)
- **PDF Libraries** (from RDR-004):
  - PyMuPDF (fitz) >= 1.23.0 - Primary text extraction
  - pdfplumber >= 0.10.0 - Table extraction fallback
- **OCR Engines** (from RDR-004):
  - Tesseract 5.x with pytesseract - Default OCR
  - EasyOCR >= 1.7.0 - Alternative for mixed content
- **MeiliSearch Client**:
  - meilisearch-python >= 0.31.0 (per pyproject.toml)
- **Supporting Libraries**:
  - pdf2image, opencv-python-headless (for OCR)
  - tenacity (retry logic)
  - tqdm/rich (progress tracking)

## Research Findings

### Investigation Process

**Research completed:**

1. **RDR-004 Review**: Complete PDF extraction pipeline (PyMuPDF + pdfplumber + OCR)
2. **RDR-008 Review**: MeiliSearch setup, index settings, client capabilities
3. **RDR-009 Review**: Dual indexing patterns, shared metadata schema
4. **MeiliSearch Native PDF Support**: Web search confirmed NO native PDF support
5. **MeiliSearch Python Client**: Verified via Chroma OpenSource collection:
   - `add_documents_in_batches(batch_size=1000)` confirmed
   - `wait_for_task()` async indexing confirmed
   - Pagination settings (`maxTotalHits`) confirmed
   - Primary key handling confirmed
   - Filter expressions syntax confirmed

### Key Discoveries

#### 1. MeiliSearch Has NO Native PDF Support

**Confirmation:**

- MeiliSearch only accepts JSON, CSV, NDJSON formats
- Binary files (PDFs) cannot be indexed directly
- Must extract text using external tools (PyMuPDF, Tesseract)

**Implication**: We MUST reuse the RDR-004 extraction pipeline.

#### 2. RDR-004 Extraction Pipeline is Production-Ready

**Complete pipeline available:**

- **PyMuPDF** (primary): 95x faster, excellent for machine-generated PDFs
- **pdfplumber** (fallback): Superior table extraction
- **Tesseract OCR** (default): 99%+ accuracy on clean scans, CPU-friendly
- **EasyOCR** (alternative): Better on handwritten/noisy, requires GPU
- **OCR trigger logic**: Activate if extracted text < 100 characters

**Extraction classes (from RDR-004):**

- `PDFExtractor`: PyMuPDF + pdfplumber fallback logic (lines 516-631)
- `OCREngine`: Tesseract + EasyOCR with confidence filtering (lines 648-797)

**Decision**: Reuse 100% of RDR-004 extraction logic, adapt only the indexing destination.

#### 3. Page-Level vs Document-Level Indexing

**Analysis:**

| Approach           | Pros                                | Cons                            | Use Case                   |
|--------------------|-------------------------------------|---------------------------------|----------------------------|
| **Page-level**     | Exact location, smaller chunks      | More documents, slight overhead | Research papers, tech docs |
| **Document-level** | Fewer documents, simpler structure  | No page precision, large chunks | Books, novels              |

**Decision**: **Page-level indexing** (RDR-004 already chunks by page).

**Rationale:**

- Users need exact page numbers for citations
- Complements Qdrant's chunk-level indexing
- RDR-004 already processes page-by-page
- MeiliSearch handles large document counts efficiently

#### 4. Shared Metadata Schema (from RDR-009)

**Metadata alignment between Qdrant (RDR-004) and MeiliSearch (this RDR):**

| Field (Qdrant)      | Field (MeiliSearch)  | Purpose              |
|---------------------|----------------------|----------------------|
| `file_path`         | `file_path`          | Location tracking    |
| `filename`          | `filename`           | File name search     |
| `page_number`       | `page_number`        | Page-level precision |
| `file_hash`         | `file_hash`          | Duplicate detection  |
| `extraction_method` | `extraction_method`  | Metadata tracking    |
| `is_image_pdf`      | `is_image_pdf`       | OCR flag             |
| `ocr_confidence`    | `ocr_confidence`     | Quality indicator    |

**Cooperative workflow:**

```bash
# 1. Semantic discovery
arc search semantic "machine learning algorithms" --collection research-papers

# Results show: paper.pdf (pages 5-7 relevant)

# 2. Exact verification
arc search text '"gradient descent algorithm"' --index research-papers \
  --filter 'file_path = paper.pdf AND page_number >= 5 AND page_number <= 7'
```

#### 5. MeiliSearch Client Capabilities (Verified via Chroma)

**Batch Document Upload:**

```python
# From meilisearch-python/meilisearch/index.py:add_documents_in_batches
def add_documents_in_batches(
    self,
    documents: Sequence[Mapping[str, Any]],
    batch_size: int = 1000,  # Default 1000, configurable
    primary_key: Optional[str] = None,
    *,
    serializer: Optional[Type[JSONEncoder]] = None,
) -> List[TaskInfo]:
```

**Findings:**

- ✅ Default batch size: 1000 documents
- ✅ Returns list of `TaskInfo` for each batch
- ✅ Primary key required
- ✅ Async indexing via task queue

**Task Tracking:**

```python
# wait_for_task confirmed in client code
def wait_for_task(
    self,
    uid: int,
    timeout_in_ms: int = 5000,
    interval_in_ms: int = 50,
) -> Task:
```

**Findings:**

- ✅ Poll-based task monitoring
- ✅ Configurable timeout and interval
- ✅ Returns task status (succeeded/failed)

**Pagination Settings:**

```python
# get_pagination_settings / update_pagination_settings confirmed
{
    "maxTotalHits": 1000  # Default, configurable
}
```

#### 6. Change Detection Strategy

**From RDR-004:**

- **File hash-based sync**: Hash PDF content, store in metadata
- **Idempotent re-indexing**: Re-run after crash, skip already-indexed files
- **Metadata query**: Query MeiliSearch for indexed `(file_path, file_hash)` pairs

**Implementation:**

```python
# Check if PDF already indexed
existing_docs = meili.get_documents(
    index_name='pdf-docs',
    filter=f'file_path = {pdf_path} AND file_hash = {computed_hash}'
)

if existing_docs:
    print(f"Skipping {pdf_path} (already indexed)")
    continue
```

#### 7. Index Settings (from RDR-008)

**PDF_DOCS_SETTINGS (RDR-008 lines 639-667):**

```python
PDF_DOCS_SETTINGS = {
    "searchableAttributes": [
        "content",      # Primary search field
        "title",        # PDF title metadata
        "author",       # PDF author metadata
        "filename",     # File name search
    ],
    "filterableAttributes": [
        "filename",
        "file_path",
        "page_number",  # Enable page filtering
        "document_type",
        "file_hash",    # For change detection
        "extraction_method",
        "is_image_pdf",
    ],
    "sortableAttributes": ["page_number"],  # Sort by page
    "typoTolerance": {
        "enabled": True,
        "minWordSizeForTypos": {
            "oneTypo": 5,
            "twoTypos": 9
        }
    },
    "stopWords": [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"
    ],
    "pagination": {
        "maxTotalHits": 10000  # Increase for large PDF collections
    }
}
```

## Proposed Solution

### Approach

**Three-Phase PDF Full-Text Indexing Pipeline:**

```text
Phase 1: PDF Extraction (REUSE RDR-004)
├─ PyMuPDF primary extraction
├─ pdfplumber fallback for complex tables
└─ OCR trigger if text < 100 chars

Phase 2: Text Preparation for MeiliSearch
├─ Extract page-level text chunks
├─ Build MeiliSearch document structure
├─ Generate file hash for change detection
└─ Prepare shared metadata

Phase 3: Batch Upload to MeiliSearch
├─ Upload 1000 documents per batch
├─ Track tasks via wait_for_task
├─ Exponential backoff retry
└─ Progress tracking (tqdm/Rich)
```

**Key Differences from RDR-004:**

| Aspect              | RDR-004 (Qdrant)       | This RDR (MeiliSearch)   |
|---------------------|------------------------|--------------------------|
| **Data type**       | Vectors + text         | Text only                |
| **Chunking**        | Token-aware (512-1024) | Page-level (full page)   |
| **Embedding**       | FastEmbed required     | NOT required             |
| **Batch size**      | 100-200 points         | 1000 documents           |
| **Index structure** | Named vectors          | Searchable attributes    |
| **Search type**     | Semantic similarity    | Exact phrase, keyword    |

### Technical Design

#### Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│           PDF Full-Text Indexing Pipeline                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: PDF Extraction (REUSE RDR-004)                    │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │  Glob PDFs  │──▶│  PDFExtract │──▶│ OCREngine   │       │
│  │  (discover) │   │  (RDR-004)  │   │ (if needed) │       │
│  └─────────────┘   └─────────────┘   └──────┬──────┘       │
│                                              │               │
│  Output: page_texts = [                     │               │
│      {page_num: 1, text: "...", metadata},  │               │
│      {page_num: 2, text: "...", metadata},  │               │
│      ...                                     │               │
│  ]                                           │               │
└──────────────────────────────────────────────┼───────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: MeiliSearch Document Preparation                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │ Hash PDF    │──▶│ Build       │──▶│ Shared      │       │
│  │ (SHA-256)   │   │ MeiliSearch │   │ Metadata    │       │
│  │             │   │ Documents   │   │ (RDR-009)   │       │
│  └─────────────┘   └──────┬──────┘   └──────┬──────┘       │
│                           │                  │               │
│  Document structure:      │                  │               │
│  {                        │                  │               │
│    "id": "doc1_page1",    │                  │               │
│    "content": "...",      │                  │               │
│    "file_path": "/path",  │                  │               │
│    "page_number": 1,      │                  │               │
│    "file_hash": "abc123", │                  │               │
│    ...metadata            │                  │               │
│  }                        │                  │               │
└───────────────────────────┼──────────────────┼───────────────┘
                            │                  │
                            ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: Batch Upload to MeiliSearch                       │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │ Batch 1000  │──▶│ Upload with │──▶│ Track Tasks │       │
│  │ documents   │   │ add_docs_   │   │ wait_for_   │       │
│  │             │   │ in_batches  │   │ task()      │       │
│  └─────────────┘   └─────────────┘   └──────┬──────┘       │
│        │                                     │               │
│        │ Retry on failure                    │               │
│        ▼                                     ▼               │
│  ┌─────────────┐                     ┌─────────────┐       │
│  │ Exponential │                     │ Progress    │       │
│  │ Backoff     │                     │ Tracking    │       │
│  └─────────────┘                     └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

#### Module Structure

```text
src/arcaneum/
├── indexing/
│   ├── pdf/  (FROM RDR-004 - REUSE)
│   │   ├── __init__.py
│   │   ├── extractor.py         # PDFExtractor class (RDR-004)
│   │   ├── ocr.py                # OCREngine class (RDR-004)
│   │   └── preprocessor.py      # Artifact removal (RDR-004)
│   ├── fulltext/  (NEW - THIS RDR)
│   │   ├── __init__.py
│   │   ├── pdf_indexer.py       # PDFFullTextIndexer (NEW)
│   │   └── sync.py               # Change detection logic (NEW)
│   └── common/
│       ├── __init__.py
│       ├── progress.py           # Progress tracking (from RDR-004)
│       └── retry.py              # Retry logic (from RDR-004)
├── fulltext/  (FROM RDR-008)
│   ├── __init__.py
│   ├── client.py                 # FullTextClient (RDR-008)
│   └── indexes.py                # PDF_DOCS_SETTINGS (RDR-008)
└── cli/
    ├── index_text.py             # CLI: arc index text pdf/code/markdown (NEW)
    └── indexes.py                # CLI: arc indexes list/create/delete (RENAME from fulltext.py)
```

**Note**: The CLI module `fulltext.py` will be renamed to `indexes.py` as part of
the CLI restructure (arcaneum-h6bo). The command `arc fulltext` becomes `arc indexes`.

#### Implementation Components

##### Component 1: PDF Full-Text Indexer (NEW)

```python
# src/arcaneum/indexing/fulltext/pdf_indexer.py

from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
from rich.progress import track

from ..pdf.extractor import PDFExtractor  # RDR-004
from ..pdf.ocr import OCREngine            # RDR-004
from ...fulltext.client import FullTextClient  # RDR-008


class PDFFullTextIndexer:
    """Index PDFs to MeiliSearch for full-text search."""

    def __init__(
        self,
        meili_client: FullTextClient,
        index_name: str,
        ocr_enabled: bool = True,
        batch_size: int = 1000
    ):
        self.meili_client = meili_client
        self.index_name = index_name
        self.batch_size = batch_size

        # Reuse RDR-004 extraction components
        self.pdf_extractor = PDFExtractor(
            fallback_enabled=True,
            table_validation=True
        )
        self.ocr_engine = OCREngine(
            engine='tesseract',
            language='eng',
            confidence_threshold=60.0
        ) if ocr_enabled else None

    def index_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Index a single PDF to MeiliSearch.

        Returns:
            Dict with indexing statistics
        """
        # Phase 1: Extract text (REUSE RDR-004)
        text, metadata = self.pdf_extractor.extract(pdf_path)

        # Check if OCR needed
        if self.ocr_engine and len(text.strip()) < 100:
            text, ocr_metadata = self.ocr_engine.process_pdf(pdf_path)
            metadata.update(ocr_metadata)

        # Phase 2: Prepare MeiliSearch documents (page-level)
        documents = self._build_meilisearch_documents(
            pdf_path, text, metadata
        )

        # Phase 3: Upload to MeiliSearch
        task_info = self.meili_client.add_documents(
            index_name=self.index_name,
            documents=documents
        )

        return {
            'pdf_path': str(pdf_path),
            'page_count': len(documents),
            'task_uid': task_info['taskUid']
        }

    def _build_meilisearch_documents(
        self,
        pdf_path: Path,
        full_text: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Build MeiliSearch documents (one per page).

        Shared metadata schema with Qdrant (RDR-009).
        """
        # Compute file hash for change detection
        file_hash = self._compute_file_hash(pdf_path)

        # Split text by pages (simple split by form feed or page markers)
        # Note: RDR-004's PDFExtractor already processes page-by-page
        # So we reconstruct pages from full_text
        pages = self._split_into_pages(full_text, metadata['page_count'])

        documents = []
        for page_num, page_text in enumerate(pages, start=1):
            doc = {
                # Primary key
                'id': f"{pdf_path.stem}_page{page_num}",

                # Searchable content
                'content': page_text,
                'filename': pdf_path.name,

                # Filterable metadata (shared with Qdrant)
                'file_path': str(pdf_path),
                'page_number': page_num,
                'file_hash': file_hash,
                'extraction_method': metadata['extraction_method'],
                'is_image_pdf': metadata['is_image_pdf'],

                # Optional metadata
                'file_size': metadata.get('file_size'),
                'page_count': metadata['page_count'],
            }

            # Add OCR-specific metadata if present
            if 'ocr_confidence' in metadata:
                doc['ocr_confidence'] = metadata['ocr_confidence']
                doc['ocr_language'] = metadata.get('ocr_language', 'eng')

            documents.append(doc)

        return documents

    def _compute_file_hash(self, pdf_path: Path) -> str:
        """Compute SHA-256 hash of PDF for change detection."""
        sha256 = hashlib.sha256()
        with open(pdf_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _split_into_pages(
        self,
        full_text: str,
        page_count: int
    ) -> List[str]:
        """
        Split full text into pages.

        Note: This is a simplified version. RDR-004's PDFExtractor
        already processes page-by-page, so in practice we'd modify
        PDFExtractor to return List[PageText] instead of concatenated text.
        """
        # Placeholder: split by form feed character
        pages = full_text.split('\f')

        # Ensure we have the expected number of pages
        if len(pages) < page_count:
            # Pad with empty pages
            pages.extend([''] * (page_count - len(pages)))
        elif len(pages) > page_count:
            # Truncate excess
            pages = pages[:page_count]

        return pages

    def index_directory(
        self,
        directory: Path,
        recursive: bool = True
    ) -> Dict[str, Any]:
        """
        Index all PDFs in a directory to MeiliSearch.

        Returns:
            Dict with indexing statistics
        """
        # Discover PDFs
        pattern = '**/*.pdf' if recursive else '*.pdf'
        pdf_files = list(directory.glob(pattern))

        stats = {
            'total_pdfs': len(pdf_files),
            'indexed_pdfs': 0,
            'skipped_pdfs': 0,
            'failed_pdfs': 0,
            'total_pages': 0
        }

        # Index PDFs with progress tracking
        for pdf_path in track(pdf_files, description="Indexing PDFs..."):
            try:
                # Check if already indexed (change detection)
                if self._is_already_indexed(pdf_path):
                    stats['skipped_pdfs'] += 1
                    continue

                # Index PDF
                result = self.index_pdf(pdf_path)
                stats['indexed_pdfs'] += 1
                stats['total_pages'] += result['page_count']

            except Exception as e:
                print(f"❌ Failed to index {pdf_path.name}: {e}")
                stats['failed_pdfs'] += 1

        return stats

    def _is_already_indexed(self, pdf_path: Path) -> bool:
        """
        Check if PDF already indexed (change detection).

        Query MeiliSearch for existing document with same file_path + file_hash.
        """
        file_hash = self._compute_file_hash(pdf_path)

        try:
            # Query for existing documents
            results = self.meili_client.search(
                index_name=self.index_name,
                query='',  # Empty query
                filter=f'file_path = {pdf_path} AND file_hash = {file_hash}',
                limit=1
            )

            return results['estimatedTotalHits'] > 0

        except Exception:
            # If query fails, assume not indexed
            return False
```

##### Component 2: CLI Integration (NEW)

```python
# src/arcaneum/cli/index_text.py
# Command: arc index text pdf

import click
from pathlib import Path
from rich.console import Console

from ..fulltext.client import FullTextClient
from ..fulltext.indexes import PDF_DOCS_SETTINGS
from ..indexing.fulltext.pdf_indexer import PDFFullTextIndexer


console = Console()


# This is a subcommand group under 'arc index text'
@click.group('text')
def index_text():
    """Index content to MeiliSearch for full-text search."""
    pass


@index_text.command('pdf')
@click.argument('directory', type=click.Path(exists=True))
@click.option('--index', required=True, help='MeiliSearch index name')
@click.option('--recursive/--no-recursive', default=True,
              help='Index PDFs recursively')
@click.option('--ocr/--no-ocr', default=True,
              help='Enable OCR for scanned PDFs')
@click.option('--batch-size', type=int, default=1000,
              help='Batch size for document upload')
@click.option('--force', is_flag=True, help='Force reindex all files')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def index_text_pdf(
    directory,
    index,
    recursive,
    ocr,
    batch_size,
    force,
    output_json
):
    """Index PDFs to MeiliSearch for full-text search.

    Example:
        arc index text pdf ./research-papers --index MyPDFs
        arc index text pdf ./docs --index Docs --no-ocr --force
    """
    console.print(f"[bold cyan]Indexing PDFs to MeiliSearch[/bold cyan]")
    console.print(f"Directory: {directory}")
    console.print(f"Index: {index}")
    console.print(f"OCR: {'Enabled' if ocr else 'Disabled'}\n")

    # Get MeiliSearch client (uses environment/auto-generated key)
    from ..paths import get_meilisearch_api_key
    import os

    url = os.environ.get('MEILISEARCH_URL', 'http://localhost:7700')
    api_key = get_meilisearch_api_key()
    meili_client = FullTextClient(url, api_key)

    # Ensure index exists with correct settings
    if not meili_client.index_exists(index):
        console.print(f"Creating index '{index}'...")
        meili_client.create_index(
            name=index,
            primary_key='id',
            settings=PDF_DOCS_SETTINGS
        )
        console.print(f"✅ Index '{index}' created with PDF settings")
    else:
        console.print(f"✅ Index '{index}' found")

    # Initialize indexer
    indexer = PDFFullTextIndexer(
        meili_client=meili_client,
        index_name=index,
        ocr_enabled=ocr,
        batch_size=batch_size
    )

    # Index directory
    stats = indexer.index_directory(
        directory=Path(directory),
        recursive=recursive,
        force_reindex=force
    )

    # Print statistics
    console.print(f"\n[bold green]Indexing Complete[/bold green]")
    console.print(f"  Total PDFs: {stats['total_pdfs']}")
    console.print(f"  Indexed: {stats['indexed_pdfs']}")
    console.print(f"  Skipped: {stats['skipped_pdfs']}")
    console.print(f"  Failed: {stats['failed_pdfs']}")
    console.print(f"  Total pages: {stats['total_pages']}")
```

### Implementation Example

**Complete Workflow:**

```bash
# 1. Start MeiliSearch (from RDR-008)
docker compose up -d meilisearch

# 2. Create MeiliSearch index with PDF settings
arc indexes create research-pdfs --type pdf

# 3. Index PDFs to MeiliSearch (this RDR)
arc index text pdf ./research-papers --index research-pdfs

# 4. Search PDFs with exact phrases
arc search text '"gradient descent algorithm"' --index research-pdfs \
  --filter 'page_number > 5'

# 5. Combined workflow (semantic + exact)
# Step 1: Semantic discovery
arc search semantic "machine learning optimization" --collection research-papers

# Step 2: Exact verification on relevant file
arc search text '"stochastic gradient descent"' --index research-pdfs \
  --filter 'file_path = paper.pdf'
```

**Dual Indexing Example (with RDR-009):**

```bash
# 1. Create corpus (both Qdrant + MeiliSearch)
arc corpus create research-pdfs --type pdf --model nomic-embed-text-v1.5

# 2. Sync directory (indexes to both systems)
arc corpus sync ./research-papers --corpus research-pdfs

# Now searchable via both:
arc search semantic "query" --collection research-pdfs
arc search text '"exact phrase"' --index research-pdfs
```

**Standalone Indexing (without dual corpus):**

```bash
# MeiliSearch-only (no embeddings required)
arc indexes create my-pdfs --type pdf
arc index text pdf ./documents --index my-pdfs

# Qdrant-only (semantic search)
arc collection create my-pdfs --type pdf --model nomic-embed-text-v1.5
arc index semantic pdf ./documents --collection my-pdfs
```

## Alternatives Considered

### Alternative 1: Document-Level Indexing (No Page Granularity)

**Description**: Index entire PDF as single MeiliSearch document

**Pros:**

- ✅ Fewer documents (1 per PDF vs N pages)
- ✅ Simpler structure
- ✅ Lower indexing overhead

**Cons:**

- ❌ No page-level precision (can't return "page 7")
- ❌ Large documents may hit size limits
- ❌ Search results less actionable (no citation location)
- ❌ Doesn't align with RDR-004's page-level chunking

**Reason for rejection**: Page-level precision is critical for research papers and
technical documentation. Users need exact page numbers for citations.

### Alternative 2: Separate Extraction Pipeline (Don't Reuse RDR-004)

**Description**: Build new PDF extraction code specific to MeiliSearch

**Pros:**

- ✅ Optimized specifically for full-text search
- ✅ No dependency on RDR-004 implementation

**Cons:**

- ❌ **Code duplication** (PyMuPDF + OCR logic repeated)
- ❌ **Maintenance burden** (two pipelines to maintain)
- ❌ **Inconsistent extraction** (different text for same PDF)
- ❌ **Wasted effort** (RDR-004 is production-ready)

**Reason for rejection**: Violates DRY principle. RDR-004 extraction is robust and
tested. Reusing it ensures consistency and reduces maintenance.

### Alternative 3: Token-Aware Chunking (Like RDR-004)

**Description**: Chunk PDFs by token limits (512-1024) instead of pages

**Pros:**

- ✅ Consistent with Qdrant indexing (RDR-004)
- ✅ Smaller chunks

**Cons:**

- ❌ **No page boundaries** (chunks span pages)
- ❌ **No page numbers** (can't return "page 7")
- ❌ **MeiliSearch doesn't use embeddings** (token limits irrelevant)
- ❌ **Breaks cooperative workflow** (Qdrant page 7 ≠ MeiliSearch results)

**Reason for rejection**: MeiliSearch doesn't generate embeddings, so token limits
don't apply. Page-level indexing preserves document structure for citations.

### Alternative 4: Elasticsearch Instead of MeiliSearch

**Description**: Use Elasticsearch for full-text search instead of MeiliSearch

**Pros:**

- ✅ More mature ecosystem
- ✅ Advanced analytics

**Cons:**

- ❌ **54.8x more memory** (2-4GB vs 96-200MB)
- ❌ **Complex setup** (SSL, JVM tuning)
- ❌ **Against RDR-008 decision** (MeiliSearch chosen for simplicity)

**Reason for rejection**: RDR-008 already decided on MeiliSearch. This RDR builds on that foundation.

## Trade-offs and Consequences

### Positive Consequences

1. **100% Code Reuse**: RDR-004 extraction pipeline fully reused
2. **Page-Level Precision**: Exact page numbers for citations
3. **Shared Metadata**: Cooperative workflows with Qdrant (RDR-009)
4. **Change Detection**: File hash-based sync prevents duplicate indexing
5. **Production-Ready OCR**: Tesseract + EasyOCR from RDR-004
6. **Batch Efficiency**: 1000 documents per batch (MeiliSearch optimized)
7. **Simple Integration**: Single CLI command for PDF indexing

### Negative Consequences

1. **More Documents**: Page-level indexing creates N documents per PDF
   - *Mitigation*: MeiliSearch handles large document counts efficiently
   - *Benefit*: Precise page-level search results

2. **No Token-Aware Chunking**: Pages may be very long or very short
   - *Mitigation*: MeiliSearch has no token limits (text-only)
   - *Benefit*: Simpler logic, preserves page boundaries

3. **Dependency on RDR-004**: Requires RDR-004 extraction classes
   - *Mitigation*: RDR-004 is stable and well-tested
   - *Benefit*: Consistent extraction across systems

### Risks and Mitigations

**Risk**: Page splitting logic fails for complex PDFs

**Mitigation**:

- PDFExtractor already processes page-by-page (RDR-004)
- Modify `PDFExtractor.extract()` to return `List[PageText]` instead of concatenated string
- Fallback: Use form feed character (`\f`) as page delimiter

**Risk**: File hash computation slow for large PDFs

**Mitigation**:

- Stream file in 8KB chunks (not loading entire PDF into memory)
- Cache file hashes in local database (future optimization)
- Change detection is optional (can disable with `--no-sync` flag)

**Risk**: OCR processing bottleneck for large collections

**Mitigation**:

- OCR is optional (`--no-ocr` flag)
- Parallel processing via multiprocessing (future optimization)
- Skip OCR if PDF already has text (< 100 char threshold)

**Risk**: MeiliSearch index size grows large

**Mitigation**:

- Monitor disk usage (MeiliSearch data in `./meili_data`)
- Implement index cleanup commands (delete old documents)
- Document retention policies in RDR

## Implementation Plan

### Prerequisites

- [x] RDR-004: PDF extraction pipeline (PyMuPDF + OCR) - **REUSE**
- [x] RDR-008: MeiliSearch server setup - **DEPENDENCY**
- [x] RDR-009: Dual indexing patterns - **REFERENCE**
- [ ] Python >= 3.12
- [ ] meilisearch-python >= 0.31.0 installed
- [ ] RDR-004 dependencies installed (PyMuPDF, Tesseract, etc.)

### Step-by-Step Implementation

#### Step 1: Create PDFFullTextIndexer Class

Create `src/arcaneum/indexing/fulltext/pdf_indexer.py`:

- Import RDR-004 extraction classes (`PDFExtractor`, `OCREngine`)
- Implement `index_pdf()` method (extract + prepare + upload)
- Implement `_build_meilisearch_documents()` (page-level documents)
- Implement `_compute_file_hash()` (SHA-256 for change detection)
- Implement `_is_already_indexed()` (query MeiliSearch)
- Implement `index_directory()` (batch processing with progress)

**Estimated effort**: 6 hours

#### Step 2: Modify RDR-004 PDFExtractor for Page-Level Output

**Current** (RDR-004):

```python
def extract(self, pdf_path: Path) -> Tuple[str, dict]:
    # Returns concatenated text
    text = '\n'.join(text_parts)
    return text, metadata
```

**Modified** (this RDR):

```python
def extract(self, pdf_path: Path) -> Tuple[List[PageText], dict]:
    # Returns list of PageText objects
    pages = [
        PageText(page_num=1, text="...", metadata={...}),
        PageText(page_num=2, text="...", metadata={...}),
        ...
    ]
    return pages, metadata
```

**Estimated effort**: 2 hours

#### Step 3: Implement CLI Command

Create `src/arcaneum/cli/index_text.py`:

- `arc index text pdf` command (subcommand under `arc index text`)
- Directory argument + options (--index, --recursive, --ocr, --batch-size, --force)
- MeiliSearch client initialization (via `get_meilisearch_api_key()`)
- Index existence check + creation with PDF_DOCS_SETTINGS
- Call `PDFFullTextIndexer.index_directory()`
- Print statistics
- Mirror patterns from `cli/index_pdfs.py` (semantic indexing)

**Estimated effort**: 3 hours

#### Step 4: Implement Change Detection

Create `src/arcaneum/indexing/fulltext/sync.py`:

- `_is_already_indexed()` implementation
- File hash computation and comparison
- MeiliSearch query for existing documents
- Skip logic based on file_hash match

**Estimated effort**: 2 hours

#### Step 5: Integration with RDR-009 Dual Indexing

Update `src/arcaneum/cli/sync.py` (from RDR-009):

- Add `--text-only` flag (index only to MeiliSearch, skip Qdrant)
- Integrate `PDFFullTextIndexer` as MeiliSearch backend
- Dual indexing workflow: Extract once → Index to both
- Note: `corpus sync` uses **chunk-level** docs (consistent with Qdrant)
- Note: `arc index text pdf` uses **page-level** docs (this RDR)

**Estimated effort**: 4 hours

#### Step 6: Testing

Create comprehensive tests:

- Unit tests for `PDFFullTextIndexer`:
  - Page-level document building
  - File hash computation
  - Change detection logic
- Integration tests:
  - End-to-end PDF indexing
  - MeiliSearch query verification
  - Dual indexing with RDR-009
- Test scenarios:
  - Text PDFs (machine-generated)
  - Image PDFs (scanned, OCR required)
  - Mixed PDFs (some pages scanned)
  - Large PDFs (>100 pages)

**Estimated effort**: 6 hours

#### Step 7: Documentation

Update documentation:

- README with full-text PDF indexing examples
- CLI reference for `index-pdfs-fulltext` command
- Cooperative workflow guide (semantic → exact)
- Troubleshooting guide (OCR issues, index size management)

**Estimated effort**: 3 hours

### Total Estimated Effort

**26 hours** (~3 days of focused work)

**Effort Breakdown:**

- Step 1: PDFFullTextIndexer (6h)
- Step 2: Modify RDR-004 extractor (2h)
- Step 3: CLI command (3h)
- Step 4: Change detection (2h)
- Step 5: RDR-009 integration (4h)
- Step 6: Testing (6h)
- Step 7: Documentation (3h)

### Files to Create

**New Modules:**

- `src/arcaneum/indexing/fulltext/__init__.py`
- `src/arcaneum/indexing/fulltext/pdf_indexer.py` - PDFFullTextIndexer class
- `src/arcaneum/indexing/fulltext/sync.py` - Change detection logic
- `src/arcaneum/cli/index_text.py` - CLI command (`arc index text pdf`)

**Tests:**

- `tests/indexing/fulltext/test_pdf_indexer.py` - Indexer tests
- `tests/integration/test_pdf_fulltext_indexing.py` - End-to-end tests

### Files to Modify

**Existing Modules:**

- `src/arcaneum/indexing/pdf/extractor.py` - Add page-level output option
- `src/arcaneum/cli/sync.py` (from RDR-009) - Add `--text-only` flag
- `src/arcaneum/cli/main.py` - Register `arc index text` subcommand group
- `src/arcaneum/cli/fulltext.py` - Rename to `indexes.py` (arcaneum-h6bo)
- `src/arcaneum/fulltext/indexes.py` - Update PDF_DOCS_SETTINGS (arcaneum-i0be)
- `README.md` - Add full-text PDF indexing examples
- `CLAUDE.md` - Update CLI quick reference

### Dependencies

Already satisfied by RDR-004 and RDR-008:

- PyMuPDF (fitz) >= 1.23.0
- pdfplumber >= 0.10.0
- pytesseract >= 0.3.10
- easyocr >= 1.7.0 (optional)
- pdf2image >= 1.16.0
- opencv-python-headless >= 4.7.0
- meilisearch-python >= 0.31.0
- tenacity >= 8.2.0
- tqdm >= 4.65.0
- rich >= 13.0.0

## Validation

### Testing Approach

1. **Unit Tests**: Test individual components (file hash, document building)
2. **Integration Tests**: Test complete PDF → MeiliSearch workflow
3. **Dual Indexing Tests**: Verify RDR-009 integration (both systems)
4. **Change Detection Tests**: Verify skip logic for already-indexed PDFs
5. **OCR Tests**: Test Tesseract + EasyOCR on scanned PDFs

### Test Scenarios

#### Scenario 1: Text PDF Indexing

- **Setup**: Machine-generated PDF with embedded text
- **Action**: `arc index text pdf ./docs --index test-pdfs`
- **Expected**:
  - PDF extracted via PyMuPDF (no OCR)
  - Page-level documents created (1 per page)
  - Uploaded to MeiliSearch in batches of 1000
  - Searchable via exact phrase queries

#### Scenario 2: Scanned PDF with OCR

- **Setup**: Image-only PDF (scanned document)
- **Action**: `arc index text pdf ./scans --index test-pdfs`
- **Expected**:
  - PDF extracted via PyMuPDF (< 100 chars)
  - OCR triggered (Tesseract)
  - OCR confidence scores stored in metadata
  - Searchable via exact phrase queries

#### Scenario 3: Change Detection (Already Indexed)

- **Setup**: PDF already indexed with file_hash in MeiliSearch
- **Action**: Re-run `arc index text pdf ./docs --index test-pdfs`
- **Expected**:
  - File hash computed
  - MeiliSearch queried for existing document
  - PDF skipped (already indexed)
  - Stats show 0 indexed, 1 skipped

#### Scenario 4: Cooperative Workflow (Semantic → Exact)

- **Setup**: PDF indexed to both Qdrant (RDR-004) and MeiliSearch (this RDR)
- **Action**:
  1. `arc search semantic "machine learning" --collection research-pdfs`
  2. Note file_path from results (e.g., `ml-paper.pdf`)
  3. `arc search text '"neural network architecture"' --index research-pdfs --filter 'file_path = ml-paper.pdf'`
- **Expected**:
  - Semantic search returns relevant PDF
  - Exact search finds specific phrase in that PDF
  - Page numbers returned for citation

#### Scenario 5: Large PDF Collection (1000+ PDFs)

- **Setup**: Directory with 1000+ PDFs
- **Action**: `arc index text pdf ./large-collection --index test-pdfs`
- **Expected**:
  - Progress bar shows indexing status
  - Batches of 1000 documents uploaded
  - All PDFs indexed successfully
  - MeiliSearch index size < 10GB

### Performance Validation

**Metrics:**

- PDF extraction speed: ~0.003s per page (PyMuPDF)
- OCR speed: ~2s per page (Tesseract CPU)
- MeiliSearch upload: 1000 documents per batch
- Indexing throughput: ~100-200 PDFs/minute (no OCR)
- Indexing throughput: ~10-20 PDFs/minute (with OCR)

**Benchmarks:**

- Index 100 text PDFs: ~5 minutes
- Index 100 scanned PDFs (OCR): ~30 minutes
- Search latency: < 50ms (typical query)
- Index size: ~10MB per 100 pages (text-only, no vectors)

### Security Validation

- No hardcoded credentials (use MEILI_MASTER_KEY env var)
- File hash verification prevents tampering
- MeiliSearch API key required for indexing
- Volume permissions: User-only write (0755)

## References

### Related RDRs

- [RDR-004: PDF Bulk Indexing](RDR-004-pdf-bulk-indexing.md) - **PRIMARY DEPENDENCY** (extraction pipeline)
- [RDR-008: Full-Text Search Server Setup](RDR-008-fulltext-search-server-setup.md) - MeiliSearch deployment
- [RDR-009: Dual Indexing Strategy](RDR-009-dual-indexing-strategy.md) - Shared metadata patterns

### Beads Issues

- [arcaneum-69](../.beads/arcaneum.db) - Original RDR request

### Official Documentation

- **MeiliSearch Documentation**: <https://www.meilisearch.com/docs>
- **meilisearch-python Client**: <https://github.com/meilisearch/meilisearch-python>
- **PyMuPDF Documentation**: <https://pymupdf.readthedocs.io/>
- **Tesseract OCR**: <https://github.com/tesseract-ocr/tesseract>

## Notes

### Key Design Decisions

1. **Reuse RDR-004 Extraction 100%**: No code duplication, consistent extraction
2. **Page-Level Indexing**: Precise page numbers for citations
3. **Shared Metadata Schema**: Cooperative workflows with Qdrant (RDR-009)
4. **File Hash Change Detection**: Idempotent re-indexing
5. **Batch Size 1000**: Optimized for MeiliSearch (vs 100-200 for Qdrant)
6. **Symmetric CLI Naming**: `arc index text pdf` mirrors `arc index semantic pdf`
7. **Standalone Option**: Full-text indexing without requiring embeddings/Qdrant

### Future Enhancements

**Hybrid Search:**

- Combine semantic (Qdrant) + exact (MeiliSearch) results
- Reciprocal Rank Fusion (RRF) for result merging
- `arc find research-pdfs "query" --hybrid`

**Parallel OCR:**

- Multiprocessing for OCR-heavy workloads
- GPU acceleration with EasyOCR
- 10x speed improvement for scanned PDFs

**Advanced Change Detection:**

- Local SQLite cache for file hashes (avoid re-computation)
- Incremental sync (index only new/modified PDFs)
- Watch mode: `arcaneum index-pdfs-fulltext --watch`

### Known Limitations

- **Page splitting**: Assumes form feed character (`\f`) as page delimiter
  - *Solution*: Modify RDR-004 `PDFExtractor` to return page-level data
- **Large pages**: No maximum page size limit
  - *Mitigation*: MeiliSearch has no document size limit
- **OCR bottleneck**: Single-threaded OCR processing
  - *Mitigation*: Parallel OCR in future enhancement

### Success Criteria

- ✅ 100% reuse of RDR-004 extraction pipeline
- ✅ Page-level indexing (1 document per page)
- ✅ Shared metadata with Qdrant (RDR-009)
- ✅ File hash-based change detection
- ✅ Batch upload 1000 documents per batch
- ✅ CLI command: `arc index text pdf` (symmetric with `arc index semantic pdf`)
- ✅ Management commands: `arc indexes list/create/delete`
- ✅ Cooperative workflow: Semantic → Exact search
- ✅ Standalone indexing: MeiliSearch-only without embeddings
- ✅ Implementation < 30 hours
- ✅ Markdownlint compliant

This RDR provides the complete specification for indexing PDFs to MeiliSearch
for full-text exact phrase and keyword search, complementary to Qdrant's
semantic search (RDR-004), reusing the robust extraction pipeline and
maintaining shared metadata for cooperative workflows.

## Appendix: CLI Command Summary

After implementation (including arcaneum-h6bo rename):

```bash
# Index content
arc index semantic pdf /path --collection X   # Qdrant (vectors)
arc index semantic code /path --collection X
arc index semantic markdown /path --collection X
arc index text pdf /path --index X            # MeiliSearch (full-text)
arc index text code /path --index X           # Future
arc index text markdown /path --index X       # Future

# Search
arc search semantic "query" --collection X    # Vector similarity
arc search text "query" --index X             # Keyword/phrase match

# Manage Qdrant
arc collection list
arc collection create NAME --type TYPE --model MODEL
arc collection delete NAME

# Manage MeiliSearch
arc indexes list
arc indexes create NAME --type TYPE
arc indexes delete NAME
arc indexes info NAME

# Dual indexing (both systems)
arc corpus create NAME --type TYPE --model MODEL
arc corpus sync /path --corpus NAME
arc corpus sync /path --corpus NAME --text-only  # MeiliSearch only
```

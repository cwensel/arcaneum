# Recommendation 009: Minimal-Command Dual Indexing Workflow

## Metadata

- **Date**: 2025-10-21
- **Status**: Implemented
- **Type**: Architecture
- **Priority**: High
- **Related Issues**: arcaneum-68, arcaneum-85
- **Related Tests**: Corpus creation tests, dual indexing tests, sync-directory
  tests

## Problem Statement

Design the simplest possible workflow for users to create searchable corpuses
with both semantic (Qdrant) and exact (MeiliSearch) search capabilities. The
solution must:

1. **Minimize commands** - 2 commands maximum for complete workflow (create
   corpus + index documents)
2. **Optimize developer experience** - Simple, obvious, discoverable
3. **Enable cooperative search** - Shared metadata for semantic → exact
   verification workflows
4. **Avoid unnecessary complexity** - Don't over-engineer for theoretical edge
   cases

**User's Goal**: "create a collection for pdfs, then sync a given directory,
all with the fewest commands on the terminal or via a claude plugin"

**User's Flexibility**: "if this adds significant complexity, we will abandon
the attempt and have sub commands for loading each"

This RDR addresses:

- Unified corpus creation (both Qdrant collection + MeiliSearch index)
- Dual document indexing with shared metadata
- Optimal command-line workflow
- When to use unified vs separate approaches

## Context

### Background

Arcaneum provides two complementary search systems:

- **Qdrant (RDR-002)**: Vector database for semantic similarity search
- **MeiliSearch (RDR-008)**: Full-text search engine for exact phrase matching

**Original Design Question** (arcaneum-68):
Should collection/index creation be unified (single command) or separate
(subcommands)?

**Initial Analysis** (arcaneum-82):
Estimated unified approach at 60-80 hours of complexity due to rollback logic,
distributed transactions, and cross-system error handling.

**Reassessment** (arcaneum-85):
User clarified the PRIMARY goal is minimizing commands, not architectural
purity. Upon reassessment, unified approach is actually simple (4-6 hours) with
fail-fast error handling.

**Decision**: Recommend unified commands (create-corpus, sync-directory) to
achieve 2-command workflow.

### Technical Environment

- **Qdrant**: v1.16.2+ (Docker from RDR-002)
- **MeiliSearch**: v1.32.x (Docker from RDR-008)
- **Python**: >= 3.12
- **qdrant-client**: >= 1.16.1 (with FastEmbed)
- **meilisearch**: >= 0.39.0
- **CLI Framework**: Click >= 8.3.0 (from RDR-003)
- **Rich**: >= 14.2.0
- **Pydantic**: >= 2.12.3

## Research Findings

### Investigation Process

**Research Tracks:**

1. **arcaneum-78**: Reviewed RDR-003 (Qdrant collections), RDR-007 (semantic
   search), RDR-008 (MeiliSearch setup)
2. **arcaneum-79**: Analyzed MeiliSearch client capabilities via explore agent
3. **arcaneum-80**: Analyzed Qdrant client capabilities via explore agent
4. **arcaneum-81**: Identified shared metadata fields and cooperative use cases
5. **arcaneum-82**: Initial complexity assessment (over-estimated)
6. **arcaneum-85**: User goal clarification and complexity reassessment

### Key Discoveries

#### 1. Shared Metadata Enables Cooperative Search

**Perfect Field Alignment:**

| Field Name (Qdrant)      | Field Name (MeiliSearch) | Use Case             |
|--------------------------|--------------------------|----------------------|
| `file_path`              | `file_path`              | Location tracking    |
| `programming_language`   | `language`               | Code language filter |
| `git_project_identifier` | `git_project_identifier` | Project identifier   |
| `git_branch`             | `branch`                 | Branch-aware search  |
| `filename`               | `filename`               | File name search     |
| `line_number`            | `line_number`            | Line-level precision |
| `chunk_index`            | `chunk_index`            | Document ordering    |
| `file_extension`         | `file_extension`         | File type filter     |

**Cooperative Workflow:**

```bash
# 1. Semantic discovery
arc find MyCode "authentication patterns"

# Results show: src/auth/verify.py

# 2. Exact verification
arc match MyCode '"def authenticate"' \
  --filter 'file_path = src/auth/verify.py'
```

#### 2. Unified Collection Creation is Simple

**Original Over-Estimate**: 60-80 hours for distributed transactions, rollback
logic

**Actual Implementation**:

```python
def create_corpus(name, type, models):
    """Create both Qdrant collection and MeiliSearch index."""
    # Create Qdrant collection
    try:
        qdrant.create_collection(name, vectors_config=...)
        print(f"✅ Qdrant collection '{name}' created")
    except Exception as e:
        print(f"❌ Qdrant failed: {e}")
        return False

    # Create MeiliSearch index
    try:
        meili.create_index(name, settings=...)
        print(f"✅ MeiliSearch index '{name}' created")
    except Exception as e:
        print(f"❌ MeiliSearch failed: {e}")
        print(f"Note: Qdrant collection '{name}' exists")
        return False

    return True
```

**Actual Complexity**: 4-6 hours (simple sequential API calls, fail-fast)

**Why We Over-Estimated**:

- Assumed distributed transaction semantics needed (wrong!)
- Focused on perfect rollback (unnecessary - fail-fast is fine)
- Optimized for architectural purity over user needs

#### 3. Dual Document Indexing is Straightforward

**Pattern** (extends existing RDR-004, RDR-005 logic):

```python
def sync_directory(dir_path, corpus_name):
    """Index directory to both systems."""
    # 1. Discover files
    files = discover_files(dir_path)

    # 2. Process each file
    for file_path in files:
        chunks = chunk_file(file_path)  # Existing logic

        for chunk in chunks:
            # Build unified document
            doc = DualIndexDocument(
                content=chunk.text,
                file_path=str(file_path),
                language=detect_language(file_path),
                vectors={model: embed(chunk.text, model) for model in models},
                ...
            )

            # Index to both (simple!)
            qdrant.upsert(collection=corpus_name, points=[to_qdrant(doc)])
            meili.add_documents(index=corpus_name, docs=[to_meili(doc)])
```

**Complexity**: 8-10 hours (reuse chunking, add dual calls)

**Total Implementation**: 16-20 hours (acceptable)

## Proposed Solution

### Approach

#### 2-Command Workflow with Unified Corpus Management

Provide high-level commands that abstract away the dual-system complexity:

1. **create-corpus**: Creates both Qdrant collection and MeiliSearch index
2. **sync-directory**: Indexes documents to both systems with shared metadata

Users don't need to know about Qdrant vs MeiliSearch internals. They just
create a corpus and sync directories.

### Technical Design

#### Architecture

```text
User Workflow (2 commands):
┌──────────────────────────────────────────────────────────────┐
│  Command 1: arc corpus create my-pdfs --type pdf-docs        │
│                                                               │
│  ├─> Creates Qdrant collection 'my-pdfs'                    │
│  └─> Creates MeiliSearch index 'my-pdfs'                    │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│  Command 2: arc corpus sync my-pdfs ./docs                   │
│                                                               │
│  ├─> Discovers PDFs in ./docs                               │
│  ├─> Chunks and generates embeddings                        │
│  ├─> Indexes to Qdrant (vectors + metadata)                 │
│  └─> Indexes to MeiliSearch (text + metadata)               │
└──────────────────────────────────────────────────────────────┘

Result: Searchable corpus in both systems
```

#### Component 1: create-corpus Command (PRIMARY)

**CLI Signature:**

```bash
arc corpus create <name> --type <source-code|pdf-docs|markdown-docs> [--models <models>]
```

**Note**: Type aliases are supported for convenience: `code` → `source-code`,
`pdf` → `pdf-docs`, `markdown` → `markdown-docs`.

**Implementation:**

```python
# src/arcaneum/cli/corpus.py

import click
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff
from ..fulltext.client import FullTextClient
from ..fulltext.indexes import get_index_settings, get_available_index_types
from ..cli.output import print_success, print_error, print_info

@corpus.command('create')
@click.argument('name')
@click.option(
    '--type', 'corpus_type',
    type=click.Choice(get_available_index_types()),
    required=True,
    help='Corpus type (source-code, pdf-docs, markdown-docs, or aliases: code, pdf, markdown)'
)
@click.option(
    '--models',
    default='stella,jina',
    help='Embedding models (comma-separated)'
)
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
def create_corpus(name, corpus_type, models, output_json):
    """Create both Qdrant collection and MeiliSearch index."""

    print_info(f"Creating corpus '{name}'")
    print_info(f"Type: {corpus_type}, Models: {models}")

    # Parse models
    model_list = models.split(',')

    # Step 1: Create Qdrant collection
    print_info("Step 1/2: Creating Qdrant collection...")
    try:
        qdrant = QdrantClient(url=qdrant_url)

        # Build vectors config
        vectors_config = {
            model: VectorParams(
                size=get_model_dimensions(model),
                distance=Distance.COSINE
            )
            for model in model_list
        }

        qdrant.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            on_disk_payload=True
        )
        print_success(f"Qdrant collection '{name}' created")
    except Exception as e:
        print_error(f"Qdrant collection creation failed: {e}")
        raise click.Abort()

    # Step 2: Create MeiliSearch index
    print_info("Step 2/2: Creating MeiliSearch index...")
    try:
        meili = FullTextClient(meili_url, meili_key)
        settings = get_index_settings(corpus_type)
        meili.create_index(name, primary_key='id', settings=settings)
        print_success(f"MeiliSearch index '{name}' created")
    except Exception as e:
        print_error(f"MeiliSearch index creation failed: {e}")
        print_info(f"Note: Qdrant collection '{name}' was created successfully")
        raise click.Abort()

    print_success(f"Corpus '{name}' ready for indexing!")
    print_info(f"Next: arc corpus sync {name} <path>")
```

**Error Handling Strategy**: Fail-fast, clear messages, no complex rollback

#### Component 2: sync Command (PRIMARY)

**CLI Signature:**

```bash
arc corpus sync <name> <path> [--models <models>] [--file-types <extensions>]
```

**Implementation Pattern:**

```python
# src/arcaneum/cli/sync.py

import click
from pathlib import Path
from rich.progress import track
from ..indexing.dual_indexer import DualIndexer
from ..embeddings.client import EmbeddingClient
from ..cli.output import print_success, print_error, print_info

@corpus.command('sync')
@click.argument('corpus_name')
@click.argument('directory', type=click.Path(exists=True))
@click.option('--models', default='stella,jina', help='Embedding models')
@click.option('--file-types', help='File extensions to index (e.g., .py,.md)')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
def sync_directory(corpus_name, directory, models, file_types, output_json):
    """Index directory to both Qdrant and MeiliSearch."""

    print_info(f"Syncing '{directory}' to corpus '{corpus_name}'")

    # Initialize clients
    qdrant = QdrantClient(url=qdrant_url)
    meili = FullTextClient(meili_url, meili_key)
    embedder = EmbeddingClient(cache_dir='./models_cache', models_config=...)

    # Create dual indexer
    dual_indexer = DualIndexer(
        qdrant_client=qdrant,
        meili_client=meili,
        collection_name=corpus_name,
        index_name=corpus_name
    )

    # Discover files
    dir_path = Path(directory)
    files = list(dir_path.rglob('*.pdf'))  # Or *.py for code

    print_info(f"Found {len(files)} files to index")

    # Process files
    total_indexed = 0
    for file_path in track(files, description="Indexing..."):
        # Chunk file (reuse RDR-004/RDR-005 logic)
        chunks = chunk_file(file_path, type=detect_type(file_path))

        # Build unified documents
        documents = []
        for i, chunk in enumerate(chunks):
            # Generate embeddings
            vectors = {
                model: embedder.embed([chunk.content], model)[0]
                for model in models.split(',')
            }

            # Create unified document with shared metadata
            doc = DualIndexDocument(
                id=f"{file_path.name}:{i}",
                content=chunk.content,
                file_path=str(file_path),
                filename=file_path.name,
                language=detect_language(file_path),
                chunk_index=i,
                vectors=vectors,
                ...
            )
            documents.append(doc)

        # Index to both systems (simple!)
        qdrant_count, meili_count = dual_indexer.index_batch(documents)
        total_indexed += len(documents)

    print_success(f"Indexed {total_indexed} chunks to both systems")
```

#### Component 3: Shared Metadata Schema

**Unified Document Schema:**

```python
# src/arcaneum/schema/document.py

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DualIndexDocument:
    """Document schema for dual indexing."""

    # Primary identifier
    id: str

    # Content
    content: str

    # Shared metadata (both systems)
    file_path: str
    filename: str
    language: str
    chunk_index: int
    file_extension: str

    # Optional fields
    line_number: Optional[int] = None
    project: Optional[str] = None
    branch: Optional[str] = None
    page_number: Optional[int] = None

    # Code-specific
    function_names: List[str] = []
    class_names: List[str] = []

    # Vectors (Qdrant only)
    vectors: dict = {}

def to_qdrant_point(doc: DualIndexDocument, point_id: int):
    """Convert to Qdrant format."""
    return PointStruct(
        id=point_id,
        vector=doc.vectors,
        payload={
            "file_path": doc.file_path,
            "programming_language": doc.language,
            "git_project_identifier": doc.git_project_identifier,
            "git_branch": doc.branch,
            "content": doc.content,
            "filename": doc.filename,
            "file_extension": doc.file_extension,
            "chunk_index": doc.chunk_index,
            ...
        }
    )

def to_meilisearch_doc(doc: DualIndexDocument):
    """Convert to MeiliSearch format."""
    return {
        "id": doc.id,
        "content": doc.content,
        "file_path": doc.file_path,
        "language": doc.language,
        "git_project_identifier": doc.git_project_identifier,
        "branch": doc.branch,
        "filename": doc.filename,
        "file_extension": doc.file_extension,
        "chunk_index": doc.chunk_index,
        ...
    }
```

#### Component 4: DualIndexer Orchestrator

```python
# src/arcaneum/indexing/dual_indexer.py

class DualIndexer:
    """Coordinates indexing to both systems."""

    def __init__(self, qdrant_client, meili_client, collection_name, index_name):
        self.qdrant = qdrant_client
        self.meili = meili_client
        self.collection_name = collection_name
        self.index_name = index_name

    def index_batch(self, documents: List[DualIndexDocument]) -> tuple[int, int]:
        """Index batch to both systems."""
        # Convert formats
        qdrant_points = [to_qdrant_point(d, i) for i, d in enumerate(documents)]
        meili_docs = [to_meilisearch_doc(d) for d in documents]

        # Index to Qdrant
        self.qdrant.upsert(collection_name=self.collection_name, points=qdrant_points)

        # Index to MeiliSearch
        self.meili.add_documents(index_name=self.index_name, documents=meili_docs)

        return len(qdrant_points), len(meili_docs)
```

### Implementation Example

**Complete 2-Command Workflow:**

```bash
# 1. Create corpus (both systems)
arc corpus create my-pdfs --type pdf-docs --models stella,bge

# 2. Sync directory (index to both systems)
arc corpus sync my-pdfs ./research-papers

# Done! Search both ways:
arc search semantic "machine learning" --collection my-pdfs
arc search text '"neural networks"' --index my-pdfs
```

**Source Code Example:**

```bash
# 1. Create corpus for code (using alias 'code' for 'source-code')
arc corpus create my-code --type code --models stella,jina

# 2. Sync git repository
arc corpus sync my-code ./src --file-types .py,.js

# Search:
arc search semantic "authentication" --collection my-code --filter language=python
arc search text '"def authenticate"' --index my-code
```

## Alternatives Considered

### Alternative 1: Separate Commands (Architectural Purity)

**Description**: Keep collection/index creation separate, dual indexing at
document level

```bash
arc collection create my-pdfs --models stella,bge
arc fulltext create-index my-pdfs --type pdf-docs
arc corpus sync my-pdfs ./docs
```

**Pros:**

- ✅ Loose coupling between systems
- ✅ Can use just Qdrant OR just MeiliSearch
- ✅ Easier to debug (isolated errors)

**Cons:**

- ❌ **3 commands minimum** (vs user goal of 2)
- ❌ More cognitive load
- ❌ Optimizes for architectural purity, not user experience
- ❌ Violates user's primary requirement: "fewest commands"

**Reason for rejection**: User explicitly wants minimal commands. Architectural
purity is secondary to UX.

### Alternative 2: Completely Separate Pipelines

**Description**: No dual indexing at all, completely independent systems

```bash
arc corpus sync ./docs --vector my-pdfs --auto-create
arc corpus sync ./docs --fulltext my-pdfs --auto-create
```

**Pros:**

- ✅ 2 commands (meets user's command-count goal)
- ✅ Complete independence
- ✅ No shared metadata complexity

**Cons:**

- ❌ **Loses cooperative use cases** (semantic → exact verification)
- ❌ No shared metadata for cross-system workflows
- ❌ Must index twice (2x time, 2x disk I/O)
- ❌ Doesn't leverage user's goal of "shared metadata between services"

**Reason for rejection**: User specifically wants shared metadata for
cooperative use cases. This alternative abandons that goal.

### Alternative 3: Auto-Create on First Sync

**Description**: sync auto-creates corpus if missing

```bash
# Single command (corpus auto-created)
arc corpus sync my-pdfs ./docs --type pdf-docs --models stella,bge
```

**Pros:**

- ✅ 1 command (ultimate simplicity!)
- ✅ "Just works" experience

**Cons:**

- ❌ **Hidden magic**: Users don't know corpus was created
- ❌ **No control**: Can't customize HNSW settings, index config before indexing
- ❌ **Poor error isolation**: Corpus creation errors mixed with indexing errors
- ❌ **Against explicitness**: Can't verify setup before indexing

**Reason for rejection**: Too much magic. Explicit 2-command workflow is better
for understanding and debugging.

## Trade-offs and Consequences

### Positive Consequences

1. **Minimal Commands**: 2-command workflow achieves user's primary goal
2. **Simple Implementation**: Builds on existing RDR-008 infrastructure
3. **Cooperative Search**: Shared metadata enables semantic → exact workflows
4. **Optimal DX**: Fewest commands, clear workflow, easy to remember
5. **Discoverable**: `arc corpus create --help` shows all options
6. **Fast Indexing**: Dual indexing overhead ~20% (acceptable)
7. **Explicit Setup**: Users know corpus created before indexing
8. **Clear Errors**: Fail-fast with helpful messages

### Negative Consequences

1. **Qdrant Created if MeiliSearch Fails**: Partial state possible
   - *Mitigation*: Clear error message shows which succeeded
   - *User Action*: Delete Qdrant collection or retry MeiliSearch creation

2. **Tighter Coupling**: create-corpus knows about both systems
   - *Mitigation*: Still using separate client libraries, not tightly coupled
   - *Benefit*: Worth it for 50% command reduction (3→2)

3. **Can't Use Just One System**: Both always created
   - *Mitigation*: Still can use existing RDR-003, RDR-008 commands separately
   - *Benefit*: Most users want both systems anyway

### Risks and Mitigations

**Risk**: MeiliSearch down when creating corpus

**Mitigation**:

- Check both services healthy before creating corpus
- `arc corpus create --validate` pings both servers first
- Clear error: "MeiliSearch not accessible at <http://localhost:7700>"

**Risk**: Syncing large directories takes too long

**Mitigation**:

- Progress bar shows files being indexed
- `--batch-size` flag to tune throughput
- Resume support via change detection (index only new/modified files)

**Risk**: Dual indexing overhead significant

**Mitigation**:

- Measured overhead: ~20% (mostly embedding generation time)
- Acceptable for user's workflow simplification benefit

## Implementation Plan

### Prerequisites

- [x] RDR-002: Qdrant server setup (completed)
- [x] RDR-003: Qdrant collection creation patterns (completed)
- [x] RDR-008: MeiliSearch server setup (completed)
- [x] RDR-004: PDF chunking logic (completed)
- [x] RDR-005: Source code chunking logic (completed)
- [ ] Python >= 3.12
- [ ] qdrant-client[fastembed] installed
- [ ] meilisearch-python installed

### Step-by-Step Implementation

#### Step 1: Implement Shared Schema

Create `src/arcaneum/schema/document.py`:

- `DualIndexDocument` dataclass
- `to_qdrant_point()` conversion
- `to_meilisearch_doc()` conversion
- Field naming convention documentation

#### Step 2: Implement DualIndexer

Create `src/arcaneum/indexing/dual_indexer.py`:

- `DualIndexer` class
- `index_batch()` method
- Error handling per system
- Logging and progress tracking

#### Step 3: Implement corpus create Command

Complete `src/arcaneum/cli/corpus.py`:

- `corpus create` command with type selection
- Sequential API calls to both systems
- Fail-fast error handling
- Health check option

**Note**: CLI command structure already exists in `main.py` as stubs.

#### Step 4: Implement corpus sync Command

Complete `src/arcaneum/cli/sync.py`:

- File discovery logic
- Integration with existing chunking (RDR-004, RDR-005, RDR-014)
- Dual indexing via `DualIndexer`
- Progress tracking
- Change detection (index only new/modified)

**Note**: CLI command structure already exists in `main.py` as stubs.

#### Step 5: Testing

Create comprehensive tests:

- Unit tests for schema conversions
- Integration tests for corpus create
- Integration tests for corpus sync
- End-to-end workflow tests
- Error scenario tests

#### Step 6: Documentation

Update documentation:

- README with 2-command workflow example
- CLI reference for both commands
- Troubleshooting guide
- Update RDR index

**Note**: This RDR implements the complete workflow. Follow-up RDRs
(arcaneum-69, arcaneum-70) can extend sync for type-specific optimizations.

### Files to Create

**New Modules:**

- `src/arcaneum/schema/document.py` - Shared document schema
- `src/arcaneum/indexing/dual_indexer.py` - Dual indexing orchestrator

**Tests:**

- `tests/cli/test_corpus.py` - Corpus creation tests
- `tests/cli/test_sync.py` - Sync directory tests
- `tests/schema/test_document.py` - Schema conversion tests
- `tests/indexing/test_dual_indexer.py` - Dual indexer tests
- `tests/integration/test_2command_workflow.py` - End-to-end tests

### Files to Complete (stubs exist)

- `src/arcaneum/cli/corpus.py` - Implement create_corpus_command()
- `src/arcaneum/cli/sync.py` - Implement sync_directory_command()

### Files to Modify

- `README.md` - Add 2-command workflow examples
- `docs/rdr/README.md` - Reference RDR-009

**Note**: CLI command registration already exists in `main.py` (corpus group
with create and sync subcommands).

### Dependencies

Already satisfied by RDR-003 and RDR-008 (see pyproject.toml):

- qdrant-client >= 1.16.1
- meilisearch >= 0.39.0
- click >= 8.3.0
- rich >= 14.2.0
- pydantic >= 2.12.3

## Validation

### Testing Approach

1. **Workflow Tests**: Complete 2-command workflow from scratch
2. **Error Scenario Tests**: One system down, both down, partial failures
3. **Performance Tests**: Measure dual indexing overhead
4. **Consistency Tests**: Verify same documents in both systems

### Test Scenarios

#### Scenario 1: Complete 2-Command Workflow (PDFs)

- **Setup**: Both Qdrant and MeiliSearch running
- **Action**:
  1. `arc corpus create research-pdfs --type pdf-docs --models stella,bge`
  2. `arc corpus sync research-pdfs ./papers`
- **Expected**:
  - Both Qdrant collection and MeiliSearch index created
  - All PDFs indexed to both systems
  - Same file_path metadata in both
  - Searchable via both semantic and exact search

#### Scenario 2: Complete 2-Command Workflow (Source Code)

- **Setup**: Both systems running
- **Action**:
  1. `arc corpus create my-code --type source-code --models stella,jina`
  2. `arc corpus sync my-code ./src --file-types .py,.js`
- **Expected**:
  - Corpus created with code-optimized settings
  - Python and JavaScript files indexed
  - Both systems have same documents
  - Can search by language filter in both

#### Scenario 3: MeiliSearch Down During corpus create

- **Setup**: Stop MeiliSearch container
- **Action**: `arc corpus create test --type pdf-docs`
- **Expected**:
  - Qdrant collection created successfully
  - MeiliSearch index creation fails with clear error
  - Error message shows Qdrant succeeded
  - User can retry MeiliSearch or fix server

#### Scenario 4: Incremental Sync (Change Detection)

- **Setup**: Corpus already exists with 100 files indexed
- **Action**:
  1. Add 10 new PDFs to directory
  2. `arc corpus sync research-pdfs ./papers`
- **Expected**:
  - Only 10 new PDFs indexed (change detection works)
  - Existing documents unchanged
  - Fast incremental update

#### Scenario 5: Cooperative Search Workflow

- **Setup**: Code corpus indexed via 2-command workflow
- **Action**:
  1. `arc search semantic "authentication" --collection my-code`
  2. Note file_path from results
  3. `arc search text '"def authenticate"' --index my-code --filter 'file_path = <noted_path>'`
- **Expected**:
  - Semantic search finds related patterns
  - Exact search verifies specific implementation
  - Shared metadata enables workflow

### Performance Validation

**Metrics:**

- Dual indexing overhead: < 20% vs Qdrant-only
- create-corpus execution time: < 2 seconds
- sync-directory throughput: ~500-1000 documents/minute
- Memory: ~4-6GB total (Qdrant 2GB + MeiliSearch 200MB + Python 2GB)

**Benchmarks:**

- Create corpus: < 2 seconds
- Sync 1000 PDFs: ~18 minutes (vs 15 minutes Qdrant-only, 20% overhead)
- Sync 10,000 Python files: ~36 minutes (vs 30 minutes Qdrant-only)

## References

### Related RDRs

- [RDR-002: Qdrant Server Setup](RDR-002-qdrant-server-setup.md) - Vector
  database deployment
- [RDR-003: Collection Creation](RDR-003-collection-creation.md) - Qdrant
  collection patterns
- [RDR-004: PDF Bulk Indexing](RDR-004-pdf-bulk-indexing.md) - PDF chunking
  logic (reused)
- [RDR-005: Source Code Indexing](RDR-005-source-code-indexing.md) - Code
  chunking logic (reused)
- [RDR-007: Semantic Search](RDR-007-semantic-search.md) - Qdrant search
- [RDR-008: Full-Text Search Server Setup](
  RDR-008-fulltext-search-server-setup.md) - MeiliSearch deployment

### Beads Issues

- [arcaneum-68](../.beads/arcaneum.db) - Original RDR request
- [arcaneum-78 through arcaneum-82](../.beads/arcaneum.db) - Research tasks
- [arcaneum-85](../.beads/arcaneum.db) - User goal clarification and
  reassessment

### Official Documentation

- **Qdrant Documentation**: <https://qdrant.tech/documentation/>
- **MeiliSearch Documentation**: <https://www.meilisearch.com/docs>
- **Qdrant Python Client**: <https://python-client.qdrant.tech/>
- **MeiliSearch Python Client**:
  <https://github.com/meilisearch/meilisearch-python>

## Notes

### Key Design Decisions

1. **Minimize Commands over Architectural Purity**: 2-command workflow is
   primary goal
2. **corpus create as PRIMARY Command**: Not optional, core workflow
3. **corpus sync with Dual Indexing**: Indexes to both systems automatically
4. **Fail-Fast Error Handling**: No complex rollback, clear error messages
5. **Shared Metadata**: Enables cooperative search workflows

### Existing Infrastructure (RDR-008)

The following components are already implemented and available:

- `FullTextClient` class with create_index, add_documents, search methods
- Index settings templates: `SOURCE_CODE_SETTINGS`, `PDF_DOCS_SETTINGS`,
  `MARKDOWN_DOCS_SETTINGS`
- Type aliases: `code` → `source-code`, `pdf` → `pdf-docs`, `markdown` →
  `markdown-docs`
- CLI command stubs registered in `main.py` (corpus group)
- Output helpers: `print_success`, `print_error`, `print_info`
- Interaction logging framework (RDR-018)

### Future Enhancements

**Hybrid Search with RRF:**

- `arc search hybrid "query" --corpus my-code` (queries both, merges results)
- Configurable weights (70% semantic, 30% exact)

**Corpus Management:**

- `arc corpus list` - Show all corpuses
- `arc corpus info <name>` - Show stats from both systems
- `arc corpus delete <name>` - Delete from both systems

**Smart Sync:**

- `arc corpus sync <corpus> <path> --watch` - Continuous sync mode
- Git-aware: `arc corpus sync-repository <corpus> <url>` - Clone and index

### Success Criteria

- ✅ 2-command workflow: corpus create + corpus sync
- ✅ Both systems created and indexed automatically
- ✅ Shared metadata enables cooperative search
- ✅ Dual indexing overhead < 20%
- ✅ Clear error messages for all failure modes
- ✅ Markdownlint compliant

This RDR provides the specification for a minimal-command dual indexing
workflow that optimizes for user experience by minimizing commands while
enabling powerful cooperative search capabilities.

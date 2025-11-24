# Recommendation 007: Semantic Search CLI for Qdrant Collections

## Metadata
- **Date**: 2025-10-20
- **Status**: Recommendation
- **Type**: Feature
- **Priority**: High
- **Related Issues**: arcaneum-7, arcaneum-52, arcaneum-53, arcaneum-54, arcaneum-56
- **Related Tests**: Search tests, embedding tests, filter tests

## Problem Statement

Create a command-line search tool that enables semantic search across Qdrant collections created by RDR-003/004/005. The tool must:

1. **Detect and use correct embedding models** from collection metadata
2. **Support flexible metadata filtering** with user-friendly DSL
3. **Format results for Claude Code UI** with file paths (line numbers deferred to full-text search)
4. **Integrate with RDR-006 pattern** for Claude Code slash command exposure
5. **Maintain simplicity** - single collection search, synchronous execution

This addresses the critical need for users to semantically search indexed content from both CLI and Claude Code, completing the indexing → search workflow. Multi-collection search deferred to future enhancement for simplicity.

## Context

### Background

Arcaneum provides bulk indexing tools via:
- **RDR-004**: PDF indexing with OCR
- **RDR-005**: Source code indexing with AST chunking
- **RDR-003**: Collection creation with named vectors

**The Missing Piece**: Users need to search indexed content
**The Workflow Gap**:
1. Users index PDFs and source code → collections created
2. **[MISSING]** Users search indexed content semantically
3. Results displayed with file paths for Claude Code navigation

**Key Design Questions** (from arcaneum-7):
- How to detect which embedding model a collection uses?
- What metadata filter DSL to provide users?
- How to format results for Claude Code UI?
- Pagination strategy?

**Scope Decision**: Multi-collection search deferred to future RDR for simplicity (v1 = single collection only)

### Technical Environment

- **Qdrant**: v1.15.4+ (Docker from RDR-002)
- **Python**: >= 3.12
- **qdrant-client**: >= 1.15.0
- **FastEmbed**: >= 0.3.0 (ONNX embeddings)
- **CLI Framework**: Click (from RDR-001)
- **Collections**: Created by RDR-003/004/005 with metadata

**Metadata Schema** (from RDR-003/004/005):
- `embedding_model`: "stella", "modernbert", "bge", "jina-code"
- `programming_language`: For source code collections
- `git_project_name`: Git repository name
- `file_path`: Absolute path to source file
- `chunk_index`: Position in document

## Research Findings

### Investigation Process

Research conducted via Beads issues arcaneum-52 through arcaneum-56:

1. **arcaneum-52**: Qdrant client search patterns - Analyzed qdrant-client codebase via Chroma collection
2. **arcaneum-53**: Query embedding strategy - Model detection, caching, named vectors
3. **arcaneum-54**: Metadata filtering DSL - User-friendly syntax + Qdrant mapping
4. **arcaneum-55**: Multi-collection search design - *Deferred to future enhancement for v1 simplicity*
5. **arcaneum-56**: Result formatting - File paths, snippets, Claude UI optimization

### Key Discoveries

#### 1. Qdrant Client Search API (arcaneum-52)

**Query Embedding Generation**:
- FastEmbed integration via `_get_or_init_model()` with caching
- Model initialization: `TextEmbedding(model_name, cache_dir, threads)`
- `query_embed()` for queries vs `embed()` for documents
- `@lru_cache` decorator for model instances

**Search Methods**:
```python
client.search(
    collection_name="MyCode",
    query_vector=(vector_name, query_vector),  # Named vector tuple
    query_filter=Filter(...),                   # Metadata filtering
    limit=10,
    score_threshold=0.7,
    with_payload=True
)
```

**Returns**: `list[ScoredPoint]` with score, id, payload, vectors

**Pagination**:
- `limit` for result count
- `offset` for pagination (warning: "large offset values may cause performance issues")

#### 2. Embedding Model Detection (arcaneum-53)

**Vector-Based Model Detection**:
- Collections use named vectors (from RDR-002/003)
- Vector names = model keys ("stella", "jina", "modernbert", "bge")
- Retrieved via: `client.get_collection(name).config.params.vectors.keys()`

**Model Detection Flow**:
```python
def detect_collection_model(
    client: QdrantClient,
    collection_name: str,
    vector_name: str = None
) -> str:
    collection_info = client.get_collection(collection_name)
    available_vectors = list(collection_info.config.params.vectors.keys())

    if not available_vectors:
        raise ValueError(f"Collection {collection_name} has no vectors")

    if vector_name:
        # Validate user-specified vector
        if vector_name not in available_vectors:
            raise ValueError(f"Vector '{vector_name}' not in {available_vectors}")
        return vector_name

    # Auto-select first vector (alphabetically)
    return sorted(available_vectors)[0]
```

**Key Insight**: Named vectors already encode the model information - no additional metadata needed!

**Model Caching**:
```python
@lru_cache(maxsize=4)  # Cache up to 4 models
def get_model(model_key: str) -> TextEmbedding:
    config = models_config[model_key]
    return TextEmbedding(
        model_name=config.name,
        cache_dir=str(cache_dir)
    )
```

**Named Vectors Support**:
- Collections can have multiple named vectors (stella, jina, etc.)
- Search specifies: `query_vector=(vector_name, vector_data)`
- Auto-selects first vector alphabetically if not specified
- User can override with `--vector-name` flag

#### 3. Metadata Filtering DSL (arcaneum-54)

**Two-Tier Approach**:

**Simple DSL** (80% use case):
```bash
arc find MyCollection "query" --filter language=python,git_project_name=myproj
```

**JSON DSL** (20% advanced):
```bash
arc find MyCollection "query" --filter '{
  "must": [{"key": "language", "match": {"value": "python"}}],
  "should": [{"key": "git_project", "match": {"any": ["proj1", "proj2"]}}]
}'
```

**Qdrant Mapping**:
```python
# Simple: key=value,key=value
def parse_simple_filter(filter_str: str) -> models.Filter:
    conditions = []
    for pair in filter_str.split(','):
        key, value = pair.split('=', 1)
        conditions.append(
            models.FieldCondition(key=key.strip(), 
                                match=models.MatchValue(value=value.strip()))
        )
    return models.Filter(must=conditions)
```

**Supported Operators**:
- `match`: Exact value
- `range`: Numeric/date ranges (gte, lt)
- `match_any`: Multiple values (OR)
- `match_text`: Text contains

#### 4. Result Formatting (arcaneum-56)

**File Path Format**: File paths (line numbers deferred to full-text search)
```
/Users/user/code/auth.py
/Documents/paper.pdf:page12
```

**Note**: Line numbers omitted for v1 simplicity. Semantic search returns chunks, not specific lines. Page numbers kept for PDFs (useful for navigation).

**Human-Readable Output**:
```
Searching for: "authentication patterns"
Found 5 results

[1] Score: 95% | Language: python | Project: myproject
    /Users/user/code/myproject/src/auth.py

    def authenticate_user(username, password):
        """Verify user credentials using bcrypt...
```

**JSON Output** (--json flag):
```json
{
  "query": "authentication patterns",
  "collections": ["MyCode"],
  "total_results": 5,
  "results": [{
    "score": 0.95,
    "location": "/path/to/file.py",
    "content": "def authenticate_user...",
    "metadata": {"language": "python", "git_project": "myproj"}
  }]
}
```

## Proposed Solution

### Approach

**CLI-First Design** (follows RDR-006 pattern):
```bash
# Basic search
arc find MyCode "authentication patterns"

# With filtering
arc find MyCode "auth" --filter language=python

# With limit
arc find MyCode "auth" --limit 20

# With pagination (second page)
arc find MyCode "auth" --limit 20 --offset 20

# JSON output
arc find MyCode "auth" --json
```

**Simplified Architecture** (single collection only):
```
┌─────────────────────────────────────────────────┐
│     Layer 1: CLI Entry Point                    │
│     python -m arcaneum.cli.main search          │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│     Layer 2: Search Orchestrator                │
│     - Detect model from collection metadata     │
│     - Generate query embedding                  │
│     - Parse filter                              │
│     - Execute single collection search          │
│     - Format results                            │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│     Layer 3: Qdrant Client                      │
│     - Execute search with filter                │
│     - Return ScoredPoint results                │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
              Qdrant Server
```

### Technical Design

#### Component 1: Query Embedding Pipeline

**Full implementation from arcaneum-53**:

```python
# src/arcaneum/search/embedder.py
from functools import lru_cache
from fastembed import TextEmbedding
from pathlib import Path

class SearchEmbedder:
    """Manages embedding models for search queries."""
    
    def __init__(self, cache_dir: Path, models_config: dict):
        self.cache_dir = cache_dir
        self.models_config = models_config
        
    @lru_cache(maxsize=4)
    def get_model(self, model_key: str) -> TextEmbedding:
        """Get or initialize cached model."""
        if model_key not in self.models_config:
            available = ", ".join(self.models_config.keys())
            raise ValueError(
                f"Model '{model_key}' not configured.\n"
                f"Available: {available}"
            )
        
        config = self.models_config[model_key]
        return TextEmbedding(
            model_name=config.name,
            cache_dir=str(self.cache_dir)
        )
    
    def generate_query_embedding(
        self,
        query: str,
        collection_name: str,
        client: QdrantClient,
        vector_name: str = None
    ) -> tuple[str, list[float]]:
        """Generate query embedding with auto-detected or specified model."""
        # Detect model from collection's vector configuration
        collection_info = client.get_collection(collection_name)
        available_vectors = list(collection_info.config.params.vectors.keys())

        if not available_vectors:
            raise ValueError(f"Collection {collection_name} has no vectors")

        # Determine which vector to use
        if vector_name:
            # User specified - validate it exists
            if vector_name not in available_vectors:
                available = ", ".join(available_vectors)
                raise ValueError(
                    f"Vector '{vector_name}' not found in collection.\n"
                    f"Available vectors: {available}"
                )
            model_key = vector_name
        else:
            # Auto-select first vector (alphabetically for consistency)
            model_key = sorted(available_vectors)[0]

        # Get cached model and generate embedding
        model = self.get_model(model_key)
        query_vector = list(model.query_embed([query]))[0]

        return (model_key, query_vector.tolist())
```

#### Component 2: Metadata Filter Parser

**Full implementation from arcaneum-54**:

```python
# src/arcaneum/search/filters.py
from qdrant_client.http import models
import json

def parse_filter(filter_arg: str) -> models.Filter:
    """Parse filter from CLI argument."""
    if not filter_arg:
        return None
    
    # Detect format
    if filter_arg.startswith('{'):
        return parse_json_filter(filter_arg)
    elif ':' in filter_arg:
        return parse_extended_filter(filter_arg)
    else:
        return parse_simple_filter(filter_arg)

def parse_simple_filter(filter_str: str) -> models.Filter:
    """Parse key=value,key=value format."""
    conditions = []
    for pair in filter_str.split(','):
        if '=' not in pair:
            continue
        key, value = pair.split('=', 1)
        conditions.append(
            models.FieldCondition(
                key=key.strip(),
                match=models.MatchValue(value=value.strip())
            )
        )
    return models.Filter(must=conditions) if conditions else None

def parse_json_filter(json_str: str) -> models.Filter:
    """Parse Qdrant JSON filter format."""
    filter_dict = json.loads(json_str)
    # Convert dict to Qdrant Filter object
    return models.Filter(**filter_dict)

def parse_extended_filter(filter_str: str) -> models.Filter:
    """Parse extended DSL: key:op:value."""
    conditions = []
    for term in filter_str.split(','):
        parts = term.split(':', 2)
        if len(parts) != 3:
            continue
        
        key, op, value = [p.strip() for p in parts]
        
        if op == 'in':
            # Multiple values: language:in:python,java
            values = value.split(',')
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=values)
                )
            )
        elif op in ('gte', 'gt', 'lte', 'lt'):
            # Range query: chunk_index:gte:10
            conditions.append(
                models.FieldCondition(
                    key=key,
                    range=models.Range(**{op: float(value)})
                )
            )
        elif op == 'contains':
            # Text search: file_path:contains:/src/
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchText(text=value)
                )
            )
    
    return models.Filter(must=conditions) if conditions else None
```

#### Component 3: Single Collection Search

**Simplified synchronous implementation**:

```python
# src/arcaneum/search/searcher.py
from dataclasses import dataclass
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.http import models

@dataclass
class SearchResult:
    """Search result format."""
    score: float
    collection: str
    location: str
    content: str
    metadata: dict[str, Any]

def search_collection(
    client: QdrantClient,
    embedder: SearchEmbedder,
    query: str,
    collection_name: str,
    vector_name: str = None,
    limit: int = 10,
    query_filter: models.Filter = None,
    score_threshold: float = None
) -> list[SearchResult]:
    """Search single collection with query embedding."""

    # Generate query embedding with auto-detected or specified model
    vector_name, query_vector = embedder.generate_query_embedding(
        query, collection_name, client, vector_name
    )

    # Execute search
    results = client.search(
        collection_name=collection_name,
        query_vector=(vector_name, query_vector),
        query_filter=query_filter,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
        with_vectors=False
    )

    # Convert to SearchResult format
    return [
        SearchResult(
            score=r.score,
            collection=collection_name,
            location=format_location(r.payload),
            content=r.payload.get("content", ""),
            metadata=r.payload
        )
        for r in results
    ]
```

**Key Features**:
- Synchronous (no asyncio complexity)
- Auto-detects embedding model from collection metadata
- Applies metadata filters
- Optional score threshold for quality filtering
- Returns unified SearchResult format

#### Component 4: Result Formatter

**Full implementation from arcaneum-56**:

```python
# src/arcaneum/search/formatter.py

def format_location(metadata: dict) -> str:
    """Format location for Claude Code."""
    file_path = metadata.get("file_path", "")

    # PDF: Include page number (useful for navigation)
    if "page_number" in metadata:
        return f"{file_path}:page{metadata['page_number']}"

    # Source code: Just file path (line numbers deferred to full-text search)
    return file_path or f"[{metadata.get('id', '?')}]"

def format_text_results(
    query: str,
    results: list[SearchResult],
    verbose: bool = False
) -> str:
    """Format results for terminal display."""
    lines = []
    lines.append(f'Searching for: "{query}"')
    lines.append(f"Found {len(results)} results\n")
    
    for i, result in enumerate(results, 1):
        # Header with score and metadata
        score_pct = int(result.score * 100)
        lines.append(f"[{i}] Score: {score_pct}% | {format_metadata(result.metadata)}")
        
        # Location
        lines.append(f"    {result.location}")
        lines.append("")
        
        # Content snippet (first 200 chars)
        snippet = extract_snippet(result.content, max_length=200)
        for line in snippet.split('\n')[:5]:
            lines.append(f"    {line}")
        lines.append("")
    
    return "\n".join(lines)

def format_json_results(
    query: str,
    collections: list[str],
    results: list[SearchResult]
) -> str:
    """Format results as JSON."""
    import json
    return json.dumps({
        "query": query,
        "collections": collections,
        "total_results": len(results),
        "results": [
            {
                "score": r.score,
                "collection": r.collection,
                "location": r.location,
                "content": r.content[:500],  # Truncate for JSON
                "metadata": r.metadata
            }
            for r in results
        ]
    }, indent=2)

def format_metadata(metadata: dict) -> str:
    """Format metadata for compact display."""
    parts = []
    
    if "programming_language" in metadata:
        parts.append(f"Language: {metadata['programming_language']}")
    if "git_project_name" in metadata:
        parts.append(f"Project: {metadata['git_project_name']}")
    if "git_branch" in metadata and metadata["git_branch"] != "main":
        parts.append(f"Branch: {metadata['git_branch']}")
    
    return " | ".join(parts) if parts else f"Collection: {metadata.get('collection', '?')}"

def extract_snippet(content: str, max_length: int = 200) -> str:
    """Extract snippet with word boundary."""
    if len(content) <= max_length:
        return content
    
    snippet = content[:max_length]
    last_space = snippet.rfind(' ')
    if last_space > max_length * 0.8:
        snippet = snippet[:last_space]
    
    return snippet + "..."
```

#### CLI Implementation

**Simplified CLI implementation** (`src/arcaneum/cli/search.py`):

```python
import click
from pathlib import Path
from qdrant_client import QdrantClient
from ..search.embedder import SearchEmbedder
from ..search.searcher import search_collection
from ..search.filters import parse_filter
from ..search.formatter import format_text_results, format_json_results
from ..config import DEFAULT_MODELS

@click.command()
@click.argument('query')
@click.option('--collection', required=True, help='Collection to search')
@click.option('--vector-name', help='Vector name to use (auto-detects if not specified)')
@click.option('--filter', 'filter_arg', help='Metadata filter (key=value or JSON)')
@click.option('--limit', type=int, default=10, help='Number of results')
@click.option('--offset', type=int, default=0, help='Number of results to skip (for pagination)')
@click.option('--score-threshold', type=float, help='Minimum score threshold')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
@click.option('--qdrant-url', default='http://localhost:6333', help='Qdrant server URL')
@click.option('--cache-dir', type=click.Path(), default='~/.cache/arcaneum', help='Model cache dir')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def search_command(
    query: str,
    collection: str,
    vector_name: str,
    filter_arg: str,
    limit: int,
    offset: int,
    score_threshold: float,
    output_json: bool,
    qdrant_url: str,
    cache_dir: str,
    verbose: bool
):
    """Search Qdrant collection semantically."""

    # Initialize
    client = QdrantClient(url=qdrant_url)
    cache_path = Path(cache_dir).expanduser()
    embedder = SearchEmbedder(cache_path, DEFAULT_MODELS)

    try:
        # Parse filter
        query_filter = parse_filter(filter_arg) if filter_arg else None

        # Execute search (synchronous)
        results = search_collection(
            client=client,
            embedder=embedder,
            query=query,
            collection_name=collection,
            vector_name=vector_name,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold
        )

        # Format output
        if output_json:
            output = format_json_results(query, [collection], results)
        else:
            output = format_text_results(query, results, verbose)

        click.echo(output)
        return 0

    except Exception as e:
        click.echo(f"[ERROR] Search failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1
```

**Integration with centralized CLI** (from RDR-001):

```python
# src/arcaneum/cli/main.py
from .search import search_command

@cli.command('search')
@click.pass_context
def search(ctx, **kwargs):
    """Search Qdrant collections (from RDR-007)."""
    from arcaneum.cli.search import search_command
    ctx.forward(search_command)
```

**Slash Command** (`commands/search.md` - follows RDR-006 pattern):

```markdown
---
description: Search Qdrant collection semantically
argument-hint: "<query>" --collection <name> [options]
---

Perform semantic search across Qdrant collections.

**Arguments:**
- "<query>": Search query (required, use quotes)
- --collection <name>: Collection to search (required)
- --vector-name <name>: Vector to use (optional, auto-detects from collection)
- --filter <filter>: Metadata filter (key=value or JSON)
- --limit <n>: Number of results (default: 10)
- --score-threshold <float>: Minimum score (optional)
- --json: Output JSON format

**Examples:**
/arc:find "authentication patterns" --collection MyCode --limit 5
/arc:find "auth" --collection MyCode --filter language=python
/arc:find "error handling" --collection Documentation --score-threshold 0.7
/arc:find "API design" --collection MyCode --vector-name jina

**Execution:**
cd ${CLAUDE_PLUGIN_ROOT}
python -m arcaneum.cli.main search $ARGUMENTS
```

## Alternatives Considered

### Alternative 1: MCP Server Wrapper (Not Chosen)

**Description**: Implement MCP server with structured tools

```python
@mcp.tool()
async def search_semantic(
    query: str,
    collection_name: str,
    limit: int = 10,
    filters: dict = None
) -> dict:
    """MCP tool for semantic search"""
    # Call CLI internally or implement directly
```

**Pros**:
- ✅ Structured tool interface with type hints
- ✅ Tools appear in Claude UI automatically
- ✅ Better parameter validation

**Cons**:
- ❌ Additional complexity (server lifecycle)
- ❌ User explicitly prefers CLI-first
- ❌ Startup overhead for each search
- ❌ More code to maintain

**Reason for rejection**: Per user preference and RDR-006 pattern, use CLI-first with optional MCP wrapper later

### Alternative 2: Embedding in Search Results

**Description**: Store and return embeddings with each result

**Pros**:
- ✅ Enables similarity comparisons
- ✅ Supports re-ranking

**Cons**:
- ❌ Increases response size significantly (768-1024 floats per result)
- ❌ Not needed for typical search use case
- ❌ Can request with `with_vectors=True` if needed

**Reason for rejection**: YAGNI - add only if users request

### Alternative 3: Built-in Hybrid Search

**Description**: Integrate full-text search (MeiliSearch) immediately

**Pros**:
- ✅ Better for exact phrase matching
- ✅ Complementary to semantic search

**Cons**:
- ❌ Additional infrastructure (MeiliSearch server)
- ❌ Complexity of maintaining two search indexes
- ❌ Not critical for v1

**Reason for rejection**: Defer to future RDR (noted in arcaneum-7 design)

## Trade-offs and Consequences

### Positive Consequences

1. **Simplicity**: Single-collection search only - no asyncio complexity
2. **Auto-Detection**: Models detected from metadata - no user configuration needed
3. **Flexible Filtering**: Simple key=value for 80%, JSON for advanced 20%
4. **Claude Integration**: File paths clickable in Claude Code UI (line numbers deferred to full-text search)
5. **Model Caching**: `@lru_cache` ensures models loaded once per session
6. **Fast Implementation**: 20.5 hours vs 28 hours (27% reduction)
7. **Consistency**: Follows RDR-006 CLI-first pattern
8. **JSON Output**: `--json` flag enables future MCP wrapper without CLI changes
9. **Synchronous**: Easier to debug and maintain than async code
10. **Clear Scope**: Single collection = unambiguous, focused feature

### Negative Consequences

1. **No Multi-Collection Search**: Must search collections separately in v1
   - *Mitigation*: Design ready in arcaneum-55, can add in v2 if needed
   - *User Impact*: Low - most searches target specific collection anyway

2. **No Hybrid Search**: Pure semantic search initially
   - *Mitigation*: Future RDR for MeiliSearch integration

3. **Filter DSL Learning Curve**: Users must learn simple DSL or JSON
   - *Mitigation*: Clear examples in `--help` and slash command docs

4. **Offset Pagination Performance**: Large offsets slow (Qdrant limitation)
   - *Mitigation*: Document in CLI help, recommend smaller pages

### Risks and Mitigations

**Risk**: Collection missing `embedding_model` metadata
**Mitigation**: Clear error message with instructions, validation in RDR-003/004/005

**Risk**: Model not in cache, slow first search
**Mitigation**: Progress indicator for model download, persistent cache

**Risk**: Large result sets consume memory
**Mitigation**: Stream results, limit default to 10

## Implementation Plan

### Prerequisites

- [x] RDR-001: Project structure (completed)
- [x] RDR-002: Qdrant server setup (completed)
- [x] RDR-003: Collection management with metadata (completed)
- [x] RDR-004: PDF indexing (completed)
- [x] RDR-005: Source code indexing (completed)
- [x] RDR-006: Claude Code integration pattern (completed)
- [ ] Collections exist with proper metadata
- [ ] Python >= 3.12
- [ ] qdrant-client[fastembed] installed

### Step-by-Step Implementation

#### Step 1: Create Search Module Structure

Create search package:
```bash
mkdir -p src/arcaneum/search
touch src/arcaneum/search/__init__.py
touch src/arcaneum/search/embedder.py
touch src/arcaneum/search/filters.py
touch src/arcaneum/search/searcher.py
touch src/arcaneum/search/formatter.py
```

**Estimated effort**: 30 minutes

#### Step 2: Implement Query Embedder

Implement `src/arcaneum/search/embedder.py`:
- `SearchEmbedder` class with model caching
- `detect_collection_model()` function
- `generate_query_embedding()` method
- Unit tests for model detection

**Files**:
- `src/arcaneum/search/embedder.py` (from Component 1 design)
- `tests/search/test_embedder.py`

**Estimated effort**: 4 hours

#### Step 3: Implement Filter Parser

Implement `src/arcaneum/search/filters.py`:
- `parse_filter()` dispatcher
- `parse_simple_filter()` for key=value
- `parse_json_filter()` for JSON
- `parse_extended_filter()` for key:op:value
- Unit tests for all formats

**Files**:
- `src/arcaneum/search/filters.py` (from Component 2 design)
- `tests/search/test_filters.py`

**Estimated effort**: 4 hours

#### Step 4: Implement Single Collection Search

Implement `src/arcaneum/search/searcher.py`:
- `search_collection()` function (synchronous)
- `SearchResult` dataclass
- Error handling
- Unit tests with mock collections

**Files**:
- `src/arcaneum/search/searcher.py` (from Component 3 design)
- `tests/search/test_searcher.py`

**Estimated effort**: 3 hours

#### Step 5: Implement Result Formatter

Implement `src/arcaneum/search/formatter.py`:
- `format_location()` for file paths
- `format_text_results()` for terminal
- `format_json_results()` for --json
- `format_metadata()` and `extract_snippet()`
- Tests for formatting edge cases

**Files**:
- `src/arcaneum/search/formatter.py` (from Component 4 design)
- `tests/search/test_formatter.py`

**Estimated effort**: 3 hours

#### Step 6: Implement CLI Command

Implement `src/arcaneum/cli/search.py`:
- `search_command()` Click command (simplified, no asyncio)
- Argument parsing and validation
- Integration with search modules
- Error handling and logging
- Update `src/arcaneum/cli/main.py`

**Files**:
- `src/arcaneum/cli/search.py` (from CLI Implementation design)
- `src/arcaneum/cli/main.py` (add search subcommand)

**Estimated effort**: 2 hours

#### Step 7: Create Slash Command

Create `commands/search.md`:
- Frontmatter with description
- Argument documentation
- Usage examples
- Bash execution block

**Files**:
- `commands/search.md` (from Slash Command design)

**Estimated effort**: 1 hour

#### Step 8: Integration Testing

End-to-end tests:
- Search single collection
- Search with filters (simple and JSON)
- JSON output format
- Error scenarios (missing collection, bad filter)
- Performance benchmarks

**Files**:
- `tests/integration/test_search_workflow.py`

**Estimated effort**: 3 hours

#### Step 9: Documentation

Create documentation:
- Update README with search examples
- Create `docs/search-guide.md`
- Document filter DSL syntax
- Add troubleshooting section

**Files**:
- `docs/search-guide.md`
- README.md updates

**Estimated effort**: 2 hours

### Total Estimated Effort

**20.5 hours** (~2.5 days of focused work)

**Effort Breakdown**:
- Step 1: Module structure (0.5h)
- Step 2: Query embedder (4h)
- Step 3: Filter parser (4h)
- Step 4: Single collection search (3h)
- Step 5: Result formatter (3h)
- Step 6: CLI command (2h)
- Step 7: Slash command (1h)
- Step 8: Integration tests (3h)
- Step 9: Documentation (2h)

### Dependencies

Reuse existing from RDR-003/004/005:
- qdrant-client[fastembed]
- click
- pydantic

## Validation

### Testing Approach

**Unit Tests**:
- Model detection from metadata
- Filter parsing (simple, JSON, extended)
- Score normalization
- Result formatting

**Integration Tests**:
- Search indexed PDF collection (from RDR-004)
- Search indexed source code collection (from RDR-005)
- Filter application (simple and JSON)
- Error handling

**Performance Tests**:
- Search 10K documents < 1s
- Model caching reduces latency by 90%

### Test Scenarios

**Scenario 1: Basic Search**
- **Setup**: PDF collection with 100 documents indexed
- **Action**: `arc find Research "machine learning"`
- **Expected**: Top 10 results, formatted with scores and paths

**Scenario 2: Filtered Search**
- **Setup**: Source code collection with multiple projects
- **Action**: `arc find MyCode "authentication" --filter git_project_name=backend`
- **Expected**: Results only from "backend" project

**Scenario 3: Score Threshold**
- **Setup**: Source code collection
- **Action**: `arc find MyCode "function" --score-threshold 0.8`
- **Expected**: Only high-confidence results (score >= 0.8)

**Scenario 4: JSON Output**
- **Setup**: Any collection
- **Action**: `arc find MyCode "test" --json`
- **Expected**: Valid JSON with schema from Component 4

**Scenario 5: Missing Model**
- **Setup**: Collection without embedding_model metadata
- **Action**: `arc find BadCollection "query"`
- **Expected**: Clear error: "Collection BadCollection missing embedding_model metadata"

**Scenario 6: Complex Filter**
- **Setup**: Source code collection
- **Action**: `arc find MyCode "auth" --filter '{"must": [{"key": "language", "match": {"value": "python"}}], "must_not": [{"key": "file_path", "match": {"text": "test"}}]}'`
- **Expected**: Python files excluding tests

### Performance Validation

**Metrics**:
- Model loading (first time): < 5s
- Model loading (cached): < 0.1s
- Search latency: < 1s for 10K docs
- Memory usage: < 200MB per model cached

## Future Enhancements

### Collection Relevance Discovery (Near-term Enhancement)

**Concept**: Help users/Claude discover which collections are relevant to a query before searching.

**Workflow**:
```bash
# Step 1: Discover relevant collections
arc collection-relevance "authentication patterns"

# Output:
# MyCode: 47 relevant chunks (stella model)
# Documentation: 12 relevant chunks (stella model)
# PDFs: 3 relevant chunks (bge model)
# Archive: 0 relevant chunks
#
# Recommended: Start with MyCode

# Step 2: Search targeted collection
arc find MyCode "authentication patterns"
```

**Implementation Sketch**:
```python
def find_relevant_collections(
    query: str,
    score_threshold: float = 0.7,
    sample_size: int = 10
) -> list[tuple[str, int, str]]:
    """Find collections relevant to query.

    Returns: [(collection_name, relevant_count, model)]
    """
    all_collections = client.get_collections()
    relevance = []

    for coll in all_collections:
        # Quick sample search to test relevance
        results = search_collection(
            query=query,
            collection_name=coll.name,
            limit=sample_size,
            score_threshold=score_threshold
        )

        # Count relevant results
        relevant_count = len(results)
        model = detect_collection_model(client, coll.name)

        relevance.append((coll.name, relevant_count, model))

    # Sort by relevance
    return sorted(relevance, key=lambda x: x[1], reverse=True)
```

**Benefits**:
- Simple iteration (no merging complexity)
- Helps Claude/user choose which collection to search
- Aligned with Claude Code's interactive workflow
- Minimal implementation (~2-3 hours)

**Use Case**: User has 10 collections, doesn't know which contains authentication code → tool shows MyCode has 47 matches → search MyCode

**Note**: This replaces traditional multi-collection search with merging. No score normalization or merge strategies needed.

### Hybrid Search (Future RDR)

Integrate MeiliSearch for full-text search:
- Reciprocal Rank Fusion (RRF) for merging semantic + lexical
- Configurable weights (e.g., 70% semantic, 30% full-text)
- Phrase matching for exact quotes

### Query Expansion

Auto-expand queries with synonyms:
- Use language models for query understanding
- "auth" → "authentication, authorization, login"

### Search History

Track and suggest past queries:
- SQLite database for search history
- `--history` flag to show recent searches

### Relevance Feedback

Learn from user selections:
- Track which results users click
- Adjust scoring based on feedback

### Export Results

Save search results:
- `--output results.json` or `results.csv`
- Batch processing workflows

## References

- [Beads Issue arcaneum-7](../.beads/arcaneum.db) - Original RDR request
- [Beads Issue arcaneum-52](../.beads/arcaneum.db) - Qdrant client research
- [Beads Issue arcaneum-53](../.beads/arcaneum.db) - Embedding strategy design
- [Beads Issue arcaneum-54](../.beads/arcaneum.db) - Filter DSL design
- [Beads Issue arcaneum-55](../.beads/arcaneum.db) - Collection iteration patterns (reference for collection-relevance feature)
- [Beads Issue arcaneum-56](../.beads/arcaneum.db) - Result formatting design
- [RDR-001: Project Structure](RDR-001-project-structure.md) - Foundation
- [RDR-002: Qdrant Server Setup](RDR-002-qdrant-server-setup.md) - Server config
- [RDR-003: Collection Creation](RDR-003-collection-creation.md) - Metadata schema
- [RDR-004: PDF Indexing](RDR-004-pdf-bulk-indexing.md) - PDF collections
- [RDR-005: Source Code Indexing](RDR-005-source-code-indexing.md) - Code collections
- [RDR-006: Claude Code Integration](RDR-006-claude-code-integration.md) - Integration pattern
- [Qdrant Documentation](https://qdrant.tech/documentation/) - API reference
- [FastEmbed Documentation](https://github.com/qdrant/fastembed) - Embedding models

## Notes

**Key Design Decisions**:

1. **CLI-First over MCP**: Direct CLI execution (per RDR-006 pattern and user preference)
   - Simpler, faster, more maintainable
   - MCP wrapper can be added later without breaking CLI

2. **Auto-Detection**: Embedding models detected from collection metadata
   - No user configuration needed
   - Fails fast with clear errors if metadata missing

3. **Two-Tier Filtering**: Simple DSL (80%) + JSON (20%)
   - Simple: `language=python,project=myproj`
   - JSON: Full Qdrant filter power
   - Auto-detect format from input

4. **Claude UI Optimization**: File paths clickable in Claude Code
   - Format: `/path/to/file.py` (no line numbers for v1)
   - Page numbers for PDFs: `/path/to/file.pdf:page12`
   - Line numbers deferred to full-text search feature

**Implementation Priority**:
1. Core search (single collection) - Foundation
2. Filter parsing - User flexibility
3. Result formatting - UX polish
4. Slash command - Claude integration

**Success Criteria**:
- Users can search indexed content within 1 second
- Filter syntax intuitive for 80% of use cases
- Results formatted perfectly for Claude Code
- Follows RDR-006 integration pattern
- Simple, maintainable codebase (no asyncio complexity)

**Development Timeline**:
- Phase 1: Core search + filtering (11.5 hours)
- Phase 2: Formatting + integration (5 hours)
- Phase 3: Testing + docs (5 hours)
- **Total**: 20.5 hours (~2.5 days)

**Migration Path for Future Enhancements**:
1. ✅ Phase 1 (Current): Single-collection search with CLI-first design
2. ⏱️ Phase 2: Add collection-relevance discovery tool
3. ⏱️ Phase 3: Add hybrid search with MeiliSearch
4. ⏱️ Phase 4: Optional FastMCP wrapper for Claude UI tool discovery

**Key Architectural Decision**: Start simple with single-collection synchronous search. Collection discovery helps users find the right collection to search.

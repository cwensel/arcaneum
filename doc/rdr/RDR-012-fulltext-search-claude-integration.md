# Recommendation 012: Claude Code Integration for Full-Text Search

## Metadata

- **Date**: 2025-10-27
- **Status**: Recommendation
- **Type**: Feature
- **Priority**: High
- **Related Issues**: arcaneum-71
- **Related Tests**: Full-text search CLI tests, slash command tests, result
  formatting tests

## Problem Statement

Create a Claude Code integration that exposes full-text search (MeiliSearch)
capabilities through CLI commands and slash commands, complementary to the
semantic search from RDR-007. The system must:

1. **Provide exact phrase matching** - Quote syntax for literal string search
   (`"def authenticate"`)
2. **Support regex and wildcards** - Complex pattern matching for precise
   queries
3. **Return precise locations** - file.py:line for code, file.pdf:page for PDFs
4. **Follow RDR-006 pattern** - CLI-first with slash commands, no MCP
5. **Complement semantic search** - Clear guidance on when to use which search
   type

This addresses the critical requirement that Claude Code users can perform exact
string searches complementary to semantic similarity searches from RDR-007.

**Expected Workflow**:

1. User: "Find authentication patterns" → Claude uses **semantic search**
   (RDR-007)
2. User: "Find exact string 'def authenticate'" → Claude uses **full-text
   search** (this RDR)

## Context

### Background

Arcaneum provides two complementary search systems:

- **Qdrant (RDR-007)**: Semantic similarity search via vector embeddings
- **MeiliSearch (RDR-008/010/011)**: Full-text exact phrase and keyword search

**The Missing Piece**: Claude Code users need discoverable access to full-text
search.

**Key Design Questions** (from arcaneum-71):

- Separate slash command: `/arc:match` vs `/arc:find`?
- How to present choice to user (semantic vs exact)?
- Result format: file:line (exact location needed)
- CLI interface: `arc match MyCode "exact phrase"`
- Integration pattern: Should mirror RDR-007 structure?
- When to use which: Guide for Claude/users?

**Complementary to RDR-007, not replacement or merger.**

### Technical Environment

- **MeiliSearch**: v1.24.0 (Docker from RDR-008)
- **Python**: >= 3.12
- **meilisearch-python**: >= 0.31.0
- **CLI Framework**: Click (from RDR-003)
- **Existing Stack**:
  - RDR-006: Claude Code integration patterns (CLI-first, slash commands)
  - RDR-007: Semantic search CLI and slash commands
  - RDR-008: MeiliSearch server setup
  - RDR-010: PDF full-text indexing
  - RDR-011: Source code full-text indexing

**Key Design Principles**:

- Mirror RDR-007's CLI structure (parallel commands)
- Follow RDR-006's integration pattern (CLI-first, no MCP)
- Leverage RDR-010/011's indexed content
- Clear distinction from semantic search

## Research Findings

### Investigation Process

Research based on prior RDRs:

1. **RDR-006 Review**: Claude Code integration patterns (CLI-first, slash commands, no MCP)
2. **RDR-007 Review**: Semantic search CLI structure and result formatting
3. **RDR-008 Review**: MeiliSearch capabilities (phrase search, filters, highlighting)
4. **RDR-010 Review**: PDF full-text indexing metadata (page numbers)
5. **RDR-011 Review**: Code full-text indexing metadata (line numbers, function names)

### Key Discoveries

#### 1. MeiliSearch Search Capabilities (from RDR-008)

**Phrase Matching Syntax**:

- Exact phrase: `"def calculate_total"` (double quotes)
- Case-insensitive by default
- Handles soft separators: `-`, `_`, `|`
- Tokenization: whitespace, quotes, separators

**Filter Expressions**:

```bash
# Simple equality
language = python

# Multiple conditions
language = python AND git_branch = main

# Numeric ranges
page_number > 5 AND page_number < 20

# Text contains
file_path CONTAINS /src/auth/
```

**Highlighting**:

- MeiliSearch returns `_formatted` field with matched text highlighted
- Useful for showing context around matches

#### 2. Result Format Requirements

**Source Code Results** (from RDR-011):

```text
[1] Score: 95% | Language: python | Project: arcaneum | Branch: main
    /path/to/src/auth/verify.py:42-67 (calculate_total function)

    def calculate_total(items):
        """Calculate the total price of items."""
        return sum(item.price for item in items)
```

**PDF Results** (from RDR-010):

```text
[1] Score: 92% | Page: 7
    /path/to/research-paper.pdf:page7

    ...gradient descent algorithm is a fundamental optimization
    technique used in machine learning...
```

**Key Differences from RDR-007 Semantic Search**:

| Aspect | Semantic (RDR-007) | Full-Text (this RDR) |
|--------|-------------------|---------------------|
| **Location** | file.py (no lines) | file.py:42-67 (precise) |
| **Score** | Similarity 0.0-1.0 | Relevance (internal) |
| **Query** | Natural language | Exact phrases, patterns |
| **Use Case** | "Find auth patterns" | "Find 'def authenticate'" |

#### 3. CLI Command Design (mirrors RDR-007)

**Parallel Command Structure**:

```bash
# Semantic search (RDR-007)
arc find MyCode "authentication patterns"

# Full-text search (this RDR)
arc match MyCode-fulltext '"def authenticate"'
```

**Why `search-text` instead of `fulltext search`**:

- Shorter, more intuitive
- Parallel to `search` (semantic)
- Avoids subcommand nesting

**Command Options**:

```bash
arc match <query>" \
  --index <name> \            # Required: MeiliSearch index name
  --filter <filter> \         # Optional: Metadata filters
  --limit <n> \               # Optional: Max results (default: 10)
  --attributes <attrs> \      # Optional: Fields to highlight
  --json                      # Optional: JSON output format
```

#### 4. Slash Command Integration (from RDR-006)

**Slash Command** (`commands/search-text.md`):

```markdown
---
description: Search MeiliSearch index with exact phrases
argument-hint: "<query>" --index <name> [options]
---

Perform full-text search for exact phrases and keywords.

**Arguments:**
- "<query>": Search query (use quotes for exact phrases)
- --index <name>: MeiliSearch index to search (required)
- --filter <filter>: Metadata filter (e.g., language=python)
- --limit <n>: Number of results (default: 10)
- --json: Output JSON format

**Examples:**
/arc:match '"def authenticate"' --index MyCode-fulltext
/arc:match 'calculate_total' --index MyCode-fulltext --filter 'language=python'
/arc:match '"neural network"' --index PDFs --filter 'page_number > 5'

**Execution:**
cd ${CLAUDE_PLUGIN_ROOT}
python -m arcaneum.cli.main search-text $ARGUMENTS
```

**Integration with RDR-006**:

- Follows CLI-first pattern (no MCP)
- Uses `${CLAUDE_PLUGIN_ROOT}` for portability
- Simple markdown file structure
- Discoverable via `/help`

#### 5. When to Use Semantic vs Full-Text (User Guidance)

**Decision Tree for Claude**:

```text
User Query Analysis:
├─ Contains quotes? ("exact phrase")
│  └─> USE FULL-TEXT (/search-text)
│
├─ Asks for "exact", "literal", "string"?
│  └─> USE FULL-TEXT (/search-text)
│
├─ Needs line numbers or precise location?
│  └─> USE FULL-TEXT (/search-text)
│
├─ Natural language concept ("patterns", "approaches")?
│  └─> USE SEMANTIC (/search)
│
├─ Wants similar code/docs?
│  └─> USE SEMANTIC (/search)
│
└─ Unsure or exploratory?
   └─> USE SEMANTIC first, then FULL-TEXT to verify
```

**Example Queries**:

| Query | Search Type | Reasoning |
|-------|-------------|-----------|
| "Find authentication code" | Semantic | Conceptual |
| "Find 'def authenticate'" | Full-text | Exact string |
| "Find similar functions" | Semantic | Similarity |
| "Find class UserAuth" | Full-text | Specific identifier |
| "Find error handling patterns" | Semantic | Pattern discovery |
| "Find 'raise ValueError'" | Full-text | Exact exception |

#### 6. Filter DSL (reuse from RDR-007)

**Simple DSL** (80% use case):

```bash
arc match MyIndex "query" --filter language=python,git_branch=main
```

**JSON DSL** (20% advanced):

```bash
arc match MyIndex "query" --filter '{
  "must": [
    {"key": "language", "match": {"value": "python"}},
    {"key": "git_branch", "match": {"value": "main"}}
  ]
}'
```

**Reuse RDR-007's filter parser** - Same logic, different backend (MeiliSearch vs Qdrant)

## Proposed Solution

### Approach

**CLI-First Full-Text Search Integration** (mirrors RDR-007 structure):

```bash
# Basic full-text search
arc match MyCode-fulltext '"def authenticate"'

# With filters
arc match MyCode-fulltext 'calculate' --filter language=python

# With limit
arc match PDFs '"neural network"' --limit 20

# JSON output
arc match MyCode-fulltext '"async def"' --json
```

**Architecture** (parallel to RDR-007):

```text
┌─────────────────────────────────────────────────┐
│     Layer 1: CLI Entry Point                    │
│     python -m arcaneum.cli.main search-text     │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│     Layer 2: Search Orchestrator                │
│     - Parse query and filters                   │
│     - Execute MeiliSearch search                │
│     - Format results with locations             │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│     Layer 3: MeiliSearch Client                 │
│     - Execute search with filters               │
│     - Return hits with highlighting             │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
              MeiliSearch Server
```

### Technical Design

#### Component 1: Filter Parser (reuse from RDR-007)

**Reuse RDR-007's filter parsing logic**:

```python
# src/arcaneum/search/filters.py (from RDR-007)
# Already supports both Qdrant and MeiliSearch filter formats

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
```

**No modifications needed** - Works for both Qdrant and MeiliSearch

#### Component 2: MeiliSearch Searcher

```python
# src/arcaneum/search/fulltext_searcher.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..fulltext.client import FullTextClient

@dataclass
class FullTextSearchResult:
    """Full-text search result with precise location."""
    score: float
    index: str
    location: str  # file.py:42-67 or file.pdf:page7
    content: str
    metadata: Dict[str, Any]
    highlights: Optional[Dict[str, str]] = None

def search_fulltext(
    client: FullTextClient,
    query: str,
    index_name: str,
    filter_expr: Optional[str] = None,
    limit: int = 10,
    attributes_to_highlight: List[str] = None
) -> List[FullTextSearchResult]:
    """
    Search MeiliSearch index with exact phrases.

    Args:
        client: FullTextClient instance
        query: Search query (use quotes for exact phrases)
        index_name: MeiliSearch index name
        filter_expr: Filter expression (simple or JSON)
        limit: Maximum results
        attributes_to_highlight: Fields to highlight

    Returns:
        List of FullTextSearchResult objects
    """
    # Default highlighting
    if attributes_to_highlight is None:
        attributes_to_highlight = ['content', 'function_name', 'class_name']

    # Execute MeiliSearch search
    results = client.search(
        index_name=index_name,
        query=query,
        filter=filter_expr,
        limit=limit,
        attributes_to_highlight=attributes_to_highlight
    )

    # Convert to FullTextSearchResult format
    search_results = []
    for hit in results['hits']:
        location = format_location(hit)

        result = FullTextSearchResult(
            score=hit.get('_rankingScore', 1.0),  # MeiliSearch internal score
            index=index_name,
            location=location,
            content=hit.get('content', ''),
            metadata=hit,
            highlights=hit.get('_formatted', {})
        )
        search_results.append(result)

    return search_results

def format_location(hit: Dict[str, Any]) -> str:
    """
    Format location for display.

    Returns:
        - Code: file.py:42-67 (function_name)
        - PDF: file.pdf:page7
    """
    file_path = hit.get('file_path', '')

    # Source code: Include line range and function/class name
    if 'start_line' in hit:
        start = hit['start_line']
        end = hit.get('end_line', start)
        name_info = ''

        if hit.get('function_name'):
            name_info = f" ({hit['function_name']} function)"
        elif hit.get('class_name'):
            name_info = f" ({hit['class_name']} class)"

        return f"{file_path}:{start}-{end}{name_info}"

    # PDF: Include page number
    if 'page_number' in hit:
        return f"{file_path}:page{hit['page_number']}"

    # Fallback: Just file path
    return file_path
```

#### Component 3: Result Formatter

```python
# src/arcaneum/search/fulltext_formatter.py

def format_text_results(
    query: str,
    results: List[FullTextSearchResult],
    verbose: bool = False
) -> str:
    """Format full-text search results for terminal display."""
    lines = []
    lines.append(f'Full-text search: "{query}"')
    lines.append(f"Found {len(results)} results\n")

    for i, result in enumerate(results, 1):
        # Header with metadata
        metadata_str = format_metadata(result.metadata)
        lines.append(f"[{i}] {metadata_str}")

        # Location (with precise line/page numbers)
        lines.append(f"    {result.location}")
        lines.append("")

        # Content snippet (use highlighted if available)
        if result.highlights and 'content' in result.highlights:
            snippet = result.highlights['content']
        else:
            snippet = result.content

        # Show first 200 chars or 5 lines
        snippet_lines = snippet.split('\n')[:5]
        for line in snippet_lines:
            lines.append(f"    {line[:200]}")
        lines.append("")

    return "\n".join(lines)

def format_json_results(
    query: str,
    index: str,
    results: List[FullTextSearchResult]
) -> str:
    """Format results as JSON."""
    import json
    return json.dumps({
        "query": query,
        "index": index,
        "search_type": "fulltext",
        "total_results": len(results),
        "results": [
            {
                "score": r.score,
                "index": r.index,
                "location": r.location,
                "content": r.content[:500],  # Truncate for JSON
                "metadata": r.metadata
            }
            for r in results
        ]
    }, indent=2)

def format_metadata(metadata: Dict[str, Any]) -> str:
    """Format metadata for compact display."""
    parts = []

    # Language for code
    if 'programming_language' in metadata:
        parts.append(f"Language: {metadata['programming_language']}")

    # Git info for code
    if 'git_project_name' in metadata:
        parts.append(f"Project: {metadata['git_project_name']}")
    if 'git_branch' in metadata and metadata['git_branch'] != 'main':
        parts.append(f"Branch: {metadata['git_branch']}")

    # Page for PDFs
    if 'page_number' in metadata:
        parts.append(f"Page: {metadata['page_number']}")

    return " | ".join(parts) if parts else f"Index: {metadata.get('index', '?')}"
```

#### Component 4: CLI Implementation

```python
# src/arcaneum/cli/search_text.py

import click
from pathlib import Path
from ..fulltext.client import FullTextClient
from ..search.fulltext_searcher import search_fulltext
from ..search.fulltext_formatter import format_text_results, format_json_results
from ..search.filters import parse_filter

@click.command('search-text')
@click.argument('query')
@click.option('--index', required=True, help='MeiliSearch index to search')
@click.option('--filter', 'filter_arg', help='Metadata filter (key=value or JSON)')
@click.option('--limit', type=int, default=10, help='Number of results')
@click.option('--attributes', help='Comma-separated fields to highlight')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
@click.option('--meili-url', default='http://localhost:7700', help='MeiliSearch server URL')
@click.option('--meili-key', envvar='MEILI_MASTER_KEY', help='MeiliSearch master key')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def search_text_command(
    query: str,
    index: str,
    filter_arg: str,
    limit: int,
    attributes: str,
    output_json: bool,
    meili_url: str,
    meili_key: str,
    verbose: bool
):
    """
    Search MeiliSearch index with exact phrases and keywords.

    Examples:

    \b
    # Exact phrase search
    arc match MyCode-fulltext '"def authenticate"'

    \b
    # Keyword search with filters
    arc match MyCode-fulltext 'calculate_total' --filter 'language=python'

    \b
    # PDF search with page filter
    arc match PDFs '"neural network"' --filter 'page_number > 5'
    """

    # Initialize MeiliSearch client
    try:
        client = FullTextClient(meili_url, meili_key)

        # Check if index exists
        if not client.index_exists(index):
            click.echo(f"[ERROR] Index '{index}' not found in MeiliSearch", err=True)
            return 1

    except Exception as e:
        click.echo(f"[ERROR] Failed to connect to MeiliSearch: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

    try:
        # Parse attributes to highlight
        highlight_attrs = None
        if attributes:
            highlight_attrs = [a.strip() for a in attributes.split(',')]

        # Execute search
        results = search_fulltext(
            client=client,
            query=query,
            index_name=index,
            filter_expr=filter_arg,
            limit=limit,
            attributes_to_highlight=highlight_attrs
        )

        # Format output
        if output_json:
            output = format_json_results(query, index, results)
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

from .search_text import search_text_command

@cli.command('search-text')
@click.pass_context
def search_text(ctx, **kwargs):
    """Full-text search via MeiliSearch (from RDR-012)."""
    ctx.forward(search_text_command)
```

#### Component 5: Slash Command

**`commands/search-text.md`**:

```markdown
---
description: Search MeiliSearch index with exact phrases
argument-hint: "<query>" --index <name> [options]
---

Perform full-text search for exact phrases and keywords across MeiliSearch indexes.

**Arguments:**

- "<query>": Search query (required, use quotes for exact phrases)
- --index <name>: MeiliSearch index to search (required)
- --filter <filter>: Metadata filter (key=value or JSON)
- --limit <n>: Number of results (default: 10)
- --attributes <attrs>: Comma-separated fields to highlight
- --json: Output JSON format

**Examples:**

```bash
# Exact phrase in source code
/arc:match '"def authenticate"' --index MyCode-fulltext

# Keyword with language filter
/arc:match 'calculate_total' --index MyCode-fulltext --filter 'language=python'

# Search specific branch
/arc:match 'UserAuth' --index MyCode-fulltext --filter 'git_branch=main'

# PDF search with page filter
/arc:match '"neural network"' --index PDFs --filter 'page_number > 5'

# JSON output for processing
/arc:match '"async def"' --index MyCode-fulltext --json
```

**When to Use Full-Text vs Semantic Search:**

- **Use /search-text** (full-text) for:
  - Exact phrase matching ("def authenticate")
  - Specific identifiers (class names, function names)
  - Precise line/page number locations
  - Regex patterns and wildcards

- **Use /search** (semantic) for:
  - Conceptual queries ("authentication patterns")
  - Finding similar code/documents
  - Exploratory discovery
  - Natural language queries

**Execution:**

```bash
cd ${CLAUDE_PLUGIN_ROOT}
python -m arcaneum.cli.main search-text $ARGUMENTS
```

**Note:** Requires MeiliSearch server running and indexes created via
RDR-010/011.

```text
Example output continued below
```

### Implementation Example

**Complete Workflow**:

```bash
# 1. Ensure MeiliSearch running (from RDR-008)
docker compose up -d meilisearch

# 2. Index content (from RDR-010/011)
# Already done: PDFs and source code indexed to MeiliSearch

# 3. Full-text search (this RDR)
arc match MyCode-fulltext '"def authenticate"'

# Output:
# Full-text search: "def authenticate"
# Found 3 results
#
# [1] Language: python | Project: arcaneum | Branch: main
#     /path/to/src/auth/verify.py:42-67 (authenticate function)
#
#     def authenticate(username, password):
#         """Verify user credentials using bcrypt."""
#         user = db.get_user(username)
#         ...

# 4. Cooperative workflow (semantic → exact)
# Step 1: Semantic discovery
arc find MyCode "authentication code"

# Step 2: Exact verification on discovered file
arc match MyCode-fulltext '"def authenticate"' \
  --filter 'file_path=/path/to/src/auth/verify.py'
```

**Via Claude Code**:

```text
User: "Find the exact function definition for authenticate"

Claude: I'll search for the exact string "def authenticate" in the code.

[Uses /arc:match '"def authenticate"' --index MyCode-fulltext]

Claude: I found the authenticate function at src/auth/verify.py:42-67.
The function signature is:

def authenticate(username, password):
    """Verify user credentials using bcrypt."""
    ...
```

## Alternatives Considered

### Alternative 1: Merge with Semantic Search (Single `/search` Command)

**Description**: Use single `/search` command, auto-detect semantic vs full-text

```bash
arc find MyCorpus "query"  # Auto-detects quotes → full-text, else semantic
```

**Pros**:

- ✅ Single command (simpler UX)
- ✅ No user decision needed

**Cons**:

- ❌ **Ambiguous**: When to use which?
- ❌ **Hidden complexity**: Auto-detection logic fragile
- ❌ **Against arcaneum-71**: User explicitly wants separate commands
- ❌ **Poor UX**: Users can't explicitly choose search type

**Reason for rejection**: User explicitly requested separation. Clear distinction is better than magic auto-detection.

### Alternative 2: MCP Server Wrapper

**Description**: Implement MCP server with structured full-text search tool

```python
@mcp.tool()
async def search_fulltext(
    query: str,
    index_name: str,
    filters: dict = None,
    limit: int = 10
) -> dict:
    """MCP tool for full-text search"""
    # Call CLI internally
```

**Pros**:

- ✅ Structured tool interface
- ✅ Type hints for parameters
- ✅ Appears in Claude UI automatically

**Cons**:

- ❌ **Against user preference**: "strong want to not use MCP" (RDR-006)
- ❌ **Complexity**: Server lifecycle management
- ❌ **Startup overhead**: MCP layer adds latency
- ❌ **Inconsistent**: Semantic search (RDR-007) uses CLI-first

**Reason for rejection**: User preference from RDR-006 applies here. CLI-first is consistent with RDR-007.

### Alternative 3: Subcommand under `fulltext`

**Description**: Use `fulltext search` instead of `search-text`

```bash
arc fulltext search "query" --index MyCode-fulltext
```

**Pros**:

- ✅ Namespacing clarity
- ✅ Consistent with `fulltext create-index`, `fulltext list-indexes`

**Cons**:

- ❌ **Longer command**: More typing
- ❌ **Not parallel to RDR-007**: Semantic is `search`, not `semantic search`
- ❌ **Less intuitive**: Users think "search text" not "fulltext subcommand"

**Reason for rejection**: `search-text` is shorter and parallel to `search` (semantic).

### Alternative 4: No Slash Command (CLI Only)

**Description**: Provide only CLI command, no `/search-text` slash command

**Pros**:

- ✅ Simpler (fewer files)
- ✅ Users can still use via Bash

**Cons**:

- ❌ **Poor discoverability**: Not in `/help`
- ❌ **Inconsistent**: Semantic search has slash command (RDR-007)
- ❌ **Against RDR-006 pattern**: Claude Code plugins should have slash commands

**Reason for rejection**: Slash commands are core to Claude Code integration (RDR-006).

## Trade-offs and Consequences

### Positive Consequences

1. **Clear Distinction**: Separate `search-text` vs `search` commands avoid confusion
2. **Precise Locations**: Returns file:line or file:page for exact navigation
3. **CLI-First**: Direct MeiliSearch execution, no MCP overhead (consistent with RDR-006)
4. **Parallel to RDR-007**: Same structure, easy to learn for users familiar with semantic search
5. **Reuses Components**: Filter parser from RDR-007, MeiliSearch client from RDR-008
6. **Complementary Workflow**: Guides users on when to use semantic vs exact search
7. **Highlighting Support**: Shows matched text in context
8. **JSON Output**: Enables programmatic use and future MCP wrapper
9. **Discoverable**: Appears in `/help` via slash command

### Negative Consequences

1. **Two Commands to Learn**: Users must understand semantic vs full-text distinction
   - *Mitigation*: Clear documentation, decision tree in slash command help
   - *Benefit*: Explicit choice is better than magic auto-detection

2. **Requires MeiliSearch**: Depends on RDR-008 setup and RDR-010/011 indexing
   - *Mitigation*: Clear error if MeiliSearch not running or index missing
   - *Benefit*: Reuses existing infrastructure

3. **No Type Hints**: Parameters not validated by Claude (CLI parsing only)
   - *Mitigation*: Clear error messages from CLI parser
   - *Benefit*: Can add MCP wrapper later without breaking CLI

### Risks and Mitigations

**Risk**: Users confused about when to use `search` vs `search-text`

**Mitigation**:

- Decision tree in slash command help
- Examples in documentation
- Error messages suggest alternative if query looks wrong (e.g., quotes in semantic search)

**Risk**: MeiliSearch server not running or index doesn't exist

**Mitigation**:

- Health check before search
- Clear error: "MeiliSearch not accessible at <http://localhost:7700>"
- Suggest: "Run `docker compose up -d meilisearch`"

**Risk**: Query syntax errors (e.g., malformed filter)

**Mitigation**:

- Validate filter syntax before sending to MeiliSearch
- Show examples in error message
- `--help` shows filter syntax

**Risk**: Results overwhelming (too many matches)

**Mitigation**:

- Default limit of 10 results
- Show "X more results available, use --limit to see more"
- Suggest more specific query or filters

## Implementation Plan

### Prerequisites

- [x] RDR-006: Claude Code integration patterns (completed)
- [x] RDR-007: Semantic search CLI (completed, reference for structure)
- [x] RDR-008: MeiliSearch server setup (completed)
- [x] RDR-010: PDF full-text indexing (completed, provides indexed content)
- [x] RDR-011: Source code full-text indexing (completed, provides indexed content)
- [ ] Python >= 3.12
- [ ] meilisearch-python >= 0.31.0 installed

### Step-by-Step Implementation

#### Step 1: Create Full-Text Searcher Module

Create `src/arcaneum/search/fulltext_searcher.py`:

- `FullTextSearchResult` dataclass
- `search_fulltext()` function (executes MeiliSearch search)
- `format_location()` function (file:line or file:page formatting)
- Error handling for missing indexes

**Estimated effort**: 3 hours

#### Step 2: Create Result Formatter Module

Create `src/arcaneum/search/fulltext_formatter.py`:

- `format_text_results()` for terminal display
- `format_json_results()` for JSON output
- `format_metadata()` helper
- Highlighting support (use `_formatted` from MeiliSearch)

**Estimated effort**: 2 hours

#### Step 3: Create CLI Command

Create `src/arcaneum/cli/search_text.py`:

- `search-text` Click command
- Argument parsing and validation
- MeiliSearch client initialization
- Index existence check
- Error handling with verbose mode
- Update `src/arcaneum/cli/main.py` to register command

**Estimated effort**: 3 hours

#### Step 4: Create Slash Command

Create `commands/search-text.md`:

- Frontmatter with description and argument hint
- Full documentation with examples
- Decision tree for semantic vs full-text
- Bash execution block with `${CLAUDE_PLUGIN_ROOT}`

**Estimated effort**: 1 hour

#### Step 5: Integration Testing

Test complete workflow:

- Search indexed code (from RDR-011)
- Search indexed PDFs (from RDR-010)
- Filter applications (language, branch, page number)
- JSON output format
- Error scenarios (missing index, server down)
- Cooperative workflow (semantic → exact)

**Files**:

- `tests/search/test_fulltext_searcher.py`
- `tests/search/test_fulltext_formatter.py`
- `tests/cli/test_search_text.py`
- `tests/integration/test_fulltext_search_workflow.py`

**Estimated effort**: 4 hours

#### Step 6: Documentation

Update documentation:

- README with full-text search examples
- CLI reference for `search-text` command
- Decision guide: when to use semantic vs exact
- Troubleshooting guide (MeiliSearch not running, index missing)
- Update `.claude-plugin/plugin.json` to include slash command

**Estimated effort**: 2 hours

### Total Estimated Effort

**15 hours** (~2 days of focused work)

**Effort Breakdown**:

- Step 1: Full-text searcher (3h)
- Step 2: Result formatter (2h)
- Step 3: CLI command (3h)
- Step 4: Slash command (1h)
- Step 5: Integration tests (4h)
- Step 6: Documentation (2h)

### Files to Create

**New Modules**:

- `src/arcaneum/search/fulltext_searcher.py` - MeiliSearch search execution
- `src/arcaneum/search/fulltext_formatter.py` - Result formatting
- `src/arcaneum/cli/search_text.py` - CLI command implementation
- `commands/search-text.md` - Slash command definition

**Tests**:

- `tests/search/test_fulltext_searcher.py` - Searcher unit tests
- `tests/search/test_fulltext_formatter.py` - Formatter unit tests
- `tests/cli/test_search_text.py` - CLI command tests
- `tests/integration/test_fulltext_search_workflow.py` - End-to-end tests

### Files to Modify

**Existing Modules**:

- `src/arcaneum/cli/main.py` - Register `search-text` command
- `.claude-plugin/plugin.json` - Add `commands/search-text.md`
- `README.md` - Add full-text search examples
- `doc/rdr/README.md` - Add RDR-012 reference

### Dependencies

Already satisfied by RDR-007 and RDR-008:

- meilisearch-python >= 0.31.0
- click >= 8.1.0
- rich >= 13.0.0
- pydantic >= 2.0.0

## Validation

### Testing Approach

1. **Unit Tests**: Test searcher, formatter components independently
2. **CLI Tests**: Test command execution, argument parsing
3. **Integration Tests**: Test complete search workflow with real indexes
4. **Cooperative Tests**: Verify semantic → exact workflow
5. **Error Tests**: Test error handling (server down, missing index)

### Test Scenarios

#### Scenario 1: Exact Phrase Search in Code

- **Setup**: Code indexed via RDR-011 with function `authenticate`
- **Action**: `arc match MyCode-fulltext '"def authenticate"'`
- **Expected**:
  - Returns document with exact match
  - Shows file:line location (e.g., verify.py:42-67)
  - Includes function context
  - Highlighting shows matched phrase

#### Scenario 2: Keyword Search with Filters

- **Setup**: Multi-language code indexed
- **Action**: `arc match MyCode-fulltext 'calculate' --filter 'language=python'`
- **Expected**:
  - Returns only Python files
  - All results contain "calculate"
  - Metadata shows language=python

#### Scenario 3: PDF Search with Page Filter

- **Setup**: PDFs indexed via RDR-010
- **Action**: `arc match PDFs '"neural network"' --filter 'page_number > 5'`
- **Expected**:
  - Returns only pages 6+
  - Shows file:pageN location
  - Content includes matched phrase

#### Scenario 4: Cooperative Workflow (Semantic → Exact)

- **Setup**: Code indexed in both Qdrant (RDR-007) and MeiliSearch (RDR-011)
- **Action**:
  1. `arc find MyCode "authentication"` (semantic)
  2. Note file path from results
  3. `arc match MyCode-fulltext '"def authenticate"' --filter 'file_path=<noted_path>'`
- **Expected**:
  - Semantic search finds relevant files
  - Exact search verifies specific implementation
  - Both use same file metadata

#### Scenario 5: Error Handling (Missing Index)

- **Setup**: MeiliSearch running but index doesn't exist
- **Action**: `arc match NonExistent "query"`
- **Expected**:
  - Clear error: "Index 'NonExistent' not found in MeiliSearch"
  - Suggests: "Create index with: arc index create NonExistent --type code"
  - Exit code 1

#### Scenario 6: JSON Output for Programmatic Use

- **Setup**: Any indexed content
- **Action**: `arc match MyCode-fulltext '"test"' --json`
- **Expected**:
  - Valid JSON output
  - Contains: query, index, search_type, total_results, results array
  - Each result has: score, location, content, metadata

### Performance Validation

**Metrics**:

- Search latency: < 50ms (typical MeiliSearch query)
- Result formatting: < 10ms
- Memory usage: < 100MB
- Index health check: < 100ms

## References

### Related RDRs

- [RDR-006: Claude Code Integration](RDR-006-claude-code-integration.md) -
  **PATTERN** (CLI-first, slash commands, no MCP)
- [RDR-007: Semantic Search](RDR-007-semantic-search.md) - **PARALLEL**
  (structure reference for this RDR)
- [RDR-008: Full-Text Search Server Setup](
  RDR-008-fulltext-search-server-setup.md) - MeiliSearch deployment and client
- [RDR-010: PDF Full-Text Indexing](RDR-010-pdf-fulltext-indexing.md) -
  Provides indexed PDF content
- [RDR-011: Source Code Full-Text Indexing](
  RDR-011-source-code-fulltext-indexing.md) - Provides indexed code content

### Beads Issues

- [arcaneum-71](../../.beads/arcaneum.db) - Original RDR request

### Official Documentation

- **MeiliSearch Documentation**: <https://www.meilisearch.com/docs>
- **meilisearch-python Client**: <https://github.com/meilisearch/meilisearch-python>
- **Claude Code Plugin Documentation**:
  <https://docs.claude.com/en/docs/claude-code/plugins>

## Notes

### Key Design Decisions

1. **Separate Command**: `search-text` vs `search` for clear distinction (per arcaneum-71)
2. **CLI-First**: Direct MeiliSearch execution, no MCP (consistent with RDR-006)
3. **Parallel to RDR-007**: Same structure as semantic search for familiarity
4. **Precise Locations**: file:line for code, file:page for PDFs (key differentiator)
5. **Reuse Components**: Filter parser from RDR-007, client from RDR-008
6. **Decision Tree**: Clear guidance on semantic vs full-text use cases

### Implementation Priority

1. Core search functionality (searcher + formatter)
2. CLI command (entry point)
3. Slash command (Claude Code integration)
4. Testing (validation)
5. Documentation (user onboarding)

### Success Criteria

- ✅ Separate `search-text` command (distinct from semantic `search`)
- ✅ Precise locations (file:line for code, file:page for PDFs)
- ✅ CLI-first with slash command (no MCP)
- ✅ Filter support (reuse RDR-007 parser)
- ✅ Highlighting support (show matched context)
- ✅ JSON output mode (future MCP compatibility)
- ✅ Clear error messages (index missing, server down)
- ✅ Decision guidance (when to use semantic vs exact)
- ✅ Implementation < 20 hours
- ✅ Markdownlint compliant

### Future Enhancements

**Hybrid Search** (Future RDR):

- Combine semantic (Qdrant) + exact (MeiliSearch) results
- Reciprocal Rank Fusion (RRF) for result merging
- `arc find MyCode "query" --hybrid`

**MCP Wrapper** (Optional):

- Add FastMCP wrapper if type hints become critical
- Call CLI with `--json` flag internally
- No changes to CLI (maintains backward compatibility)

**Advanced Filters**:

- Regex support in query: `search-text 'function\s+\w+'`
- Fuzzy matching: `search-text 'authenticate' --fuzzy`
- Date range filters for git commits

**Result Caching**:

- Cache frequent queries for faster response
- Invalidate on index updates

This RDR provides the complete specification for exposing full-text search
(MeiliSearch) to Claude Code users through CLI commands and slash commands,
complementary to semantic search (RDR-007), following CLI-first patterns
(RDR-006), and leveraging indexed content from RDR-010 (PDFs) and RDR-011
(source code).

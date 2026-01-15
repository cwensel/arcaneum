# Recommendation 012: Claude Code Integration for Full-Text Search

## Metadata

- **Date**: 2025-10-27
- **Updated**: 2026-01-15
- **Status**: Partially Implemented
- **Type**: Feature
- **Priority**: High
- **Related Issues**: arcaneum-71, arcaneum-3l8l, arcaneum-4t78, arcaneum-72j3,
  arcaneum-jo4x, arcaneum-9vrc, arcaneum-yfge
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

- **MeiliSearch**: v1.12 (Docker from RDR-008)
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

**Filter Expressions** (MeiliSearch native syntax):

```bash
# Simple equality
language = python

# Multiple conditions
language = python AND git_branch = main

# Numeric ranges
page_number > 5 AND page_number < 20

# Array membership
programming_language IN [python, javascript, typescript]

# Text contains
file_path CONTAINS /src/auth/
```

**Note**: Filter syntax uses MeiliSearch native format, which differs from
Qdrant filter syntax in RDR-007.

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
| ------ | ------------------ | -------------------- |
| **Location** | file.py (no lines) | file.py:42-67 (precise) |
| **Score** | Similarity 0.0-1.0 | Relevance (internal) |
| **Query** | Natural language | Exact phrases, patterns |
| **Use Case** | "Find auth patterns" | "Find 'def authenticate'" |

#### 3. CLI Command Design (mirrors RDR-007)

**Unified Search Group Structure**:

```bash
# Semantic search (RDR-007)
arc search semantic "authentication patterns" --collection MyCode

# Full-text search (this RDR)
arc search text "def authenticate" --index MyCode-fulltext
```

**Why unified `search` group with subcommands**:

- Consistent grouping of related search functionality
- Clear distinction via subcommand name (`semantic` vs `text`)
- Parallel option patterns (`--collection` for Qdrant, `--index` for MeiliSearch)

**Command Options**:

```bash
arc search text "<query>" \
  --index <name> \            # Required: MeiliSearch index name
  --filter <filter> \         # Optional: MeiliSearch filter expression
  --limit <n> \               # Optional: Max results (default: 10)
  --offset <n> \              # Optional: Pagination offset (default: 0)
  --json                      # Optional: JSON output format
  --verbose                   # Optional: Detailed output
```

#### 4. Slash Command Integration (from RDR-006)

**Slash Command** (`commands/search.md`) - unified search command:

```markdown
---
description: Search across collections
argument-hint: semantic "query" --collection NAME
---

Search your indexed content using semantic search (most common) or full-text search.

**Subcommands (required):**
- `semantic`: Vector-based semantic search (Qdrant) - use `--collection`
- `text`: Keyword-based full-text search (MeiliSearch) - use `--index`

**Examples:**
# Semantic search (most common)
/arc:search semantic "identity proofing" --collection Standards

# Full-text keyword search
/arc:search text "def authenticate" --index MyCode-fulltext

**Execution:**
arc search $ARGUMENTS
```

**Integration with RDR-006**:

- Follows CLI-first pattern (no MCP)
- Unified search command with subcommands
- Discoverable via `/help`

#### 5. When to Use Semantic vs Full-Text (User Guidance)

**Decision Tree for Claude**:

```text
User Query Analysis:
├─ Contains quotes? ("exact phrase")
│  └─> USE FULL-TEXT (arc search text)
│
├─ Asks for "exact", "literal", "string"?
│  └─> USE FULL-TEXT (arc search text)
│
├─ Needs line numbers or precise location?
│  └─> USE FULL-TEXT (arc search text)
│
├─ Natural language concept ("patterns", "approaches")?
│  └─> USE SEMANTIC (arc search semantic)
│
├─ Wants similar code/docs?
│  └─> USE SEMANTIC (arc search semantic)
│
└─ Unsure or exploratory?
   └─> USE SEMANTIC first, then FULL-TEXT to verify
```

**Example Queries**:

| Query | Search Type | Reasoning |
| ----- | ----------- | --------- |
| "Find authentication code" | Semantic | Conceptual |
| "Find 'def authenticate'" | Full-text | Exact string |
| "Find similar functions" | Semantic | Similarity |
| "Find class UserAuth" | Full-text | Specific identifier |
| "Find error handling patterns" | Semantic | Pattern discovery |
| "Find 'raise ValueError'" | Full-text | Exact exception |

#### 6. Filter Syntax (MeiliSearch Native)

**Note**: MeiliSearch uses its own SQL-like filter syntax. This is **NOT Lucene
query syntax**. Key differences:

| Aspect | MeiliSearch | Lucene |
| ------ | ----------- | ------ |
| Format | `field = value` | `field:value` |
| Purpose | Post-search filtering | Combined query |
| Default | Explicit operators required | OR between terms |
| Wildcards | `STARTS WITH`, `CONTAINS` | `te?t`, `test*` |

Filters are passed directly to MeiliSearch without transformation. The full
MeiliSearch filter syntax is supported.

**Comparison Operators**:

| Operator | Example |
| -------- | ------- |
| `=` | `language = python` |
| `!=` | `language != java` |
| `>` | `page_number > 5` |
| `>=` | `start_line >= 100` |
| `<` | `page_number < 20` |
| `<=` | `end_line <= 500` |
| `TO` | `page_number 5 TO 20` (inclusive range) |

**Logical Operators** (precedence: NOT > AND > OR):

| Operator | Example |
| -------- | ------- |
| `AND` | `language = python AND git_branch = main` |
| `OR` | `language = python OR language = javascript` |
| `NOT` | `NOT language = java` |

Use parentheses for explicit grouping:

```bash
--filter "(language = python OR language = javascript) AND git_branch = main"
```

**Collection Operators**:

| Operator | Example |
| -------- | ------- |
| `IN` | `language IN [python, javascript, typescript]` |
| `NOT IN` | `language NOT IN [java, c++]` |

**Existence Operators**:

| Operator | Example |
| -------- | ------- |
| `EXISTS` | `function_name EXISTS` |
| `NOT EXISTS` | `NOT class_name EXISTS` |
| `IS EMPTY` | `tags IS EMPTY` |
| `IS NOT EMPTY` | `tags IS NOT EMPTY` |
| `IS NULL` | `optional_field IS NULL` |
| `IS NOT NULL` | `optional_field IS NOT NULL` |

**String Pattern Operators** (experimental):

| Operator | Example |
| -------- | ------- |
| `CONTAINS` | `file_path CONTAINS "/src/auth/"` |
| `NOT CONTAINS` | `file_path NOT CONTAINS "/test/"` |
| `STARTS WITH` | `filename STARTS WITH "test_"` |
| `NOT STARTS WITH` | `filename NOT STARTS WITH "_"` |

**CLI Examples**:

```bash
# Simple equality
arc search text "query" --index MyIndex --filter "language = python"

# Multiple conditions with AND
arc search text "query" --index MyIndex --filter "language = python AND git_branch = main"

# Array membership
arc search text "query" --index MyIndex --filter "programming_language IN [python, javascript]"

# Numeric range
arc search text "query" --index MyIndex --filter "page_number 5 TO 20"

# Complex filter with grouping
arc search text "query" --index MyIndex \
  --filter "(language = python OR language = javascript) AND NOT file_path CONTAINS /test/"

# Check field existence
arc search text "query" --index MyIndex --filter "function_name EXISTS"
```

**MeiliSearch filter syntax reference**:
<https://www.meilisearch.com/docs/learn/filtering_and_sorting/filter_expression_reference>

## Proposed Solution

### Approach

**CLI-First Full-Text Search Integration** (mirrors RDR-007 structure):

```bash
# Basic full-text search
arc search text "def authenticate" --index MyCode-fulltext

# With filters
arc search text "calculate" --index MyCode-fulltext --filter "language = python"

# With limit and offset
arc search text "neural network" --index PDFs --limit 20 --offset 10

# JSON output
arc search text "async def" --index MyCode-fulltext --json

# Verbose output
arc search text "authenticate" --index MyCode-fulltext --verbose
```

**Architecture** (parallel to RDR-007):

```text
┌─────────────────────────────────────────────────┐
│     Layer 1: CLI Entry Point                    │
│     arc search text <query> --index <name>      │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│     Layer 2: fulltext.py                        │
│     - search_text_command()                     │
│     - Parse query and filters                   │
│     - Format results with Rich console          │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│     Layer 3: FullTextClient                     │
│     - Execute search with filters               │
│     - Return hits with highlighting             │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
              MeiliSearch Server
```

### Technical Design

#### Component 1: CLI Search Command (Implemented)

**Location**: `src/arcaneum/cli/fulltext.py:33-187`

The search command is implemented inline in fulltext.py rather than as separate
modules. This simpler approach works well for the current scope.

```python
# src/arcaneum/cli/fulltext.py

def search_text_command(
    query: str,
    index_name: str,
    filter_arg: Optional[str],
    limit: int,
    offset: int,
    output_json: bool,
    verbose: bool
):
    """
    Implementation for 'arc search text' command.
    Called from main.py search group.
    """
    client = get_client()

    # Verify server and index exist
    if not client.health_check():
        raise ResourceNotFoundError("MeiliSearch server not available")
    if not client.index_exists(index_name):
        raise ResourceNotFoundError(f"Index '{index_name}' not found")

    # Execute search with interaction logging (RDR-018)
    interaction_logger.start("search", "text", index=index_name, query=query)
    results = client.search(
        index_name, query,
        filter=filter_arg,
        limit=limit,
        offset=offset,
        attributes_to_highlight=['content']
    )

    # Format output (JSON or Rich console)
    if output_json:
        # Structured JSON output
        print(json.dumps({...}))
    else:
        # Rich console formatting with highlighting
        for hit in results['hits']:
            # Display location, metadata, highlighted content
```

**CLI Registration** (`src/arcaneum/cli/main.py:528-539`):

```python
@search.command('text')
@click.argument('query')
@click.option('--index', required=True, help='MeiliSearch index to search')
@click.option('--filter', 'filter_arg', help='Metadata filter expression')
@click.option('--limit', type=int, default=10)
@click.option('--offset', type=int, default=0)
@click.option('--json', 'output_json', is_flag=True)
@click.option('--verbose', '-v', is_flag=True)
def search_text(query, index_name, filter_arg, limit, offset, output_json, verbose):
    """Keyword-based full-text search"""
    from arcaneum.cli.fulltext import search_text_command
    search_text_command(query, index_name, filter_arg, limit, offset, output_json, verbose)
```

#### Component 2: Location Formatting (Gap - Needs Enhancement)

**Current Implementation** (`fulltext.py:146-148`):

```python
location = hit.get('filename', hit.get('file_path', 'Unknown'))
if 'line_number' in hit:
    location += f":{hit['line_number']}"
```

**Required Enhancement** (see arcaneum-jo4x):

The current implementation only shows a single line number. RDR-011 provides
richer metadata that should be displayed:

```python
def format_location(hit: Dict[str, Any]) -> str:
    """Format location with full context from RDR-011 metadata."""
    file_path = hit.get('file_path') or hit.get('filename', 'Unknown')

    # Source code: Include line range and function/class name
    if 'start_line' in hit:
        start = hit['start_line']
        end = hit.get('end_line', start)
        location = f"{file_path}:{start}-{end}"

        if hit.get('function_name'):
            location += f" ({hit['function_name']} function)"
        elif hit.get('class_name'):
            location += f" ({hit['class_name']} class)"
        return location

    # PDF: Include page number
    if 'page_number' in hit:
        return f"{file_path}:page{hit['page_number']}"

    return file_path
```

#### Component 3: Slash Command (Implemented)

**Location**: `commands/search.md`

Unified search command supporting both semantic and text subcommands:

```markdown
---
description: Search across collections
argument-hint: semantic "query" --collection NAME
---

**Subcommands:**
- `semantic`: Vector-based semantic search (Qdrant)
- `text`: Keyword-based full-text search (MeiliSearch)

**Examples:**
/arc:search semantic "identity proofing" --collection Standards
/arc:search text "def authenticate" --index MyCode-fulltext

**Execution:**
arc search $ARGUMENTS
```

### Implementation Example

**Complete Workflow**:

```bash
# 1. Ensure MeiliSearch running (from RDR-008)
docker compose -f deploy/docker-compose.yml up -d meilisearch

# 2. Index content (from RDR-010/011)
arc index text code ./my-project --index MyCode-fulltext

# 3. Full-text search (this RDR)
arc search text "def authenticate" --index MyCode-fulltext

# Output:
# Search Results (12ms)
# Found 3 matches in 'MyCode-fulltext'
#
# 1. /path/to/src/auth/verify.py:42
#    def authenticate(username, password):
#        """Verify user credentials...

# 4. Cooperative workflow (semantic → exact)
# Step 1: Semantic discovery
arc search semantic "authentication code" --collection MyCode

# Step 2: Exact verification on discovered file
arc search text "def authenticate" --index MyCode-fulltext \
  --filter "file_path CONTAINS /auth/"
```

**Via Claude Code**:

```text
User: "Find the exact function definition for authenticate"

Claude: I'll search for the exact string "def authenticate" in the code.

[Uses /arc:search text "def authenticate" --index MyCode-fulltext]

Claude: I found the authenticate function at src/auth/verify.py:42.
The function signature is:

def authenticate(username, password):
    """Verify user credentials using bcrypt."""
    ...
```

## Alternatives Considered

### Alternative 1: Separate Top-Level Commands (Original Design)

**Description**: Use separate `arc match` and `arc find` top-level commands

```bash
arc find MyCode "authentication patterns"   # Semantic
arc match MyCode-fulltext "def authenticate" # Full-text
```

**Outcome**: Rejected in favor of unified `arc search` group with subcommands.
The implemented approach (`arc search semantic` / `arc search text`) provides
better organization and clearer parallel structure.

### Alternative 2: MCP Server Wrapper

**Description**: Implement MCP server with structured full-text search tool

**Outcome**: Rejected per RDR-006. CLI-first approach maintained for consistency.

### Alternative 3: Subcommand under `fulltext`

**Description**: Use `fulltext search` instead of unified search group

```bash
arc fulltext search "query" --index MyCode-fulltext
```

**Outcome**: Rejected. The `arc indexes` group handles index management, while
`arc search text` handles searching. This separation follows the Qdrant pattern
(`arc collection` for management, `arc search semantic` for searching).

### Alternative 4: Separate Modules (Original Design)

**Description**: Create separate `fulltext_searcher.py` and `fulltext_formatter.py`

**Outcome**: Implemented inline in `fulltext.py` instead. The simpler approach
works well for current scope. Can be refactored if complexity grows.

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
- [x] Python >= 3.12
- [x] meilisearch >= 0.31.0 installed

### Completed Implementation

#### CLI Search Command (Done)

- [x] `arc search text` command - `src/arcaneum/cli/fulltext.py:33-187`
- [x] Filter support (MeiliSearch native syntax)
- [x] JSON and Rich console output modes
- [x] Highlighting via `_formatted` field
- [x] Pagination (limit/offset)
- [x] Interaction logging (RDR-018)
- [x] Health checks and error handling

#### CLI Registration (Done)

- [x] Unified search group in `main.py:502-539`
- [x] `arc search semantic` for Qdrant
- [x] `arc search text` for MeiliSearch

#### Slash Command (Done)

- [x] `commands/search.md` - unified search command
- [x] Documents both semantic and text subcommands
- [x] Usage examples and guidance

#### Index Management Commands (Done)

- [x] `arc indexes list` - list indexes
- [x] `arc indexes create` - create with type settings
- [x] `arc indexes info` - index details
- [x] `arc indexes delete` - delete index
- [x] `arc indexes verify` - health verification
- [x] `arc indexes items` - list indexed documents
- [x] `arc indexes export/import` - backup/restore
- [x] `arc indexes list-projects` - git project listing
- [x] `arc indexes delete-project` - project deletion

### Remaining Work

#### Step 1: Location Formatting Enhancement (arcaneum-jo4x)

Update `fulltext.py:144-176` to use RDR-011 metadata fully:

- [ ] Show `start_line-end_line` range instead of single line
- [ ] Include function/class name in location display
- [ ] Format PDF locations as `file.pdf:pageN`

**Estimated effort**: 2 hours

#### Step 2: Integration Tests

Create comprehensive test suite:

- [ ] `tests/cli/test_search_text.py` - CLI command tests
- [ ] `tests/integration/test_fulltext_search_workflow.py` - End-to-end tests

**Estimated effort**: 4 hours

### Files Modified (Completed)

- `src/arcaneum/cli/fulltext.py` - Search command implementation
- `src/arcaneum/cli/main.py` - CLI registration
- `commands/search.md` - Slash command definition

### Files to Modify (Remaining)

- `src/arcaneum/cli/fulltext.py:144-176` - Location formatting enhancement

### Dependencies

Already satisfied:

- meilisearch >= 0.31.0
- click >= 8.3.0
- rich >= 14.2.0

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
- **Action**: `arc search text "def authenticate" --index MyCode-fulltext`
- **Expected**:
  - Returns document with exact match
  - Shows file:line location (e.g., verify.py:42-67)
  - Includes function context
  - Highlighting shows matched phrase

#### Scenario 2: Keyword Search with Filters

- **Setup**: Multi-language code indexed
- **Action**: `arc search text "calculate" --index MyCode-fulltext --filter "language = python"`
- **Expected**:
  - Returns only Python files
  - All results contain "calculate"
  - Metadata shows language=python

#### Scenario 3: PDF Search with Page Filter

- **Setup**: PDFs indexed via RDR-010
- **Action**: `arc search text "neural network" --index PDFs --filter "page_number > 5"`
- **Expected**:
  - Returns only pages 6+
  - Shows file:pageN location
  - Content includes matched phrase

#### Scenario 4: Cooperative Workflow (Semantic → Exact)

- **Setup**: Code indexed in both Qdrant (RDR-007) and MeiliSearch (RDR-011)
- **Action**:
  1. `arc search semantic "authentication" --collection MyCode`
  2. Note file path from results
  3. `arc search text "def authenticate" --index MyCode-fulltext --filter "file_path CONTAINS /auth/"`
- **Expected**:
  - Semantic search finds relevant files
  - Exact search verifies specific implementation
  - Both use same file metadata

#### Scenario 5: Error Handling (Missing Index)

- **Setup**: MeiliSearch running but index doesn't exist
- **Action**: `arc search text "query" --index NonExistent`
- **Expected**:
  - Clear error: "Index 'NonExistent' not found"
  - Exit code 1

#### Scenario 6: JSON Output for Programmatic Use

- **Setup**: Any indexed content
- **Action**: `arc search text "test" --index MyCode-fulltext --json`
- **Expected**:
  - Valid JSON output
  - Contains: query, index, hits, estimatedTotalHits, processingTimeMs
  - Each hit has: file_path, content, `_formatted` (highlights)

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

1. **Unified Search Group**: `arc search text` and `arc search semantic` under
   single `search` command group
2. **CLI-First**: Direct MeiliSearch execution, no MCP (consistent with RDR-006)
3. **Parallel to RDR-007**: Same structure as semantic search for familiarity
4. **Precise Locations**: file:line for code, file:page for PDFs (key differentiator)
5. **MeiliSearch Native Filters**: Pass filters directly to MeiliSearch (not
   using RDR-007 parser)
6. **Decision Tree**: Clear guidance on semantic vs full-text use cases

### Implementation Status

**Completed**:

1. ✅ CLI search command (`arc search text`)
2. ✅ CLI registration in unified search group
3. ✅ Slash command (`/arc:search text`)
4. ✅ Index management commands (`arc indexes`)
5. ✅ JSON and Rich console output modes
6. ✅ Highlighting support
7. ✅ Interaction logging (RDR-018)

**Remaining**:

1. ⬜ Location formatting enhancement (start_line-end_line, function names)
2. ⬜ Integration tests

### Success Criteria

- ✅ Unified search group (`arc search text` / `arc search semantic`)
- ✅ Precise locations (file:line for code, file:page for PDFs) - partial
- ✅ CLI-first with slash command (no MCP)
- ✅ Filter support (MeiliSearch native syntax)
- ✅ Highlighting support (show matched context)
- ✅ JSON output mode
- ✅ Clear error messages (index missing, server down)
- ✅ Decision guidance (when to use semantic vs exact)
- ✅ Implementation < 20 hours
- ✅ Markdownlint compliant

### Future Enhancements

**Hybrid Search** (see arcaneum-yfge):

MeiliSearch now provides native hybrid search (v1.6+) combining keyword (BM25)
and semantic (vector) search:

```python
# MeiliSearch native hybrid search
results = index.search('query', {
    'hybrid': {
        'semanticRatio': 0.5,  # 0=keyword only, 1=semantic only
        'embedder': 'default'
    }
})
```

Two approaches for Arcaneum:

1. **Cross-Backend Hybrid** (current architecture):
   - Keep Qdrant for semantic + MeiliSearch for full-text
   - Implement Reciprocal Rank Fusion (RRF) at application layer
   - `arc search hybrid "query" --collection MyCode --index MyCode-fulltext`

2. **MeiliSearch-Only Hybrid** (alternative):
   - Configure MeiliSearch embedders (OpenAI, HuggingFace, REST, Ollama)
   - Use native `semanticRatio` parameter
   - Simpler operations but less embedding model flexibility

**Federated Search** (MeiliSearch v1.10+):

Search across multiple indexes with merged results:

```python
results = client.multi_search(
    queries=[
        {'indexUid': 'source_code', 'q': 'auth'},
        {'indexUid': 'documentation', 'q': 'auth'}
    ],
    federation={'limit': 50}  # Merges into single result list
)
```

**MCP Wrapper** (Optional):

- Add FastMCP wrapper if type hints become critical
- Call CLI with `--json` flag internally
- No changes to CLI (maintains backward compatibility)

**Advanced Filters**:

- Date range filters for git commits
- Complex boolean expressions

This RDR provides the complete specification for exposing full-text search
(MeiliSearch) to Claude Code users through CLI commands and slash commands,
complementary to semantic search (RDR-007), following CLI-first patterns
(RDR-006), and leveraging indexed content from RDR-010 (PDFs) and RDR-011
(source code).

---
description: Search across collections
argument-hint: semantic "query" --corpus NAME
---

Search your indexed content using semantic search (most common) or full-text search.

**Quick Start - Most Common Usage:**

```bash
arc search semantic "your query here" --corpus CorpusName
```

**IMPORTANT:** The subcommand (`semantic` or `text`) comes BEFORE the query.

**Subcommands (required):**

- `semantic`: Vector-based semantic search (Qdrant)
- `text`: Keyword-based full-text search (MeiliSearch)

**Examples:**

```text
# Semantic search (most common)
/arc:search semantic "identity proofing" --corpus Standards
/arc:search semantic "authentication logic" --corpus MyCode --limit 5

# Multi-corpus search
/arc:search semantic "authentication" --corpus Code --corpus Docs

# Full-text keyword search
/arc:search text "def authenticate" --corpus MyCode
```

**Common Options:**

- --corpus: Corpus/collection to search (can specify multiple times)
- --limit: Number of results to return (default: 10)
- --offset: Number of results to skip for pagination (default: 0)
- --filter: Metadata filter (key=value or JSON)
- --json: Output in JSON format
- --verbose: Show detailed information

**Semantic Search Options:**

- --vector-name: Vector name (auto-detected if not specified)
- --score-threshold: Minimum similarity score

**Execution:**

```bash
arc search $ARGUMENTS
```

**When to Use Each:**

**Semantic Search** (vector-based):

- Finding conceptually similar code/documents
- Cross-language semantic matching
- "What does this" or "How to" questions
- Fuzzy concept matching

**Full-Text Search** (keyword-based):

- Exact keyword or phrase matching
- Function/variable name search
- Quoted phrase search
- Boolean operators (AND, OR, NOT)

**Result Format:**

Both commands show:

- Relevance score (similarity for semantic, rank for text)
- Source file path
- Matching content snippet
- Metadata (git info for code, page numbers for PDFs)

**Related Commands:**

- /arc:corpus create - Create corpus for dual indexing (recommended)
- /arc:corpus sync - Index content to both systems
- /arc:corpus list - List available corpora
- /arc:collection list - See available collections (semantic only)

**Implementation:**

- RDR-007: Semantic search via Qdrant
- RDR-012: Full-text search via MeiliSearch
- RDR-006: Claude Code integration

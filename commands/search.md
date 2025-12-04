---
description: Search across collections
argument-hint: semantic "query" --collection NAME
---

Search your indexed content using semantic search (most common) or full-text search.

**Quick Start - Most Common Usage:**

```bash
arc search semantic "your query here" --collection CollectionName
```

**IMPORTANT:** The subcommand (`semantic` or `text`) comes BEFORE the query.

**Subcommands (required):**

- `semantic`: Vector-based semantic search (Qdrant) - use `--collection`
- `text`: Keyword-based full-text search (MeiliSearch) - use `--index`

**Examples:**

```text
# Semantic search (most common)
/arc:search semantic "identity proofing" --collection Standards
/arc:search semantic "authentication logic" --collection MyCode --limit 5

# Full-text keyword search
/arc:search text "def authenticate" --index MyCode-fulltext
```

**Common Options:**

- --collection: Collection to search (required for semantic)
- --limit: Number of results to return (default: 10)
- --offset: Number of results to skip for pagination (default: 0)
- --filter: Metadata filter (key=value or JSON)
- --json: Output in JSON format
- --verbose: Show detailed information

**Semantic Search Options:**

- --vector-name: Vector name (auto-detected if not specified)
- --score-threshold: Minimum similarity score

**Full-Text Search Options:**

- --index: MeiliSearch index name (required)

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

- /collection list - See available collections
- /index pdf - Index PDFs for searching
- /index code - Index code for searching

**Implementation:**

- RDR-007: Semantic search via Qdrant
- RDR-012: Full-text search via MeiliSearch
- RDR-006: Claude Code integration

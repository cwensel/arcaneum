---
description: Search across collections
argument-hint: <semantic|text> <query> [options]
---

Search your indexed content using vector-based semantic search or keyword-based full-text search.

**Subcommands:**

- semantic: Vector-based semantic search (Qdrant)
- text: Keyword-based full-text search (MeiliSearch)

**Common Options:**

- --limit: Number of results to return (default: 10)
- --filter: Metadata filter (key=value or JSON)
- --json: Output in JSON format
- --verbose: Show detailed information

**Semantic Search Options:**

- --collection: Collection to search (required)
- --vector-name: Vector name (auto-detected if not specified)
- --score-threshold: Minimum similarity score

**Full-Text Search Options:**

- --index: MeiliSearch index name (required)

**Examples:**

```text
/search semantic "authentication logic" --collection MyCode --limit 5
/search text "def authenticate" --index MyCode-fulltext
/search semantic "fraud detection patterns" --collection PDFs --score-threshold 0.7
```

**Execution:**

```bash
cd ${CLAUDE_PLUGIN_ROOT}
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
- /index pdfs - Index PDFs for searching
- /index source - Index code for searching

**Implementation:**

- RDR-007: Semantic search via Qdrant
- RDR-012: Full-text search via MeiliSearch
- RDR-006: Claude Code integration

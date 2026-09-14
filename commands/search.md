---
description: Search indexed corpora with semantic or ranked full-text retrieval
argument-hint: <semantic|text> "query" --corpus NAME
---

Search indexed content using semantic or full-text search. Both modes are first-class:
choose the one that matches the query, and use both when unsure.

**Quick Start:**

```bash
# Conceptual or paraphrased query
arc search semantic "your query here" --corpus CorpusName

# Exact phrase (the inner quotes are part of the query)
arc search text '"def authenticate"' --corpus CorpusName
```

**IMPORTANT:** The subcommand (`semantic` or `text`) comes BEFORE the query.

**Subcommands (required):**

- `semantic`: Vector-based semantic search (Qdrant)
- `text`: Keyword-based full-text search (MeiliSearch)

**Examples:**

```text
# Conceptual search
/arc:search semantic "identity proofing" --corpus Standards

# Exact identifier
/arc:search text "authenticate_user" --corpus MyCode

# Exact phrase
/arc:search text '"def authenticate"' --corpus MyCode

# Unsure: run both and combine the useful results
/arc:search semantic "retry backoff" --corpus Docs --limit 5
/arc:search text "retry backoff" --corpus Docs --limit 5
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

- Keyword and identifier lookup
- Function/variable name search
- Quoted phrase search
- Typo-tolerant keyword retrieval

**Why use full-text corpus search instead of grep?**

- It searches focused corpora, reducing unrelated matches.
- Sources may be PDFs, external repositories, or files absent from the local checkout.
- Results are bounded, ranked, and structure-aware, which is usually more token-efficient
  than consuming exhaustive matching lines.
- Use `rg` or grep instead for regular expressions, exhaustive occurrence enumeration, or
  the authoritative state of local files that may not have been indexed yet.

**Result Format:**

Both commands show:

- Ranked result order
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

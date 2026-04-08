---
name: arc-search
description: Search indexed corpora using semantic (vector) OR full-text (keyword) search via the arc CLI. Use when the user asks to search, find, look up, or query a corpus, collection, knowledge base, codebase, docs, PDFs, or markdown. Covers both conceptual queries and exact-term lookups.
allowed-tools: Bash(arc:*), Read
---

# arc search

A corpus is dual-indexed: semantic search hits Qdrant, full-text search hits MeiliSearch.
**Both are first-class — pick the one that matches the query, and use both when unsure.**

## Choose the right mode

Use **`semantic`** when the query is:

- Conceptual or paraphrased ("how does auth work")
- A natural-language question
- About *meaning* or *intent*
- Cross-domain ("rate limiting strategies")

Use **`text`** (full-text) when the query is:

- An exact identifier, symbol, error string, or file name
- A specific function name, class, CLI flag, or env var
- A literal phrase the user expects to appear verbatim
- A known acronym, ticket ID, version string, or quoted text

**When unsure, run BOTH and merge results.** Full-text is cheap and often surfaces
hits semantic misses (rare tokens, code symbols, exact error messages).
Do not default to semantic alone.

## Discover what's available

```bash
arc collection list                    # show every corpus/collection
arc corpus info MyCorpus               # inspect one corpus (both sides)
```

## Semantic search (Qdrant)

```bash
arc search semantic "QUERY" --corpus NAME [OPTIONS]
```

Options:

- `--corpus NAME` — repeat for multi-corpus search (`--corpus A --corpus B`)
- `--limit N` — number of results (default small; raise for broad surveys)
- `--offset N` — pagination
- `--score-threshold 0.0-1.0` — drop low-confidence hits
- `--filter "key=value"` or `--filter '{"key":"value"}'` — metadata filter
- `--vector-name NAME` — pick a specific embedding model (auto-detected otherwise)
- `--json` — structured output for parsing
- `-v` / `--verbose` — show scores and metadata

## Full-text search (MeiliSearch)

```bash
arc search text "QUERY" --corpus NAME [OPTIONS]
```

Options:

- `--corpus NAME` — repeat for multi-corpus search
- `--limit N`, `--offset N` — pagination
- `--filter "key=value"` or `--filter '{"key":"value"}'` — metadata filter
- `--json`, `-v` — same as semantic

Note: full-text has no `--score-threshold` or `--vector-name` (no embeddings involved).

## Common patterns

```bash
# Find a function by name — full-text wins
arc search text "parse_frontmatter" --corpus Code

# Conceptual question — semantic wins
arc search semantic "how is the embedding cache invalidated" --corpus Code

# Unknown — run both
arc search semantic "retry backoff" --corpus Docs --limit 5
arc search text "retry backoff" --corpus Docs --limit 5

# Multi-corpus
arc search semantic "auth flow" --corpus Code --corpus Docs

# Filtered (e.g. only python files)
arc search text "TODO" --corpus Code --filter "language=python" --json
```

## Subcommand placement

The subcommand (`semantic` | `text`) MUST come before the query:

- ✅ `arc search semantic "query" --corpus X`
- ❌ `arc search "query" --corpus X --semantic`
- ❌ `arc search --corpus X "query"`

---
name: arc-search
description: Search indexed corpora using semantic (vector) OR full-text (keyword) search via the arc CLI. Use when the user asks to search, find, look up, or query a corpus, collection, knowledge base, codebase, docs, PDFs, or markdown. Covers both conceptual queries and exact-term lookups, including sources outside the local checkout.
allowed-tools: Bash(arc:*), Read
---

# arc search

A corpus is dual-indexed: semantic search hits Qdrant, full-text search hits MeiliSearch.
**Both are first-class — pick the one that matches the query, and use both when unsure.**

Prefer Arcaneum over local file-search tools when the target is an indexed corpus. Corpus
search works across focused source sets such as external repositories, PDFs, and files that
may not exist in the current checkout. It returns bounded, ranked, structure-aware results,
which are usually more token-efficient than consuming an exhaustive list of matching lines.

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

Do not substitute `rg`, grep, or another local file-search tool merely because the query is
literal. When the source is an Arcaneum corpus, use `arc search text` first.

**When unsure, run BOTH and merge results.** Full-text is cheap and often surfaces
hits semantic misses (rare tokens, code symbols, exact error messages).
Do not default to semantic alone.

## Discover what's available

```bash
arc corpus list                        # show dual-indexed corpora
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
- `--filter FILTER` — metadata filter (see [Filters](#filters))
- `--vector-name NAME` — pick a specific embedding model (auto-detected otherwise)
- `--json` — structured output for parsing (see [Results and cites](#results-and-cites))
- `-v` / `--verbose` — show more detail; with `--json`, add each hit's stored `metadata`

## Full-text search (MeiliSearch)

```bash
arc search text "QUERY" --corpus NAME [OPTIONS]
```

Options:

- `--corpus NAME` — repeat for multi-corpus search
- `--limit N`, `--offset N` — pagination
- `--filter FILTER` — metadata filter (see [Filters](#filters))
- `--json`, `-v` — same as semantic

Note: full-text has no `--score-threshold` or `--vector-name` (no embeddings involved).
Preserve double quotes inside the query argument for an exact phrase:

```bash
arc search text '"def authenticate"' --corpus Code
```

Full-text search does not evaluate regular expressions. Use `rg` when the task requires
regex, exhaustive occurrence enumeration, or the authoritative state of local files that
may not have been indexed yet.

## Filters

`--filter` takes the same syntax in both modes:

- `key=value`, with terms joined by `,` (all must match): `--filter "filename=db(31).pdf,page_number:gte:80"`
- `key:op:value`, where op is `match`, `in`, `gte`, `gt`, `lte`, `lt` or `contains`.
  Separate `in` values with `|`: `--filter "language:in:python|java"`
- JSON `{"field": value}`, where a list means `in`: `--filter '{"filename": "db(31).pdf"}'`

Use the JSON form when a value contains `,`, `"`, `<`, `>`, `!=` or an upper-case `AND`, `OR`, `NOT`,
`TO`, `IN` or `EXISTS`. `contains` works in semantic only.

A field the corpus can't filter on fails with `bad-filter` and lists the fields it can filter on.
For PDF corpora these are `filename`, `file_path` and `page_number`, among others. For markdown corpora
they are `filename`, `file_path`, `section` and `tags`. In code corpora, semantic search names
`language`, `project` and `branch` as `programming_language`, `git_project_name` and `git_branch`.

Advanced: each backend's native syntax also works, but only in its own mode. Text search takes a
MeiliSearch expression (`"page_number > 80 AND filename = \"db(31).pdf\""`). Semantic search takes
Qdrant JSON with a top-level `must`, `should` or `must_not`.

## Results and cites

Every output format prints each hit's cite. Both modes' `--json` share one schema:
`data.hits[]` holds `corpus`, `chunk_id`, `locator`, `cite`, `score`, `snippet` and
`snippet_truncated`. The snippet is 500 characters unless `--max-content-chars N` says
otherwise (`0` is unlimited).

A cite is `<corpus>:<path>` plus `#page=N` for a PDF, `#<heading-slug>` for markdown, or
`#L<start>-<end>` for a code hit that carries a line range (semantic code hits cite the path
alone). The path is relative to the corpus's sync root, or absolute with `~` for the home
directory until a sync records one. `#`, `%` and space in the path appear as `%23`, `%25` and
`%20`. Example: `DevRef:Database-Books/db(31).pdf#page=89`. Quote a cite in backticks when
citing a hit; `chunk_id` names the chunk itself.

Scores compare only within one mode, and text scores only within one corpus: multi-corpus
text results interleave each corpus's ranking rather than sorting by score.

## Common patterns

```bash
# Find a function by name — full-text wins
arc search text "parse_frontmatter" --corpus Code

# Find an exact phrase — preserve the inner double quotes
arc search text '"def authenticate"' --corpus Code

# Conceptual question — semantic wins
arc search semantic "how is the embedding cache invalidated" --corpus Code

# Unknown — run both
arc search semantic "retry backoff" --corpus Docs --limit 5
arc search text "retry backoff" --corpus Docs --limit 5

# Multi-corpus
arc search semantic "auth flow" --corpus Code --corpus Docs

# Filtered (e.g. one PDF, from page 80 on): the same filter in both modes
arc search text "transaction" --corpus DevRef --filter "filename=db(31).pdf,page_number:gte:80" --json
arc search semantic "transaction isolation" --corpus DevRef --filter '{"filename": "db(31).pdf"}' --json
```

## Subcommand placement

The subcommand (`semantic` | `text`) MUST come before the query:

- ✅ `arc search semantic "query" --corpus X`
- ❌ `arc search "query" --corpus X --semantic`
- ❌ `arc search --corpus X "query"`

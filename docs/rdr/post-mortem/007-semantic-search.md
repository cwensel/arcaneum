# Post-Mortem: RDR-007 Semantic Search CLI

## RDR Summary

RDR-007 proposed building a CLI-first semantic search tool for Qdrant
collections, covering query embedding with auto-detected models, a
two-tier metadata filter DSL, human-readable and JSON result formatting,
and Claude Code slash command integration. The scope was explicitly
limited to single-collection, synchronous search with multi-collection
deferred to a future enhancement.

## Implementation Status

Implemented (with significant enhancements beyond original scope).

The core search pipeline shipped and is actively used. The implementation
followed the planned four-component architecture (embedder, filters,
searcher, formatter) but diverged on command naming, added multi-corpus
support (originally deferred), and reused shared embedding infrastructure
instead of building standalone components. Documentation deliverables
(search guide) were not created.

---

## Implementation vs. Plan

### What Was Implemented as Planned

- **Search module structure**: `src/arcaneum/search/` package with
  `__init__.py`, `embedder.py`, `filters.py`, `searcher.py`,
  `formatter.py` -- matches the plan exactly.
- **Model auto-detection from collection vectors**: The
  `detect_collection_model()` logic (inspect named vectors, auto-select
  first alphabetically, validate user-specified vector) was implemented
  as designed in the RDR.
- **Two-tier filter DSL**: `parse_simple_filter()` (key=value),
  `parse_json_filter()` (full Qdrant JSON), and `parse_extended_filter()`
  (key:op:value) all implemented with the planned operator set (match,
  in, gte/gt/lte/lt, contains).
- **SearchResult dataclass**: Core fields (score, collection, location,
  content, metadata) match the plan.
- **Result formatting**: `format_text_results()`, `format_json_results()`,
  `format_metadata()`, `extract_snippet()`, and `format_location()` all
  implement the planned output formats with numbered results, score
  percentages, file paths, and content snippets.
- **Location formatting**: Source code shows file path only; PDFs include
  page numbers (`file.pdf:page12`) as planned.
- **CLI options**: `--vector-name`, `--filter`, `--limit`, `--offset`,
  `--score-threshold`, `--json`, `--verbose` all present.
- **Synchronous execution**: No asyncio complexity, as planned.
- **Slash command**: `commands/search.md` exists with execution block.

### What Diverged from the Plan

- **Command naming and structure**: The RDR planned `arc find <collection>
  "<query>"` as a flat command. The implementation uses `arc search
  semantic "<query>" --corpus <name>` as a subcommand under a `search`
  group (which also contains `text` for full-text search). This changed
  the argument from positional to an option, and introduced a required
  subcommand keyword. The change was driven by the need to colocate
  semantic and full-text search under a unified `search` group once
  MeiliSearch integration (RDR-010/012) was implemented.

- **Collection renamed to corpus**: The RDR used `--collection`
  throughout. The implementation uses `--corpus` as the primary option
  (with `--collection` retained as a hidden deprecated alias). This
  reflects a project-wide nomenclature shift where "corpus" means the
  unified Qdrant + MeiliSearch pair.

- **Embedding backend**: The RDR specified FastEmbed's `TextEmbedding`
  directly with `@lru_cache` for model instances. The implementation
  delegates to `EmbeddingClient` from `arcaneum.embeddings.client`,
  which supports both FastEmbed and SentenceTransformers backends. The
  embedder handles backend dispatch (`query_embed()` for FastEmbed,
  `encode()` for SentenceTransformers). This was necessary because
  several important models (stella, jina-code variants) use
  SentenceTransformers, not FastEmbed.

- **Qdrant search API**: The RDR specified `client.search()` with
  `query_vector=(vector_name, query_vector)` tuple syntax. The
  implementation uses `client.query_points()` with separate `query`
  and `using` parameters, which is the newer API introduced in
  qdrant-client 1.16+.

- **CLI is a plain function, not a Click command**: The RDR showed
  `search_command` as a `@click.command()` decorated function with all
  options. The implementation has `search_command()` as a plain function
  called from `main.py` where the Click decorators live. This separates
  CLI parsing from business logic.

- **Multi-corpus search with merge**: The RDR explicitly deferred
  multi-collection search. The implementation loops over multiple
  `--corpus` values, collects results, sorts by score, and applies
  pagination after merging. Missing corpora are handled gracefully
  (warn and continue if other corpora succeed).

- **Qdrant client creation**: The RDR instantiated `QdrantClient(url=...)`
  inline with a `--qdrant-url` CLI option. The implementation uses a
  shared `create_qdrant_client(for_search=True)` utility that reads
  from config files and environment variables, with separate timeout
  handling for search vs indexing operations.

### What Was Added Beyond the Plan

- **Multi-corpus search**: Full merge-and-sort across multiple corpora
  with graceful partial failure handling (deferred in the RDR but
  implemented as part of the corpus unification effort).
- **Interaction logging** (RDR-018): Every search logs start/finish
  with query, corpora, filters, result count, duration, and execution
  context (terminal vs Claude Code) to `~/.arcaneum/logs/`.
- **Retry with exponential backoff**: `@retry` decorator from tenacity
  on `search_collection()` retries connection/timeout errors up to 3
  times.
- **explain_search()**: Diagnostic function that shows the execution
  plan (model, vector, dimensions, collection stats) without running
  the search.
- **format_summary()**: Verbose-mode summary showing search statistics
  (avg/max/min score, execution time, filter description).
- **build_filter_description()**: Human-readable filter description
  generator for verbose output and logging.
- **JSON sanitization**: `_sanitize_for_json()` strips control
  characters from PDF text that would break JSON parsing.
- **Metadata filtering for JSON output**: Non-verbose JSON output
  filters metadata to essential fields only (`_filter_metadata()`).
- **Rich console output**: Uses Rich library for terminal rendering
  instead of plain `click.echo()`.
- **Custom error classes**: `InvalidArgumentError` and
  `ResourceNotFoundError` with distinct exit codes (2, 3) instead
  of generic exceptions.
- **resolve_corpora()**: Backwards-compatibility layer accepting both
  `--corpus` and deprecated `--collection` options.
- **SearchResult.point_id**: Extra field for debugging.

### What Was Planned but Not Implemented

- **docs/search-guide.md**: The RDR planned a standalone search guide
  document. This was not created. Documentation lives in
  `commands/search.md` and `CLAUDE.md` instead.
- **Per-module unit tests**: The RDR planned `tests/search/test_embedder.py`,
  `tests/search/test_filters.py`, `tests/search/test_searcher.py`,
  `tests/search/test_formatter.py`. None of these exist. Testing is
  done at the CLI integration level in
  `tests/unit/cli/test_search_commands.py` with mocked dependencies.
- **Integration test suite**: `tests/integration/test_search_workflow.py`
  was planned but not created.
- **Performance benchmarks**: The RDR specified "Search 10K documents
  < 1s" and "Model caching reduces latency by 90%". No formal
  benchmarks were created.

---

## Drift Classification

| Category | Count | Examples |
| -------- | ----- | -------- |
| **Unvalidated assumption** | 1 | FastEmbed-only backend assumption; several key models (stella, jina-code) use SentenceTransformers |
| **Framework API detail** | 1 | `client.search()` with tuple syntax replaced by `client.query_points()` with `using` parameter |
| **Missing failure mode** | 2 | PDF text with control characters breaking JSON; connection/timeout failures needing retry |
| **Missing Day 2 operation** | 2 | Per-module unit tests not created; search guide documentation not written |
| **Deferred critical constraint** | 1 | Multi-corpus search was deferred but turned out to be needed for corpus unification |
| **Over-specified code** | 1 | Standalone SearchEmbedder with @lru_cache and models_config rewritten to delegate to shared EmbeddingClient |
| **Under-specified architecture** | 1 | CLI command taxonomy (flat command vs search group with subcommands) not locked before implementation |
| **Scope underestimation** | 1 | Interaction logging, error class integration, and corpus naming emerged as cross-cutting work |
| **Internal contradiction** | 0 | |
| **Missing cross-cutting concern** | 2 | Interaction logging (RDR-018) needed; corpus nomenclature shift affected all option names |

---

## RDR Quality Assessment

### What the RDR Got Right

- **Four-component architecture**: The decomposition into embedder,
  filters, searcher, and formatter mapped directly to the final module
  structure. Every planned file was created.
- **Named-vector model detection**: The insight that vector names
  encode model identity (no separate metadata needed) was correct and
  implemented verbatim.
- **Two-tier filter DSL**: The simple/extended/JSON filter approach
  survived implementation unchanged. The operator set was accurate.
- **Result format design**: The human-readable output format (numbered
  results, score percentages, indented snippets) and JSON structure
  were implemented closely to the plan.
- **Scope decision to defer multi-collection**: While multi-corpus was
  eventually added, the initial single-collection implementation was
  the right first step and the deferred design from arcaneum-55 was
  available when needed.
- **Trade-off analysis**: The rejection of MCP server wrapper, embedded
  vectors, and built-in hybrid search were all validated by
  implementation experience.

### What the RDR Missed

- **Embedding backend diversity**: The RDR assumed all models use
  FastEmbed with `query_embed()`. In practice, the project's most
  important models (stella for PDFs, jina-code variants for code) use
  SentenceTransformers with `encode()`. This required backend dispatch
  logic not anticipated in the plan.
- **CLI taxonomy evolution**: The RDR designed `arc find` as a
  standalone command. The project evolved to `arc search semantic` /
  `arc search text` under a unified group, reflecting the addition of
  MeiliSearch full-text search. The RDR should have anticipated that
  the search command namespace would need to accommodate multiple search
  types.
- **PDF content sanitization**: The RDR's JSON output design did not
  account for control characters in OCR-extracted PDF text that break
  JSON serialization.
- **Connection resilience**: No consideration of Qdrant server
  connection failures, timeouts, or retry strategies for the search
  path.
- **Test strategy**: The per-module unit test plan was abandoned in
  favor of CLI-level integration tests with mocks. The RDR did not
  consider whether isolated unit tests or integration tests would
  provide more value for a relatively thin orchestration layer.

### What the RDR Over-specified

- **Complete Python code for SearchEmbedder**: The full class
  implementation with `@lru_cache` and `models_config` parameter was
  rewritten to delegate to the existing `EmbeddingClient`. The detailed
  code was wasted effort since the real implementation reused shared
  infrastructure.
- **Complete Python code for CLI command**: The full Click-decorated
  `search_command()` was substantially rewritten. The inline
  `QdrantClient()` creation, `--qdrant-url` option, and `--cache-dir`
  option were all replaced by shared utilities.
- **Offset pagination detail**: The RDR spent prose on offset pagination
  performance warnings. In practice, multi-corpus merge required
  client-side pagination (fetch limit+offset from each corpus, sort,
  slice), making the Qdrant-level offset concern moot.
- **Future enhancements section**: The collection-relevance discovery
  sketch with implementation code (~50 lines) has not been implemented
  and the approach may not be needed given that multi-corpus search was
  added directly.

---

## Key Takeaways for RDR Process Improvement

1. **Verify backend/library assumptions with the existing codebase
   before specifying implementation code**: The RDR assumed FastEmbed
   was the only embedding backend, but the project already used
   SentenceTransformers for key models. A quick check of
   `EMBEDDING_MODELS` in `embeddings/client.py` would have revealed
   the multi-backend requirement and avoided specifying code that had
   to be rewritten.

2. **Audit for existing shared infrastructure before designing
   standalone components**: The RDR designed a self-contained
   `SearchEmbedder` with its own caching and model configuration. The
   project already had `EmbeddingClient` with all of that
   functionality. RDRs should include a "reuse audit" step that lists
   existing modules that overlap with proposed components.

3. **Design CLI command taxonomy at the group level, not the command
   level**: The RDR named the command `arc find` without considering
   how it would coexist with full-text search (planned in the same
   roadmap). Defining the `arc search {semantic,text}` group structure
   upfront would have prevented the naming change and the
   backwards-compatibility layer.

4. **Specify test strategy rather than test file lists**: The RDR
   listed four unit test files that were never created. A strategy-level
   statement ("test at the CLI integration level with mocked Qdrant
   and embedding clients") would have been more accurate and actionable
   than enumerating per-module test files for what turned out to be a
   thin orchestration layer.

5. **Include a "failure modes" section for any component that touches
   external services or untrusted input**: The RDR omitted connection
   retry logic and PDF content sanitization, both of which required
   real implementation effort. A checklist item like "What happens when
   the external service is down?" and "What happens when payload content
   contains unexpected characters?" would have caught these gaps.

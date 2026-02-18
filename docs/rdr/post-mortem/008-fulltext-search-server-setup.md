# Post-Mortem: RDR-008 Full-Text Search Server Setup (MeiliSearch)

## RDR Summary

RDR-008 proposed deploying MeiliSearch alongside Qdrant in a unified Docker Compose
setup to provide complementary full-text search capabilities, including exact phrase
matching, filtered search, and typo-tolerant keyword lookup. The approach recommended
MeiliSearch over Elasticsearch for resource efficiency and simplicity, with a Python
client wrapper, index configuration templates, CLI commands, and a management script.

## Implementation Status

Implemented

The core MeiliSearch deployment, Python client, index configuration, and CLI
integration are all in production. The implementation went substantially beyond
the original plan, expanding into adjacent RDRs (RDR-009 dual indexing, RDR-010
PDF full-text, RDR-011 code full-text) that built on this foundation. The CLI
command structure was rearchitected to use `arc indexes` (mirroring `arc collection`)
instead of the planned `arc fulltext`, and container management was promoted from
shell scripts to a full `arc container` CLI group.

---

## Implementation vs. Plan

### What Was Implemented as Planned

- **Docker Compose structure**: MeiliSearch added to `deploy/docker-compose.yml`
  alongside Qdrant with named volumes, health check, resource limits, and environment
  variables, closely matching the RDR specification
- **Three named volumes**: `meilisearch-arcaneum-data`, `meilisearch-arcaneum-dumps`,
  `meilisearch-arcaneum-snapshots` as specified
- **Port mapping**: Port 7700 for HTTP API as planned
- **Resource limits**: 4GB memory / 4 CPUs limits with 1GB / 2 CPUs reservations
  exactly as specified
- **Health check configuration**: Uses `wget` against `/health` with the same
  intervals and retry parameters from the RDR
- **Memory configuration**: `MEILI_MAX_INDEXING_MEMORY=2.5GiB` set as recommended
  (2/3 of 4GB container limit, addressing MeiliSearch issue #4686)
- **Management script**: `scripts/meilisearch-manage.sh` implements all seven planned
  operations (start, stop, restart, logs, status, create-dump, list-indexes)
- **Python client core**: `FullTextClient` in `src/arcaneum/fulltext/client.py`
  with `create_index`, `get_index`, `list_indexes`, `delete_index`, `add_documents`,
  `search`, and `health_check` methods matching the planned API
- **Index configuration templates**: `SOURCE_CODE_SETTINGS` and `PDF_DOCS_SETTINGS`
  in `src/arcaneum/fulltext/indexes.py` with the planned searchable attributes,
  filterable attributes, typo tolerance, and stop word configurations
- **CLI search command**: `arc search text` integrated into the search group
  alongside `arc search semantic`, exactly matching the planned complementary workflow
- **`.env.example` updated**: Contains `MEILISEARCH_URL` and `MEILISEARCH_API_KEY`
- **`arc doctor` integration**: MeiliSearch connection check added to diagnostics
- **pyproject.toml dependency**: `meilisearch` package added to dependencies

### What Diverged from the Plan

- **MeiliSearch Docker image version**: The RDR planned `getmeili/meilisearch:v1.32`.
  The implementation uses `v1.12` because the RDR was written speculatively for a
  future version that was not yet released at implementation time.

- **Python client version requirement**: The RDR specified `meilisearch>=0.39.0`.
  The implementation uses `meilisearch>=0.31.0` to match the actual v1.12 server
  compatibility.

- **API key management**: The RDR planned manual API key configuration via `.env`
  file with `MEILISEARCH_API_KEY=your_secure_master_key_here_min_16_chars`. The
  implementation auto-generates a 32-character URL-safe base64 key stored in
  `~/.config/arcaneum/meilisearch.key` because requiring manual key setup created
  unnecessary friction for local development.

- **CLI command group naming**: The RDR planned `arc fulltext` as the admin
  command group. The implementation uses `arc indexes` to mirror the `arc collection`
  naming pattern for Qdrant, providing a more consistent mental model for users
  managing both datastores.

- **CLI subcommand naming**: The RDR planned `create-index`, `list-indexes`,
  `delete-index`. The implementation uses shorter names: `create`, `list`, `delete`
  because the parent group (`arc indexes`) already provides context.

- **Client constructor API key**: The RDR specified `api_key: str` as required.
  The implementation uses `api_key: Optional[str] = None` to support the
  auto-generated key workflow and development mode.

- **Container management approach**: The RDR planned management via shell script
  (`scripts/meilisearch-manage.sh`). The implementation added a full `arc container`
  CLI group (`start`, `stop`, `restart`, `status`, `logs`, `reset`) that manages
  both Qdrant and MeiliSearch containers together, with auto-generated API key
  injection via environment variables.

- **MEILI_ENV configuration**: The RDR hardcoded `MEILI_ENV=production`. The
  implementation uses `MEILI_ENV=${MEILI_ENV:-production}` to allow override for
  development/testing.

- **Search command option naming**: The RDR planned `--index` for specifying the
  target index. The implementation uses `--corpus` (with `--index` retained as a
  hidden deprecated alias) to unify the naming with semantic search's `--corpus`
  option.

### What Was Added Beyond the Plan

- **Multi-corpus search**: `arc search text` supports `--corpus` specified multiple
  times with round-robin result interleaving across indexes, not anticipated by the RDR
- **Additional index types**: `SOURCE_CODE_FULLTEXT_SETTINGS` for function-level
  granularity (RDR-011) and `MARKDOWN_DOCS_SETTINGS` for markdown documents, plus
  type aliases (`code`, `code-fulltext`, `pdf`, `markdown`)
- **Extended client methods**: `index_exists()`, `add_documents_sync()`,
  `add_documents_batch_parallel()`, `delete_documents_by_file_paths()`,
  `get_index_stats()`, `get_index_settings()`, `update_index_settings()`,
  `get_version()`, `get_stats()`, `get_all_file_paths()`,
  `get_chunk_counts_by_file()` -- all needed by downstream indexing RDRs
- **Additional CLI commands**: `info`, `update-settings`, `verify`, `items`,
  `export`, `import`, `list-projects`, `delete-project` (10 total vs 4 planned)
- **Full-text indexing pipeline**: `src/arcaneum/indexing/fulltext/` module with
  `code_indexer.py`, `ast_extractor.py`, `pdf_indexer.py`, `sync.py` for indexing
  content into MeiliSearch (handled via RDR-009, RDR-010, RDR-011)
- **Indexing CLI**: `arc index text code`, `arc index text pdf`,
  `arc index text markdown` for populating MeiliSearch indexes
- **Dual-index corpus management**: `arc corpus` group for unified Qdrant +
  MeiliSearch operations (create, sync, parity, delete)
- **Interaction logging**: All search and index management commands log operations
  via `interaction_logger` (RDR-018)
- **Filter error diagnostics**: When a search fails due to non-filterable attributes,
  the CLI fetches available filterable attributes and suggests corrections
- **Centralized client factory**: `create_meili_client()` in `cli/utils.py`
  for consistent client creation across all commands (parallel to `create_qdrant_client()`)
- **Index export/import**: JSONL-based backup and migration with settings preservation
- **Git-aware change detection**: `GitCodeMetadataSync` for incremental reindexing
  based on commit hashes

### What Was Planned but Not Implemented

- **`tests/cli/test_fulltext.py`**: CLI tests for the fulltext commands were
  planned but not created. Integration tests exist for the client and index
  configurations but not for CLI command execution
- **`tests/integration/test_dual_search.py`**: The dual search integration test
  was planned but not created as a standalone test file. Dual functionality is
  tested indirectly through the corpus sync workflow
- **`docs/fulltext-search.md`**: User documentation was planned but not created.
  Usage is documented only in CLI `--help` text and the RDR itself
- **README update**: The README was planned to be updated with a MeiliSearch
  section but was not

---

## Drift Classification

| Category | Count | Examples |
| -------- | ----- | -------- |
| **Unvalidated assumption** | 1 | MeiliSearch v1.32 and meilisearch-python v0.39.0 specified but not available at implementation time |
| **Framework API detail** | 1 | `list_indexes()` return type needed robust handling for both dict and Index object responses from meilisearch-python |
| **Missing failure mode** | 0 | |
| **Missing Day 2 operation** | 2 | Index export/import for backup/migration; auto-generated API key management for zero-config setup |
| **Deferred critical constraint** | 0 | |
| **Over-specified code** | 3 | Client code substantially expanded (7 to 18+ methods); CLI rearchitected from 4 to 10 commands with different naming; index templates expanded from 2 to 4 types |
| **Under-specified architecture** | 2 | API key lifecycle (generation, storage, rotation) not designed; container management UX not decided (shell script vs CLI) |
| **Scope underestimation** | 2 | CLI grew from 4 admin commands to 10; full-text client became a major module supporting indexing, sync, and change detection beyond just search |
| **Internal contradiction** | 0 | |
| **Missing cross-cutting concern** | 1 | Indexing pipeline (how data gets into MeiliSearch) was out of scope but turned out to be the majority of the implementation work via RDR-009/010/011 |

---

## RDR Quality Assessment

### What the RDR Got Right

- **Technology selection**: MeiliSearch over Elasticsearch was the correct call.
  The resource efficiency claims (96-200MB vs 2-4GB) held true, and the simple
  API accelerated implementation
- **Docker Compose structure**: The unified docker-compose.yml with independent
  services pattern worked exactly as designed. Port assignments, volume naming
  conventions, and health check patterns were all directly usable
- **Complementary architecture**: The core insight that semantic and full-text
  search should be parallel but independent proved correct and became the
  foundation for the entire corpus abstraction (RDR-009)
- **Index schema design**: Searchable/filterable attribute selections for
  source code and PDF documents were accurate. The typo tolerance thresholds
  (7/12 for code, 5/9 for docs) remain unchanged
- **Memory configuration**: The MEILI_MAX_INDEXING_MEMORY guidance (2/3 of container
  limit) was critical and directly incorporated
- **Alternatives analysis**: The Elasticsearch comparison with concrete resource
  numbers was valuable for decision confidence. The TypeSense and Solr evaluations
  were appropriately brief

### What the RDR Missed

- **API key lifecycle**: The RDR treated API key management as a simple environment
  variable, but the actual implementation required auto-generation, secure file
  storage, and environment variable override. This is a recurring pattern: secrets
  management always needs more design than "put it in .env"
- **Indexing workflow**: The RDR explicitly deferred indexing to future RDRs (RDR-009
  through RDR-011), but the client API had to be significantly extended to support
  indexing operations. The RDR should have identified the client extension points
  even if the indexing logic was out of scope
- **CLI naming consistency**: The RDR proposed `arc fulltext` without considering
  that `arc collection` was the Qdrant management group. The parallel naming pattern
  (`arc indexes` / `arc collection`) was obvious in retrospect
- **Container management UX**: The RDR assumed shell scripts would suffice, but
  the need for API key injection and multi-service coordination drove the creation
  of `arc container` commands

### What the RDR Over-specified

- **Complete Python source code**: The RDR included full implementations for
  `client.py`, `indexes.py`, and `fulltext.py` (CLI). All three were substantially
  rewritten during implementation. The code samples consumed significant RDR space
  but the actual implementations diverged in class structure, method signatures,
  error handling, and feature set. Pseudocode or interface-only definitions would
  have been more appropriate
- **Management script detail**: The full bash script was included but the
  implementation is nearly identical. The script was simple enough that a task list
  ("implement start/stop/restart/logs/status/create-dump/list-indexes") would have
  sufficed
- **MeiliSearch version pinning to unreleased version**: Specifying v1.32 and
  meilisearch-python v0.39.0 when neither was available at implementation time
  created immediate divergence. Version ranges or "latest stable at implementation
  time" would have been more useful
- **Future RDR integration section**: The detailed speculation about RDR-009
  through RDR-071 integration patterns was partly useful for establishing intent
  but the specifics (e.g., "use PDF_DOCS_SETTINGS from this RDR") turned out to
  be approximately correct but with different specifics

---

## Key Takeaways for RDR Process Improvement

1. **Specify interfaces, not implementations, for code that has upstream
   dependencies**: The client code was over-specified because downstream RDRs
   (009, 010, 011) needed additional methods. If the RDR had specified the client
   interface (method signatures and contracts) rather than full implementations,
   it would have been easier to extend without "diverging from the plan."

2. **Use version ranges or "current stable" instead of pinning to unreleased
   versions**: The RDR specified MeiliSearch v1.32 and meilisearch-python v0.39.0,
   neither of which existed at implementation time. This created unnecessary
   divergence entries. RDRs should pin to currently released versions with a note
   about upgrade path.

3. **Design the secret/credential lifecycle explicitly**: Every RDR that introduces
   a service requiring authentication should include a subsection on key generation,
   storage location, rotation, and override mechanisms. The pattern of "just add
   it to .env" consistently underestimates the actual UX work needed for
   zero-configuration local development.

4. **Name CLI commands by checking existing naming patterns first**: The `arc fulltext`
   name was proposed without surveying the existing CLI namespace. A simple check
   ("what are the existing top-level command groups and what naming pattern do they
   follow?") would have caught the inconsistency before the RDR was locked.

5. **Separate the "data ingestion" design from the "data query" design, even in
   a query-focused RDR**: RDR-008 focused on search but deferred indexing entirely.
   The client still needed extensive additions to support indexing, which could have
   been anticipated by asking "what client methods will the indexing RDRs need?"
   during research.

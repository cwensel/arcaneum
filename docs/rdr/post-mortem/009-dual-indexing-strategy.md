# Post-Mortem: RDR-009 Minimal-Command Dual Indexing Workflow

## RDR Summary

RDR-009 proposed a 2-command workflow (`corpus create` + `corpus sync`) to
create searchable corpora across both Qdrant (semantic search) and MeiliSearch
(full-text search) with shared metadata. The approach prioritized minimal CLI
commands and cooperative search workflows over architectural purity, rejecting
both separate-command and auto-create alternatives.

## Implementation Status

Implemented

The core 2-command workflow is fully operational: `arc corpus create` creates
both a Qdrant collection and MeiliSearch index, and `arc corpus sync` indexes
documents to both systems with shared metadata. The implementation significantly
exceeded the original scope with additional corpus management commands (list,
info, items, verify, delete, parity) and operational features like git-aware
sync, parallel chunking, GPU/CPU mode selection, and change detection.

---

## Implementation vs. Plan

### What Was Implemented as Planned

- **2-command workflow**: The core `arc corpus create <name> --type <type>` and
  `arc corpus sync <name> <path>` commands work exactly as proposed, creating
  both Qdrant collections and MeiliSearch indexes in a single operation.
- **Fail-fast error handling**: Corpus creation uses sequential API calls with
  clear error messages. If Qdrant succeeds but MeiliSearch fails, the user is
  informed of the partial state with actionable guidance, matching the RDR's
  error handling strategy.
- **DualIndexDocument shared schema**: Implemented in
  `src/arcaneum/schema/document.py` as a dataclass with shared metadata fields
  (file_path, filename, language, chunk_index, file_extension) and conversion
  functions `to_qdrant_point()` and `to_meilisearch_doc()`, closely matching
  the RDR specification.
- **DualIndexer orchestrator**: Implemented in
  `src/arcaneum/indexing/dual_indexer.py` with `index_batch()`,
  `index_single()`, `delete_by_file_path()`, and `delete_by_project_identifier()`
  methods. The class coordinates indexing to both systems with consistent
  document IDs.
- **Type aliases**: The mapping of short names (`pdf`, `code`, `markdown`) to
  canonical names (`pdf-docs`, `source-code`, `markdown-docs`) works as planned.
- **MeiliSearch index settings templates**: `SOURCE_CODE_SETTINGS`,
  `PDF_DOCS_SETTINGS`, and `MARKDOWN_DOCS_SETTINGS` in
  `src/arcaneum/fulltext/indexes.py` match the planned configuration patterns.
- **Shared metadata field alignment**: Both Qdrant payload and MeiliSearch
  documents share file_path, filename, file_extension, chunk_index, language
  (with the planned naming difference: `programming_language` in Qdrant vs
  `language` in MeiliSearch), and git metadata fields.
- **Collection metadata storage**: Metadata (type, model, created_at) stored
  as a reserved-UUID point in Qdrant collections, enabling type enforcement
  and model inference during sync.
- **Health check before MeiliSearch operations**: The `corpus create` command
  verifies MeiliSearch availability before attempting index creation.
- **Search commands with --corpus flag**: Both `arc search semantic` and
  `arc search text` support `--corpus` (with backwards-compatible `--collection`
  and `--index` hidden options), enabling the cooperative search workflow
  described in the RDR.

### What Diverged from the Plan

- **Document schema uses dataclass, not Pydantic**: The RDR listed Pydantic
  >= 2.12.3 as a dependency and the pseudocode implied a dataclass. The actual
  implementation uses Python `dataclasses.dataclass` rather than a Pydantic
  `BaseModel`. This was pragmatic because the schema is a simple data container
  that does not need Pydantic's validation, serialization, or JSON Schema
  features. Pydantic is still used in `config.py` for configuration management.

- **Content field naming in Qdrant**: The RDR pseudocode showed
  `payload["content"]` for Qdrant. The implementation uses `payload["text"]`
  instead, aligning with the existing convention established by RDR-004 and
  RDR-005 indexers, while MeiliSearch documents use `"content"` as planned.

- **Sync command accepts multiple paths and files**: The RDR specified
  `arc corpus sync <name> <path>` taking a single directory. The implementation
  accepts multiple paths via variadic arguments (`nargs=-1`), individual files,
  and a `--from-file` option for reading paths from a file or stdin. This was
  needed to support real-world workflows like piping file lists.

- **No separate `--models` default override on sync**: The RDR showed
  `--models` as a required option on sync. The implementation reads the
  configured model from collection metadata set during `corpus create`,
  falling back to the `--models` flag only if metadata is missing. This
  eliminates the need to re-specify models on every sync.

- **Corpus type uses short names in CLI**: The RDR specified
  `--type <source-code|pdf-docs|markdown-docs>` with aliases. The CLI actually
  uses `click.Choice(['pdf', 'code', 'markdown'])` with short names as the
  primary choice values, mapped internally to canonical names for MeiliSearch
  settings lookup.

- **Batch size for MeiliSearch is configurable on DualIndexer, not on CLI**:
  The RDR mentioned `--batch-size` as a CLI flag for sync. The implementation
  does not expose `--batch-size` on `corpus sync` but instead uses
  `DualIndexer.DEFAULT_BATCH_SIZE = 100` with parallel batch upload for
  multiple batches.

### What Was Added Beyond the Plan

- **`corpus list` command**: Lists all corpora with parity status (synced,
  qdrant_only, meili_only), querying both systems and merging results. Supports
  `--verbose` for model and chunk count details.

- **`corpus info` command**: Shows combined Qdrant/MeiliSearch statistics for
  a corpus including unique item counts, chunk counts, parity status, vector
  configurations, and HNSW parameters.

- **`corpus items` command**: Lists all indexed items with per-item parity
  status between systems, showing chunk counts from both Qdrant (Q) and
  MeiliSearch (M) columns. Supports code-specific display (repos, branches,
  commit hashes) vs file-specific display (filenames, sizes).

- **`corpus verify` command**: Performs fsck-like integrity checks across both
  systems, detecting incomplete chunk sets, missing documents, and health
  issues.

- **`corpus delete` command**: Deletes both Qdrant collection and MeiliSearch
  index with confirmation prompt, partial deletion handling, and JSON output.

- **`corpus parity` command**: A sophisticated repair tool that detects and
  backfills missing entries between systems. Supports dry-run mode, chunk count
  verification, metadata repair, and automatic creation of missing MeiliSearch
  indexes for qdrant_only corpora.

- **Git-aware sync modes**: `--git-update` skips repos with unchanged commit
  hash; `--git-version` keeps multiple versions indexed. These required
  `GitMetadataSync` and `GitProjectDiscovery` classes not anticipated by
  the RDR.

- **Parallel code chunking**: Uses `ProcessPoolExecutor` with configurable
  `--text-workers` for parallel AST-based code chunking, with process
  priority management (`os.nice(10)`) to avoid starving the main process.

- **Change detection with parity awareness**: The sync command checks both
  Qdrant and MeiliSearch for existing files, identifies files missing from
  either system, and handles backfill automatically during normal sync.

- **File exclusion patterns**: Extensive default exclusion for minified files,
  generated code, vendor directories, and bundle outputs. Content-based
  minification detection reads the first 64KB to check for long lines.

- **Directory prefix skipping**: `--skip-dir-prefix` (default: `_`) skips
  directories starting with configurable prefixes, with
  `--no-skip-dir-prefix` to disable.

- **GPU/CPU mode selection**: `--no-gpu` flag for CPU-only embedding,
  `--max-embedding-batch` for OOM recovery, and `--cpu-workers` for batch
  parallelization in CPU mode.

- **Adaptive progress bar**: `AdaptiveProgress` class extends Rich's
  `Progress` with a smoothing window that grows from 120s to 300s for
  more stable ETAs on long-running syncs.

- **Interaction logging (RDR-018)**: All corpus commands log operations via
  `interaction_logger.start()` / `interaction_logger.finish()` for
  telemetry.

- **Oversized code file skipping**: Files over 1MB (`_MAX_CODE_FILE_SIZE`)
  are skipped during code corpus sync to avoid GPU OOM from generated or
  data files.

### What Was Planned but Not Implemented

- **End-to-end workflow test (`tests/integration/test_2command_workflow.py`)**:
  The RDR specified this file. It does not exist. Integration tests for
  fulltext search workflow exist but no dedicated 2-command workflow test.

- **`--validate` flag on corpus create**: The RDR mentioned
  `arc corpus create --validate` to ping both servers before creating. The
  implementation does check MeiliSearch health inline but does not offer a
  separate `--validate` dry-run flag.

- **Resume support via change detection on sync**: The RDR mentioned
  "Resume support via change detection (index only new/modified files)."
  The implementation has new-file detection (skip already-indexed files) but
  does not resume mid-file if a sync is interrupted.

- **`--file-types` flag behavior**: While `--file-types` exists on the sync
  command, the default file extensions are derived from corpus type internally,
  and the flag only overrides the extension filter rather than enabling
  type-specific chunking logic as the RDR implied.

---

## Drift Classification

| Category | Count | Examples |
| -------- | ----- | -------- |
| **Unvalidated assumption** | 1 | RDR assumed 16-20 hours total implementation; actual scope was significantly larger due to Day 2 operations |
| **Framework API detail** | 2 | Content field name (`text` vs `content` in Qdrant); CLI type choices use short names not canonical names |
| **Missing failure mode** | 1 | MeiliSearch filter-based deletion not fully implemented (returns 0,0 in DualIndexer) |
| **Missing Day 2 operation** | 3 | Parity repair, corpus deletion, corpus verification not planned |
| **Deferred critical constraint** | 0 | |
| **Over-specified code** | 2 | DualIndexer pseudocode and sync_directory pseudocode were substantially rewritten |
| **Under-specified architecture** | 2 | Git-aware sync modes and change detection parity logic not planned |
| **Scope underestimation** | 2 | Sync command grew to 3000+ lines with parallel chunking, GPU management, exclusion logic; corpus management commands added 6 subcommands |
| **Internal contradiction** | 0 | |
| **Missing cross-cutting concern** | 2 | GPU memory management for embedding; file exclusion patterns for generated/minified code |

---

## RDR Quality Assessment

### What the RDR Got Right

- **User-driven prioritization**: The RDR correctly identified that minimizing
  commands (2-command workflow) was the primary goal over architectural purity.
  This framing survived implementation intact and the resulting
  `corpus create` and `corpus sync` workflow is the actual user experience.

- **Shared metadata field alignment table**: The table mapping Qdrant field
  names to MeiliSearch field names was directly useful during implementation.
  The naming conventions (`programming_language` vs `language`,
  `git_project_name` vs `project`) are reflected in the actual conversion
  functions.

- **Fail-fast over distributed transactions**: The decision to use simple
  sequential API calls with fail-fast error handling (no rollback) was
  correct. The implementation follows this pattern exactly, and the partial
  state scenario (Qdrant created but MeiliSearch fails) is handled with
  clear error messages.

- **Complexity reassessment**: The RDR honestly documented the over-estimate
  (60-80 hours) and corrected it to 16-20 hours. While the final scope was
  larger than 20 hours, the core dual-indexing logic itself was indeed simple.

- **Alternative analysis**: The three alternatives (separate commands,
  separate pipelines, auto-create) were well-analyzed with clear rejection
  reasons. None of these alternatives were adopted during implementation.

### What the RDR Missed

- **Day 2 corpus management**: The RDR mentioned `corpus list`, `corpus info`,
  and `corpus delete` only as "Future Enhancements" but all three were needed
  immediately upon implementation to make the system usable. Users need to
  see what exists, inspect status, and clean up mistakes. These should have
  been in the core plan.

- **Parity drift between systems**: The RDR did not consider that Qdrant and
  MeiliSearch can fall out of sync (partial uploads, one system down during
  sync, interrupted operations). The `corpus parity` command was a significant
  implementation effort to detect and repair these drifts.

- **Git-aware indexing for code corpora**: The RDR treated code indexing as
  simple file discovery but real code corpora need git awareness: tracking
  which repos/branches/commits are indexed, skipping unchanged repos, and
  supporting multi-version indexing.

- **GPU/CPU memory management**: The RDR's performance section estimated
  "~4-6GB total" but did not address MPS GPU OOM, batch size tuning,
  oversized file skipping, or CPU fallback. These became essential for
  production use on Apple Silicon.

- **File exclusion and filtering**: The RDR mentioned `--file-types` but did
  not anticipate the need to exclude minified files, generated code, vendor
  directories, or other problematic content that wastes indexing time or causes
  OOM.

### What the RDR Over-specified

- **Pseudocode for all four components**: The RDR included detailed Python
  code for `create_corpus`, `sync_directory`, `DualIndexDocument`, and
  `DualIndexer`. All four were substantially rewritten during implementation.
  The pseudocode gave a false sense of precision while the actual
  implementation needed different method signatures, error handling patterns,
  and data flow. High-level interface descriptions would have been more useful.

- **Performance benchmarks**: The RDR provided specific throughput numbers
  ("Sync 1000 PDFs: ~18 minutes", "Sync 10,000 Python files: ~36 minutes")
  that were speculative estimates without benchmarking. These numbers were
  neither validated nor referenced during implementation.

- **Qdrant/MeiliSearch field mapping table**: While the field alignment concept
  was useful, the specific field list was incomplete (missing file_hash,
  file_size, quick_hash, document_type, section, tags, headings, git_remote_url,
  git_version_identifier) and some planned fields had different final names.

---

## Key Takeaways for RDR Process Improvement

1. **Include Day 2 operations in the core plan, not as future enhancements**:
   The RDR deferred `corpus list`, `corpus info`, and `corpus delete` to
   "Future Enhancements" but all three were required immediately. Any command
   that creates resources must plan for listing, inspecting, and deleting those
   resources in the same RDR. A simple heuristic: if `create` is in scope,
   `list`, `info`, and `delete` must be too.

2. **Replace pseudocode with interface contracts and state diagrams**: All four
   pseudocode blocks in this RDR were substantially rewritten. The effort spent
   writing detailed Python code would have been better spent defining input/output
   contracts, error states, and data flow diagrams. Pseudocode creates a false
   sense of implementation readiness.

3. **Plan for drift between distributed systems explicitly**: When an RDR
   introduces dual writes to independent systems, it must address what happens
   when they diverge. This RDR's error handling only covered creation-time
   failures, not ongoing parity drift from interrupted syncs, partial uploads,
   or one system being unavailable during indexing. A "what can go wrong after
   Day 1" section would have surfaced the parity problem earlier.

4. **Estimate scope based on the full command surface, not just the happy path**:
   The RDR estimated 16-20 hours based on `create` and `sync` alone. The actual
   implementation included 8 corpus subcommands (create, sync, list, info,
   items, verify, delete, parity), parallel chunking, git-aware modes, GPU
   management, and extensive file filtering. Counting the number of CLI commands
   and operational scenarios upfront would have produced a more realistic
   estimate.

5. **Drop speculative performance numbers unless backed by benchmarks**: The
   RDR included specific throughput estimates and overhead percentages that
   were never validated. These numbers added length without adding value. If
   performance matters, require a benchmark spike as a prerequisite. If it does
   not, omit the numbers and state the performance constraint qualitatively
   (e.g., "dual indexing overhead should be acceptable for interactive use").

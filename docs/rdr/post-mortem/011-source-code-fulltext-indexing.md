# Post-Mortem: RDR-011 Source Code Full-Text Indexing

## RDR Summary

RDR-011 proposed a git-aware source code full-text indexing pipeline for MeiliSearch
that indexes at function/class granularity using tree-sitter AST parsing. The system
would complement RDR-005's semantic search (Qdrant) with exact phrase and keyword
matching, supporting multi-branch git repositories, metadata-based change detection,
and integration with RDR-009's dual indexing via `arc corpus sync`.

## Implementation Status

Implemented

All core components were implemented: the AST function/class extractor, the MeiliSearch
code indexer with git-aware change detection, the CLI command (`arc index text code`),
index management commands (`arc indexes list-projects`, `arc indexes delete-project`),
and the `SOURCE_CODE_FULLTEXT_SETTINGS` index configuration. Comprehensive unit,
integration, and multi-branch tests were created.

---

## Implementation vs. Plan

### What Was Implemented as Planned

- **AST function/class extractor** (`src/arcaneum/indexing/fulltext/ast_extractor.py`):
  Created as a new module using tree-sitter directly via `get_parser()`, exactly as
  specified. The `CodeDefinition` dataclass, `ASTFunctionExtractor` class, and
  `DEFINITION_TYPES` mapping all match the RDR's design closely.

- **CodeDefinition dataclass**: Fields match the spec (`name`, `qualified_name`,
  `code_type`, `start_line`, `end_line`, `content`, `file_path`). A `line_count`
  computed property was added.

- **DEFINITION_TYPES language mapping**: Python, JavaScript, TypeScript, Java, Go,
  and Rust node types match the RDR. Additional languages (C, C++, C#, Ruby, PHP,
  Swift, Kotlin, Scala) were included as the RDR indicated with "130+ languages
  supported."

- **LANGUAGE_MAP**: File extension to tree-sitter language mapping implemented
  with all extensions listed in the RDR plus additional ones (bash, R, Lua, Vim,
  Elisp, Clojure, Elixir, Erlang, Haskell, OCaml, Nim, Perl).

- **Fallback strategy**: Unsupported languages and AST failures fall back to
  module-level indexing, exactly as planned.

- **Module-level code extraction**: The `_extract_module_code()` method identifies
  lines not covered by any definition, as the RDR specified.

- **Nested function handling**: Qualified names (`MyClass.method`) implemented via
  recursive `_extract_from_node()` with `parent_name` parameter.

- **Source code MeiliSearch indexer** (`src/arcaneum/indexing/fulltext/code_indexer.py`):
  The `SourceCodeFullTextIndexer` class follows the RDR's design with git discovery,
  AST extraction, document building, and batch upload.

- **MeiliSearch document schema**: All fields from the RDR are present: `id`,
  `content`, `function_name`, `class_name`, `qualified_name`, `filename`,
  `git_project_identifier`, `git_project_name`, `git_branch`, `git_commit_hash`,
  `git_remote_url`, `file_path`, `start_line`, `end_line`, `line_count`, `code_type`,
  `programming_language`. The `file_extension` field was added beyond the original
  schema.

- **Git-aware metadata sync** (`GitCodeMetadataSync` in
  `src/arcaneum/indexing/fulltext/sync.py`): Implements the MeiliSearch-as-source-of-truth
  pattern with `get_indexed_projects()`, `should_reindex_project()`,
  `delete_project_documents()`, and result caching.

- **Batch upload**: Default batch size of 1000 documents with per-batch waiting,
  as specified.

- **SOURCE_CODE_FULLTEXT_SETTINGS** (`src/arcaneum/fulltext/indexes.py`): Searchable
  attributes, filterable attributes, sortable attributes, typo tolerance, stop words,
  and pagination settings all match the RDR specification exactly.

- **CLI command** `arc index text code`: Registered as subcommand with `--index`,
  `--force`, `--depth`, `--batch-size`, `--verbose`, `--json` options. Mirrors
  `arc index text pdf` pattern as planned.

- **Index management commands**: `arc indexes list-projects` and
  `arc indexes delete-project` implemented in `src/arcaneum/cli/fulltext.py` as
  specified.

- **Test files created**: `tests/indexing/fulltext/test_ast_extractor.py`,
  `tests/indexing/fulltext/test_code_indexer.py`, and
  `tests/integration/test_fulltext_code_indexing.py` all match the planned test
  file locations.

### What Diverged from the Plan

- **Document ID sanitization**: The RDR proposed IDs like
  `{git_project_identifier}:{file_path}:{qualified_name}:{start_line}` using colons.
  The implementation sanitizes IDs by replacing non-alphanumeric characters (including
  `#` from identifiers and `/` from paths) with underscores, collapsing multiple
  underscores, and truncating to 511 characters. This was necessary because MeiliSearch
  IDs only accept alphanumeric characters, hyphens, and underscores.

- **Deletion approach**: The RDR proposed MeiliSearch filter-based deletion via
  `index.delete_documents(filter=...)`. The implementation uses a search-then-delete
  approach: querying for documents matching the `git_project_identifier` filter,
  collecting their IDs, and deleting by ID in batches. This is because the
  meilisearch-python client's `delete_documents()` method works with document IDs,
  not arbitrary filter expressions, for reliable deletion.

- **Sync queries use search, not get_documents**: The RDR proposed querying via
  `index.get_documents()` with `attributes_to_retrieve`. The implementation uses
  `client.search()` with empty query and pagination. This is because the
  `get_documents` API in meilisearch-python has different parameter semantics than
  shown in the RDR pseudocode, while `search()` provides consistent offset/limit
  pagination and filter support.

- **Python decorated definitions**: The RDR did not mention handling Python decorators
  specifically. The implementation adds `decorated_definition` to
  `DEFINITION_TYPES["python"]` and includes special logic in `_extract_from_node()`
  to find the actual definition inside a decorator, ensuring `@decorator` content is
  included in the definition's line range and content.

- **tree-sitter name extraction**: The RDR showed only
  `node.child_by_field_name("name")`. The implementation tries multiple field names
  (`"name"`, `"identifier"`) and falls back to searching for `identifier` or `name`
  child node types directly, because different languages use different field names
  for definition identifiers.

- **`file_extension` field**: The RDR's document schema did not include `file_extension`
  as a document field (it was only in `filterableAttributes`). The implementation
  adds it to each document to support the filterable attribute.

### What Was Added Beyond the Plan

- **Parallel file processing**: The `SourceCodeFullTextIndexer` supports parallel
  AST extraction via `ProcessPoolExecutor` with configurable `--workers` option
  (`None`=auto at cpu/2, `0` or `1`=sequential, `N`=explicit). This required a
  module-level `_extract_definitions_worker()` function for pickling, conversion
  of `CodeDefinition` objects to dicts for cross-process serialization, and
  integration with `src/arcaneum/indexing/common/multiprocessing.py` for consistent
  fork context and signal handling. The RDR did not mention parallelization.

- **Simple (non-git) mode**: A `--no-git` flag and `_index_code_simple()` function
  provide whole-file indexing without git awareness, useful for indexing individual
  files or directories outside git repositories. The RDR only specified git-aware mode.

- **`--from-file` option**: The CLI command accepts file lists from a file or stdin,
  mirroring the PDF indexing command's capability. Not mentioned in the RDR.

- **`index_single_project()` method**: A convenience method for indexing a specific
  git repository without directory discovery was added to
  `SourceCodeFullTextIndexer`. The RDR only described `index_directory()`.

- **`get_project_document_count()` method**: Added to `GitCodeMetadataSync` to
  support the `delete-project` CLI command's confirmation prompt showing how many
  documents will be deleted.

- **`detect_language()` and `supports_ast_extraction()` methods**: Public utility
  methods added to `ASTFunctionExtractor` for external inspection of language support.

- **`get_supported_languages()` and `get_supported_extensions()` class methods**:
  Added to `ASTFunctionExtractor` for discoverability.

- **Interaction logging (RDR-018)**: All CLI commands include `interaction_logger`
  calls for telemetry. Not mentioned in RDR-011 (RDR-018 was a separate concern).

- **Process priority (`os.nice(10)`)**: Worker processes set themselves to low
  priority for background processing. Not specified in the RDR.

- **Additional index management commands**: `arc indexes verify`, `arc indexes items`,
  `arc indexes export`, `arc indexes import`, and `arc indexes update-settings` were
  implemented in `fulltext.py` beyond what RDR-011 specified (which only called for
  `list-projects` and `delete-project`).

- **`format_location()` in search output**: The full-text search result display was
  enhanced to show line ranges and function/class names for source code results
  (e.g., `file.py:42-67 (my_func function)`).

### What Was Planned but Not Implemented

- **RDR-009 dual indexing verification**: Step 4 of the implementation plan called
  for verifying that `DualIndexDocument` schema supports function-level fields and
  that `arc corpus sync` correctly routes code to both systems. While `arc corpus sync`
  does index code to both Qdrant and MeiliSearch, the MeiliSearch side of corpus sync
  uses the chunk-level schema (from RDR-005/RDR-008), not the function-level schema
  from RDR-011. The `DualIndexer.delete_by_file()` and `delete_by_project()` methods
  note that MeiliSearch filter deletion is not implemented ("Return 0 for meili since
  we can't easily count"). The standalone `arc index text code` command does use
  the function-level schema, but `arc corpus sync` and the standalone path remain
  architecturally separate rather than sharing a single-pass pipeline as the RDR's
  dual indexing diagram depicted.

- **Documentation updates (Step 7)**: The RDR called for README updates, CLI reference,
  cooperative workflow guide, and troubleshooting guide. These documentation updates
  are not part of the code implementation and their status is unknown from the codebase
  alone.

- **Performance targets validation**: The RDR specified targets (50-100 files/sec,
  >95% AST success, <500ms deletion, <5s metadata query). No performance benchmarking
  infrastructure was added to validate these targets.

---

## Drift Classification

| Category | Count | Examples |
| -------- | ----- | -------- |
| **Unvalidated assumption** | 1 | RDR assumed `index.delete_documents(filter=...)` would work directly; actual API requires search-then-delete-by-ID |
| **Framework API detail** | 3 | MeiliSearch ID character constraints (no colons/hashes); `get_documents` vs `search` API semantics; `delete_documents` filter parameter |
| **Missing failure mode** | 1 | Python decorated definitions not handled in RDR's AST extraction design |
| **Missing Day 2 operation** | 0 | |
| **Deferred critical constraint** | 1 | Dual indexing via `arc corpus sync` still uses chunk-level schema, not function-level; the single-pass pipeline shown in the RDR diagram was not realized |
| **Over-specified code** | 2 | `GitMetadataSync.get_indexed_projects()` pseudocode used `index.get_documents()` (rewritten to use `search()`); document ID format rewritten entirely |
| **Under-specified architecture** | 1 | Parallel processing for AST extraction was not considered; required worker serialization, process pool management, and signal handling infrastructure |
| **Scope underestimation** | 1 | Simple (non-git) mode, `--from-file` support, additional index management commands, and interaction logging added significant implementation beyond the plan |
| **Internal contradiction** | 0 | |
| **Missing cross-cutting concern** | 1 | `file_extension` field needed in document schema to support the filterable attribute declared in settings |

### Drift Category Definitions

- **Unvalidated assumption** -- a claim presented as fact but never verified by spike/POC
- **Framework API detail** -- method signatures, interface contracts, or config syntax wrong
- **Missing failure mode** -- what breaks, what fails silently, recovery path not considered
- **Missing Day 2 operation** -- bootstrap, CI/CD, removal, rollback, migration not planned
- **Deferred critical constraint** -- downstream use case that validates the approach was out of scope
- **Over-specified code** -- implementation code that was substantially rewritten
- **Under-specified architecture** -- architectural decision that should have been made but wasn't
- **Scope underestimation** -- sub-feature that grew into its own major effort
- **Internal contradiction** -- research findings or stated principles conflicting with the proposal
- **Missing cross-cutting concern** -- versioning, licensing, config cache, deployment model, etc.

---

## RDR Quality Assessment

### What the RDR Got Right

- **Function/class-level granularity decision**: The analysis of whole-file vs
  line-based vs function/class-level was thorough, and the chosen approach proved
  correct in implementation. The edge cases (module-level code, nested functions,
  large functions) were all relevant and handled.

- **Tree-sitter direct usage over LlamaIndex CodeSplitter**: The explicit clarification
  (added 2026-01-15) that tree-sitter should be used directly, not through CodeSplitter,
  prevented a likely misimplementation. The comparison table between `ast_chunker.py`
  and `ast_extractor.py` was directly useful.

- **MeiliSearch index settings**: `SOURCE_CODE_FULLTEXT_SETTINGS` was implemented
  verbatim. The choices for searchable, filterable, and sortable attributes, typo
  tolerance thresholds, empty stop words, and high pagination limits were all correct.

- **Git-aware change detection pattern**: The MeiliSearch-as-source-of-truth approach
  with composite identifiers, commit hash comparison, and branch-specific operations
  was directly implementable and worked as designed.

- **CLI naming convention**: The `arc index text code` naming parallel to
  `arc index text pdf` was correct and implemented exactly. The `arc indexes`
  management commands fit naturally.

- **Batch upload strategy**: The 1000-document batch size with per-batch waiting
  was implemented as-is and proved appropriate.

- **Research issue decomposition**: Breaking the investigation into seven focused
  beads issues (arcaneum-86 through arcaneum-92) produced thorough coverage of
  each concern.

### What the RDR Missed

- **MeiliSearch API constraints**: The RDR's pseudocode used MeiliSearch API
  patterns that did not match the actual meilisearch-python client behavior.
  Document IDs cannot contain colons or hash characters. `delete_documents()`
  with a filter expression works differently than shown. `get_documents()` has
  different parameter semantics than `search()`. A brief spike against the actual
  API would have caught these.

- **Python decorator handling**: Decorated functions are common in Python codebases,
  but the RDR's DEFINITION_TYPES for Python only listed `function_definition` and
  `class_definition`. The `decorated_definition` node type wraps the actual
  definition and must be handled specially to include decorator content in line ranges.

- **Parallelization requirements**: For large codebases, sequential AST extraction
  is a bottleneck. The implementation added parallel processing with
  `ProcessPoolExecutor`, which required non-trivial infrastructure (worker
  serialization, process pool management, signal handling, `os.nice()` for priority).

- **Document field completeness**: The filterable attribute `file_extension` was
  declared in `SOURCE_CODE_FULLTEXT_SETTINGS` but not included in the document
  schema. The implementation had to add it.

- **Dual indexing gap**: The RDR depicted a single-pass pipeline diagram where
  `arc corpus sync` would use both `ast_chunker.py` (for Qdrant chunks) and
  `ast_extractor.py` (for MeiliSearch function-level documents) in parallel.
  In practice, `arc corpus sync` feeds MeiliSearch with chunk-level documents,
  not function-level documents. The standalone `arc index text code` command is
  the only path that produces function-level MeiliSearch documents. This gap means
  users wanting both semantic and function-level full-text search must run two
  separate commands.

### What the RDR Over-specified

- **Extensive pseudocode for standard patterns**: The `GitMetadataSync`,
  `SourceCodeFullTextIndexer`, and `ASTFunctionExtractor` classes each had
  50-100 lines of pseudocode that were substantially rewritten during implementation.
  The pseudocode gave a false sense of precision (especially around MeiliSearch API
  calls) while masking API incompatibilities.

- **Alternative analysis depth**: Four alternatives were analyzed in detail (whole-file,
  line-based, sequential dual indexing, regex extraction). The function/class-level
  approach was clearly superior, and the alternatives could have been dismissed more
  briefly. The regex alternative in particular was never seriously viable given
  existing tree-sitter infrastructure.

- **Performance targets without benchmarking plan**: Specific targets (50-100
  files/sec, >95% AST success rate, <500ms deletion) were stated without a plan
  to measure them. No performance test infrastructure was created during
  implementation.

---

## Key Takeaways for RDR Process Improvement

1. **Validate framework API calls with a spike before writing pseudocode**: Three
   of the implementation divergences (document ID format, delete_documents filter,
   get_documents semantics) came from MeiliSearch API assumptions in pseudocode
   that were never tested. A 30-minute spike against the actual client library
   would have caught all three. RDR pseudocode that uses external APIs should be
   marked "Assumed" unless verified, and the Finalization Gate should require at
   least one verified API interaction per external service.

2. **Specify document schema fields to match index settings**: The RDR declared
   `file_extension` as a filterable attribute in `SOURCE_CODE_FULLTEXT_SETTINGS`
   but did not include it in the document schema. When an RDR specifies index/database
   settings, cross-reference every filterable/searchable/sortable attribute against
   the document schema to ensure all referenced fields are actually populated.

3. **Include parallelization strategy for CPU-bound pipelines**: The RDR's architecture
   diagram showed a sequential pipeline without considering that AST parsing across
   hundreds of files is CPU-bound. For any pipeline that processes many files, the
   RDR should explicitly address whether parallelization is needed, and if so, outline
   the serialization and process management approach. The "Batch Upload Optimization"
   research track (arcaneum-91) focused on MeiliSearch upload batching but missed the
   AST extraction bottleneck.

4. **Distinguish pseudocode intent from API-accurate code**: RDR pseudocode served
   two purposes that should be separated: (1) communicating architectural intent
   and data flow, and (2) providing copy-paste implementation guidance. The first
   purpose was well served; the second created false confidence that led to rework.
   Future RDRs should label pseudocode blocks as either "conceptual" (showing intent)
   or "verified" (tested against actual APIs).

5. **Verify dual indexing integration end-to-end, not just at the schema level**: The
   RDR stated that `DualIndexDocument` schema "already supports function-level fields"
   and declared the dual indexing verified. However, schema compatibility does not
   guarantee pipeline integration. The `arc corpus sync` pipeline still produces
   chunk-level MeiliSearch documents, not function-level ones. When an RDR depends on
   another RDR's infrastructure, verify the actual data flow path, not just type
   compatibility.

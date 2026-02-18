# Post-Mortem: RDR-005 Git-Aware Source Code Indexing with AST Chunking

## RDR Summary

RDR-005 proposed a git-aware source code indexing pipeline for Qdrant that
discovers git projects with depth control, extracts metadata (commit hash,
branch, remote URL), chunks code with tree-sitter AST parsing (165+ languages),
and uses metadata-based sync with Qdrant as the single source of truth. The
approach centered on a composite identifier pattern (`project#branch`) for
multi-branch support and filter-based branch-specific deletion.

## Implementation Status

Implemented

All core modules shipped and are in active production use. The six planned
source files were created as specified. The MCP plugin wrapper was not
implemented. Several features beyond the original plan were added during
implementation, including GPU acceleration, streaming uploads, collection
verification, and full-text code indexing via a separate RDR (RDR-011).

---

## Implementation vs. Plan

### What Was Implemented as Planned

- **Git project discovery** with recursive `.git` directory search and
  configurable depth control using `find` command
  (`src/arcaneum/indexing/git_operations.py`)
- **Git metadata extraction** with robust error handling for detached HEAD
  states, shallow clones, missing remotes, and credential sanitization
- **Composite identifier pattern** (`project#branch`) for multi-branch
  support, implemented as a `GitMetadata.identifier` property
- **Metadata-based sync** using Qdrant scroll API as single source of truth,
  with caching, force-refresh, and stale identifier detection
  (`src/arcaneum/indexing/git_metadata_sync.py`)
- **AST-aware chunking** via tree-sitter-language-pack + LlamaIndex
  CodeSplitter with automatic fallback to line-based chunking
  (`src/arcaneum/indexing/ast_chunker.py`)
- **15+ language support** with a comprehensive LANGUAGE_MAP covering all
  originally required languages plus 20+ additional ones
- **Filter-based branch-specific deletion** for efficient re-indexing
  (Qdrant filter on `git_project_identifier`)
- **Batch upload with retry logic** using tenacity exponential backoff
- **Metadata schema** matching the plan: `git_project_identifier`,
  `git_project_name`, `git_branch`, `git_commit_hash`, `git_remote_url`,
  `text_extraction_method`, `ast_chunked`, `programming_language`
- **Incremental sync** with commit hash comparison and force mode bypass
- **Read-only git operations** -- no git mutations
- **CLI integration** via `arc index source` command with depth, force,
  model, chunk size, and verbose options
- **Dependency selection**: GitPython, tree-sitter-language-pack,
  llama-index-core, fastembed, tenacity, rich -- all as planned
- **Test coverage** for git operations, AST chunking, metadata sync,
  Qdrant indexer, and pipeline integration

### What Diverged from the Plan

- **Depth calculation adjusted**: The RDR planned `maxdepth = depth + 1`.
  The implementation uses `depth + 2` because searching for `.git` directories
  requires accounting for both the project directory level and the `.git`
  directory itself. The code includes detailed comments explaining the
  reasoning. This is a correctness fix discovered during implementation.

- **Default embedding model differs**: The RDR recommended
  jina-code-embeddings-1.5b (1536D, 32K context) as primary. The
  implementation defaults to jina-embeddings-v2-base-code (768D, 8K
  context) because it has proven FastEmbed support and lower resource
  requirements. The 1.5b model is supported as an option
  (`jina-code-1.5b`) but is not the default.

- **Batch size increased**: The RDR planned 150 chunks per upload batch.
  The implementation uses 512 as default, optimized for GPU-accelerated
  embedding throughput based on production tuning (referenced as
  arcaneum-2m1i, arcaneum-i7oa).

- **store_type value simplified**: The RDR specified `"source-code"` as
  the store_type value. The implementation uses `"code"` for brevity.

- **Embedding architecture abstracted**: The RDR planned direct FastEmbed
  integration (`FastEmbedModel` class). The implementation introduced an
  `EmbeddingClient` abstraction that wraps multiple backends with GPU
  support (MPS/CUDA), parallel embedding workers, batch size auto-tuning,
  and OOM recovery with CPU fallback.

- **File processing parallelized**: The RDR described a sequential
  file processing loop. The implementation uses `ProcessPoolExecutor`
  with configurable worker count (default: `cpu_count() // 2`) for
  parallel AST chunking, with a module-level `_process_file_worker`
  function for pickling compatibility.

- **Code analysis fields declared but not populated**: The metadata schema
  includes `has_functions`, `has_classes`, and `has_imports` fields
  exactly as planned, but they always default to `False`. The pipeline
  never sets them because the AST chunker does not perform structural
  analysis -- it only chunks code.

- **Remote URL fallback return type**: The RDR's `_extract_remote_url`
  returned `"unknown"` when no remotes exist. The implementation returns
  `None`, which is more Pythonic and avoids ambiguity between "no remote"
  and a remote literally named "unknown".

### What Was Added Beyond the Plan

- **GPU acceleration**: Full MPS (Apple Silicon) and CUDA support with
  automatic device detection, batch size auto-tuning based on available
  GPU memory, OOM recovery with CPU fallback, and cache clearing between
  projects
- **Streaming upload mode**: Embeds and uploads each batch immediately
  instead of accumulating all embeddings in memory, reducing memory
  usage from O(total_chunks) to O(batch_size)
- **Collection verification** (`src/arcaneum/indexing/verify.py`):
  fsck-like integrity checking that detects incomplete chunk sets and
  identifies projects needing repair
- **Repair mode**: `repair_targets` parameter allows targeted re-indexing
  of specific projects found incomplete during verification
- **Full-text code indexing** (RDR-011): A parallel pipeline for
  MeiliSearch using `ASTFunctionExtractor` that extracts discrete
  function/class definitions with line ranges, reusing RDR-005's git
  infrastructure
- **Bulk upload mode**: Disables HNSW index construction during upload
  for 1.3-1.5x speedup, then rebuilds index after completion
- **Hard max chars enforcement**: Post-processing step that re-splits
  oversized chunks to prevent embedding OOM, with overlap for context
  continuity
- **Minified code handling**: Long-line splitting with natural break
  point detection (semicolons, braces, commas)
- **CPU monitoring and pipeline profiling**: Stage-level timing breakdown
  for file processing, embedding, and upload phases
- **Process priority management**: Worker processes set to low priority
  (`os.nice(10)`) with opt-out via `--not-nice` flag
- **Interaction logging** (RDR-018): Structured logging of indexing
  operations for observability
- **Version identifier**: `GitMetadata.version_identifier` property
  (`project#branch@commit_short`) enabling multi-version indexing where
  multiple commits of the same branch coexist
- **Collection type metadata**: Type validation ensuring code collections
  are not accidentally used as PDF/markdown collections
- **DualIndexDocument integration**: `apply_git_metadata()` helper
  function centralizing git metadata assignment for consistency between
  semantic and full-text indexing paths

### What Was Planned but Not Implemented

- **MCP plugin wrapper** (`plugins/qdrant-indexer/mcp_server.py`): The
  RDR planned an MCP server exposing `index_source_code()`,
  `check_indexing_status()`, and `delete_project()` tools. This was not
  built. The CLI command serves as the primary interface instead.
- **Level 2 file mtime optimization**: The RDR planned a file-level
  mtime check after commit-level change detection to skip unchanged
  files within a changed commit. The implementation uses commit-hash-only
  detection and re-indexes all files when the commit changes, which is
  simpler and avoids the mtime unreliability concerns the RDR itself
  flagged.
- **gRPC transport**: The RDR mentioned gRPC support for 2-3x faster
  uploads. The `QdrantIndexer` docstring references it, but the
  implementation uses HTTP exclusively. The bulk upload mode
  optimization partially compensates.
- **Performance benchmarks**: The RDR specified detailed performance
  targets (100-200 files/sec, <1s metadata extraction, >95% AST success
  rate, <500ms deletion, <5s metadata query). No formal benchmark suite
  was created.
- **Security validation tests**: The RDR specified credential
  sanitization tests and metadata security validation. Unit tests cover
  URL sanitization, but there is no dedicated security test suite.

---

## Drift Classification

| Category | Count | Examples |
| -------- | ----- | -------- |
| **Unvalidated assumption** | 2 | jina-code-1.5b as practical default; 150-chunk batch size optimal |
| **Framework API detail** | 1 | Depth calculation off by 1 (depth+1 vs depth+2) |
| **Missing failure mode** | 1 | GPU OOM during embedding not anticipated (required streaming mode, CPU fallback, hard max chars) |
| **Missing Day 2 operation** | 1 | Collection verification/repair not planned |
| **Deferred critical constraint** | 1 | MCP plugin deferred indefinitely |
| **Over-specified code** | 2 | Code analysis fields (has_functions etc.) never populated; Level 2 mtime optimization unnecessary |
| **Under-specified architecture** | 2 | Embedding abstraction layer not anticipated; parallel file processing not designed |
| **Scope underestimation** | 1 | Full-text indexing (RDR-011) grew from "future enhancement" to its own pipeline |
| **Internal contradiction** | 0 | |
| **Missing cross-cutting concern** | 2 | GPU/device management not considered; memory management during large indexing jobs |

---

## RDR Quality Assessment

### What the RDR Got Right

- **Composite identifier pattern**: The `project#branch` design proved
  exactly right for multi-branch support. It shipped unchanged and
  enabled branch-specific deletion, querying, and comparison as designed.
- **Metadata-based sync architecture**: Using Qdrant as single source of
  truth (following RDR-004 pattern) eliminated external state management
  complexity. The scroll-based approach shipped almost verbatim.
- **tree-sitter + LlamaIndex selection**: The technology choice for AST
  chunking was well-researched and correct. The 165+ language coverage
  claim held up, and the fallback-to-line-based strategy works reliably.
- **Filter-based deletion advantage**: The research finding that Qdrant
  filter-based deletion is 40-100x faster than ChromaDB's ID-based
  approach informed the architecture correctly and simplified the
  change detection strategy.
- **No SQLite checkpoint decision**: Removing the SQLite checkpoint
  database in favor of idempotent re-indexing was the right call. It
  reduced complexity without measurable cost.
- **Read-only git operations constraint**: This safety requirement
  survived implementation unchanged and prevented any risk of git state
  corruption.
- **Research-driven alternatives analysis**: The rejection of ASTChunk
  (4 languages), jina-v3 (worse performance + licensing), and
  incremental-only updates were all validated by implementation
  experience.

### What the RDR Missed

- **GPU/device management**: The RDR did not consider GPU acceleration
  at all, despite targeting a 1.5B parameter embedding model. The
  implementation required significant GPU memory management (MPS OOM
  detection, batch size auto-tuning, CPU fallback, cache clearing).
  This became one of the largest implementation efforts.
- **Memory pressure during large jobs**: Processing large repositories
  creates significant memory pressure from accumulated embeddings,
  chunk objects, and process pool overhead. The streaming upload mode,
  explicit GC calls, and future reference cleanup were all unplanned.
- **Collection integrity and repair**: No consideration was given to
  what happens when indexing is interrupted mid-project. The verification
  and repair features were added after production use revealed the need.
- **Minified/compressed code**: The RDR assumed code files have
  reasonable line lengths. Minified JavaScript (entire file on one line)
  caused memory issues that required special handling.
- **Process priority and system responsiveness**: Running CPU-intensive
  AST parsing and embedding on all cores makes the system unresponsive.
  The nice/priority management was not anticipated.

### What the RDR Over-specified

- **Code analysis fields**: The metadata schema included `has_functions`,
  `has_classes`, and `has_imports` fields that were never populated. The
  AST chunker splits code but does not analyze structure. These fields
  add payload size with no value.
- **Level 2 file mtime optimization**: The RDR designed a two-level
  change detection strategy with file-level mtime checks. This was
  unnecessary because commit-level detection is sufficient -- when a
  commit changes, re-indexing all files is fast enough given the AST
  chunking and embedding pipeline bottleneck.
- **Detailed performance targets**: The RDR specified targets like
  "100-200 files/sec" and "<500ms deletion" that were never formally
  validated. The actual performance characteristics depend heavily on
  model size, GPU availability, and file complexity, making static
  targets misleading.
- **Extensive code samples**: The RDR included complete class
  implementations that were substantially rewritten. The
  `SourceCodeIndexer` class gained parallel processing, streaming mode,
  profiling, and repair support -- none of which existed in the
  pseudocode.

---

## Key Takeaways for RDR Process Improvement

1. **Spike GPU/device requirements when specifying large model usage**:
   The RDR recommended a 1.5B parameter model without considering GPU
   memory management, OOM recovery, or device-specific batch limits.
   Future RDRs that specify embedding models should include a spike on
   device requirements and memory constraints, especially for Apple
   Silicon MPS where memory behavior differs from NVIDIA CUDA.

2. **Distinguish schema fields that require implementation from those
   that are aspirational**: The `has_functions`, `has_classes`, and
   `has_imports` fields were specified in the metadata schema but had no
   corresponding implementation plan in any step. Future RDRs should
   mark schema fields that require additional implementation work
   (structural analysis, in this case) and either include that work in
   the implementation plan or defer the fields entirely.

3. **Design for interruption recovery in any pipeline that processes
   batches**: The RDR stated "idempotent re-indexing" as the crash
   recovery strategy but did not design verification or repair
   mechanisms. Any pipeline that processes items in batches should plan
   for partial completion detection and targeted repair, not just full
   re-runs.

4. **Replace speculative performance targets with benchmark acceptance
   criteria**: The RDR included specific performance numbers (100-200
   files/sec, <500ms deletion) that were never validated because no
   benchmark harness was built. Future RDRs should either commit to
   building benchmarks as an implementation step or omit specific
   numbers in favor of qualitative goals (e.g., "deletion should be
   sub-second" rather than "<500ms").

5. **Account for memory lifecycle in batch processing pipelines**: The
   RDR's pipeline design accumulated all chunks and embeddings in memory
   before uploading. For large repositories this caused memory exhaustion.
   Future RDRs designing batch pipelines should explicitly address memory
   bounds -- whether through streaming, explicit cleanup, or back-pressure
   mechanisms.

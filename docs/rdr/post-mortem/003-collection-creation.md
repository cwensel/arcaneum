# Post-Mortem: RDR-003 Collection Creation CLI

## RDR Summary

RDR-003 proposed a CLI tool for managing Qdrant collections with named vectors and
multiple embedding models. The approach centered on a lightweight, FastEmbed-powered
CLI with explicit configuration via CLI flags and an optional YAML config file, with
no reliance on environment variables. It specified commands for initialization
(`arc init`), collection lifecycle (`arc collection create/list/info/delete`),
and model management (`arc models list/download/info`).

## Implementation Status

Partially Implemented

The core collection management commands (`arc collection create/list/info/delete`)
were implemented and expanded well beyond the plan. However, several foundational
elements of the RDR were abandoned or replaced: the `arc init` command was never
built, the `arc models download/info` commands were dropped in favor of lazy model
loading, the project-local config file workflow (`arcaneum.yaml`) was never
implemented, and the recommended lightweight FastEmbed-only approach was replaced
by a dual-backend system using sentence-transformers with full PyTorch as the
primary embedding backend.

---

## Implementation vs. Plan

### What Was Implemented as Planned

- **CLI framework is Click**: The RDR recommended Click (or Typer), and Click was
  used throughout. The entry point is registered as `arc` in `pyproject.toml`
  under `[project.scripts]`.
- **Collection CRUD commands**: `arc collection create`, `list`, `info`, and
  `delete` all exist with the planned functionality (named vectors, HNSW
  configuration, confirmation prompts for delete).
- **`arc models list`**: Lists available models with alias, actual model name,
  dimensions, and description in a Rich table.
- **Configuration Pydantic models**: `config.py` implements `ModelConfig`,
  `QdrantConfig`, `CacheConfig`, `CollectionTemplate`, and `ArcaneumConfig`
  almost exactly as specified. `load_config()` and `save_config()` functions exist.
- **Rich terminal output**: All commands use `rich.console.Console` and
  `rich.table.Table` for formatted output, as planned.
- **Named vectors architecture**: Collections are created with named vectors per
  the RDR-002 architecture. `build_vectors_config()` in `cli/utils.py` maps
  model names to `VectorParams` with COSINE distance.
- **HNSW configuration**: `--hnsw-m` and `--hnsw-ef` flags on `collection create`
  with defaults of 16 and 100 respectively, matching the plan.
- **JSON output mode**: All collection commands support `--json` for machine-readable
  output, which the RDR anticipated for Claude Code plugin integration.
- **Confirmation prompt for delete**: `collection delete` requires `--confirm` flag
  or interactive confirmation, as specified.
- **Integration tests for collection creation**: `tests/test_collection_creation.py`
  covers single-vector, multi-vector, indexes, HNSW config, list, and delete.

### What Diverged from the Plan

- **FastEmbed as primary backend replaced by sentence-transformers**: The RDR
  recommended FastEmbed (ONNX Runtime, ~100MB dependencies) as the primary
  embedding library, citing 2-3x CPU speed and lightweight distribution. The
  implementation uses sentence-transformers with full PyTorch (~7GB+) as the
  primary backend for most models (stella, jina-code variants, codesage, nomic,
  minilm, gte, e5). FastEmbed is only used for BGE and Jina v2/v3 models. This
  happened because GPU acceleration (MPS/CUDA) required PyTorch, and the
  sentence-transformers ecosystem provided access to a much wider model selection.

- **`arc init` command dropped**: The RDR designed a workspace initialization
  command that would create a config file, validate Qdrant connectivity, and set
  up cache directories. This was never implemented. The `arc doctor` command
  partially fills the validation role, and `arc container start` handles service
  setup, but there is no single initialization workflow.

- **Project-local config file (`arcaneum.yaml`) never implemented**: The RDR
  designed a workflow around a project-local YAML config file with model
  definitions, collection templates, and Qdrant settings. In practice, model
  definitions live in the `EMBEDDING_MODELS` dictionary in `embeddings/client.py`,
  and Qdrant configuration uses XDG-compliant paths (`~/.config/arcaneum/`) with
  environment variable overrides. No project-local config files are used.

- **Environment variables used despite explicit prohibition**: The RDR stated
  "All configuration via CLI flags or config file (no hidden environment
  variables)." The implementation uses several environment variables:
  `ARC_QDRANT_URL`, `ARC_QDRANT_TIMEOUT`, `ARC_SSL_VERIFY`, `ARC_NO_GPU`,
  `MEILISEARCH_URL`, `MEILISEARCH_API_KEY`, `HF_HOME`,
  `SENTENCE_TRANSFORMERS_HOME`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `TOKENIZERS_PARALLELISM`. This reflects the reality that environment variables
  are necessary for library configuration, SSL handling, and GPU control.

- **Model roster completely changed**: The RDR specified 4 models: stella
  (aliased to bge-large), modernbert, bge, and jina. The implementation has 16+
  model entries. stella is the actual dunzhang/stella_en_1.5B_v5 (not a bge
  alias). modernbert was dropped from the model registry entirely. Multiple jina
  variants (code v2, code 0.5b, code 1.5b, v3, base-en), multiple bge variants
  (large, base, small), and new models (codesage, nomic-code, minilm, gte-base,
  e5-base) were added.

- **Collection create uses `--type` flag, not `--models` flag**: The RDR
  designed `collection create` to accept `--models stella,jina` for specifying
  which named vectors to include. The implementation uses `--type` (pdf, code,
  markdown) which infers a default model, with `--model` as an optional
  override. The emphasis shifted from multi-model collections to single-model
  collections with type metadata.

- **`collections/manager.py` not created**: The RDR specified a separate
  collection management utilities module. Collection logic was instead
  distributed across `cli/collections.py` (command implementations),
  `cli/utils.py` (client creation, vector config building), and
  `indexing/collection_metadata.py` (type enforcement via metadata points).

- **YAML output format dropped**: The RDR specified table/JSON/YAML output for
  `collection list`. Only table and JSON were implemented.

### What Was Added Beyond the Plan

- **GPU acceleration subsystem** (RDR-013): MPS (Apple Silicon) and CUDA
  support with dynamic batch sizing, OOM recovery with progressive batch
  reduction, GPU cache management, timeout detection for hung GPU operations,
  and CPU fallback after GPU poisoning. This is a substantial subsystem not
  anticipated by RDR-003.

- **Collection type enforcement**: A metadata system stores collection type
  (pdf/code/markdown) as a reserved point in each collection, preventing
  cross-type indexing. This includes `CollectionType`, `set_collection_metadata`,
  `get_collection_metadata`, and `validate_collection_type`.

- **Collection verify command** (`arc collection verify`): An fsck-like
  integrity checker that scans collections for incomplete chunk sets, reporting
  per-file and per-project completion status with repair hints.

- **Collection items command** (`arc collection items`): Lists all indexed
  files/repos in a collection with chunk counts, grouped by file path (for
  PDF/markdown) or git project identifier (for code).

- **Collection export/import** (RDR-017): Binary and JSONL export formats with
  glob-based filtering, path detachment/attachment for portability, and repo
  filtering for code collections.

- **MeiliSearch integration** (RDR-008, RDR-010): Full-text search alongside
  vector search, with the `arc indexes` command group and `arc search text`
  command.

- **Corpus commands** (RDR-009): Dual-index management (`arc corpus
  create/sync/parity/verify/items/delete`) that orchestrates both Qdrant
  collections and MeiliSearch indexes together.

- **Docker container management**: `arc container start/stop/restart/status/logs/reset`
  for managing Qdrant and MeiliSearch containers.

- **Diagnostics command** (`arc doctor`): System health checks for Python
  version, dependencies, Qdrant connectivity, MeiliSearch connectivity,
  embedding models, temp directory, and environment.

- **Structured error handling**: Custom exception hierarchy (`ArcaneumError`,
  `InvalidArgumentError`, `ResourceNotFoundError`) with specific exit codes
  and `HelpfulGroup` for better CLI error messages.

- **Interaction logging** (RDR-018): All collection commands log operation
  metadata for analytics.

- **XDG Base Directory compliance**: Model cache at `~/.cache/arcaneum/models`,
  data at `~/.local/share/arcaneum`, config at `~/.config/arcaneum`, with
  automatic migration from legacy `~/.arcaneum/`.

- **SSL configuration**: `ARC_SSL_VERIFY=false` support for corporate proxies
  with self-signed certificates.

- **Process priority control**: `--process-priority` and `--not-nice` flags
  for background indexing.

### What Was Planned but Not Implemented

- **`arc init` command**: Workspace initialization with config file creation,
  Qdrant validation, and cache directory setup. Partially replaced by `arc
  doctor` and `arc container start`.

- **`arc models download`**: Explicit model download command. Models are instead
  lazy-loaded on first use, which eliminated the need for explicit downloads.

- **`arc models info`**: Detailed model information display (dimensions, chunk
  size, overlap, cache status). `arc models list` provides partial coverage.

- **Project-local `arcaneum.yaml` workflow**: Config file as a first-class
  project artifact checked into git. The entire config-file-driven workflow
  (config hierarchy of flags > file > defaults) was not implemented as designed.

- **Config file management commands**: No `arc config` commands for managing
  YAML config files (though `arc config show-cache-dir` and `clear-cache` exist
  for cache management).

- **`save_config()` usage**: While the function exists in `config.py`, no CLI
  command invokes it because no init command was built.

- **Model preloading** (`preload_models` in EmbeddingClient): The RDR specified
  a method to preload models during init. The implementation has no preload
  utility; models load lazily.

- **Documentation files**: `docs/cli-reference.md`, `docs/configuration.md`,
  and `examples/arcaneum.yaml` were never created. CLI reference information
  lives in `CLAUDE.md` and `--help` output.

- **`test_config.py`**, **`test_cli.py`**, **`test_embeddings.py`**: Only
  `test_collection_creation.py` was created. Config loading, CLI flag parsing,
  and embedding client tests were not written.

---

## Drift Classification

| Category | Count | Examples |
| -------- | ----- | -------- |
| **Unvalidated assumption** | 2 | FastEmbed as sufficient embedding backend (GPU needs forced PyTorch); env vars unnecessary (libraries require them) |
| **Framework API detail** | 1 | stella model aliased to bge-large in RDR, but actual stella model (1.5B params) is a different model entirely |
| **Missing failure mode** | 1 | No GPU OOM handling planned; became a major subsystem (timeout, retry, cache clear, CPU fallback) |
| **Missing Day 2 operation** | 2 | No migration path when config approach changed; no container lifecycle management planned |
| **Deferred critical constraint** | 1 | GPU acceleration needs drove the switch from FastEmbed to sentence-transformers |
| **Over-specified code** | 3 | Full `arc init` command code, `arc models download/info` code, `EmbeddingClient.preload_models()` code -- all rewritten or dropped |
| **Under-specified architecture** | 2 | No plan for collection type enforcement; no plan for how config would actually flow from file to commands at runtime |
| **Scope underestimation** | 2 | Embedding client grew from 50-line wrapper to 1600-line GPU-aware system; model roster grew from 4 to 16+ entries |
| **Internal contradiction** | 1 | RDR said "no environment variables" but the config hierarchy listed CLI flags > config file > defaults, with no way to configure SSL, GPU, or library internals without env vars |
| **Missing cross-cutting concern** | 2 | XDG directory compliance not planned; SSL/proxy configuration not considered |

---

## RDR Quality Assessment

### What the RDR Got Right

- **Click as CLI framework**: The recommendation to use Click was validated.
  Click's group/command structure maps well to the `arc collection create`
  style commands. Typer was not needed.

- **Pydantic for configuration models**: The config model hierarchy
  (`ModelConfig`, `QdrantConfig`, `CacheConfig`, `CollectionTemplate`,
  `ArcaneumConfig`) transferred almost directly to implementation. Pydantic
  validation provides type safety and clear defaults.

- **Named vectors architecture**: The decision to use Qdrant named vectors
  (from RDR-002) was sound and carried through to implementation. The
  `build_vectors_config()` utility follows the pattern the RDR described.

- **HNSW configuration exposure**: Making `m` and `ef_construct` configurable
  via CLI flags with sensible defaults (16/100) was practical and used as
  designed.

- **Rich for terminal output**: The recommendation to use Rich for tables and
  formatted output was fully adopted and provides a good user experience.

- **Opensource project evaluation**: The decision to build from scratch rather
  than fork qdrant-cli or qdrant-loader was correct. Neither tool would have
  supported the project's eventual scope (dual-index, GPU embedding, type
  enforcement).

- **Research on model token constraints**: The chunking parameters research
  (chunk sizes, overlap percentages, char-to-token ratios) informed the
  `_build_default_models()` logic in config.py, even though specific values
  changed.

### What the RDR Missed

- **GPU acceleration as a first-class requirement**: The entire GPU subsystem
  (MPS, CUDA, OOM recovery, batch sizing, memory management) was not
  anticipated. This drove the most significant architectural change (FastEmbed
  to sentence-transformers) and represents the largest implementation effort
  not in the plan.

- **Collection type enforcement**: The need to prevent mixing PDFs, code, and
  markdown in the same collection was not identified. This required a metadata
  system using reserved points and type validation on every indexing operation.

- **Environment variable necessity**: The "no env vars" principle was
  aspirational but impractical. SSL certificate handling, GPU control, model
  cache paths, and threading configuration all require environment variables
  that cannot be replaced by CLI flags because they must be set before library
  initialization.

- **XDG Base Directory compliance**: The RDR specified `./models_cache` as the
  default cache directory. The implementation correctly follows XDG conventions
  (`~/.cache/arcaneum/models`), which is important for system integration but
  was not considered in the plan.

- **Model availability and naming**: The RDR assumed stella was an alias for
  bge-large. In practice, stella is a 1.5B-parameter model requiring
  sentence-transformers and GPU support. The RDR also listed modernbert as a
  target model, but it was never added to the actual model registry. These
  inaccuracies suggest the model research was done at a surface level without
  loading and testing the actual models.

- **Lazy model loading as the better pattern**: The RDR designed explicit
  download and preload commands. The implementation discovered that lazy
  loading on first use with `local_files_only` caching is a superior pattern:
  simpler workflow, no explicit download step, and automatic cache management.

### What the RDR Over-specified

- **Complete `arc init` implementation code**: 40+ lines of Click command code
  for a command that was never built. The implementation found that `arc doctor`
  plus `arc container start` covered the actual user needs more directly.

- **Complete `arc models download/info` implementation code**: 80+ lines of
  Click command code for commands that were replaced by lazy model loading.
  The explicit download/info workflow was unnecessary complexity.

- **`EmbeddingClient.preload_models()` method**: Specified with implementation
  code, but preloading was never needed because lazy loading was sufficient.

- **Performance benchmarks**: Specific predictions (e.g., "100 texts stella:
  <5 seconds on CPU") were speculative and became irrelevant when GPU
  acceleration changed the performance characteristics entirely.

- **ChromaDB migration path**: Detailed migration steps from ChromaDB to Qdrant
  were specified, but the project moved directly to Qdrant without a migration
  phase, making this section unused.

- **Alternative analysis depth**: Five alternatives were analyzed in detail
  (env vars, global config, config required, separate init subcommands,
  existing tools). While thorough, the actual divergences came from
  undiscovered requirements (GPU, SSL, type enforcement) rather than from
  the alternatives considered.

---

## Key Takeaways for RDR Process Improvement

1. **Validate library recommendations by running the actual code path**: The
   FastEmbed recommendation was based on spec comparison (100MB vs 7GB, 2-3x
   faster) but never validated with the target models. Loading stella
   (1.5B params) would have immediately revealed it requires
   sentence-transformers/PyTorch, not FastEmbed. A 30-minute spike loading
   each target model would have changed the entire dependency strategy.

2. **Treat "no environment variables" claims as constraints requiring explicit
   verification**: The RDR stated "no env vars" as a design principle but did
   not verify whether the chosen libraries (FastEmbed, qdrant-client,
   sentence-transformers) could actually be configured without them. Future
   RDRs should list each dependency's configuration mechanism and verify the
   "no env var" constraint is achievable before locking.

3. **Separate architectural decisions from implementation code in the RDR**:
   RDR-003 included 350+ lines of Python code for commands that were
   substantially rewritten or never built. The architectural value (Click
   groups, Pydantic models, named vectors pattern) survived; the specific
   code did not. Writing pseudocode or interface contracts instead of
   implementation code would have saved effort without losing design value.

4. **Include a "downstream requirements" section that identifies known future
   integrations**: GPU support (RDR-013), collection type enforcement, and
   MeiliSearch integration all drove architectural changes that could have
   been anticipated. The RDR listed related RDRs (004-007) but did not
   analyze whether their requirements would constrain this design. A brief
   "what will RDR-005 need from the embedding client?" analysis would have
   surfaced the GPU requirement early.

5. **Flag model-specific claims with expiration dates**: The model roster,
   dimension counts, and performance characteristics changed significantly
   between RDR authoring and implementation. Model ecosystems evolve rapidly.
   Future RDRs should mark model-specific claims as "valid as of [date],
   re-verify before implementation" to signal that these details are
   inherently unstable.

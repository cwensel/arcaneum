# Post-Mortem: RDR-013 Indexing Pipeline Performance Optimization

## RDR Summary

RDR-013 proposed a four-phase optimization strategy to address severe performance
bottlenecks in the Arcaneum indexing pipeline, where embedding generation consumed
99% of total indexing time. The approach recommended sequential phases: foundation
improvements (batch sizing, bulk upload mode, monitoring), parallelization with GPU
acceleration, advanced refinements (caching, metadata sync), and architectural changes
(streaming, distributed processing). The RDR predicted 15-37x realistic cumulative
speedup across Phases 1-3.

## Implementation Status

Partially Implemented

Phases 1 and 2 were implemented with significant divergences from the plan.
Phase 2 was substantially more complex than anticipated due to GPU memory management
challenges on Apple Silicon (MPS). Phase 3 was deferred entirely. Phase 4's streaming
architecture was implemented ahead of schedule as part of Phase 2, driven by memory
pressure discovered during GPU work. Performance profiles (`--fast`, `--turbo`) were
not implemented; instead, individual CLI flags provide equivalent control.

---

## Implementation vs. Plan

### What Was Implemented as Planned

- **CPU monitoring with psutil** (Phase 1, Step 1.4): The `CPUMonitor` class in
  `src/arcaneum/monitoring/cpu_stats.py` closely matches the RDR specification,
  including `cpu_percent`, `cpu_percent_per_core`, `num_threads`, `elapsed_time`,
  and a `get_summary()` method. Integrated into both PDF and source code pipelines.

- **Qdrant bulk mode** (Phase 1/3): `enable_bulk_mode()` and `disable_bulk_mode()`
  implemented in `src/arcaneum/indexing/qdrant_indexer.py` exactly as specified, using
  `OptimizersConfigDiff(indexing_threshold=0)` during uploads and restoring to 20000
  after. The `upload_chunks()` method accepts `bulk_mode=True` and uses `wait=False`
  during bulk uploads, matching the RDR design.

- **GPU device detection** (Phase 2, Step 2.1): The `_detect_device()` method in
  `EmbeddingClient` checks for MPS then CUDA availability, returning the appropriate
  device string, as specified in the RDR.

- **GPU acceleration for SentenceTransformers** (Phase 2, Step 2.1): Models are loaded
  with `device=self._device` (MPS/CUDA/CPU), matching the RDR's approach. FastEmbed
  models use `CoreMLExecutionProvider` when available.

- **Parallel file processing** (Phase 2, Step 2.3): Source code uses
  `ProcessPoolExecutor` via `create_process_pool()` with a module-level worker function
  `_process_file_worker()` for pickling compatibility, matching the RDR design pattern.
  Default workers are `cpu_count() // 2` as recommended for laptop responsiveness.

- **Parallel OCR processing** (Phase 2, Step 2.4): OCR uses `multiprocessing.Pool` with
  a module-level `_ocr_single_page_worker()` function, processing pages in parallel with
  memory-efficient batching and image serialization, closely matching the RDR approach.

- **Process priority control** (Phase 2, partial): The `--process-priority` flag with
  `low`/`normal`/`high` options and worker-level `os.nice(10)` implement the RDR's
  concept of background priority for indexing, though not as performance profiles.

### What Diverged from the Plan

- **Parallel embedding became sequential for GPU**: The RDR planned
  `ThreadPoolExecutor`-based parallel embedding across batches (Step 2.2). The
  implementation discovered that GPU models have internal parallelism within each batch,
  making thread-level parallelism counterproductive (it introduces lock serialization
  overhead). `embed_parallel()` processes batches sequentially for GPU, using
  `ThreadPoolExecutor` only for CPU models. This was a fundamental insight that
  invalidated the RDR's primary speedup mechanism.

- **Batch sizes exceeded plan significantly**: The RDR proposed increasing from
  100 to 200-300. The implementation uses 512 as the default (`QdrantIndexer.DEFAULT_BATCH_SIZE`),
  with dynamic GPU-aware sizing ranging from 128 to 1024 based on model dimensions and
  available GPU memory. The `_get_optimal_batch_size()` method and
  `estimate_safe_batch_size_v2()` in `memory.py` provide runtime adaptation that the
  RDR did not anticipate.

- **GPU complexity far exceeded "just add device=mps"**: The RDR described GPU as
  "Low complexity: just add `device='mps'`". The actual implementation required:
  OOM recovery with progressive batch reduction, GPU poisoning (disabling GPU after
  timeout), CPU fallback model loading, embedding validation (NaN/Inf/zero/duplicate
  detection), Metal command buffer timeout handling via daemon threads, GPU cache
  clearing strategies based on model size, and per-model `mps_max_batch` limits.
  This consumed the majority of Phase 2 effort.

- **Performance profiles replaced by individual flags**: Instead of `--fast`/`--turbo`
  profiles, the implementation provides `--no-gpu`, `--process-priority`,
  `--embedding-batch-size`, `--not-nice`, and `--no-streaming` as independent controls.
  GPU is enabled by default (the RDR proposed it off by default), which is the opposite
  of the "user-friendly default" philosophy. The rationale: GPU acceleration is the
  primary performance win, so it should be on by default.

- **Streaming architecture moved from Phase 4 to Phase 2**: The RDR placed streaming
  (generator-based pipeline) in Phase 4 as an architectural improvement. It was
  implemented during Phase 2 as `accumulate=False` mode in `embed_parallel()` with
  `on_batch_complete` callbacks, driven by the discovery that GPU memory pressure
  required immediate upload-after-embed to avoid holding all embeddings in memory.

- **gRPC not implemented**: The RDR proposed switching PDF indexing from HTTP to gRPC
  (Phase 1, Step 1.1). The codebase still uses HTTP for all Qdrant communication.
  The RDR's own analysis rated this as <0.1% improvement, so it was correctly deprioritized.

- **Progress implementation differs from plan**: The RDR proposed `tqdm` progress bars
  (Step 2.5). The implementation uses `rich.progress` components and custom inline
  progress printing with carriage return updates and ETA calculations. The functionality
  is equivalent but the mechanism differs.

### What Was Added Beyond the Plan

- **GPU OOM recovery system**: Progressive batch reduction (halving until minimum),
  GPU cache clearing with synchronization, retry with pause intervals, and helpful
  error messages with suggestions. Not anticipated by the RDR.

- **GPU poisoning and CPU fallback**: When `model.encode()` times out (GPU hang in
  Metal retry loop), the GPU is "poisoned" (disabled for the session) and a fresh
  CPU-only model is loaded. This prevents fatal SIGABRT from conflicting Metal
  command buffers.

- **Embedding validation**: `_validate_embeddings()` checks for NaN, Inf, zero
  vectors, extreme norms, identical embeddings, and low variance. Metal/MPS OOM can
  corrupt embeddings without raising Python exceptions.

- **Dynamic GPU memory estimation**: `src/arcaneum/utils/memory.py` provides
  `get_gpu_memory_info()` and `estimate_safe_batch_size_v2()` for runtime batch size
  calculation based on available GPU memory, model parameters, and device type.

- **CPU threading configuration**: `_configure_cpu_threading()` sets
  `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `TOKENIZERS_PARALLELISM` to prevent
  thread over-subscription when running in CPU mode.

- **Per-model MPS batch limits**: `mps_max_batch` in `EMBEDDING_MODELS` config
  provides model-specific maximum batch sizes for MPS to prevent system lockups
  (e.g., stella 1.5B needs batch=2, nomic-code 7B needs batch=1).

- **Text pre-truncation**: Pre-truncates texts exceeding safe character limits before
  embedding to prevent tokenizer OOM on generated code with high token density.

- **Memory cleanup patterns**: Explicit `gc.collect()`, `_clear_gpu_cache()`, and
  `del` patterns between files and batches to prevent memory accumulation.

- **Shared embedding client**: Single `EmbeddingClient` shared across PDF workers
  instead of per-worker instances, avoiding duplicate 2GB+ model loading.

- **Pipeline profiler**: `src/arcaneum/monitoring/pipeline_profiler.py` provides
  per-stage timing breakdown (file processing, embedding, upload) with `--profile` flag.

- **Model cache**: `src/arcaneum/embeddings/model_cache.py` provides process-level
  model persistence, saving 7-8 seconds on repeated CLI invocations.

- **Memory-aware worker limits**: `calculate_safe_workers()` in `memory.py` adjusts
  worker counts based on available system memory.

### What Was Planned but Not Implemented

- **Performance profiles (`--fast`, `--turbo`)**: Replaced by individual CLI flags.
  The concept of bundled profiles was not implemented.

- **Embedding cache** (Phase 3, Step 3.1): No `src/arcaneum/embeddings/cache.py`
  exists. The SQLite-based cache for avoiding re-embedding duplicate content was
  deferred.

- **Metadata sync optimization** (Phase 3, Step 3.3): `git_metadata_sync.py` still
  uses `limit=100` for scroll queries, not the proposed `limit=1000`. No persistent
  file-based cache for indexed projects was implemented.

- **Benchmark suite** (Phase 3, Step 3.4): No `tests/benchmarks/` directory exists.
  Performance is validated manually.

- **gRPC for Qdrant communication** (Phase 1, Step 1.1): HTTP is still used. Correctly
  deprioritized given the <0.1% improvement estimate.

- **Smart file filtering module** (Phase 4, Step 4.3): No standalone
  `src/arcaneum/indexing/filters.py` exists. However, equivalent functionality exists
  in `src/arcaneum/cli/sync.py` with `DEFAULT_EXCLUDE_FILE_PATTERNS` and
  `DEFAULT_EXCLUDE_DIRS`, including `node_modules`, `*.min.js`, generated protobuf
  files, and build directories.

- **Distributed processing with Ray** (Phase 4, Step 4.2): Not implemented, as
  expected for an optional future enhancement.

---

## Drift Classification

| Category | Count | Examples |
| -------- | ----- | -------- |
| **Unvalidated assumption** | 2 | GPU "low complexity" claim; ThreadPoolExecutor speedup for GPU embedding |
| **Framework API detail** | 1 | tqdm proposed but rich.progress used (project already depended on rich) |
| **Missing failure mode** | 3 | MPS OOM corruption without exceptions; GPU timeout/hang in Metal retry loop; text pre-truncation for high-density code |
| **Missing Day 2 operation** | 0 | |
| **Deferred critical constraint** | 1 | GPU memory management for Apple Silicon unified memory was out of scope |
| **Over-specified code** | 2 | ThreadPoolExecutor embedding code; EmbeddingCache SQLite implementation for deferred Phase 3 |
| **Under-specified architecture** | 2 | No design for GPU failure recovery; no design for streaming embed-upload pipeline |
| **Scope underestimation** | 1 | GPU acceleration grew from "add device=mps" to a major subsystem (OOM recovery, poisoning, validation, timeout, CPU fallback) |
| **Internal contradiction** | 1 | RDR proposed GPU off by default for "user-friendly" defaults, but GPU is the primary speedup; implementation correctly defaulted GPU to on |
| **Missing cross-cutting concern** | 1 | Memory management (gc.collect patterns, GPU cache clearing, shared model instances) not addressed in original plan |

### Drift Category Definitions

- **Unvalidated assumption** -- a claim presented as fact
  but never verified by spike/POC
- **Framework API detail** -- method signatures, interface
  contracts, or config syntax wrong
- **Missing failure mode** -- what breaks, what fails
  silently, recovery path not considered
- **Missing Day 2 operation** -- bootstrap, CI/CD,
  removal, rollback, migration not planned
- **Deferred critical constraint** -- downstream use case
  that validates the approach was out of scope
- **Over-specified code** -- implementation code that was
  substantially rewritten
- **Under-specified architecture** -- architectural
  decision that should have been made but wasn't
- **Scope underestimation** -- sub-feature that grew into
  its own major effort
- **Internal contradiction** -- research findings or stated
  principles conflicting with the proposal
- **Missing cross-cutting concern** -- versioning,
  licensing, config cache, deployment model, etc.

---

## RDR Quality Assessment

### What the RDR Got Right

- **Bottleneck identification was accurate**: The analysis that embedding generation
  consumed 99% of indexing time and that parallelization + GPU would provide the
  primary speedup was correct. The priority ordering of bottlenecks was sound.

- **Phased approach was valuable**: The phase structure allowed Phases 1-2 to be
  implemented independently and validated, with Phase 3 correctly deferred when
  Phases 1-2 delivered sufficient improvement.

- **GPU model compatibility matrix was verified and useful**: The testing of stella,
  jina-code, bge-*, and jina-v3 with MPS/CoreML was accurate and directly informed
  implementation decisions. The recommendation to use stella for PDFs and jina-code
  for source code as GPU-compatible defaults was correct.

- **Qdrant bulk mode design was directly implementable**: The `enable_bulk_mode()`
  and `disable_bulk_mode()` pattern with `indexing_threshold=0` was implemented
  almost verbatim from the RDR specification.

- **CPU monitoring diagnosis was correct**: The observation that htop showed low
  process CPU (14%) while ONNX Runtime threads consumed actual cycles, and the
  psutil-based solution, was accurate and implemented as designed.

- **Alternative rejection rationale was sound**: Ray Data (too complex for single
  machine), GPU-only (limits portability), and Rust rewrite (development cost)
  were all correctly rejected with clear reasoning.

### What the RDR Missed

- **GPU failure modes on Apple Silicon**: The RDR described GPU as "Low complexity:
  just add `device='mps'`" with an estimated 1.5-3x speedup. In reality, MPS
  introduces: silent embedding corruption on OOM, infinite hang in Metal retry loops,
  SIGABRT from concurrent command buffer access, system-wide lockups from unified
  memory exhaustion, and buffer allocation failures for large batch outputs. Each
  required its own mitigation strategy.

- **Memory management as a cross-cutting concern**: The RDR mentioned "Increased memory
  usage" as a negative consequence but did not design for it. The implementation
  required systematic `gc.collect()` patterns, GPU cache clearing strategies keyed to
  model size, shared embedding client patterns, text pre-truncation, and streaming
  upload to prevent OOM across the pipeline.

- **GPU internal parallelism invalidates thread-level parallelism**: The core assumption
  of Phase 2 -- that ThreadPoolExecutor across embedding batches would yield 2-4x
  speedup -- was wrong for GPU models. GPU hardware already parallelizes within each
  batch; adding Python threads introduces lock contention without benefit. This was
  only discovered during implementation profiling.

- **Streaming as a memory necessity, not an architectural luxury**: The RDR placed
  streaming in Phase 4 as an architectural improvement for "avoiding holding all
  chunks in memory." In practice, streaming was required in Phase 2 because GPU memory
  pressure made it impossible to accumulate all embeddings before upload.

### What the RDR Over-specified

- **Complete code samples for deferred phases**: The RDR included full implementation
  code for Phase 3 (EmbeddingCache SQLite class, metadata sync with file cache) and
  Phase 4 (streaming architecture, smart file filtering). Phases 3-4 were deferred,
  and when streaming was implemented (ahead of Phase 4), the actual design differed
  substantially from the RDR code. The 200+ lines of Phase 3-4 code provided no value.

- **Speculative performance estimates per profile**: The RDR provided detailed
  speedup estimates for three performance profiles across four test scenarios
  (e.g., "Fast profile: 6-12x with 87% CPU + GPU"). These numbers were theoretical
  and the profiles were never implemented. The actual speedup depends on model,
  hardware, and workload in ways the estimates could not capture.

- **ThreadPoolExecutor embedding parallelism code**: The `embed_parallel()` method
  in the RDR used ThreadPoolExecutor with `as_completed()` -- this was implemented
  but then discovered to be counterproductive for GPU models and effectively abandoned
  for the primary use case.

---

## Key Takeaways for RDR Process Improvement

1. **Require a spike for any claim rated "Low complexity" that touches hardware
   acceleration**: The RDR stated GPU was "Low complexity: just add `device='mps'`"
   without a spike to validate failure modes. A 30-minute spike encoding 1000 texts
   on MPS would have revealed OOM corruption, timeouts, and memory pressure, changing
   the Phase 2 estimate from days to weeks. Any RDR claiming hardware integration is
   simple should require a proof-of-concept before locking.

2. **Do not write implementation code for phases beyond the current scope**: The RDR
   included complete Python code for Phase 3 (EmbeddingCache) and Phase 4 (streaming,
   filters) -- over 200 lines that were never used or were substantially rewritten.
   RDRs should describe the architecture and interfaces for deferred phases but reserve
   implementation code for the phases being implemented now.

3. **Treat memory management as a first-class design concern for GPU workloads**:
   The RDR listed "Increased memory usage" as a risk with "Add memory monitoring" as
   mitigation, but did not design memory management into the pipeline. Future RDRs
   involving GPU or large-model workloads should include a dedicated memory management
   section covering: cache clearing strategy, object lifetime, peak memory estimation,
   and OOM recovery paths.

4. **Validate parallelism assumptions against the specific execution model**: The RDR
   assumed ThreadPoolExecutor across embedding batches would yield 2-4x speedup based
   on general CPU parallelism principles. GPU execution models are fundamentally
   different -- internal parallelism within batches means external thread parallelism
   adds overhead, not speedup. RDRs should specify which parallelism model applies
   (CPU thread, CPU process, GPU kernel, or GPU stream) and validate the assumption
   matches the execution target.

5. **Prefer composable flags over bundled profiles when the parameter space is
   uncertain**: The RDR designed three bundled profiles (default/fast/turbo) that
   assumed specific optimal configurations. The implementation discovered that optimal
   settings depend on model size, GPU memory, file count, and chunk density -- too many
   variables for three fixed profiles. Independent flags (`--no-gpu`,
   `--embedding-batch-size`, `--process-priority`) proved more practical. RDRs should
   prefer composable controls when the optimization space is not well understood.

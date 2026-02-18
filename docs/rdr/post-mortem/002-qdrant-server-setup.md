# Post-Mortem: RDR-002 Qdrant Server Setup with Client-side Embeddings

## RDR Summary

RDR-002 proposed a standardized Qdrant server deployment using Docker Compose with
client-side embedding generation via FastEmbed. The core architectural decision was
to use named vectors (multiple embedding models per collection) instead of separate
collections per model, with volume persistence, a management shell script, and
Python modules for collection initialization and embedding generation.

## Implementation Status

Partially Implemented

The foundational decisions (Docker Compose, client-side embeddings, named vectors,
volume persistence) were all implemented and remain in production. However, almost
every concrete artifact diverged significantly from the plan: the Docker Compose
configuration was restructured, the embedding client was rewritten to support two
backends with GPU acceleration, the collection architecture shifted from fixed
collections to user-defined ones, and model caching moved to XDG-compliant paths.
The static `collections/init.py` module exists but is effectively dead code,
superseded by the dynamic CLI-driven collection creation.

---

## Implementation vs. Plan

### What Was Implemented as Planned

- **Docker Compose deployment**: Qdrant runs via Docker Compose with the
  `qdrant/qdrant` image, `qdrant-arcaneum` container name, ports 6333/6334,
  `unless-stopped` restart policy, and identical resource limits (4G memory/2 CPU
  limit, 2G/1 CPU reservation).
- **Client-side embedding generation**: All embedding happens in the Python
  application, not on the Qdrant server. This was the RDR's critical finding and
  it proved correct.
- **Named vectors architecture**: Collections support multiple named vectors per
  collection. The `VectorParams` with cosine distance is used as specified.
- **Management shell script**: `scripts/qdrant-manage.sh` exists and closely
  matches the RDR's proposed script, including start/stop/restart/logs/status
  commands with health check verification.
- **HNSW configuration**: Default `m=16`, `ef_construct=100` values are used
  exactly as planned.
- **On-disk payload**: `on_disk_payload=True` is the default for collections,
  as recommended.
- **Payload indexes**: Keyword indexes on `programming_language`,
  `git_project_root`, and other fields are created during collection setup.
- **`.gitignore` entries**: `qdrant_storage/`, `qdrant_snapshots/`, and
  `models_cache/` are all in `.gitignore` as planned.
- **`collections/init.py` module**: Exists at `src/arcaneum/collections/init.py`
  with the exact structure proposed, including `COLLECTION_CONFIGS` dictionary
  and `init_collections()` function.

### What Diverged from the Plan

- **Docker Compose file location**: The RDR planned `docker-compose.yml` at the
  repository root. The implementation placed it at `deploy/docker-compose.yml`.
  This was a project organization decision to separate deployment artifacts from
  source code.

- **Qdrant version**: The RDR pinned `qdrant/qdrant:v1.15.4`. The implementation
  uses `v1.16.2`. Normal version drift from ongoing development.

- **Docker Compose `version` key**: The RDR specified `version: '3.8'`. The
  implementation correctly omits the deprecated `version` field, following modern
  Docker Compose conventions.

- **Volume strategy**: The RDR planned bind mounts (`./qdrant_storage:/qdrant/storage`).
  The implementation uses Docker named volumes
  (`qdrant-arcaneum-storage:/qdrant/storage`) with the `local` driver. Named
  volumes are more portable, avoid permission issues, and are managed by the Docker
  daemon rather than tied to a specific host directory.

- **Models cache not mounted on Qdrant container**: The RDR planned a
  `./models_cache:/models` volume and `SENTENCE_TRANSFORMERS_HOME=/models`
  environment variable on the Qdrant container. Since Qdrant does not generate
  embeddings (the RDR's own critical finding), mounting a model cache on the Qdrant
  container serves no purpose. The implementation correctly sets
  `SENTENCE_TRANSFORMERS_HOME` in the Python application code, pointing to
  `~/.cache/arcaneum/models`.

- **Qdrant storage optimization**: The RDR did not plan any Qdrant storage
  optimizer configuration. The implementation adds six `QDRANT__STORAGE__*`
  environment variables for disk-biased, low-memory operation (WAL capacity,
  segment numbers, flush intervals, indexing thresholds). This was needed for
  real-world memory constraints.

- **Embedding backend**: The RDR planned FastEmbed exclusively. The implementation
  uses a dual-backend architecture: FastEmbed for ONNX-based models (BGE family)
  and SentenceTransformers for larger models (stella, jina-code, nomic-code). This
  was driven by the discovery that many high-quality models (especially code-specific
  and large models) are not available in FastEmbed's ONNX format.

- **Model catalog expansion**: The RDR specified 4 models (stella 1024D, modernbert
  1024D, bge 1024D, jina 768D). The implementation supports 16+ models across
  multiple families (jina-code variants, codesage, nomic-code, bge variants, minilm,
  gte-base, e5-base, jina-v3). The `modernbert` model from the RDR was never
  implemented.

- **Collection architecture shift**: The RDR planned fixed collections (`source-code`,
  `pdf-docs`, `markdown-docs`) with predetermined named vectors. The implementation
  uses user-defined collection names via `arc collection create MyPDFs --type pdf`
  with dynamic model selection. The static `COLLECTION_CONFIGS` in
  `collections/init.py` is effectively dead code.

- **XDG Base Directory compliance**: The RDR planned `./models_cache/` as a
  relative directory. The implementation follows XDG Base Directory specification:
  models in `~/.cache/arcaneum/models`, data in `~/.local/share/arcaneum/`,
  config in `~/.config/arcaneum/`. Includes legacy `~/.arcaneum/` migration.

- **Multi-model default pattern**: The RDR strongly advocated multi-model named
  vectors (4 models per collection). In practice, collections typically use a single
  model per collection (e.g., `--type code` defaults to `jina-code-0.5b`). Multi-model
  is supported but not the default pattern.

### What Was Added Beyond the Plan

- **CLI container management**: Full `arc container start/stop/restart/status/logs/reset`
  commands in Python (`src/arcaneum/cli/docker.py`), superseding the shell script for
  daily use. Manages both Qdrant and MeiliSearch.
- **GPU acceleration with OOM recovery**: Extensive MPS (Apple Silicon) and CUDA
  support with adaptive batch sizing, timeout-based GPU poisoning, CPU fallback,
  embedding validation, and multi-layer OOM recovery. This became a major effort
  tracked across multiple issues.
- **Backup and restore scripts**: `scripts/qdrant-backup.sh` and
  `scripts/qdrant-restore.sh` for snapshot-based collection backup, not planned in
  the RDR.
- **MeiliSearch integration**: The Docker Compose file includes MeiliSearch
  (`getmeili/meilisearch:v1.12`) as a second service (RDR-008), with auto-generated
  API keys stored at `~/.config/arcaneum/meilisearch.key`.
- **Global model cache**: `src/arcaneum/embeddings/model_cache.py` provides
  thread-safe, process-lifetime caching of loaded models, reducing model reload
  overhead.
- **SSL configuration handling**: Environment-driven SSL verification control
  (`ARC_SSL_VERIFY=false`) for corporate proxy environments.
- **Collection export/import**: Full export/import workflow with binary and JSONL
  formats, path detach/attach for portability, and glob-based filtering (RDR-017).
- **Collection verification**: `arc collection verify` and `arc corpus verify` for
  fsck-like integrity checking of chunk completeness.
- **Dual-index corpus system**: `arc corpus` commands that manage paired
  Qdrant + MeiliSearch indexes with parity checking and backfill.
- **CPU threading optimization**: Configurable OMP/MKL thread counts, tokenizer
  parallelism control, and `--cpu-workers` flag to prevent thread over-subscription.

### What Was Planned but Not Implemented

- **`docs/qdrant-setup.md`**: Step 5 of the implementation plan called for a usage
  documentation file. This was never created; CLI help text and `CLAUDE.md` serve
  as the primary documentation instead.
- **`modernbert` embedding model**: Listed as a supported model in the RDR but
  never added to `EMBEDDING_MODELS`. The `collections/init.py` static config still
  references it, but this module is dead code.
- **Multi-model per collection as default**: The RDR envisioned every collection
  having 3-4 named vectors by default (stella, modernbert, bge, jina). In practice,
  collections are created with 1 model. The multi-model approach proved
  operationally complex (4x embedding time, 4x storage) without demonstrated
  retrieval benefit.

---

## Drift Classification

| Category | Count | Examples |
| -------- | ----- | -------- |
| **Unvalidated assumption** | 2 | Multi-model default would improve retrieval; modernbert availability in FastEmbed |
| **Framework API detail** | 1 | Docker Compose `version: '3.8'` deprecated |
| **Missing failure mode** | 1 | No GPU OOM handling planned; became major implementation effort |
| **Missing Day 2 operation** | 1 | No backup/restore scripts planned |
| **Deferred critical constraint** | 0 | |
| **Over-specified code** | 2 | `collections/init.py` static configs became dead code; multi-model collection examples never matched reality |
| **Under-specified architecture** | 2 | No model backend strategy (FastEmbed vs SentenceTransformers); no directory/path management strategy |
| **Scope underestimation** | 2 | Embedding client grew from ~50 lines to ~1600 lines; container management grew from shell script to full CLI |
| **Internal contradiction** | 1 | RDR's critical finding was "Qdrant does NOT generate embeddings" but then mounted models_cache on the Qdrant container |
| **Missing cross-cutting concern** | 2 | XDG directory compliance; SSL certificate handling for corporate environments |

### Drift Category Definitions

- **Unvalidated assumption** -- a claim presented as fact but never verified by
  spike/POC
- **Framework API detail** -- method signatures, interface contracts, or config
  syntax wrong
- **Missing failure mode** -- what breaks, what fails silently, recovery path not
  considered
- **Missing Day 2 operation** -- bootstrap, CI/CD, removal, rollback, migration
  not planned
- **Deferred critical constraint** -- downstream use case that validates the
  approach was out of scope
- **Over-specified code** -- implementation code that was substantially rewritten
- **Under-specified architecture** -- architectural decision that should have been
  made but was not
- **Scope underestimation** -- sub-feature that grew into its own major effort
- **Internal contradiction** -- research findings or stated principles conflicting
  with the proposal
- **Missing cross-cutting concern** -- versioning, licensing, config cache,
  deployment model, etc.

---

## RDR Quality Assessment

### What the RDR Got Right

- **Client-side embedding as the core decision**: The research finding that Qdrant
  does not generate embeddings was the most valuable insight. It correctly shaped
  the entire architecture and has held up through all subsequent development.
- **Named vectors over multiple collections**: This recommendation proved correct.
  Named vectors are the active pattern, even though single-model is more common
  than multi-model.
- **Docker Compose simplicity**: The single-service Docker Compose approach was
  correct for local development and single-server production. No complex
  orchestration was needed.
- **Resource limits**: The 4G memory limit and 2G reservation were practical values
  that carried through to production unchanged.
- **Volume persistence strategy**: The principle of persistent volumes for Qdrant
  storage and snapshots was correct, even though the implementation changed from
  bind mounts to named volumes.
- **HNSW configuration**: The `m=16`, `ef_construct=100` defaults were sensible and
  survived unchanged.
- **Alternatives analysis**: The rejection of server-side embeddings (Alternative 2)
  and multiple collections per model (Alternative 1) were both correct decisions
  that saved significant rework.

### What the RDR Missed

- **Embedding model diversity problem**: The RDR assumed all models would be
  available via FastEmbed (ONNX). In reality, many high-quality models (stella,
  jina-code variants, nomic-code) require SentenceTransformers with PyTorch. The
  dual-backend architecture was a necessary surprise.
- **GPU acceleration as a requirement**: The RDR did not mention GPU at all.
  Embedding large codebases (100K+ chunks) on CPU is impractical. GPU support
  (especially Apple Silicon MPS) became a major feature with extensive OOM handling
  that spans hundreds of lines of code.
- **Model cache location strategy**: The RDR treated model caching as a container
  volume concern. Since embeddings are client-side, model cache management is
  entirely an application concern. XDG Base Directory compliance was needed for
  a proper CLI tool.
- **Collection naming and lifecycle**: The RDR assumed fixed, pre-defined collection
  names. Real users need to create, name, type, delete, verify, export, and import
  collections dynamically.
- **Qdrant storage tuning**: The RDR did not anticipate the need for disk-biased
  storage optimization on memory-constrained machines. The six `QDRANT__STORAGE__*`
  environment variables in the actual deployment were discovered through operational
  experience.

### What the RDR Over-specified

- **Multi-model code samples**: The RDR included extensive code samples for
  4-model named vector configurations, multi-model indexing, and model-switching
  queries. None of this code was used as written; the real patterns are simpler
  (single-model per collection).
- **Static `COLLECTION_CONFIGS`**: The RDR specified a fixed dictionary of
  collection configurations with hardcoded model names. This became dead code
  immediately when dynamic CLI-driven collection creation was implemented.
- **`EmbeddingClient` as a thin wrapper**: The RDR's proposed embedding client was
  ~40 lines with simple model initialization and batch embedding. The actual client
  is ~1600 lines handling dual backends, GPU detection, OOM recovery, batch
  optimization, embedding validation, CPU threading, and model cache management.
  The RDR's code sample provided no architectural insight for the actual
  implementation.
- **Performance validation targets**: The RDR specified targets like "query latency
  < 50ms for 10K vectors" and "model loading < 2 seconds (cached)." These numbers
  were never formally validated and do not reflect the actual performance concerns
  (GPU OOM, batch throughput, memory fragmentation).
- **`docs/qdrant-setup.md`**: Planned but never needed; CLI help and CLAUDE.md
  proved sufficient.

---

## Key Takeaways for RDR Process Improvement

1. **Run a spike for each embedding model before listing it as supported**: The RDR
   listed modernbert as a supported model without verifying it was available in
   FastEmbed. A 15-minute spike loading each model via the planned library would
   have revealed the FastEmbed coverage gap and motivated the dual-backend
   architecture earlier.

2. **Separate container configuration from application architecture**: The RDR
   conflated Qdrant server deployment (Docker Compose) with client-side application
   concerns (model caching, embedding generation). The models_cache volume mounted
   on the Qdrant container was an internal contradiction with the RDR's own finding
   that embedding is client-side. Future RDRs should draw a clear boundary between
   server infrastructure and application code.

3. **Prototype the CLI interface before specifying internal modules**: The RDR
   specified `collections/init.py` with hardcoded collection names and model lists,
   which became dead code once the CLI supported dynamic collection creation. Writing
   the CLI commands first (even as pseudocode) would have revealed that static
   configuration does not match how users interact with the tool.

4. **Include GPU/accelerator strategy for any compute-intensive workload**: The
   RDR's "client-side embedding" decision implied a significant compute workload
   but did not address how to make that workload efficient. For any RDR involving
   ML model inference, the plan should explicitly address device selection, memory
   management, and failure modes (OOM, timeouts, hardware-specific quirks).

5. **Plan code samples at the interface level, not the implementation level**: The
   RDR's code samples for collection creation, embedding generation, and querying
   were all rewritten. Code samples in RDRs are most valuable when they demonstrate
   the API contract (function signatures, input/output shapes, error handling) rather
   than full implementations that will be discarded.

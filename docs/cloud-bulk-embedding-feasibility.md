# Cloud bulk embedding feasibility

Status: **open analysis** (desk research, 2026-09-27). No measurements yet.

Could bulk indexing offload embedding to ephemeral AWS compute, managed by
[clusterless](https://github.com/ClusterlessHQ/clusterless), to make large
syncs faster and more stable? Short answer: yes, but not through Lambda or
Fargate. AWS Batch on ECS Managed Instances is the AWS-managed GPU option.
Faster embedding may not be the dominant win: MeiliSearch indexing is the
larger local CPU cost, and it cannot usefully be offloaded (see
[Full-text indexing](#full-text-indexing-meilisearch)).

Evidence labels: **Documented** (official docs or source reading), **Measured
elsewhere** (third-party benchmark, not our models or hardware),
**Extrapolated** (derived by scaling), **Assumed** (needs validation).

Tracking: kata umbrella `arcaneum#rrky`. Clusterless capability is tracked
separately so it can land independently: `clusterless#m1kd` (GPU + Spot) and
`clusterless#2cy0` (fan-out).

## Current state

- Embedding is local: fastembed (ONNX, CoreML) and sentence-transformers
  (MPS/CUDA). There is no remote backend. Every call goes through
  `EmbeddingClient.embed()` (`src/arcaneum/embeddings/client.py:1411`),
  one call per file per model (`src/arcaneum/cli/sync.py:3892`).
- Much of the stability machinery exists because of Metal: RDR-020, the
  RDR-022 spawned accelerator worker, the sticky `_gpu_poisoned` fallback,
  and per-model `mps_max_batch` caps.
- Dedup is file-level only (`compute_file_hash`, `sync.py:1693`). A
  chunk-level embedding cache was deferred in RDR-013 Phase 3.
- Qdrant bulk mode (`indexing_threshold` 0/20000) exists in
  `qdrant_indexer.py:321-360`, but `DualIndexer` does not use it.
- Measured baseline (RDR-024): building the 347,881-document PapersFast
  MeiliSearch index cost 33,591 s (9.3 h) of server-side merge time. Merge
  cost scales with index size, not batch size. Offloading embedding does not
  touch this cost.

## AWS compute options

| Service | GPU | Evidence |
| --- | --- | --- |
| Fargate | No | **Documented**: "Use Amazon EC2 for GPU workloads, which are not supported on AWS Fargate today." |
| Lambda, including Lambda Managed Instances | No | **Documented** (secondary sources): no GPU resource type. |
| SageMaker Serverless Inference | No | **Documented**: GPUs listed under unsupported features; 6 GB memory cap. |
| ECS Managed Instances | Yes | **Documented**: AWS-managed instances with NVIDIA drivers preinstalled; g4dn, g5, p3, p4d, g6f listed as a subset. |
| AWS Batch on ECS Managed Instances (2026-08-25) | Yes | **Documented**: GPU; On-Demand, Spot, or reserved capacity; no `minvCpus` warm pool. |
| Bedrock embeddings (Titan v2, Cohere v3/v4) | Managed | **Documented**: batch jobs over S3 JSONL. Different models, and query-time search would depend on AWS. Rejected for now. |

ECS Managed Instances adds a management fee on top of the EC2 price. AWS cut
it 35% for G-series on 2026-07-01; the absolute fee was not found.

## Proposed shape

1. **arc**: chunk locally, drop chunks already embedded (chunk-hash cache),
   write shards (parquet or JSONL, zstd) and a source manifest for each lot
   to S3.
2. **Trigger**: `cls arcs exec --lot`, or an `ArcNotifyEvent` sent with
   EventBridge `PutEvents`, starts the arc directly and skips S3 boundary
   polling (clusterless `ArcExec.java:157-188`).
3. **Batch arc** (ECS Managed Instances, GPU or CPU; Fargate for a CPU
   prototype): arcaneum's own embedding code in batch mode reads shards and
   writes float32 vector shards keyed by chunk hash, idempotently per shard.
   It writes the `complete` sink manifest last.
4. **arc**: poll the sink manifest at its deterministic S3 path, download
   vectors, and bulk-load Qdrant with indexing deferred.

The worker runs arcaneum's code instead of TEI or Infinity, so vectors match
local query embeddings. Fastembed models run on `onnxruntime-gpu` with the
same ONNX graph; sentence-transformers models run on CUDA. `batch_scheduler.py`
already buckets by length, which gives most of the dynamic-batching benefit
TEI provides. Parity is checked by cosine agreement on a sample
(target > 0.999, **Assumed** achievable).

## Using the cold start

Expect ~5–15 minutes before first compute. AWS's scale-to-zero GPU example
measured ~13 minutes from submit to first result (**Documented**).

- **Pipeline lots.** Publish a lot every N chunks as chunking proceeds. The
  first lot absorbs instance launch; later lots land on warm capacity. This
  fits clusterless's lot model without protocol changes.
- **Ship only the delta**, using a chunk-hash cache keyed by (chunk hash,
  model, model revision, preprocessing).
- **Index MeiliSearch immediately**, since it needs no vectors. This
  requires a third manifest state ("text indexed, vectors pending").
  Otherwise the manifest-after-durable invariant (`sync.py:284-305`,
  RDR-009/024) breaks and `--parity` misclassifies in-flight files.
- **Prepare Qdrant**: create the collection with `indexing_threshold=0`,
  upload with `wait=False` in parallel, and rebuild once at the end.
- **Shrink startup**: bake weights into the image and use SOCI lazy loading
  (a 1.3 GB image pull went from 20 s to 2.8 s, **Documented**). Count tokens
  locally so shards are sized by token budget, not characters.

## CPU vs GPU

Assumptions: 1M chunks × 512 tokens (512M tokens), roughly the size of the
`Claude` collection (1.05M points), with Spot prices in us-east-1. Throughput
figures are **Extrapolated** from third-party anchors, e.g. BGE-M3 at ~277
texts/s on L40S (**Measured elsewhere**). Estimates for decoder-based models
(0.6B–7B) vary widely between sources; FLOP-based sanity checks suggest the
lower published figures are pessimistic.

| Model class (registry) | CPU, c7i.8xlarge (~$0.53/h) | GPU | Verdict |
| --- | --- | --- | --- |
| Small (bge-small, minilm) | ~2 h, ~$1 | L4 g6.xlarge: ~0.5 h, ~$0.20 | Either; CPU is a good fallback |
| Base (jina-code, bge-base) | ~4 h, ~$2 | L4: ~1 h, ~$0.40 | GPU |
| 0.6B (qwen3-embed) | a day or more | L4: ~4–9 h, ~$2–3 | GPU |
| 1.5B (jina-code-1.5b) | impractical | L40S g6e.xlarge: ~10–28 h | GPU, fan-out |
| 7B (nomic-code) | impractical | L40S: ~35–84 h | GPU, fan-out; T4 lacks memory |

Takeaways:

- At this scale, cost is single-digit dollars except for the largest models.
  The decision is wall-clock.
- Fan-out across instances (`clusterless#2cy0`) is the lever for large
  models.
- Keep a CPU path. GPU Spot capacity is less predictable than CPU Spot
  (**Documented**), and new accounts may have zero G-instance quota
  (**Assumed**).

Spot prices, us-east-1 (**Documented** unless noted): g4dn.xlarge $0.265/h,
g5.xlarge $0.442/h, g6.xlarge ~$0.32/h (**Extrapolated**), g6e.xlarge
~$0.84/h (**Extrapolated**), c7i.8xlarge $0.526/h, c7g.16xlarge $0.706/h.

## Clusterless gaps

Tracked in `clusterless#m1kd` and `clusterless#2cy0`; details are in those
issues.

- Only Fargate compute environments are built. `ComputeResource.computeType`
  and `useSpot` are `@JsonIgnore`, so deploy JSON cannot set them.
- There is no GPU field in `BatchRuntimeProps`.
- CDK 2.271.0 supports Managed Instances only through the L1
  `CfnComputeEnvironment` (`ecsSettings.managedInstancesProvider`).
- One Batch job runs per lot; there are no array jobs or Map fan-out.
- `BatchResultHandler` throws on partial-only results, so workloads must be
  idempotent per shard for Spot retries.
- Idle cost is already near zero: public subnets, no NAT gateway, and Batch
  scales to zero.

## Prior art

- **Content-hash embedding cache**: Cursor (server-side, keyed by chunk
  hash), LangChain `CacheBackedEmbeddings`, LlamaIndex `IngestionPipeline`.
- **Merkle-tree delta sync**: Cursor, and Bloop, which cold-indexed a 9 GB
  repo in ~4m20s and re-synced a 200-commit PR in under 15 s.
- **S3 JSONL/parquet as the batch transport**: OpenAI Batch API, Bedrock
  batch inference, Cohere Embed Jobs, Ray Data.
- **Separate CPU prep from GPU encode**: TEI and Infinity with dynamic
  batching; Ray Data `map_batches`.
- **Defer vector indexing during bulk load**: Qdrant bulk upload guidance.
- **Bounded per-worker concurrency with autoscaling to zero**: Modal's
  30M-review example reached 575k tokens/s on L40S, scaling down after 5
  idle minutes.

## Full-text indexing (MeiliSearch)

MeiliSearch indexing is the larger day-to-day CPU cost. Offloading
embedding does not touch it. Tracking: kata umbrella `arcaneum#9gv0`. The
batch-resolvable set carries label `batch:meili-indexing`.

### Current cost

Live read-only check on 2026-09-27, v1.12.8 (**Documented**):

- 13 indexes, 25 GB. `Claude` (656k docs) spends 100–461 s per add of at
  most 300 docs. `PapersFast` (348k docs) spends 95–126 s per add of 39–154
  docs. `DevRef` (101k docs) spends 4–10 s.
- Per-task cost grows with index size, not batch size (RDR-024: 0.7 s at 0
  docs, 20.3 s at 300k). About 40% of 4,007 historical tasks carried fewer
  than 50 docs, and each paid that cost.
- Sync submits one add per file (`sync.py:4013`). RDR-024 cross-file
  batching is Draft and not implemented. Recent batches include long runs of
  single-task deletions.
- Settings (`src/arcaneum/fulltext/indexes.py`) leave `facetSearch` and
  `prefixSearch` on and configure `sortableAttributes`. `arc search text`
  uses filters, highlighting, and typo tolerance, but no facet search, sort,
  or search-as-you-type.
- `meilisearch-tuning.md` already fixed the Docker VM page-cache bottleneck
  and rejected `proximityPrecision: byAttribute` for phrase-ranking quality.

### Levers, in order

1. **Upgrade** v1.12.8 to current stable (v1.54.0, 2026-09-21), per the
   release notes (**Documented**):
   - v1.32: parallel payload extraction ("7x speedup on a four-million-
     document insertion using four CPUs").
   - v1.35: multithreaded facet and prefix post-processing.
   - v1.43: faster facet-search indexing.
   - v1.44: lower prefix memory.
   - v1.45–1.47: new settings indexer.

   Facet and prefix post-processing is a plausible cause of per-task cost
   growing with index size (**Assumed**; v1.12 has no per-step timings).
   The DB is version-bound. Migration is either in-place `--upgrade-db`
   (dumpless upgrade, stabilized in v1.51) or `arc container backup`
   followed by restore, which reindexes once. `arcaneum#wk39`.
2. **Trim unused features**: `facetSearch: false` (keep facetDistribution
   for `mpk8`), granular filterable attributes (v1.14+), drop unused
   sortable attributes, and evaluate `prefixSearch: disabled` against
   identifier queries. Keep `proximityPrecision: byWord`. Land it in the same
   reindex as the upgrade. `arcaneum#3h9w`.
3. **Fix the write pattern**: group deletions ahead of additions
   (`arcaneum#w2x0`; delete-by-filter cannot be autobatched with adds). Then
   re-validate RDR-024's premise on the new engine, finalize it, and
   implement cross-file batching (`arcaneum#q0hv`).
4. **Defer bulk builds**: load a side index with large batches and settings
   applied first, verify, then swap it in with `swap-indexes`. This moves
   CPU off interactive syncs; it does not remove it (`arcaneum#7mke`).

### Offloading Meili: not viable

- Dumps reindex on import, so there is no CPU saving on the receiver
  (**Documented**).
- Snapshots are whole-instance only. S3 snapshot upload is Enterprise-only.
- A raw `data.ms` copy requires the identical version. arm64 to x86
  portability is undocumented.
- `/export` pushes documents to a live receiver, which most likely
  reindexes them (**Assumed**).

### Alternative lexical backends

If Meili still hurts after levers 1–3, evaluate a backend whose index can be
built remotely (e.g. in the clusterless embedding job) and shipped
(`arcaneum#60dc`, related to `xtqt`):

| Option | Phrase | Identifiers | Typo tolerance | Build elsewhere and ship |
| --- | --- | --- | --- | --- |
| Qdrant BM25 sparse + full-text `MatchPhrase` filter (1.15+) | Yes, as a filter | Good, with custom tokenization | No | Sparse vectors: yes. Payload index builds in Qdrant. |
| Tantivy (`tantivy-py`) | Yes, positional | Excellent with code tokenizers (Bloop) | Manual only | Yes; cross-arch **Assumed** |
| SQLite FTS5 (trigram / standard) | Yes, standard tokenizer | Good with trigram | No | Yes, one file |
| MeiliSearch (current) | Yes | Good | Yes | No |

The Qdrant-native option would also remove Qdrant/Meili parity entirely,
since chunk text is already in the Qdrant payload. The BM25 plus
phrase-filter combination is **Assumed** workable and needs a spike.

## Next steps

Measure before committing to the remote design:

1. `arcaneum#bv7m`: benchmark CPU vs GPU throughput, cost, and vector parity
   with the existing accelerator harness (under $10).
2. `arcaneum#45ed`: attribute bulk sync wall-clock across phases. Is
   embedding the bottleneck?
3. `arcaneum#vby2` (chunk-hash cache) and `arcaneum#nq22` (Qdrant bulk mode)
   pay off locally regardless of the cloud decision.
4. `arcaneum#tpcs`: RDR seed for the remote design, gated on 1 and 2.

Full-text, independent of the cloud work and likely the bigger local win:

1. Batch `batch:meili-indexing`: `arcaneum#wk39` (upgrade), `arcaneum#3h9w`
   (settings trim, after the upgrade), `arcaneum#w2x0` (grouped deletes).
2. `arcaneum#q0hv`: re-validate and implement RDR-024 after the upgrade.
3. `arcaneum#7mke` and `arcaneum#60dc`: seeds, gated on the results above.

## Sources

- [AWS Fargate FAQs](https://aws.amazon.com/fargate/faqs/)
- [SageMaker Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
- [Lambda Managed Instances](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances.html)
- [Use GPUs with ECS Managed Instances](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/managed-instances-gpu.html)
- [AWS Batch on ECS Managed Instances announcement](https://aws.amazon.com/about-aws/whats-new/2026/08/aws-batch-on-ecs-managed-instances/)
- [AWS Batch: when to use ECS Managed Instances](https://docs.aws.amazon.com/batch/latest/userguide/when-to-use-ecs-managed-instances.html)
- [ECS Managed Instances GPU fee reduction](https://aws.amazon.com/about-aws/whats-new/2026/07/amazon-ecs-managed-instances-gpu-price/)
- [GPU batch inference on ECS Managed Instances with scale to zero](https://aws.amazon.com/blogs/containers/run-gpu-batch-inference-on-amazon-ecs-managed-instances-with-scale-to-zero/)
- [SOCI cold-start reduction](https://aws.amazon.com/blogs/machine-learning/reducing-container-cold-start-times-using-soci-index-on-dlami-and-dlc/)
- [Qdrant bulk upload](https://qdrant.tech/documentation/manage-data/bulk-upload/)
- [Meilisearch: importing large datasets](https://www.meilisearch.com/docs/capabilities/indexing/how_to/import_large_datasets)
- [Cursor: secure codebase indexing](https://cursor.com/blog/secure-codebase-indexing)
- [Qdrant/Bloop case study](https://qdrant.tech/blog/case-study-bloop/)
- [Modal: Amazon reviews embeddings](https://modal.com/docs/examples/amazon_embeddings)
- [Anyscale: RAG at scale](https://www.anyscale.com/blog/rag-at-scale-10x-cheaper-embedding-computations-with-anyscale-and-pinecone)
- [RunPod GPU embedding benchmark](https://www.runpod.io/blog/gpu-embedding-workloads-benchmark)
- [HF Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference/index)
- [Bedrock batch inference](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html)
- [Meilisearch releases](https://github.com/meilisearch/meilisearch/releases)
- [Meilisearch 1.12 new indexer](https://www.meilisearch.com/blog/introducing-indexer-2024)
- [Meilisearch 1.14 granular filters](https://www.meilisearch.com/blog/meilisearch-1-14)
- [Meilisearch dumps](https://www.meilisearch.com/docs/learn/data_backup/dumps)
- [Qdrant full-text search](https://qdrant.tech/documentation/search/text-search/full-text-search/)
- [Qdrant 1.15](https://qdrant.tech/blog/qdrant-1.15.x/)
- [Qdrant BM25 inference](https://qdrant.tech/documentation/inference/inference-bm25/)
- [Tantivy architecture](https://github.com/quickwit-oss/tantivy/blob/main/ARCHITECTURE.md)
- [zoekt](https://github.com/sourcegraph/zoekt)

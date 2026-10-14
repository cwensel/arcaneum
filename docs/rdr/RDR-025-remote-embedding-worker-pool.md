# Recommendation 025: Remote Embedding Worker Pool

> Revise during planning; lock at implementation.
> If wrong, abandon code and iterate RDR.

## Metadata

- **Date**: 2026-10-14
- **Status**: Draft
- **Type**: Architecture
- **Priority**: Medium
- **Related Issues**: Extends RDR-022 (spawned accelerator worker
  protocol); intermediate step toward kata `arcaneum#tpcs` under umbrella
  `arcaneum#rrky` (`docs/cloud-bulk-embedding-feasibility.md`); deferred
  work relates to RDR-024 (cross-file batching) and `arcaneum#vby2`
  (chunk-hash embedding cache)

## Problem Statement

The workstation running `arc` is shared with other concurrent projects.
Document embedding during `arc corpus sync` and `arc corpus repair` is the
largest CPU and memory consumer `arc` itself places on it. `nice -n 20`
lowers its priority but does not remove the work, the threads, or the
resident model.

The goal is to **lower local CPU and memory use**, not to make embedding
faster. Document embedding should run on spare machines on the local
network, in Docker containers, over a simple RPC, while producing vectors
identical (within float tolerance) to those the workstation produces, so
remote and local vectors can be mixed in one collection and queried with
locally embedded queries.

Phase 1 covers `arctic-m` document embedding by default and `qwen3-embed`
document embedding on request, for long unattended runs such as a repair or
forced resync. Query embedding stays local in Phase 1.

## Context

### Background

Evidence from the workstation's own logs and a local profile
(2026-10-13/14):

- **Sync is mostly encoding.** Memory-probe samples (~1 Hz) for two
  `arc corpus sync` runs on 2026-10-14 attribute 76% and 84% of samples to
  the `encoding` phase, 12–15% to `file` (extraction and chunking), and
  4–9% to `indexing`. Both runs encoded in-process on CPU (no MPS memory,
  no accelerator worker). Samples are wall-clock, not CPU time; encoding
  uses all cores minus two while extraction is mostly single-threaded, so
  encoding's CPU share is likely higher still.
- **Search cost depends on the model.** Over the same two days, semantic
  searches against `qwen3-embed` corpora had median 9.2 s (671 calls) and
  against `arctic-m` corpora median 2.7 s (156 calls). A local profile
  attributes 10.0 s of a 15.2 s `qwen3-embed` search to model load and first
  query, but only 0.3 s of a 4.0 s `arctic-m` search. Remote query
  embedding would therefore help `qwen3-embed` search substantially and
  `arctic-m` search hardly at all; it is deferred (see Deferred Work).
- **Memory is tight.** During sync the workstation sat near 75% RAM use with
  about 18 GB of swap in use; `arc` reached 2.7 GB RSS.

Models in use (Qdrant collection vectors, 2026-10-14):

- `arctic-m` (fastembed ONNX, 768D): `Claude` (657k points), `PapersFast`
  (390k), `StateMachineRes` (19k).
- `qwen3-embed` (sentence-transformers, 1024D), ~263k points across 12
  collections: `DevRef` (101k), `RetrofitRDR` (69k), `NextLit` (29k),
  `SchemaEvo` (18k), `SchemaEvoRes` (14k), `MergeLit`, `LLM`, `SqlSafety`
  (~7k each), `SpecDrivenDev`, `CodeMaintenance`, `SchemaEvoMarket`
  (~3–4k each), `StateMachineLit` (1k).

### Technical Environment

- Workstation: Apple M2 Pro (8 performance + 4 efficiency cores), 32 GB,
  macOS, Python 3.12, fastembed 0.8.0.
- Candidate workers: `dev-0`, `dev-1` — Intel i7-3720QM (Ivy Bridge,
  4 cores / 8 threads, AVX and F16C but **no AVX2 or FMA**), 16 GB RAM
  (~12 GB available), Ubuntu with Linux 6.8, Docker 29.8.1, no GPU, local
  network.
- Embedding entry points: `EmbeddingClient.embed()`
  (`src/arcaneum/embeddings/client.py:1411`), called once per file per
  model by sync (`src/arcaneum/cli/sync.py:4081`, `4496`, `5848`) and by
  the indexing pipelines through `embed_parallel()`
  (`client.py:1766`).
- RDR-022 protocol: `src/arcaneum/embeddings/worker_protocol.py` (versioned
  envelopes, request IDs, `WorkerBackend.encode()`), currently over
  `multiprocessing` spawn queues.
- Existing command options: `arc corpus sync` and `arc corpus repair` both
  accept `--no-gpu` and `--max-embedding-batch`; neither has a remote
  option.

## Research Findings

### Investigation

**Where the remote boundary must sit.** The RDR-022 worker is only used for
the sentence-transformers accelerator path (`client.py:980-1000`,
`1729`). CPU encodes short-circuit to in-process
`_encode_on_cpu_fallback` (`client.py:1716-1717`), and fastembed models are
always in-process. A remote backend placed behind the accelerator worker
would therefore miss every encode on this workstation. The boundary has to
be "embed these document texts with model M", at
`EmbeddingClient.embed()` / `embed_parallel()`.

**Model revisions.** sentence-transformers registry entries pin a commit
hash and refuse to load without one (`client.py:406-408`). fastembed
registry entries such as `arctic-m` (`client.py:309-316`) carry no
revision. fastembed 0.8.0 has no revision parameter: a non-offline download
resolves the repository's current HEAD (`model_info(repo).sha`,
`fastembed/common/model_management.py:235`). It does accept
`specific_model_path`, which loads a model directory directly and skips
download (`model_management.py:402-404`; accepted by `TextEmbedding` via
`fastembed/text/onnx_embedding.py:209,251,256`). On 2026-10-14 the
workstation cache, the dev-0 cache, and Hugging Face HEAD all resolve
`Snowflake/snowflake-arctic-embed-m` to `fc74610d18462d218e312aa986ec5c8a75a98152`
(last modified 2024-12-13).

**Transport.** `multiprocessing.connection` frames each message with a
4-byte length, switching to an 8-byte length above 2 GiB
(`multiprocessing/connection.py:406-416`), so payload size is not a
practical limit; a 64-chunk `arctic-m` batch is about 196 KB of float32.
`Listener`/`Client` take an optional `authkey` and use an HMAC challenge
(`connection.py:928-949`).

**Feasibility spike (2026-10-14).** Real chunks sampled from Qdrant payloads
(96 `NextLit` chunks for `qwen3-embed`, p50 2,385 chars; 192 `PapersFast`
chunks for `arctic-m`, p50 1,443 chars) were embedded with arcaneum's own
`EmbeddingClient(use_gpu=False)` at commit `b7a8a2e`. Remote: `dev-0` in a
`python:3.12-slim` container that installed arcaneum and its
`sentence-transformers` extra at startup into a cached venv volume,
CPU-only torch, models downloaded on first use into a mounted cache.
Local: the workstation, same script, under `nice -n 20` while the machine
carried a load average of ~18 on 12 cores (its normal contended state, not
an idle baseline).

| Measure | `arctic-m` workstation | `arctic-m` dev-0 | `qwen3-embed` workstation | `qwen3-embed` dev-0 |
| --- | --- | --- | --- | --- |
| Threads (`OMP_NUM_THREADS`) | 10 | 6 | 10 | 6 |
| First load (s) | 0.6 | 7.3 | 8.4 | 30.0 |
| Encode (s) | 50.0 | 189.9 | 106.2 | 1,524.7 |
| Chunks/s | 3.84 | 1.01 | 0.90 | 0.063 |
| Peak RSS (MB) | 5,774 | 7,070 | 4,307 | 6,004 |
| Resident query latency (s) | — | 0.02–0.05 | — | 0.46–0.62 |

- **Parity**: dev-0 vectors match workstation vectors for both models:
  per-chunk cosine 1.000000 (min), max absolute difference 9.7e-8
  (`arctic-m`) and 5.7e-7 (`qwen3-embed`), identical nearest neighbor for
  every chunk within each sample.
- **Throughput**: per box, dev-0 is ~3.8× slower than the contended
  workstation for `arctic-m` (ONNX) and ~14× slower for `qwen3-embed`
  (torch). torch reports CPU capability `DEFAULT` on Ivy Bridge, i.e. no
  AVX2 kernels; onnxruntime degrades far less. Across two boxes,
  `arctic-m` runs at roughly half the workstation's rate, while
  `qwen3-embed` runs at ~450 chunks/hour, which suits unattended repairs
  and forced resyncs but not routine syncs.
- **Memory**: peak RSS is 6.0–7.1 GB per model in separate processes,
  driven largely by batch activations (`arctic-m` peaks higher than the
  larger `qwen3-embed`). Either model fits on a 16 GB box with ~12 GB
  available; both resident in one server would not fit comfortably.
- **Install at startup**: a stock `python:3.12-slim` container with a
  cached venv volume ran arcaneum's CPU path unchanged (torch 2.14.1+cpu,
  numpy 2.5.4, onnxruntime 1.31.0, sentence-transformers 5.7.0). Install
  duration was not isolated.

#### Dependency Source Verification

| Dependency | Source Searched? | Key Findings |
| --- | --- | --- |
| `multiprocessing.connection` (stdlib 3.12) | Yes | 4-byte length frame, 8-byte above 2 GiB (`connection.py:406-416`); optional `authkey` HMAC challenge (`connection.py:928-949`). |
| fastembed 0.8.0 | Yes | No revision parameter; download resolves repo HEAD (`model_management.py:235`); `specific_model_path` loads a local directory and skips download (`model_management.py:402-404`, `onnx_embedding.py:209-256`). |
| huggingface_hub `snapshot_download` | Yes (as called by fastembed) | Accepts `revision` via kwargs passthrough in fastembed's own call (`model_management.py:226-231`, `258-262`). |
| sentence-transformers 5.7.0 / torch 2.14.1+cpu | No (spike only) | Ran on Ivy Bridge; torch selects `DEFAULT` CPU capability. Registry already pins revisions. |

### Key Discoveries

- **Verified**: arcaneum's CPU embedding path runs unchanged on `dev-0` in
  a stock Python container with install-at-startup and on-demand model
  download, and its vectors match the workstation's.
- **Verified**: the candidate machines have enough memory for one resident
  model each, not two; compute, not memory, limits them, severely so for
  `qwen3-embed`.
- **Documented**: the remote boundary must wrap `EmbeddingClient`'s
  document path, not the RDR-022 accelerator worker (`client.py:1716`).
- **Documented**: fastembed cannot pin a revision itself, but arcaneum can
  download a pinned snapshot and pass it as `specific_model_path`.
- **Verified**: `qwen3-embed` search latency is dominated by local model
  load; `arctic-m` search latency is not.

### Critical Assumptions

- [x] Remote vectors agree with local CPU vectors for every model in use
  (cosine ≥ 0.9999 per chunk) — **Status**: Verified for `arctic-m` and
  `qwen3-embed`, the only models in use (min cosine 1.000000)
  — **Method**: Spike
- [x] fastembed models can be pinned so a worker loads the same weights as
  the workstation — **Status**: Verified (pinned snapshot download plus
  `specific_model_path`; both machines currently resolve the same snapshot)
  — **Method**: Source Search
- [x] The transport carries batch payloads without size limits relevant to
  this use — **Status**: Verified — **Method**: Source Search

## Proposed Solution

### Approach

Run arcaneum itself as an embedding server inside Docker on each worker
machine, and route the workstation's **document** embedding for selected
models to a pool of those servers over the local network.

1. **Server (`arc models serve`)**: a long-lived process wrapping a CPU
   `EmbeddingClient`. It accepts RDR-022 envelopes over
   `multiprocessing.connection.Listener` on a LAN port, loads a model
   lazily on first request (downloading the pinned revision into a mounted
   cache volume), and keeps it resident.
2. **Container**: stock `python:3.12-slim` image. At startup a script
   installs a pinned arcaneum version into a cached venv volume,
   reinstalling only when the requested version changes, then runs
   `arc models serve`. No arcaneum-specific image is built or distributed;
   model weights live in a cache volume, not in an image.
3. **Client pool**: a configured list of `host:port` endpoints. One request
   in flight per endpoint (RDR-022's rule, applied per endpoint). An
   `embed()` call is split into sub-batches dispatched to whichever
   endpoints are idle and serve the model; results are reassembled in
   order by request ID.
4. **Routing**: per-model configuration decides whether a model's document
   embedding is remote. Phase 1 defaults: `arctic-m` remote whenever
   endpoints are configured; `qwen3-embed` local unless the command is run
   with `--remote-embed`.

Security is an explicit non-goal: the pool runs on a trusted local network,
pickle over the socket is acceptable because both ends run the same pinned
arcaneum, and an optional static `authkey` is the only access control.

### Technical Design

**Handshake.** The first message on a connection is `initialize`, carrying
protocol version, arcaneum version, model key, pinned model revision,
prompt-policy version, and expected dimensions. The server loads the model
(downloading the pinned revision if needed) and replies `initialized` with
the same fields as resolved on its side. Any mismatch excludes that endpoint
for that model. While a download or load is in progress the server answers
heartbeats with a `loading` state; encode deadlines start only after
`initialized`, so first use never trips a timeout.

**fastembed revision pinning.** fastembed registry entries gain a
`revision` field (`arctic-m`: `fc74610d18462d218e312aa986ec5c8a75a98152`).
When a fastembed model has a pinned revision, both the workstation and the
server obtain that snapshot themselves and construct `TextEmbedding` with
`specific_model_path` pointing at it. fastembed models without a pinned
revision are not eligible for remote routing.

**Encode.** `encode` carries document texts and model key; `encoded`
returns an independently owned C-contiguous float32 array, validated with
the existing `_validate_embeddings` checks before use.

**Model placement.** A 16 GB worker holds one resident model comfortably,
not two. Each endpoint keeps at most one model resident; a request for
another model evicts the first. Endpoint configuration may restrict which
models an endpoint serves so that, if desired, a long `qwen3-embed` run
can be confined to one box while the other keeps serving `arctic-m`. The
server's encode batch size is configurable to bound peak RSS.

**Pool behavior.**

- An endpoint that errors, disconnects or exceeds the encode deadline is
  marked down and its in-flight batch is retried on another endpoint.
  Encoding has no side effects, so retry is safe.
- Down endpoints are probed with the existing heartbeat message and
  rejoin when healthy.
- When no endpoint can serve a remote-routed model, a configured policy
  applies: `wait` (default; block and keep probing, reporting the
  unreachable endpoints), `local` (embed locally, today's behavior), or
  `fail` (stop before writing that file's chunks).

```text
// Illustrative — verify API signatures during implementation
RemoteEmbeddingPool(endpoints: list[Endpoint], unavailable: "wait"|"local"|"fail")
  .embed(texts: list[str], model: str) -> np.ndarray   # (n, dim) float32
  .health() -> list[EndpointHealth]  # endpoint, state, resident model, versions
```

**Configuration.** Endpoints, per-model routing and the unavailable policy
live in arcaneum config, with an environment override for endpoints
(following the `ARC_QDRANT_URL` pattern), e.g.
`ARC_EMBED_ENDPOINTS=dev-0:7300,dev-1:7300`. `arc corpus sync` and
`arc corpus repair` gain `--remote-embed` (route every model in the run to
the pool) and `--local-embed` (route none), overriding configuration for
that invocation.

### Existing Infrastructure Audit

| Proposed Component | Existing Module | Decision |
| --- | --- | --- |
| Message envelopes, validation | `embeddings/worker_protocol.py` (`make_message`, `_validate_message`, `MessageType`) | Reuse: same envelope and version field over a socket transport |
| Worker serve loop | `worker_protocol._worker_main` | Extend: run over a connection-like object so queue and socket transports share it |
| Server-side encoding | `EmbeddingClient` CPU path | Reuse: the server embeds through the same client so preprocessing and prompt policy are identical |
| Output validation | `EmbeddingClient._validate_embeddings` | Reuse on the client side for remote results |
| fastembed model loading | `EmbeddingClient` fastembed construction and cache checks (`client.py:2377-2441`) | Extend: pinned snapshot plus `specific_model_path` when a revision is set |
| Thread sizing | `client.py:1024-1040` | Reuse on the server; the server sizes threads to its own cores |
| Command options | `--no-gpu`, `--max-embedding-batch` on sync and repair | Extend: add `--remote-embed` / `--local-embed` alongside |

### Decision Rationale

The spike shows the existing CPU path runs unchanged on the candidate
machines with vectors equal to within 6e-7, so the only new parts are a
transport, a pool, a server entry point, and fastembed revision pinning.
Running arcaneum's own code on the server keeps preprocessing, truncation
and prompt policy identical to local embedding, which is what lets remote
and local vectors share a collection. `multiprocessing.connection` is the
smallest transport that carries the existing envelopes unchanged on a
trusted network.

Starting with `arctic-m` puts the frequent, small `Claude` corpus syncs on
the workers, where two boxes run at about half the workstation's rate.
`qwen3-embed` is 14× slower per box, so it is routed remotely only when
asked, for long unattended runs where wall-clock does not matter.

## Alternatives Considered

### Alternative 1: Ollama endpoints for `qwen3-embed`

**Description**: Run the official `ollama/ollama` container on workers and
embed `qwen3-embed` corpora through its `/api/embed`, applying the
registry's query prompt client-side.

Spike on dev-0 (2026-10-14, Ollama 0.40.4, same 96 chunks and five queries,
batches of 8, llama.cpp on 4 threads), compared with sentence-transformers
vectors:

| Tag | Chunks/s | Query (s) | Server RSS | Doc cosine min / p50 | Query cosine min | Top-10 overlap |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3-embedding:0.6b` (Q8_0) | 0.084 | 0.43–0.78 | ~2.6 GB | 0.99621 / 0.99932 | 0.99914 | 0.98 |
| `qwen3-embedding:0.6b-fp16` | 0.077 | 0.44–0.66 | ~3.2 GB | 0.99664 / 1.00000 | 1.00000 | 1.00 |

Top-10 overlap is the mean overlap of each query's top ten chunks (Ollama
query against sentence-transformers documents) with the all-
sentence-transformers ranking. With fp16, 95 of 96 chunks match at cosine
≥ 0.9999; the one outlier (0.99664) is a chunk ending in markdown emphasis,
likely a tokenization difference (**Assumed**).

**Pros**:

- fp16 vectors are compatible with existing `qwen3-embed` collections.
- About half the server memory of the torch path (3.2 GB vs 6.0 GB).
- Off-the-shelf server; the model is pulled at startup.

**Cons**:

- Only ~1.2–1.3× faster than torch on the pre-AVX2 workers.
- Prompt policy is reimplemented outside sentence-transformers, and model
  identity is an Ollama digest rather than the registry revision.
- Covers `qwen3-embed` only; `arctic-m`, the Phase 1 model, still needs an
  arcaneum endpoint.

**Reason for rejection**: Phase 1 targets `arctic-m`, which Ollama does not
cover with verified parity, and `qwen3-embed` through `arc models serve`
already matches exactly. Revisit if `qwen3-embed` worker memory becomes the
constraint (e.g. when query routing makes it resident). Ollama Cloud is not
an option today: its model list (`https://ollama.com/api/tags`,
2026-10-14) has no embedding models, and ollama/ollama#17129 requesting
them is open.

### Alternative 2: HTTP + msgpack transport

**Description**: Serve the envelopes over HTTP with msgpack bodies.

**Pros**:

- Language-neutral, inspectable, no pickle.

**Cons**:

- More code (server framework, array encoding) with no benefit on a
  trusted LAN where both ends run identical arcaneum.

**Reason for rejection**: Revisit when the same server is exposed beyond
the LAN (e.g. the `arcaneum#tpcs` cloud path).

### Briefly Rejected

- **Remote query embedding in Phase 1**: helps only `qwen3-embed` search,
  and keeping `qwen3-embed` resident pins a 6 GB model to one of two
  workers; deferred.
- **TEI or Infinity servers**: different preprocessing and prompt handling
  from arcaneum's, so vectors could drift from locally embedded queries.
- **Run all of `arc corpus sync` on the remote machine**: indexed file paths
  are corpus identity, and git hooks trigger syncs locally.
- **Thread caps only (`OMP_NUM_THREADS`)**: lowers peak contention but keeps
  all the work and the resident model local; worth doing separately.
- **Prebuilt arcaneum image**: requires building and shipping an image per
  release; install-at-startup with a cached venv avoids that.

## Trade-offs

### Consequences

- Positive: `arctic-m` sync embedding CPU and model memory leave the
  workstation by default once endpoints are configured.
- Positive: repairs and forced resyncs of `qwen3-embed` corpora can run
  unattended without loading the workstation.
- Positive: the server and pool are the shape the cloud path
  (`arcaneum#tpcs`, `clusterless#2cy0`) needs for fan-out.
- Negative: sync wall-clock depends on worker speed: about 2× longer for
  `arctic-m` across two workers, and hours rather than minutes for
  `qwen3-embed` with `--remote-embed`.
- Negative: with the default `wait` policy, a sync stalls while all
  workers are down instead of falling back.

### Risks and Mitigations

- **Risk**: A worker loads different weights or code than the workstation
  and silently mixes incompatible vectors into a collection.
  **Mitigation**: Handshake rejects any version, revision, prompt-policy or
  dimension mismatch; fastembed models are remote-eligible only with a
  pinned revision.
- **Risk**: First-use model download or arcaneum reinstall makes a worker
  look hung.
  **Mitigation**: `loading` heartbeat state; deadlines start after
  `initialized`.
- **Risk**: A `--remote-embed` `qwen3-embed` run evicts `arctic-m` on both
  workers and stalls routine syncs.
  **Mitigation**: Per-endpoint model restriction confines `qwen3-embed` to
  one worker when desired.
- **Risk**: Pickle deserialization from the network.
  **Mitigation**: Accepted for a trusted LAN; bind to the LAN interface and
  document that the port must not be exposed.

### Failure Modes

- All endpoints down during sync: `wait` blocks visibly, naming the
  unreachable endpoints; `fail` stops before writing that file's chunks;
  `local` logs that it fell back.
- Endpoint dies mid-batch: batch retried elsewhere; logged with endpoint
  and request ID.
- Handshake mismatch: endpoint excluded for that model; shown by
  `arc models workers`.
- Diagnosis: `arc models workers` reports each endpoint's state, versions
  and resident model; the existing memory-probe log shows whether the
  workstation process loaded a model.

## Implementation Plan

### Prerequisites

- [x] All Critical Assumptions verified
- [x] Docker available on each worker (Docker 29.8.1 on `dev-0`, `dev-1`)

### Minimum Viable Validation

With `dev-0` and `dev-1` configured and the unavailable policy set to
`fail`:

1. `arc corpus sync` of a scratch `arctic-m` corpus of a few markdown files
   completes, the workstation's memory-probe log shows no local model load,
   and stored vectors match a local CPU re-embedding at cosine ≥ 0.9999.
2. `arc corpus sync --remote-embed --force` of a scratch `qwen3-embed`
   corpus of two small PDFs completes with the same checks.
3. `arc search semantic` against both scratch corpora (local query
   embedding) returns the same top results as when the corpora are indexed
   locally.

### Phase 1: Code Implementation

#### Step 1: Transport-neutral worker loop

Run `_worker_main` over a connection-like object; keep the spawn-queue
transport passing its existing tests.

#### Step 2: fastembed revision pinning

Add `revision` to fastembed registry entries (`arctic-m` first); load
pinned snapshots via `specific_model_path` on both sides.

#### Step 3: `arc models serve`

Server entry point: listener, handshake with `loading` state, lazy model
load with single-resident eviction, encode, heartbeat.

#### Step 4: Remote pool client and routing

Endpoint config, handshake, sub-batch dispatch, retry, health probing,
unavailable policy, per-model routing in `EmbeddingClient.embed()` and
`embed_parallel()`, and `--remote-embed` / `--local-embed` on
`arc corpus sync` and `arc corpus repair`.

#### Step 5: `arc models workers`

List configured endpoints with state, versions and resident model.

#### Step 6: Container startup script

Installs a pinned arcaneum version into a cached venv volume and runs the
server; documented `docker run` invocation.

### Phase 2: Operational Activation

#### Activation Step 1: Start workers

Run the documented container on `dev-0` and `dev-1` with cache and venv
volumes and the LAN port published.

#### Activation Step 2: Configure the workstation

Set the endpoints in arcaneum config and confirm with
`arc models workers`.

### Deferred Work

- **Remote query embedding** for `qwen3-embed` corpora (local model load is
  ~10 s per search; a resident remote model answers in ~0.6 s). Needs a
  decision on dedicating a worker, or the Ollama alternative's lower
  memory.
- **Cross-file batching** for embedding (the RDR-024 shape), so corpora of
  small files keep more than one endpoint busy.

### Day 2 Operations

| Resource | List | Info | Delete | Verify | Backup |
| --- | --- | --- | --- | --- | --- |
| Endpoint config | In scope (`arc models workers`) | In scope (state, versions, resident model) | In scope (config edit) | In scope (handshake + heartbeat) | N/A |
| Worker cache and venv volumes | Deferred | Deferred | Deferred (`docker volume rm`) | N/A | N/A |

The volumes hold re-downloadable caches; Docker's own commands suffice
initially.

### New Dependencies

None on the workstation. Workers need Docker and network access to PyPI
and Hugging Face on first start.

## Validation

### Testing Strategy

1. **Scenario**: Socket transport round-trip with the deterministic fake
   backend.
   **Expected**: Same results and error envelopes as the queue transport.
2. **Scenario**: Handshake with mismatched protocol version, arcaneum
   version, revision, prompt policy or dimension.
   **Expected**: Endpoint excluded for that model; no encode sent.
3. **Scenario**: Server still downloading or loading a model.
   **Expected**: Client sees `loading`, does not time out, and proceeds
   after `initialized`.
4. **Scenario**: Endpoint killed mid-batch with a second endpoint healthy.
   **Expected**: Batch retried; result identical and in order.
5. **Scenario**: All endpoints down under each policy.
   **Expected**: `wait` blocks and recovers, `fail` stops before writing
   chunks, `local` embeds locally.
6. **Scenario**: fastembed model without a pinned revision configured for
   remote routing.
   **Expected**: Refused with a message naming the missing revision.
7. **Scenario**: `--remote-embed` and `--local-embed` on sync and repair.
   **Expected**: Override per-model routing for that invocation only.

### Performance Expectations

The goal is local resource reduction, measured as workstation CPU time and
peak RSS of `arc corpus sync` with and without the pool. Remote throughput
is in the feasibility spike above; no throughput target is set.

## Finalization Gate

### Contradiction Check

No contradictions found between research findings, design principles, and
proposed solution. The earlier draft proposed remote query embedding for
all models; the local profile showed it would not help `arctic-m` search,
so query routing moved to Deferred Work.

### Assumption Verification

All three Critical Assumptions are verified: vector parity by spike for
both models in use, fastembed pinning by source search of fastembed 0.8.0,
and transport framing by source search of the Python 3.12 stdlib.

#### API Verification

| API Call | Library | Verification |
| --- | --- | --- |
| `multiprocessing.connection.Listener` / `Client` (`authkey`) | stdlib 3.12 | Source Search |
| `Connection.send_bytes` length framing | stdlib 3.12 | Source Search |
| `TextEmbedding(..., specific_model_path=...)` | fastembed 0.8.0 | Source Search |
| `snapshot_download(repo_id, revision=..., cache_dir=...)` | huggingface_hub | Source Search (fastembed's own call site); exercised in Step 2 tests |
| `EmbeddingClient.embed()` on CPU in a Linux container | arcaneum | Spike |

### Scope Verification

The Minimum Viable Validation is in scope: the three checks run against
`dev-0` and `dev-1` as the last step of Phase 1, before activation.

### Cross-Cutting Concerns

- **Versioning**: Handshake carries protocol and arcaneum versions; the
  container reinstalls on version change.
- **Build tool compatibility**: N/A
- **Licensing**: N/A (no new dependencies)
- **Deployment model**: Stock Python container, install at startup,
  cache and venv volumes.
- **IDE compatibility**: N/A
- **Incremental adoption**: Off unless endpoints are configured;
  `qwen3-embed` stays local unless `--remote-embed` is given.
- **Secret/credential lifecycle**: N/A (trusted LAN; optional static
  `authkey`)
- **Memory management**: The workstation loads no model for remote-routed
  work. Workers keep one resident model (6.0–7.1 GB peak measured) with a
  configurable batch size.

### Proportionality

Right-sized. The Ollama spike table is retained because it records why a
lower-memory option was not chosen for Phase 1.

## References

- RDR-022: Spawned Accelerator Worker Protocol
- RDR-024: Adaptive Cross-File Index Batching
- `docs/cloud-bulk-embedding-feasibility.md`
- Katas: `arcaneum#rrky`, `arcaneum#tpcs`, `arcaneum#45ed`, `arcaneum#vby2`
- `src/arcaneum/embeddings/worker_protocol.py`,
  `src/arcaneum/embeddings/client.py`, `src/arcaneum/search/embedder.py`,
  `src/arcaneum/cli/sync.py`
- fastembed 0.8.0: `fastembed/common/model_management.py`,
  `fastembed/text/onnx_embedding.py`
- Python 3.12 `multiprocessing/connection.py`
- Ollama Cloud docs (`https://docs.ollama.com/cloud`); ollama/ollama#17129

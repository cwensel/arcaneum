# Recommendation 024: Adaptive Cross-File Index Batching

> Revise during planning; lock at implementation.
> If wrong, abandon code and iterate RDR.

## Metadata

- **Date**: 2026-08-18
- **Status**: Draft
- **Type**: Technical Debt
- **Priority**: Medium
- **Related Issues**: Extends RDR-009 (dual indexing strategy);
  builds on RDR-023 (merge-cost findings)

## Problem Statement

`arc corpus sync` submits one MeiliSearch task per file. Because MeiliSearch
merge cost scales with total index size rather than batch size, every small
file pays a full-index merge. On a large corpus this dominates sync wall-clock
time, and the cost grows as the index grows.

The fix is to batch documents across file boundaries. The constraint is that
the per-file manifest checkpoint — which makes sync resumable — is currently
coupled to the per-file index call. This RDR proposes decoupling them without
reintroducing the rollback complexity RDR-009 deliberately rejected.

## Context

### Background

Observed during a `PapersFast` sync of 6 new files. Each file logged
`Timed out waiting for MeiliSearch task NNNNN; retrying wait (2/3)` and took
80–146 s server-side despite carrying only 46–55 documents.

Investigation of the live MeiliSearch task queue found the retry warnings are
not a defect: they are `c7bedcc` ("poll MeiliSearch tasks past the retry cap
while progressing") working as designed. Every task reported `succeeded`. The
60 s default timeout in `add_documents_sync` is intentionally non-binding —
`c7bedcc` bounded the attempt cap to *stalled* tasks so that a task the server
still reports as `processing` resets the budget and keeps polling. Before that
commit this same situation surfaced as a cancelled sync.

So the slowness is not a regression. It is the expected cost curve of the
index at its current size, now large enough to be user-visible.

### Technical Environment

- MeiliSearch v1.12 (`meilisearch-arcaneum` container),
  `MEILI_MAX_INDEXING_MEMORY=2.5GiB`, `MEILI_MAX_INDEXING_THREADS=4`,
  `MEILI_HTTP_PAYLOAD_SIZE_LIMIT=100MB`, 4 GiB container memory limit
- Qdrant v1.18.2 (`qdrant-arcaneum`)
- `DualIndexer.batch_size` default 300 (`dual_indexer.py:47`)
- Per-file index + manifest sequence at `sync.py:3408-3417`
- File manifests introduced by `221089e`, hardened by `4fda3fa`, `7d0988b`,
  `0ff1fd5`, `568eeca`, `9ce1fb7`, `89ede1a`, `7273aa0`, `f96fa3c`

## Research Findings

### Investigation

Analyzed the live MeiliSearch task history for the `PapersFast` index: 4,007
successful `documentAdditionOrUpdate` tasks spanning 2026-08-03 to 2026-08-18,
covering the full life of the index. Cumulative document counts were used to
reconstruct index size at the time each task ran.

Also read `dual_indexer.py:64-128`, the sync indexing call site at
`sync.py:3398-3420`, `_upsert_file_manifest` at `sync.py:284-305`, the
MeiliSearch wait logic at `fulltext/client.py:39-101`, and RDR-009 and RDR-023.

#### Dependency Source Verification

| Dependency | Source Searched? | Key Findings |
| --- | --- | --- |
| meilisearch-python | Yes | `index.add_documents` returns `TaskInfo` with `task_uid`; `client.wait_for_task(uid, timeout_in_ms=)` raises `MeilisearchTimeoutError` on deadline. Confirmed at `fulltext/client.py:66`. |
| MeiliSearch server v1.12 | Yes (live) | `GET /tasks` exposes `duration` (ISO-8601), `details.indexedDocuments`, `status`. `GET /stats` exposes `numberOfDocuments` per index. |
| qdrant-client | No | Batching for Qdrant is unchanged by this RDR; existing `upsert` loop at `dual_indexer.py:99-102` is reused as-is. |

### Key Discoveries

- **Verified** — Task duration for a fixed ~46-document batch scales with
  index size, measured across the index's full history:

  | Index size at run | Median task duration |
  | --- | --- |
  | 0 | 0.7 s |
  | 50,000 | 1.6 s |
  | 100,000 | 3.1 s |
  | 150,000 | 3.9 s |
  | 200,000 | 6.1 s |
  | 250,000 | 10.1 s |
  | 300,000 | 20.3 s |

  Identical work, 29× more expensive at 300k than at the start. Per-document
  cost rises from 0.007 s/doc to 0.383 s/doc across the same span. The curve is
  smooth and superlinear — no discontinuity, so no regression event.

- **Verified** — 1,581 of 4,007 tasks (~40%) carried fewer than 50 documents.
  Each paid a full-index merge for a handful of documents.

- **Verified** — Cumulative server-side indexing cost to build the current
  347,881-document index: 33,591 s (9.3 hours), disproportionately spent late
  on the curve.

- **Verified** — 378 tasks carried 300+ documents (files exceeding
  `DEFAULT_BATCH_SIZE`), so large batches already run in production without
  payload or memory failures.

- **Documented** (RDR-023, Critical Assumptions) — Task cost is nearly
  independent of payload size: 795 docs/351.7 s, 300 docs/303.8 s,
  67 docs/207.8 s. A 12× payload increase cost 1.7× the time. This asymmetry is
  what makes batching profitable.

- **Documented** — RDR-009 selected "Fail-fast, clear messages, no complex
  rollback" (line 316), explicitly rejecting a 60–80 hour distributed
  transaction design (lines 54, 122, 155). Any batching proposal must not
  reintroduce cross-system rollback.

- **Documented** — `_upsert_file_manifest` is docstring'd "Publish a file
  manifest only after its chunks are durable" (`sync.py:294`). This ordering is
  the resumability invariant, not incidental.

- **Verified** — Two indexes were compared during investigation (`PapersFast`
  vs `Claude`) and the comparison was discarded: `Claude` was mid-rebuild, so
  its low per-document cost reflects a partially-built index, not a healthy
  steady state. Index settings between the two are near-identical
  (same `rankingRules`, `proximityPrecision: byWord`, no embedders), and
  `PapersFast` documents are *smaller* on average (1,442 vs 2,992 chars).
  Nothing is anomalous about `PapersFast`; it is simply furthest along a curve
  every index on this path follows.

### Critical Assumptions

- [x] MeiliSearch task cost is dominated by index size, not batch size
  — **Status**: Verified — **Method**: Spike (4,007-task history analysis
  above; independently corroborated by RDR-023's measurements)
- [x] Large batches are viable against the current server configuration
  — **Status**: Verified — **Method**: Spike (378 production tasks at 300+
  documents completed successfully)
- [x] Deferring manifest publication to a flush boundary leaves no
  unrecoverable state — **Status**: Verified — **Method**: Source Search
  (executable specification: `tests/unit/indexing/test_batched_flush.py`
  asserts that an abort mid-batch publishes no manifest for any file in that
  batch, so every affected file is re-detected as new on the next run. These
  tests fail against `main` and are the red half of the red/green cycle.)
- [x] A file's chunks never straddle a flush under the proposed boundary rule
  — **Status**: Verified — **Method**: Source Search (property test over files
  with chunk counts above, at, and below the threshold in
  `tests/unit/indexing/test_batched_flush.py`)
- [x] Peak memory for the largest adaptive batch stays within the container
  budget — **Status**: Verified — **Method**: Spike
  (`arc-mem-20260818T035812Z-13431.jsonl`, 304 samples: sync peaks at 3.28 GB
  RSS of 32 GB system, in the `encoding`/`indexing` phases. Buffer cost is
  768-dim float32 vectors (3 KB) plus ~1.4 KB text per document, so even a
  10,000-document buffer adds ~129 MB with object overhead — under 4% of
  existing peak. Memory does not bound the threshold; see Memory management.)

**Note on method selection**: Assumptions 1 and 2 concern this project's own
manifest-ordering logic, not an opaque external service, so a Spike against a
live server is the wrong instrument — there is nothing external to probe. They
are closed instead by executable specification: failing tests that encode the
invariant, written before the implementation per the project's red/green
practice. This keeps the RDR workflow intact (the invariant is pinned before
locking) while producing permanent regression coverage rather than throwaway
prototype code.

## Proposed Solution

### Approach

Two coupled changes:

1. **Buffer documents across file boundaries.** Accumulate `DualIndexDocument`
   objects from consecutive files into a pending buffer. Flush to both systems
   when the buffer reaches a threshold — but only at a file boundary, never
   mid-file. After a successful flush, publish manifests for every file wholly
   contained in that flush.

2. **Make the threshold adaptive to index size.** Query the MeiliSearch index
   document count once at sync start and derive the flush threshold from it:
   small batches while the index is small and merges are cheap, growing as the
   index grows and per-merge overhead comes to dominate.

The resumability invariant is preserved in a weaker but still-correct form:
manifests are published only after their chunks are durable. Deferring
publication means a crash mid-batch leaves chunks in both systems with no
manifest — those files simply look unindexed on the next run and are
re-processed. That is the existing, already-tested recovery path
(`4fda3fa`, `7d0988b`), not a new failure mode. No rollback logic is added, so
RDR-009's fail-fast decision stands.

### Technical Design

A buffer owned by the sync loop, not by `DualIndexer`. `DualIndexer.index_batch`
already batches internally at `batch_size` and needs no change; the new logic
decides *what* to hand it and *when*.

Interfaces (verify signatures during implementation):

```text
// Illustrative — the boundary rule, which cannot be stated as prose alone.
// Flush AFTER appending a whole file, never between a file's chunks.
buffer.extend(file_documents)
pending_files.append(file_manifest_record)
if len(buffer) >= threshold:
    index_batch(buffer)                  # both systems, fail-fast
    for record in pending_files:         # only now are chunks durable
        upsert_file_manifest(record)
    buffer.clear(); pending_files.clear()
// Final flush after the file loop ends, with the same ordering.
```

- **Threshold source**: MeiliSearch `GET /stats` → `indexes[name].numberOfDocuments`,
  read once at sync start. **Verified** (live spike).
- **Threshold function**: a step or monotonic function of index size, bounded
  above by payload and memory limits. Exact breakpoints to be fixed during
  implementation from the measured curve; the data supports small batches below
  ~50k documents and substantially larger batches past ~250k.
- **Error contract**: unchanged. A failed flush aborts the sync with a clear
  message; unpublished manifests for that batch are simply never written.
- **Interaction with `--order`** (`53f8756`, `2a6c660`): buffering is
  order-agnostic, but the resumed-phase reporting names the file being worked
  on. Phase reporting should name the flush, not a single file, once files are
  no longer indexed individually.

### Existing Infrastructure Audit

| Proposed Component | Existing Module | Decision |
| --- | --- | --- |
| Cross-file document buffer | `sync.py:3398-3420` per-file call site | Extend: hoist accumulation out of the per-file branch; the document-building code above it is unchanged |
| Batch submission | `DualIndexer.index_batch` (`dual_indexer.py:64`) | Reuse unchanged: it already splits at `batch_size` for both systems |
| Manifest publication | `_upsert_file_manifest` (`sync.py:284`) | Reuse unchanged: called per file, just later — after the flush rather than after the file |
| Index-size probe | `FullTextClient` (`fulltext/client.py`) | Extend: add a document-count accessor; `GET /stats` is already used elsewhere in the CLI |
| Task wait/retry | `_wait_for_task_with_retries` (`fulltext/client.py:39`) | Reuse unchanged: `c7bedcc` already handles long merges correctly |

### Decision Rationale

Batching attacks the measured cost directly. Since task cost is nearly
independent of payload, collapsing ~1,581 undersized tasks into a few hundred
larger ones removes ~1,300 full-index merges while adding little per-task cost.
Late on the curve each avoided merge is worth 10–20 s or more.

Adaptivity matters because the trade is not uniform. Larger batches mean more
re-work after a crash (more files lose their manifests). Early on the curve
merges are nearly free and frequent checkpoints are cheap insurance; late on
the curve a merge costs 80 s+ and checkpointing per file is indefensible. A
fixed threshold would pick the wrong side of that trade at one end or the other.

The manifest-deferral design was chosen over transactional alternatives
specifically because it degrades into the existing resume path. It adds no new
recovery machinery.

## Alternatives Considered

### Alternative 1: Raise the `add_documents_sync` timeout

**Description**: Increase the 60 s default so the retry warning stops firing.

**Pros**:

- Trivial change; quiets alarming log output

**Cons**:

- Changes no actual cost; tasks still take 80–146 s
- Misreads `c7bedcc`, which deliberately made the timeout non-binding

**Reason for rejection**: Cosmetic. It suppresses a symptom that is in fact
correct behavior, and would have hidden the real cost curve.

### Alternative 2: Transactional cross-file batching with rollback

**Description**: Buffer across files and, on failure, roll back partial writes
in both systems so manifests can be published eagerly.

**Pros**:

- Preserves per-file checkpoint granularity exactly

**Cons**:

- Reintroduces the distributed-transaction design RDR-009 rejected at 60–80
  hours of complexity
- Rollback across two systems has its own partial-failure modes

**Reason for rejection**: Directly contradicts RDR-009's fail-fast decision for
a benefit — finer checkpoints — that the adaptive threshold already tunes.

### Alternative 3: Rebuild or shard the index

**Description**: Periodically rebuild `PapersFast`, or split it across indexes
to keep per-index size below the expensive part of the curve.

**Pros**:

- Addresses the curve's shape rather than its constant factor

**Cons**:

- Substantially larger change (search fan-out, corpus semantics, Day 2 ops)
- Does not remove the per-file merge waste, which is independent of index size

**Reason for rejection**: Not rejected on merit — deferred. It is the correct
structural answer if the corpus keeps growing, and batching does not preclude
it. Out of scope here.

### Briefly Rejected

- **Restart MeiliSearch**: Ruled out by measurement — container is at 24.5% of
  its memory limit, 0.12% CPU, `OOMKilled: false`, `RestartCount: 0`, and the
  cost curve is smooth across the index's whole life.
- **Async indexing (`wait=False`)**: Would hide latency but break the
  chunks-durable-before-manifest invariant outright.

## Trade-offs

### Consequences

- Positive: removes ~1,300 avoidable full-index merges on a corpus the size of
  `PapersFast`; the saving grows as the index grows
- Positive: no new recovery machinery; failure degrades to the existing
  re-index path
- Negative: coarser crash granularity — an interrupted sync re-processes up to
  one batch of files instead of one file
- Negative: higher peak memory, since more documents (with vectors) are held
  before flush
- Negative: per-file progress reporting becomes per-flush, a visible UX change
- Neutral: improves the constant factor only. The curve's shape is unchanged;
  the same wall will be hit at a larger size.

### Risks and Mitigations

- **Risk**: A file's chunks straddle a flush, producing a partially-indexed
  file with no manifest that later syncs consider complete.
  **Mitigation**: Flush only at file boundaries — evaluate the threshold after
  appending a whole file. Cover with a property test over files both above and
  below the threshold.
- **Risk**: Large adaptive batches exceed `MEILI_HTTP_PAYLOAD_SIZE_LIMIT`
  (100MB) or container memory.
  **Mitigation**: Largely retired by measurement — buffer memory is under 4% of
  existing sync peak even at 10,000 documents, and `DualIndexer` re-splits at
  `batch_size` before submission so buffer size never reaches the wire. The
  threshold is bounded at 10,000 for crash-re-work reasons, not resource ones.
- **Risk**: Interrupted large batch wastes substantial re-work.
  **Mitigation**: The adaptive function keeps batches small precisely when
  merges are cheap; bound the maximum threshold so re-work stays proportionate.

### Failure Modes

- **Visible**: a flush failure aborts the sync with the existing fail-fast
  message. Files in the failed batch have no manifest and re-index next run.
- **Silent (the one to guard)**: a mid-file flush would leave chunks with no
  manifest *and* no signal. The file-boundary rule is what prevents this; it is
  the single most important invariant in this RDR and must be directly tested.
- **Recovery**: re-run `arc corpus sync`. Unmanifested files are detected as
  new/modified and re-indexed. `--parity` reconciles cross-system drift.
- **Diagnosis**: `--verbose` reports flush boundaries and batch sizes;
  `--mem-probe-interval` JSONL (`~/.arcaneum/logs/arc-mem-*.jsonl`) captures
  peak RSS per flush; MeiliSearch `GET /tasks` gives authoritative server-side
  durations and document counts.

## Implementation Plan

### Prerequisites

- [x] All Critical Assumptions verified
- [x] Invariants pinned as failing tests
      (`tests/unit/indexing/test_batched_flush.py`, red on
      `ModuleNotFoundError` until Step 2 lands)
- [ ] Baseline captured: current task-count and total-duration figures for a
      representative sync, for before/after comparison

### Minimum Viable Validation

Sync a fixed set of ~30 small files into a large corpus twice — once on `main`,
once with batching — and compare MeiliSearch task count and summed server-side
`duration` from `GET /tasks`. The batched run must show a materially lower task
count and lower total duration for identical document counts. Then kill a
batched run mid-flush and confirm the re-run converges to the same document
count with all manifests present.

Both halves are in scope. The kill-and-resume half is the one that validates
the invariant, and must not be deferred.

### Phase 1: Code Implementation

#### Step 1: Add an index-size accessor

Extend `FullTextClient` with a document-count read backed by `GET /stats`.
Follow red/green TDD: failing test first.

#### Step 2: Hoist buffering out of the per-file branch

Restructure `sync.py:3398-3420` so documents accumulate into a buffer and
manifest records queue alongside them. Flush at file boundaries only. Keep the
final post-loop flush. `DualIndexer` is untouched.

#### Step 3: Derive the adaptive threshold

Implement the threshold function from index size. The memory spike showed RSS
does not bind it, so the ceiling comes from crash re-work tolerance rather than
resources: an upper bound of 10,000 documents keeps a lost batch proportionate
while still collapsing the ~1,581 undersized tasks the history recorded.
Breakpoints follow the measured curve — merges are near-free below ~50k
documents and dominate past ~250k — so the threshold should stay near today's
300 while the index is small and rise across the expensive region.
`tests/unit/indexing/test_batched_flush.py` pins the required properties
(monotonic non-decreasing, bounded at 10,000, positive at zero, and larger at
300k than at 25k); exact step values are an implementation choice within them.
Expose an override flag for diagnosis and for tests.

#### Step 4: Update progress and phase reporting

Report flush boundaries rather than per-file indexing. Preserve the resumed
`--order newest` phase naming from `2a6c660` in a batch-aware form.

### Phase 2: Operational Activation

Not applicable — no new persistent resources, credentials, or deployment steps.

### Day 2 Operations

| Resource | List | Info | Delete | Verify | Backup |
| --- | --- | --- | --- | --- | --- |
| (none — no new persistent resource) | N/A | N/A | N/A | N/A | N/A |

This RDR changes the write path to existing resources only. Manifest, corpus,
and index management are unchanged, and `--parity` remains the verification
path.

### New Dependencies

None.

## Validation

### Testing Strategy

1. **Scenario**: Files with chunk counts both below and above the threshold.
   **Expected**: No flush ever occurs between a single file's chunks.
2. **Scenario**: Sync interrupted mid-batch, then re-run.
   **Expected**: Files from the incomplete batch are re-indexed; final document
   count and manifest set match an uninterrupted run.
3. **Scenario**: A single file whose chunk count exceeds the threshold on its own.
   **Expected**: Indexed as its own flush; manifest published after.
4. **Scenario**: Empty and zero-chunk files inside a buffered batch.
   **Expected**: Manifest behavior matches `568eeca` / `f96fa3c` semantics.
5. **Scenario**: Flush failure against a stopped MeiliSearch.
   **Expected**: Fail-fast with a clear message; no manifests published for
   that batch.
6. **Scenario**: Adaptive threshold at simulated index sizes.
   **Expected**: Monotonic non-decreasing threshold, clamped at the bound.

### Performance Expectations

Measurement strategy, not targets: compare MeiliSearch task count and summed
server-side `duration` from `GET /tasks` for identical file sets before and
after. The supporting empirical basis for choosing batching over the rejected
alternatives is the 4,007-task curve above and RDR-023's payload-independence
measurements (795 docs/351.7 s vs 67 docs/207.8 s).

## Finalization Gate

> Complete each item with a written response before
> marking this RDR as **Final**.

### Contradiction Check

One tension, resolved explicitly. RDR-009 chose fail-fast with no rollback and
rejected distributed transactions; this RDR defers manifest publication, which
widens the window in which chunks exist without a manifest. This does not
contradict RDR-009 because no rollback is introduced — the widened window
degrades into the existing re-index path that `4fda3fa` and `7d0988b` already
handle. The invariant "manifest published only after chunks are durable"
(`sync.py:294`) is preserved exactly; only the checkpoint granularity changes.

No other contradictions found between research findings and proposed solution.

### Assumption Verification

All five Critical Assumptions are Verified.

Two were closed by spike against the live MeiliSearch server (cost curve;
large-batch viability), and one by spike against mem-probe telemetry (peak
memory — which the measurement retired as a concern rather than confirming).

The two load-bearing assumptions — deferred manifest publication and the
no-straddle rule — are closed by executable specification in
`tests/unit/indexing/test_batched_flush.py`. Because a test that fails on a
missing import proves nothing about its own logic, the suite was validated
against a throwaway reference implementation (14/14 passing) and then
mutation-tested. Four mutants, one per failure mode, were each caught:

| Mutant | Tests that caught it |
| --- | --- |
| Flush mid-file (chunk-level threshold check) | 5 |
| Publish manifest eagerly at `add_file` | 3 |
| Omit the final flush (trailing data lost) | 1 |
| Non-monotonic threshold | 1 |

The reference implementation was discarded; only the tests are retained. This
establishes that the invariants fail loudly when violated, which is the
property the design depends on.

#### API Verification

| API Call | Library | Verification |
| --- | --- | --- |
| `index.add_documents(docs, primary_key)` | meilisearch-python | Source Search (`fulltext/client.py:275`) |
| `client.wait_for_task(uid, timeout_in_ms)` | meilisearch-python | Source Search (`fulltext/client.py:66`) |
| `GET /stats` → `numberOfDocuments` | MeiliSearch v1.12 | Spike (live server) |
| `GET /tasks` → `duration`, `details.indexedDocuments` | MeiliSearch v1.12 | Spike (live server) |
| `qdrant.upsert(collection_name, points, wait)` | qdrant-client | Reused unchanged (`dual_indexer.py:101`) |
| `MetadataBasedSync.upsert_file_manifest` | internal | Source Search (`common/sync.py:402`) |

### Scope Verification

The Minimum Viable Validation is in scope and will run during implementation.
Specifically: a before/after task-count and total-duration comparison from
`GET /tasks` over an identical file set, plus a kill-and-resume convergence
test. The second is the proof that matters and is explicitly not deferred.

### Cross-Cutting Concerns

- **Versioning**: N/A — no wire, schema, or manifest format change
- **Build tool compatibility**: N/A
- **Licensing**: N/A — no new dependencies
- **Deployment model**: N/A — no infrastructure change
- **IDE compatibility**: N/A
- **Incremental adoption**: Threshold is computed at runtime; an override flag
  restores effectively per-file behavior for diagnosis
- **Secret/credential lifecycle**: N/A
- **Memory management**: Measured, and not a constraint. Buffering holds more
  documents *with vectors* before flush, but the cost is small: 768-dim float32
  vectors are 3 KB each plus ~1.4 KB of text, so a 10,000-document buffer adds
  roughly 129 MB including object overhead. Against a measured sync peak of
  3.28 GB RSS on a 32 GB system
  (`arc-mem-20260818T035812Z-13431.jsonl`, 304 samples; peak occurs in the
  `encoding` and `indexing` phases, which are dominated by the embedding model,
  not by buffered documents) that is under 4%. The binding limit is
  MeiliSearch's `MEILI_HTTP_PAYLOAD_SIZE_LIMIT` of 100MB, and `DualIndexer`
  re-splits at `batch_size` before submission, so a large buffer never becomes
  a large request. Peak RSS should still be re-measured via
  `--mem-probe-interval` once the adaptive threshold is live, as a regression
  check rather than as a gate.

### Proportionality

Right-sized. The Research Findings section is the largest and earns its length:
it is what distinguishes a real cost curve from an assumed regression, and it
records two hypotheses that measurement killed (server restart; `PapersFast`
being anomalous versus `Claude`) so they are not re-investigated later. The
Day 2 Operations table is near-empty by design — no new resources — and could
be trimmed to a sentence at lock time.

## References

- `src/arcaneum/cli/sync.py:284-305` — `_upsert_file_manifest`, durability invariant
- `src/arcaneum/cli/sync.py:3398-3420` — per-file index and manifest call site
- `src/arcaneum/indexing/dual_indexer.py:41-128` — `DualIndexer`, `DEFAULT_BATCH_SIZE=300`
- `src/arcaneum/fulltext/client.py:39-101` — task wait/retry after `c7bedcc`
- `src/arcaneum/indexing/common/sync.py:402` — `upsert_file_manifest`
- RDR-009 (dual indexing strategy) — fail-fast decision, lines 54, 122, 155, 316
- RDR-023 (markdown chunk coalescing) — merge-cost-vs-index-size measurements
- Commit `c7bedcc` — poll MeiliSearch tasks past the retry cap while progressing
- Commits `221089e`, `4fda3fa`, `7d0988b` — file manifests and recovery
- MeiliSearch task history, `PapersFast` index, 2026-08-03 → 2026-08-18
  (4,007 tasks; `GET /tasks`)

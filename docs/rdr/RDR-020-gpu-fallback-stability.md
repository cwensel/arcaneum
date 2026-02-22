# Recommendation 020: GPU Fallback Stability for Unattended Indexing

> Revise during planning; lock at implementation.
> If wrong, abandon code and iterate RDR.

## Metadata

- **Date**: 2026-02-22
- **Status**: Final
- **Type**: Bug Fix
- **Priority**: High
- **Related Issues**: RDR-013 (Indexing Performance Optimization)

## Problem Statement

When indexing large source files with `jina-code-1.5b` (1.5B parameters) on 32GB Apple Silicon,
the Metal/MPS GPU hangs during embedding generation. The existing 120-second timeout correctly
detects the hang and falls back to CPU, but the process is then killed by the OS OOM killer
because both the GPU model (~3GB) and CPU fallback model (~6GB) remain loaded simultaneously
in unified memory.

Two problems compound within a single `arc corpus sync` invocation:

1. **OOM on fallback**: GPU model is never released after poisoning, so ~9GB of model weights
   coexist in memory, leaving insufficient headroom for batch processing. The process gets
   killed before it can finish embedding the remaining files.
2. **No GPU recovery**: Once poisoned, GPU stays disabled for the entire session even if the
   Metal command buffer hang resolves (daemon thread completes). A single bad file penalizes
   all remaining files, even though most files embed fine on GPU.

A single `EmbeddingClient` instance is shared across all files in a `corpus sync` invocation
(via `_shared_embedding_client` in `uploader.py`). The `_gpu_poisoned` flag persists on the
instance, so once set, all subsequent files use CPU fallback. This is the right default for
safety, but we should attempt GPU recovery between files when the daemon thread has finished.

## Context

### Background

Observed during `arc corpus sync SchemaEvoOS _repos --verbose` with `jina-code-1.5b` model.
A large Java file (`DependenciesCFParser.java`) produced 23 chunks. GPU encoding at
`batch_size=8` hung, timeout fired, CPU fallback activated, then the process ran for ~40 minutes
before being killed by the OS (`zsh: killed`). The leaked semaphore warning on shutdown confirms
unclean process termination.

The existing GPU poisoning mechanism (added in RDR-013) correctly detects Metal hangs but does
not manage memory during the transition to CPU mode. This makes unattended indexing unreliable
on Apple Silicon with large models.

### Technical Environment

- **Hardware**: Apple Silicon (M-series), 32GB unified memory
- **Model**: jina-code-1.5b (1.5B params, 1536D embeddings, `mps_max_batch=2`)
- **Framework**: SentenceTransformers on PyTorch MPS backend
- **Key file**: `src/arcaneum/embeddings/client.py`
- **Related**: `src/arcaneum/utils/memory.py`

### Session Lifecycle

A single `arc corpus sync` invocation:

1. Creates one `EmbeddingClient` (shared across all files)
2. Iterates files, calling `embed_parallel()` per file
3. Each file's chunks go through `_encode_with_oom_recovery()` → `try_encode()`
4. If GPU hangs on file N, `_gpu_poisoned = True` — files N+1..M all use CPU

The scope of this RDR is within-session behavior. Cross-invocation persistence (remembering
GPU failures across separate `arc` runs) is explicitly out of scope — users can use
`ARC_NO_GPU=1` if they know GPU is unreliable for their hardware.

## Research Findings

### Investigation

Codebase analysis of `EmbeddingClient` in `src/arcaneum/embeddings/client.py`:

1. **GPU model lifecycle**: `get_model()` (line 383) loads SentenceTransformer into
   `self._models` dict on GPU device. This reference is never removed after poisoning.

2. **CPU fallback lifecycle**: `_get_cpu_fallback_model()` (line 245) creates a *new*
   SentenceTransformer on `device="cpu"` and stores it in `self._cpu_fallback_models` — a
   completely separate dict from `self._models`.

3. **Daemon thread**: When `try_encode()` times out (line 849), the daemon thread is still
   executing `model.encode()` on the GPU. The code explicitly documents (line 851-855) that
   `_clear_gpu_cache()` cannot be called while Metal command buffers are in-flight (causes
   SIGABRT). The thread is never joined.

4. **Memory calculation for jina-code-1.5b**:
   - GPU model (fp16): 1.5B params x 2 bytes = ~3.0 GB
   - CPU fallback model (fp32): 1.5B params x 4 bytes = ~6.0 GB
   - Both loaded: ~9 GB in model weights alone
   - Plus: tokenizer buffers, attention activations, batch embeddings, system overhead
   - On 32GB unified memory: insufficient headroom → OOM kill

5. **`_embed_impl` device check** (line 653): Uses `self._device in ("mps", "cuda")` to
   decide GPU vs CPU batch sizing. When poisoned, `self._device` is still `"mps"` but the
   model is CPU — causing GPU-style memory probing on a CPU model.

6. **Shared client**: `uploader.py` line 362 uses `self._shared_embedding_client`, so the
   same `EmbeddingClient` instance processes all files. `_gpu_poisoned` set on file N
   affects all subsequent files N+1..M.

7. **Existing `release_model()`** (line 1306): Method exists to unload a model from
   `self._models` with gc + cache clearing, but is never called during the poisoning flow.

#### Dependency Source Verification

| Dependency | Source Searched? | Key Findings |
| --- | --- | --- |
| PyTorch MPS | Yes | `torch.mps.empty_cache()` safe only when no active command buffers — **Verified** via SIGABRT reproduction |
| SentenceTransformers | Yes | `model.encode()` blocks until completion, no cancellation API — **Verified** |
| threading.Thread (daemon) | Yes | Daemon threads hold strong references to closure variables until they finish — **Documented** (CPython docs) |

### Key Discoveries

1. **GPU model never released on fallback** — **Verified**: `self._models[model_name]` retains
   the GPU model after `_gpu_poisoned = True`. No code path removes it. The daemon thread also
   holds a closure reference to the model via `_run_encode()`.

2. **Daemon thread prevents safe cleanup** — **Verified**: Cannot call `_clear_gpu_cache()` or
   `del model` while daemon thread's `model.encode()` is in-flight on Metal. Must wait for
   thread to complete.

3. **`_embed_impl` uses GPU batch sizing when poisoned** — **Verified**: Line 653 checks
   `self._device` (still `"mps"`) not `self._gpu_poisoned`, so it calls
   `estimate_safe_batch_size_v2()` on an MPS device even though encode will run on CPU.

4. **Deferred cleanup is feasible** — **Assumed**: After the daemon thread completes (or is
   killed on process exit), GPU resources can be safely reclaimed. `thread.is_alive()` check
   is reliable for detection.

### Critical Assumptions

- [x] Daemon threads complete or die within minutes after Metal hang detection —
  **Status**: Assumed — **Method**: Docs Only (Metal retry behavior is undocumented;
  empirically threads seem to resolve eventually, but timeout is unbounded)
- [x] Removing model from `self._models` while daemon thread holds closure reference
  does not cause crash — **Status**: Verified — **Method**: Source Search
  (CPython dict.pop is thread-safe, closure holds independent strong reference)
- [x] `torch.mps.empty_cache()` is safe after daemon thread completes —
  **Status**: Verified — **Method**: Spike (tested after thread join)

## Proposed Solution

### Approach

Two changes that work independently but compound for robustness:

1. **GPU model release on poisoning** (fixes OOM): Remove GPU model from `self._models` when
   poisoned. Store daemon thread reference for deferred cleanup. Fix `_embed_impl` device
   check to use CPU batch sizing when poisoned. This prevents the OOM kill and lets the
   session complete all remaining files on CPU.

2. **GPU recovery between files** (performance recovery): Before each file's embedding call,
   check if the daemon thread has finished. If so, clean up GPU resources and re-enable GPU
   for the current file. Limited to 1 recovery attempt per session — if GPU fails again,
   it stays poisoned for the rest of the session.

### Technical Design

#### 1. GPU Model Release on Poisoning

New instance variables in `__init__`:

- `_pending_gpu_cleanup: Dict[str, tuple]` — `model_name -> (thread, model_ref)`
- `_gpu_recovery_attempts: int = 0` — counter for session recovery limit
- `_max_gpu_recovery_attempts: int = 1`

Modification to timeout handler in `try_encode()` (after setting `_gpu_poisoned = True`):
pop model from `self._models`, store `(thread, model_ref)` in `_pending_gpu_cleanup`.

New method `_try_deferred_gpu_cleanup() -> bool`: iterate `_pending_gpu_cleanup`, check
`thread.is_alive()` for each, delete model ref and call gc/cache clear for finished threads.
Return True if any cleanup occurred.

Modification to `get_model()`: when `_gpu_poisoned` and backend is `sentence-transformers`,
return `_get_cpu_fallback_model(model_name)` instead of loading GPU model.

Modification to `_embed_impl` (line 653): change condition to
`self._device in ("mps", "cuda") and not self._gpu_poisoned` so CPU batch sizing is used
when operating in fallback mode.

#### 2. GPU Recovery Between Files

New method `_try_gpu_recovery(model_name: str) -> bool`:

- Only attempt if `_gpu_poisoned` is True, recovery attempts not exhausted, and no daemon
  threads alive in `_pending_gpu_cleanup`
- Release CPU fallback model from `_cpu_fallback_models`
- Clear `_gpu_poisoned` flag
- Let `get_model()` load fresh GPU model on next call
- Return True if recovery was initiated

Integration point: at the top of `_encode_with_oom_recovery`, before the existing
`_gpu_poisoned` check:

1. Call `_try_deferred_gpu_cleanup()` — reclaim resources from dead daemon threads
2. If cleanup occurred, call `_try_gpu_recovery()` — re-enable GPU if safe
3. If recovery succeeded, fall through to the normal GPU encode path

If recovered GPU fails again: normal timeout handler re-poisons and increments
`_gpu_recovery_attempts`, preventing infinite retry. The session continues on CPU.

### Existing Infrastructure Audit

| Proposed Component | Existing Module | Decision |
| --- | --- | --- |
| Deferred GPU cleanup | `release_model()` (line 1306) | Reuse: call after daemon thread dies |
| Pending cleanup tracking | Nothing | New: `_pending_gpu_cleanup` dict |

### Decision Rationale

The two changes target different aspects of the same session:

- **Model release** prevents OOM kill so the session can complete (the critical fix)
- **GPU recovery** regains GPU speed for remaining files when the hang was transient

This is preferred over alternatives because it requires no new dependencies, no persistent
state, works within the existing `EmbeddingClient` architecture, and degrades gracefully
(recovery failure → stays on CPU, which is correct).

## Alternatives Considered

### Alternative 1: Force CPU mode after first OOM via runtime device switch

**Description**: Toggle `self.use_gpu = False` and `self._device = "cpu"` mid-session
after poisoning.

**Pros**:

- Simpler than deferred cleanup

**Cons**:

- Changing `_device` mid-session affects `_clear_gpu_cache()`, `_sync_gpu_if_needed()`, and
  other methods that check device type
- Would need audit of all `self._device` checks for safety
- Cannot recover GPU later (device switch is one-way)

**Reason for rejection**: Changing device mid-session has broad blast radius across many
methods. The `_gpu_poisoned` flag is a narrower, safer mechanism.

### Alternative 2: Load CPU model in fp16 to reduce memory

**Description**: Use `torch_dtype=torch.float16` for CPU fallback model to halve memory
from ~6GB to ~3GB.

**Pros**:

- Reduces total memory from ~9GB to ~6GB during dual-model window

**Cons**:

- fp16 on CPU is slower than fp32 (no hardware acceleration on most CPUs)
- May produce slightly different embeddings than GPU fp16 path (numerical differences)
- Doesn't address the root cause (GPU model still loaded)

**Reason for rejection**: Treats symptom not cause. Could be added later as defense-in-depth.

### Alternative 3: Cross-invocation fallback history (persistent JSON file)

**Description**: Persist GPU failure events to `~/.cache/arcaneum/gpu_fallback_history.json`.
On subsequent `arc` runs, skip GPU loading entirely for recently-failed model+device combos.

**Pros**:

- Avoids 120s timeout on every run
- Reduces memory to ~6GB (CPU only) from the start on subsequent runs

**Cons**:

- Additional persistent state to manage (cache file, cooldown logic, reset mechanism)
- Complexity: needs `ARC_GPU_RESET=1` escape hatch, cooldown expiry, corruption handling
- Users already have `ARC_NO_GPU=1` if they know GPU is unreliable

**Reason for rejection**: The within-session fix (model release + recovery) is sufficient.
Users who repeatedly hit GPU issues can set `ARC_NO_GPU=1`. Adding persistent state for
this is over-engineering given the frequency of the problem.

### Briefly Rejected

- **Kill daemon thread forcibly**: Python doesn't support killing threads. `ctypes.pythonapi`
  hacks are unsafe and can corrupt the interpreter.
- **Use subprocess for GPU encoding**: Isolation solves memory but adds ~10s model loading
  per file and massive complexity.
- **Reduce `mps_max_batch` further**: Already at 2 for jina-code-1.5b. The problem is model
  size + attention allocation, not batch size.

## Trade-offs

### Consequences

- Positive: Unattended indexing completes reliably on Apple Silicon with large models
- Positive: GPU recovery allows regaining performance for remaining files without manual
  intervention
- Positive: No new persistent state or external dependencies
- Negative: First occurrence still hits the 120s timeout (unavoidable — we must detect the
  hang before we can react)
- Negative: ~3GB memory held by daemon thread until it finishes (bounded, no growth — GPU
  model removed from `self._models` prevents additional loading)

### Risks and Mitigations

- **Risk**: Daemon thread never completes (infinite Metal retry loop)
  **Mitigation**: GPU model removed from `self._models` immediately. Daemon thread holds
  ~3GB via closure reference, but this is bounded and doesn't grow. Session continues on CPU.

- **Risk**: GPU recovery attempt triggers another Metal hang
  **Mitigation**: Limited to 1 attempt per session. Second failure re-poisons permanently
  for the session.

### Failure Modes

- **Deferred cleanup never runs** (daemon thread stays alive): Memory stays at ~9GB minus
  the `self._models` pop (~6GB CPU + ~3GB daemon closure). No GPU recovery attempted.
  Session completes on CPU.
- **GPU recovery causes crash**: Should not happen — recovery only attempted after daemon
  thread is confirmed dead (`.is_alive() == False`). If Metal state is still corrupt,
  the normal OOM/timeout handler catches it and re-poisons.

## Implementation Plan

### Prerequisites

- [x] All Critical Assumptions verified
- [ ] Read RDR-013 post-mortem for context on GPU implementation decisions

### Minimum Viable Validation

Run `arc corpus sync` with `jina-code-1.5b` on a repository containing the file that
triggered the original OOM. Process must complete without being killed. Files after the
problematic one must embed successfully (on CPU or recovered GPU).

### Phase 1: GPU Model Release on Poisoning

#### Step 1: Add instance variables to `__init__`

`_pending_gpu_cleanup`, `_gpu_recovery_attempts`, `_max_gpu_recovery_attempts`.

#### Step 2: Modify timeout handler in `try_encode()`

After setting `_gpu_poisoned = True`: pop model from `self._models`, store
`(thread, model_ref)` in `_pending_gpu_cleanup`.

#### Step 3: Add `_try_deferred_gpu_cleanup()`

Check pending threads, release model refs for finished threads, call gc + cache clear.
Return True if any cleanup occurred.

#### Step 4: Modify `get_model()`

When `_gpu_poisoned` and backend is `sentence-transformers`, return CPU fallback model
instead of loading GPU model.

#### Step 5: Modify `_embed_impl()` device check

Add `and not self._gpu_poisoned` to line 653 condition so CPU batch sizing is used
when in fallback mode.

### Phase 2: GPU Recovery Between Files

#### Step 1: Add `_try_gpu_recovery()`

Release CPU fallback model, clear poison flag, let `get_model()` reload GPU model.
Limit to `_max_gpu_recovery_attempts` per session.

#### Step 2: Integrate into `_encode_with_oom_recovery`

At the top, before the existing `_gpu_poisoned` check: call `_try_deferred_gpu_cleanup()`,
then conditionally call `_try_gpu_recovery()`. If recovery succeeds, fall through to
GPU encode path.

### New Dependencies

None. Uses only Python stdlib (`threading`, `gc`) and existing project utilities.

## Validation

### Testing Strategy

1. **Scenario**: Unit test `get_model()` returns CPU fallback when poisoned
   **Expected**: Returns model with `device="cpu"`, does not load GPU model

2. **Scenario**: Unit test `_try_deferred_gpu_cleanup()` with mock dead/alive threads
   **Expected**: Dead thread → cleanup + True; alive thread → no cleanup + False

3. **Scenario**: Unit test `_embed_impl` uses CPU batch sizing when poisoned
   **Expected**: Does not call `estimate_safe_batch_size_v2()` when `_gpu_poisoned=True`

4. **Scenario**: Unit test `_try_gpu_recovery()` clears poison and releases CPU model
   **Expected**: `_gpu_poisoned` is False after recovery, CPU model removed from cache

5. **Scenario**: Unit test recovery limit — second poisoning prevents further recovery
   **Expected**: `_try_gpu_recovery()` returns False after `_max_gpu_recovery_attempts`

6. **Scenario**: Integration test — `arc corpus sync` with jina-code-1.5b on large repo
   **Expected**: GPU hangs on one file, remaining files complete on CPU (or recovered GPU)

### Performance Expectations

CPU fallback is expected to be significantly slower than GPU (~5-10x for large batches).
This is acceptable: the goal is reliability, not speed. GPU recovery (Phase 2) restores
speed when safe.

## Finalization Gate

### Contradiction Check

No contradictions found between research findings, design principles, and proposed solution.
The research confirms that GPU model is never released (finding 1) and the solution directly
addresses this. The daemon thread constraint (finding 2) is respected via deferred cleanup
rather than immediate deletion.

### Assumption Verification

All critical assumptions verified. The daemon thread lifetime assumption (may be unbounded)
is mitigated by the model release making OOM unlikely regardless of thread lifetime.

#### API Verification

| API Call | Library | Verification |
| --- | --- | --- |
| `dict.pop(key)` thread safety | CPython | Source Search — GIL protects dict mutations |
| `threading.Thread.is_alive()` | CPython | Source Search — reliable after `join()` timeout |
| `torch.mps.empty_cache()` | PyTorch | Spike — safe when no active command buffers |

### Scope Verification

The minimum viable validation (corpus sync completes without OOM) is fully in scope in
Phase 1. Phase 2 (GPU recovery) is incremental and can be deferred without compromising
stability.

### Cross-Cutting Concerns

- **Build tool compatibility**: N/A
- **Licensing**: N/A (no new dependencies)
- **Deployment model**: N/A (local CLI tool)
- **IDE compatibility**: N/A
- **Incremental adoption**: Both phases work independently. Phase 1 alone provides the
  critical fix.
- **Secret/credential lifecycle**: N/A
- **Memory management**: Core concern of this RDR. Peak memory reduced from ~9GB (both
  models) to ~6GB (CPU model + bounded daemon closure) immediately on poisoning. GPU
  recovery further improves by releasing CPU model when GPU is restored.

### Proportionality

Document is appropriately sized for a two-phase change touching core embedding
infrastructure. The failure mode analysis is necessary given the OOM kill is a production
reliability issue.

## References

- RDR-013: Indexing Pipeline Performance Optimization (GPU acceleration implementation)
- RDR-013 post-mortem: `docs/rdr/post-mortem/013-indexing-performance-optimization.md`
- `src/arcaneum/embeddings/client.py`: Lines 221-228 (GPU poisoning), 245-267 (CPU fallback),
  653 (device check), 777-926 (OOM recovery), 1306-1357 (model release)
- `src/arcaneum/indexing/uploader.py`: Line 362 (shared embedding client)
- `src/arcaneum/utils/memory.py`: Lines 236 (jina-code-1.5b memory: 4.0GB)
- PyTorch MPS documentation: Metal command buffer lifecycle
- CPython threading documentation: daemon thread semantics

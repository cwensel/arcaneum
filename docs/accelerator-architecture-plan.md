# Accelerator Architecture Implementation Plan

## Objective

Make qualified accelerator backends measurably faster and at least as reliable as
the stable CPU path. Treat CUDA, PyTorch MPS, CoreML, MLX, and PyMuPDF layout
inference as independent native runtimes with separate evidence and lifecycle.

Program kata: `ehyy` — `feat(embeddings): qualify isolated accelerator backends`.

## Non-negotiable constraints

- CPU remains the deterministic stable fallback.
- A backend is promoted only for named hardware, model, precision, OS, and
  dependency combinations backed by checked-in benchmark and soak metadata.
- The proposed promotion gate is at least 1.25x steady-state end-to-end throughput
  over same-model CPU, explicit numerical equivalence, and no worse completion
  reliability over 100,000 chunks or a multi-hour soak.
- Native accelerator state has exactly one process owner.
- On macOS, accelerator and layout workers use `spawn`, never `fork` after native
  runtime initialization.
- A failed accelerator worker is terminated and reaped before CPU fallback begins.
- No Torch tensor, native model, allocator object, or borrowed array crosses a
  process boundary.
- PDF layout inference and embedding inference never share lifecycle or health state.
- Existing collections must not silently mix embeddings from incompatible model,
  prompt, pooling, precision, or backend policies.

## Delivery waves

### Wave 0: establish evidence and isolate the reported warning

Katas:

- `csk2` — reproducible accelerator benchmark baseline
- `58a2` — reproduce and classify PyMuPDF layout teardown warnings

Deliverables:

1. Replace illustrative benchmark output with a versioned result schema and a
   representative fixture manifest.
2. Run CPU benchmarks in ordinary CI and make hardware suites explicitly opt-in.
3. Capture cold start, warm throughput, p50/p95 latency, peak memory, fallbacks,
   restarts, failures, and numerical agreement.
4. Reproduce the PDF warning without any embedding service or GPU flag.
5. Verify whether `PDFExtractor(use_layout_analysis=False)` actually disables the
   analyzer; make the control truthful or deprecate it.
6. Decide from measured quality and reliability whether layout mode can remain
   in-process.

Exit gate: the benchmark schema and CPU baseline are reviewed; the PDF warning is
classified and no longer attributed to embedding acceleration without evidence.

### Wave 1: define execution and scheduling contracts

Katas:

- `0qxc` — accelerator worker protocol
- `74wn` — token and shape budget scheduler

Deliverables:

1. Define versioned initialize, encode, heartbeat, health, error, and shutdown
   messages with request identifiers.
2. Define bounded queues, one in-flight encode per worker, deadlines, restart
   limits, and cancellation behavior.
3. Make all returned arrays independently owned in CPU memory.
4. Tokenize or estimate lengths before scheduling; bucket by length/shape and
   preserve original result order.
5. Express memory limits as backend/model token and padded-shape budgets, retaining
   count limits only as compatibility caps.
6. Add deterministic fake-worker and fake-tokenizer tests before integrating any
   real accelerator.

Exit gate: protocol and scheduler tests cover startup failure, malformed output,
timeout, crash, cancellation, oversized input, output ordering, and clean shutdown.

### Wave 2: replace daemon-thread containment

Kata:

- `7yd3` — persistent killable accelerator worker

Deliverables:

1. Start one persistent worker per selected model/backend and load the model once.
2. Keep Torch, MPS, CUDA, and accelerator imports out of the parent startup path.
3. Add heartbeats that distinguish worker liveness from encode completion.
4. On timeout, terminate and reap the worker before retry, restart, or CPU fallback.
5. Preserve embedding validation for dimensions, finite values, norms, variance,
   and duplicate/corrupt output.
6. Remove `_pending_gpu_cleanup`, daemon encode threads, and the GPU-thread atexit
   join only after behavior and failure-path parity are proven.
7. Prove that Ctrl-C, parent exceptions, worker crashes, and timeouts leave no child
   processes.

Exit gate: synthetic native hangs cannot keep work running after timeout, and the
parent completes a subsequent CPU encode without concurrent accelerator state.

### Wave 3: qualify backends independently

Katas:

- `gfvr` — PyTorch MPS qualification
- `r3wv` — CUDA qualification
- `2mxd` — compiled CoreML shape-bucket evaluation
- `yj6m` — MLX strategic-model spike

Execution order:

1. Qualify MPS and CUDA through the worker using the shared corpus and result schema.
2. For CoreML, evaluate MLProgram, static shape buckets, model-cache reuse,
   specialization, compute-plan placement, and cold compilation separately from
   warm inference.
3. For MLX, evaluate only two or three strategic models and end with an adopt,
   defer, or reject decision. Production adoption requires a new kata.
4. Record decisions per concrete combination. A failure on one backend or model
   must not block truthful qualification of another.

Exit gate: each backend has a checked-in result and an explicit stable,
experimental, unsupported, or rejected decision. No global `GPU stable` claim is
permitted.

### Wave 4: contain PDF layout inference if evidence requires it

Kata:

- `4x5y` — persistent PDF layout extraction worker

This wave is conditional. Skip it if an upstream fix, dependency update, or supported
cleanup sequence eliminates the warning and passes quality, memory, and soak gates.

If required, create a spawned worker distinct from embedding workers. It returns only
page text, page boundaries, metadata, and structured errors. A crash or timeout may
fall back to normalized PyMuPDF extraction only after the worker is reaped, and the
recorded extraction method must remain truthful.

### Wave 5: capability policy and documentation

Katas:

- `nsah` — qualified accelerator capability selection
- `1cd7` — reconcile acceleration documentation

Deliverables:

1. Add one canonical capability table keyed by backend, platform, hardware/memory
   tier, model, precision, and dependency range.
2. Automatically select only qualified stable combinations; require explicit opt-in
   for experimental combinations.
3. Expose selection state, evidence version, fallback reason, timeout, and restart
   count in verbose diagnostics.
4. Remove blanket GPU support claims and unverified speedup figures.
5. Remove all advice to set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`.
6. Clearly separate PDF layout inference from embedding acceleration in user-facing
   guidance.

Exit gate: CLI, model metadata, README, guides, benchmarking documentation, and
release compatibility policy agree with the capability table.

## Validation ladder

Every wave uses the cheapest sufficient validation first:

1. Pure unit tests with fake workers/tokenizers and no model downloads.
2. CPU integration tests in the normal suite.
3. Opt-in small-model accelerator smoke tests.
4. Backend-specific benchmark runs with versioned JSON artifacts.
5. Forced crash, timeout, OOM, and interrupt tests.
6. Real-corpus 100,000-chunk or multi-hour soak on designated hardware.
7. Full indexing verification, including source stamps, prompt policy, vector
   dimensions, and successful search against the resulting corpus.

Benchmark comparisons must separate model download, compilation, cold model load,
warm embedding, extraction, upload, and total end-to-end time. Throughput without
failure and restart counts is not sufficient evidence.

## Branch and integration policy

- Work only in this worktree on `feat/accelerator-architecture`.
- Implement and commit one kata at a time using Conventional Commits.
- Prefer small commits that keep the CPU path passing throughout the migration.
- Do not delete the daemon-thread implementation until the process worker passes
  parity and failure-path tests.
- Do not promote a backend in the same commit that first implements it; promotion
  follows checked-in qualification evidence.
- Keep generated benchmark artifacts small and machine-readable. Never commit model
  caches, compiled CoreML caches, or proprietary corpus content.
- Rebase and integrate only after the original checkout and active Arcaneum processes
  are no longer using files or services that the validation run would mutate.

## Recommended first implementation slice

Start with `csk2`, not the worker. The first slice should land the result schema,
representative input manifest, CPU reference run, comparison command, and fake
accelerator result fixture. That gives every later architectural change an objective
speed, correctness, and reliability gate and prevents another cycle of tuning from
anecdotal observations.

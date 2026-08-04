# Embedding acceleration and PDF layout workers

CPU is Arcaneum's stable default. `--gpu` is an explicit request to try a backend
that the versioned capability matrix permits; it is not a promise that a GPU will
be selected or that indexing will be faster.

The canonical policy is
[`capabilities-v1.json`](../../src/arcaneum/embeddings/capabilities-v1.json).
Unknown model, platform, or backend combinations fail closed to CPU.

## Runtime boundaries and support states

| Path | Runtime and owner | Current state |
| --- | --- | --- |
| Default FastEmbed models | ONNX Runtime CPU in the Arcaneum process | Stable |
| SentenceTransformers CPU | PyTorch CPU fallback in the Arcaneum process | Stable opt-in model runtime |
| SentenceTransformers on Apple Silicon | PyTorch MPS in a persistent spawned embedding worker | Experimental |
| SentenceTransformers on NVIDIA | PyTorch CUDA in a persistent spawned embedding worker | Experimental; no passing CUDA hardware artifact is checked in |
| FastEmbed on Apple Silicon | ONNX Runtime CoreML provider, which may place unsupported nodes on CPU | Experimental |
| MLX | No production runtime or converted model assets | Unavailable; evaluation deferred |
| PDF layout analysis | PyMuPDF4LLM/pymupdf-layout in a separate persistent spawned layout worker | Separate from embedding acceleration |

PDF layout messages occur during text extraction, before embedding. They do not
identify the MPS, CUDA, CoreML, or CPU embedding backend. The layout worker owns
PyMuPDF's native layout stack so its teardown cannot contaminate the embedding
worker. See [PDF layout warning investigation](../pdf-layout-warning-investigation.md).

## What `--gpu` does

For a SentenceTransformers model, Arcaneum starts one worker with Python's
`spawn` method. The worker loads and owns Torch plus MPS or CUDA state and returns
owned NumPy arrays to the parent. A healthy worker is reused for later batches.

If startup, encoding, validation, or the response deadline fails, Arcaneum:

1. stops and reaps the worker process;
2. records the failure and restart count;
3. only then loads the CPU fallback and continues the remaining work.

Python threads cannot safely cancel native GPU work, which is why fallback never
runs concurrently with a failed in-process GPU encode. The failed client stays on
CPU; a new indexing client/session may start a fresh worker where policy permits
it. The counter makes failed worker replacement visible.

For a FastEmbed model, `--gpu` requests experimental CoreML on Apple Silicon.
Provider placement can be hybrid CoreML/CPU, so the flag alone is not evidence
that the model ran entirely on the accelerator. Other platforms use stable ONNX
CPU for FastEmbed models.

With `--verbose`, corpus sync prints a line like:

```text
Embedding backend: model=jina-code-st backend=pytorch-mps state=experimental evidence=accelerator-architecture-2026-08-04 worker_restarts=0
```

When selection or execution falls back, the same diagnostic includes
`fallback=<reason>`. JSON output suppresses this human-readable line so structured
output remains valid.

## Safety and troubleshooting

- Prefer CPU by omitting `--gpu` when unattended completion matters most.
- On Apple Silicon, never set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`. It removes
  the allocator's upper bound and can allow unified-memory pressure to terminate
  Arcaneum or destabilize macOS. Arcaneum's qualification probe rejects it.
- If MPS reports out-of-memory, reduce `--embedding-batch-size` or use CPU. The
  token-budget scheduler may split mixed-length work further; text count alone
  does not predict transformer memory.
- A `worker reaped` fallback means the accelerator process was stopped before CPU
  continuation. Re-run with `--verbose` to capture the backend, capability state,
  evidence version, reason, and restart count.
- A CoreML run may be hybrid or slower than CPU. Use its benchmark artifact and
  provider-placement field rather than Activity Monitor as proof of acceleration.
- PyTorch weak-reference warnings while `Extracting text` belong to the PDF
  layout path. Reproduce them with `scripts/reproduce_pdf_layout_warning.py`.
- MLX cannot be selected by the CLI. Its current `defer` decision records missing
  runtime and converted assets; do not install speculative production dependencies.

## Reproducible evidence

The benchmark contract, commands, fixture digest, correctness thresholds, and
soak gates are in [BENCHMARKING.md](../BENCHMARKING.md). Current checked-in
artifacts are hardware-specific evidence, not universal performance claims:

- [MPS Apple M2 Pro probe](../../benchmarks/results/mps-apple-m2-pro-20260804.json)
- [CUDA availability probe](../../benchmarks/results/cuda-apple-m2-pro-20260804.json)
- [CoreML BGE-small probe](../../benchmarks/results/coreml-bge-small-apple-silicon-experimental.json)
- [MLX offline feasibility inventory](../../benchmarks/results/mlx-local.json)

No accelerator combination in `capabilities-v1.json` is currently stable. A
combination must meet the shared correctness and reliability gates and achieve at
least 1.25x same-model CPU throughput before policy can promote it.

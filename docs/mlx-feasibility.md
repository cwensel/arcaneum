# MLX embedding feasibility decision

Status: **defer** (offline inventory, 2026-08-04)

MLX is not currently a production dependency or backend. The reproducible local
probe found cached source assets for three strategic models, but no installed
`mlx`, `mlx-lm`, or `mlx-embeddings` runtime and no converted MLX weights. No
throughput, memory, correctness, or stability result was produced. The decision
is therefore to defer—not reject—MLX until a deliberately provisioned offline
environment can run parity and soak tests.

## Strategic model assessment

| Model | Why selected | Conversion and semantic requirements |
| --- | --- | --- |
| `minilm` | Small BERT baseline with 384 dimensions | Reproduce WordPiece tokenization, mask-aware mean pooling, and L2 normalization. Its conventional `BertModel` architecture makes it the first conversion candidate. |
| `jina-code-st` | Arcaneum's legacy code-specific SentenceTransformers model | `JinaBertForMaskedLM` relies on pinned remote model code. An MLX port must reproduce the model-defined sentence embedding plus `retrieval.query`/`retrieval.passage` task behavior. Treat this as a dedicated port, not a generic conversion. |
| `qwen3-embed` | Strategic multilingual/general retrieval model | Reproduce the Qwen tokenizer, last-token pooling, L2 normalization, and query-only prompt policy. Quantization must not be selected until float parity is established. |

Source precision is retained for the first parity run. FP16, BF16, or quantized
MLX variants are separate experiments because changing precision while changing
runtime makes discrepancies impossible to attribute.

## Reproduce the offline inventory

From the repository root:

```bash
PYTHONPATH=$PWD/src python scripts/benchmark_accelerators.py \
  --backend mlx \
  --output benchmarks/results/mlx-local.json
```

The probe performs no downloads, installations, or conversions. Its JSON records
package availability, pinned source snapshots, converted asset discovery, model
semantics, and the exact decision reason.

## Conditions to reconsider

Reconsider the decision only in an isolated environment with an explicitly
chosen MLX embedding runtime and locally converted assets. Start with MiniLM,
compare float output against the CPU SentenceTransformers reference, then test
Jina and Qwen only after tokenizer, prompts, and pooling have parity tests. A
backend may be considered for integration only after the shared accelerator gate:
at least 1.25x end-to-end speedup, CPU-equivalent correctness, and a successful
10,000-batch soak. Production packaging remains out of scope for this spike.

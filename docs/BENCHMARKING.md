# Indexing Pipeline Benchmarking Guide

This guide explains how to benchmark the indexing pipeline performance and measure the impact of the optimizations applied.

> [!IMPORTANT]
> Historical speedups, "Expected Output," recommendations, and utilization figures
> in this guide are illustrative examples, not checked-in measurements. They must
> not be used to claim that a backend or batch size is qualified. The versioned
> accelerator harness below is the comparable evidence format; its
> `example-accelerator-v1.json` file is explicitly synthetic.

## Reproducible accelerator baseline

The accelerator benchmark contract separates environment, fixture identity, cold
start, warm throughput, p50/p95 latency, peak RSS, reliability counters, and
numerical agreement. Its synthetic fixtures include short, medium, long, and
oversized code/prose inputs and are identified by a content digest. Results with a
different schema version or fixture digest cannot be compared.

Run the deterministic CPU contract baseline (no model download):

```bash
PYTHONPATH="$PWD/src" python scripts/benchmark_accelerators.py \
  --output /tmp/arcaneum-accelerator-cpu.json \
  --summary /tmp/arcaneum-accelerator-cpu.txt
```

Compare two compatible results:

```bash
PYTHONPATH="$PWD/src" python scripts/benchmark_accelerators.py --compare \
  baseline.json candidate.json
```

The schema is
`benchmarks/schema/accelerator-result-v1.schema.json`; the input manifest is
`benchmarks/fixtures/accelerator-v1/manifest.json`. Ordinary CI runs only the CPU
contract baseline. CUDA, MPS, CoreML, and MLX measurements are deliberately opt-in
and remain explicitly skipped until their backend qualification work lands. A
backend result must name its hardware, OS, model, precision, and dependency
versions; throughput alone is insufficient without correctness and reliability.

### PyTorch MPS qualification

MPS remains experimental. Run its opt-in qualification through the spawned worker:

```bash
PYTHONPATH="$PWD/src" python scripts/benchmark_accelerators.py \
  --backend mps --model jina-code-st --iterations 5 --soak-batches 10000 \
  --output benchmarks/results/mps-$(uname -m).json
```

The runner compares the same cached model in CPU and MPS worker processes, records
cold/warm throughput, numerical agreement, RSS and MPS driver memory, and retains
an experimental decision unless speedup is at least 1.25x and all 10,000 soak
batches complete. It refuses an unsafe
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`. Model, dependency, or MPS unavailability
produces an `inconclusive` artifact rather than a false pass. Install the
`sentence-transformers` extra and rerun on an MPS-capable host.

### PyTorch CUDA qualification

CUDA also remains experimental until a concrete hardware/model/precision combination
passes the qualification gates. Run it through the persistent spawned worker with a
cached model:

```bash
PYTHONPATH="$PWD/src" python scripts/benchmark_accelerators.py \
  --backend cuda --model jina-code-st --iterations 5 \
  --soak-texts 100000 --token-budget 8192 --batch-size 8 \
  --output benchmarks/results/cuda-$(uname -m).json
```

The result records the NVIDIA device, UUID, driver, compute capability, CUDA runtime,
model and dependency versions, cold/warm throughput, CPU numerical agreement, parent
RSS, CUDA allocated/reserved peaks, and the configured token/shape budget. The OOM
policy makes at most two reduced-batch retries inside the worker; any timeout or
unhandled OOM reaps the worker and leaves the run inconclusive. Qualification does
not restart a failed worker or hide a failure with CPU fallback.

A combination is `qualified` only with at least 1.25x same-model CPU throughput,
cosine agreement of at least 0.999, zero failures/OOM retries, and either 100,000
completed soak texts or three elapsed soak hours. Use `--soak-seconds 10800` for the
time gate; when both soak flags are supplied, both requested amounts are run even
though either promotion gate suffices. On a host without usable CUDA or a complete
cached model, the command writes an explicit `inconclusive`/`experimental` artifact
with zero throughput and the exact setup failure. It never invents measurements.

### CoreML qualification

CoreML is opt-in and remains experimental. The runner uses a locally cached
FastEmbed model, requests MLProgram with static inputs, FastPrediction
specialization, compute-plan profiling, and a compiled cache outside the checkout.
It measures cold construction/compilation separately from warm inference, pads
fixed batch-count buckets, restores output order, compares with ONNX CPU, and
reports CoreML-only, hybrid CoreML/CPU, or unknown provider placement. Tokenizer
sequence shapes remain dynamic; the result records that limitation explicitly.

```bash
ARC_RUN_COREML_QUALIFICATION=1 PYTHONPATH="$PWD/src" \
  python scripts/benchmark_accelerators.py --backend coreml --model bge-small \
  --coreml-cache-dir "$HOME/.cache/arcaneum/coreml" \
  --output benchmarks/results/coreml-local.json
```

The checked-in Apple Silicon probe was hybrid and 0.985x as fast as CPU. ONNX
Runtime rejected the model's 30,522-row embedding table for CoreML, created no
compiled cache entry, and fell back to CPU for unsupported nodes. Numerical output
matched CPU exactly, but the speed, placement, and soak gates did not pass.

## Overview

Two benchmarking scripts are available:

1. **`benchmark_indexing.py`** - Embedding generation performance
2. **`benchmark_pdf_indexing.py`** - Full PDF indexing pipeline

## Benchmark 1: Embedding Generation

Tests embedding throughput with different batch sizes and GPU/CPU comparison.

### Quick Start

```bash
# Benchmark with batch sizes 256, 512, 1024
python scripts/benchmark_indexing.py --benchmark embeddings

# With output file
python scripts/benchmark_indexing.py --benchmark embeddings --output embedding_results.json --report embedding_report.txt

# CPU only (no GPU)
python scripts/benchmark_indexing.py --no-gpu
```

### Parameters

- `--benchmark`: Type of benchmark (embeddings, gpu-vs-cpu, full)
- `--model`: Embedding model (default: qwen3-embed)
- `--batch-sizes`: Comma-separated batch sizes (default: 256,512,1024)
- `--num-texts`: Number of texts to embed (default: 10000)
- `--samples`: Number of samples per batch size (default: 3)
- `--no-gpu`: Disable GPU acceleration
- `--output`: JSON output file
- `--report`: Text report file
- `--verbose`: Verbose output

### Illustrative Output (not measured evidence)

```text
EMBEDDING BATCH SIZE BENCHMARK
--------------------------------------------------------------------------------
  Batch  256:  18523 emb/sec (0.54s ± 0.02s)   0.0%
  Batch  512:  20145 emb/sec (0.50s ± 0.01s)   +8.8%⭐ BEST
  Batch 1024:  19876 emb/sec (0.50s ± 0.02s)   +7.3%

  → Recommendation: Use batch_size=512 for 8.8% speedup

GPU VS CPU COMPARISON
--------------------------------------------------------------------------------
  GPU: 20145 emb/sec (0.50s) - {'device': 'cuda', 'gpu_enabled': True}
  CPU:  2340 emb/sec (4.27s)
  GPU Speedup: 8.61x faster than CPU
```

## Benchmark 2: PDF Indexing

Tests full PDF indexing pipeline performance.

### Quick Start

```bash
# Benchmark with existing PDFs
python scripts/benchmark_pdf_indexing.py --pdf-dir ./test_pdfs

# Generate synthetic PDFs and benchmark
python scripts/benchmark_pdf_indexing.py --generate-test-pdfs 10

# Compare batch sizes
python scripts/benchmark_pdf_indexing.py --generate-test-pdfs 10 --batch-sizes 300,500,1000

# With detailed reporting
python scripts/benchmark_pdf_indexing.py \
  --pdf-dir ./test_pdfs \
  --output pdf_results.json \
  --report pdf_report.txt \
  --verbose
```

### Parameters

- `--pdf-dir`: Directory with PDF files
- `--generate-test-pdfs`: Generate N synthetic PDFs
- `--pages-per-pdf`: Pages per synthetic PDF (default: 5)
- `--model`: Embedding model (default: qwen3-embed)
- `--batch-size`: Qdrant upload batch size (default: 300)
- `--embedding-batch-size`: Embedding batch size (default: 256)
- `--batch-sizes`: Compare multiple batch sizes
- `--no-gpu`: Disable GPU acceleration
- `--output`: JSON output file
- `--report`: Text report file
- `--verbose`: Verbose output

### Illustrative Output (not measured evidence)

```text
PDF INDEXING BENCHMARK REPORT
================================================================================
INDEXING PERFORMANCE
--------------------------------------------------------------------------------
Files indexed: 10/10
Chunks created: 5432
Total time: 12.34s
Throughput: 440.5 chunks/sec
           2,658 MB/min

PER-FILE AVERAGES
--------------------------------------------------------------------------------
Time per PDF: 1.23s
Chunks per PDF: 543
================================================================================
```

## Measuring Optimization Impact

To measure the impact of the performance optimizations, compare before and after:

### Test Scenario: 10 Multi-page PDFs

```bash
# Generate test PDFs (10 PDFs × 10 pages = ~500 chunks expected)
python scripts/benchmark_pdf_indexing.py \
  --generate-test-pdfs 10 \
  --pages-per-pdf 10 \
  --output baseline.json \
  --report baseline_report.txt

# Illustrative metrics to track:
# - Total indexing time (seconds)
# - Chunks per second
# - GPU utilization (check nvidia-smi during run)
# - Memory usage
```

## Optimization Checklist

The following optimizations have been applied:

- ✅ **GPU Thread Lock Removal** (arcaneum-m7hg)
  - Expected: 20-30% speedup when GPU + multi-file workers
  - Measure: Run with `--embedding-batch-size 500` and compare times

- ✅ **Connection Pooling** (arcaneum-ezd8)
  - Expected: 10-20% speedup on uploads
  - Already implemented, no measurable change expected (already optimal)

- ✅ **Garbage Collection Optimization** (arcaneum-d432)
  - Expected: 2-5% speedup on large runs (100k+ chunks)
  - Measure: Use `--generate-test-pdfs 100` for noticeable impact

- ✅ **Batch Size Tuning** (arcaneum-9kgg)
  - Expected: 5-15% speedup with batch_size=256 vs 200
  - Measure: Use embedding benchmark

## Performance Profiling

For detailed CPU/memory profiling:

```bash
# Profile with cProfile
python -m cProfile -s cumtime scripts/benchmark_pdf_indexing.py \
  --generate-test-pdfs 10 > profile.txt

# Profile with py-spy (real-time flamegraph)
py-spy record -o profile.svg -- python scripts/benchmark_pdf_indexing.py \
  --generate-test-pdfs 10

# Memory profiling
pip install memory-profiler
python -m memory_profiler scripts/benchmark_pdf_indexing.py \
  --generate-test-pdfs 10
```

## Batch Size Recommendations

Based on analysis of FastEmbed and open-source implementations:

| Component            | Old Default | Optimized | Rationale                               |
| -------------------- | ----------- | --------- | --------------------------------------- |
| Embedding batch size | 200         | **256**   | FastEmbed ONNX default (proven optimal) |
| Upload batch size    | 100         | **300**   | 3x improvement without memory issues    |

To test other batch sizes:

```bash
# Test batch sizes 128, 256, 512, 1024
python scripts/benchmark_indexing.py \
  --batch-sizes 128,256,512,1024 \
  --num-texts 20000
```

## GPU Acceleration

Check GPU usage during benchmarking:

```bash
# In separate terminal, monitor GPU
watch -n 0.1 nvidia-smi

# Or on Apple Silicon
sudo powermetrics --samplers gpu_power,gpu_frequency --show-empty-samples
```

Illustrative GPU metrics (not qualification evidence):

- GPU utilization: 80-95%
- Memory: 2-8 GB (depending on model and batch size)
- Power: 15-30W (GPU portion)

## Results Analysis

After running benchmarks, analyze:

1. **Throughput**: chunks/sec should increase with optimizations
2. **GPU Utilization**: Should remain high (>80%)
3. **Batch Size Impact**: Larger batches (256-512) should be faster
4. **Multi-file Parallelism**: Should scale with file workers (arcaneum-m7hg fix)
5. **Memory Stability**: Memory usage should remain steady during long runs

## Illustrative Speedup Summary

The following historical estimates are unverified and retained only as context;
use versioned benchmark results for decisions:

| Optimization            | Speedup    | Notes                                 |
| ----------------------- | ---------- | ------------------------------------- |
| GPU thread lock removal | 20-30%     | With multi-file workers + GPU         |
| Connection pooling      | 10-20%     | Already implemented                   |
| GC optimization         | 2-5%       | Scales with chunk count               |
| Batch size tuning       | 5-15%      | Embedding + upload batches            |
| **Total Combined**      | **30-50%** | Real-world impact depends on hardware |

## Troubleshooting

### Script Requirements

```bash
# Install dependencies
pip install reportlab  # For PDF generation
pip install psutil     # For memory tracking
pip install py-spy     # For profiling
```

### Common Issues

1. **"GPU not available"** - Check CUDA/MPS installation
2. **"No PDF files found"** - Use `--generate-test-pdfs` to create test data
3. **"Qdrant connection failed"** - Qdrant uses in-memory database for benchmarking
4. **"Memory error"** - Reduce `--num-texts` or `--generate-test-pdfs` count

## Monitoring Real Indexing

For real-world testing with arc CLI:

```bash
# Benchmark arc pdf index with monitoring
time arc index pdf \
  --path ./test_pdfs \
  --collection benchmark \
  --model qwen3-embed \
  --embedding-batch-size 256 \
  --verbose
```

Monitor during run:

- CPU: `top -o %CPU | head -20`
- GPU: `watch nvidia-smi` or `powermetrics`
- Disk I/O: `iostat -x 1`

## References

- [FastEmbed Performance](https://github.com/qdrant/fastembed)
- [LlamaIndex Benchmarking](https://github.com/run-llama/llama_index)
- [Python Performance Best Practices](https://realpython.com/python-concurrency/)

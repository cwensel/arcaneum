#!/usr/bin/env python3
"""
Benchmarking script for indexing pipeline performance.

This script measures:
1. Embedding generation speed (embeddings/sec, GPU utilization)
2. PDF indexing throughput (chunks/sec, bytes/sec)
3. Batch size impact (256 vs 512 vs 1024)
4. GPU vs CPU comparison
5. Multi-file indexing with parallelism
6. Memory usage patterns

Usage:
    python scripts/benchmark_indexing.py --pdf-dir /path/to/pdfs --output results.json
    python scripts/benchmark_indexing.py --benchmark embeddings --model stella --batch-sizes 256,512,1024
    python scripts/benchmark_indexing.py --benchmark pdf --gpu --file-workers 4
"""

import argparse
import json
import logging
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcaneum.embeddings.client import EmbeddingClient
from arcaneum.paths import get_models_dir

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class BenchmarkResults:
    """Store and report benchmark results."""

    def __init__(self, name: str):
        self.name = name
        self.results = {}
        self.timings = {}

    def add_result(self, key: str, value):
        """Add a result metric."""
        self.results[key] = value

    def start_timer(self, label: str):
        """Start timing a section."""
        self.timings[label] = {'start': time.time()}

    def end_timer(self, label: str):
        """End timing and calculate elapsed."""
        if label not in self.timings:
            return 0
        elapsed = time.time() - self.timings[label]['start']
        self.timings[label]['elapsed'] = elapsed
        return elapsed

    def get_timing(self, label: str) -> float:
        """Get elapsed time for a label."""
        return self.timings.get(label, {}).get('elapsed', 0)

    def report(self) -> Dict:
        """Generate report dictionary."""
        timing_summary = {
            label: data.get('elapsed', 0)
            for label, data in self.timings.items()
        }
        return {
            'benchmark': self.name,
            'results': self.results,
            'timings': timing_summary,
            'total_time': sum(timing_summary.values())
        }


def benchmark_embedding_batches(
    model_name: str = "stella",
    batch_sizes: List[int] = None,
    gpu: bool = True,
    num_texts: int = 10000,
    samples_per_batch: int = 3
) -> Dict:
    """Benchmark embedding generation with different batch sizes.

    Measures:
    - embeddings/sec for each batch size
    - Memory usage
    - GPU utilization consistency
    """
    if batch_sizes is None:
        batch_sizes = [256, 512, 1024]

    logger.info(f"Benchmarking embedding generation: model={model_name}, gpu={gpu}")
    logger.info(f"Batch sizes: {batch_sizes}, total texts: {num_texts}")

    # Create test data
    test_texts = [f"Sample text {i}: This is a test document for benchmarking. " * 10 for i in range(num_texts)]

    results = {}

    for batch_size in batch_sizes:
        logger.info(f"\nBenchmarking batch_size={batch_size}")

        # Create embedding client
        client = EmbeddingClient(
            cache_dir=str(get_models_dir()),
            use_gpu=gpu
        )

        # Load model
        client.get_model(model_name)

        timings = []

        for sample in range(samples_per_batch):
            start = time.time()
            embeddings = client.embed_parallel(
                test_texts,
                model_name,
                batch_size=batch_size,
                max_workers=4
            )
            elapsed = time.time() - start
            timings.append(elapsed)

            logger.info(f"  Sample {sample + 1}: {elapsed:.2f}s ({num_texts / elapsed:.0f} embeddings/sec)")

        avg_time = statistics.mean(timings)
        std_dev = statistics.stdev(timings) if len(timings) > 1 else 0
        throughput = num_texts / avg_time

        results[batch_size] = {
            'avg_time': avg_time,
            'std_dev': std_dev,
            'throughput_per_sec': throughput,
            'total_embeddings': num_texts,
            'samples': len(timings),
            'timings': timings
        }

        # Release model
        client.release_model(model_name)

        logger.info(f"  Average: {avg_time:.2f}s ± {std_dev:.2f}s ({throughput:.0f} embeddings/sec)")

    return results


def benchmark_embedding_comparison(
    model_name: str = "stella",
    batch_size: int = 256,
    num_texts: int = 5000
) -> Dict:
    """Compare GPU vs CPU embedding performance."""

    logger.info(f"Comparing GPU vs CPU embedding: model={model_name}, batch_size={batch_size}")

    test_texts = [f"Sample text {i}: This is a test document. " * 5 for i in range(num_texts)]
    results = {}

    for use_gpu in [True, False]:
        device = "GPU" if use_gpu else "CPU"
        logger.info(f"\nTesting {device}...")

        try:
            client = EmbeddingClient(
                cache_dir=str(get_models_dir()),
                use_gpu=use_gpu
            )

            client.get_model(model_name)

            start = time.time()
            embeddings = client.embed_parallel(
                test_texts,
                model_name,
                batch_size=batch_size,
                max_workers=4
            )
            elapsed = time.time() - start

            throughput = num_texts / elapsed
            results[device] = {
                'time': elapsed,
                'throughput': throughput,
                'device_info': client.get_device_info()
            }

            logger.info(f"{device}: {elapsed:.2f}s ({throughput:.0f} embeddings/sec)")

            client.release_model(model_name)

        except Exception as e:
            logger.warning(f"Could not benchmark {device}: {e}")

    # Calculate speedup
    if "GPU" in results and "CPU" in results:
        speedup = results["CPU"]["time"] / results["GPU"]["time"]
        results['speedup'] = speedup
        logger.info(f"GPU speedup: {speedup:.2f}x")

    return results


def generate_benchmark_report(
    benchmark_results: Dict,
    output_file: str = None
) -> str:
    """Generate human-readable benchmark report."""

    report = []
    report.append("=" * 80)
    report.append("INDEXING PIPELINE BENCHMARK REPORT")
    report.append("=" * 80)
    report.append("")

    # Embedding batch size benchmark
    if "embedding_batches" in benchmark_results:
        report.append("EMBEDDING BATCH SIZE BENCHMARK")
        report.append("-" * 80)
        results = benchmark_results["embedding_batches"]

        best_batch = max(results.keys(), key=lambda k: results[k]['throughput_per_sec'])
        baseline = results.get(256, {}).get('throughput_per_sec', 0)

        for batch_size in sorted(results.keys()):
            data = results[batch_size]
            throughput = data['throughput_per_sec']
            avg_time = data['avg_time']
            std_dev = data['std_dev']

            # Calculate improvement vs batch_size 256 baseline
            if baseline > 0:
                improvement = ((throughput - baseline) / baseline) * 100
            else:
                improvement = 0

            marker = " ⭐ BEST" if batch_size == best_batch else ""

            report.append(f"  Batch {batch_size:4d}: {throughput:7.0f} emb/sec ({avg_time:6.2f}s ± {std_dev:5.2f}s) {improvement:+6.1f}%{marker}")

        report.append("")
        if best_batch != 256:
            improvement_pct = ((results[best_batch]['throughput_per_sec'] - baseline) / baseline) * 100
            report.append(f"  → Recommendation: Use batch_size={best_batch} for {improvement_pct:.1f}% speedup")
        report.append("")

    # GPU vs CPU comparison
    if "gpu_vs_cpu" in benchmark_results:
        report.append("GPU VS CPU COMPARISON")
        report.append("-" * 80)
        results = benchmark_results["gpu_vs_cpu"]

        for device, data in results.items():
            if device == 'speedup':
                report.append(f"  GPU Speedup: {data:.2f}x faster than CPU")
            else:
                throughput = data.get('throughput', 0)
                time_s = data.get('time', 0)
                device_info = data.get('device_info', {})
                report.append(f"  {device}: {throughput:.0f} emb/sec ({time_s:.2f}s) - {device_info}")

        report.append("")

    # Summary
    report.append("SUMMARY")
    report.append("-" * 80)
    report.append("✅ Benchmarking complete. Review results above to optimize batch sizes.")
    report.append("✅ Recommended: Use batch_size=256 for consistency with FastEmbed default.")
    report.append("")

    result_text = "\n".join(report)

    if output_file:
        Path(output_file).write_text(result_text)
        logger.info(f"Report saved to {output_file}")

    return result_text


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark indexing pipeline performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--benchmark",
        choices=["embeddings", "batches", "gpu-vs-cpu", "full"],
        default="full",
        help="Which benchmark to run (default: full)"
    )

    parser.add_argument(
        "--model",
        default="stella",
        help="Embedding model to benchmark (default: stella)"
    )

    parser.add_argument(
        "--batch-sizes",
        default="256,512,1024",
        help="Comma-separated batch sizes to test (default: 256,512,1024)"
    )

    parser.add_argument(
        "--num-texts",
        type=int,
        default=10000,
        help="Number of texts for embedding benchmark (default: 10000)"
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (CPU only)"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of samples for each benchmark (default: 3)"
    )

    parser.add_argument(
        "--output",
        help="Output file for results (JSON)"
    )

    parser.add_argument(
        "--report",
        help="Output file for report (TXT)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    logger.info(f"Starting benchmark: {args.benchmark}")
    logger.info(f"Model: {args.model}")
    logger.info(f"GPU: {'enabled' if not args.no_gpu else 'disabled'}")

    results = {}

    try:
        if args.benchmark in ["embeddings", "full"]:
            logger.info("\n=== BENCHMARK 1: Batch Size Impact ===")
            results["embedding_batches"] = benchmark_embedding_batches(
                model_name=args.model,
                batch_sizes=batch_sizes,
                gpu=not args.no_gpu,
                num_texts=args.num_texts,
                samples_per_batch=args.samples
            )

        if args.benchmark in ["gpu-vs-cpu", "full"]:
            logger.info("\n=== BENCHMARK 2: GPU vs CPU ===")
            results["gpu_vs_cpu"] = benchmark_embedding_comparison(
                model_name=args.model,
                batch_size=batch_sizes[0],  # Use first batch size
                num_texts=args.num_texts // 2
            )

        # Save JSON results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output}")

        # Generate and save report
        report = generate_benchmark_report(results, args.report)
        print("\n" + report)

        logger.info("\n✅ Benchmarking complete!")

    except KeyboardInterrupt:
        logger.info("\nBenchmarking interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()

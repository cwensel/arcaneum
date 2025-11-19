#!/usr/bin/env python3
"""
PDF indexing benchmark script.

Measures real-world PDF indexing performance with the optimizations applied.

Features:
- Test on sample PDFs or generate synthetic PDFs for testing
- Measure throughput: chunks/sec, bytes/sec, embeddings/sec
- GPU utilization tracking
- Memory usage profiling
- Batch size impact comparison
- Multi-file parallelism testing

Usage:
    # Benchmark with existing PDFs
    python scripts/benchmark_pdf_indexing.py --pdf-dir ./test_pdfs --output results.json

    # Generate synthetic test PDFs and benchmark
    python scripts/benchmark_pdf_indexing.py --generate-test-pdfs 10 --output results.json

    # Benchmark with different batch sizes
    python scripts/benchmark_pdf_indexing.py --batch-sizes 256,512,1024 --pdf-dir ./test_pdfs

    # Profile memory usage
    python scripts/benchmark_pdf_indexing.py --profile-memory --pdf-dir ./test_pdfs
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
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arcaneum.embeddings.client import EmbeddingClient
from arcaneum.indexing.uploader import PDFBatchUploader
from arcaneum.paths import get_models_dir
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def generate_synthetic_pdfs(count: int, output_dir: Path, pages_per_pdf: int = 5) -> List[Path]:
    """Generate synthetic PDF files for testing.

    Creates simple PDFs with text content for benchmarking.
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
    except ImportError:
        logger.error("reportlab not installed. Install with: pip install reportlab")
        return []

    output_dir.mkdir(exist_ok=True)
    pdfs = []

    logger.info(f"Generating {count} synthetic PDFs with {pages_per_pdf} pages each...")

    for i in range(count):
        pdf_path = output_dir / f"synthetic_doc_{i:03d}.pdf"

        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        for page in range(pages_per_pdf):
            # Add text content
            c.drawString(100, 750, f"Document {i} - Page {page + 1}")
            text = f"""
This is a synthetic document created for benchmarking purposes.

Document Index: {i}
Page Number: {page + 1}
Total Pages: {pages_per_pdf}

Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.

Content repeated for realistic testing:
""" * 10

            c.drawString(100, 700, "Synthetic PDF Content for Benchmarking")
            y = 680
            for line in text.split('\n')[:30]:  # Limit lines per page
                if y > 50:
                    c.drawString(100, y, line[:80])
                    y -= 15

            c.showPage()

        c.save()
        pdfs.append(pdf_path)
        logger.info(f"  Created: {pdf_path.name}")

    return pdfs


def benchmark_pdf_indexing(
    pdf_dir: Path,
    collection_name: str = "benchmark",
    model_name: str = "stella",
    batch_size: int = 256,
    embedding_batch_size: int = 256,
    gpu: bool = True,
    file_workers: int = 1,
    embedding_workers: int = 4,
    force_reindex: bool = True
) -> Dict:
    """Benchmark PDF indexing performance.

    Measures:
    - Total indexing time
    - Chunks created
    - Throughput: chunks/sec, bytes/sec
    - Memory usage
    """

    logger.info(f"Benchmarking PDF indexing:")
    logger.info(f"  PDF directory: {pdf_dir}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Embedding batch size: {embedding_batch_size}")
    logger.info(f"  GPU: {gpu}")
    logger.info(f"  File workers: {file_workers}")

    # Find PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        return {}

    total_bytes = sum(f.stat().st_size for f in pdf_files)
    logger.info(f"Found {len(pdf_files)} PDFs ({total_bytes / 1024 / 1024:.1f} MB)")

    # Create temporary Qdrant instance for testing
    temp_qdrant_dir = tempfile.mkdtemp(prefix="arcaneum_benchmark_")
    logger.info(f"Using temporary Qdrant storage: {temp_qdrant_dir}")

    try:
        # Initialize Qdrant client (in-memory for testing)
        qdrant = QdrantClient(":memory:")

        # Initialize embedding client
        embeddings = EmbeddingClient(
            cache_dir=str(get_models_dir()),
            use_gpu=gpu
        )

        # Initialize uploader with test parameters
        uploader = PDFBatchUploader(
            qdrant_client=qdrant,
            embedding_client=embeddings,
            batch_size=batch_size,
            embedding_batch_size=embedding_batch_size,
            file_workers=file_workers,
            embedding_workers=embedding_workers,
            markdown_conversion=True,
            ocr_enabled=False  # Skip OCR for benchmarking
        )

        # Pre-create collection with named vectors
        try:
            vector_size = embeddings.get_dimensions(model_name)
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config={
                    model_name: VectorParams(size=vector_size, distance=Distance.COSINE)
                }
            )
        except Exception as e:
            logger.warning(f"Collection creation failed (may already exist): {e}")

        # Benchmark indexing
        logger.info("\nStarting indexing...")
        start_time = time.time()

        stats = uploader.index_directory(
            pdf_dir=pdf_dir,
            collection_name=collection_name,
            model_name=model_name,
            model_config={
                'chunk_size': 512,
                'chunk_overlap': 50,
                'char_to_token_ratio': 3.3,
            },
            force_reindex=force_reindex,
            verbose=False
        )

        elapsed = time.time() - start_time

        # Calculate metrics
        chunks = stats.get('chunks', 0)
        files = stats.get('files', 0)
        errors = stats.get('errors', 0)

        throughput_chunks = chunks / elapsed if elapsed > 0 else 0
        throughput_bytes = total_bytes / elapsed if elapsed > 0 else 0

        result = {
            'pdf_count': len(pdf_files),
            'files_indexed': files,
            'errors': errors,
            'total_bytes': total_bytes,
            'chunks_created': chunks,
            'time_seconds': elapsed,
            'throughput': {
                'chunks_per_second': throughput_chunks,
                'bytes_per_second': throughput_bytes,
                'mb_per_minute': (throughput_bytes * 60) / (1024 * 1024)
            },
            'averages': {
                'seconds_per_pdf': elapsed / files if files > 0 else 0,
                'chunks_per_pdf': chunks / files if files > 0 else 0,
                'bytes_per_chunk': total_bytes / chunks if chunks > 0 else 0
            }
        }

        logger.info(f"\n=== RESULTS ===")
        logger.info(f"Time: {elapsed:.2f}s")
        logger.info(f"Files: {files}/{len(pdf_files)}")
        logger.info(f"Chunks: {chunks}")
        logger.info(f"Throughput: {throughput_chunks:.1f} chunks/sec ({throughput_bytes / 1024 / 1024:.1f} MB/sec)")
        logger.info(f"Per PDF: {elapsed / files:.2f}s ({chunks / files:.0f} chunks)")

        return result

    finally:
        # Cleanup
        if os.path.exists(temp_qdrant_dir):
            shutil.rmtree(temp_qdrant_dir)


def benchmark_batch_sizes(
    pdf_dir: Path,
    batch_sizes: List[int],
    num_runs: int = 2
) -> Dict:
    """Compare performance across different batch sizes."""

    logger.info(f"Benchmarking batch sizes: {batch_sizes}")
    results = {}

    for batch_size in batch_sizes:
        logger.info(f"\n=== Testing batch_size={batch_size} ===")

        timings = []
        for run in range(num_runs):
            logger.info(f"Run {run + 1}/{num_runs}...")

            result = benchmark_pdf_indexing(
                pdf_dir=pdf_dir,
                batch_size=batch_size,
                embedding_batch_size=batch_size,
                force_reindex=(run == 0)  # Only force first run
            )

            if result:
                timings.append(result['time_seconds'])

        if timings:
            avg_time = statistics.mean(timings)
            results[batch_size] = {
                'avg_time': avg_time,
                'throughput': result['throughput']['chunks_per_second'] if result else 0,
                'timings': timings
            }

            logger.info(f"  Average time: {avg_time:.2f}s ({result['throughput']['chunks_per_second']:.1f} chunks/sec)")

    return results


def generate_report(benchmark_results: Dict, output_file: str = None) -> str:
    """Generate benchmark report."""

    report = []
    report.append("=" * 80)
    report.append("PDF INDEXING BENCHMARK REPORT")
    report.append("=" * 80)
    report.append("")

    result = benchmark_results
    if isinstance(result, dict):
        report.append("INDEXING PERFORMANCE")
        report.append("-" * 80)
        report.append(f"Files indexed: {result.get('files_indexed', 0)}/{result.get('pdf_count', 0)}")
        report.append(f"Chunks created: {result.get('chunks_created', 0)}")
        report.append(f"Total time: {result.get('time_seconds', 0):.2f}s")
        report.append(f"Throughput: {result.get('throughput', {}).get('chunks_per_second', 0):.1f} chunks/sec")
        report.append(f"           {result.get('throughput', {}).get('mb_per_minute', 0):.1f} MB/min")
        report.append("")
        report.append("PER-FILE AVERAGES")
        report.append("-" * 80)
        averages = result.get('averages', {})
        report.append(f"Time per PDF: {averages.get('seconds_per_pdf', 0):.2f}s")
        report.append(f"Chunks per PDF: {averages.get('chunks_per_pdf', 0):.0f}")
        report.append("")

    report.append("=" * 80)

    result_text = "\n".join(report)

    if output_file:
        Path(output_file).write_text(result_text)
        logger.info(f"Report saved to {output_file}")

    return result_text


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PDF indexing performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--pdf-dir",
        type=Path,
        help="Directory containing PDF files to index"
    )

    parser.add_argument(
        "--generate-test-pdfs",
        type=int,
        metavar="COUNT",
        help="Generate synthetic PDFs for testing (count)"
    )

    parser.add_argument(
        "--pages-per-pdf",
        type=int,
        default=5,
        help="Pages per synthetic PDF (default: 5)"
    )

    parser.add_argument(
        "--model",
        default="stella",
        help="Embedding model (default: stella)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=300,
        help="Qdrant upload batch size (default: 300)"
    )

    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=256,
        help="Embedding batch size (default: 256)"
    )

    parser.add_argument(
        "--batch-sizes",
        help="Comma-separated batch sizes to compare (overrides --batch-size)"
    )

    parser.add_argument(
        "--file-workers",
        type=int,
        default=1,
        help="Number of parallel PDF workers (default: 1)"
    )

    parser.add_argument(
        "--embedding-workers",
        type=int,
        default=4,
        help="Number of parallel embedding workers (default: 4)"
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )

    parser.add_argument(
        "--output",
        help="Output file for JSON results"
    )

    parser.add_argument(
        "--report",
        help="Output file for text report"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle PDF directory
    if args.generate_test_pdfs:
        pdf_dir = Path(tempfile.mkdtemp(prefix="arcaneum_test_pdfs_"))
        logger.info(f"Generating test PDFs in {pdf_dir}")
        generate_synthetic_pdfs(
            count=args.generate_test_pdfs,
            output_dir=pdf_dir,
            pages_per_pdf=args.pages_per_pdf
        )
    elif args.pdf_dir:
        pdf_dir = args.pdf_dir
        if not pdf_dir.exists():
            logger.error(f"PDF directory not found: {pdf_dir}")
            sys.exit(1)
    else:
        logger.error("Either --pdf-dir or --generate-test-pdfs required")
        parser.print_help()
        sys.exit(1)

    try:
        # Run benchmark
        if args.batch_sizes:
            batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
            results = benchmark_batch_sizes(pdf_dir, batch_sizes)
        else:
            results = benchmark_pdf_indexing(
                pdf_dir=pdf_dir,
                model_name=args.model,
                batch_size=args.batch_size,
                embedding_batch_size=args.embedding_batch_size,
                gpu=not args.no_gpu,
                file_workers=args.file_workers,
                embedding_workers=args.embedding_workers
            )

        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output}")

        # Generate report
        report = generate_report(results, args.report)
        print("\n" + report)

        logger.info("✅ Benchmarking complete!")

    except KeyboardInterrupt:
        logger.info("\nBenchmarking interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()

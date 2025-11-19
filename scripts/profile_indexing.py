#!/usr/bin/env python3
"""
Profile PDF indexing pipeline with cProfile to identify CPU bottlenecks.
Usage: python profile_indexing.py --pdfs NUM_PDFS [--pages PAGES_PER_PDF] [--output OUTFILE]
"""
import argparse
import cProfile
import pstats
import sys
import tempfile
from pathlib import Path
from io import StringIO
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def run_indexing_benchmark(num_pdfs: int, pages_per_pdf: int, collection_name: str = "profile_test"):
    """Run PDF indexing for profiling."""
    import os
    import random
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    # Import after logging setup
    from arcaneum.cli.utils import create_qdrant_client
    from arcaneum.embeddings.model_cache import get_cached_model
    from arcaneum.paths import get_models_dir
    from arcaneum.indexing.uploader import PdfUploader

    logger.info(f"Generating {num_pdfs} test PDFs with {pages_per_pdf} pages each...")

    # Create temporary PDF directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate test PDFs
        for pdf_idx in range(num_pdfs):
            pdf_path = tmpdir / f"test_{pdf_idx:03d}.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=letter)

            for page_idx in range(pages_per_pdf):
                # Add text content
                c.drawString(50, 750, f"PDF {pdf_idx}, Page {page_idx}")
                c.drawString(50, 730, "This is test content for profiling the indexing pipeline.")
                c.drawString(50, 710, "Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
                c.drawString(50, 690, f"Random text: {random.randint(1000, 9999)}")
                c.showPage()

            c.save()
            logger.info(f"  Created: {pdf_path.name}")

        # Create Qdrant client
        logger.info("Connecting to Qdrant...")
        with tempfile.TemporaryDirectory() as qdrant_dir:
            qdrant_url = f"http://localhost:6333"
            try:
                qdrant_client = create_qdrant_client(url=qdrant_url)
            except Exception as e:
                logger.error(f"Could not connect to Qdrant: {e}")
                logger.error("Starting local Qdrant instance would require docker/docker-compose")
                raise

            # Create collection
            try:
                from qdrant_client.models import VectorParams, Distance
                vector_size = 1024
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "default": VectorParams(size=vector_size, distance=Distance.COSINE)
                    }
                )
            except Exception:
                pass  # Collection might already exist

            # Load model
            logger.info("Loading embedding model...")
            model = get_cached_model(
                model_name="stella",
                cache_dir=str(get_models_dir()),
                use_gpu=True
            )

            # Create uploader
            uploader = PdfUploader(
                qdrant_client=qdrant_client,
                embedding_client=model,
                file_workers=1,
                embedding_batch_size=256
            )

            # Index PDFs
            logger.info(f"Indexing {num_pdfs} PDFs...")
            stats = uploader.index_directory(
                pdf_dir=tmpdir,
                collection_name=collection_name,
                model_name="stella",
                force_reindex=True,
                verbose=False
            )

            logger.info(f"Indexing complete: {stats}")


def main():
    parser = argparse.ArgumentParser(description="Profile PDF indexing pipeline")
    parser.add_argument("--pdfs", type=int, default=50, help="Number of test PDFs to generate")
    parser.add_argument("--pages", type=int, default=5, help="Pages per PDF")
    parser.add_argument("--output", default="profile_results.prof", help="Output profile file")
    parser.add_argument("--stats", action="store_true", help="Print stats instead of saving")

    args = parser.parse_args()

    logger.info(f"Starting profiling: {args.pdfs} PDFs × {args.pages} pages")
    logger.info("=" * 60)

    # Run profiling
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        run_indexing_benchmark(args.pdfs, args.pages)
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        profiler.disable()

    # Print results
    if args.stats:
        ps = pstats.Stats(profiler, stream=sys.stdout)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        print("\n" + "=" * 60)
        print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
        print("=" * 60)
        ps.print_stats(30)

        print("\n" + "=" * 60)
        print("TOP 30 FUNCTIONS BY TOTAL TIME")
        print("=" * 60)
        ps.sort_stats('time')
        ps.print_stats(30)
    else:
        # Save profile
        profiler.dump_stats(args.output)
        logger.info(f"Profile saved to: {args.output}")
        logger.info(f"View with: python -m pstats {args.output}")


if __name__ == "__main__":
    sys.exit(main())

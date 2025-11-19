#!/usr/bin/env python3
"""
Profile PDF indexing with cProfile using the existing benchmark infrastructure.
Generates test PDFs and indexes them, collecting CPU profiling data.

Usage: python profile_with_cprofile.py --pdfs NUM --pages PAGES
"""
import argparse
import cProfile
import pstats
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Profile PDF indexing with cProfile")
    parser.add_argument("--pdfs", type=int, default=20, help="Number of test PDFs")
    parser.add_argument("--pages", type=int, default=5, help="Pages per PDF")
    parser.add_argument("--output", default="indexing_profile.prof", help="Output .prof file")

    args = parser.parse_args()

    # Import indexing components
    import tempfile
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from qdrant_client.models import VectorParams, Distance
    from arcaneum.cli.utils import create_qdrant_client
    from arcaneum.embeddings.model_cache import get_cached_model
    from arcaneum.paths import get_models_dir
    from arcaneum.indexing.uploader import PDFBatchUploader

    print(f"Profiling: {args.pdfs} PDFs × {args.pages} pages")
    print("=" * 70)

    def profile_task():
        """Task to profile."""
        # Generate test PDFs
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            print(f"Generating {args.pdfs} test PDFs...")

            for pdf_idx in range(args.pdfs):
                pdf_path = tmpdir / f"test_{pdf_idx:03d}.pdf"
                c = canvas.Canvas(str(pdf_path), pagesize=letter)

                for page_idx in range(args.pages):
                    y = 750
                    c.drawString(50, y, f"PDF {pdf_idx}, Page {page_idx}")
                    y -= 20
                    c.drawString(50, y, "Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
                    y -= 20
                    c.drawString(50, y, "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")
                    y -= 20
                    c.drawString(50, y, f"Document ID: {pdf_idx}-{page_idx}")
                    c.showPage()

                c.save()

            print(f"Created {args.pdfs} PDFs in {tmpdir}")

            # Create Qdrant client with temporary storage
            with tempfile.TemporaryDirectory() as qdrant_dir:
                print(f"Using temporary Qdrant storage: {qdrant_dir}")

                # Use in-memory Qdrant (localhost with temp dir)
                qdrant_url = "http://localhost:6333"
                print(f"Connecting to Qdrant at {qdrant_url}...")

                try:
                    qdrant_client = create_qdrant_client(url=qdrant_url)
                except Exception as e:
                    print(f"ERROR: Could not connect to Qdrant: {e}")
                    print("Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
                    sys.exit(1)

                # Create collection
                collection_name = f"profile_test_{os.getpid()}"
                print(f"Creating collection: {collection_name}")

                try:
                    qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "default": VectorParams(size=1024, distance=Distance.COSINE)
                        }
                    )
                except Exception as e:
                    print(f"Error creating collection: {e}")

                # Load embedding model
                print("Loading embedding model...")
                embeddings = get_cached_model(
                    model_name="stella",
                    cache_dir=str(get_models_dir()),
                    use_gpu=True
                )

                # Create uploader
                print("Creating PDF uploader...")
                uploader = PDFBatchUploader(
                    qdrant_client=qdrant_client,
                    embedding_client=embeddings,
                    file_workers=1,
                    embedding_batch_size=256
                )

                # Index PDFs
                print(f"Indexing {args.pdfs} PDFs ({args.pages} pages each)...")
                stats = uploader.index_directory(
                    pdf_directory=tmpdir,
                    collection_name=collection_name,
                    model_name="stella",
                    force_reindex=True,
                    verbose=False
                )

                print(f"Indexing complete!")
                print(f"  Files: {stats.get('files_indexed', 0)}/{stats.get('pdf_count', 0)}")
                print(f"  Chunks: {stats.get('chunks_created', 0)}")
                print(f"  Time: {stats.get('time_seconds', 0):.2f}s")

    # Run profiling
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        profile_task()
    finally:
        profiler.disable()

    # Save and print results
    profiler.dump_stats(args.output)
    print(f"\nProfile saved to: {args.output}")

    # Print top functions by time
    print("\n" + "=" * 70)
    print("TOP 40 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 70)
    ps = pstats.Stats(args.output)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(40)

    print("\n" + "=" * 70)
    print("TOP 40 FUNCTIONS BY TOTAL TIME")
    print("=" * 70)
    ps.sort_stats('time')
    ps.print_stats(40)

    print(f"\nFull profile: python -m pstats {args.output}")
    print("Use 'help' in pstats for analysis commands")


if __name__ == "__main__":
    main()

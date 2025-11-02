"""Batch upload orchestrator for PDF indexing (RDR-004)."""

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from pathlib import Path
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type
import logging
import os
import sys

from ..embeddings.client import EmbeddingClient
from .common.sync import MetadataBasedSync, compute_file_hash
from .pdf.extractor import PDFExtractor
from .pdf.ocr import OCREngine
from .pdf.chunker import PDFChunker
from ..monitoring.cpu_stats import create_monitor

logger = logging.getLogger(__name__)


class PDFBatchUploader:
    """Orchestrate bulk PDF indexing with batching and error recovery."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_client: EmbeddingClient,
        batch_size: int = 100,
        parallel_workers: int = 4,
        max_retries: int = 5,
        ocr_enabled: bool = True,
        ocr_engine: str = 'tesseract',
        ocr_language: str = 'eng',
        ocr_threshold: int = 100,
        ocr_workers: Optional[int] = None,
        embedding_workers: int = 4,
        embedding_batch_size: int = 200,
        batch_across_files: bool = False,
    ):
        """Initialize batch uploader.

        Args:
            qdrant_client: Qdrant client instance
            embedding_client: Embedding client instance
            batch_size: Number of points per batch
            parallel_workers: Number of parallel upload workers
            max_retries: Maximum retry attempts for failed uploads
            ocr_enabled: Enable OCR for scanned PDFs (default: True)
            ocr_engine: OCR engine ('tesseract' or 'easyocr')
            ocr_language: OCR language code
            ocr_threshold: Trigger OCR if text < N characters
            ocr_workers: Number of parallel OCR workers (None = cpu_count)
            embedding_workers: Number of parallel workers for embedding generation (default: 4)
            embedding_batch_size: Batch size for embedding generation (default: 200)
        """
        self.qdrant = qdrant_client
        self.embeddings = embedding_client
        self.batch_size = batch_size
        self.parallel_workers = parallel_workers
        self.max_retries = max_retries
        self.embedding_workers = embedding_workers
        self.embedding_batch_size = embedding_batch_size

        # Initialize components
        self.extractor = PDFExtractor(
            fallback_enabled=True,
            table_validation=True
        )

        self.ocr_enabled = ocr_enabled
        self.ocr_threshold = ocr_threshold
        self.batch_across_files = batch_across_files

        if ocr_enabled:
            self.ocr = OCREngine(
                engine=ocr_engine,
                language=ocr_language,
                confidence_threshold=60.0,
                image_dpi=300,
                image_scale=2.0,
                ocr_workers=ocr_workers
            )
        else:
            self.ocr = None

        # Metadata-based sync (queries Qdrant directly, no separate DB)
        self.sync = MetadataBasedSync(qdrant_client)

    def index_directory(
        self,
        pdf_dir: Path,
        collection_name: str,
        model_name: str,
        model_config: Dict,
        force_reindex: bool = False,
        verbose: bool = False
    ) -> Dict:
        """Index PDFs in directory with incremental sync.

        Args:
            pdf_dir: Directory containing PDFs
            collection_name: Qdrant collection name
            model_name: Embedding model to use
            model_config: Model configuration (chunk_size, overlap, etc.)
            force_reindex: Bypass metadata sync and reindex all files
            verbose: If True, show tqdm progress bar; if False, show per-file progress

        Returns:
            Statistics dict with files, chunks, errors counts
        """
        # Suppress tokenizers fork warning
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

        # Start CPU monitoring (RDR-013 Phase 1)
        cpu_monitor = create_monitor()
        if cpu_monitor:
            cpu_monitor.start()

        # Initialize chunker
        chunker = PDFChunker(
            model_config=model_config,
            overlap_percent=0.15,  # 15% overlap (NVIDIA recommendation)
            late_chunking_enabled=model_config.get('late_chunking', False),
            min_doc_tokens=2000,
            max_doc_tokens=8000
        )

        # Discover all PDFs
        all_pdf_files = sorted(pdf_dir.rglob("*.pdf"))
        logger.info(f"Found {len(all_pdf_files)} total PDF files")

        # Filter to unindexed files via metadata queries (unless force_reindex)
        if verbose:
            print(f"🔍 Scanning collection for existing files...")

        if force_reindex:
            pdf_files = all_pdf_files
            logger.info(f"Force reindex: processing all {len(pdf_files)} PDFs")
            if verbose:
                print(f"🔄 Force reindex: {len(pdf_files)} PDFs to process")
        else:
            pdf_files = self.sync.get_unindexed_files(collection_name, all_pdf_files)
            skipped = len(all_pdf_files) - len(pdf_files)
            logger.info(f"Incremental sync: {len(pdf_files)} new/modified, {skipped} already indexed")

            if verbose:
                print(f"📊 Found {len(all_pdf_files)} PDFs: {len(pdf_files)} new/modified, {skipped} already indexed")

        if not pdf_files:
            logger.info("No PDFs to index")
            if not verbose:
                print("All PDFs up to date")
            else:
                print("✅ All PDFs are up to date")
            return {"files": 0, "chunks": 0, "errors": 0}

        # Show count for minimal mode
        if not verbose:
            print(f"Found {len(pdf_files)} PDF(s)")
            print(f"Processing {len(pdf_files)} PDF(s)...")
        else:
            print()

        # Process PDFs with clear stage-based progress
        batch = []
        point_id = self._get_next_point_id(collection_name)
        stats = {"files": 0, "chunks": 0, "errors": 0}

        try:
            total_pdfs = len(pdf_files)

            for pdf_idx, pdf_path in enumerate(pdf_files, 1):
                try:
                    # Track chunks created and uploaded for this PDF
                    file_chunk_count = 0
                    chunks_uploaded_before = stats["chunks"]

                    # Stage 0: Computing hash (silent - too fast to show)
                    if verbose:
                        print(f"\n[{pdf_idx}/{total_pdfs}] {pdf_path.name}", flush=True)
                        print(f"  → computing hash", flush=True)

                    # Compute file hash for incremental indexing
                    file_hash = compute_file_hash(pdf_path)

                    # Stage 1: Extract text
                    if not verbose:
                        print(f"\r[{pdf_idx}/{total_pdfs}] {pdf_path.name} → extracting{' '*20}", end="", flush=True)
                    else:
                        print(f"  → extracting text", flush=True)

                    if not verbose:
                        # Suppress PyMuPDF warnings to stderr
                        stderr_fd = sys.stderr.fileno()
                        with open(os.devnull, 'w') as devnull:
                            old_stderr = os.dup(stderr_fd)
                            os.dup2(devnull.fileno(), stderr_fd)
                            try:
                                text, extract_meta = self.extractor.extract(pdf_path)
                            finally:
                                os.dup2(old_stderr, stderr_fd)
                                os.close(old_stderr)
                    else:
                        text, extract_meta = self.extractor.extract(pdf_path)
                        print(f"     extracted {len(text)} chars", flush=True)

                    # Stage 2: OCR (if needed)
                    if self.ocr_enabled and len(text) < self.ocr_threshold:
                        # Get page count for progress display
                        import pymupdf as fitz
                        try:
                            temp_doc = fitz.open(pdf_path)
                            page_count = len(temp_doc)
                            temp_doc.close()
                        except:
                            page_count = 0

                        if not verbose:
                            print(f"\r[{pdf_idx}/{total_pdfs}] {pdf_path.name} → OCR ({page_count}p){' '*20}", end="", flush=True)
                        else:
                            print(f"  → running OCR ({page_count} pages)", flush=True)

                        text, ocr_meta = self.ocr.process_pdf(pdf_path, verbose=verbose)
                        extract_meta.update(ocr_meta)

                        if verbose:
                            print(f"     OCR complete", flush=True)

                    # Chunk text with file metadata (including hash for sync)
                    base_metadata = {
                        'filename': pdf_path.name,
                        'file_path': str(pdf_path),
                        'file_hash': file_hash,  # For incremental sync
                        'file_size': pdf_path.stat().st_size,
                        'store_type': 'pdf',
                        **extract_meta
                    }

                    # Stage 3: Chunking
                    if not verbose:
                        print(f"\r[{pdf_idx}/{total_pdfs}] {pdf_path.name} → chunking ({len(text)} chars){' '*15}", end="", flush=True)
                    else:
                        print(f"  → chunking ({len(text)} chars)", flush=True)

                    chunks = chunker.chunk(text, base_metadata)
                    file_chunk_count = len(chunks)

                    if verbose:
                        print(f"     created {file_chunk_count} chunks", flush=True)

                    # Stage 4: Embedding
                    texts = [chunk.text for chunk in chunks]
                    if not verbose:
                        print(f"\r[{pdf_idx}/{total_pdfs}] {pdf_path.name} → embedding ({file_chunk_count} chunks, parallel){' '*15}", end="", flush=True)
                    else:
                        print(f"  → embedding ({file_chunk_count} chunks, parallel)", flush=True)

                    # Parallel embedding (Phase 2 RDR-013) for 2-4x speedup
                    embeddings = self.embeddings.embed_parallel(
                        texts,
                        model_name,
                        max_workers=self.embedding_workers,
                        batch_size=self.embedding_batch_size
                    )

                    if verbose:
                        print(f"     embedded {file_chunk_count} chunks", flush=True)

                    # Stage 5: Create points and upload
                    for chunk, embedding in zip(chunks, embeddings):
                        point = PointStruct(
                            id=point_id,
                            vector={model_name: embedding},  # Named vector
                            payload={
                                'text': chunk.text,
                                **chunk.metadata
                            }
                        )
                        batch.append(point)
                        point_id += 1

                        # Upload when batch full (only if batching across files)
                        if self.batch_across_files and len(batch) >= self.batch_size:
                            if verbose:
                                print(f"  → uploading batch ({len(batch)} chunks)", flush=True)
                            self._upload_batch(collection_name, batch)
                            stats["chunks"] += len(batch)
                            batch = []

                    # Upload this PDF's chunks immediately (atomic mode - default)
                    if not self.batch_across_files and len(batch) > 0:
                        if not verbose:
                            print(f"\r[{pdf_idx}/{total_pdfs}] {pdf_path.name} → uploading ({len(batch)} chunks){' '*15}", end="", flush=True)
                        else:
                            print(f"  → uploading ({len(batch)} chunks)", flush=True)

                        self._upload_batch(collection_name, batch)
                        stats["chunks"] += len(batch)
                        batch = []

                    stats["files"] += 1

                    # Stage 6: Complete
                    if not verbose:
                        status_line = f"[{pdf_idx}/{total_pdfs}] {pdf_path.name} ✓ ({file_chunk_count} chunks)"
                        print(f"\r{status_line:<80}")
                    else:
                        print(f"  ✓ complete ({file_chunk_count} chunks)", flush=True)

                except Exception as e:
                    # Show error
                    error_msg = str(e)
                    error_type = type(e).__name__
                    if not verbose:
                        status_line = f"[{pdf_idx}/{total_pdfs}] {pdf_path.name} ✗ ({error_type})"
                        print(f"\r{status_line:<80}")
                    else:
                        print(f"  ✗ ERROR: {error_msg}", file=sys.stderr, flush=True)

                    stats["errors"] += 1
                    continue

            # Upload remaining batch (only needed if batching across files)
            if batch:
                print(f"\n→ Uploading final batch ({len(batch)} chunks)", flush=True)
                try:
                    self._upload_batch(collection_name, batch)
                    stats["chunks"] += len(batch)
                    print(f"  ✓ Final batch uploaded", flush=True)
                except Exception as e:
                    print(f"\nERROR: Final batch upload failed: {e}", file=sys.stderr)
                    print(f"  Lost {len(batch)} chunks from batch buffer!", file=sys.stderr)

        except KeyboardInterrupt:
            # Let the signal handler in CLI handle it
            # Just re-raise to propagate to CLI
            raise

        # Summary
        print()
        if logger.isEnabledFor(logging.INFO):
            # Verbose mode: detailed output
            logger.info("=" * 60)
            logger.info("INDEXING COMPLETE")
            logger.info(f"Files processed: {stats['files']}")
            logger.info(f"Chunks uploaded: {stats['chunks']}")
            logger.info(f"Errors: {stats['errors']}")

            # Show CPU statistics (RDR-013 Phase 1)
            if cpu_monitor:
                logger.info("")
                cpu_stats = cpu_monitor.get_stats()
                logger.info(f"CPU usage: {cpu_stats['cpu_percent']:.1f}% total ({cpu_stats['cpu_percent_per_core']:.1f}% per core)")
                logger.info(f"Threads: {cpu_stats['num_threads']} | Cores: {cpu_stats['num_cores']}")
                logger.info(f"Elapsed: {cpu_stats['elapsed_time']:.1f}s")

            logger.info("=" * 60)
        else:
            # Normal mode: clean summary
            print(f"✅ Indexed {stats['files']} PDFs → {stats['chunks']} chunks")
            if stats['errors'] > 0:
                print(f"⚠️  {stats['errors']} errors")

            # Show CPU stats in compact mode (RDR-013 Phase 1)
            if cpu_monitor:
                print(f"ℹ️  {cpu_monitor.get_summary()}")

        return stats

    def _get_next_point_id(self, collection_name: str) -> int:
        """Get next available point ID for collection."""
        try:
            info = self.qdrant.get_collection(collection_name)
            return info.points_count
        except:
            return 0

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),  # Max 10 second wait
        retry=retry_if_not_exception_type(KeyboardInterrupt),  # Don't retry on Ctrl+C
        reraise=True
    )
    def _upload_batch(self, collection_name: str, points: List[PointStruct]):
        """Upload batch with exponential backoff retry."""
        try:
            result = self.qdrant.upload_points(
                collection_name=collection_name,
                points=points,
                batch_size=self.batch_size,
                parallel=self.parallel_workers,
                max_retries=3,  # Inner retry for rate limiting
                wait=True
            )

            logger.debug(f"Batch uploaded: {len(points)} points")
            return result

        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            raise

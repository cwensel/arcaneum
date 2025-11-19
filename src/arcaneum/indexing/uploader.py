"""Batch upload orchestrator for PDF indexing (RDR-004)."""

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import logging
import os
import sys

from ..embeddings.client import EmbeddingClient
from .common.sync import MetadataBasedSync, compute_file_hash, compute_quick_hash
from .pdf.extractor import PDFExtractor
from .pdf.ocr import OCREngine
from .pdf.chunker import PDFChunker
from ..monitoring.cpu_stats import create_monitor
from ..utils.memory import calculate_safe_workers, log_memory_stats

logger = logging.getLogger(__name__)


class PDFBatchUploader:
    """Orchestrate bulk PDF indexing with batching and error recovery."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_client: EmbeddingClient,
        batch_size: int = 300,
        parallel_workers: int = 4,
        max_retries: int = 5,
        ocr_enabled: bool = True,
        ocr_engine: str = 'tesseract',
        ocr_language: str = 'eng',
        ocr_threshold: int = 100,
        ocr_workers: Optional[int] = None,
        embedding_workers: int = 4,
        embedding_batch_size: int = 256,
        batch_across_files: bool = False,
        file_workers: int = 1,
        max_memory_gb: Optional[float] = None,
        pdf_timeout: int = 600,
        ocr_page_timeout: int = 60,
        embedding_timeout: int = 300,
        markdown_conversion: bool = True,
        preserve_images: bool = False,
    ):
        """Initialize batch uploader.

        Args:
            qdrant_client: Qdrant client instance
            embedding_client: Embedding client instance
            batch_size: Number of points per batch (default: 300, optimized from 100)
            parallel_workers: Number of parallel upload workers
            max_retries: Maximum retry attempts for failed uploads
            ocr_enabled: Enable OCR for scanned PDFs (default: True)
            ocr_engine: OCR engine ('tesseract' or 'easyocr')
            ocr_language: OCR language code
            ocr_threshold: Trigger OCR if text < N characters
            ocr_workers: Number of parallel OCR workers (None = cpu_count)
            embedding_workers: Number of parallel workers for embedding generation (default: 4)
            embedding_batch_size: Batch size for embedding generation (default: 256, optimized from 200 per arcaneum-9kgg)
            batch_across_files: Whether to batch uploads across files
            file_workers: Number of PDF files to process in parallel (default: 1)
            max_memory_gb: Maximum memory to use in GB (None = auto-calculate from available)
            pdf_timeout: Timeout in seconds for processing a single PDF (default: 600)
            ocr_page_timeout: Timeout in seconds for OCR processing a single page (default: 60)
            embedding_timeout: Timeout in seconds for embedding generation (default: 300)
            markdown_conversion: Convert PDF to markdown with structure (default: True, RDR-016)
            preserve_images: Extract images for multimodal search (default: False, RDR-016)
        """
        self.qdrant = qdrant_client
        self.embeddings = embedding_client
        self.batch_size = batch_size
        self.parallel_workers = parallel_workers
        self.max_retries = max_retries
        self.embedding_workers = embedding_workers
        self.embedding_batch_size = embedding_batch_size
        self.max_memory_gb = max_memory_gb
        self.pdf_timeout = pdf_timeout
        self.ocr_page_timeout = ocr_page_timeout
        self.embedding_timeout = embedding_timeout

        # Apply memory-aware worker limits
        # Estimate 500MB per PDF file worker (images, OCR, embeddings)
        # NOTE: All workers share the same embedding model instance (locked for thread-safety).
        # No additional memory per worker needed (arcaneum-3fs3).
        memory_per_worker_mb = 500

        safe_file_workers, warning = calculate_safe_workers(
            requested_workers=max(1, file_workers),
            estimated_memory_per_worker_mb=memory_per_worker_mb,
            max_memory_gb=max_memory_gb,
            min_workers=1
        )
        self.file_workers = safe_file_workers

        # Lock for thread-safe access to embedding client (arcaneum-3fs3)
        # SentenceTransformer models are NOT thread-safe. Multiple workers accessing
        # the same model instance can cause GPU memory conflicts and state corruption.
        # Solution: Serialize access to embedding operations with a lock.
        import threading
        self._embedding_lock = threading.Lock()

        if warning:
            logger.warning(warning)
            print(warning, flush=True)

        # GPU + file workers now works efficiently
        # Single-threaded embedding with larger batches (arcaneum-m7hg)
        # No serialization penalty - GPU has internal parallelism per batch
        if embedding_client.use_gpu and self.file_workers > 1 and False:  # Disabled: no longer a bottleneck
            gpu_info = (
                f"ℹ️  GPU acceleration with {self.file_workers} file workers: "
                f"Efficient batching strategy (single-threaded per batch for internal GPU parallelism)"
            )
            logger.debug(gpu_info)

        # Initialize components (RDR-016: markdown conversion default)
        self.extractor = PDFExtractor(
            fallback_enabled=True,
            table_validation=True,
            markdown_conversion=markdown_conversion,
            preserve_images=preserve_images,
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
                ocr_workers=ocr_workers,
                max_memory_gb=max_memory_gb,
                page_timeout=ocr_page_timeout
            )
        else:
            self.ocr = None

        # Log memory stats at initialization
        log_memory_stats("Initialization: ")

        # Metadata-based sync (queries Qdrant directly, no separate DB)
        self.sync = MetadataBasedSync(qdrant_client)

    def _process_single_pdf(
        self,
        pdf_path: Path,
        collection_name: str,
        model_name: str,
        model_config: Dict,
        chunker: 'PDFChunker',
        point_id_start: int,
        verbose: bool,
        pdf_idx: int,
        total_pdfs: int
    ) -> Tuple[List[PointStruct], int, Optional[str]]:
        """Process a single PDF: extract, OCR, chunk, embed, create points.

        Args:
            pdf_path: Path to PDF file
            collection_name: Collection name (for upload)
            model_name: Embedding model name
            model_config: Model configuration
            chunker: PDFChunker instance
            point_id_start: Starting point ID for this PDF
            verbose: Verbose output flag
            pdf_idx: Current PDF index (for progress)
            total_pdfs: Total number of PDFs (for progress)

        Returns:
            Tuple of (points list, chunk count, error message or None)
        """
        try:
            # Stage 0: Computing hashes (two-pass sync)
            if verbose:
                print(f"\n[{pdf_idx}/{total_pdfs}] {pdf_path.name}", flush=True)
                print(f"  → computing hashes", flush=True)

            # Pass 1: Fast metadata hash (mtime+size)
            quick_hash = compute_quick_hash(pdf_path)
            # Pass 2: Full content hash for storage and deep verification
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

            # Chunk text with file metadata (two-pass sync support)
            base_metadata = {
                'filename': pdf_path.name,
                'file_path': str(pdf_path.absolute()),  # Always store absolute path for consistent lookups
                'quick_hash': quick_hash,  # Pass 1: Fast metadata-based hash (mtime+size)
                'file_hash': file_hash,     # Pass 2: Full content hash (for deep verification)
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

            # CRITICAL FIX (arcaneum-3fs3): Lock access to shared embedding client for multi-worker mode
            # SentenceTransformer models are NOT thread-safe. Multiple workers using the same
            # model instance causes GPU memory conflicts and state corruption.
            # Solution: Serialize embedding operations with a lock while reusing the shared model.
            # This avoids the overhead of loading separate model instances per worker.
            if self.file_workers > 1:
                # Multi-worker mode: use lock to serialize access to shared model
                with self._embedding_lock:
                    embeddings = self.embeddings.embed_parallel(
                        texts,
                        model_name,
                        max_workers=self.embedding_workers,
                        batch_size=self.embedding_batch_size,
                        timeout=self.embedding_timeout
                    )
            else:
                # Single-worker mode: no locking needed (no contention)
                embeddings = self.embeddings.embed_parallel(
                    texts,
                    model_name,
                    max_workers=self.embedding_workers,
                    batch_size=self.embedding_batch_size,
                    timeout=self.embedding_timeout
                )

            if verbose:
                print(f"     embedded {file_chunk_count} chunks", flush=True)

            # Stage 5: Create and upload points in batches (streaming upload)
            # This avoids holding all points in memory at once
            UPLOAD_BATCH_SIZE = 1000
            points_batch = []
            point_id = point_id_start
            uploaded_count = 0

            for chunk, embedding in zip(chunks, embeddings):
                point = PointStruct(
                    id=point_id,
                    vector={model_name: embedding},
                    payload={
                        'text': chunk.text,
                        **chunk.metadata
                    }
                )
                points_batch.append(point)
                point_id += 1

                # Upload batch when threshold reached
                if len(points_batch) >= UPLOAD_BATCH_SIZE:
                    if not verbose:
                        print(f"\r[{pdf_idx}/{total_pdfs}] {pdf_path.name} → uploading batch ({uploaded_count}/{file_chunk_count}){' '*15}", end="", flush=True)
                    else:
                        print(f"  → uploading batch ({len(points_batch)} chunks, {uploaded_count}/{file_chunk_count} total)", flush=True)

                    self._upload_batch(collection_name, points_batch)
                    uploaded_count += len(points_batch)

                    # Clear batch (no gc.collect per batch - do once per file instead)
                    points_batch.clear()

            # Upload remaining points
            if points_batch:
                if not verbose:
                    print(f"\r[{pdf_idx}/{total_pdfs}] {pdf_path.name} → uploading final batch ({uploaded_count}/{file_chunk_count}){' '*15}", end="", flush=True)
                else:
                    print(f"  → uploading final batch ({len(points_batch)} chunks)", flush=True)

                self._upload_batch(collection_name, points_batch)
                uploaded_count += len(points_batch)
                points_batch.clear()

            # Clear large lists to free memory and garbage collect ONCE per file (arcaneum-d432)
            # This reduces gc.collect() overhead from 100+ calls to ~1 per file
            del texts, embeddings, chunks, points_batch
            import gc
            gc.collect()

            # Return empty list since we already uploaded
            # uploaded_count is used for stats
            return ([], uploaded_count, None)

        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            return ([], 0, f"{error_type}: {error_msg}")

    def index_directory(
        self,
        pdf_dir: Path,
        collection_name: str,
        model_name: str,
        model_config: Dict,
        force_reindex: bool = False,
        verbose: bool = False
    ) -> Dict:
        """Index PDFs in directory with incremental sync and optional parallel processing.

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
            pdf_files, renames, already_indexed = self.sync.get_unindexed_files(collection_name, all_pdf_files)

            # Handle renames first (metadata update only, no reindexing)
            if renames:
                self.sync.handle_renames(collection_name, renames)

            logger.info(f"Incremental sync: {len(pdf_files)} new/modified, "
                       f"{len(renames)} renamed, {len(already_indexed)} already indexed")

            if verbose:
                if renames:
                    print(f"📝 Renamed {len(renames)} PDFs (metadata updated)")
                print(f"📊 Found {len(all_pdf_files)} PDFs: {len(pdf_files)} new/modified, "
                      f"{len(renames)} renamed, {len(already_indexed)} already indexed")

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

        # Process PDFs with optional parallel processing (arcaneum-108)
        point_id = self._get_next_point_id(collection_name)
        stats = {"files": 0, "chunks": 0, "errors": 0}

        try:
            total_pdfs = len(pdf_files)

            # Use parallel processing if file_workers > 1
            if self.file_workers > 1:
                # Parallel mode: Use ThreadPoolExecutor to process multiple PDFs concurrently
                # Pre-allocate point ID ranges (generous allocation: 1000 chunks per PDF)
                point_id_step = 1000

                with ThreadPoolExecutor(max_workers=self.file_workers) as executor:
                    # Submit all PDF processing jobs
                    future_to_pdf = {}
                    for pdf_idx, pdf_path in enumerate(pdf_files, 1):
                        future = executor.submit(
                            self._process_single_pdf,
                            pdf_path,
                            collection_name,
                            model_name,
                            model_config,
                            chunker,
                            point_id + (pdf_idx - 1) * point_id_step,
                            verbose,
                            pdf_idx,
                            total_pdfs
                        )
                        future_to_pdf[future] = (pdf_idx, pdf_path)

                    # Collect results as they complete
                    for future in as_completed(future_to_pdf):
                        pdf_idx, pdf_path = future_to_pdf[future]
                        try:
                            points, file_chunk_count, error = future.result(timeout=self.pdf_timeout)
                        except TimeoutError:
                            error = f"TimeoutError: PDF processing exceeded {self.pdf_timeout}s"
                            points, file_chunk_count = [], 0

                        if error:
                            # Show error
                            if not verbose:
                                status_line = f"[{pdf_idx}/{total_pdfs}] {pdf_path.name} ✗ ({error.split(':')[0]})"
                                print(f"\r{status_line:<80}")
                            else:
                                print(f"  ✗ ERROR: {error}", file=sys.stderr, flush=True)
                            stats["errors"] += 1
                        else:
                            # Points are already uploaded via streaming (points list will be empty)
                            # Just update stats and show completion
                            if file_chunk_count > 0:
                                stats["chunks"] += file_chunk_count
                                stats["files"] += 1

                                # Complete
                                if not verbose:
                                    status_line = f"[{pdf_idx}/{total_pdfs}] {pdf_path.name} ✓ ({file_chunk_count} chunks)"
                                    print(f"\r{status_line:<80}")
                                else:
                                    print(f"  ✓ complete ({file_chunk_count} chunks)", flush=True)

            else:
                # Sequential mode: Process PDFs one at a time
                for pdf_idx, pdf_path in enumerate(pdf_files, 1):
                    points, file_chunk_count, error = self._process_single_pdf(
                        pdf_path,
                        collection_name,
                        model_name,
                        model_config,
                        chunker,
                        point_id,
                        verbose,
                        pdf_idx,
                        total_pdfs
                    )

                    if error:
                        # Show error
                        if not verbose:
                            status_line = f"[{pdf_idx}/{total_pdfs}] {pdf_path.name} ✗ ({error.split(':')[0]})"
                            print(f"\r{status_line:<80}")
                        else:
                            print(f"  ✗ ERROR: {error}", file=sys.stderr, flush=True)
                        stats["errors"] += 1
                    else:
                        # Points are already uploaded via streaming (points list will be empty)
                        # Just update stats and show completion
                        if file_chunk_count > 0:
                            stats["chunks"] += file_chunk_count
                            stats["files"] += 1
                            point_id += file_chunk_count

                            # Complete
                            if not verbose:
                                status_line = f"[{pdf_idx}/{total_pdfs}] {pdf_path.name} ✓ ({file_chunk_count} chunks)"
                                print(f"\r{status_line:<80}")
                            else:
                                print(f"  ✓ complete ({file_chunk_count} chunks)", flush=True)

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

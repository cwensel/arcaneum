"""Batch upload orchestrator for PDF indexing (RDR-004)."""

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type
import logging
import os
import sys

from ..embeddings.client import EmbeddingClient
from .common.sync import MetadataBasedSync, compute_file_hash
from .pdf.extractor import PDFExtractor
from .pdf.ocr import OCREngine
from .pdf.chunker import PDFChunker

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
        ocr_enabled: bool = False,
        ocr_engine: str = 'tesseract',
        ocr_language: str = 'eng',
        ocr_threshold: int = 100,
        batch_across_files: bool = False,
    ):
        """Initialize batch uploader.

        Args:
            qdrant_client: Qdrant client instance
            embedding_client: Embedding client instance
            batch_size: Number of points per batch
            parallel_workers: Number of parallel upload workers
            max_retries: Maximum retry attempts for failed uploads
            ocr_enabled: Enable OCR for scanned PDFs
            ocr_engine: OCR engine ('tesseract' or 'easyocr')
            ocr_language: OCR language code
            ocr_threshold: Trigger OCR if text < N characters
        """
        self.qdrant = qdrant_client
        self.embeddings = embedding_client
        self.batch_size = batch_size
        self.parallel_workers = parallel_workers
        self.max_retries = max_retries

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
                image_scale=2.0
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

        # Process PDFs with clean progress bar
        batch = []
        point_id = self._get_next_point_id(collection_name)
        stats = {"files": 0, "chunks": 0, "errors": 0}

        try:
            total_pdfs = len(pdf_files)

            # Use tqdm for verbose mode, simple print for minimal mode
            if verbose:
                pbar = tqdm(total=total_pdfs, desc="Indexing PDFs", unit="file",
                           bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            else:
                pbar = None

            for pdf_idx, pdf_path in enumerate(pdf_files, 1):
                try:
                    # Track chunks created and uploaded for this PDF
                    file_chunk_count = 0
                    chunks_uploaded_before = stats["chunks"]

                    # Minimal mode: just show filename, let processing happen quietly
                    if not verbose:
                        print(f"[{pdf_idx}/{total_pdfs}] {pdf_path.name}", end="", flush=True)
                    else:
                        print(f"\n[{pdf_idx}/{total_pdfs}] Processing {pdf_path.name}...", flush=True)
                        print(f"  → Computing hash", flush=True)

                    # Compute file hash for incremental indexing
                    file_hash = compute_file_hash(pdf_path)

                    # Extract text (suppress stderr warnings in non-verbose mode)
                    if verbose:
                        print(f"  → Extracting text", flush=True)
                        text, extract_meta = self.extractor.extract(pdf_path)
                        print(f"  → Extracted {len(text)} chars", flush=True)
                    else:
                        # Suppress PyMuPDF warnings to stderr (os already imported at module level)
                        stderr_fd = sys.stderr.fileno()
                        with open(os.devnull, 'w') as devnull:
                            old_stderr = os.dup(stderr_fd)
                            os.dup2(devnull.fileno(), stderr_fd)
                            try:
                                text, extract_meta = self.extractor.extract(pdf_path)
                            finally:
                                os.dup2(old_stderr, stderr_fd)
                                os.close(old_stderr)

                    # Check if OCR needed
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
                            # Show OCR indicator inline
                            print(f" [OCR:{page_count}p]", end="", flush=True)

                        text, ocr_meta = self.ocr.process_pdf(pdf_path, verbose=verbose)
                        extract_meta.update(ocr_meta)

                    # Chunk text with file metadata (including hash for sync)
                    base_metadata = {
                        'filename': pdf_path.name,
                        'file_path': str(pdf_path),
                        'file_hash': file_hash,  # For incremental sync
                        'file_size': pdf_path.stat().st_size,
                        'store_type': 'pdf',
                        **extract_meta
                    }

                    # Chunking (silent in minimal mode)
                    if verbose:
                        print(f"  → Chunking text ({len(text)} chars)", flush=True)

                    chunks = chunker.chunk(text, base_metadata)
                    file_chunk_count = len(chunks)

                    if verbose:
                        print(f"  → Created {file_chunk_count} chunks", flush=True)

                    # Generate embeddings in batches (handle locally to maintain line control)
                    texts = [chunk.text for chunk in chunks]
                    embeddings = []

                    if verbose:
                        print(f"  → Embedding {file_chunk_count} chunks", flush=True)

                    # Batch embedding locally (like source code indexing does)
                    EMBEDDING_BATCH_SIZE = 100
                    for batch_start in range(0, file_chunk_count, EMBEDDING_BATCH_SIZE):
                        batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, file_chunk_count)
                        batch_texts = texts[batch_start:batch_end]

                        # Update same line with batch progress
                        if not verbose and file_chunk_count > EMBEDDING_BATCH_SIZE:
                            print(
                                f"\r[{pdf_idx}/{total_pdfs}] {pdf_path.name} → embedding {batch_end}/{file_chunk_count}" + " "*20,
                                end="",
                                flush=True
                            )

                        # Call EmbeddingClient with SMALL batch (won't re-batch)
                        batch_embeddings = self.embeddings.embed(batch_texts, model_name)
                        embeddings.extend(batch_embeddings)

                    # Create points (silent in minimal mode)

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
                                print(f"  → Uploading batch of {len(batch)} chunks", flush=True)
                            self._upload_batch(collection_name, batch)
                            stats["chunks"] += len(batch)
                            batch = []

                    # Upload this PDF's chunks immediately (atomic mode - default)
                    if not self.batch_across_files and len(batch) > 0:
                        # Show upload stage
                        if not verbose:
                            print(f"\r[{pdf_idx}/{total_pdfs}] {pdf_path.name} → uploading" + " "*30, end="", flush=True)
                        else:
                            print(f"  → Uploading {len(batch)} chunks", flush=True)

                        self._upload_batch(collection_name, batch)
                        stats["chunks"] += len(batch)
                        batch = []

                    if verbose:
                        print(f"  ✓ Completed {pdf_path.name}", flush=True)

                    stats["files"] += 1

                    # Show completion with created vs uploaded counts
                    chunks_uploaded_this_file = stats["chunks"] - chunks_uploaded_before

                    if not verbose:
                        if self.batch_across_files:
                            # Batching mode: may have chunks pending
                            chunks_in_batch = len([p for p in batch if p.payload.get('filename') == pdf_path.name])
                            if chunks_in_batch > 0:
                                print(f" ✓ ({file_chunk_count} chunks, {chunks_in_batch} pending)")
                            else:
                                print(f" ✓ ({file_chunk_count} chunks)")
                        else:
                            # Atomic mode: all chunks should be uploaded immediately
                            if file_chunk_count == chunks_uploaded_this_file:
                                print(f" ✓ ({file_chunk_count} chunks)")
                            else:
                                print(f" ⚠ ({file_chunk_count} created, {chunks_uploaded_this_file} uploaded - FAILED)")
                    else:
                        chunks_in_batch = len([p for p in batch if p.payload.get('filename') == pdf_path.name])
                        print(f"  Chunks: {file_chunk_count} created, {chunks_uploaded_this_file} uploaded, {chunks_in_batch} in batch", flush=True)
                        if pbar:
                            pbar.update(1)

                except Exception as e:
                    # Show detailed error
                    error_msg = str(e)
                    if not verbose:
                        # Minimal: show error type
                        error_type = type(e).__name__
                        print(f" ✗ ({error_type})")
                    else:
                        # Verbose: show full error
                        print(f"  ✗ ERROR: {error_msg}", file=sys.stderr, flush=True)
                        if pbar:
                            pbar.update(1)

                    stats["errors"] += 1
                    continue

            # Upload remaining batch (only needed if batching across files)
            if batch:
                if verbose or not self.batch_across_files:
                    print(f"\n  → Uploading final batch: {len(batch)} chunks", flush=True)
                try:
                    self._upload_batch(collection_name, batch)
                    stats["chunks"] += len(batch)
                    if verbose:
                        print(f"  ✓ Final batch uploaded", flush=True)
                except Exception as e:
                    print(f"\nERROR: Final batch upload failed: {e}", file=sys.stderr)
                    print(f"  Lost {len(batch)} chunks from batch buffer!", file=sys.stderr)

            # Close progress bar if used
            if verbose and pbar:
                pbar.close()

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
            logger.info("=" * 60)
        else:
            # Normal mode: clean summary
            print(f"✅ Indexed {stats['files']} PDFs → {stats['chunks']} chunks")
            if stats['errors'] > 0:
                print(f"⚠️  {stats['errors']} errors")

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

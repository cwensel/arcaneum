"""Batch upload orchestrator for PDF indexing (RDR-004)."""

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type
import logging
import os

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
        force_reindex: bool = False
    ) -> Dict:
        """Index PDFs in directory with incremental sync.

        Args:
            pdf_dir: Directory containing PDFs
            collection_name: Qdrant collection name
            model_name: Embedding model to use
            model_config: Model configuration (chunk_size, overlap, etc.)
            force_reindex: Bypass metadata sync and reindex all files

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
        print(f"🔍 Scanning collection for existing files...")
        if force_reindex:
            pdf_files = all_pdf_files
            logger.info(f"Force reindex: processing all {len(pdf_files)} PDFs")
            print(f"🔄 Force reindex: {len(pdf_files)} PDFs to process")
        else:
            pdf_files = self.sync.get_unindexed_files(collection_name, all_pdf_files)
            skipped = len(all_pdf_files) - len(pdf_files)
            logger.info(f"Incremental sync: {len(pdf_files)} new/modified, {skipped} already indexed")

            # Always show sync status to user
            print(f"📊 Found {len(all_pdf_files)} PDFs: {len(pdf_files)} new/modified, {skipped} already indexed")

        if not pdf_files:
            logger.info("No PDFs to index")
            print("✅ All PDFs are up to date")
            return {"files": 0, "chunks": 0, "errors": 0}

        print()

        # Process PDFs with clean progress bar
        batch = []
        point_id = self._get_next_point_id(collection_name)
        stats = {"files": 0, "chunks": 0, "errors": 0}

        try:
            with tqdm(total=len(pdf_files), desc="Indexing PDFs", unit="file",
                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

                for pdf_path in pdf_files:
                    try:
                        # Compute file hash for incremental indexing
                        file_hash = compute_file_hash(pdf_path)

                        # Extract text
                        text, extract_meta = self.extractor.extract(pdf_path)

                        # Check if OCR needed
                        if self.ocr_enabled and len(text) < self.ocr_threshold:
                            logger.debug(f"Triggering OCR for {pdf_path.name}")
                            text, ocr_meta = self.ocr.process_pdf(pdf_path)
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

                        chunks = chunker.chunk(text, base_metadata)

                        # Generate embeddings
                        texts = [chunk.text for chunk in chunks]
                        embeddings = self.embeddings.embed(texts, model_name)

                        # Create points
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

                            # Upload when batch full
                            if len(batch) >= self.batch_size:
                                self._upload_batch(collection_name, batch)
                                stats["chunks"] += len(batch)
                                batch = []

                        stats["files"] += 1
                        pbar.update(1)

                    except Exception as e:
                        logger.error(f"Failed: {pdf_path.name}: {e}")
                        stats["errors"] += 1
                        pbar.update(1)
                        continue

                # Upload remaining batch
                if batch:
                    self._upload_batch(collection_name, batch)
                    stats["chunks"] += len(batch)

        except KeyboardInterrupt:
            print("\n\n⚠️  Indexing interrupted by user")
            print(f"📊 Partial progress: {stats['files']} files indexed, {stats['chunks']} chunks uploaded")
            if stats['errors'] > 0:
                print(f"⚠️  {stats['errors']} errors occurred")
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

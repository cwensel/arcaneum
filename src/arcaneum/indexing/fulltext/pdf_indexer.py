"""PDF full-text indexer for MeiliSearch (RDR-010).

Indexes PDFs to MeiliSearch for exact phrase and keyword search,
complementary to Qdrant's semantic search (RDR-004).
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from rich.progress import Progress, TaskID, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from ..pdf.extractor import PDFExtractor
from ..pdf.ocr import OCREngine
from ...fulltext.client import FullTextClient

logger = logging.getLogger(__name__)


class PDFFullTextIndexer:
    """Index PDFs to MeiliSearch for full-text search.

    This class reuses the RDR-004 extraction pipeline (PDFExtractor + OCREngine)
    and indexes extracted text to MeiliSearch at page-level granularity.

    Key differences from Qdrant indexing:
    - No embedding generation (text-only)
    - Page-level documents (not token-aware chunks)
    - Larger batch size (1000 vs 100-300 for vectors)

    Attributes:
        meili_client: MeiliSearch client instance
        index_name: Target MeiliSearch index name
        batch_size: Documents per batch upload (default: 1000)
        ocr_enabled: Whether to use OCR for scanned PDFs
    """

    def __init__(
        self,
        meili_client: FullTextClient,
        index_name: str,
        ocr_enabled: bool = True,
        ocr_language: str = 'eng',
        ocr_workers: Optional[int] = None,
        batch_size: int = 1000,
        markdown_conversion: bool = True,
    ):
        """Initialize PDF full-text indexer.

        Args:
            meili_client: MeiliSearch client instance
            index_name: Target MeiliSearch index name
            ocr_enabled: Enable OCR for scanned PDFs (default: True)
            ocr_language: OCR language code (default: 'eng')
            ocr_workers: Parallel OCR workers (None = cpu_count)
            batch_size: Documents per batch upload (default: 1000)
            markdown_conversion: Use markdown conversion (default: True)
        """
        self.meili_client = meili_client
        self.index_name = index_name
        self.batch_size = batch_size
        self.ocr_enabled = ocr_enabled

        # Reuse RDR-004 extraction components
        self.pdf_extractor = PDFExtractor(
            fallback_enabled=True,
            table_validation=True,
            markdown_conversion=markdown_conversion,
        )

        self.ocr_engine = OCREngine(
            engine='tesseract',
            language=ocr_language,
            confidence_threshold=60.0,
            ocr_workers=ocr_workers,
        ) if ocr_enabled else None

    def index_pdf(
        self,
        pdf_path: Path,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Index a single PDF to MeiliSearch.

        Extracts text using RDR-004 pipeline, builds page-level documents,
        and uploads to MeiliSearch.

        Args:
            pdf_path: Path to PDF file
            verbose: Show detailed progress

        Returns:
            Dict with indexing statistics:
            - pdf_path: Path to indexed PDF
            - page_count: Number of pages indexed
            - task_uid: MeiliSearch task ID
            - extraction_method: Method used for extraction
        """
        # Phase 1: Extract text (REUSE RDR-004)
        text, metadata = self.pdf_extractor.extract(pdf_path)

        # Check if OCR needed (text too short suggests image PDF)
        if self.ocr_engine and len(text.strip()) < 100:
            if verbose:
                logger.info(f"OCR triggered for {pdf_path.name} (text < 100 chars)")
            text, ocr_metadata = self.ocr_engine.process_pdf(pdf_path, verbose=verbose)
            metadata.update(ocr_metadata)

        # Phase 2: Prepare MeiliSearch documents (page-level)
        documents = self._build_meilisearch_documents(
            pdf_path, text, metadata
        )

        # Phase 3: Upload to MeiliSearch
        if documents:
            result = self.meili_client.add_documents_sync(
                index_name=self.index_name,
                documents=documents
            )
            # Result can be a Task object (Pydantic) or dict
            if hasattr(result, 'uid'):
                task_uid = result.uid
            elif isinstance(result, dict):
                task_uid = result.get('taskUid', result.get('uid', 'unknown'))
            else:
                task_uid = 'unknown'
        else:
            task_uid = None

        return {
            'pdf_path': str(pdf_path),
            'page_count': len(documents),
            'task_uid': task_uid,
            'extraction_method': metadata.get('extraction_method', 'unknown'),
        }

    def _build_meilisearch_documents(
        self,
        pdf_path: Path,
        full_text: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build MeiliSearch documents (one per page).

        Creates page-level documents with shared metadata schema (RDR-009).

        Args:
            pdf_path: Path to PDF file
            full_text: Extracted text content
            metadata: Extraction metadata

        Returns:
            List of MeiliSearch document dictionaries
        """
        # Compute file hash for change detection
        file_hash = self._compute_file_hash(pdf_path)

        # Get page count
        page_count = metadata.get('page_count', 1)

        # Split text into pages
        pages = self._split_into_pages(full_text, page_count, metadata)

        documents = []
        for page_num, page_text in enumerate(pages, start=1):
            # Skip empty pages
            if not page_text.strip():
                continue

            # Generate unique ID (file stem + page number)
            # Use hash of file path for uniqueness across directories
            path_hash = hashlib.md5(str(pdf_path.absolute()).encode()).hexdigest()[:8]
            # Sanitize stem: MeiliSearch IDs only allow alphanumeric, hyphen, underscore
            # Replace invalid characters and truncate to avoid 511 byte limit
            sanitized_stem = re.sub(r'[^a-zA-Z0-9_-]', '_', pdf_path.stem)[:200]
            doc_id = f"{sanitized_stem}_{path_hash}_p{page_num}"

            doc = {
                # Primary key
                'id': doc_id,

                # Searchable content
                'content': page_text,
                'filename': pdf_path.name,

                # Filterable metadata (shared with Qdrant, RDR-009)
                'file_path': str(pdf_path.absolute()),
                'page_number': page_num,
                'file_hash': file_hash,
                'extraction_method': metadata.get('extraction_method', 'unknown'),
                'is_image_pdf': metadata.get('is_image_pdf', False),

                # Additional metadata
                'file_size': metadata.get('file_size'),
                'page_count': page_count,
                'document_type': 'pdf',
            }

            # Add OCR-specific metadata if present
            if 'ocr_confidence' in metadata:
                doc['ocr_confidence'] = metadata['ocr_confidence']
                doc['ocr_language'] = metadata.get('ocr_language', 'eng')

            documents.append(doc)

        return documents

    def _compute_file_hash(self, pdf_path: Path) -> str:
        """Compute SHA-256 hash of PDF for change detection.

        Streams file in 8KB chunks for memory efficiency.

        Args:
            pdf_path: Path to PDF file

        Returns:
            SHA-256 hex digest
        """
        sha256 = hashlib.sha256()
        with open(pdf_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _split_into_pages(
        self,
        full_text: str,
        page_count: int,
        metadata: Dict[str, Any]
    ) -> List[str]:
        """Split full text into pages.

        Uses page boundaries from metadata if available (preferred),
        otherwise falls back to form feed character splitting.

        Args:
            full_text: Concatenated text from all pages
            page_count: Expected number of pages
            metadata: Extraction metadata (may contain page_boundaries)

        Returns:
            List of text strings, one per page
        """
        # Check for page boundaries from extractor
        page_boundaries = metadata.get('page_boundaries', [])

        if page_boundaries:
            # Use precise page boundaries from extraction
            pages = []
            for i, boundary in enumerate(page_boundaries):
                start_char = boundary.get('start_char', 0)
                length = boundary.get('page_text_length', 0)
                end_char = start_char + length

                # Get next boundary start or end of text
                if i + 1 < len(page_boundaries):
                    next_start = page_boundaries[i + 1].get('start_char', len(full_text))
                    end_char = min(end_char, next_start)

                page_text = full_text[start_char:end_char]
                pages.append(page_text)

            return pages

        # Fallback: split by form feed character (used by some extractors)
        if '\f' in full_text:
            pages = full_text.split('\f')
        else:
            # Try splitting by page markers in markdown output
            # PyMuPDF4LLM adds "-----" between pages
            page_marker_pattern = r'\n-{5,}\n'
            if re.search(page_marker_pattern, full_text):
                pages = re.split(page_marker_pattern, full_text)
            else:
                # Last resort: treat entire text as single page
                pages = [full_text]

        # Adjust to expected page count
        if len(pages) < page_count:
            # Pad with empty pages
            pages.extend([''] * (page_count - len(pages)))
        elif len(pages) > page_count:
            # Truncate excess (shouldn't happen normally)
            pages = pages[:page_count]

        return pages

    def index_directory(
        self,
        directory: Path,
        recursive: bool = True,
        force_reindex: bool = False,
        verbose: bool = False,
        file_list: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        """Index all PDFs in a directory to MeiliSearch.

        Args:
            directory: Directory containing PDF files
            recursive: Search subdirectories (default: True)
            force_reindex: Reindex all files even if already indexed
            verbose: Show detailed progress
            file_list: Optional explicit list of PDF files to index

        Returns:
            Dict with indexing statistics
        """
        # Discover PDFs
        if file_list:
            pdf_files = [f for f in file_list if f.suffix.lower() == '.pdf']
        else:
            pattern = '**/*.pdf' if recursive else '*.pdf'
            pdf_files = list(directory.glob(pattern))

        stats = {
            'total_pdfs': len(pdf_files),
            'indexed_pdfs': 0,
            'skipped_pdfs': 0,
            'failed_pdfs': 0,
            'total_pages': 0,
            'errors': [],
        }

        if not pdf_files:
            logger.info("No PDF files found to index")
            return stats

        # Index PDFs with progress tracking
        # Use transient=False so progress bar stays visible, and use console.print
        # for verbose output so it appears cleanly above the progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task(
                "[cyan]Indexing PDFs to MeiliSearch",
                total=len(pdf_files)
            )

            for pdf_path in pdf_files:
                try:
                    # Check if already indexed (change detection)
                    if not force_reindex and self._is_already_indexed(pdf_path):
                        stats['skipped_pdfs'] += 1
                        if verbose:
                            progress.console.print(
                                f"  [dim]Skipped:[/dim] {pdf_path.name} [dim](already indexed)[/dim]"
                            )
                        progress.update(task, advance=1)
                        continue

                    # Index PDF
                    result = self.index_pdf(pdf_path, verbose=verbose)
                    stats['indexed_pdfs'] += 1
                    stats['total_pages'] += result['page_count']

                    if verbose:
                        progress.console.print(
                            f"  [green]Indexed:[/green] {pdf_path.name} "
                            f"[dim]({result['page_count']} pages)[/dim]"
                        )

                except Exception as e:
                    progress.console.print(
                        f"  [red]Failed:[/red] {pdf_path.name}: {e}"
                    )
                    stats['failed_pdfs'] += 1
                    stats['errors'].append({
                        'file': str(pdf_path),
                        'error': str(e)
                    })

                progress.update(task, advance=1)

        return stats

    def _is_already_indexed(self, pdf_path: Path) -> bool:
        """Check if PDF already indexed (change detection).

        Queries MeiliSearch for existing document with same file_path + file_hash.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if PDF is already indexed with same content
        """
        file_hash = self._compute_file_hash(pdf_path)
        file_path_str = str(pdf_path.absolute())

        try:
            # Query for existing documents with matching path and hash
            # MeiliSearch filter syntax uses = for equality
            filter_expr = f'file_path = "{file_path_str}" AND file_hash = "{file_hash}"'

            results = self.meili_client.search(
                index_name=self.index_name,
                query='',  # Empty query to just filter
                filter=filter_expr,
                limit=1
            )

            # Check actual hits returned, not estimatedTotalHits
            # estimatedTotalHits can be unreliable for filtered queries
            hits = results.get('hits', [])
            logger.debug(
                f"Change detection for {pdf_path.name}: "
                f"hits={len(hits)}, filter={filter_expr[:100]}..."
            )
            return len(hits) > 0

        except Exception as e:
            # If query fails, assume not indexed
            # Common cause: filterable attributes not configured
            logger.warning(
                f"Change detection query failed for {pdf_path.name}: {e}. "
                f"Ensure index has file_path and file_hash as filterable attributes."
            )
            return False

    def delete_pdf_documents(self, pdf_path: Path) -> bool:
        """Delete all documents for a PDF from the index.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if deletion succeeded
        """
        file_path_str = str(pdf_path.absolute())

        try:
            # Get all document IDs for this file
            filter_expr = f'file_path = "{file_path_str}"'

            results = self.meili_client.search(
                index_name=self.index_name,
                query='',
                filter=filter_expr,
                limit=1000  # Get all pages
            )

            hits = results.get('hits', [])
            if not hits:
                return True

            # Delete documents by ID
            doc_ids = [hit['id'] for hit in hits]
            index = self.meili_client.get_index(self.index_name)
            task = index.delete_documents(doc_ids)
            self.meili_client.client.wait_for_task(task.task_uid)

            return True

        except Exception as e:
            logger.error(f"Failed to delete documents for {pdf_path}: {e}")
            return False

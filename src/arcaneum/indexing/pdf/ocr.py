"""OCR engine using Tesseract (RDR-004)."""

import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from multiprocessing import cpu_count
import os
import io
import logging
import tempfile
import gc

from ...utils.memory import calculate_safe_workers
from ..common.multiprocessing import get_mp_context, worker_init

logger = logging.getLogger(__name__)


def _ocr_single_page_worker(
    page_image_bytes: bytes,
    page_num: int,
    language: str,
    confidence_threshold: float,
    image_scale: float
) -> Tuple[int, str, float]:
    """Process a single PDF page with OCR (module-level for ProcessPoolExecutor).

    Args:
        page_image_bytes: PIL Image serialized as PNG bytes
        page_num: Page number (1-indexed)
        language: Language code
        confidence_threshold: Minimum confidence score
        image_scale: Scale factor for accuracy

    Returns:
        Tuple of (page_num, text, confidence)
    """
    # Set low priority UNLESS disabled by --not-nice flag (arcaneum-mql4)
    if os.environ.get('ARCANEUM_DISABLE_WORKER_NICE') != '1':
        try:
            if hasattr(os, 'nice'):
                os.nice(10)  # Background priority for OCR workers
        except Exception:
            pass  # Ignore if we can't set priority

    image = None
    img_array = None
    gray = None
    thresh = None
    denoised = None

    try:
        # Disable OpenCV threading to prevent fork-related crashes on macOS (segfault in cv2.resize)
        # See: https://github.com/opencv/opencv/issues/5150
        cv2.setNumThreads(0)

        # Deserialize image
        image = Image.open(io.BytesIO(page_image_bytes))

        # Preprocess image
        img_array = np.array(image)

        # Scale image
        if image_scale != 1.0:
            width = int(img_array.shape[1] * image_scale)
            height = int(img_array.shape[0] * image_scale)
            img_array = cv2.resize(img_array, (width, height), interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Denoise
        denoised = cv2.medianBlur(thresh, 3)

        # Tesseract OCR
        data = pytesseract.image_to_data(
            denoised,
            lang=language,
            config='--psm 3 --oem 1',
            output_type=pytesseract.Output.DICT
        )

        # Extract text with confidence filtering
        filtered_text = []
        confidences = []

        for i, conf in enumerate(data['conf']):
            if conf == -1:
                continue
            if conf >= confidence_threshold:
                text = data['text'][i]
                if text.strip():
                    filtered_text.append(text)
                    confidences.append(conf)

        text = ' '.join(filtered_text)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        result = (page_num, text, avg_conf)

        # Explicit cleanup of large objects before return
        del image, img_array, gray, thresh, denoised

        return result

    except Exception as e:
        logger.error(f"OCR failed for page {page_num}: {e}")
        return (page_num, "", 0.0)

    finally:
        # Ensure cleanup even on exception
        try:
            del image, img_array, gray, thresh, denoised
        except:
            pass
        # Force garbage collection in worker
        gc.collect()


class OCREngine:
    """OCR engine using Tesseract."""

    def __init__(
        self,
        language: str = 'eng',
        confidence_threshold: float = 60.0,
        image_dpi: int = 300,
        image_scale: float = 2.0,
        ocr_workers: Optional[int] = None,
        page_batch_size: int = 20,
        max_memory_gb: Optional[float] = None,
        page_timeout: int = 60
    ):
        """Initialize OCR engine.

        Args:
            language: Language code (e.g., 'eng', 'fra', 'spa')
            confidence_threshold: Minimum confidence score (0-100 for Tesseract)
            image_dpi: DPI for PDF to image conversion
            image_scale: Scale factor for OCR accuracy (2x recommended)
            ocr_workers: Number of parallel workers for page processing (None = cpu_count)
            page_batch_size: Number of pages to process in each batch (for memory efficiency)
            max_memory_gb: Maximum memory to use in GB (None = auto-calculate from available)
            page_timeout: Timeout in seconds for processing a single page (default: 60)
        """
        self.language = language
        self.confidence_threshold = confidence_threshold
        self.image_dpi = image_dpi
        self.image_scale = image_scale
        self.page_batch_size = page_batch_size
        self.max_memory_gb = max_memory_gb
        self.page_timeout = page_timeout

        # Configure parallel workers with memory-aware limits
        # Estimate 50MB per OCR worker (processed images, OCR models)
        requested_workers = cpu_count() if ocr_workers is None else ocr_workers
        safe_ocr_workers, warning = calculate_safe_workers(
            requested_workers=requested_workers,
            estimated_memory_per_worker_mb=50,
            max_memory_gb=max_memory_gb,
            min_workers=1
        )
        self.ocr_workers = safe_ocr_workers

        if warning:
            logger.warning(f"OCR: {warning}")

    def process_pdf(self, pdf_path: Path, verbose: bool = False) -> Tuple[str, dict]:
        """Perform OCR on PDF with parallel page processing and memory-efficient batching.

        Args:
            pdf_path: Path to PDF file
            verbose: If True, show per-page progress

        Returns:
            Tuple of (text, metadata)
        """
        # Get total page count without loading images
        pdf_info = pdfinfo_from_path(pdf_path)
        total_pages = pdf_info['Pages']

        if verbose:
            print(f"  → OCR: Processing {total_pages} pages in batches of {self.page_batch_size} (parallel, {self.ocr_workers} workers)...", flush=True)

        # Process pages in batches to avoid memory exhaustion
        page_results = {}  # {page_num: (text, confidence)}

        for batch_start in range(1, total_pages + 1, self.page_batch_size):
            batch_end = min(batch_start + self.page_batch_size - 1, total_pages)

            if verbose:
                print(f"  → OCR: Batch pages {batch_start}-{batch_end}...", flush=True)

            # Process this batch with memory-efficient approach
            batch_results = self._process_batch(
                pdf_path, batch_start, batch_end, total_pages, verbose
            )
            page_results.update(batch_results)

            # Explicit cleanup between batches
            del batch_results
            gc.collect()

        # Assemble results in page order
        text_parts = []
        confidence_scores = []

        for page_num in sorted(page_results.keys()):
            page_text, page_conf = page_results[page_num]
            if page_text.strip():
                text_parts.append(page_text)
                confidence_scores.append(page_conf)

        text = '\n'.join(text_parts)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        metadata = {
            'extraction_method': 'ocr_tesseract',
            'is_image_pdf': True,
            'ocr_confidence': avg_confidence,
            'ocr_language': self.language,
            'page_count': total_pages,
            'ocr_pages_processed': len(text_parts),
        }

        if verbose:
            print(f"  → OCR complete: {len(text)} chars, avg conf: {avg_confidence:.0f}%", flush=True)

        return text, metadata

    def _process_batch(
        self,
        pdf_path: Path,
        first_page: int,
        last_page: int,
        total_pages: int,
        verbose: bool
    ) -> Dict[int, Tuple[str, float]]:
        """Process a batch of PDF pages with memory-efficient image handling.

        Args:
            pdf_path: Path to PDF file
            first_page: First page number (1-indexed)
            last_page: Last page number (1-indexed)
            total_pages: Total page count
            verbose: If True, show per-page progress

        Returns:
            Dict mapping page_num to (text, confidence)
        """
        batch_results = {}

        # Use temporary directory for image files (auto-cleanup)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert batch to images on disk (not in memory)
            # Using paths_only=False here because we need to serialize for workers
            # But using output_folder keeps memory usage low
            images = convert_from_path(
                pdf_path,
                dpi=self.image_dpi,
                first_page=first_page,
                last_page=last_page,
                output_folder=temp_dir,
                fmt='jpeg',  # Use JPEG instead of PPM (30MB → ~2MB per page)
                thread_count=min(4, self.ocr_workers)  # Moderate threading for I/O
            )

            # Serialize images for worker processes
            serialized_images = []
            for idx, image in enumerate(images):
                page_num = first_page + idx
                # Convert PIL Image to PNG bytes
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_bytes = img_buffer.getvalue()
                serialized_images.append((page_num, img_bytes))

                # Close image to free memory immediately
                image.close()

            # Clear images list
            del images
            gc.collect()

            # Process pages in parallel using multiprocessing.Pool with fork context
            # Use shared context and signal handler for proper Ctrl-C handling
            ctx = get_mp_context()
            pool = None
            async_results = []

            try:
                pool = ctx.Pool(
                    processes=self.ocr_workers,
                    initializer=worker_init
                )
                # Submit all page jobs for this batch
                for page_num, img_bytes in serialized_images:
                    async_result = pool.apply_async(
                        _ocr_single_page_worker,
                        (img_bytes, page_num, self.language,
                         self.confidence_threshold, self.image_scale)
                    )
                    async_results.append((async_result, page_num))

                # Collect results as they complete
                for async_result, page_num in async_results:
                    try:
                        result_page_num, page_text, page_conf = async_result.get(timeout=self.page_timeout)
                        batch_results[result_page_num] = (page_text, page_conf)

                        # Show per-page progress
                        if verbose and page_text.strip():
                            print(f"  → OCR: Page {result_page_num}/{total_pages} ({len(page_text)} chars, conf: {page_conf:.0f}%)", flush=True)

                    except TimeoutError:
                        logger.error(f"OCR timeout for page {page_num} (exceeded {self.page_timeout}s)")
                        batch_results[page_num] = ("", 0.0)
                    except Exception as e:
                        logger.error(f"OCR failed for page {page_num}: {e}")
                        batch_results[page_num] = ("", 0.0)

            except KeyboardInterrupt:
                logger.warning("Interrupted - terminating OCR workers...")
                raise
            finally:
                if pool:
                    pool.terminate()
                    pool.join()
                # Clear serialized images
                del serialized_images, async_results
                gc.collect()

        # temp_dir is automatically cleaned up here

        return batch_results

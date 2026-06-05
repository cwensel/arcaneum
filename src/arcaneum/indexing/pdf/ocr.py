"""OCR engine using Tesseract (RDR-004)."""

import gc
import io
import logging
import os
import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image

from ...utils.memory import calculate_safe_workers
from ..common.multiprocessing import get_mp_context, worker_init

logger = logging.getLogger(__name__)


def _load_cv2():
    """Import OpenCV only when OCR preprocessing actually needs it.

    On macOS, importing OpenCV alongside packages that load PyAV can emit
    Objective-C duplicate class warnings from bundled FFmpeg libraries. Normal
    text-PDF extraction does not need OpenCV, so keep it out of the import path
    until OCR is running.
    """
    import cv2

    return cv2


def _safe_confidence(confidence: Any) -> Optional[float]:
    """Normalize Tesseract confidence values, skipping structural rows."""
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return None
    if conf < 0:
        return None
    return conf


def _ocr_data_value(data: Dict[str, List[Any]], key: str, index: int, default: Any) -> Any:
    values = data.get(key)
    if not values or index >= len(values):
        return default
    return values[index]


def _ocr_single_page_worker(
    page_image_bytes: bytes,
    page_num: int,
    language: str,
    confidence_threshold: float,
    image_scale: float,
) -> Tuple[int, str, float, Dict[str, Any]]:
    """Process a single PDF page with OCR (module-level for ProcessPoolExecutor).

    Args:
        page_image_bytes: PIL Image serialized as PNG bytes
        page_num: Page number (1-indexed)
        language: Language code
        confidence_threshold: Minimum confidence score
        image_scale: Scale factor for accuracy

    Returns:
        Tuple of (page_num, text, confidence, page_metadata)
    """
    # Set low priority UNLESS disabled by --not-nice flag (arcaneum-mql4)
    if os.environ.get("ARCANEUM_DISABLE_WORKER_NICE") != "1":
        try:
            if hasattr(os, "nice"):
                os.nice(10)  # Background priority for OCR workers
        except Exception:
            pass  # Ignore if we can't set priority

    image = None
    img_array = None
    gray = None
    thresh = None
    denoised = None

    try:
        cv2 = _load_cv2()

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
            denoised, lang=language, config="--psm 3 --oem 1", output_type=pytesseract.Output.DICT
        )

        # Extract all words with confidence metadata. Do not discard low-confidence
        # words: they can be the only recall signal in scanned/aged PDFs.
        lines: Dict[Tuple[int, int, int], List[Tuple[int, str]]] = {}
        confidences = []
        low_confidence_words = []
        word_count = 0

        for i, conf in enumerate(data["conf"]):
            conf_value = _safe_confidence(conf)
            if conf_value is None:
                continue
            text = data["text"][i]
            if not text.strip():
                continue

            block_num = int(_ocr_data_value(data, "block_num", i, 0) or 0)
            par_num = int(_ocr_data_value(data, "par_num", i, 0) or 0)
            line_num = int(_ocr_data_value(data, "line_num", i, 0) or 0)
            word_num = int(_ocr_data_value(data, "word_num", i, word_count + 1) or (word_count + 1))
            line_key = (block_num, par_num, line_num)
            lines.setdefault(line_key, []).append((word_num, text))

            word_count += 1
            confidences.append(conf_value)
            if conf_value < confidence_threshold:
                low_confidence_words.append(
                    {
                        "text": text,
                        "confidence": conf_value,
                    }
                )

        text_lines = []
        for line_key in sorted(lines.keys()):
            words = [word for _, word in sorted(lines[line_key], key=lambda item: item[0])]
            if words:
                text_lines.append(" ".join(words))

        text = "\n".join(text_lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        page_metadata = {
            "page_number": page_num,
            "confidence": avg_conf,
            "word_count": word_count,
            "low_confidence_word_count": len(low_confidence_words),
            "low_confidence_words": low_confidence_words,
            "failed": False,
        }

        result = (page_num, text, avg_conf, page_metadata)

        # Explicitly release large objects before return.
        image = img_array = gray = thresh = denoised = None

        return result

    except Exception as e:
        logger.error(f"OCR failed for page {page_num}: {e}")
        return (
            page_num,
            "",
            0.0,
            {
                "page_number": page_num,
                "confidence": 0.0,
                "word_count": 0,
                "low_confidence_word_count": 0,
                "low_confidence_words": [],
                "failed": True,
                "error": str(e),
            },
        )

    finally:
        # Ensure cleanup even on exception
        image = img_array = gray = thresh = denoised = None
        # Force garbage collection in worker
        gc.collect()


def _offset_page_boundaries(
    page_boundaries: List[Dict[str, Any]], offset: int
) -> List[Dict[str, Any]]:
    """Return page boundaries shifted by a character offset."""
    shifted = []
    for boundary in page_boundaries:
        shifted_boundary = dict(boundary)
        shifted_boundary["start_char"] = shifted_boundary.get("start_char", 0) + offset
        shifted.append(shifted_boundary)
    return shifted


def merge_extracted_text_with_ocr(
    extracted_text: str,
    extracted_metadata: Dict[str, Any],
    ocr_text: str,
    ocr_metadata: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Merge OCR fallback output without discarding the original extraction.

    Existing markdown/layout text remains first so chunking still sees the
    structured extraction. OCR is appended as recall enrichment and the merge
    strategy is recorded for downstream payloads and JSON reports.
    """
    original_text = extracted_text or ""
    original_metadata = dict(extracted_metadata or {})
    merged_metadata = dict(original_metadata)
    original_method = original_metadata.get("extraction_method", "unknown")
    ocr_method = ocr_metadata.get("extraction_method", "ocr_tesseract")

    if not ocr_text or not ocr_text.strip():
        merged_metadata.update(ocr_metadata)
        merged_metadata["original_extraction_method"] = original_method
        merged_metadata["extraction_method"] = (
            f"{original_method}+{ocr_method}" if original_text.strip() else ocr_method
        )
        merged_metadata["ocr_merge_strategy"] = "ocr_failed_no_text"
        merged_metadata["page_boundaries"] = original_metadata.get("page_boundaries", [])
        merged_metadata["original_text_length"] = len(original_text)
        merged_metadata["ocr_text_length"] = 0
        return original_text, merged_metadata

    if original_text.strip():
        separator = "\n\n[OCR fallback text]\n\n"
        merged_text = f"{original_text}{separator}{ocr_text}"
        original_boundaries = original_metadata.get("page_boundaries", [])
        ocr_boundaries = _offset_page_boundaries(
            ocr_metadata.get("page_boundaries", []),
            len(original_text) + len(separator),
        )
        merged_metadata.update(ocr_metadata)
        merged_metadata["original_extraction_method"] = original_method
        merged_metadata["extraction_method"] = f"{original_method}+{ocr_method}"
        merged_metadata["ocr_merge_strategy"] = "append_ocr_to_extracted_text"
        merged_metadata["page_boundaries"] = [*original_boundaries, *ocr_boundaries]
        merged_metadata["original_text_length"] = len(original_text)
        merged_metadata["ocr_text_length"] = len(ocr_text)
        return merged_text, merged_metadata

    merged_metadata.update(ocr_metadata)
    merged_metadata["original_extraction_method"] = original_method
    merged_metadata["extraction_method"] = ocr_method
    merged_metadata["ocr_merge_strategy"] = "ocr_only_empty_extraction"
    merged_metadata["original_text_length"] = 0
    merged_metadata["ocr_text_length"] = len(ocr_text)
    return ocr_text, merged_metadata


class OCREngine:
    """OCR engine using Tesseract."""

    def __init__(
        self,
        language: str = "eng",
        confidence_threshold: float = 60.0,
        image_dpi: int = 300,
        image_scale: float = 2.0,
        ocr_workers: Optional[int] = None,
        page_batch_size: int = 20,
        max_memory_gb: Optional[float] = None,
        page_timeout: int = 60,
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
            min_workers=1,
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
        total_pages = pdf_info["Pages"]

        if verbose:
            print(
                f"  → OCR: Processing {total_pages} pages in batches of "
                f"{self.page_batch_size} (parallel, {self.ocr_workers} workers)...",
                flush=True,
            )

        # Process pages in batches to avoid memory exhaustion
        page_results = {}  # {page_num: (text, confidence, page_metadata)}

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
        page_boundaries = []
        ocr_pages = []
        failed_pages = 0
        low_confidence_word_count = 0
        current_pos = 0

        for page_num in sorted(page_results.keys()):
            page_text, page_conf, page_metadata = page_results[page_num]
            ocr_pages.append(page_metadata)
            low_confidence_word_count += page_metadata.get("low_confidence_word_count", 0)
            if page_metadata.get("failed"):
                failed_pages += 1
            if page_text.strip():
                page_boundaries.append(
                    {
                        "page_number": page_num,
                        "start_char": current_pos,
                        "page_text_length": len(page_text),
                    }
                )
                text_parts.append(page_text)
                confidence_scores.append(page_conf)
                current_pos += len(page_text) + 1

        text = "\n".join(text_parts)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        metadata = {
            "extraction_method": "ocr_tesseract",
            "is_image_pdf": True,
            "ocr_confidence": avg_confidence,
            "ocr_language": self.language,
            "page_count": total_pages,
            "ocr_pages_processed": len(ocr_pages) - failed_pages,
            "ocr_pages_failed": failed_pages,
            "ocr_low_confidence_word_count": low_confidence_word_count,
            "ocr_pages": ocr_pages,
            "page_boundaries": page_boundaries,
        }

        if verbose:
            print(
                f"  → OCR complete: {len(text)} chars, avg conf: {avg_confidence:.0f}%", flush=True
            )

        return text, metadata

    def _process_batch(
        self, pdf_path: Path, first_page: int, last_page: int, total_pages: int, verbose: bool
    ) -> Dict[int, Tuple[str, float, Dict[str, Any]]]:
        """Process a batch of PDF pages with memory-efficient image handling.

        Args:
            pdf_path: Path to PDF file
            first_page: First page number (1-indexed)
            last_page: Last page number (1-indexed)
            total_pages: Total page count
            verbose: If True, show per-page progress

        Returns:
            Dict mapping page_num to (text, confidence, page_metadata)
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
                fmt="jpeg",  # Use JPEG instead of PPM (30MB → ~2MB per page)
                thread_count=min(4, self.ocr_workers),  # Moderate threading for I/O
            )

            # Serialize images for worker processes
            serialized_images = []
            for idx, image in enumerate(images):
                page_num = first_page + idx
                # Convert PIL Image to PNG bytes
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="PNG")
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
                pool = ctx.Pool(processes=self.ocr_workers, initializer=worker_init)
                # Submit all page jobs for this batch
                for page_num, img_bytes in serialized_images:
                    async_result = pool.apply_async(
                        _ocr_single_page_worker,
                        (
                            img_bytes,
                            page_num,
                            self.language,
                            self.confidence_threshold,
                            self.image_scale,
                        ),
                    )
                    async_results.append((async_result, page_num))

                # Collect results as they complete
                for async_result, page_num in async_results:
                    try:
                        result_page_num, page_text, page_conf, page_metadata = async_result.get(
                            timeout=self.page_timeout
                        )
                        batch_results[result_page_num] = (page_text, page_conf, page_metadata)

                        # Show per-page progress
                        if verbose and page_text.strip():
                            print(
                                f"  → OCR: Page {result_page_num}/{total_pages} "
                                f"({len(page_text)} chars, conf: {page_conf:.0f}%)",
                                flush=True,
                            )

                    except TimeoutError:
                        logger.error(
                            f"OCR timeout for page {page_num} (exceeded {self.page_timeout}s)"
                        )
                        batch_results[page_num] = (
                            "",
                            0.0,
                            {
                                "page_number": page_num,
                                "confidence": 0.0,
                                "word_count": 0,
                                "low_confidence_word_count": 0,
                                "low_confidence_words": [],
                                "failed": True,
                                "error": f"timeout after {self.page_timeout}s",
                            },
                        )
                    except Exception as e:
                        logger.error(f"OCR failed for page {page_num}: {e}")
                        batch_results[page_num] = (
                            "",
                            0.0,
                            {
                                "page_number": page_num,
                                "confidence": 0.0,
                                "word_count": 0,
                                "low_confidence_word_count": 0,
                                "low_confidence_words": [],
                                "failed": True,
                                "error": str(e),
                            },
                        )

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

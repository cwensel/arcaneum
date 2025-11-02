"""OCR engine supporting Tesseract and EasyOCR (RDR-004)."""

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import io
import logging

logger = logging.getLogger(__name__)


def _ocr_single_page_worker(
    page_image_bytes: bytes,
    page_num: int,
    engine: str,
    language: str,
    confidence_threshold: float,
    image_scale: float
) -> Tuple[int, str, float]:
    """Process a single PDF page with OCR (module-level for ProcessPoolExecutor).

    Args:
        page_image_bytes: PIL Image serialized as PNG bytes
        page_num: Page number (1-indexed)
        engine: OCR engine ('tesseract' or 'easyocr')
        language: Language code
        confidence_threshold: Minimum confidence score
        image_scale: Scale factor for accuracy

    Returns:
        Tuple of (page_num, text, confidence)
    """
    try:
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

        # Perform OCR based on engine
        if engine == 'tesseract':
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

        else:  # easyocr
            import easyocr
            # Create reader per worker (can't pickle Reader)
            reader = easyocr.Reader([language])
            results = reader.readtext(denoised, detail=1)

            # Filter by confidence
            filtered_text = []
            confidences = []

            for bbox, text, conf in results:
                if conf >= (confidence_threshold / 100.0):
                    filtered_text.append(text)
                    confidences.append(conf * 100)

            text = ' '.join(filtered_text)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0

        return (page_num, text, avg_conf)

    except Exception as e:
        logger.error(f"OCR failed for page {page_num}: {e}")
        return (page_num, "", 0.0)


class OCREngine:
    """OCR engine supporting Tesseract and EasyOCR."""

    def __init__(
        self,
        engine: str = 'tesseract',
        language: str = 'eng',
        confidence_threshold: float = 60.0,
        image_dpi: int = 300,
        image_scale: float = 2.0,
        ocr_workers: Optional[int] = None
    ):
        """Initialize OCR engine.

        Args:
            engine: OCR engine to use ('tesseract' or 'easyocr')
            language: Language code (e.g., 'eng', 'fra', 'spa')
            confidence_threshold: Minimum confidence score (0-100 for Tesseract, 0-1 for EasyOCR)
            image_dpi: DPI for PDF to image conversion
            image_scale: Scale factor for OCR accuracy (2x recommended)
            ocr_workers: Number of parallel workers for page processing (None = cpu_count)
        """
        self.engine = engine
        self.language = language
        self.confidence_threshold = confidence_threshold
        self.image_dpi = image_dpi
        self.image_scale = image_scale

        # Configure parallel workers (default: all CPUs for OCR-heavy workload)
        if ocr_workers is None:
            self.ocr_workers = cpu_count()
        else:
            self.ocr_workers = max(1, ocr_workers)

        # For easyocr, workers will create their own readers (can't pickle)
        if engine == 'easyocr':
            try:
                import easyocr
                # Just verify import works
            except ImportError:
                logger.error("EasyOCR not installed. Install with: pip install easyocr")
                raise

    def process_pdf(self, pdf_path: Path, verbose: bool = False) -> Tuple[str, dict]:
        """Perform OCR on PDF with parallel page processing.

        Args:
            pdf_path: Path to PDF file
            verbose: If True, show per-page progress

        Returns:
            Tuple of (text, metadata)
        """
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=self.image_dpi)
        total_pages = len(images)

        if verbose:
            print(f"  → OCR: Processing {total_pages} pages (parallel, {self.ocr_workers} workers)...", flush=True)

        # Serialize images for worker processes
        serialized_images = []
        for page_num, image in enumerate(images, 1):
            # Convert PIL Image to PNG bytes
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()
            serialized_images.append((page_num, img_bytes))

        # Process pages in parallel
        page_results = {}  # {page_num: (text, confidence)}

        with ProcessPoolExecutor(max_workers=self.ocr_workers) as executor:
            # Submit all page jobs
            future_to_page = {}
            for page_num, img_bytes in serialized_images:
                future = executor.submit(
                    _ocr_single_page_worker,
                    img_bytes,
                    page_num,
                    self.engine,
                    self.language,
                    self.confidence_threshold,
                    self.image_scale
                )
                future_to_page[future] = page_num

            # Collect results as they complete
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    result_page_num, page_text, page_conf = future.result()
                    page_results[result_page_num] = (page_text, page_conf)

                    # Show per-page progress
                    if verbose and page_text.strip():
                        print(f"  → OCR: Page {result_page_num}/{total_pages} ({len(page_text)} chars, conf: {page_conf:.0f}%)", flush=True)

                except Exception as e:
                    logger.error(f"OCR failed for page {page_num}: {e}")
                    page_results[page_num] = ("", 0.0)

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
            'extraction_method': f'ocr_{self.engine}',
            'is_image_pdf': True,
            'ocr_confidence': avg_confidence,
            'ocr_language': self.language,
            'page_count': total_pages,
            'ocr_pages_processed': len(text_parts),
        }

        if verbose:
            print(f"  → OCR complete: {len(text)} chars, avg conf: {avg_confidence:.0f}%", flush=True)

        return text, metadata

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for better OCR accuracy."""
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Scale image (2x recommended for accuracy)
        if self.image_scale != 1.0:
            width = int(img_array.shape[1] * self.image_scale)
            height = int(img_array.shape[0] * self.image_scale)
            img_array = cv2.resize(img_array, (width, height), interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply thresholding (Otsu's method)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Denoise
        denoised = cv2.medianBlur(thresh, 3)

        return denoised

    def _ocr_tesseract(self, image: np.ndarray) -> Tuple[str, float]:
        """Perform OCR with Tesseract."""
        try:
            # Get detailed data with confidence
            data = pytesseract.image_to_data(
                image,
                lang=self.language,
                config='--psm 3 --oem 1',  # Auto segmentation, LSTM engine
                output_type=pytesseract.Output.DICT
            )

            # Extract text with confidence filtering
            filtered_text = []
            confidences = []

            for i, conf in enumerate(data['conf']):
                if conf == -1:  # No text detected
                    continue
                if conf >= self.confidence_threshold:
                    text = data['text'][i]
                    if text.strip():
                        filtered_text.append(text)
                        confidences.append(conf)

            text = ' '.join(filtered_text)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0

            return text, avg_conf

        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            raise

    def _ocr_easyocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Perform OCR with EasyOCR."""
        try:
            # EasyOCR returns [(bbox, text, confidence), ...]
            results = self.reader.readtext(image, detail=1)

            # Filter by confidence and extract text
            filtered_text = []
            confidences = []

            for bbox, text, conf in results:
                if conf >= (self.confidence_threshold / 100.0):  # EasyOCR uses 0-1 scale
                    filtered_text.append(text)
                    confidences.append(conf * 100)  # Convert to percentage

            text = ' '.join(filtered_text)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0

            return text, avg_conf

        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            raise

"""OCR engine supporting Tesseract and EasyOCR (RDR-004)."""

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class OCREngine:
    """OCR engine supporting Tesseract and EasyOCR."""

    def __init__(
        self,
        engine: str = 'tesseract',
        language: str = 'eng',
        confidence_threshold: float = 60.0,
        image_dpi: int = 300,
        image_scale: float = 2.0
    ):
        """Initialize OCR engine.

        Args:
            engine: OCR engine to use ('tesseract' or 'easyocr')
            language: Language code (e.g., 'eng', 'fra', 'spa')
            confidence_threshold: Minimum confidence score (0-100 for Tesseract, 0-1 for EasyOCR)
            image_dpi: DPI for PDF to image conversion
            image_scale: Scale factor for OCR accuracy (2x recommended)
        """
        self.engine = engine
        self.language = language
        self.confidence_threshold = confidence_threshold
        self.image_dpi = image_dpi
        self.image_scale = image_scale

        if engine == 'easyocr':
            try:
                import easyocr
                self.reader = easyocr.Reader([language])
            except ImportError:
                logger.error("EasyOCR not installed. Install with: pip install easyocr")
                raise

    def process_pdf(self, pdf_path: Path) -> Tuple[str, dict]:
        """Perform OCR on PDF.

        Returns:
            Tuple of (text, metadata)
        """
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=self.image_dpi)

        text_parts = []
        confidence_scores = []

        for page_num, image in enumerate(images, 1):
            # Preprocess image
            processed_image = self._preprocess_image(image)

            # Perform OCR
            if self.engine == 'tesseract':
                page_text, page_conf = self._ocr_tesseract(processed_image)
            else:  # easyocr
                page_text, page_conf = self._ocr_easyocr(processed_image)

            if page_text.strip():
                text_parts.append(page_text)
                confidence_scores.append(page_conf)
                logger.debug(f"Page {page_num}: {len(page_text)} chars, confidence: {page_conf:.1f}")

        text = '\n'.join(text_parts)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        metadata = {
            'extraction_method': f'ocr_{self.engine}',
            'is_image_pdf': True,
            'ocr_confidence': avg_confidence,
            'ocr_language': self.language,
            'page_count': len(images),
        }

        logger.info(f"OCR completed: {len(text)} chars, avg confidence: {avg_confidence:.1f}%")

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

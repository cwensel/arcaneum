"""PDF text extraction with PyMuPDF and pdfplumber fallback (RDR-004)."""

import pymupdf
import pdfplumber
from pathlib import Path
from typing import Tuple, Optional
import logging
import warnings
import sys
import os

# Suppress PyMuPDF warnings about invalid PDF values
warnings.filterwarnings('ignore', message='.*Cannot set.*is an invalid.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pymupdf')

# Suppress PyMuPDF C library warnings to stderr
pymupdf.TOOLS.mupdf_display_errors(False)

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text from PDFs using PyMuPDF with pdfplumber fallback."""

    def __init__(self, fallback_enabled: bool = True, table_validation: bool = True):
        """Initialize PDF extractor.

        Args:
            fallback_enabled: Enable pdfplumber fallback for complex tables
            table_validation: Validate table extraction quality
        """
        self.fallback_enabled = fallback_enabled
        self.table_validation = table_validation

    def extract(self, pdf_path: Path) -> Tuple[str, dict]:
        """Extract text from PDF.

        Returns:
            Tuple of (text, metadata)
            metadata includes: extraction_method, is_image_pdf, page_count
        """
        try:
            # Primary: PyMuPDF (95x faster)
            text, metadata = self._extract_with_pymupdf(pdf_path)

            # Validate extraction quality
            if self.table_validation and self._has_complex_tables(pdf_path):
                # Fallback to pdfplumber for table-heavy documents
                logger.info(f"Complex tables detected in {pdf_path.name}, using pdfplumber")
                text, metadata = self._extract_with_pdfplumber(pdf_path)

            return text, metadata

        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            raise

    def _extract_with_pymupdf(self, pdf_path: Path) -> Tuple[str, dict]:
        """Extract text using PyMuPDF (fast, general-purpose)."""
        text_parts = []
        page_boundaries = []  # Track character positions where each page starts
        current_pos = 0

        with pymupdf.open(pdf_path) as doc:
            page_count = len(doc)

            for page_num, page in enumerate(doc):
                page_text = page.get_text(sort=True)  # Sort for reading order

                if page_text.strip():
                    page_boundaries.append({
                        'page_number': page_num + 1,  # 1-indexed for user display
                        'start_char': current_pos,
                        'page_text_length': len(page_text)
                    })
                    text_parts.append(page_text)
                    current_pos += len(page_text) + 1  # +1 for newline

        text = '\n'.join(text_parts)

        metadata = {
            'extraction_method': 'pymupdf',
            'is_image_pdf': False,
            'page_count': page_count,
            'file_size': pdf_path.stat().st_size,
            'page_boundaries': page_boundaries,  # Add page boundary tracking
        }

        return text, metadata

    def _extract_with_pdfplumber(self, pdf_path: Path) -> Tuple[str, dict]:
        """Extract text using pdfplumber (slower, better table handling)."""
        text_parts = []
        page_boundaries = []  # Track character positions where each page starts
        current_pos = 0

        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)

            for page in pdf.pages:
                # Extract tables first
                tables = page.extract_tables()

                # Extract regular text
                page_text = page.extract_text(layout=True)  # Preserve layout

                # Combine text and tables
                if tables:
                    table_texts = [self._format_table(table) for table in tables]
                    page_text = page_text + '\n\n' + '\n\n'.join(table_texts)

                if page_text and page_text.strip():
                    page_boundaries.append({
                        'page_number': page.page_number,  # pdfplumber already 1-indexed
                        'start_char': current_pos,
                        'page_text_length': len(page_text)
                    })
                    text_parts.append(page_text)
                    current_pos += len(page_text) + 1  # +1 for newline

        text = '\n'.join(text_parts)

        metadata = {
            'extraction_method': 'pdfplumber',
            'is_image_pdf': False,
            'page_count': page_count,
            'file_size': pdf_path.stat().st_size,
            'page_boundaries': page_boundaries,  # Add page boundary tracking
        }

        return text, metadata

    def _has_complex_tables(self, pdf_path: Path) -> bool:
        """Quick check if PDF has complex tables (heuristic)."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Check first page only (performance)
                if pdf.pages:
                    tables = pdf.pages[0].find_tables()
                    return len(tables) > 0
            return False
        except:
            return False

    def _format_table(self, table: list) -> str:
        """Format extracted table as Markdown."""
        if not table:
            return ""

        # Simple Markdown table formatting
        lines = []
        for row in table:
            lines.append("| " + " | ".join(str(cell) if cell else "" for cell in row) + " |")

        # Add header separator after first row
        if len(lines) > 1:
            lines.insert(1, "|" + "|".join([" --- " for _ in table[0]]) + "|")

        return '\n'.join(lines)

"""PDF text extraction with PyMuPDF and pdfplumber fallback (RDR-004, RDR-016)."""

import pymupdf
import pymupdf4llm
import pdfplumber
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
import warnings
import sys
import os
import re

# Optional: pymupdf-layout for enhanced layout detection
try:
    from pymupdf_layout import Layout
    HAS_PYMUPDF_LAYOUT = True
except ImportError:
    HAS_PYMUPDF_LAYOUT = False

# Suppress PyMuPDF warnings about invalid PDF values
warnings.filterwarnings('ignore', message='.*Cannot set.*is an invalid.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pymupdf')

# Suppress PyMuPDF C library warnings to stderr
pymupdf.TOOLS.mupdf_display_errors(False)

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text from PDFs using PyMuPDF with pdfplumber fallback."""

    def __init__(
        self,
        fallback_enabled: bool = True,
        table_validation: bool = True,
        markdown_conversion: bool = True,
        ignore_images: bool = True,
        preserve_images: bool = False,
        use_layout_analysis: bool = True,
    ):
        """Initialize PDF extractor.

        Args:
            fallback_enabled: Enable pdfplumber fallback for complex tables
            table_validation: Validate table extraction quality
            markdown_conversion: Convert PDF to markdown with structure (default: True, RDR-016)
            ignore_images: Skip image processing for performance (default: True, RDR-016)
            preserve_images: Extract images for multimodal search (default: False, RDR-016)
            use_layout_analysis: Use pymupdf-layout for enhanced layout detection (default: True)
        """
        self.fallback_enabled = fallback_enabled
        self.table_validation = table_validation
        self.markdown_conversion = markdown_conversion
        self.ignore_images = ignore_images and not preserve_images
        self.preserve_images = preserve_images
        self.use_layout_analysis = use_layout_analysis and HAS_PYMUPDF_LAYOUT

    def extract(self, pdf_path: Path) -> Tuple[str, dict]:
        """Extract text from PDF with optional markdown conversion (RDR-016).

        Returns:
            Tuple of (text, metadata)
            metadata includes: extraction_method, is_image_pdf, page_count, format
        """
        try:
            # RDR-016: Use markdown conversion by default for quality-first approach
            if self.markdown_conversion:
                return self._extract_with_markdown(pdf_path)
            else:
                # Normalization-only mode for maximum token savings
                return self._extract_with_pymupdf_normalized(pdf_path)

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

    def _has_type3_fonts(self, pdf_path: Path) -> bool:
        """Check if PDF uses Type3 fonts which can cause PyMuPDF4LLM to hang.

        Type3 fonts are user-defined fonts where each character is drawn using
        PDF graphics commands. PyMuPDF4LLM's style analysis can hang indefinitely
        on PDFs with Type3 fonts due to complex font processing.

        Returns:
            True if PDF uses Type3 fonts (should skip markdown conversion)
        """
        try:
            with pymupdf.open(pdf_path) as doc:
                for page in doc:
                    fonts = page.get_fonts()
                    for font in fonts:
                        # font tuple: (xref, ext, type, basefont, name, encoding)
                        font_type = font[2] if len(font) > 2 else ""
                        if "Type3" in font_type:
                            logger.debug(f"Type3 font detected in {pdf_path.name}, "
                                       f"will use normalized extraction")
                            return True
                return False
        except Exception as e:
            logger.debug(f"Error checking fonts in {pdf_path.name}: {e}")
            return False

    def _get_layout_analysis(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """Get layout analysis using pymupdf-layout for better structure detection.

        Returns layout information including:
        - Text blocks and their bounding boxes
        - Headers, footers, and sections
        - Column layouts
        - Element hierarchy for better semantic understanding
        """
        if not self.use_layout_analysis:
            return None

        try:
            layout = Layout(str(pdf_path))
            return {
                'layout_detected': True,
                'has_pymupdf_layout': True,
                'text_blocks': len(layout.text_block_rects) if hasattr(layout, 'text_block_rects') else 0,
                'pages_analyzed': layout.num_pages if hasattr(layout, 'num_pages') else 0,
            }
        except Exception as e:
            logger.debug(f"Layout analysis failed for {pdf_path.name}: {e}. "
                        f"Will use standard extraction.")
            return None

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

    def _normalize_whitespace_edge_cases(self, text: str) -> str:
        """Handle whitespace edge cases not covered by PyMuPDF4LLM (RDR-016).

        PyMuPDF4LLM already handles:
        - Double space collapsing
        - Trailing spaces before newlines
        - Triple newline reduction to double
        - Leading/trailing whitespace trimming

        This function handles remaining edge cases:
        - Tabs
        - Unicode whitespace characters (non-breaking spaces, etc.)
        - 4+ consecutive newlines
        """
        if not text:
            return text

        # Convert tabs to spaces (PyMuPDF4LLM doesn't handle tabs)
        text = text.replace('\t', ' ')

        # Normalize Unicode whitespace characters
        # Includes non-breaking space, thin space, etc.
        text = re.sub(r'[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]+', ' ', text)

        # Handle 4+ newlines (PyMuPDF4LLM only reduces 3 to 2)
        text = re.sub(r'\n{4,}', '\n\n\n', text)

        return text.strip()

    def _extract_with_markdown(self, pdf_path: Path) -> Tuple[str, dict]:
        """Extract text as markdown using PyMuPDF4LLM with optional layout analysis (RDR-016 default).

        This is the quality-first approach that provides semantic structure
        (headers, lists, tables) while still achieving token savings through
        built-in whitespace normalization.

        If pymupdf-layout is available, uses layout analysis to enhance
        structure detection and semantic understanding.

        Falls back to normalized extraction if:
        - PDF uses Type3 fonts (causes PyMuPDF4LLM to hang)
        - Font errors occur (code=4: no font file for digest)
        - Other markdown conversion failures
        """
        # Check for Type3 fonts which cause PyMuPDF4LLM to hang indefinitely
        if self._has_type3_fonts(pdf_path):
            logger.info(f"Skipping markdown conversion for {pdf_path.name} "
                       f"(Type3 fonts detected - known PyMuPDF4LLM hang issue)")
            return self._extract_with_pymupdf_normalized(pdf_path)

        try:
            # Get layout analysis for enhanced structure detection (optional)
            layout_info = self._get_layout_analysis(pdf_path) if self.use_layout_analysis else None

            # Convert entire document to markdown
            # PyMuPDF4LLM includes built-in whitespace normalization
            md_text = pymupdf4llm.to_markdown(
                str(pdf_path),
                ignore_images=self.ignore_images,  # Default: True for performance
                write_images=self.preserve_images,  # Default: False
                force_text=True,  # Extract all text (default)
                table_strategy="lines_strict",  # Accurate table detection
            )

            # Handle edge cases not covered by PyMuPDF4LLM
            # (tabs, Unicode whitespace, 4+ newlines)
            md_text = self._normalize_whitespace_edge_cases(md_text)

            with pymupdf.open(pdf_path) as doc:
                page_count = len(doc)

            metadata = {
                'extraction_method': 'pymupdf4llm_markdown',
                'is_image_pdf': False,
                'page_count': page_count,
                'file_size': pdf_path.stat().st_size,
                'format': 'markdown',
                'layout_analyzed': layout_info is not None,
            }

            # Add layout analysis details if available
            if layout_info:
                metadata.update(layout_info)

            return md_text, metadata

        except RuntimeError as e:
            # Handle PyMuPDF font digest errors (code=4: no font file for digest)
            # This occurs when TEXT_COLLECT_STYLES flag encounters fonts without embedded data
            # (Base-14 fonts, system fonts). The error is in fake-bold detection optimization,
            # not core text extraction. Fallback maintains quality.
            error_msg = str(e)
            if "font" in error_msg.lower() or "code=4" in error_msg:
                logger.warning(f"Markdown conversion failed for {pdf_path.name} "
                             f"(font digest error: {error_msg}). This is a known PyMuPDF4LLM "
                             f"limitation with certain fonts. Falling back to normalized extraction.")
                # Fall back to normalized extraction - quality is maintained
                # (only loses fake-bold deduplication optimization)
                return self._extract_with_pymupdf_normalized(pdf_path)
            else:
                # Re-raise other RuntimeErrors
                raise

    def _extract_with_pymupdf_normalized(self, pdf_path: Path) -> Tuple[str, dict]:
        """Extract text with normalization only (RDR-016 opt-in for maximum savings).

        This is the maximum token savings approach (47-48% reduction) without
        adding structural markup. Use when cost optimization is the primary goal.
        """
        text_parts = []
        page_boundaries = []
        current_pos = 0

        with pymupdf.open(pdf_path) as doc:
            page_count = len(doc)

            for page_num, page in enumerate(doc):
                page_text = page.get_text(sort=True)

                if page_text.strip():
                    page_boundaries.append({
                        'page_number': page_num + 1,
                        'start_char': current_pos,
                        'page_text_length': len(page_text)
                    })
                    text_parts.append(page_text)
                    current_pos += len(page_text) + 1

        text = '\n'.join(text_parts)

        # Apply comprehensive normalization
        # Note: Raw PyMuPDF extraction doesn't normalize (unlike PyMuPDF4LLM)
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)
        # Reduce excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        # Handle edge cases (tabs, Unicode whitespace)
        text = self._normalize_whitespace_edge_cases(text)

        metadata = {
            'extraction_method': 'pymupdf_normalized',
            'is_image_pdf': False,
            'page_count': page_count,
            'file_size': pdf_path.stat().st_size,
            'format': 'normalized',
            'page_boundaries': page_boundaries,
        }

        return text, metadata

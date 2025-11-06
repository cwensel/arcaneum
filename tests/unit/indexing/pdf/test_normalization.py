"""Unit tests for PDF text normalization (RDR-016)."""

import pytest
from pathlib import Path
from arcaneum.indexing.pdf.extractor import PDFExtractor


class TestWhitespaceNormalization:
    """Test whitespace normalization edge cases."""

    def setup_method(self):
        """Setup extractor for tests."""
        self.extractor = PDFExtractor()

    def test_tab_conversion(self):
        """Test tabs are converted to spaces."""
        input_text = "Hello\tworld\t\ttest"
        expected = "Hello world  test"
        result = self.extractor._normalize_whitespace_edge_cases(input_text)
        assert result == expected

    def test_unicode_whitespace_normalization(self):
        """Test Unicode whitespace characters are normalized."""
        # Non-breaking space (U+00A0)
        input_text = "Hello\u00A0world"
        result = self.extractor._normalize_whitespace_edge_cases(input_text)
        assert result == "Hello world"

        # Multiple Unicode whitespace types
        input_text = "Test\u00A0\u2000\u2001word"
        result = self.extractor._normalize_whitespace_edge_cases(input_text)
        assert result == "Test word"

    def test_excessive_newlines(self):
        """Test 4+ newlines are reduced to 3."""
        input_text = "Line1\n\n\n\nLine2"
        expected = "Line1\n\n\nLine2"
        result = self.extractor._normalize_whitespace_edge_cases(input_text)
        assert result == expected

        # Test 5+ newlines
        input_text = "Line1\n\n\n\n\n\nLine2"
        result = self.extractor._normalize_whitespace_edge_cases(input_text)
        assert result == expected

    def test_leading_trailing_whitespace(self):
        """Test leading and trailing whitespace is stripped."""
        input_text = "   \n\nHello world\n\n   "
        expected = "Hello world"
        result = self.extractor._normalize_whitespace_edge_cases(input_text)
        assert result == expected

    def test_empty_text(self):
        """Test empty text handling."""
        assert self.extractor._normalize_whitespace_edge_cases("") == ""
        assert self.extractor._normalize_whitespace_edge_cases(None) is None

    def test_combined_edge_cases(self):
        """Test combination of multiple edge cases."""
        input_text = "\tHello\u00A0world\n\n\n\n\tTest\u2000line\n\n\n\n\n  "
        # Should convert tabs, normalize Unicode spaces, reduce newlines, strip
        result = self.extractor._normalize_whitespace_edge_cases(input_text)
        assert "\t" not in result
        assert "\u00A0" not in result
        assert "\u2000" not in result
        assert result.count("\n\n\n\n") == 0  # No 4+ newlines
        assert result == result.strip()


class TestMarkdownExtraction:
    """Test markdown extraction with PyMuPDF4LLM."""

    def test_markdown_mode_enabled(self):
        """Test extractor uses markdown conversion by default."""
        extractor = PDFExtractor(markdown_conversion=True)
        assert extractor.markdown_conversion is True

    def test_normalization_only_mode(self):
        """Test extractor can use normalization-only mode."""
        extractor = PDFExtractor(markdown_conversion=False)
        assert extractor.markdown_conversion is False

    def test_preserve_images_disables_ignore(self):
        """Test preserve_images overrides ignore_images."""
        extractor = PDFExtractor(ignore_images=True, preserve_images=True)
        assert extractor.ignore_images is False
        assert extractor.preserve_images is True

    def test_ignore_images_default(self):
        """Test ignore_images is True by default."""
        extractor = PDFExtractor()
        assert extractor.ignore_images is True
        assert extractor.preserve_images is False


class TestExtractionMetadata:
    """Test extraction metadata includes correct format information."""

    def test_metadata_structure(self):
        """Test metadata includes required fields."""
        # Note: This is a structural test, actual PDF processing
        # would require fixture PDFs which we'll add in integration tests
        extractor = PDFExtractor(markdown_conversion=True)
        assert extractor.markdown_conversion is True

        extractor_normalized = PDFExtractor(markdown_conversion=False)
        assert extractor_normalized.markdown_conversion is False

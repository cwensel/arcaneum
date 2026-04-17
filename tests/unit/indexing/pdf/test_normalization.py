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

    def test_markdown_conversion_default_is_true(self):
        """Markdown conversion is the default extraction mode."""
        extractor = PDFExtractor()
        assert extractor.markdown_conversion is True

    def test_normalization_only_mode(self):
        """markdown_conversion=False switches to plain-text normalization."""
        extractor = PDFExtractor(markdown_conversion=False)
        assert extractor.markdown_conversion is False

    def test_preserve_images_disables_ignore(self):
        """preserve_images must force ignore_images to False, overriding it."""
        extractor = PDFExtractor(ignore_images=True, preserve_images=True)
        assert extractor.ignore_images is False
        assert extractor.preserve_images is True

    def test_ignore_images_default(self):
        """ignore_images defaults True and preserve_images defaults False."""
        extractor = PDFExtractor()
        assert extractor.ignore_images is True
        assert extractor.preserve_images is False

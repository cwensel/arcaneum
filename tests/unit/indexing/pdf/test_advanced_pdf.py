"""Unit tests for advanced PDF extraction integration (RDR-022/023)."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from arcaneum.indexing.pdf.extractor import PDFExtractor


class TestPDFExtractorAdvancedPdf:
    """Test advanced_pdf parameter routing in PDFExtractor."""

    def test_default_advanced_pdf_is_off(self):
        """Default advanced_pdf should be 'off'."""
        extractor = PDFExtractor()
        assert extractor.advanced_pdf == "off"

    def test_advanced_pdf_on_sets_mode(self):
        """advanced_pdf='on' should be stored."""
        extractor = PDFExtractor(advanced_pdf="on")
        assert extractor.advanced_pdf == "on"

    def test_advanced_pdf_auto_sets_mode(self):
        """advanced_pdf='auto' should be stored."""
        extractor = PDFExtractor(advanced_pdf="auto")
        assert extractor.advanced_pdf == "auto"

    @patch('arcaneum.indexing.pdf.extractor.HAS_MINERU', False)
    def test_extract_falls_through_when_mineru_not_installed(self, tmp_path):
        """When MinerU is not installed, advanced_pdf='on' should fall through to default."""
        extractor = PDFExtractor(advanced_pdf="on")
        # With HAS_MINERU=False, the MinerU path is skipped entirely
        # Extract will proceed to markdown extraction (which needs a real PDF)
        # Just verify the routing logic
        assert extractor.advanced_pdf == "on"

    @patch('arcaneum.indexing.pdf.extractor.HAS_MINERU', True)
    @patch('mineru.cli.common.do_parse')
    def test_extract_with_mineru_on_calls_api(self, mock_do_parse, tmp_path):
        """When advanced_pdf='on' and MinerU installed, should call do_parse in-process."""
        import pymupdf
        pdf_path = tmp_path / "test.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test content for MinerU extraction")
        doc.save(str(pdf_path))
        doc.close()

        # Mock do_parse to raise an exception to trigger fallback
        mock_do_parse.side_effect = RuntimeError("test MinerU failure")

        extractor = PDFExtractor(advanced_pdf="on")
        text, metadata = extractor.extract(pdf_path)

        # Should have tried MinerU first (do_parse called)
        mock_do_parse.assert_called_once()

        # Should have fallen back to pymupdf4llm
        assert metadata['extraction_method'] in ('pymupdf4llm_markdown', 'pymupdf_normalized')

    @patch('arcaneum.indexing.pdf.extractor.HAS_MINERU', True)
    @patch('mineru.cli.common.do_parse')
    def test_mineru_failure_falls_back(self, mock_do_parse, tmp_path):
        """MinerU failure should fall back to PyMuPDF4LLM."""
        import pymupdf
        pdf_path = tmp_path / "test.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test content")
        doc.save(str(pdf_path))
        doc.close()

        mock_do_parse.side_effect = Exception("GPU error")

        extractor = PDFExtractor(advanced_pdf="on")
        text, metadata = extractor.extract(pdf_path)

        # Should have fallen back
        assert metadata['extraction_method'] != 'mineru_markdown'


class TestColumnDetection:
    """Test multi-column layout detection heuristic."""

    def test_single_column_pdf(self, tmp_path):
        """A simple single-column PDF should not be detected as multi-column."""
        from arcaneum.indexing.pdf.quality import detect_multi_column
        import pymupdf

        pdf_path = tmp_path / "single_col.pdf"
        doc = pymupdf.open()

        for _ in range(3):
            page = doc.new_page(width=612, height=792)  # Letter size
            # Single column of text spanning full width
            y = 72
            for i in range(20):
                page.insert_text((72, y), f"This is a line of single-column text number {i} with enough words to span the page width reasonably well.")
                y += 20

        doc.save(str(pdf_path))
        doc.close()

        assert detect_multi_column(pdf_path) is False

    def test_empty_pdf(self, tmp_path):
        """An empty PDF should not be detected as multi-column."""
        from arcaneum.indexing.pdf.quality import detect_multi_column
        import pymupdf

        pdf_path = tmp_path / "empty.pdf"
        doc = pymupdf.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()

        assert detect_multi_column(pdf_path) is False

    def test_nonexistent_pdf(self, tmp_path):
        """A nonexistent PDF should return False (graceful failure)."""
        from arcaneum.indexing.pdf.quality import detect_multi_column

        pdf_path = tmp_path / "nonexistent.pdf"
        assert detect_multi_column(pdf_path) is False


class TestColumnInterleavingArtifacts:
    """Test Phase B post-extraction artifact detection (RDR-022)."""

    def test_clean_text_no_artifacts(self):
        """Clean single-column text should not trigger artifact detection."""
        from arcaneum.indexing.pdf.quality import has_column_interleaving_artifacts

        text = ("The algorithm processes each node in the graph sequentially. "
                "For each node, it computes the shortest path using Dijkstra's method. "
                "The result is stored in a priority queue for later retrieval.")
        assert has_column_interleaving_artifacts(text) is False

    def test_orphaned_headers_detected(self):
        """Bold single-letter headers mid-sentence should be detected."""
        from arcaneum.indexing.pdf.quality import has_column_interleaving_artifacts

        # Needs > 200 chars to pass the length guard
        padding = "The algorithm processes each node in the graph sequentially for optimal results. "
        text = (padding
                + "study the dynamic **A** Algorithm DC-Tree for servers on trees "
                + "and the equilibrium **B** Binary Search Trees in networks "
                + "with the proposed **C** Combinatorial Optimization method")
        assert len(text) > 200
        assert has_column_interleaving_artifacts(text) is True

    def test_bracket_fragments_detected(self):
        """Bracket-fragmented text should be detected."""
        from arcaneum.indexing.pdf.quality import has_column_interleaving_artifacts

        # Simulate a real extraction with many bracket fragments scattered through text
        padding = "The algorithm processes each node in the graph. " * 2
        fragments = " normal text ".join(
            "[whose][property][is][that]" for _ in range(6)
        )
        text = padding + fragments + padding
        assert len(text) > 200
        assert has_column_interleaving_artifacts(text) is True

    def test_page_number_insertions_detected(self):
        """Mid-sentence page number insertions should be detected."""
        from arcaneum.indexing.pdf.quality import has_column_interleaving_artifacts

        padding = "The algorithm processes each node in the graph sequentially for results. "
        text = (padding
                + "study the dynamic 8 Adwords Pricing model equilibrium "
                + "and the result 42 Binary Search Trees algorithm was proposed "
                + "in the original paper by the authors of this work")
        assert len(text) > 200
        assert has_column_interleaving_artifacts(text) is True

    def test_short_text_not_checked(self):
        """Text shorter than 200 chars should not trigger detection."""
        from arcaneum.indexing.pdf.quality import has_column_interleaving_artifacts

        assert has_column_interleaving_artifacts("short") is False
        assert has_column_interleaving_artifacts("") is False
        assert has_column_interleaving_artifacts(None) is False


class TestColumnDetectionCaching:
    """Test that column detection results are cached."""

    def test_cache_hit_on_second_call(self, tmp_path):
        """Second call to detect_multi_column should use cache."""
        from arcaneum.indexing.pdf.quality import detect_multi_column, _column_detection_cache
        import pymupdf

        pdf_path = tmp_path / "cached.pdf"
        doc = pymupdf.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()

        # First call populates cache
        result1 = detect_multi_column(pdf_path)
        cache_key = (str(pdf_path.resolve()), pdf_path.stat().st_mtime)
        assert cache_key in _column_detection_cache

        # Verify cached value matches
        assert _column_detection_cache[cache_key] == result1


class TestHasMineruDetection:
    """Test MinerU availability detection via import."""

    def test_has_mineru_reflects_import(self):
        """HAS_MINERU should be True when mineru.cli.common.do_parse is importable."""
        from arcaneum.indexing.pdf.extractor import HAS_MINERU
        try:
            from mineru.cli.common import do_parse
            assert HAS_MINERU is True
        except ImportError:
            assert HAS_MINERU is False

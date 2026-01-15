"""Integration tests for PDF full-text indexing (RDR-010).

These tests require:
- Running MeiliSearch server
- Sample PDF files

Run with: pytest tests/indexing/fulltext/test_pdf_fulltext_integration.py -v
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from arcaneum.fulltext.client import FullTextClient
from arcaneum.fulltext.indexes import PDF_DOCS_SETTINGS
from arcaneum.indexing.fulltext.pdf_indexer import PDFFullTextIndexer
from arcaneum.paths import get_meilisearch_api_key


MEILISEARCH_URL = os.environ.get("MEILISEARCH_URL", "http://localhost:7700")
MEILISEARCH_API_KEY = os.environ.get("MEILISEARCH_API_KEY") or get_meilisearch_api_key()


@pytest.fixture
def meili_client():
    """Provide MeiliSearch client connected to test server."""
    try:
        client = FullTextClient(url=MEILISEARCH_URL, api_key=MEILISEARCH_API_KEY)

        # Skip tests if server not available
        if not client.health_check():
            pytest.skip("MeiliSearch server not available")

        # Test that we can list indexes (verifies API key)
        client.list_indexes()

    except Exception as e:
        pytest.skip(f"MeiliSearch server not available: {e}")

    yield client

    # Cleanup: delete test indexes
    try:
        indexes = client.list_indexes()
        for idx in indexes:
            if idx['uid'].startswith("test_pdf_"):
                client.delete_index(idx['uid'])
    except Exception:
        pass


@pytest.fixture
def test_index(meili_client):
    """Create a test index with PDF settings."""
    index_name = "test_pdf_fulltext"

    # Delete if exists
    if meili_client.index_exists(index_name):
        meili_client.delete_index(index_name)

    # Create with PDF settings
    meili_client.create_index(
        name=index_name,
        primary_key='id',
        settings=PDF_DOCS_SETTINGS
    )

    yield index_name

    # Cleanup
    try:
        meili_client.delete_index(index_name)
    except Exception:
        pass


class TestPDFFullTextIndexerIntegration:
    """Integration tests for PDF indexing to MeiliSearch."""

    def test_index_single_pdf_mock_extraction(self, meili_client, test_index):
        """Test indexing a single PDF with mocked extraction."""
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 test content")
            pdf_path = Path(f.name)

        try:
            # Create indexer with mocked extraction
            with patch('arcaneum.indexing.fulltext.pdf_indexer.PDFExtractor') as mock_extractor:
                with patch('arcaneum.indexing.fulltext.pdf_indexer.OCREngine'):
                    # Mock extraction to return test data
                    mock_extractor.return_value.extract.return_value = (
                        "This is test content from page 1.\fThis is test content from page 2.",
                        {
                            "extraction_method": "pymupdf4llm_markdown",
                            "is_image_pdf": False,
                            "page_count": 2,
                            "file_size": 100,
                        }
                    )

                    indexer = PDFFullTextIndexer(
                        meili_client=meili_client,
                        index_name=test_index,
                        ocr_enabled=False,
                        batch_size=100
                    )

                    # Index the PDF
                    result = indexer.index_pdf(pdf_path)

            assert result['page_count'] == 2
            assert 'task_uid' in result

            # Verify documents were indexed
            stats = meili_client.get_index_stats(test_index)
            assert stats['numberOfDocuments'] == 2

            # Search for indexed content
            search_results = meili_client.search(
                test_index,
                "test content",
                limit=10
            )

            assert search_results['estimatedTotalHits'] >= 1

        finally:
            pdf_path.unlink()

    def test_index_directory_mock_extraction(self, meili_client, test_index):
        """Test indexing a directory of PDFs with mocked extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test PDF files
            pdf1 = Path(tmpdir) / "doc1.pdf"
            pdf2 = Path(tmpdir) / "doc2.pdf"
            pdf1.write_bytes(b"%PDF-1.4 content 1")
            pdf2.write_bytes(b"%PDF-1.4 content 2")

            # Create indexer with mocked extraction
            with patch('arcaneum.indexing.fulltext.pdf_indexer.PDFExtractor') as mock_extractor:
                with patch('arcaneum.indexing.fulltext.pdf_indexer.OCREngine'):
                    # Mock extraction for each file
                    mock_extractor.return_value.extract.side_effect = [
                        ("Document one content about machine learning.", {
                            "extraction_method": "pymupdf4llm_markdown",
                            "is_image_pdf": False,
                            "page_count": 1,
                            "file_size": 100,
                        }),
                        ("Document two content about deep learning.", {
                            "extraction_method": "pymupdf4llm_markdown",
                            "is_image_pdf": False,
                            "page_count": 1,
                            "file_size": 100,
                        }),
                    ]

                    indexer = PDFFullTextIndexer(
                        meili_client=meili_client,
                        index_name=test_index,
                        ocr_enabled=False,
                        batch_size=100
                    )

                    # Index directory
                    stats = indexer.index_directory(
                        directory=Path(tmpdir),
                        recursive=True,
                        force_reindex=True,
                        verbose=False
                    )

            assert stats['total_pdfs'] == 2
            assert stats['indexed_pdfs'] == 2
            assert stats['failed_pdfs'] == 0

            # Verify documents in index
            index_stats = meili_client.get_index_stats(test_index)
            assert index_stats['numberOfDocuments'] == 2

    def test_change_detection_skip_indexed(self, meili_client, test_index):
        """Test that already-indexed files are skipped."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 test")
            pdf_path = Path(f.name)

        try:
            with patch('arcaneum.indexing.fulltext.pdf_indexer.PDFExtractor') as mock_extractor:
                with patch('arcaneum.indexing.fulltext.pdf_indexer.OCREngine'):
                    mock_extractor.return_value.extract.return_value = (
                        "Test content",
                        {
                            "extraction_method": "pymupdf",
                            "is_image_pdf": False,
                            "page_count": 1,
                            "file_size": 100,
                        }
                    )

                    indexer = PDFFullTextIndexer(
                        meili_client=meili_client,
                        index_name=test_index,
                        ocr_enabled=False
                    )

                    # First indexing
                    result1 = indexer.index_pdf(pdf_path)
                    assert result1['page_count'] == 1

                    # Second indexing should detect existing
                    is_indexed = indexer._is_already_indexed(pdf_path)
                    assert is_indexed is True

        finally:
            pdf_path.unlink()

    def test_search_exact_phrase(self, meili_client, test_index):
        """Test exact phrase search on indexed PDFs."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 test")
            pdf_path = Path(f.name)

        try:
            with patch('arcaneum.indexing.fulltext.pdf_indexer.PDFExtractor') as mock_extractor:
                with patch('arcaneum.indexing.fulltext.pdf_indexer.OCREngine'):
                    mock_extractor.return_value.extract.return_value = (
                        "The quick brown fox jumps over the lazy dog.",
                        {
                            "extraction_method": "pymupdf",
                            "is_image_pdf": False,
                            "page_count": 1,
                            "file_size": 100,
                        }
                    )

                    indexer = PDFFullTextIndexer(
                        meili_client=meili_client,
                        index_name=test_index,
                        ocr_enabled=False
                    )

                    indexer.index_pdf(pdf_path)

            # Search for exact phrase
            results = meili_client.search(
                test_index,
                '"quick brown fox"',
                limit=10
            )

            assert results['estimatedTotalHits'] >= 1
            assert "quick brown fox" in results['hits'][0]['content'].lower()

        finally:
            pdf_path.unlink()

    def test_search_with_filter(self, meili_client, test_index):
        """Test filtered search by page number."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 test")
            pdf_path = Path(f.name)

        try:
            with patch('arcaneum.indexing.fulltext.pdf_indexer.PDFExtractor') as mock_extractor:
                with patch('arcaneum.indexing.fulltext.pdf_indexer.OCREngine'):
                    mock_extractor.return_value.extract.return_value = (
                        "Page one content\fPage two content\fPage three content",
                        {
                            "extraction_method": "pymupdf",
                            "is_image_pdf": False,
                            "page_count": 3,
                            "file_size": 100,
                        }
                    )

                    indexer = PDFFullTextIndexer(
                        meili_client=meili_client,
                        index_name=test_index,
                        ocr_enabled=False
                    )

                    indexer.index_pdf(pdf_path)

            # Search with page filter
            results = meili_client.search(
                test_index,
                "content",
                filter="page_number = 2",
                limit=10
            )

            assert results['estimatedTotalHits'] >= 1
            assert results['hits'][0]['page_number'] == 2

        finally:
            pdf_path.unlink()

    def test_delete_pdf_documents(self, meili_client, test_index):
        """Test deleting all documents for a PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 test")
            pdf_path = Path(f.name)

        try:
            with patch('arcaneum.indexing.fulltext.pdf_indexer.PDFExtractor') as mock_extractor:
                with patch('arcaneum.indexing.fulltext.pdf_indexer.OCREngine'):
                    mock_extractor.return_value.extract.return_value = (
                        "Page 1\fPage 2",
                        {
                            "extraction_method": "pymupdf",
                            "is_image_pdf": False,
                            "page_count": 2,
                            "file_size": 100,
                        }
                    )

                    indexer = PDFFullTextIndexer(
                        meili_client=meili_client,
                        index_name=test_index,
                        ocr_enabled=False
                    )

                    # Index PDF
                    indexer.index_pdf(pdf_path)

                    # Verify indexed
                    stats = meili_client.get_index_stats(test_index)
                    assert stats['numberOfDocuments'] == 2

                    # Delete documents
                    result = indexer.delete_pdf_documents(pdf_path)
                    assert result is True

                    # Verify deleted
                    stats = meili_client.get_index_stats(test_index)
                    assert stats['numberOfDocuments'] == 0

        finally:
            pdf_path.unlink()


class TestPDFDocsSettingsIntegration:
    """Tests for PDF_DOCS_SETTINGS in MeiliSearch."""

    def test_pdf_settings_applied(self, meili_client, test_index):
        """Test that PDF_DOCS_SETTINGS are correctly applied."""
        settings = meili_client.get_index_settings(test_index)

        # Check searchable attributes
        assert "content" in settings["searchableAttributes"]
        assert "filename" in settings["searchableAttributes"]

        # Check filterable attributes (RDR-010 additions)
        assert "file_path" in settings["filterableAttributes"]
        assert "page_number" in settings["filterableAttributes"]
        assert "file_hash" in settings["filterableAttributes"]
        assert "extraction_method" in settings["filterableAttributes"]
        assert "is_image_pdf" in settings["filterableAttributes"]

        # Check sortable
        assert "page_number" in settings["sortableAttributes"]

        # Check pagination (RDR-010: increased)
        assert settings["pagination"]["maxTotalHits"] == 10000

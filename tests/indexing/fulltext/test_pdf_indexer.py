"""Unit tests for PDFFullTextIndexer (RDR-010).

Tests the PDF full-text indexing functionality for MeiliSearch.
"""

import hashlib
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

from arcaneum.indexing.fulltext.pdf_indexer import PDFFullTextIndexer
from arcaneum.indexing.fulltext.sync import (
    compute_file_hash,
    find_files_to_index,
    get_orphaned_files,
)


class TestComputeFileHash:
    """Tests for file hash computation."""

    def test_compute_file_hash_simple(self):
        """Test hash computation for a simple file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Hello, World!")
            f.flush()
            temp_path = Path(f.name)

        try:
            result = compute_file_hash(temp_path)
            # Expected SHA-256 of "Hello, World!"
            expected = hashlib.sha256(b"Hello, World!").hexdigest()
            assert result == expected
        finally:
            temp_path.unlink()

    def test_compute_file_hash_large_file(self):
        """Test hash computation for a large file (chunked reading)."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Write 1MB of data
            data = b"x" * (1024 * 1024)
            f.write(data)
            f.flush()
            temp_path = Path(f.name)

        try:
            result = compute_file_hash(temp_path)
            expected = hashlib.sha256(data).hexdigest()
            assert result == expected
        finally:
            temp_path.unlink()

    def test_compute_file_hash_empty_file(self):
        """Test hash computation for empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = compute_file_hash(temp_path)
            expected = hashlib.sha256(b"").hexdigest()
            assert result == expected
        finally:
            temp_path.unlink()


class TestFindFilesToIndex:
    """Tests for change detection logic."""

    def test_find_new_files(self):
        """Test detection of new files (not in index)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = Path(tmpdir) / "new1.pdf"
            file2 = Path(tmpdir) / "new2.pdf"
            file1.write_bytes(b"PDF content 1")
            file2.write_bytes(b"PDF content 2")

            pdf_files = [file1, file2]
            indexed_files = {}  # Empty index

            new, modified, unchanged = find_files_to_index(pdf_files, indexed_files)

            assert len(new) == 2
            assert len(modified) == 0
            assert len(unchanged) == 0

    def test_find_modified_files(self):
        """Test detection of modified files (hash changed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "modified.pdf"
            file1.write_bytes(b"New PDF content")

            pdf_files = [file1]
            # Index has old hash
            indexed_files = {
                str(file1.absolute()): "old_hash_value"
            }

            new, modified, unchanged = find_files_to_index(pdf_files, indexed_files)

            assert len(new) == 0
            assert len(modified) == 1
            assert len(unchanged) == 0

    def test_find_unchanged_files(self):
        """Test detection of unchanged files (hash matches)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "unchanged.pdf"
            file1.write_bytes(b"PDF content")

            pdf_files = [file1]
            # Index has matching hash
            current_hash = compute_file_hash(file1)
            indexed_files = {
                str(file1.absolute()): current_hash
            }

            new, modified, unchanged = find_files_to_index(pdf_files, indexed_files)

            assert len(new) == 0
            assert len(modified) == 0
            assert len(unchanged) == 1

    def test_force_reindex(self):
        """Test that force_reindex treats all files as new."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "existing.pdf"
            file1.write_bytes(b"PDF content")

            pdf_files = [file1]
            current_hash = compute_file_hash(file1)
            indexed_files = {
                str(file1.absolute()): current_hash
            }

            new, modified, unchanged = find_files_to_index(
                pdf_files, indexed_files, force_reindex=True
            )

            assert len(new) == 1
            assert len(modified) == 0
            assert len(unchanged) == 0


class TestGetOrphanedFiles:
    """Tests for orphan detection."""

    def test_find_orphaned_files(self):
        """Test detection of files in index but not on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Only one file exists
            existing = Path(tmpdir) / "exists.pdf"
            existing.write_bytes(b"content")

            pdf_files = [existing]
            indexed_files = {
                str(existing.absolute()): "hash1",
                "/path/to/deleted.pdf": "hash2",
            }

            orphaned = get_orphaned_files(indexed_files, pdf_files)

            assert len(orphaned) == 1
            assert "/path/to/deleted.pdf" in orphaned

    def test_no_orphans(self):
        """Test when all indexed files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.pdf"
            file1.write_bytes(b"content")

            pdf_files = [file1]
            indexed_files = {
                str(file1.absolute()): "hash1",
            }

            orphaned = get_orphaned_files(indexed_files, pdf_files)

            assert len(orphaned) == 0


class TestPDFFullTextIndexerDocumentBuilding:
    """Tests for document building functionality."""

    @pytest.fixture
    def mock_meili_client(self):
        """Create mock MeiliSearch client."""
        client = Mock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        client.add_documents_sync.return_value = {"taskUid": 123, "status": "succeeded"}
        client.search.return_value = {"hits": [], "estimatedTotalHits": 0}
        return client

    @pytest.fixture
    def indexer(self, mock_meili_client):
        """Create indexer with mocked client."""
        with patch('arcaneum.indexing.fulltext.pdf_indexer.PDFExtractor') as mock_extractor:
            with patch('arcaneum.indexing.fulltext.pdf_indexer.OCREngine') as mock_ocr:
                indexer = PDFFullTextIndexer(
                    meili_client=mock_meili_client,
                    index_name="test-index",
                    ocr_enabled=False,  # Disable OCR for unit tests
                    batch_size=100
                )
                # Access the mocked extractor
                indexer.pdf_extractor = mock_extractor.return_value
                return indexer

    def test_build_documents_simple(self, indexer):
        """Test document building with simple text."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"PDF content")
            pdf_path = Path(f.name)

        try:
            metadata = {
                "extraction_method": "pymupdf",
                "is_image_pdf": False,
                "page_count": 2,
                "file_size": 100,
            }

            documents = indexer._build_meilisearch_documents(
                pdf_path,
                "Page 1 content\f Page 2 content",
                metadata
            )

            assert len(documents) == 2

            # Check first document
            doc1 = documents[0]
            assert "id" in doc1
            assert doc1["page_number"] == 1
            assert doc1["filename"] == pdf_path.name
            assert doc1["extraction_method"] == "pymupdf"
            assert doc1["is_image_pdf"] is False
            assert "file_hash" in doc1
            assert "content" in doc1

        finally:
            pdf_path.unlink()

    def test_build_documents_with_page_boundaries(self, indexer):
        """Test document building with page boundaries from metadata."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"PDF")
            pdf_path = Path(f.name)

        try:
            # Simulate page boundaries from PDFExtractor
            full_text = "First page content\nSecond page content"
            metadata = {
                "extraction_method": "pymupdf",
                "is_image_pdf": False,
                "page_count": 2,
                "file_size": 100,
                "page_boundaries": [
                    {"page_number": 1, "start_char": 0, "page_text_length": 18},
                    {"page_number": 2, "start_char": 19, "page_text_length": 19},
                ]
            }

            documents = indexer._build_meilisearch_documents(
                pdf_path, full_text, metadata
            )

            assert len(documents) == 2
            assert "First page" in documents[0]["content"]
            assert "Second page" in documents[1]["content"]

        finally:
            pdf_path.unlink()

    def test_build_documents_unique_ids(self, indexer):
        """Test that document IDs are unique across files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two files with same name in different dirs
            dir1 = Path(tmpdir) / "dir1"
            dir2 = Path(tmpdir) / "dir2"
            dir1.mkdir()
            dir2.mkdir()

            file1 = dir1 / "same.pdf"
            file2 = dir2 / "same.pdf"
            file1.write_bytes(b"content1")
            file2.write_bytes(b"content2")

            metadata = {
                "extraction_method": "pymupdf",
                "is_image_pdf": False,
                "page_count": 1,
                "file_size": 100,
            }

            docs1 = indexer._build_meilisearch_documents(file1, "content", metadata)
            docs2 = indexer._build_meilisearch_documents(file2, "content", metadata)

            # IDs should be different despite same filename
            assert docs1[0]["id"] != docs2[0]["id"]

    def test_build_documents_skips_empty_pages(self, indexer):
        """Test that empty pages are skipped."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"PDF")
            pdf_path = Path(f.name)

        try:
            metadata = {
                "extraction_method": "pymupdf",
                "is_image_pdf": False,
                "page_count": 3,
                "file_size": 100,
            }

            # Second page is empty
            documents = indexer._build_meilisearch_documents(
                pdf_path,
                "Page 1\f\fPage 3",
                metadata
            )

            # Only 2 documents (pages 1 and 3)
            assert len(documents) == 2
            page_numbers = [d["page_number"] for d in documents]
            assert 1 in page_numbers
            assert 3 in page_numbers

        finally:
            pdf_path.unlink()

    def test_build_documents_with_ocr_metadata(self, indexer):
        """Test document building includes OCR metadata when present."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"PDF")
            pdf_path = Path(f.name)

        try:
            metadata = {
                "extraction_method": "ocr_tesseract",
                "is_image_pdf": True,
                "page_count": 1,
                "file_size": 100,
                "ocr_confidence": 95.5,
                "ocr_language": "eng",
            }

            documents = indexer._build_meilisearch_documents(
                pdf_path, "OCR text", metadata
            )

            assert len(documents) == 1
            doc = documents[0]
            assert doc["is_image_pdf"] is True
            assert doc["extraction_method"] == "ocr_tesseract"
            assert doc["ocr_confidence"] == 95.5
            assert doc["ocr_language"] == "eng"

        finally:
            pdf_path.unlink()


class TestPDFFullTextIndexerPageSplitting:
    """Tests for page splitting logic."""

    @pytest.fixture
    def indexer(self):
        """Create indexer with mocked dependencies."""
        mock_client = Mock()
        with patch('arcaneum.indexing.fulltext.pdf_indexer.PDFExtractor'):
            with patch('arcaneum.indexing.fulltext.pdf_indexer.OCREngine'):
                return PDFFullTextIndexer(
                    meili_client=mock_client,
                    index_name="test",
                    ocr_enabled=False
                )

    def test_split_by_form_feed(self, indexer):
        """Test splitting text by form feed character."""
        text = "Page 1\fPage 2\fPage 3"
        metadata = {"page_count": 3}

        pages = indexer._split_into_pages(text, 3, metadata)

        assert len(pages) == 3
        assert pages[0] == "Page 1"
        assert pages[1] == "Page 2"
        assert pages[2] == "Page 3"

    def test_split_by_page_boundaries(self, indexer):
        """Test splitting using page boundaries from metadata."""
        text = "First page contentSecond page content"
        metadata = {
            "page_count": 2,
            "page_boundaries": [
                {"page_number": 1, "start_char": 0, "page_text_length": 18},
                {"page_number": 2, "start_char": 18, "page_text_length": 19},
            ]
        }

        pages = indexer._split_into_pages(text, 2, metadata)

        assert len(pages) == 2
        assert pages[0] == "First page content"
        assert pages[1] == "Second page content"

    def test_split_single_page(self, indexer):
        """Test splitting single page document."""
        text = "All content on one page"
        metadata = {"page_count": 1}

        pages = indexer._split_into_pages(text, 1, metadata)

        assert len(pages) == 1
        assert pages[0] == text

    def test_split_pads_missing_pages(self, indexer):
        """Test that missing pages are padded with empty strings."""
        text = "Page 1\fPage 2"
        metadata = {"page_count": 5}

        pages = indexer._split_into_pages(text, 5, metadata)

        assert len(pages) == 5
        assert pages[0] == "Page 1"
        assert pages[1] == "Page 2"
        assert pages[2] == ""
        assert pages[3] == ""
        assert pages[4] == ""


class TestPDFFullTextIndexerChangeDetection:
    """Tests for change detection in indexer."""

    @pytest.fixture
    def indexer(self):
        """Create indexer with mocked client."""
        mock_client = Mock()
        mock_client.search.return_value = {"hits": [], "estimatedTotalHits": 0}

        with patch('arcaneum.indexing.fulltext.pdf_indexer.PDFExtractor'):
            with patch('arcaneum.indexing.fulltext.pdf_indexer.OCREngine'):
                return PDFFullTextIndexer(
                    meili_client=mock_client,
                    index_name="test",
                    ocr_enabled=False
                )

    def test_is_already_indexed_false(self, indexer):
        """Test file not in index returns False."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"content")
            pdf_path = Path(f.name)

        try:
            indexer.meili_client.search.return_value = {
                "hits": [],
                "estimatedTotalHits": 0
            }

            result = indexer._is_already_indexed(pdf_path)
            assert result is False

        finally:
            pdf_path.unlink()

    def test_is_already_indexed_true(self, indexer):
        """Test file in index returns True."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"content")
            pdf_path = Path(f.name)

        try:
            file_hash = compute_file_hash(pdf_path)

            indexer.meili_client.search.return_value = {
                "hits": [{"id": "doc1", "file_hash": file_hash}],
                "estimatedTotalHits": 1
            }

            result = indexer._is_already_indexed(pdf_path)
            assert result is True

        finally:
            pdf_path.unlink()

    def test_is_already_indexed_handles_error(self, indexer):
        """Test that search errors return False (assume not indexed)."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"content")
            pdf_path = Path(f.name)

        try:
            indexer.meili_client.search.side_effect = Exception("Search failed")

            result = indexer._is_already_indexed(pdf_path)
            assert result is False

        finally:
            pdf_path.unlink()

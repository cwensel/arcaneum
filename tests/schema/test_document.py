"""Unit tests for shared document schema (RDR-009)."""

import pytest

from arcaneum.schema.document import (
    DualIndexDocument,
    to_qdrant_point,
    to_meilisearch_doc,
)


class TestDualIndexDocument:
    """Tests for DualIndexDocument dataclass."""

    def test_default_values(self):
        """Test that DualIndexDocument has sensible defaults."""
        doc = DualIndexDocument()
        assert doc.id != ""  # UUID generated
        assert doc.content == ""
        assert doc.file_path == ""
        assert doc.chunk_index == 0
        assert doc.chunk_count == 1
        assert doc.vectors == {}
        assert doc.function_names == []

    def test_create_with_values(self):
        """Test creating document with specific values."""
        doc = DualIndexDocument(
            id="test-id-123",
            content="Hello world",
            file_path="/path/to/file.py",
            filename="file.py",
            language="python",
            chunk_index=2,
            chunk_count=5,
            file_extension=".py",
            line_number=42,
            function_names=["foo", "bar"],
        )

        assert doc.id == "test-id-123"
        assert doc.content == "Hello world"
        assert doc.file_path == "/path/to/file.py"
        assert doc.filename == "file.py"
        assert doc.language == "python"
        assert doc.chunk_index == 2
        assert doc.chunk_count == 5
        assert doc.file_extension == ".py"
        assert doc.line_number == 42
        assert doc.function_names == ["foo", "bar"]

    def test_optional_fields_default_to_none(self):
        """Test that optional fields default to None."""
        doc = DualIndexDocument()

        assert doc.line_number is None
        assert doc.page_number is None
        assert doc.project is None
        assert doc.branch is None
        assert doc.title is None

    def test_vectors_independent_between_instances(self):
        """Test that vectors dict is not shared between instances."""
        doc1 = DualIndexDocument()
        doc2 = DualIndexDocument()

        doc1.vectors["model1"] = [1.0, 2.0, 3.0]

        assert "model1" not in doc2.vectors


class TestToQdrantPoint:
    """Tests for to_qdrant_point conversion."""

    def test_basic_conversion(self):
        """Test converting document to Qdrant point."""
        doc = DualIndexDocument(
            id="test-123",
            content="Sample content",
            file_path="/path/to/file.py",
            filename="file.py",
            file_extension=".py",
            chunk_index=0,
            chunk_count=3,
            language="python",
            vectors={"stella": [0.1, 0.2, 0.3]},
        )

        point = to_qdrant_point(doc)

        assert point.id == "test-123"
        assert point.vector == {"stella": [0.1, 0.2, 0.3]}
        assert point.payload["text"] == "Sample content"
        assert point.payload["file_path"] == "/path/to/file.py"
        assert point.payload["filename"] == "file.py"
        assert point.payload["programming_language"] == "python"

    def test_raises_on_missing_vectors(self):
        """Test that conversion raises ValueError if no vectors."""
        doc = DualIndexDocument(
            content="No vectors",
            file_path="/path/to/file.py",
        )

        with pytest.raises(ValueError, match="no vectors"):
            to_qdrant_point(doc)

    def test_optional_fields_included(self):
        """Test that optional fields are included when present."""
        doc = DualIndexDocument(
            content="Code chunk",
            file_path="/path/to/file.py",
            filename="file.py",
            file_extension=".py",
            chunk_index=0,
            chunk_count=1,
            language="python",
            line_number=42,
            project="myproject",
            branch="main",
            git_project_identifier="myproject#main",
            function_names=["foo", "bar"],
            vectors={"model": [0.1, 0.2]},
        )

        point = to_qdrant_point(doc)

        assert point.payload["line_number"] == 42
        assert point.payload["git_project_name"] == "myproject"
        assert point.payload["git_branch"] == "main"
        assert point.payload["git_project_identifier"] == "myproject#main"
        assert point.payload["function_names"] == ["foo", "bar"]

    def test_optional_fields_omitted_when_none(self):
        """Test that optional fields are not included when None."""
        doc = DualIndexDocument(
            content="Minimal doc",
            file_path="/path/to/file.py",
            filename="file.py",
            file_extension=".py",
            chunk_index=0,
            chunk_count=1,
            vectors={"model": [0.1, 0.2]},
        )

        point = to_qdrant_point(doc)

        assert "line_number" not in point.payload
        assert "page_number" not in point.payload
        assert "git_project_name" not in point.payload
        assert "title" not in point.payload

    def test_uses_provided_point_id(self):
        """Test that explicit point_id is used."""
        doc = DualIndexDocument(
            id="doc-uuid-123",
            content="Content",
            file_path="/path/file.py",
            filename="file.py",
            file_extension=".py",
            chunk_index=0,
            chunk_count=1,
            vectors={"model": [0.1]},
        )

        point = to_qdrant_point(doc, point_id=42)

        assert point.id == 42


class TestToMeilisearchDoc:
    """Tests for to_meilisearch_doc conversion."""

    def test_basic_conversion(self):
        """Test converting document to MeiliSearch format."""
        doc = DualIndexDocument(
            id="test-456",
            content="Sample markdown content",
            file_path="/path/to/file.md",
            filename="file.md",
            file_extension=".md",
            chunk_index=1,
            chunk_count=5,
            language="markdown",
        )

        meili_doc = to_meilisearch_doc(doc)

        assert meili_doc["id"] == "test-456"
        assert meili_doc["content"] == "Sample markdown content"
        assert meili_doc["file_path"] == "/path/to/file.md"
        assert meili_doc["filename"] == "file.md"
        assert meili_doc["language"] == "markdown"
        assert meili_doc["chunk_index"] == 1

    def test_vectors_not_included(self):
        """Test that vectors are not included in MeiliSearch doc."""
        doc = DualIndexDocument(
            id="test-789",
            content="Content",
            file_path="/path/file.py",
            filename="file.py",
            file_extension=".py",
            chunk_index=0,
            vectors={"stella": [0.1, 0.2, 0.3]},
        )

        meili_doc = to_meilisearch_doc(doc)

        assert "vectors" not in meili_doc

    def test_optional_fields_included(self):
        """Test that optional fields are included when present."""
        doc = DualIndexDocument(
            id="test-opt",
            content="PDF content",
            file_path="/path/to/doc.pdf",
            filename="doc.pdf",
            file_extension=".pdf",
            chunk_index=0,
            page_number=5,
            title="Important Document",
            author="John Doe",
            document_type="pdf",
        )

        meili_doc = to_meilisearch_doc(doc)

        assert meili_doc["page_number"] == 5
        assert meili_doc["title"] == "Important Document"
        assert meili_doc["author"] == "John Doe"
        assert meili_doc["document_type"] == "pdf"

    def test_optional_fields_omitted_when_none(self):
        """Test that optional fields are not included when None."""
        doc = DualIndexDocument(
            id="test-min",
            content="Minimal",
            file_path="/path/file.txt",
            filename="file.txt",
            file_extension=".txt",
            chunk_index=0,
        )

        meili_doc = to_meilisearch_doc(doc)

        assert "line_number" not in meili_doc
        assert "page_number" not in meili_doc
        assert "title" not in meili_doc
        assert "project" not in meili_doc

    def test_code_specific_fields(self):
        """Test code-specific fields in conversion."""
        doc = DualIndexDocument(
            id="code-doc",
            content="def foo():\n    pass",
            file_path="/src/main.py",
            filename="main.py",
            file_extension=".py",
            chunk_index=0,
            language="python",
            function_names=["foo", "bar"],
            class_names=["MyClass"],
        )

        meili_doc = to_meilisearch_doc(doc)

        assert meili_doc["function_names"] == ["foo", "bar"]
        assert meili_doc["class_names"] == ["MyClass"]

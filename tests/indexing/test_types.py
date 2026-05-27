"""Tests for source code indexing type definitions."""

import pytest
from arcaneum.indexing.types import GitMetadata, CodeChunkMetadata, CodeChunk


class TestGitMetadata:
    """Tests for GitMetadata dataclass."""

    def test_identifier_property(self):
        """Test composite identifier generation."""
        metadata = GitMetadata(
            project_root="/path/to/repo",
            commit_hash="a" * 40,
            branch="feature-x",
            project_name="my-project"
        )

        assert metadata.identifier == "my-project#feature-x"

    def test_version_identifier_property(self):
        """Test version_identifier generation with project#branch@commit."""
        metadata = GitMetadata(
            project_root="/path/to/repo",
            commit_hash="abc123def456abc123def456abc123def456abc1",
            branch="feature-x",
            project_name="my-project"
        )

        assert metadata.version_identifier == "my-project#feature-x@abc123d"

class TestCodeChunkMetadata:
    """Tests for CodeChunkMetadata dataclass."""

    def test_default_values(self):
        """Test default values for optional fields."""
        metadata = CodeChunkMetadata(
            git_project_identifier="project#main",
            file_path="/repo/src/main.py",
            filename="main.py",
            file_extension=".py",
            programming_language="python",
            file_size=1024,
            line_count=50,
            chunk_index=0,
            chunk_count=1,
            text_extraction_method="ast_python",
            git_project_root="/repo",
            git_project_name="project",
            git_branch="main",
            git_commit_hash="a" * 40
        )

        assert metadata.ast_chunked is False
        assert metadata.has_functions is False
        assert metadata.has_classes is False
        assert metadata.has_imports is False
        assert metadata.embedding_model == "jina-embeddings-v2-base-code"
        assert metadata.store_type == "code"
        assert metadata.source_hash == "a" * 40
        assert metadata.chunking_version == "code-ast:v1"

    def test_to_payload(self):
        """Test conversion to Qdrant payload dictionary."""
        metadata = CodeChunkMetadata(
            git_project_identifier="project#main",
            file_path="/repo/src/main.py",
            filename="main.py",
            file_extension=".py",
            programming_language="python",
            file_size=1024,
            line_count=50,
            chunk_index=0,
            chunk_count=2,
            text_extraction_method="ast_python",
            git_project_root="/repo",
            git_project_name="project",
            git_branch="main",
            git_commit_hash="a" * 40,
            source_hash="source123",
            chunking_version="code-ast:v1:400:20",
            ast_chunked=True,
            has_functions=True
        )

        payload = metadata.to_payload()

        assert isinstance(payload, dict)
        assert payload["git_project_identifier"] == "project#main"
        assert payload["file_path"] == "/repo/src/main.py"
        assert payload["ast_chunked"] is True
        assert payload["has_functions"] is True
        assert payload["chunk_count"] == 2
        assert payload["source_hash"] == "source123"
        assert payload["chunking_version"] == "code-ast:v1:400:20"

    def test_from_payload(self):
        """Test reconstruction from Qdrant payload."""
        original = CodeChunkMetadata(
            git_project_identifier="project#feature",
            file_path="/repo/src/test.java",
            filename="test.java",
            file_extension=".java",
            programming_language="java",
            file_size=2048,
            line_count=100,
            chunk_index=1,
            chunk_count=3,
            text_extraction_method="ast_java",
            git_project_root="/repo",
            git_project_name="project",
            git_branch="feature",
            git_commit_hash="b" * 40,
            git_remote_url="https://github.com/user/repo.git",
            source_hash="source456",
            chunking_version="code-ast:v1:400:20",
            has_classes=True
        )

        payload = original.to_payload()
        reconstructed = CodeChunkMetadata.from_payload(payload)

        assert reconstructed.git_project_identifier == original.git_project_identifier
        assert reconstructed.file_path == original.file_path
        assert reconstructed.has_classes == original.has_classes
        assert reconstructed.git_remote_url == original.git_remote_url
        assert reconstructed.source_hash == original.source_hash
        assert reconstructed.chunking_version == original.chunking_version


class TestCodeChunk:
    """Tests for CodeChunk dataclass."""

    def test_convenience_properties(self):
        """Test convenience property accessors."""
        metadata = CodeChunkMetadata(
            git_project_identifier="myproject#develop",
            file_path="/repo/src/utils.js",
            filename="utils.js",
            file_extension=".js",
            programming_language="javascript",
            file_size=512,
            line_count=30,
            chunk_index=0,
            chunk_count=1,
            text_extraction_method="ast_javascript",
            git_project_root="/repo",
            git_project_name="myproject",
            git_branch="develop",
            git_commit_hash="c" * 40
        )

        chunk = CodeChunk(content="function test() {}", metadata=metadata)

        assert chunk.file_path == "/repo/src/utils.js"
        assert chunk.git_project_identifier == "myproject#develop"

    def test_to_point_without_embedding_fails(self):
        """Test that to_point() raises error when embedding is None."""
        metadata = CodeChunkMetadata(
            git_project_identifier="project#main",
            file_path="/repo/src/main.py",
            filename="main.py",
            file_extension=".py",
            programming_language="python",
            file_size=1024,
            line_count=50,
            chunk_index=0,
            chunk_count=1,
            text_extraction_method="ast_python",
            git_project_root="/repo",
            git_project_name="project",
            git_branch="main",
            git_commit_hash="a" * 40
        )

        chunk = CodeChunk(content="code", metadata=metadata)

        with pytest.raises(ValueError, match="embedding is None"):
            chunk.to_point()

    def test_to_point_with_embedding(self):
        """Test successful conversion to PointStruct."""
        metadata = CodeChunkMetadata(
            git_project_identifier="project#main",
            file_path="/repo/src/main.py",
            filename="main.py",
            file_extension=".py",
            programming_language="python",
            file_size=1024,
            line_count=50,
            chunk_index=0,
            chunk_count=1,
            text_extraction_method="ast_python",
            git_project_root="/repo",
            git_project_name="project",
            git_branch="main",
            git_commit_hash="a" * 40
        )

        chunk = CodeChunk(
            content="code",
            metadata=metadata,
            embedding=[0.1] * 768  # Mock 768D embedding
        )

        point = chunk.to_point()

        assert point.id == chunk.id
        assert point.vector == chunk.embedding
        assert point.payload["git_project_identifier"] == "project#main"

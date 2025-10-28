"""
Type definitions for source code indexing.

This module defines the core data structures used for git-aware source code indexing
with AST chunking and multi-branch support (RDR-005).
"""

from dataclasses import dataclass, field
from typing import Optional, List
from uuid import uuid4


@dataclass
class GitMetadata:
    """Git repository metadata extracted from a project.

    Attributes:
        project_root: Absolute path to the git repository root
        commit_hash: Full 40-character SHA of current HEAD
        branch: Branch name (or tag/detached HEAD identifier)
        project_name: Derived from remote URL or directory basename
        remote_url: Sanitized remote URL (origin > upstream > first remote)
    """
    project_root: str
    commit_hash: str
    branch: str
    project_name: str
    remote_url: Optional[str] = None

    @property
    def identifier(self) -> str:
        """Composite identifier combining project name and branch.

        Format: "project-name#branch"
        Examples: "arcaneum#main", "myproject#feature-x"

        This enables multi-branch support where multiple branches of the same
        repository can coexist in the collection.
        """
        return f"{self.project_name}#{self.branch}"


@dataclass
class CodeChunkMetadata:
    """Metadata for a single code chunk in Qdrant.

    This schema follows RDR-005 and enables:
    - Multi-branch support via git_project_identifier
    - Branch-specific deletion using filter-based queries
    - Metadata-based sync (Qdrant as source of truth)
    - AST-aware chunking tracking
    """

    # PRIMARY identifier for multi-branch support
    git_project_identifier: str  # "project-name#branch" (e.g., "arcaneum#main")

    # Common file fields
    file_path: str  # Absolute path to source file
    filename: str  # Basename of file
    file_extension: str  # e.g., ".py", ".java", ".js"
    programming_language: str  # e.g., "python", "java", "javascript"
    file_size: int  # Size in bytes
    line_count: int  # Number of lines in original file
    chunk_index: int  # Zero-based index of this chunk within the file
    chunk_count: int  # Total number of chunks for this file
    text_extraction_method: str  # e.g., "ast_python", "ast_java", "line_based"

    # Git metadata fields (all required for git-only mode)
    git_project_root: str  # Absolute path to git repository root
    git_project_name: str  # Component of identifier, kept for filtering
    git_branch: str  # Component of identifier, kept for filtering
    git_commit_hash: str  # Full 40-char SHA
    git_remote_url: Optional[str] = None  # Sanitized remote URL

    # Code analysis fields
    ast_chunked: bool = False  # True if AST parsing succeeded
    has_functions: bool = False  # True if chunk contains function definitions
    has_classes: bool = False  # True if chunk contains class definitions
    has_imports: bool = False  # True if chunk contains import statements

    # Embedding metadata
    embedding_model: str = "jina-embeddings-v2-base-code"
    store_type: str = "source-code"

    def to_payload(self) -> dict:
        """Convert to Qdrant payload dictionary.

        Returns dictionary suitable for Qdrant point payload.
        """
        return {
            "git_project_identifier": self.git_project_identifier,
            "file_path": self.file_path,
            "filename": self.filename,
            "file_extension": self.file_extension,
            "programming_language": self.programming_language,
            "file_size": self.file_size,
            "line_count": self.line_count,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "text_extraction_method": self.text_extraction_method,
            "git_project_root": self.git_project_root,
            "git_project_name": self.git_project_name,
            "git_branch": self.git_branch,
            "git_commit_hash": self.git_commit_hash,
            "git_remote_url": self.git_remote_url,
            "ast_chunked": self.ast_chunked,
            "has_functions": self.has_functions,
            "has_classes": self.has_classes,
            "has_imports": self.has_imports,
            "embedding_model": self.embedding_model,
            "store_type": self.store_type,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "CodeChunkMetadata":
        """Create from Qdrant payload dictionary."""
        return cls(
            git_project_identifier=payload["git_project_identifier"],
            file_path=payload["file_path"],
            filename=payload["filename"],
            file_extension=payload["file_extension"],
            programming_language=payload["programming_language"],
            file_size=payload["file_size"],
            line_count=payload["line_count"],
            chunk_index=payload["chunk_index"],
            chunk_count=payload["chunk_count"],
            text_extraction_method=payload["text_extraction_method"],
            git_project_root=payload["git_project_root"],
            git_project_name=payload["git_project_name"],
            git_branch=payload["git_branch"],
            git_commit_hash=payload["git_commit_hash"],
            git_remote_url=payload.get("git_remote_url"),
            ast_chunked=payload.get("ast_chunked", False),
            has_functions=payload.get("has_functions", False),
            has_classes=payload.get("has_classes", False),
            has_imports=payload.get("has_imports", False),
            embedding_model=payload.get("embedding_model", "jina-embeddings-v2-base-code"),
            store_type=payload.get("store_type", "source-code"),
        )


@dataclass
class CodeChunk:
    """A single chunk of source code with embedding and metadata.

    This represents a unit of code to be indexed in Qdrant, including:
    - The actual code content
    - Vector embedding for semantic search
    - Rich metadata for filtering and branch management
    """
    content: str  # The actual code text
    metadata: CodeChunkMetadata  # Rich metadata
    embedding: Optional[List[float]] = None  # Vector embedding (computed lazily)
    id: str = field(default_factory=lambda: str(uuid4()))  # Unique ID for Qdrant

    @property
    def file_path(self) -> str:
        """Convenience accessor for file path."""
        return self.metadata.file_path

    @property
    def git_project_identifier(self) -> str:
        """Convenience accessor for composite identifier."""
        return self.metadata.git_project_identifier

    def to_point(self, vector_name: Optional[str] = None):
        """Convert to Qdrant PointStruct.

        Returns a PointStruct ready for uploading to Qdrant.
        Requires embedding to be computed first.

        Args:
            vector_name: Name of vector if using named vectors (e.g., "stella")
                        If None, uses unnamed vector
        """
        from qdrant_client.models import PointStruct

        if self.embedding is None:
            raise ValueError("Cannot convert to PointStruct: embedding is None")

        # Handle named vs unnamed vectors
        if vector_name:
            vector = {vector_name: self.embedding}
        else:
            vector = self.embedding

        # Build payload with metadata AND content (for search result display)
        payload = self.metadata.to_payload()
        payload["text"] = self.content  # Add content field for search snippets

        return PointStruct(
            id=self.id,
            vector=vector,
            payload=payload
        )


@dataclass
class Chunk:
    """Simple chunk container for AST chunking output.

    Used by ASTCodeChunker before full metadata is attached.
    """
    content: str  # The code text
    method: str  # Extraction method (e.g., "ast_python", "line_based")

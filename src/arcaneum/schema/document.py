"""Shared document schema for dual indexing (RDR-009).

This module defines the unified document schema used for indexing to both
Qdrant (vector search) and MeiliSearch (full-text search) with shared metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from uuid import uuid4


@dataclass
class DualIndexDocument:
    """Document schema for dual indexing to Qdrant and MeiliSearch.

    This schema enables cooperative search workflows where semantic search
    results can be verified with exact phrase matching using shared metadata.

    Shared metadata fields (both systems):
        - id: Unique document identifier
        - content: The text content
        - file_path: Absolute path to source file
        - filename: Basename of file
        - language: Programming language or document type
        - chunk_index: Zero-based index of this chunk within the file
        - chunk_count: Total number of chunks for this file
        - file_extension: File extension (e.g., ".py", ".pdf")

    Optional fields:
        - line_number: Starting line number (for code)
        - page_number: Page number (for PDFs)
        - project: Project or repository name
        - branch: Git branch (for code)
        - git_project_identifier: Composite "project#branch" identifier
        - git_commit_hash: Git commit SHA
        - function_names: List of function names in chunk (for code)
        - class_names: List of class names in chunk (for code)
        - title: Document title (for PDFs/markdown)
        - author: Document author (for PDFs)
        - headings: Section headings (for markdown)
        - tags: Document tags (for markdown)

    Vector-only fields:
        - vectors: Dict of model_name -> embedding vector
    """

    # Primary identifier
    id: str = field(default_factory=lambda: str(uuid4()))

    # Content
    content: str = ""

    # Shared metadata (both systems)
    file_path: str = ""
    filename: str = ""
    language: str = ""
    chunk_index: int = 0
    chunk_count: int = 1
    file_extension: str = ""

    # Optional fields
    line_number: Optional[int] = None
    page_number: Optional[int] = None
    project: Optional[str] = None
    branch: Optional[str] = None
    git_project_identifier: Optional[str] = None
    git_commit_hash: Optional[str] = None
    git_remote_url: Optional[str] = None
    git_version_identifier: Optional[str] = None  # "project#branch@commit" for multi-version indexing

    # Code-specific
    function_names: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)

    # PDF-specific
    title: Optional[str] = None
    author: Optional[str] = None
    document_type: Optional[str] = None

    # Markdown-specific
    headings: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    section: Optional[str] = None

    # File metadata
    file_hash: Optional[str] = None
    file_size: Optional[int] = None
    quick_hash: Optional[str] = None  # Metadata-based hash (mtime+size) for fast change detection

    # Vectors (Qdrant only)
    vectors: Dict[str, List[float]] = field(default_factory=dict)


def to_qdrant_point(doc: DualIndexDocument, point_id: Optional[int] = None):
    """Convert DualIndexDocument to Qdrant PointStruct.

    Args:
        doc: Document to convert
        point_id: Optional numeric ID for point (uses doc.id if not provided)

    Returns:
        Qdrant PointStruct ready for upsert

    Raises:
        ValueError: If document has no vectors
    """
    from qdrant_client.models import PointStruct

    if not doc.vectors:
        raise ValueError("Cannot convert to PointStruct: document has no vectors")

    # Build payload with all metadata
    payload = {
        "text": doc.content,  # Content field for search snippets
        "file_path": doc.file_path,
        "filename": doc.filename,
        "file_extension": doc.file_extension,
        "chunk_index": doc.chunk_index,
        "chunk_count": doc.chunk_count,
    }

    # Add language with Qdrant field name
    if doc.language:
        payload["programming_language"] = doc.language

    # Add optional fields if present
    if doc.line_number is not None:
        payload["line_number"] = doc.line_number

    if doc.page_number is not None:
        payload["page_number"] = doc.page_number

    if doc.project:
        payload["git_project_name"] = doc.project

    if doc.branch:
        payload["git_branch"] = doc.branch

    if doc.git_project_identifier:
        payload["git_project_identifier"] = doc.git_project_identifier

    if doc.git_commit_hash:
        payload["git_commit_hash"] = doc.git_commit_hash

    if doc.git_remote_url:
        payload["git_remote_url"] = doc.git_remote_url

    if doc.git_version_identifier:
        payload["git_version_identifier"] = doc.git_version_identifier

    if doc.function_names:
        payload["function_names"] = doc.function_names

    if doc.class_names:
        payload["class_names"] = doc.class_names

    if doc.title:
        payload["title"] = doc.title

    if doc.author:
        payload["author"] = doc.author

    if doc.document_type:
        payload["document_type"] = doc.document_type

    if doc.headings:
        payload["headings"] = doc.headings

    if doc.tags:
        payload["tags"] = doc.tags

    if doc.section:
        payload["section"] = doc.section

    if doc.file_hash:
        payload["file_hash"] = doc.file_hash

    if doc.file_size is not None:
        payload["file_size"] = doc.file_size

    if doc.quick_hash:
        payload["quick_hash"] = doc.quick_hash

    # Use document id or provided point_id
    pid = point_id if point_id is not None else doc.id

    return PointStruct(
        id=pid,
        vector=doc.vectors,  # Named vectors dict
        payload=payload
    )


def to_meilisearch_doc(doc: DualIndexDocument) -> Dict[str, Any]:
    """Convert DualIndexDocument to MeiliSearch document format.

    Args:
        doc: Document to convert

    Returns:
        Dictionary suitable for MeiliSearch add_documents
    """
    meili_doc = {
        "id": doc.id,
        "content": doc.content,
        "file_path": doc.file_path,
        "filename": doc.filename,
        "file_extension": doc.file_extension,
        "chunk_index": doc.chunk_index,
    }

    # Add language with MeiliSearch field name
    if doc.language:
        meili_doc["language"] = doc.language

    # Add optional fields if present
    if doc.line_number is not None:
        meili_doc["line_number"] = doc.line_number

    if doc.page_number is not None:
        meili_doc["page_number"] = doc.page_number

    if doc.project:
        meili_doc["project"] = doc.project

    if doc.branch:
        meili_doc["branch"] = doc.branch

    if doc.git_project_identifier:
        meili_doc["git_project_identifier"] = doc.git_project_identifier

    if doc.git_version_identifier:
        meili_doc["git_version_identifier"] = doc.git_version_identifier

    if doc.function_names:
        meili_doc["function_names"] = doc.function_names

    if doc.class_names:
        meili_doc["class_names"] = doc.class_names

    if doc.title:
        meili_doc["title"] = doc.title

    if doc.author:
        meili_doc["author"] = doc.author

    if doc.document_type:
        meili_doc["document_type"] = doc.document_type

    if doc.headings:
        meili_doc["headings"] = doc.headings

    if doc.tags:
        meili_doc["tags"] = doc.tags

    if doc.section:
        meili_doc["section"] = doc.section

    return meili_doc

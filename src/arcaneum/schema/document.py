"""Shared document schema for dual indexing (RDR-009).

This module defines the unified document schema used for indexing to both
Qdrant (vector search) and MeiliSearch (full-text search) with shared metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple
from uuid import uuid4

from arcaneum import __version__ as ARCANEUM_VERSION

PERSISTED_SCHEMA_VERSION = 1
PERSISTED_SCHEMA_VERSION_FIELD = "schema_version"
PERSISTED_APP_VERSION_FIELD = "app_version"

# Migration contract:
# - Adding optional payload fields is backward-compatible within this version.
# - Renaming, removing, or changing the type/meaning of persisted payload fields
#   requires bumping PERSISTED_SCHEMA_VERSION and adding a verifier/backfill path.
# - Existing collections without schema_version are treated as legacy v0 and
#   should be repaired by reindexing or by a targeted metadata backfill when the
#   payload shape is still compatible.


def current_app_version() -> str:
    """Return the Arcaneum version stored with persisted payloads."""
    return ARCANEUM_VERSION


def persisted_metadata_fields(
    schema_version: int = PERSISTED_SCHEMA_VERSION,
    app_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Return schema fields required on persisted search records."""
    return {
        PERSISTED_SCHEMA_VERSION_FIELD: schema_version,
        PERSISTED_APP_VERSION_FIELD: app_version or current_app_version(),
    }


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
    schema_version: int = PERSISTED_SCHEMA_VERSION
    app_version: str = field(default_factory=current_app_version)
    file_path: str = ""
    filename: str = ""
    language: str = ""
    chunk_index: int = 0
    chunk_count: int = 1
    file_extension: str = ""

    # Optional fields
    line_number: Optional[int] = None
    page_number: Optional[int] = None
    section_type: Optional[str] = None
    section_title: Optional[str] = None
    project: Optional[str] = None
    branch: Optional[str] = None
    git_project_identifier: Optional[str] = None
    git_commit_hash: Optional[str] = None
    git_remote_url: Optional[str] = None
    git_version_identifier: Optional[str] = (
        None  # "project#branch@commit" for multi-version indexing
    )

    # Code-specific
    function_names: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)

    # PDF-specific
    title: Optional[str] = None
    author: Optional[str] = None
    document_type: Optional[str] = None
    page_count: Optional[int] = None
    extraction_floor: Optional[bool] = None
    ocr_confidence: Optional[float] = None
    ocr_language: Optional[str] = None
    ocr_pages_processed: Optional[int] = None
    ocr_pages_failed: Optional[int] = None
    ocr_low_confidence_word_count: Optional[int] = None
    ocr_merge_strategy: Optional[str] = None
    ocr_triggered_by: Optional[str] = None

    # Markdown-specific
    headings: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    section: Optional[str] = None

    # File metadata
    file_hash: Optional[str] = None
    source_hash: Optional[str] = None
    chunking_version: Optional[str] = None
    file_size: Optional[int] = None
    quick_hash: Optional[str] = None  # Metadata-based hash (mtime+size) for fast change detection
    quality_manifest: Optional[Dict[str, Any]] = None

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
        **persisted_metadata_fields(doc.schema_version, doc.app_version),
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

    if doc.section_type:
        payload["section_type"] = doc.section_type

    if doc.section_title:
        payload["section_title"] = doc.section_title

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

    if doc.page_count is not None:
        payload["page_count"] = doc.page_count

    if doc.extraction_floor:
        payload["extraction_floor"] = doc.extraction_floor

    if doc.ocr_confidence is not None:
        payload["ocr_confidence"] = doc.ocr_confidence

    if doc.ocr_language:
        payload["ocr_language"] = doc.ocr_language

    if doc.ocr_pages_processed is not None:
        payload["ocr_pages_processed"] = doc.ocr_pages_processed

    if doc.ocr_pages_failed is not None:
        payload["ocr_pages_failed"] = doc.ocr_pages_failed

    if doc.ocr_low_confidence_word_count is not None:
        payload["ocr_low_confidence_word_count"] = doc.ocr_low_confidence_word_count

    if doc.ocr_merge_strategy:
        payload["ocr_merge_strategy"] = doc.ocr_merge_strategy

    if doc.ocr_triggered_by:
        payload["ocr_triggered_by"] = doc.ocr_triggered_by

    if doc.headings:
        payload["headings"] = doc.headings

    if doc.tags:
        payload["tags"] = doc.tags

    if doc.section:
        payload["section"] = doc.section

    if doc.file_hash:
        payload["file_hash"] = doc.file_hash

    if doc.source_hash:
        payload["source_hash"] = doc.source_hash

    if doc.chunking_version:
        payload["chunking_version"] = doc.chunking_version

    if doc.file_size is not None:
        payload["file_size"] = doc.file_size

    if doc.quick_hash:
        payload["quick_hash"] = doc.quick_hash

    if doc.quality_manifest:
        payload["quality_manifest"] = doc.quality_manifest

    # Use document id or provided point_id
    pid = point_id if point_id is not None else doc.id

    return PointStruct(
        id=pid,
        vector=doc.vectors,  # Named vectors dict
        payload=payload,
    )


# The fields of a MeiliSearch document after ``id``, ``schema_version`` and
# ``app_version``, in document order: each MeiliSearch name (the
# DualIndexDocument attribute), the Qdrant payload key to_qdrant_point stores
# it under, and when a document keeps it. to_meilisearch_doc and
# qdrant_payload_to_meilisearch_doc both read it, so a Qdrant text hit carries
# exactly a MeiliSearch hit's fields (RDR-027 Q5).
_ALWAYS = "always"
_SET = "set"  # kept unless None
_TRUTHY = "truthy"  # kept when truthy

MEILISEARCH_FIELDS: Tuple[Tuple[str, str, str], ...] = (
    ("content", "text", _ALWAYS),
    ("file_path", "file_path", _ALWAYS),
    ("filename", "filename", _ALWAYS),
    ("file_extension", "file_extension", _ALWAYS),
    ("chunk_index", "chunk_index", _ALWAYS),
    ("language", "programming_language", _TRUTHY),
    ("line_number", "line_number", _SET),
    ("page_number", "page_number", _SET),
    ("section_type", "section_type", _TRUTHY),
    ("section_title", "section_title", _TRUTHY),
    ("project", "git_project_name", _TRUTHY),
    ("branch", "git_branch", _TRUTHY),
    ("git_project_identifier", "git_project_identifier", _TRUTHY),
    ("git_version_identifier", "git_version_identifier", _TRUTHY),
    ("function_names", "function_names", _TRUTHY),
    ("class_names", "class_names", _TRUTHY),
    ("title", "title", _TRUTHY),
    ("author", "author", _TRUTHY),
    ("document_type", "document_type", _TRUTHY),
    ("ocr_confidence", "ocr_confidence", _SET),
    ("ocr_language", "ocr_language", _TRUTHY),
    ("ocr_pages_processed", "ocr_pages_processed", _SET),
    ("ocr_pages_failed", "ocr_pages_failed", _SET),
    ("ocr_low_confidence_word_count", "ocr_low_confidence_word_count", _SET),
    ("ocr_merge_strategy", "ocr_merge_strategy", _TRUTHY),
    ("ocr_triggered_by", "ocr_triggered_by", _TRUTHY),
    ("headings", "headings", _TRUTHY),
    ("tags", "tags", _TRUTHY),
    ("section", "section", _TRUTHY),
    ("quality_manifest", "quality_manifest", _TRUTHY),
)


def _kept(rule: str, value: Any) -> bool:
    if rule == _ALWAYS:
        return True
    if rule == _SET:
        return value is not None
    return bool(value)


def to_meilisearch_doc(doc: DualIndexDocument) -> Dict[str, Any]:
    """Convert DualIndexDocument to MeiliSearch document format.

    Args:
        doc: Document to convert

    Returns:
        Dictionary suitable for MeiliSearch add_documents
    """
    meili_doc = {
        "id": doc.id,
        **persisted_metadata_fields(doc.schema_version, doc.app_version),
    }
    for name, _payload_key, rule in MEILISEARCH_FIELDS:
        value = getattr(doc, name)
        if _kept(rule, value):
            meili_doc[name] = value
    return meili_doc


def qdrant_payload_to_meilisearch_doc(point_id: Any, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Project a Qdrant chunk point onto the MeiliSearch document of the same chunk.

    The payload's MeiliSearch fields are kept under their MeiliSearch names
    (``text`` as ``content``, ``programming_language`` as ``language``, the
    ``git_*`` names as ``project`` and ``branch``), and every Qdrant-only
    field is dropped, so a Qdrant text hit's ``--verbose`` metadata equals a
    MeiliSearch hit's (RDR-027 Q5). A field the payload lacks stays absent.
    """
    doc: Dict[str, Any] = {"id": str(point_id)}
    for key in (PERSISTED_SCHEMA_VERSION_FIELD, PERSISTED_APP_VERSION_FIELD):
        if key in payload:
            doc[key] = payload[key]
    for name, payload_key, rule in MEILISEARCH_FIELDS:
        if payload_key in payload and _kept(rule, payload[payload_key]):
            doc[name] = payload[payload_key]
    return doc

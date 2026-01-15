"""Index configuration templates for MeiliSearch (RDR-008)."""

from typing import Dict, Any


# Index settings for source code
SOURCE_CODE_SETTINGS: Dict[str, Any] = {
    "searchableAttributes": [
        "content",
        "filename",
        "function_names",
        "class_names",
    ],
    "filterableAttributes": [
        "language",
        "project",
        "branch",
        "file_path",
        "file_extension",
        "git_project_identifier",
    ],
    "sortableAttributes": [],
    "typoTolerance": {
        "enabled": True,
        "minWordSizeForTypos": {
            "oneTypo": 7,   # Higher threshold for code
            "twoTypos": 12
        }
    },
    "stopWords": [],  # Preserve all code keywords
    "pagination": {
        "maxTotalHits": 1000
    }
}


# Index settings for PDF documents (RDR-010: enhanced for full-text indexing)
PDF_DOCS_SETTINGS: Dict[str, Any] = {
    "searchableAttributes": [
        "content",
        "title",
        "author",
        "filename",
    ],
    "filterableAttributes": [
        "filename",
        "file_path",
        "page_number",
        "document_type",
        "file_hash",           # RDR-010: Change detection
        "extraction_method",   # RDR-010: Metadata tracking
        "is_image_pdf",        # RDR-010: OCR flag
    ],
    "sortableAttributes": ["page_number"],
    "typoTolerance": {
        "enabled": True,
        "minWordSizeForTypos": {
            "oneTypo": 5,
            "twoTypos": 9
        }
    },
    "stopWords": [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"
    ],
    "pagination": {
        "maxTotalHits": 10000  # RDR-010: Increase for large PDF collections
    }
}


# Index settings for markdown documents
MARKDOWN_DOCS_SETTINGS: Dict[str, Any] = {
    "searchableAttributes": [
        "content",
        "title",
        "filename",
        "headings",
    ],
    "filterableAttributes": [
        "filename",
        "file_path",
        "section",
        "tags",
    ],
    "sortableAttributes": [],
    "typoTolerance": {
        "enabled": True,
        "minWordSizeForTypos": {
            "oneTypo": 5,
            "twoTypos": 9
        }
    },
    "stopWords": [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"
    ],
    "pagination": {
        "maxTotalHits": 1000
    }
}


def get_index_settings(index_type: str) -> Dict[str, Any]:
    """
    Get index settings by type.

    Args:
        index_type: 'source-code', 'pdf-docs', 'markdown-docs', 'code', 'pdf', or 'markdown'

    Returns:
        Index settings dictionary

    Raises:
        ValueError: If index_type is unknown
    """
    # Map type aliases to canonical names
    type_aliases = {
        "code": "source-code",
        "pdf": "pdf-docs",
        "markdown": "markdown-docs",
    }

    canonical_type = type_aliases.get(index_type, index_type)

    settings_map = {
        "source-code": SOURCE_CODE_SETTINGS,
        "pdf-docs": PDF_DOCS_SETTINGS,
        "markdown-docs": MARKDOWN_DOCS_SETTINGS,
    }

    if canonical_type not in settings_map:
        raise ValueError(
            f"Unknown index type: {index_type}. "
            f"Available: {list(settings_map.keys())} (aliases: {list(type_aliases.keys())})"
        )

    return settings_map[canonical_type].copy()


def get_available_index_types() -> list[str]:
    """Get list of available index types."""
    return ["source-code", "pdf-docs", "markdown-docs", "code", "pdf", "markdown"]

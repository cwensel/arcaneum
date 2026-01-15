"""Full-text search module (RDR-008).

This module provides MeiliSearch integration for Arcaneum, enabling:
- Exact phrase matching with quote syntax ("def authenticate")
- Typo-tolerant keyword search
- Filtered search by metadata (language, project, file_path)
- Line-number precision for code search results
"""

from arcaneum.fulltext.client import FullTextClient
from arcaneum.fulltext.indexes import (
    SOURCE_CODE_SETTINGS,
    PDF_DOCS_SETTINGS,
    MARKDOWN_DOCS_SETTINGS,
    get_index_settings,
    get_available_index_types,
)

__all__ = [
    "FullTextClient",
    "SOURCE_CODE_SETTINGS",
    "PDF_DOCS_SETTINGS",
    "MARKDOWN_DOCS_SETTINGS",
    "get_index_settings",
    "get_available_index_types",
]

"""Search module for semantic search across Qdrant collections (RDR-007)."""

from .embedder import SearchEmbedder
from .filters import parse_filter, build_filter_description
from .searcher import SearchResult, search_collection, explain_search, format_location
from .formatter import (
    format_text_results,
    format_json_results,
    format_metadata,
    extract_snippet,
    format_summary
)

__all__ = [
    # Embedder
    "SearchEmbedder",
    # Filters
    "parse_filter",
    "build_filter_description",
    # Searcher
    "SearchResult",
    "search_collection",
    "explain_search",
    "format_location",
    # Formatter
    "format_text_results",
    "format_json_results",
    "format_metadata",
    "extract_snippet",
    "format_summary",
]

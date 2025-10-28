"""Result formatters for search output (RDR-007)."""

import json
from typing import List, Dict, Any
from .searcher import SearchResult


def format_text_results(
    query: str,
    results: List[SearchResult],
    verbose: bool = False
) -> str:
    """Format search results for human-readable terminal display.

    Output format:
        Searching for: "query"
        Found N results

        [1] Score: 95% | Language: python | Project: myproject
            /path/to/file.py

            def authenticate_user(username, password):
                \"\"\"Verify user credentials...

    Args:
        query: Original search query
        results: List of SearchResult objects
        verbose: If True, show more metadata and longer snippets

    Returns:
        Formatted string for terminal output
    """
    lines = []

    # Header
    lines.append(f'Searching for: "{query}"')
    lines.append(f"Found {len(results)} result{'s' if len(results) != 1 else ''}")
    lines.append("")  # Blank line

    # Format each result
    for i, result in enumerate(results, 1):
        # Score and metadata header
        score_pct = int(result.score * 100)
        metadata_str = format_metadata(result.metadata, verbose=verbose)

        if metadata_str:
            lines.append(f"[{i}] Score: {score_pct}% | {metadata_str}")
        else:
            lines.append(f"[{i}] Score: {score_pct}%")

        # Location
        lines.append(f"    {result.location}")
        lines.append("")  # Blank line

        # Content snippet
        snippet_length = 400 if verbose else 200
        snippet = extract_snippet(result.content, max_length=snippet_length)

        # Show first few lines of snippet (max 5 lines in normal mode, 10 in verbose)
        max_lines = 10 if verbose else 5
        snippet_lines = snippet.split('\n')[:max_lines]

        for line in snippet_lines:
            lines.append(f"    {line}")

        lines.append("")  # Blank line between results

    return "\n".join(lines)


def format_json_results(
    query: str,
    collection: str,
    results: List[SearchResult],
    verbose: bool = False
) -> str:
    """Format search results as JSON.

    Output structure:
        {
            "query": "search query",
            "collection": "MyCollection",
            "total_results": 5,
            "results": [
                {
                    "score": 0.95,
                    "location": "/path/to/file.py",
                    "content": "...",
                    "metadata": {...}
                }
            ]
        }

    Args:
        query: Original search query
        collection: Collection name
        results: List of SearchResult objects
        verbose: If True, include full metadata and longer content

    Returns:
        JSON string
    """
    # Truncate content unless verbose mode
    content_length = None if verbose else 500

    output = {
        "query": query,
        "collection": collection,
        "total_results": len(results),
        "results": [
            {
                "score": r.score,
                "location": r.location,
                "content": r.content[:content_length] if content_length else r.content,
                "metadata": r.metadata if verbose else _filter_metadata(r.metadata),
                "point_id": r.point_id
            }
            for r in results
        ]
    }

    return json.dumps(output, indent=2)


def format_metadata(metadata: Dict[str, Any], verbose: bool = False) -> str:
    """Format metadata for compact inline display.

    Extracts key fields for display:
    - programming_language (for code)
    - git_project_name (for code)
    - git_branch (if not main)
    - page_number (for PDFs)

    Args:
        metadata: Full metadata dictionary
        verbose: If True, show more fields

    Returns:
        Formatted metadata string (e.g., "Language: python | Project: backend")
    """
    parts = []

    # Source code metadata
    if "programming_language" in metadata:
        parts.append(f"Language: {metadata['programming_language']}")

    if "git_project_name" in metadata:
        parts.append(f"Project: {metadata['git_project_name']}")

    # Show branch if not main/master
    if verbose and "git_branch" in metadata:
        branch = metadata["git_branch"]
        if branch not in ("main", "master"):
            parts.append(f"Branch: {branch}")

    # PDF metadata
    if "page_number" in metadata:
        parts.append(f"Page: {metadata['page_number']}")

    # Show collection if available and verbose
    if verbose and "collection" in metadata:
        parts.append(f"Collection: {metadata['collection']}")

    return " | ".join(parts) if parts else ""


def extract_snippet(content: str, max_length: int = 200) -> str:
    """Extract content snippet with word boundary awareness.

    Truncates at word boundaries to avoid cutting off mid-word.

    Args:
        content: Full text content
        max_length: Maximum character length

    Returns:
        Truncated snippet (with "..." if truncated)
    """
    if len(content) <= max_length:
        return content

    # Find word boundary near max_length
    snippet = content[:max_length]
    last_space = snippet.rfind(' ')

    # Only truncate at word boundary if we're at least 80% of max_length
    if last_space > max_length * 0.8:
        snippet = snippet[:last_space]

    return snippet + "..."


def _filter_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Filter metadata to essential fields for non-verbose JSON output.

    Args:
        metadata: Full metadata dictionary

    Returns:
        Filtered metadata with only essential fields
    """
    # Essential fields to always include
    essential_fields = {
        "file_path",
        "programming_language",
        "git_project_name",
        "git_branch",
        "page_number",
        "chunk_index"
    }

    # Return only essential fields that exist
    return {
        k: v for k, v in metadata.items()
        if k in essential_fields
    }


def format_summary(
    query: str,
    collection: str,
    results: List[SearchResult],
    filter_description: str = None,
    execution_time_ms: float = None
) -> str:
    """Format a summary of search results.

    Useful for verbose mode to show search statistics.

    Args:
        query: Original search query
        collection: Collection name
        results: List of SearchResult objects
        filter_description: Human-readable filter description
        execution_time_ms: Search execution time in milliseconds

    Returns:
        Formatted summary string
    """
    lines = [
        "Search Summary",
        "=" * 50,
        f"Query:        {query}",
        f"Collection:   {collection}",
        f"Results:      {len(results)}",
    ]

    if filter_description:
        lines.append(f"Filters:      {filter_description}")

    if results:
        avg_score = sum(r.score for r in results) / len(results)
        max_score = max(r.score for r in results)
        min_score = min(r.score for r in results)

        lines.extend([
            f"Avg Score:    {avg_score:.2%}",
            f"Max Score:    {max_score:.2%}",
            f"Min Score:    {min_score:.2%}",
        ])

    if execution_time_ms is not None:
        lines.append(f"Time:         {execution_time_ms:.1f}ms")

    lines.append("=" * 50)

    return "\n".join(lines)

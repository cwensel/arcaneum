"""Result formatters for search output (RDR-007)."""

from typing import Any, Dict, List

from ..utils.search_output import content_preview
from .searcher import SearchResult


def format_text_results(
    query: str,
    results: List[SearchResult],
    offset: int = 0,
    verbose: bool = False,
    max_content_chars: int | None = None,
    cites: List[str] | None = None,
) -> str:
    """Format search results for human-readable terminal display.

    Output format:
        Searching for: "query"
        Found N results (showing results 11-20)

        [11] Score: 95% | Language: python | Project: myproject
            /path/to/file.py

            def authenticate_user(username, password):
                \"\"\"Verify user credentials...

    Args:
        query: Original search query
        results: List of SearchResult objects
        offset: Number of results skipped (for pagination)
        verbose: If True, show more metadata and longer snippets
        max_content_chars: Override snippet and line limits; zero means unlimited
        cites: Each result's cite (RDR-026 Q4), printed in place of its location

    Returns:
        Formatted string for terminal output
    """
    lines = []

    # Header with pagination info
    lines.append(f'Searching for: "{query}"')
    if offset > 0:
        start = offset + 1
        end = offset + len(results)
        lines.append(
            f"Found {len(results)} result{'s' if len(results) != 1 else ''} (showing results {start}-{end})"
        )
    else:
        lines.append(f"Found {len(results)} result{'s' if len(results) != 1 else ''}")
    lines.append("")  # Blank line

    # Format each result (numbered from offset + 1)
    for i, result in enumerate(results, offset + 1):
        location = cites[i - offset - 1] if cites else result.location
        # Score and metadata header
        score_pct = int(result.score * 100)
        metadata_str = format_metadata(result.metadata, verbose=verbose)

        if metadata_str:
            lines.append(f"[{i}] Score: {score_pct}% | {metadata_str}")
        else:
            lines.append(f"[{i}] Score: {score_pct}%")

        # Location
        lines.append(f"    {location}")
        lines.append("")  # Blank line

        if max_content_chars is not None:
            snippet_lines = content_preview(result.content, max_content_chars).split("\n")
        else:
            # Preserve the legacy snippet and line limits unless explicitly overridden.
            snippet_length = 400 if verbose else 200
            snippet = extract_snippet(result.content, max_length=snippet_length)
            max_lines = 10 if verbose else 5
            snippet_lines = snippet.split("\n")[:max_lines]

        for line in snippet_lines:
            lines.append(f"    {line}")

        lines.append("")  # Blank line between results

    return "\n".join(lines)


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
    last_space = snippet.rfind(" ")

    # Only truncate at word boundary if we're at least 80% of max_length
    if last_space > max_length * 0.8:
        snippet = snippet[:last_space]

    return snippet + "..."


def format_summary(
    query: str,
    collection: str,
    results: List[SearchResult],
    filter_description: str = None,
    execution_time_ms: float = None,
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

        lines.extend(
            [
                f"Avg Score:    {avg_score:.2%}",
                f"Max Score:    {max_score:.2%}",
                f"Min Score:    {min_score:.2%}",
            ]
        )

    if execution_time_ms is not None:
        lines.append(f"Time:         {execution_time_ms:.1f}ms")

    lines.append("=" * 50)

    return "\n".join(lines)

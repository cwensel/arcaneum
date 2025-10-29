"""CLI command for semantic search (RDR-007 with RDR-006 enhancements)."""

import click
import logging
import sys
import time
from pathlib import Path
from qdrant_client import QdrantClient
from rich.console import Console

from ..search import (
    SearchEmbedder,
    parse_filter,
    build_filter_description,
    search_collection,
    format_text_results,
    format_json_results,
    format_summary
)
from .errors import InvalidArgumentError, ResourceNotFoundError

console = Console()
logger = logging.getLogger(__name__)


def search_command(
    query: str,
    collection: str,
    vector_name: str,
    filter_arg: str,
    limit: int,
    score_threshold: float,
    output_json: bool,
    verbose: bool
):
    """Search Qdrant collection semantically.

    Args:
        query: Search query text
        collection: Collection name to search
        vector_name: Optional specific vector to use (auto-detects if None)
        filter_arg: Optional metadata filter string
        limit: Maximum number of results
        score_threshold: Optional minimum similarity score
        output_json: If True, output JSON format
        verbose: If True, show detailed output and logging
    """
    # Setup logging based on verbose flag
    if verbose:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('qdrant_client').setLevel(logging.INFO)
    else:
        # Clean output mode - only warnings and errors
        logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')
        logging.getLogger('arcaneum').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.ERROR)
        logging.getLogger('qdrant_client').setLevel(logging.ERROR)

    try:
        # Initialize Qdrant client
        # TODO: Make Qdrant URL configurable via environment or config file
        qdrant_url = "http://localhost:6333"
        client = QdrantClient(url=qdrant_url)

        # Initialize embedder
        # TODO: Make cache dir configurable
        cache_dir = Path.home() / ".cache" / "arcaneum"
        embedder = SearchEmbedder(cache_dir=cache_dir, verify_ssl=True)

        # Parse metadata filter if provided
        query_filter = None
        filter_description = None
        if filter_arg:
            try:
                query_filter = parse_filter(filter_arg)
                filter_description = build_filter_description(query_filter)
                if verbose:
                    logger.info(f"Filter: {filter_description}")
            except ValueError as e:
                raise InvalidArgumentError(f"Invalid filter: {e}")

        # Execute search
        if verbose:
            logger.info(f"Searching collection '{collection}' for: \"{query}\"")

        start_time = time.time()

        results = search_collection(
            client=client,
            embedder=embedder,
            query=query,
            collection_name=collection,
            vector_name=vector_name,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold
        )

        execution_time_ms = (time.time() - start_time) * 1000

        if verbose:
            logger.info(f"Search completed in {execution_time_ms:.1f}ms")

        # Format and output results
        if output_json:
            # JSON output mode
            output = format_json_results(
                query=query,
                collection=collection,
                results=results,
                verbose=verbose
            )
            # Use print() not console.print() for JSON to avoid Rich wrapping
            print(output)
        else:
            # Human-readable text output
            if verbose:
                # Show summary in verbose mode
                summary = format_summary(
                    query=query,
                    collection=collection,
                    results=results,
                    filter_description=filter_description,
                    execution_time_ms=execution_time_ms
                )
                console.print(summary)
                console.print()  # Blank line

            # Show results
            output = format_text_results(
                query=query,
                results=results,
                verbose=verbose
            )
            console.print(output)

        # Exit with success
        sys.exit(0)

    except (InvalidArgumentError, ResourceNotFoundError):
        raise  # Re-raise our custom exceptions for main() to handle
    except ValueError as e:
        # Convert ValueError to InvalidArgumentError for consistency
        raise InvalidArgumentError(str(e))

    except Exception as e:
        # Unexpected errors (Qdrant connection, etc.)
        console.print(f"[ERROR] Search failed: {e}", style="red")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

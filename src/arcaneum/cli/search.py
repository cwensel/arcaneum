"""CLI command for semantic search (RDR-007 with RDR-006 enhancements)."""

import click
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple
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
from ..paths import get_models_dir
from .errors import InvalidArgumentError, ResourceNotFoundError
from .interaction_logger import interaction_logger
from .utils import create_qdrant_client

console = Console()
logger = logging.getLogger(__name__)


def resolve_corpora(corpora: tuple, legacy_option: str, option_name: str) -> List[str]:
    """Resolve corpus targets with backwards compatibility.

    Args:
        corpora: Tuple of corpus names from --corpus option
        legacy_option: Value from legacy --collection or --index option
        option_name: Name of legacy option for error messages ('collection' or 'index')

    Returns:
        List of corpus names to search

    Raises:
        click.UsageError: If both options specified or neither specified
    """
    if corpora and legacy_option:
        raise click.UsageError(f"Cannot use both --corpus and --{option_name}")

    if legacy_option:
        return [legacy_option]  # Silently accept legacy option

    if not corpora:
        raise click.UsageError("Missing required option: --corpus")

    return list(corpora)


def search_command(
    query: str,
    corpora: List[str],
    vector_name: str,
    filter_arg: str,
    limit: int,
    offset: int,
    score_threshold: float,
    output_json: bool,
    verbose: bool
):
    """Search Qdrant collection(s) semantically.

    Args:
        query: Search query text
        corpora: List of collection/corpus names to search
        vector_name: Optional specific vector to use (auto-detects if None)
        filter_arg: Optional metadata filter string
        limit: Maximum number of results
        offset: Number of results to skip (for pagination)
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
        # Initialize Qdrant client with proper timeout and retry configuration
        client = create_qdrant_client(for_search=True)

        # Initialize embedder
        # Use same cache directory as index commands for consistency
        cache_dir = get_models_dir()
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

        # Execute search with interaction logging (RDR-018)
        corpora_str = ", ".join(corpora)
        if verbose:
            logger.info(f"Searching corpus '{corpora_str}' for: \"{query}\"")

        start_time = time.time()

        # Start interaction logging
        interaction_logger.start(
            "search", "semantic",
            corpora=corpora,
            query=query,
            limit=limit,
            offset=offset,
            filters=filter_arg if filter_arg else None,
            score_threshold=score_threshold,
        )

        try:
            # Search each corpus and merge results
            all_results = []
            missing_corpora = []

            for corpus_name in corpora:
                try:
                    corpus_results = search_collection(
                        client=client,
                        embedder=embedder,
                        query=query,
                        collection_name=corpus_name,
                        vector_name=vector_name,
                        limit=limit + offset,  # Get extra for merging
                        offset=0,  # Apply offset after merge
                        query_filter=query_filter,
                        score_threshold=score_threshold
                    )
                    # Tag results with source corpus
                    for result in corpus_results:
                        result['_corpus'] = corpus_name
                    all_results.extend(corpus_results)
                except Exception as e:
                    error_str = str(e).lower()
                    if 'not found' in error_str or 'doesn\'t exist' in error_str:
                        missing_corpora.append(corpus_name)
                        if len(corpora) > 1:
                            # Warn about missing corpus but continue with others
                            if verbose:
                                logger.warning(f"Corpus '{corpus_name}' not found, skipping")
                        else:
                            # Single corpus not found is an error
                            raise ResourceNotFoundError(f"Corpus '{corpus_name}' not found")
                    else:
                        raise

            # If all corpora are missing, error out
            if len(missing_corpora) == len(corpora):
                raise ResourceNotFoundError(f"No matching corpora found: {', '.join(corpora)}")

            # Sort merged results by score (descending) and apply pagination
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            results = all_results[offset:offset + limit]

            execution_time_ms = (time.time() - start_time) * 1000

            if verbose:
                logger.info(f"Search completed in {execution_time_ms:.1f}ms")
                if missing_corpora:
                    logger.warning(f"Missing corpora: {', '.join(missing_corpora)}")

            # Log successful search
            interaction_logger.finish(result_count=len(results))
        except Exception as e:
            # Log failed search
            interaction_logger.finish(error=str(e))
            raise

        # Use first corpus name for backwards-compatible output format
        # (or combined name for multi-corpus)
        display_collection = corpora[0] if len(corpora) == 1 else f"[{', '.join(corpora)}]"

        # Format and output results
        if output_json:
            # JSON output mode
            output = format_json_results(
                query=query,
                collection=display_collection,
                results=results,
                limit=limit,
                offset=offset,
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
                    collection=display_collection,
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
                limit=limit,
                offset=offset,
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

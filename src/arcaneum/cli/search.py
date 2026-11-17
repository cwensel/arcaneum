"""CLI commands for semantic search and passages (RDR-007, RDR-006, RDR-026).

The commands format ``arcaneum.retrieval.search_semantic`` and
``arcaneum.retrieval.get_passage``; the lookups themselves and their coded
errors live in the print-free core (RDR-026).
"""

import logging
import sys
import time
from typing import List

from rich.console import Console

from ..retrieval import get_passage, search_semantic
from ..search import format_summary, format_text_results
from ..utils.search_output import format_compact_results
from .interaction_logger import interaction_logger
from .output import format_json_response
from .utils import create_meili_client, create_qdrant_client, resolve_config_path

console = Console()
logger = logging.getLogger(__name__)


def search_command(
    query: str,
    corpora: List[str],
    vector_name: str,
    filter_arg: str,
    limit: int,
    offset: int,
    score_threshold: float,
    output_json: bool,
    verbose: bool,
    include_references: bool = False,
    output_format: str | None = None,
    max_content_chars: int | None = None,
    allow_download: bool = True,
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
        include_references: Include PDF bibliography chunks
        output_format: Text, compact, or JSON output; None preserves output_json
        max_content_chars: Explicit content cap; zero is unlimited, None uses format defaults
        allow_download: Download a query model missing from the local cache
            (``--offline`` turns it off: such a model is semantic-unavailable)
    """
    # Setup logging based on verbose flag
    if verbose:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("qdrant_client").setLevel(logging.INFO)
    else:
        # Clean output mode - only warnings and errors
        logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
        logging.getLogger("arcaneum").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("qdrant_client").setLevel(logging.ERROR)

    corpora_str = ", ".join(corpora)
    if verbose:
        logger.info(f"Searching corpus '{corpora_str}' for: \"{query}\"")

    start_time = time.time()

    with interaction_logger.track(
        "search",
        "semantic",
        corpora=corpora,
        query=query,
        limit=limit,
        offset=offset,
        filters=filter_arg if filter_arg else None,
        score_threshold=score_threshold,
        include_references=include_references,
    ) as ctx:
        result = search_semantic(
            query,
            corpora,
            vector_name=vector_name,
            filter=filter_arg,
            limit=limit,
            offset=offset,
            score_threshold=score_threshold,
            include_references=include_references,
            allow_download=allow_download,
            client=create_qdrant_client(for_search=True),
            verbose=verbose,
        )
        ctx["result_count"] = len(result.results)

    results = result.results
    execution_time_ms = (time.time() - start_time) * 1000
    if verbose:
        if result.filter_description:
            logger.info(f"Filter: {result.filter_description}")
        logger.info(f"Search completed in {execution_time_ms:.1f}ms")
        if result.missing_corpora:
            logger.warning(f"Missing corpora: {', '.join(result.missing_corpora)}")

    # Format and output results; the cites come from the core's hit builder.
    cites = [hit["cite"] for hit in result.to_hits(max_content_chars=0)]
    if output_format == "compact":
        print(
            format_compact_results(
                query,
                corpora,
                [
                    {"cite": cite, "content": r.content, "score": r.score}
                    for cite, r in zip(cites, results)
                ],
                offset=offset,
                max_content_chars=500 if max_content_chars is None else max_content_chars,
            )
        )
    elif output_json or output_format == "json":
        data = result.to_data(verbose=verbose, max_content_chars=max_content_chars)
        # print() not console.print() for JSON to avoid Rich wrapping; the
        # envelope's ensure_ascii escapes what PDF text can carry.
        print(format_json_response("success", f"Found {len(data['hits'])} results", data))
    else:
        # Human-readable text output
        if verbose:
            summary = format_summary(
                query=query,
                collection=result.display_collection,
                results=results,
                filter_description=result.filter_description,
                execution_time_ms=execution_time_ms,
            )
            console.print(summary)
            console.print()  # Blank line

        output = format_text_results(
            query=query,
            results=results,
            cites=cites,
            offset=offset,
            verbose=verbose,
            max_content_chars=max_content_chars,
        )
        if max_content_chars is None:
            console.print(output)
        else:
            console.print(output, markup=False, highlight=False)

    sys.exit(0)


def _chunk_span(chunks) -> str:
    indexes = [chunk.chunk_index for chunk in chunks if chunk.chunk_index is not None]
    if not indexes:
        return "1 chunk"
    if indexes[0] == indexes[-1]:
        return f"chunk {indexes[0]}"
    return f"chunks {indexes[0]}-{indexes[-1]}"


def passage_command(
    chunk_id: str,
    corpus: str,
    *,
    before: int,
    after: int,
    max_chars: int,
    output_json: bool,
):
    """Print the text around a search hit (``arc search passage``).

    Args:
        chunk_id: A hit's ``chunk_id``, from either search mode
        corpus: The corpus that holds the chunk
        before: Neighbouring chunks before the hit (0-5)
        after: Neighbouring chunks after the hit (0-5)
        max_chars: Bound on the passage text; zero is unlimited
        output_json: Print the ``{status, message, data, errors}`` envelope
    """
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    with interaction_logger.track(
        "search", "passage", corpora=[corpus], chunk_id=chunk_id, before=before, after=after
    ) as ctx:
        result = get_passage(
            corpus,
            chunk_id,
            before=before,
            after=after,
            max_chars=max_chars,
            qdrant_client=create_qdrant_client(
                config_path=resolve_config_path(migrate=False), for_search=True
            ),
            meili_client=create_meili_client(generate_key=False),
        )
        ctx["result_count"] = len(result.chunks)

    if output_json:
        data = result.to_data()
        print(format_json_response("success", f"Passage of {len(result.chunks)} chunks", data))
        return

    # Plain print: passage text is source text, never rich markup.
    print(f"{result.cite} ({_chunk_span(result.chunks)})")
    print(f"first: {result.first_chunk_id}")
    print(f"last:  {result.last_chunk_id}")
    print()
    print(result.text)
    if result.truncated:
        print("[truncated]")

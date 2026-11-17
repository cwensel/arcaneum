"""Semantic (Qdrant) search for the retrieval core (RDR-026)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..cli.concurrency import acquire_embedder_slot
from ..cli.errors import (
    ArcaneumError,
    BadFilterError,
    InvalidArgumentError,
    SemanticUnavailableError,
)
from ..cli.search_merge import fetch_from_corpora, per_corpus_limit
from ..paths import get_models_dir
from ..search.embedder import ModelNotCachedError, QueryEmbeddingError, SearchEmbedder
from ..search.filters import build_filter_description, parse_filter
from ..search.formatter import build_json_results
from ..search.searcher import SearchResult, search_collection
from .failures import is_connection_failure, service_down

logger = logging.getLogger(__name__)


class _MissingCollection(Exception):
    """Sentinel raised by the per-corpus fetch when a collection is absent.

    Avoids string-matching Qdrant error messages: ``fetch_from_corpora``'s
    ``is_missing`` predicate matches on this type.
    """


@dataclass
class SemanticSearchResult:
    """Merged semantic hits across the requested corpora."""

    query: str
    corpora: List[str]
    limit: int
    offset: int
    results: List[SearchResult]
    missing_corpora: List[str] = field(default_factory=list)
    filter_description: Optional[str] = None

    @property
    def display_collection(self) -> str:
        """The corpus name, or the bracketed list for a multi-corpus search."""
        if len(self.corpora) == 1:
            return self.corpora[0]
        return f"[{', '.join(self.corpora)}]"

    def to_data(
        self, verbose: bool = False, max_content_chars: int | None = None
    ) -> Dict[str, Any]:
        """The JSON ``data`` the CLI and the MCP server emit."""
        data = build_json_results(
            query=self.query,
            collection=self.display_collection,
            results=self.results,
            limit=self.limit,
            offset=self.offset,
            verbose=verbose,
            max_content_chars=max_content_chars,
        )
        data["missing_corpora"] = list(self.missing_corpora)
        return data


def search_semantic(
    query: str,
    corpora: List[str],
    *,
    vector_name: Optional[str] = None,
    filter: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    score_threshold: Optional[float] = None,
    include_references: bool = False,
    allow_download: bool = True,
    client=None,
    embedder: Optional[SearchEmbedder] = None,
    verbose: bool = False,
) -> SemanticSearchResult:
    """Search Qdrant collections semantically and merge hits by score.

    Args:
        query: Search query text
        corpora: Collection/corpus names to search
        vector_name: Named vector to use (auto-detected when None)
        filter: Metadata filter string (``search/filters.py`` dialect)
        limit: Maximum number of merged results
        offset: Number of merged results to skip
        score_threshold: Minimum similarity score
        include_references: Include PDF bibliography chunks
        allow_download: Download a query model missing from the local cache;
            when False such a model is ``semantic-unavailable`` (RDR-026 Q9)
        client: Qdrant client (created from config when None)
        embedder: Query embedder (created inside the embedder slot when None)
        verbose: Log skipped corpora

    Raises:
        UnknownCorpusError: no requested corpus exists
        BadFilterError: the filter does not parse
        ServiceDownError: Qdrant is unreachable
        SemanticUnavailableError: no embedder slot, or the model is uncached
            with downloads off
        InvalidArgumentError: the collection cannot serve the query (no such
            vector, unconfigured model, stale prompt policy)
        ArcaneumError: any other search failure
    """
    query_filter = None
    filter_description = None
    if filter:
        try:
            query_filter = parse_filter(filter)
            filter_description = build_filter_description(query_filter)
        except ValueError as e:
            raise BadFilterError(f"Invalid filter: {e}") from e

    try:
        if client is None:
            from ..cli.utils import create_qdrant_client

            client = create_qdrant_client(for_search=True)

        # Cap concurrent embedder loads (file-lock semaphore). Cold-loading
        # the embedding model in N parallel processes thrashes RAM/swap on
        # consumer hardware; ARCANEUM_SEARCH_CONCURRENCY tunes the cap.
        with acquire_embedder_slot():
            if embedder is None:
                embedder = SearchEmbedder(cache_dir=get_models_dir(), allow_download=allow_download)

            fetch_limit = per_corpus_limit(corpora, limit, offset)

            def _fetch(corpus_name: str) -> List[SearchResult]:
                if not client.collection_exists(corpus_name):
                    raise _MissingCollection(corpus_name)
                return search_collection(
                    client=client,
                    embedder=embedder,
                    query=query,
                    collection_name=corpus_name,
                    vector_name=vector_name,
                    limit=fetch_limit,
                    offset=0,  # Apply offset after merge
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                    include_references=include_references,
                )

            per_corpus, missing_corpora = fetch_from_corpora(
                corpora,
                _fetch,
                lambda exc: isinstance(exc, _MissingCollection),
                logger,
                verbose,
            )
    except ArcaneumError:
        raise
    except ModelNotCachedError as e:
        raise SemanticUnavailableError(str(e)) from e
    except QueryEmbeddingError as e:
        # Not Qdrant: the model host or the local model failed.
        raise ArcaneumError(f"Search failed: {e}") from e
    except Exception as e:
        if is_connection_failure(e):
            raise service_down("Qdrant", e) from e
        if isinstance(e, ValueError):
            raise InvalidArgumentError(str(e)) from e
        raise ArcaneumError(f"Search failed: {e}") from e

    merged = [hit for hits in per_corpus for hit in hits]
    merged.sort(key=lambda hit: hit.score, reverse=True)

    return SemanticSearchResult(
        query=query,
        corpora=list(corpora),
        limit=limit,
        offset=offset,
        results=merged[offset : offset + limit],
        missing_corpora=missing_corpora,
        filter_description=filter_description,
    )

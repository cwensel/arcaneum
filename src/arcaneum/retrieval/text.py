"""Full-text (MeiliSearch) search for the retrieval core (RDR-026)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..cli.errors import ArcaneumError, BadFilterError, ServiceDownError
from ..cli.search_merge import fetch_from_corpora, per_corpus_limit
from ..search.filters import parse_filter, to_meilisearch_filter, unfilterable_field_message
from .failures import is_connection_failure, service_down
from .hits import build_hit, read_corpora_source_roots, search_data

logger = logging.getLogger(__name__)

MEILI_START_HINT = "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"


class _MissingCorpus(Exception):
    """Sentinel raised by the per-corpus fetch when an index is absent.

    MeiliSearch's index_exists() is an upfront check, so there is no driver
    exception to inspect: ``fetch_from_corpora``'s ``is_missing`` predicate
    matches on this type.
    """


@dataclass
class TextSearchResult:
    """Merged MeiliSearch hits across the requested corpora."""

    query: str
    corpora: List[str]
    limit: int
    offset: int
    hits: List[Dict[str, Any]]
    estimated_total_hits: int = 0
    processing_time_ms: int = 0
    missing_corpora: List[str] = field(default_factory=list)
    source_roots: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def display_index(self) -> str:
        """The corpus name, or the bracketed list for a multi-corpus search."""
        if len(self.corpora) == 1:
            return self.corpora[0]
        return f"[{', '.join(self.corpora)}]"

    def to_hits(
        self, verbose: bool = False, max_content_chars: int | None = None
    ) -> List[Dict[str, Any]]:
        """The merged hits as RDR-026 Q2 hits, in round-robin order."""
        hits = []
        for hit in self.hits:
            corpus = hit.get("_corpus") or ""  # set by search_text's per-corpus fetch
            hits.append(
                build_hit(
                    corpus=corpus,
                    chunk_id=hit.get("id"),
                    payload=hit,
                    content=hit.get("content") or "",
                    score=hit.get("_rankingScore"),
                    roots=self.source_roots.get(corpus, ()),
                    max_content_chars=max_content_chars,
                    verbose=verbose,
                )
            )
        return hits

    def to_data(
        self, verbose: bool = False, max_content_chars: int | None = None
    ) -> Dict[str, Any]:
        """The JSON ``data`` the CLI and the MCP server emit (RDR-026 Q2)."""
        return search_data(
            query=self.query,
            corpora=self.corpora,
            missing_corpora=self.missing_corpora,
            limit=self.limit,
            offset=self.offset,
            hits=self.to_hits(verbose=verbose, max_content_chars=max_content_chars),
        )


def _is_filter_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        getattr(exc, "code", None) == "invalid_search_filter"
        or "invalid_search_filter" in message
        or "is not filterable" in message
    )


def _bad_filter(client, corpus_name: str, exc: Exception) -> BadFilterError:
    """Name the rejected attribute and list the index's filterable ones."""
    attr_match = re.search(r"Attribute `([^`]+)`", str(exc))
    bad_attr = attr_match.group(1) if attr_match else None

    try:
        settings = client.get_index_settings(corpus_name)
        filterable = list(settings.get("filterableAttributes") or [])
    except Exception:
        filterable = None

    if bad_attr and filterable is not None:
        return BadFilterError(unfilterable_field_message(bad_attr, filterable, corpus_name))

    if bad_attr:
        lines = [f"Filter attribute '{bad_attr}' is not filterable."]
    else:
        lines = [f"Invalid filter: {getattr(exc, 'message', None) or exc}"]
    if filterable:
        lines.append(f"Available filterable attributes for index '{corpus_name}':")
        lines.extend(f"  - {attr}" for attr in filterable)
    elif filterable is not None:
        lines.append(
            f"Index '{corpus_name}' has no filterable attributes configured; "
            "filters cannot be used with this index."
        )
    return BadFilterError("\n".join(lines))


def search_text(
    query: str,
    corpora: List[str],
    *,
    filter: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    client=None,
    qdrant_client=None,
    verbose: bool = False,
) -> TextSearchResult:
    """Search MeiliSearch indexes and merge hits round-robin across corpora.

    Args:
        query: Search query (quotes for exact phrases)
        corpora: MeiliSearch index names to search
        filter: Metadata filter (``search/filters.py``: the neutral dialect,
            translated with every value quoted, or a native MeiliSearch
            expression)
        limit: Maximum number of merged hits
        offset: Number of merged hits to skip
        client: FullTextClient (created from config when None)
        qdrant_client: Qdrant client used only to read each corpus's
            ``source_roots`` for its cites; None, or a failed read, cites the
            absolute path (RDR-026 Q4)
        verbose: Log skipped corpora

    Raises:
        UnknownCorpusError: no requested corpus exists
        BadFilterError: the filter does not parse, is Qdrant JSON, uses
            ``contains``, or MeiliSearch rejects it (e.g. an unfilterable field)
        ServiceDownError: MeiliSearch is unreachable
        ArcaneumError: any other search failure
    """
    meili_filter = None
    if filter:
        try:
            meili_filter = to_meilisearch_filter(parse_filter(filter))
        except ValueError as e:
            raise BadFilterError(f"Invalid filter: {e}") from e

    if client is None:
        from ..cli.utils import create_meili_client

        client = create_meili_client()

    if not client.health_check():
        raise ServiceDownError(f"MeiliSearch server not available. {MEILI_START_HINT}")

    fetch_limit = per_corpus_limit(corpora, limit, offset)
    totals = {"processing_time_ms": 0, "estimated_total_hits": 0}

    def _fetch(corpus_name: str) -> list:
        if not client.index_exists(corpus_name):
            raise _MissingCorpus(corpus_name)
        try:
            results = client.search(
                corpus_name,
                query,
                filter=meili_filter,
                limit=fetch_limit,
                offset=0,  # Apply offset after merge
                attributes_to_highlight=["content"],
                show_ranking_score=True,
            )
        except Exception as e:
            if meili_filter and _is_filter_error(e):
                raise _bad_filter(client, corpus_name, e) from e
            raise
        hits = results.get("hits", [])
        for hit in hits:
            hit["_corpus"] = corpus_name
        totals["processing_time_ms"] += results.get("processingTimeMs", 0)
        totals["estimated_total_hits"] += results.get("estimatedTotalHits", 0)
        return hits

    try:
        per_corpus_hits, missing_corpora = fetch_from_corpora(
            corpora,
            _fetch,
            lambda exc: isinstance(exc, _MissingCorpus),
            logger,
            verbose,
        )
    except ArcaneumError:
        raise
    except Exception as e:
        if is_connection_failure(e):
            raise service_down("MeiliSearch", e) from e
        raise ArcaneumError(f"Search failed: {e}") from e

    # MeiliSearch scores are not shown comparable across indexes (RDR-026
    # Q7), so a multi-corpus search merges positionally, round-robin: it keeps
    # each corpus's own ranking but does not rank across corpora.
    all_hits = [hit for hits in per_corpus_hits for hit in hits]
    if len(corpora) > 1:
        hits_by_corpus: Dict[str, List[Dict[str, Any]]] = {}
        for hit in all_hits:
            hits_by_corpus.setdefault(hit.get("_corpus", "unknown"), []).append(hit)
        longest = max((len(h) for h in hits_by_corpus.values()), default=0)
        all_hits = [
            hits_by_corpus[corpus][i]
            for i in range(longest)
            for corpus in corpora
            if corpus in hits_by_corpus and i < len(hits_by_corpus[corpus])
        ]

    page = all_hits[offset : offset + limit]
    return TextSearchResult(
        query=query,
        corpora=list(corpora),
        limit=limit,
        offset=offset,
        hits=page,
        source_roots=read_corpora_source_roots(
            qdrant_client, sorted({hit.get("_corpus") for hit in page if hit.get("_corpus")})
        ),
        estimated_total_hits=totals["estimated_total_hits"],
        processing_time_ms=totals["processing_time_ms"],
        missing_corpora=missing_corpora,
    )

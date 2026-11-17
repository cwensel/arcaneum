"""Full-text (MeiliSearch) search for the retrieval core (RDR-026)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..cli.errors import ArcaneumError, BadFilterError, ServiceDownError
from ..cli.search_merge import fetch_from_corpora, per_corpus_limit
from ..utils.search_output import limit_json_content
from .failures import is_connection_failure, service_down

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

    @property
    def display_index(self) -> str:
        """The corpus name, or the bracketed list for a multi-corpus search."""
        if len(self.corpora) == 1:
            return self.corpora[0]
        return f"[{', '.join(self.corpora)}]"

    def to_data(self, max_content_chars: int | None = None) -> Dict[str, Any]:
        """The JSON ``data`` the CLI and the MCP server emit."""
        hits = self.hits
        if max_content_chars is not None:
            hits = [limit_json_content(hit, max_content_chars) for hit in hits]
        return {
            "query": self.query,
            "index": self.display_index,
            "corpora": list(self.corpora),
            "missing_corpora": list(self.missing_corpora),
            "hits": hits,
            "estimatedTotalHits": self.estimated_total_hits,
            "processingTimeMs": self.processing_time_ms,
            "limit": self.limit,
            "offset": self.offset,
        }


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
    if bad_attr:
        lines = [f"Filter attribute '{bad_attr}' is not filterable."]
    else:
        lines = [f"Invalid filter: {getattr(exc, 'message', None) or exc}"]

    try:
        settings = client.get_index_settings(corpus_name)
        filterable = list(settings.get("filterableAttributes") or [])
    except Exception:
        filterable = None

    if filterable:
        lines.append(f"Available filterable attributes for index '{corpus_name}':")
        lines.extend(f"  - {attr}" for attr in filterable)
        if bad_attr:
            for attr in filterable:
                if bad_attr.lower() in attr.lower() or attr.lower() in bad_attr.lower():
                    lines.append(f'Hint: Try --filter "{attr}=<value>"')
                    break
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
    verbose: bool = False,
) -> TextSearchResult:
    """Search MeiliSearch indexes and merge hits round-robin across corpora.

    Args:
        query: Search query (quotes for exact phrases)
        corpora: MeiliSearch index names to search
        filter: MeiliSearch filter expression
        limit: Maximum number of merged hits
        offset: Number of merged hits to skip
        client: FullTextClient (created from config when None)
        verbose: Log skipped corpora

    Raises:
        UnknownCorpusError: no requested corpus exists
        BadFilterError: MeiliSearch rejects the filter
        ServiceDownError: MeiliSearch is unreachable
        ArcaneumError: any other search failure
    """
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
                filter=filter,
                limit=fetch_limit,
                offset=0,  # Apply offset after merge
                attributes_to_highlight=["content"],
            )
        except Exception as e:
            if filter and _is_filter_error(e):
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

    return TextSearchResult(
        query=query,
        corpora=list(corpora),
        limit=limit,
        offset=offset,
        hits=all_hits[offset : offset + limit],
        estimated_total_hits=totals["estimated_total_hits"],
        processing_time_ms=totals["processing_time_ms"],
        missing_corpora=missing_corpora,
    )

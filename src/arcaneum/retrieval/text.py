"""Full-text search for the retrieval core (RDR-026), through a ``TextBackend`` (RDR-027 Q1)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..cli.errors import ArcaneumError, BadFilterError
from ..cli.search_merge import fetch_from_corpora, per_corpus_limit
from ..search.filters import parse_filter
from .failures import is_connection_failure, service_down
from .hits import build_hit, read_corpora_source_roots, search_data

if TYPE_CHECKING:
    from .text_backend import TextBackend

logger = logging.getLogger(__name__)


class _MissingCorpus(Exception):
    """Sentinel raised by the per-corpus fetch when an index is absent.

    The backend's exists() is an upfront check, so there is no driver
    exception to inspect: ``fetch_from_corpora``'s ``is_missing`` predicate
    matches on this type.
    """


@dataclass
class TextSearchResult:
    """Merged text-backend hits across the requested corpora."""

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


def search_text(
    query: str,
    corpora: List[str],
    *,
    filter: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    text_backend: Optional["TextBackend"] = None,
    qdrant_client=None,
    verbose: bool = False,
) -> TextSearchResult:
    """Search the corpora's text backend and merge hits round-robin across corpora.

    Args:
        query: Search query (quotes for exact phrases)
        corpora: Corpus names to search
        filter: Metadata filter (``search/filters.py``: the neutral dialect,
            translated with every value quoted, or a native MeiliSearch
            expression)
        limit: Maximum number of merged hits
        offset: Number of merged hits to skip
        text_backend: The corpora's ``TextBackend`` (MeiliSearch from config
            when None)
        qdrant_client: Qdrant client used only to read each corpus's
            ``source_roots`` for its cites; None, or a failed read, cites the
            absolute path (RDR-026 Q4)
        verbose: Log skipped corpora

    Raises:
        UnknownCorpusError: no requested corpus exists
        BadFilterError: the filter does not parse, or the backend refuses it
            (MeiliSearch: Qdrant JSON, ``contains``, an unfilterable field)
        ServiceDownError: the text backend is unreachable
        ArcaneumError: any other search failure
    """
    parsed = None
    if filter:
        try:
            parsed = parse_filter(filter)
        except ValueError as e:
            raise BadFilterError(f"Invalid filter: {e}") from e

    if text_backend is None:
        from ..cli.utils import create_meili_client
        from ..fulltext.backend import MeiliTextBackend

        text_backend = MeiliTextBackend(create_meili_client())

    text_backend.check_filter(parsed)
    text_backend.health()

    fetch_limit = per_corpus_limit(corpora, limit, offset)
    totals = {"processing_time_ms": 0, "estimated_total_hits": 0}

    def _fetch(corpus_name: str) -> list:
        if not text_backend.exists(corpus_name):
            raise _MissingCorpus(corpus_name)
        page = text_backend.search(
            corpus_name,
            query,
            filter=parsed,
            limit=fetch_limit,
            offset=0,  # Apply offset after merge
        )
        for hit in page.hits:
            hit["_corpus"] = corpus_name
        totals["processing_time_ms"] += page.processing_time_ms
        totals["estimated_total_hits"] += page.estimated_total_hits
        return page.hits

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
            raise service_down(text_backend.service, e) from e
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

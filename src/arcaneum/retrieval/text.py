"""Full-text search for the retrieval core (RDR-026), through a ``TextBackend`` (RDR-027 Q1).

Each corpus is searched on the backend its metadata records (RDR-027 Q2):
MeiliSearch, or Qdrant's text index over the chunk points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..cli.errors import ArcaneumError, BadFilterError, ResourceNotFoundError
from ..cli.search_merge import fetch_from_corpora, per_corpus_limit
from ..indexing.collection_metadata import text_backend_of
from ..search.filters import parse_filter
from .failures import is_connection_failure, service_down
from .hits import build_hit, read_corpora_metadata, roots_of, search_data

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
    missing_corpora: List[str] = field(default_factory=list)
    source_roots: Dict[str, List[str]] = field(default_factory=dict)
    # Each corpus's type, for highlighting with its stop words (RDR-027 Q5).
    corpus_types: Dict[str, Optional[str]] = field(default_factory=dict)

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
    """Search each corpus on its text backend and merge hits round-robin across corpora.

    Each corpus's metadata is read once, before any search, for its
    ``text_backend`` and its ``source_roots``. A corpus whose metadata records
    no backend, or cannot be read (Qdrant down, or a MeiliSearch-only index),
    is a MeiliSearch corpus.

    Args:
        query: Search query (quotes for exact phrases)
        corpora: Corpus names to search
        filter: Metadata filter (``search/filters.py``: the neutral dialect,
            translated for each backend, or a native expression)
        limit: Maximum number of merged hits
        offset: Number of merged hits to skip
        text_backend: The ``TextBackend`` for MeiliSearch corpora (MeiliSearch
            from config when None); its health is checked only when a
            MeiliSearch corpus is searched
        qdrant_client: Qdrant client that reads each corpus's metadata and
            serves the Qdrant corpora; None, or a failed read, makes every
            corpus a MeiliSearch corpus cited by absolute path (RDR-026 Q4)
        verbose: Log skipped corpora

    Raises:
        UnknownCorpusError: no requested corpus exists
        ResourceNotFoundError: a Qdrant corpus has lost its text index
        BadFilterError: the filter does not parse, or a corpus's backend
            refuses it (MeiliSearch: Qdrant JSON, ``contains``, an
            unfilterable field; Qdrant: a MeiliSearch expression)
        ServiceDownError: a searched corpus's backend is unreachable
        ArcaneumError: any other search failure
    """
    parsed = None
    if filter:
        try:
            parsed = parse_filter(filter)
        except ValueError as e:
            raise BadFilterError(f"Invalid filter: {e}") from e

    metadata = read_corpora_metadata(qdrant_client, corpora)
    try:
        backend_names = {corpus: text_backend_of(metadata[corpus]) for corpus in corpora}
    except ValueError as e:
        raise ArcaneumError(str(e)) from e
    corpus_types = {corpus: metadata[corpus].get("collection_type") for corpus in corpora}

    backends: Dict[str, "TextBackend"] = {}
    if "meili" in backend_names.values():
        if text_backend is None:
            from ..cli.utils import create_meili_client
            from ..fulltext.backend import MeiliTextBackend

            text_backend = MeiliTextBackend(create_meili_client())
        backends["meili"] = text_backend
    if "qdrant" in backend_names.values():
        from .qdrant_text import QdrantTextBackend

        backends["qdrant"] = QdrantTextBackend(
            qdrant_client,
            corpus_types={c: t for c, t in corpus_types.items() if backend_names[c] == "qdrant"},
        )

    # Every filter refusal precedes every request.
    for backend in backends.values():
        backend.check_filter(parsed)
    for backend in backends.values():
        backend.health()

    fetch_limit = per_corpus_limit(corpora, limit, offset)

    def _fetch(corpus_name: str) -> list:
        backend = backends[backend_names[corpus_name]]
        try:
            if not backend.exists(corpus_name):
                if backend.name == "qdrant":
                    # Its metadata was read, so the collection exists.
                    raise ResourceNotFoundError(
                        f"Corpus '{corpus_name}' is served by Qdrant but has no text index. "
                        f"Rebuild it with: arc corpus repair {corpus_name}"
                    )
                raise _MissingCorpus(corpus_name)
            page = backend.search(
                corpus_name,
                query,
                filter=parsed,
                limit=fetch_limit,
                offset=0,  # Apply offset after merge
            )
        except (ArcaneumError, _MissingCorpus):
            raise
        except Exception as e:
            if is_connection_failure(e):
                raise service_down(backend.service, e) from e
            raise
        for hit in page.hits:
            hit["_corpus"] = corpus_name
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
        raise ArcaneumError(f"Search failed: {e}") from e

    # Scores are not comparable across corpora (RDR-026 Q7), and BM25's are
    # unbounded where MeiliSearch's lie in 0-1 (RDR-027 Q5), so a
    # multi-corpus search merges positionally, round-robin: it keeps each
    # corpus's own ranking but does not rank across corpora.
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
        source_roots={corpus: roots_of(metadata[corpus]) for corpus in corpora},
        corpus_types=corpus_types,
        missing_corpora=missing_corpora,
    )

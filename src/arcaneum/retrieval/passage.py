"""Passage lookup for the retrieval core (RDR-026 Q8).

``get_passage`` expands one chunk into the text around it: the chunk plus up
to ``before`` and ``after`` neighbours from the same file and version, in
``chunk_index`` order, with chunk overlap stitched away.

1. The id resolves through whichever system holds it: Qdrant ``retrieve`` when
   it is a valid point id, else (or on a miss) MeiliSearch ``get_document``.
   The system that holds the chunk is the seed's source (RDR-026 Q5: chunk ids
   need not be shared across systems).
2. The window comes from the seed's source only, since a MeiliSearch-to-Qdrant
   backfill re-chunks a file and one system's ordinals do not select the
   other's content. A Qdrant seed scrolls on ``file_path``, the
   ``chunk_index`` range and the seed's version keys, creating no payload
   index. A MeiliSearch seed pages a ``file_path`` search and selects
   client-side, since MeiliSearch filters on neither ``chunk_index`` nor
   ``git_version_identifier``. No window crosses ``file_path`` or version.
3. Chunks splice by character offsets where both carry them; otherwise the
   longest suffix of one chunk that equals a prefix of the next is trimmed,
   and chunks that share no overlap join with a blank line.
4. ``max_chars`` bounds the text and sets ``truncated``.

Nothing here writes: no payload index, no MeiliSearch settings change.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..cli.errors import (
    ArcaneumError,
    InvalidArgumentError,
    ServiceDownError,
    UnknownChunkError,
    UnknownCorpusError,
)
from ..utils.search_output import limit_content
from .failures import is_connection_failure, service_down
from .hits import SCHEMA_VERSION, _int, _sanitize, build_cite, read_source_roots

logger = logging.getLogger(__name__)

MAX_NEIGHBOURS = 5
DEFAULT_NEIGHBOURS = 1

# The keys that tell versions of one file_path apart (``--git-version``).
VERSION_KEYS = ("git_project_identifier", "git_version_identifier")

# A text overlap shorter than this is taken for a coincidence: trimming a
# chance match ("the ") would drop real text, while a blank-line join keeps it.
MIN_OVERLAP_CHARS = 16

QDRANT = "qdrant"
MEILISEARCH = "meilisearch"

_QDRANT_SCROLL_PAGE = 64
_QDRANT_SCROLL_PAGES = 8

_UNSIGNED = re.compile(r"[0-9]+")
_MEILI_ID = re.compile(r"[A-Za-z0-9_-]{1,511}")


@dataclass(frozen=True)
class PassageChunk:
    """One chunk of a passage: its id, its ordinal in the file and its text."""

    chunk_id: str
    chunk_index: Optional[int]
    text: str
    payload: Mapping[str, Any]


@dataclass
class PassageResult:
    """A window of chunks around one seed chunk, stitched into one text."""

    corpus: str
    chunk_id: str
    cite: str
    chunks: List[PassageChunk]
    text: str
    truncated: bool
    source: str

    @property
    def first_chunk_id(self) -> str:
        return self.chunks[0].chunk_id

    @property
    def last_chunk_id(self) -> str:
        return self.chunks[-1].chunk_id

    def to_data(self) -> Dict[str, Any]:
        """The JSON ``data`` the CLI and the MCP server emit (RDR-026 Q8)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "corpus": self.corpus,
            "chunk_id": self.chunk_id,
            "cite": self.cite,
            "first_chunk_id": self.first_chunk_id,
            "last_chunk_id": self.last_chunk_id,
            "chunks": [
                {"chunk_id": chunk.chunk_id, "chunk_index": chunk.chunk_index}
                for chunk in self.chunks
            ],
            "text": self.text,
            "truncated": self.truncated,
        }


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------


def overlap_length(previous: str, following: str) -> int:
    """Length of the longest suffix of ``previous`` that is a prefix of ``following``.

    An overlap can be most of a chunk (a file's short last chunk, or the code
    chunker's line fallback, which keeps whole lines), but never all of one:
    every chunk starts past the previous one's start and adds text past its
    overlap, so a whole-chunk match is repeated content and is not trimmed.
    Zero when the match is shorter than ``MIN_OVERLAP_CHARS``, which a chance
    match ("the ") rarely reaches. Without offsets, repetition that matches
    a proper suffix and prefix is indistinguishable from overlap.
    """
    bound = min(len(previous), len(following)) - 1
    for length in range(bound, MIN_OVERLAP_CHARS - 1, -1):
        if previous.endswith(following[:length]):
            return length
    return 0


def _offset_overlap(previous: PassageChunk, following: PassageChunk) -> Optional[int]:
    """The overlap the chunks' character offsets give, when it agrees with their text."""
    end = _int(previous.payload.get("chunk_end_char"))
    start = _int(following.payload.get("chunk_start_char"))
    if end is None or start is None or _int(previous.payload.get("chunk_start_char")) is None:
        return None
    overlap = end - start
    if not 0 <= overlap < len(following.text) or overlap > len(previous.text):
        return None
    if overlap and not previous.text.endswith(following.text[:overlap]):
        return None
    return overlap


def _continuation(previous: PassageChunk, following: PassageChunk) -> str:
    """What ``following`` adds after ``previous`` in the stitched text."""
    consecutive = (
        previous.chunk_index is not None
        and following.chunk_index is not None
        and following.chunk_index == previous.chunk_index + 1
    )
    if consecutive:
        overlap = _offset_overlap(previous, following)
        if overlap is None:
            overlap = overlap_length(previous.text, following.text) or None
        if overlap is not None:
            return following.text[overlap:]
    return "\n\n" + following.text


def stitch(chunks: List[PassageChunk]) -> str:
    """The chunks' text in order, each overlap appearing once."""
    if not chunks:
        return ""
    parts = [chunks[0].text]
    for previous, following in zip(chunks, chunks[1:]):
        parts.append(_continuation(previous, following))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def _qdrant_point_id(chunk_id: str):
    """``chunk_id`` as a Qdrant point id (unsigned integer or UUID), or None."""
    if _UNSIGNED.fullmatch(chunk_id):
        number = int(chunk_id)
        return number if number < 2**64 else None
    try:
        return str(uuid.UUID(chunk_id))
    except ValueError:
        return None


def _is_chunk(payload: Mapping[str, Any]) -> bool:
    """False for collection metadata and file-manifest points, which are not chunks."""
    from ..indexing.collection_metadata import FILE_MANIFEST_PAYLOAD_KEY, METADATA_PAYLOAD_KEY

    return not payload.get(METADATA_PAYLOAD_KEY) and FILE_MANIFEST_PAYLOAD_KEY not in payload


def _content(payload: Mapping[str, Any]) -> str:
    # Qdrant stores chunk text as ``text`` (older points as ``content``);
    # MeiliSearch as ``content``.
    content = payload.get("content")
    if content is None:
        content = payload.get("text")
    return _sanitize(str(content or ""))


def _chunk(chunk_id: Any, payload: Mapping[str, Any]) -> PassageChunk:
    return PassageChunk(
        chunk_id=str(chunk_id),
        chunk_index=_int(payload.get("chunk_index")),
        text=_content(payload),
        payload=payload,
    )


def _qdrant_seed(client, corpus: str, chunk_id: str) -> Tuple[bool, Optional[PassageChunk]]:
    """Whether Qdrant holds the corpus, and the seed chunk when it holds the id."""
    if not client.collection_exists(corpus):
        return False, None
    point_id = _qdrant_point_id(chunk_id)
    if point_id is None:
        return True, None
    points = client.retrieve(
        collection_name=corpus, ids=[point_id], with_payload=True, with_vectors=False
    )
    for point in points or []:
        payload = point.payload or {}
        if _is_chunk(payload):
            return True, _chunk(point.id, payload)
    return True, None


def _meili_seed(backend, corpus: str, chunk_id: str) -> Tuple[bool, Optional[PassageChunk]]:
    """Whether the text backend holds the corpus, and the seed chunk when it holds the id."""
    try:
        backend.health()
    except ServiceDownError as e:
        raise ConnectionError("health check failed") from e
    if not backend.exists(corpus):
        return False, None
    # Not a document id MeiliSearch accepts: never sent, so it cannot reach
    # another route of the documents endpoint.
    if not _MEILI_ID.fullmatch(chunk_id):
        return True, None
    document = backend.get_document(corpus, chunk_id)
    if document is None:
        return True, None
    return True, _chunk(document.get("id", chunk_id), document)


def _version(payload: Mapping[str, Any]) -> Tuple[Any, ...]:
    return tuple(payload.get(key) or None for key in VERSION_KEYS)


def _qdrant_window(client, corpus: str, seed: PassageChunk, low: int, high: int):
    """The Qdrant points of the seed's file and version with ``chunk_index`` in range.

    Returns the candidates and whether the scroll stopped at its page bound.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

    must = [
        FieldCondition(key="file_path", match=MatchValue(value=seed.payload["file_path"])),
        FieldCondition(key="chunk_index", range=Range(gte=low, lte=high)),
    ]
    for key, value in zip(VERSION_KEYS, _version(seed.payload)):
        if value is not None:
            must.append(FieldCondition(key=key, match=MatchValue(value=value)))
    window_filter = Filter(must=must)

    candidates = []
    offset = None
    for _page in range(_QDRANT_SCROLL_PAGES):
        points, offset = client.scroll(
            collection_name=corpus,
            scroll_filter=window_filter,
            limit=_QDRANT_SCROLL_PAGE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        candidates.extend(_chunk(point.id, point.payload or {}) for point in points)
        if offset is None:
            break
    # A scroll stopped at its page bound (other versions of an unversioned
    # path can fill it) may have missed neighbours.
    return candidates, offset is not None


_MEILI_LIST_FIELDS = ["id", "file_path", "chunk_index", *VERSION_KEYS]


def _meili_window(backend, corpus: str, seed: PassageChunk, low: int, high: int):
    """The text backend's chunks of the seed's file and version with ``chunk_index`` in range.

    Lists the file's chunks (ids and ordinals only), then fetches the window's
    documents. Returns the candidates and whether the listing stopped short of
    the file (MeiliSearch's ``maxTotalHits``).
    """
    listing, total = backend.file_documents(corpus, seed.payload["file_path"], _MEILI_LIST_FIELDS)
    wanted: List[str] = []
    for hit in listing:
        index = _int(hit.get("chunk_index"))
        if (
            index is not None
            and low <= index <= high
            and _version(hit) == _version(seed.payload)
            and str(hit.get("id")) != seed.chunk_id
        ):
            wanted.append(str(hit.get("id")))

    candidates = []
    for document_id in wanted:
        document = backend.get_document(corpus, document_id)
        if document is not None:
            candidates.append(_chunk(document.get("id", document_id), document))
    return candidates, total > len(listing)


def _select(seed: PassageChunk, candidates: List[PassageChunk], low: int, high: int):
    """One chunk per ordinal in range, from the seed's file and version, seed first."""
    by_index: Dict[int, PassageChunk] = {seed.chunk_index: seed}
    for chunk in sorted(candidates, key=lambda c: c.chunk_id):
        if (
            chunk.chunk_index is None
            or not low <= chunk.chunk_index <= high
            or not _is_chunk(chunk.payload)
            or chunk.payload.get("file_path") != seed.payload.get("file_path")
            or _version(chunk.payload) != _version(seed.payload)
        ):
            continue
        by_index.setdefault(chunk.chunk_index, chunk)
    return [by_index[index] for index in sorted(by_index)]


def _check_arguments(chunk_id: str, before: int, after: int, max_chars: int) -> None:
    if not isinstance(chunk_id, str) or not chunk_id:
        raise InvalidArgumentError("chunk_id must be a non-empty chunk id")
    for name, value in (("before", before), ("after", after)):
        if not 0 <= value <= MAX_NEIGHBOURS:
            raise InvalidArgumentError(
                f"{name} must be between 0 and {MAX_NEIGHBOURS}, got {value}"
            )
    if max_chars < 0:
        raise InvalidArgumentError(f"max_chars must be at least 0, got {max_chars}")


def _not_found(corpus: str, chunk_id: str, corpus_found: bool, down: Dict[str, BaseException]):
    """The error for a chunk no reachable system holds."""
    if down:
        reasons = [f"{service} is unreachable: {error}" for service, error in down.items()]
        if len(down) == 1:
            reasons.append(f"chunk {chunk_id} is not in the other system")
        return ServiceDownError("; ".join(reasons))
    if not corpus_found:
        return UnknownCorpusError(f"Unknown corpus: {corpus}")
    return UnknownChunkError(
        f"No chunk {chunk_id} in corpus {corpus}: the id is unknown, or a resync "
        "replaced it; search again for a current id"
    )


def get_passage(
    corpus: str,
    chunk_id: str,
    *,
    before: int = DEFAULT_NEIGHBOURS,
    after: int = DEFAULT_NEIGHBOURS,
    max_chars: int = 0,
    qdrant_client=None,
    text_backend=None,
    meilisearch_enabled: Optional[bool] = None,
) -> PassageResult:
    """The chunk ``chunk_id`` of ``corpus`` with up to ``before``/``after`` neighbours.

    Args:
        corpus: The corpus that holds the chunk
        chunk_id: A hit's ``chunk_id``, from either search mode
        before: Neighbouring chunks before the hit, 0 to ``MAX_NEIGHBOURS``
        after: Neighbouring chunks after the hit, 0 to ``MAX_NEIGHBOURS``
        max_chars: Bound on the stitched text; zero is unlimited
        qdrant_client: Qdrant client (created from config when None)
        text_backend: The corpus's ``TextBackend`` (MeiliSearch from config
            when None)
        meilisearch_enabled: False never looks in MeiliSearch, so a chunk
            Qdrant lacks is not found (``meilisearch.enabled`` from config
            when None; arcaneum#183)

    Raises:
        InvalidArgumentError: a bound is out of range, or the id is empty
        UnknownCorpusError: neither system holds the corpus
        UnknownChunkError: neither system holds the chunk
        ServiceDownError: a system that might hold the chunk is unreachable
        ArcaneumError: any other lookup failure
    """
    _check_arguments(chunk_id, before, after, max_chars)

    down: Dict[str, BaseException] = {}
    corpus_found = False
    seed = None
    source = QDRANT

    try:
        if qdrant_client is None:
            from ..cli.utils import create_qdrant_client, resolve_config_path

            # A read-only lookup: read a legacy config in place, never migrate it.
            qdrant_client = create_qdrant_client(
                config_path=resolve_config_path(migrate=False), for_search=True
            )
        corpus_found, seed = _qdrant_seed(qdrant_client, corpus, chunk_id)
    except ArcaneumError:
        raise
    except Exception as e:
        if not is_connection_failure(e):
            raise ArcaneumError(f"Passage lookup failed in Qdrant: {e}") from e
        down["Qdrant"] = e
        qdrant_client = None

    if seed is None and meilisearch_enabled is None:
        from ..config import resolve_meili_settings

        meilisearch_enabled = resolve_meili_settings().enabled
    if seed is None and meilisearch_enabled:
        source = MEILISEARCH
        try:
            if text_backend is None:
                from ..cli.utils import create_meili_client
                from ..fulltext.backend import MeiliTextBackend

                # A read-only lookup: use an existing key, never generate one.
                text_backend = MeiliTextBackend(create_meili_client(generate_key=False))
            found, seed = _meili_seed(text_backend, corpus, chunk_id)
            corpus_found = corpus_found or found
        except ArcaneumError:
            raise
        except Exception as e:
            if not is_connection_failure(e):
                raise ArcaneumError(f"Passage lookup failed in MeiliSearch: {e}") from e
            down["MeiliSearch"] = e

    if seed is None:
        raise _not_found(corpus, chunk_id, corpus_found, down)

    partial = False
    chunks = [seed]
    file_path = seed.payload.get("file_path")
    if seed.chunk_index is not None and isinstance(file_path, str) and file_path:
        low = max(seed.chunk_index - before, 0)
        high = seed.chunk_index + after
        fetch = _qdrant_window if source == QDRANT else _meili_window
        client = qdrant_client if source == QDRANT else text_backend
        try:
            candidates, partial = fetch(client, corpus, seed, low, high)
        except ArcaneumError:
            raise
        except Exception as e:
            service = "Qdrant" if source == QDRANT else "MeiliSearch"
            if is_connection_failure(e):
                raise service_down(service, e) from e
            raise ArcaneumError(f"Passage lookup failed in {service}: {e}") from e
        chunks = _select(seed, candidates, low, high)
        # A listing cut short cannot tell a missing neighbour from the file's end.
        partial = partial and len(chunks) < high - low + 1

    text, cut = limit_content(stitch(chunks), max_chars)
    roots = read_source_roots(qdrant_client, corpus) if qdrant_client is not None else []
    return PassageResult(
        corpus=corpus,
        chunk_id=seed.chunk_id,
        cite=build_cite(corpus, seed.payload, roots),
        chunks=chunks,
        text=text,
        truncated=cut or partial,
        source=source,
    )

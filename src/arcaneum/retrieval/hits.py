"""The cite-ready search hit both search modes emit (RDR-026 Q2, Q4, Q12).

A hit is ``{corpus, chunk_id, locator, cite, score, snippet,
snippet_truncated[, metadata]}``. Its cite follows the RDR-026 Q4 grammar:

    cite     = corpus ":" locator
    locator  = path [ "#" fragment ]
    fragment = "page=" N | "L" A "-" B | slug

The path is relative to the longest ``source_roots`` entry that prefixes the
stored ``file_path`` (recorded by ``arc corpus sync``); without one it is the
absolute path with ``$HOME`` folded to ``~``. ``#``, ``%`` and space are
percent-encoded in the path, and nothing else is, so a cite holds no space and
splits at its first ``:`` and its only ``#``. The passage operation reuses this
builder for its own ``cite``.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..utils.search_output import limit_content

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# The snippet bound when the caller names no ``max_content_chars``; it is the
# 500 characters semantic JSON has always used.
DEFAULT_SNIPPET_CHARS = 500

SOURCE_ROOTS_FIELD = "source_roots"

_CORPUS = re.compile(r"[A-Za-z0-9_-]+")
_PAGE = re.compile(r"page=\d+")
_LINES = re.compile(r"L\d+-\d+")
_SLUG = re.compile(r"(?:[^\W_]|-)+")
_ENCODED = {"%": "%25", "#": "%23", " ": "%20"}
_DECODE = re.compile(r"%(25|23|20)")


@dataclass(frozen=True)
class Cite:
    """A parsed cite: its corpus, its locator, and the locator's two parts."""

    corpus: str
    locator: str
    path: str
    fragment: Optional[str]


def encode_path(path: str) -> str:
    """Percent-encode ``%``, ``#`` and space, and nothing else."""
    return "".join(_ENCODED.get(ch, ch) for ch in path)


def decode_path(path: str) -> str:
    """Reverse ``encode_path``."""
    return _DECODE.sub(lambda m: {"25": "%", "23": "#", "20": " "}[m.group(1)], path)


def _normalize(path: str) -> str:
    return os.path.expanduser(path).rstrip("/")


def _root_prefix(root: str) -> str:
    """``root`` as the prefix it strips: ``/``-terminated, ``/`` itself kept."""
    expanded = os.path.expanduser(root)
    return expanded.rstrip("/") + "/" if expanded else ""


def display_path(file_path: str, roots: Iterable[str] = ()) -> str:
    """The cite's path: root-relative, else absolute with ``$HOME`` as ``~``."""
    path = os.path.expanduser(file_path)
    for prefix in sorted((_root_prefix(r) for r in roots), key=len, reverse=True):
        if prefix and path.startswith(prefix):
            return path[len(prefix) :]
    home = _normalize("~")
    if home and (path == home or path.startswith(home + "/")):
        return "~" + path[len(home) :]
    return path


def heading_slug(section: Any) -> Optional[str]:
    """Slug of the deepest heading in a ``section`` (a ``" > "`` path or a list).

    Lowercased; letters, digits and ``-`` kept, whitespace turned into ``-``,
    everything else dropped. None when there is no heading or nothing is left.
    """
    if isinstance(section, (list, tuple)):
        headings = [str(h) for h in section if h]
    elif isinstance(section, str):
        headings = section.split(" > ")
    else:
        headings = []
    if not headings:
        return None
    heading = headings[-1].strip().lstrip("#").strip().lower()
    slug = "".join(
        "-" if ch.isspace() else ch for ch in heading if ch.isspace() or ch == "-" or ch.isalnum()
    )
    return slug or None


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fragment(payload: Mapping[str, Any]) -> Optional[str]:
    page = _int(payload.get("page_number"))
    if page is not None:
        return f"page={page}"
    start, end = _int(payload.get("start_line")), _int(payload.get("end_line"))
    if start is not None and end is not None:
        return f"L{start}-{end}"
    # ``arc corpus sync`` stores the heading trail as ``section``;
    # ``arc index markdown`` stores it as ``header_path``.
    section = payload.get("section")
    return heading_slug(section if section else payload.get("header_path"))


def build_locator(payload: Mapping[str, Any], roots: Iterable[str] = ()) -> str:
    """The locator of a stored chunk: its encoded path plus page, lines or heading.

    A code hit gets ``#L`` only when it carries both ``start_line`` and
    ``end_line``; corpus code chunks record neither, so their cite is the path
    alone (owner, 2026-11-16, on arcaneum#145).
    """
    file_path = payload.get("file_path") or payload.get("filename") or ""
    locator = encode_path(display_path(str(file_path), roots))
    fragment = _fragment(payload)
    return f"{locator}#{fragment}" if fragment else locator


def build_cite(corpus: str, payload: Mapping[str, Any], roots: Iterable[str] = ()) -> str:
    """``<corpus>:<locator>`` for a stored chunk."""
    return f"{corpus}:{build_locator(payload, roots)}"


def parse_cite(cite: str) -> Cite:
    """Split a cite into corpus, locator, decoded path and fragment.

    Raises:
        ValueError: the string is not a cite this grammar produces
    """
    corpus, sep, locator = cite.partition(":")
    if not sep or not _CORPUS.fullmatch(corpus):
        raise ValueError(f"Not a cite (no corpus): {cite!r}")
    path, hash_sep, fragment = locator.partition("#")
    # The builder escapes every space, so a raw one means a foreign string.
    # Other whitespace is left as is (RDR-026 Q4 escapes nothing else).
    if not path or " " in locator:
        raise ValueError(f"Not a cite (bad path): {cite!r}")
    if hash_sep and not (
        _PAGE.fullmatch(fragment)
        or _LINES.fullmatch(fragment)
        or (_SLUG.fullmatch(fragment) and fragment == fragment.lower())
    ):
        raise ValueError(f"Not a cite (bad fragment): {cite!r}")
    return Cite(
        corpus=corpus,
        locator=locator,
        path=decode_path(path),
        fragment=fragment if hash_sep else None,
    )


def roots_of(metadata: Mapping[str, Any]) -> List[str]:
    """The ``source_roots`` a corpus's metadata records."""
    roots = metadata.get(SOURCE_ROOTS_FIELD)
    if not isinstance(roots, list):
        return []
    return [root for root in roots if isinstance(root, str) and root]


def _is_missing_collection(exc: Exception) -> bool:
    """Whether a metadata read failed because Qdrant holds no such collection."""
    return getattr(exc, "status_code", None) == 404 or "not found" in str(exc).lower()


def read_source_roots(client, corpus: str) -> List[str]:
    """The corpus's recorded ``source_roots``; empty when absent or unreadable.

    A failed read (Qdrant down during a text search, say) is not an error: the
    cite falls back to the absolute path (RDR-026 Q4).
    """
    return read_corpora_source_roots(client, [corpus]).get(corpus, [])


def read_corpora_metadata(client, corpora: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Each corpus's collection metadata, read once; empty when absent or unreadable.

    A corpus Qdrant does not hold (a MeiliSearch-only index) reads as empty.
    Any other failed read ends the lookups: the rest of the corpora read as
    empty, so an unreachable Qdrant costs one timeout, not one per corpus.
    """
    from ..indexing.collection_metadata import _retrieve_collection_metadata

    metadata: Dict[str, Dict[str, Any]] = {}
    for corpus in corpora:
        if client is None:
            metadata[corpus] = {}
            continue
        try:
            metadata[corpus] = _retrieve_collection_metadata(client, corpus) or {}
        except Exception as e:
            logger.debug("No metadata for %s: %s", corpus, e)
            metadata[corpus] = {}
            if not _is_missing_collection(e):
                client = None
    return metadata


def read_corpora_source_roots(client, corpora: Iterable[str]) -> Dict[str, List[str]]:
    """Each corpus's ``source_roots``, as ``read_corpora_metadata`` reads them."""
    return {
        corpus: roots_of(metadata)
        for corpus, metadata in read_corpora_metadata(client, corpora).items()
    }


def _sanitize(text: str) -> str:
    """Replace control characters other than newline, return and tab with a space."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)


def build_hit(
    *,
    corpus: str,
    chunk_id: Any,
    payload: Mapping[str, Any],
    content: str,
    score: Optional[float],
    roots: Sequence[str] = (),
    max_content_chars: Optional[int] = None,
    verbose: bool = False,
    content_keys: Iterable[str] = ("content",),
) -> Dict[str, Any]:
    """One hit of the RDR-026 Q2 schema.

    Args:
        corpus: The corpus the hit came from
        chunk_id: Qdrant point id or MeiliSearch document id
        payload: The stored fields of the chunk
        content: The chunk text
        score: Similarity (semantic) or ``_rankingScore`` (text)
        roots: The corpus's ``source_roots``
        max_content_chars: Snippet bound; None uses ``DEFAULT_SNIPPET_CHARS``,
            zero is unlimited
        verbose: Add ``metadata``: every stored field except the content and
            the search engine's ``_``-prefixed fields
        content_keys: Payload keys holding the chunk text, left out of metadata
    """
    bound = DEFAULT_SNIPPET_CHARS if max_content_chars is None else max_content_chars
    snippet, truncated = limit_content(_sanitize(content or ""), bound)
    locator = build_locator(payload, roots)
    hit: Dict[str, Any] = {
        "corpus": corpus,
        "chunk_id": None if chunk_id is None else str(chunk_id),
        "locator": locator,
        "cite": f"{corpus}:{locator}",
        "score": score,
        "snippet": snippet,
        "snippet_truncated": truncated,
    }
    if verbose:
        skip = set(content_keys)
        hit["metadata"] = {
            key: value
            for key, value in payload.items()
            if key not in skip and not key.startswith("_")
        }
    return hit


def search_data(
    *,
    query: str,
    corpora: Sequence[str],
    missing_corpora: Sequence[str],
    limit: int,
    offset: int,
    hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """The search ``data`` both modes emit (RDR-026 Q2)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "query": query,
        "corpora": list(corpora),
        "missing_corpora": list(missing_corpora),
        "limit": limit,
        "offset": offset,
        "hits": hits,
    }

"""The text-backend seam of the retrieval core (RDR-027 Q1).

A ``TextBackend`` is one full-text store a corpus can use. Search, passage
lookup and listing hold a backend, never a store's client, so a corpus can
choose its backend. ``arcaneum.fulltext.backend.MeiliTextBackend`` is the
MeiliSearch implementation and ``arcaneum.retrieval.qdrant_text.QdrantTextBackend``
the Qdrant one.

Nothing here imports a store's SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from ..schema.document import DualIndexDocument
from ..search.filters import ParsedFilter

# A hit is the chunk's stored fields with ``id``, ``content`` (the chunk text)
# and, when the store ranks, ``_rankingScore``.
RawHit = Dict[str, Any]


@dataclass
class TextPage:
    """One corpus's hits for one query.

    A page carries no total: text search reports the hits it returns, as
    semantic search does (RDR-027 Q5).
    """

    hits: List[RawHit]


@dataclass
class Coverage:
    """What a corpus's text store holds (RDR-027 Q8).

    ``chunks`` counts the stored chunks, ``searchable_chunks`` those a query
    can match (every chunk but those flagged as yielding no token), and
    ``indexed`` the chunks the text index holds, None when the index is
    missing. Coverage is complete when ``indexed`` equals
    ``searchable_chunks``. ``files`` and ``searchable_files`` count both per
    ``file_path`` when asked for; ``flagged`` is a bounded sample of the
    zero-token chunks as ``(file_path, chunk_index)``.
    """

    chunks: int
    searchable_chunks: int
    files: Dict[str, int] = field(default_factory=dict)
    indexed: Optional[int] = None
    searchable_files: Dict[str, int] = field(default_factory=dict)
    flagged: List[Tuple[Optional[str], Optional[int]]] = field(default_factory=list)

    @property
    def zero_token_chunks(self) -> int:
        """Chunks whose text yields no token: a chunking finding, never a gap."""
        return self.chunks - self.searchable_chunks

    @property
    def status(self) -> str:
        """``synced``, ``coverage_gap`` or ``text_index_missing``."""
        if self.indexed is None:
            return "text_index_missing"
        return "synced" if self.indexed == self.searchable_chunks else "coverage_gap"

    @property
    def complete(self) -> bool:
        return self.status == "synced"

    def to_data(self) -> Dict[str, Any]:
        """The backend-neutral JSON the status commands emit."""
        return {
            "status": self.status,
            "chunks": self.chunks,
            "searchable_chunks": self.searchable_chunks,
            "text_chunks": self.indexed or 0,
            "zero_token_chunks": self.zero_token_chunks,
            "zero_token_sample": [
                {"file_path": path, "chunk_index": index} for path, index in self.flagged
            ],
        }


@runtime_checkable
class TextBackend(Protocol):
    """A full-text store for corpora.

    Reads raise ``ServiceDownError`` (``health``) or the store's own errors,
    which callers classify with ``retrieval.failures``; a filter the store
    refuses raises ``BadFilterError``.
    """

    name: str  # the corpus's ``text_backend`` value (RDR-027 Q2)
    service: str  # the store's name in messages

    def health(self) -> None:
        """Return when the store answers; raise ``ServiceDownError`` otherwise."""

    def exists(self, corpus: str) -> bool:
        """Whether the store holds ``corpus``."""

    def check_filter(self, filter: Optional[ParsedFilter]) -> None:
        """Raise ``BadFilterError`` for a filter this store cannot run, before any request."""

    def search(
        self,
        corpus: str,
        query: str,
        *,
        filter: Optional[ParsedFilter] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> TextPage:
        """Ranked hits for ``query`` in ``corpus``."""

    def get_document(self, corpus: str, chunk_id: str) -> Optional[Dict[str, Any]]:
        """One chunk's stored fields, or None when the corpus does not hold it."""

    def file_documents(
        self, corpus: str, file_path: str, fields: Sequence[str]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """The listed ``fields`` of a file's chunks, and how many chunks the file has.

        The listing may stop short of the total where the store caps it.
        """

    def has_file(self, corpus: str, file_path: str) -> bool:
        """Whether any chunk of the corpus has ``file_path``: a write's read-back."""

    def add(self, corpus: str, docs: Sequence[DualIndexDocument]) -> None:
        """Store chunks, replacing any with the same id."""

    def delete_files(self, corpus: str, file_paths: Sequence[str]) -> None:
        """Remove every chunk of each file."""

    def rename(self, corpus: str, moves: Mapping[Tuple[str, str], Sequence[str]]) -> None:
        """Move chunks to a new path: ``(old_path, new_path)`` to the moved chunk ids."""

    def ensure(self, corpus: str, corpus_type: str) -> None:
        """Create the corpus's index and settings when absent."""

    def drop(self, corpus: str) -> None:
        """Remove the corpus; a missing corpus is not an error."""

    def coverage(self, corpus: str, *, files: bool = True) -> Coverage:
        """The corpus's chunk counts, in total and, with ``files``, per file."""

    def corpora(self) -> Dict[str, int]:
        """Every corpus the store holds, with its chunk count."""

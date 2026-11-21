"""Qdrant as a ``TextBackend`` (RDR-027 Q3-Q5).

A Qdrant text corpus searches the chunk text its points already hold, through a
text index on the ``text`` payload: the word tokenizer, lowercase, ASCII
folding, phrase matching, BM25 scoring, the corpus type's MeiliSearch stop
words, no stemming, and ``memory: cold`` (Q3). A query's quoted phrases become
``MatchPhrase`` filters and its other words one ``MatchText`` filter, ordered
by a BM25 ``TextQuery``; when no chunk holds every term, the words' filter is
dropped and BM25 ranks any term, the phrases still required (Q4). Hits carry
the MeiliSearch document's fields (Q5), so both backends emit one hit.

Reads never write: only ``ensure`` creates the index, and corpus create calls
it. ``tokenize`` applies Q3's rules in Python, for highlighting and for the
zero-token flag of Q8.

qdrant_client is imported where it is used (RDR-026 Q11).
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from ..cli.errors import BadFilterError, ServiceDownError
from ..indexing.collection_metadata import (
    FILE_MANIFEST_PAYLOAD_KEY,
    FILE_MANIFEST_PAYLOAD_VALUE,
    metadata_exclusion_filter,
)
from ..schema.document import DualIndexDocument, qdrant_payload_to_meilisearch_doc, to_qdrant_point
from ..search.filters import (
    FilterError,
    ParsedFilter,
    check_filterable,
    semantic_filterable_fields,
    to_qdrant_filter,
)
from .text_backend import Coverage, TextPage

TEXT_FIELD = "text"
QDRANT_START_HINT = "Start with: docker compose -f deploy/docker-compose.yml up -d qdrant"

# Q8's flag on a chunk whose text yields no token (set by sync, arcaneum#174).
TEXT_TOKENS_FIELD = "text_tokens"

_WORD = re.compile(r"[^\W_]+")  # Qdrant's word tokenizer: runs of letters and digits
_PHRASE = re.compile(r'"([^"]*)"')
_ADD_BATCH_SIZE = 100
_SCROLL_PAGE_SIZE = 1000


# ---------------------------------------------------------------------------
# Q3's tokenizer
# ---------------------------------------------------------------------------


def stop_words(corpus_type: Optional[str]) -> List[str]:
    """The corpus type's MeiliSearch stop words, which its text index also drops."""
    from ..fulltext.indexes import get_index_settings

    if not corpus_type:
        return []
    try:
        return list(get_index_settings(corpus_type).get("stopWords") or [])
    except ValueError:
        return []


def _normalize(word: str) -> str:
    """Lowercase and fold accents to ASCII, keeping letters with no ASCII form."""
    decomposed = unicodedata.normalize("NFKD", word.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def token_spans(text: str) -> Iterator[Tuple[int, int, str]]:
    """Each word of ``text``: its start, its end, and its token before stop words."""
    for match in _WORD.finditer(text or ""):
        token = _normalize(match.group())
        if token:
            yield match.start(), match.end(), token


def tokenize(text: str, corpus_type: Optional[str]) -> List[str]:
    """The tokens a corpus's text index holds for ``text`` (Q3's rules)."""
    stops = set(stop_words(corpus_type))
    return [token for _start, _end, token in token_spans(text) if token not in stops]


def zero_token(text: str, corpus_type: Optional[str]) -> bool:
    """Whether ``text`` yields no token in a corpus's text index (Q8's flag)."""
    stops = set(stop_words(corpus_type))
    return all(token in stops for _start, _end, token in token_spans(text))


def text_index_params(corpus_type: Optional[str]):
    """Q3's text index on ``text`` for a corpus type."""
    from qdrant_client.http import models

    stops = stop_words(corpus_type)
    return models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        lowercase=True,
        ascii_folding=True,
        phrase_matching=True,
        stopwords=models.StopwordsSet(custom=stops) if stops else None,
        scoring=models.TextScoringParams(type=models.TextScoringType.BM25),
        memory=models.Memory.COLD,
    )


def split_query(query: str) -> Tuple[List[str], str]:
    """A query's quoted phrases, and its other words as one string.

    An unpaired quote is dropped, so it joins no phrase.
    """
    phrases = [" ".join(p.split()) for p in _PHRASE.findall(query or "") if p.strip()]
    rest = _PHRASE.sub(" ", query or "").replace('"', " ")
    return phrases, " ".join(rest.split())


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


def _has_text_index(info: Any) -> bool:
    entry = (getattr(info, "payload_schema", None) or {}).get(TEXT_FIELD)
    data_type = getattr(entry, "data_type", None)
    return getattr(data_type, "value", data_type) == "text"


def _point_id(chunk_id: Any) -> Optional[Any]:
    """``chunk_id`` as a Qdrant point id, or None when it cannot be one."""
    if isinstance(chunk_id, int) and not isinstance(chunk_id, bool):
        return chunk_id if chunk_id >= 0 else None
    text = str(chunk_id)
    if text.isdigit():
        return int(text)
    try:
        return str(uuid.UUID(text))
    except ValueError:
        return None


def _is_reserved(payload: Mapping[str, Any]) -> bool:
    from ..indexing.collection_metadata import METADATA_PAYLOAD_KEY

    return bool(payload.get(METADATA_PAYLOAD_KEY)) or (
        payload.get(FILE_MANIFEST_PAYLOAD_KEY) == FILE_MANIFEST_PAYLOAD_VALUE
    )


def _chunk_filter(query_filter=None):
    """``query_filter`` restricted to chunk points: no metadata point, no file manifest."""
    from qdrant_client.http import models

    combined = metadata_exclusion_filter(query_filter)
    manifest = models.FieldCondition(
        key=FILE_MANIFEST_PAYLOAD_KEY, match=models.MatchValue(value=FILE_MANIFEST_PAYLOAD_VALUE)
    )
    return combined.model_copy(update={"must_not": [*(combined.must_not or []), manifest]})


def _file_filter(file_paths: Sequence[str]):
    from qdrant_client.http import models

    return _chunk_filter(
        models.Filter(
            must=[
                models.FieldCondition(
                    key="file_path", match=models.MatchAny(any=list(file_paths))
                )
            ]
        )
    )


class QdrantTextBackend:
    """A ``TextBackend`` over one Qdrant server: a text index per corpus collection.

    A corpus is a Qdrant text corpus once its collection has the text index;
    the collection itself, and its vectors, belong to ``arc corpus``.

    Args:
        client: Qdrant client
        corpus_types: Each corpus's type, when the caller has read it; any
            other corpus's type is read from its metadata on first use
    """

    name = "qdrant"
    service = "Qdrant"

    def __init__(self, client, corpus_types: Optional[Mapping[str, Optional[str]]] = None):
        self.client = client
        self._corpus_types: Dict[str, Optional[str]] = dict(corpus_types or {})

    # -- the protocol --------------------------------------------------------
    def health(self) -> None:
        try:
            self.client.get_collections()
        except Exception as e:
            raise ServiceDownError(f"Qdrant server not available. {QDRANT_START_HINT}") from e

    def exists(self, corpus: str) -> bool:
        if not self.client.collection_exists(corpus):
            return False
        return _has_text_index(self.client.get_collection(corpus))

    def check_filter(self, filter: Optional[ParsedFilter]) -> None:
        self._qdrant_filter(filter)

    def search(
        self,
        corpus: str,
        query: str,
        *,
        filter: Optional[ParsedFilter] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> TextPage:
        from qdrant_client.http import models

        user_filter = self._qdrant_filter(filter)
        corpus_type = self._corpus_type(corpus)
        fields = semantic_filterable_fields(corpus_type)
        if fields is not None:
            try:
                check_filterable(filter, fields, corpus)
            except FilterError as e:
                raise BadFilterError(str(e)) from e

        phrases, words = split_query(query)
        if not phrases and not words:
            # MeiliSearch's placeholder search: every chunk, in id order.
            return self._page(self._query(corpus, None, [], user_filter, limit, offset))

        ranking = models.TextQuery(text=" ".join((query or "").replace('"', " ").split()))
        phrase_conditions = [
            models.FieldCondition(key=TEXT_FIELD, match=models.MatchPhrase(phrase=phrase))
            for phrase in phrases
        ]
        conditions = list(phrase_conditions)
        if tokenize(words, corpus_type):
            conditions.append(
                models.FieldCondition(key=TEXT_FIELD, match=models.MatchText(text=words))
            )
        points = self._query(corpus, ranking, conditions, user_filter, limit, offset)
        if not points and len(conditions) > len(phrase_conditions):
            # Fall back to BM25 over any term only when no chunk holds every
            # term, not when this page is past the last strict hit.
            if offset == 0 or not self._query(corpus, ranking, conditions, user_filter, 1, 0):
                points = self._query(corpus, ranking, phrase_conditions, user_filter, limit, offset)
        return self._page(points)

    def get_document(self, corpus: str, chunk_id: str) -> Optional[Dict[str, Any]]:
        point_id = _point_id(chunk_id)
        if point_id is None:
            return None
        points = self.client.retrieve(
            collection_name=corpus, ids=[point_id], with_payload=True, with_vectors=False
        )
        if not points or _is_reserved(points[0].payload or {}):
            return None
        return qdrant_payload_to_meilisearch_doc(points[0].id, points[0].payload or {})

    def file_documents(
        self, corpus: str, file_path: str, fields: Sequence[str]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Every chunk of the file, so the listing always reaches the total."""
        wanted = set(fields)
        documents = [
            {k: v for k, v in qdrant_payload_to_meilisearch_doc(p.id, p.payload or {}).items() if k in wanted}
            for p in self._scroll(corpus, _file_filter([file_path]), with_payload=True)
        ]
        return documents, len(documents)

    def has_file(self, corpus: str, file_path: str) -> bool:
        points, _next = self.client.scroll(
            collection_name=corpus,
            scroll_filter=_file_filter([file_path]),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(points)

    def add(self, corpus: str, docs: Sequence[DualIndexDocument]) -> None:
        """Upsert each chunk's point, vectors and all, as ``DualIndexer`` writes it.

        A Qdrant corpus's text lives in its chunk points, so sync writes them
        once, through ``DualIndexer``, and never calls this as well (Q7). A
        document without vectors raises ``ValueError`` rather than replace a
        point's vectors with none.
        """
        points = [to_qdrant_point(doc) for doc in docs]
        for i in range(0, len(points), _ADD_BATCH_SIZE):
            self.client.upsert(
                collection_name=corpus, points=points[i : i + _ADD_BATCH_SIZE], wait=True
            )

    def delete_files(self, corpus: str, file_paths: Sequence[str]) -> None:
        """Delete the files' chunk points; their file manifests are sync's to manage."""
        from qdrant_client.http import models

        if not file_paths:
            return
        self.client.delete(
            collection_name=corpus,
            points_selector=models.FilterSelector(filter=_file_filter(file_paths)),
            wait=True,
        )

    def rename(self, corpus: str, moves: Mapping[Tuple[str, str], Sequence[str]]) -> None:
        """Set ``file_path`` and ``filename`` on every moved chunk; the text is untouched."""
        for (_old_path, new_path), chunk_ids in moves.items():
            ids = [pid for pid in (_point_id(c) for c in chunk_ids) if pid is not None]
            if ids:
                self.client.set_payload(
                    collection_name=corpus,
                    payload={"file_path": new_path, "filename": Path(new_path).name},
                    points=ids,
                    wait=True,
                )

    def ensure(self, corpus: str, corpus_type: str) -> None:
        """Create Q3's text index, and Q8's ``text_tokens`` bool index, when missing."""
        from qdrant_client.http import models

        self._corpus_types[corpus] = corpus_type
        info = self.client.get_collection(corpus)
        wanted = []
        if not _has_text_index(info):
            wanted.append((TEXT_FIELD, text_index_params(corpus_type)))
        if TEXT_TOKENS_FIELD not in (getattr(info, "payload_schema", None) or {}):
            wanted.append((TEXT_TOKENS_FIELD, models.PayloadSchemaType.BOOL))
        for field_name, field_schema in wanted:
            try:
                self.client.create_payload_index(
                    collection_name=corpus,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=True,
                )
            except Exception as exc:
                # Another process may create it between the read and the create.
                message = str(exc).lower()
                if "already exists" not in message and "409" not in message:
                    raise

    def drop(self, corpus: str) -> None:
        """Drop the text index; the collection and its vectors stay."""
        if self.exists(corpus):
            self.client.delete_payload_index(
                collection_name=corpus, field_name=TEXT_FIELD, wait=True
            )

    def coverage(self, corpus: str) -> Coverage:
        """Chunk points, those not flagged as yielding no token (Q8), and chunks per file."""
        from qdrant_client.http import models

        files: Counter = Counter()
        for point in self._scroll(corpus, _chunk_filter(), with_payload=["file_path"]):
            files[(point.payload or {}).get("file_path")] += 1
        tokenless = models.Filter(
            must=[
                models.FieldCondition(
                    key=TEXT_TOKENS_FIELD, match=models.MatchValue(value=False)
                )
            ]
        )
        flagged = self.client.count(
            collection_name=corpus, count_filter=_chunk_filter(tokenless), exact=True
        ).count
        chunks = sum(files.values())
        return Coverage(chunks=chunks, searchable_chunks=chunks - flagged, files=dict(files))

    def corpora(self) -> Dict[str, int]:
        """Each collection with the text index, with its chunk count."""
        counts: Dict[str, int] = {}
        for collection in self.client.get_collections().collections:
            if _has_text_index(self.client.get_collection(collection.name)):
                counts[collection.name] = self.client.count(
                    collection_name=collection.name, count_filter=_chunk_filter(), exact=True
                ).count
        return counts

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _qdrant_filter(filter: Optional[ParsedFilter]):
        try:
            return to_qdrant_filter(filter)
        except FilterError as e:
            raise BadFilterError(f"Invalid filter: {e}") from e

    def _corpus_type(self, corpus: str) -> Optional[str]:
        if corpus not in self._corpus_types:
            from ..indexing.collection_metadata import _retrieve_collection_metadata

            try:
                metadata = _retrieve_collection_metadata(self.client, corpus) or {}
            except Exception:
                metadata = {}
            self._corpus_types[corpus] = metadata.get("collection_type")
        return self._corpus_types[corpus]

    def _query(self, corpus, ranking, conditions, user_filter, limit, offset) -> list:
        from qdrant_client.http import models

        must = [*conditions, *([user_filter] if user_filter is not None else [])]
        response = self.client.query_points(
            collection_name=corpus,
            query=ranking,
            using=TEXT_FIELD if ranking is not None else None,
            query_filter=_chunk_filter(models.Filter(must=must) if must else None),
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        return list(response.points)

    @staticmethod
    def _page(points) -> TextPage:
        return TextPage(
            hits=[
                {
                    **qdrant_payload_to_meilisearch_doc(p.id, p.payload or {}),
                    "_rankingScore": float(p.score),
                }
                for p in points
            ]
        )

    def _scroll(self, corpus: str, scroll_filter, with_payload) -> Iterator[Any]:
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=corpus,
                scroll_filter=scroll_filter,
                limit=_SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=with_payload,
                with_vectors=False,
            )
            yield from points
            if offset is None:
                return

"""MeiliSearch as a ``TextBackend`` (RDR-027 Q1).

``MeiliTextBackend`` wraps a ``FullTextClient``. Beside the protocol's
operations it carries the MeiliSearch-only ones that repair and parity need
(a document scan, partial updates, queued uploads), so no module outside this
package reaches the MeiliSearch SDK.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from ..cli.errors import BadFilterError, ServiceDownError
from ..retrieval.text_backend import Coverage, TextPage
from ..schema.document import DualIndexDocument, to_meilisearch_doc
from ..search.filters import (
    ParsedFilter,
    to_meilisearch_filter,
    unfilterable_field_message,
)
from .indexes import get_index_settings

MEILI_START_HINT = "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"

# Partial updates of a rename can touch every chunk of many files.
MEILI_RENAME_UPDATE_TIMEOUT_MS = 300000

_ADD_BATCH_SIZE = 1000
_FILE_PAGE_SIZE = 1000


def connection_error_types() -> Tuple[type[BaseException], ...]:
    """The SDK errors that mean MeiliSearch is unreachable."""
    from meilisearch.errors import MeilisearchCommunicationError

    return (MeilisearchCommunicationError,)


def _is_filter_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        getattr(exc, "code", None) == "invalid_search_filter"
        or "invalid_search_filter" in message
        or "is not filterable" in message
    )


def _as_dict(document: Any) -> Dict[str, Any]:
    return document if isinstance(document, dict) else dict(vars(document))


def _status(result: Any, key: str) -> Any:
    return getattr(result, key, None) or (result.get(key) if isinstance(result, dict) else None)


class MeiliTextBackend:
    """A ``TextBackend`` over one MeiliSearch server, an index per corpus."""

    name = "meili"
    service = "MeiliSearch"

    def __init__(self, client):
        self.client = client

    # -- the protocol --------------------------------------------------------
    def health(self) -> None:
        if not self.client.health_check():
            raise ServiceDownError(f"MeiliSearch server not available. {MEILI_START_HINT}")

    def exists(self, corpus: str) -> bool:
        return self.client.index_exists(corpus)

    def check_filter(self, filter: Optional[ParsedFilter]) -> None:
        self._native_filter(filter)

    def search(
        self,
        corpus: str,
        query: str,
        *,
        filter: Optional[ParsedFilter] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> TextPage:
        native = self._native_filter(filter)
        try:
            results = self.client.search(
                corpus,
                query,
                filter=native,
                limit=limit,
                offset=offset,
                attributes_to_highlight=["content"],
                show_ranking_score=True,
            )
        except Exception as e:
            if native and _is_filter_error(e):
                raise self._bad_filter(corpus, e) from e
            raise
        return TextPage(hits=results.get("hits", []))

    def get_document(self, corpus: str, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self.client.get_document(corpus, chunk_id)

    def file_documents(
        self, corpus: str, file_path: str, fields: Sequence[str]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Pages a placeholder search filtered on ``file_path``.

        The total is MeiliSearch's estimate, which exceeds the listing when
        paging stops at ``maxTotalHits``.
        """
        from ..search.filters import _meili_value

        path_filter = f"file_path = {_meili_value(file_path)}"
        documents: List[Dict[str, Any]] = []
        total = 0
        while True:
            response = self.client.search(
                corpus,
                "",
                filter=path_filter,
                limit=_FILE_PAGE_SIZE,
                offset=len(documents),
                attributes_to_retrieve=list(fields),
            )
            hits = response.get("hits", [])
            total = response.get("estimatedTotalHits", response.get("totalHits", 0)) or 0
            documents.extend(hits)
            if len(hits) < _FILE_PAGE_SIZE:
                break
        return documents, total

    def has_file(self, corpus: str, file_path: str) -> bool:
        escaped_path = file_path.replace("\\", "\\\\").replace('"', '\\"')
        result = self.client.search(
            index_name=corpus,
            query="",
            filter=f'file_path = "{escaped_path}"',
            limit=1,
        )
        return bool(result.get("hits", []))

    def add(self, corpus: str, docs: Sequence[DualIndexDocument]) -> None:
        meili_docs = [to_meilisearch_doc(doc) for doc in docs]
        batches = [
            meili_docs[i : i + _ADD_BATCH_SIZE] for i in range(0, len(meili_docs), _ADD_BATCH_SIZE)
        ]
        self.client.add_documents_batch_parallel(corpus, batches)

    def delete_files(self, corpus: str, file_paths: Sequence[str]) -> None:
        self.client.delete_documents_by_file_paths(corpus, list(file_paths))

    def rename(self, corpus: str, moves: Mapping[Tuple[str, str], Sequence[str]]) -> None:
        """One partial update of ``file_path`` and ``filename`` for every moved chunk."""
        from pathlib import Path

        updates = [
            {"id": chunk_id, "file_path": new_path, "filename": Path(new_path).name}
            for (_old_path, new_path), chunk_ids in moves.items()
            for chunk_id in chunk_ids
        ]
        if updates:
            self.update_documents(corpus, updates, timeout_ms=MEILI_RENAME_UPDATE_TIMEOUT_MS)

    def ensure(self, corpus: str, corpus_type: str) -> None:
        if not self.client.index_exists(corpus):
            self.client.create_index(
                corpus, primary_key="id", settings=get_index_settings(corpus_type)
            )

    def drop(self, corpus: str) -> None:
        if self.client.index_exists(corpus):
            self.client.delete_index(corpus)

    def coverage(self, corpus: str, *, files: bool = True) -> Coverage:
        """Every document is searchable and indexed: MeiliSearch has no separate text index."""
        chunks = self.client.get_index_stats(corpus).get("numberOfDocuments", 0)
        per_file = self.client.get_chunk_counts_by_file(corpus) if files else {}
        return Coverage(
            chunks=chunks,
            searchable_chunks=chunks,
            files=per_file,
            indexed=chunks,
            searchable_files=dict(per_file),
        )

    def corpora(self) -> Dict[str, int]:
        """Chunk counts from the global stats, else per index; an unreadable index counts 0."""
        try:
            all_stats = self.client.get_stats().get("indexes", {})
        except Exception:
            all_stats = None
        counts: Dict[str, int] = {}
        for idx in self.client.list_indexes():
            name = idx["uid"]
            try:
                stats = all_stats.get(name, {}) if all_stats is not None else None
                if stats is None:
                    stats = self.client.get_index_stats(name)
                counts[name] = stats.get("numberOfDocuments", 0)
            except Exception:
                counts[name] = 0
        return counts

    # -- MeiliSearch only ----------------------------------------------------
    def filterable_attributes(self, corpus: str) -> List[str]:
        settings = self.client.get_index_settings(corpus)
        return list(settings.get("filterableAttributes") or [])

    def iter_documents(
        self, corpus: str, fields: Sequence[str], batch_size: int = 1000
    ) -> Iterator[Dict[str, Any]]:
        """Every document's ``fields``, through the documents endpoint.

        Unlike search, the documents endpoint is not capped by
        ``maxTotalHits``. Fields a document lacks are absent from its dict.
        """
        index = self.client.get_index(corpus)
        offset = 0
        while True:
            result = index.get_documents(
                {"offset": offset, "limit": batch_size, "fields": list(fields)}
            )
            docs = result.results if hasattr(result, "results") else result.get("results", [])
            if not docs:
                return
            for doc in docs:
                yield _as_dict(doc)
            offset += len(docs)
            if len(docs) < batch_size:
                return

    def update_documents(
        self, corpus: str, documents: Sequence[Mapping[str, Any]], timeout_ms: Optional[int] = None
    ) -> None:
        """Merge fields into existing documents by id, waiting for the task.

        With no ``timeout_ms`` the SDK's default wait applies.
        """
        task = self.client.get_index(corpus).update_documents(list(documents))
        if timeout_ms is None:
            self.client.client.wait_for_task(task.task_uid)
        else:
            self.client.client.wait_for_task(task.task_uid, timeout_in_ms=timeout_ms)

    def clear(self, corpus: str, timeout_ms: int = 300000) -> None:
        """Delete every document of the index, keeping the index and its settings."""
        task = self.client.get_index(corpus).delete_all_documents()
        status, error = self.wait_for_task(task.task_uid, timeout_ms)
        if status != "succeeded":
            raise RuntimeError(f"Clearing MeiliSearch index '{corpus}' {status}: {error}")

    def enqueue_documents(self, corpus: str, documents: Sequence[Mapping[str, Any]]) -> int:
        """Queue an upload without waiting; returns its task uid."""
        return self.client.get_index(corpus).add_documents(list(documents)).task_uid

    def wait_for_task(self, task_uid: int, timeout_ms: int) -> Tuple[Any, Any]:
        """Wait for a queued task; returns its status and error."""
        result = self.client.client.wait_for_task(task_uid, timeout_in_ms=timeout_ms)
        return _status(result, "status"), _status(result, "error")

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _native_filter(filter: Optional[ParsedFilter]) -> Optional[str]:
        try:
            return to_meilisearch_filter(filter)
        except ValueError as e:
            raise BadFilterError(f"Invalid filter: {e}") from e

    def _bad_filter(self, corpus: str, exc: Exception) -> BadFilterError:
        """Name the rejected attribute and list the index's filterable ones."""
        attr_match = re.search(r"Attribute `([^`]+)`", str(exc))
        bad_attr = attr_match.group(1) if attr_match else None

        try:
            filterable = self.filterable_attributes(corpus)
        except Exception:
            filterable = None

        if bad_attr and filterable is not None:
            return BadFilterError(unfilterable_field_message(bad_attr, filterable, corpus))

        if bad_attr:
            lines = [f"Filter attribute '{bad_attr}' is not filterable."]
        else:
            lines = [f"Invalid filter: {getattr(exc, 'message', None) or exc}"]
        if filterable:
            lines.append(f"Available filterable attributes for index '{corpus}':")
            lines.extend(f"  - {attr}" for attr in filterable)
        elif filterable is not None:
            lines.append(
                f"Index '{corpus}' has no filterable attributes configured; "
                "filters cannot be used with this index."
            )
        return BadFilterError("\n".join(lines))

"""Text-backend fixtures (RDR-027 Q1).

``text_backend`` is parametrized by backend name, so a later backend joins the
contract tests by adding its name to ``TEXT_BACKENDS`` and a branch to
``make_text_backend``.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

TEXT_BACKENDS = ["meilisearch"]


class FakeFullTextClient:
    """An in-memory ``FullTextClient``: the calls ``MeiliTextBackend`` makes.

    Search matches every query word as a case-insensitive substring of
    ``content`` and filters ``field = "value"`` on the index's filterable
    attributes, refusing any other field as MeiliSearch does.
    """

    def __init__(self, healthy: bool = True):
        self.healthy = healthy
        self.indexes: dict[str, dict] = {}  # name -> {"settings": ..., "docs": {id: doc}}
        self.client = SimpleNamespace(wait_for_task=self._wait_for_task)
        self._tasks: dict[int, str] = {}

    # -- service -------------------------------------------------------------
    def health_check(self) -> bool:
        return self.healthy

    def _wait_for_task(self, task_uid, timeout_in_ms=None):
        return SimpleNamespace(status=self._tasks.get(task_uid, "succeeded"), error=None)

    def _task(self) -> SimpleNamespace:
        uid = len(self._tasks) + 1
        self._tasks[uid] = "succeeded"
        return SimpleNamespace(task_uid=uid)

    # -- indexes -------------------------------------------------------------
    def index_exists(self, name: str) -> bool:
        return name in self.indexes

    def create_index(self, name, primary_key="id", settings=None):
        self.indexes[name] = {"settings": dict(settings or {}), "docs": {}}

    def delete_index(self, name):
        self.indexes.pop(name, None)

    def list_indexes(self):
        return [{"uid": name} for name in self.indexes]

    def get_stats(self):
        return {
            "indexes": {
                name: {"numberOfDocuments": len(index["docs"])}
                for name, index in self.indexes.items()
            }
        }

    def get_index_stats(self, name):
        return {"numberOfDocuments": len(self.indexes[name]["docs"])}

    def get_index_settings(self, name):
        return dict(self.indexes[name]["settings"])

    def get_index(self, name):
        docs = self.indexes[name]["docs"]

        def update_documents(updates):
            for update in updates:
                docs.setdefault(update["id"], {}).update(update)
            return self._task()

        def get_documents(params):
            fields = params.get("fields")
            ordered = list(docs.values())[params["offset"] : params["offset"] + params["limit"]]
            if fields:
                ordered = [{k: v for k, v in d.items() if k in fields} for d in ordered]
            return SimpleNamespace(results=[SimpleNamespace(**d) for d in ordered])

        def add_documents(batch, primary_key=None):
            for doc in batch:
                docs[doc["id"]] = dict(doc)
            return self._task()

        return SimpleNamespace(
            update_documents=update_documents,
            get_documents=get_documents,
            add_documents=add_documents,
        )

    # -- documents -----------------------------------------------------------
    def add_documents_batch_parallel(self, name, batches, primary_key=None, timeout_ms=None):
        docs = self.indexes[name]["docs"]
        total = 0
        for batch in batches:
            for doc in batch:
                docs[doc["id"]] = dict(doc)
                total += 1
        return {"total_documents": total, "task_count": len(batches)}

    def delete_documents_by_file_paths(self, name, file_paths, timeout_ms=None):
        docs = self.indexes[name]["docs"]
        for doc_id in [i for i, d in docs.items() if d.get("file_path") in set(file_paths)]:
            del docs[doc_id]
        return {"deleted_count": len(file_paths), "task_uids": []}

    def get_document(self, name, document_id):
        doc = self.indexes.get(name, {}).get("docs", {}).get(document_id)
        return dict(doc) if doc is not None else None

    def get_chunk_counts_by_file(self, name):
        counts: dict[str, int] = {}
        for doc in self.indexes[name]["docs"].values():
            counts[doc["file_path"]] = counts.get(doc["file_path"], 0) + 1
        return counts

    def search(
        self,
        index_name,
        query,
        filter=None,
        limit=20,
        offset=0,
        attributes_to_highlight=None,
        show_ranking_score=False,
        attributes_to_retrieve=None,
    ):
        name = index_name
        docs = list(self.indexes[name]["docs"].values())
        for word in query.lower().split():
            docs = [d for d in docs if word.strip('"') in str(d.get("content", "")).lower()]
        if filter:
            filterable = self.indexes[name]["settings"].get("filterableAttributes", [])
            match = re.fullmatch(r'(\w+) = "((?:[^"\\]|\\.)*)"', filter)
            if match is None or match.group(1) not in filterable:
                error = RuntimeError(
                    f"invalid_search_filter: Attribute `{filter.split()[0]}` is not filterable."
                )
                error.code = "invalid_search_filter"
                raise error
            docs = [d for d in docs if str(d.get(match.group(1))) == match.group(2)]
        hits = []
        for rank, doc in enumerate(docs[offset : offset + limit]):
            hit = dict(doc)
            if attributes_to_retrieve:
                hit = {k: v for k, v in hit.items() if k in attributes_to_retrieve}
            if show_ranking_score:
                hit["_rankingScore"] = 1.0 / (rank + 1)
            hits.append(hit)
        return {"hits": hits, "estimatedTotalHits": len(docs), "processingTimeMs": 1}


def make_text_backend(name: str, healthy: bool = True):
    """A text backend over an in-memory store, up or down."""
    if name == "meilisearch":
        from arcaneum.fulltext.backend import MeiliTextBackend

        return MeiliTextBackend(FakeFullTextClient(healthy=healthy))
    raise ValueError(f"unknown text backend {name!r}")


@pytest.fixture(params=TEXT_BACKENDS)
def text_backend_name(request) -> str:
    return request.param


@pytest.fixture
def text_backend(text_backend_name):
    return make_text_backend(text_backend_name)


@pytest.fixture
def down_text_backend(text_backend_name):
    return make_text_backend(text_backend_name, healthy=False)

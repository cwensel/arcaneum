"""Text-backend fixtures (RDR-027 Q1).

``text_backend`` is parametrized by backend name, so a later backend joins the
contract tests by adding its name to ``TEXT_BACKENDS`` and a branch to
``make_text_backend``.

``LocalQdrant`` stands in for a Qdrant server; ``tests/integration`` runs the
Qdrant backend against a real 1.20 server.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

TEXT_BACKENDS = ["meilisearch", "qdrant"]

# The contract's corpus. A Qdrant text corpus is a collection `arc corpus
# create` made, which the backend's ensure() then indexes, so the Qdrant
# fixture creates it first.
CONTRACT_CORPORA = ("Docs",)
VECTOR_NAME = "stub"  # the contract chunks' one named vector


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


class LocalQdrant:
    """qdrant-client's local mode, reporting the payload indexes asked of it.

    Local mode builds no payload index and reports none, and runs a text query
    only over a text index with the default tokenization. This records each
    requested index and reports it in ``get_collection``'s ``payload_schema``,
    as the server does, and hands local mode a default-tokenized, scored text
    index in place of a requested text index, so text queries run. Local mode
    therefore ignores the stop words and ASCII folding of RDR-027 Q3; the
    requested parameters stay in ``requested`` for tests to inspect.
    ``index_calls`` records every index creation, and ``query_calls`` every
    ``query_points`` argument set.
    """

    def __init__(self, healthy: bool = True):
        from qdrant_client import QdrantClient

        self._client = QdrantClient(location=":memory:")
        self.healthy = healthy
        self.requested: dict[tuple[str, str], object] = {}
        self.index_calls: list[tuple[str, str]] = []
        self.query_calls: list[dict] = []

    def __getattr__(self, name):
        return getattr(self._client, name)

    def get_collections(self):
        if not self.healthy:
            raise ConnectionError("connection refused")
        return self._client.get_collections()

    def create_collection_with_vector(self, name: str) -> None:
        from qdrant_client.http import models

        self._client.create_collection(
            name,
            vectors_config={
                VECTOR_NAME: models.VectorParams(size=2, distance=models.Distance.COSINE)
            },
        )

    def create_payload_index(self, collection_name, field_name, field_schema=None, **kwargs):
        import warnings

        from qdrant_client.http import models

        self.index_calls.append((collection_name, field_name))
        self.requested[(collection_name, field_name)] = field_schema
        if not isinstance(field_schema, models.TextIndexParams):
            return None  # local mode builds no other index
        local_schema = models.TextIndexParams(
            type=models.TextIndexType.TEXT, scoring=field_schema.scoring
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return self._client.create_payload_index(
                collection_name, field_name, field_schema=local_schema, **kwargs
            )

    def delete_payload_index(self, collection_name, field_name, **kwargs):
        if self.requested.pop((collection_name, field_name), None) is None:
            return None
        return self._client.delete_payload_index(collection_name, field_name, **kwargs)

    def get_collection(self, collection_name):
        from qdrant_client.http import models

        info = self._client.get_collection(collection_name)
        schema = {}
        for (collection, field_name), params in self.requested.items():
            if collection != collection_name:
                continue
            if isinstance(params, models.TextIndexParams):
                schema[field_name] = models.PayloadIndexInfo(
                    data_type=models.PayloadSchemaType.TEXT, params=params, points=0
                )
            else:
                schema[field_name] = models.PayloadIndexInfo(data_type=params, points=0)
        return info.model_copy(update={"payload_schema": schema})

    def query_points(self, collection_name, **kwargs):
        self.query_calls.append(kwargs)
        return self._client.query_points(collection_name, **kwargs)


def make_text_backend(name: str, healthy: bool = True):
    """A text backend over an in-memory store, up or down."""
    if name == "meilisearch":
        from arcaneum.fulltext.backend import MeiliTextBackend

        return MeiliTextBackend(FakeFullTextClient(healthy=healthy))
    if name == "qdrant":
        from arcaneum.retrieval.qdrant_text import QdrantTextBackend

        client = LocalQdrant(healthy=healthy)
        for corpus in CONTRACT_CORPORA:
            client.create_collection_with_vector(corpus)
        return QdrantTextBackend(client)
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

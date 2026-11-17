"""The print-free retrieval core shared by the CLI and the MCP server (RDR-026).

Each operation returns a result dataclass whose ``to_data()`` is the JSON
``data`` both front ends emit, and fails with an ``ArcaneumError`` carrying a
stable ``code``. None of them writes to stdout, and listing writes nothing to
Qdrant.
"""

import json
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from meilisearch.errors import MeilisearchApiError, MeilisearchCommunicationError
from qdrant_client.http.exceptions import ResponseHandlingException

from arcaneum.cli import errors
from arcaneum.search.searcher import SearchResult


def _hit(score, corpus, path, point_id):
    return SearchResult(
        score=score,
        collection=corpus,
        location=path,
        content=f"content of {path}",
        metadata={"file_path": path, "content": f"content of {path}"},
        point_id=point_id,
    )


def _meili_filter_error(attribute="filename"):
    body = {
        "message": f"Attribute `{attribute}` is not filterable.",
        "code": "invalid_search_filter",
        "type": "invalid_request",
        "link": "https://docs.meilisearch.com/errors#invalid_search_filter",
    }
    return MeilisearchApiError(
        "bad filter", SimpleNamespace(status_code=400, text=json.dumps(body))
    )


class TestErrorCodes:
    """Q3: every failure carries a stable code and its mapped exit code."""

    @pytest.mark.parametrize(
        "cls, code, exit_code",
        [
            (errors.UnknownCorpusError, "unknown-corpus", 3),
            (errors.UnknownChunkError, "unknown-chunk", 3),
            (errors.BadFilterError, "bad-filter", 2),
            (errors.InvalidArgumentError, "bad-argument", 2),
            (errors.ServiceDownError, "service-down", 1),
            (errors.SemanticUnavailableError, "semantic-unavailable", 1),
            (errors.SearchSlotUnavailable, "semantic-unavailable", 1),
            (errors.OperationTimeoutError, "timeout", 1),
        ],
    )
    def test_code_and_exit(self, cls, code, exit_code):
        error = cls("boom")

        assert error.code == code
        assert error.exit_code == exit_code
        assert error.to_error() == {"code": code, "message": "boom"}

    def test_not_found_errors_stay_resource_not_found(self):
        assert issubclass(errors.UnknownCorpusError, errors.ResourceNotFoundError)
        assert issubclass(errors.BadFilterError, errors.InvalidArgumentError)


# ---------------------------------------------------------------------------
# search_semantic
# ---------------------------------------------------------------------------


@pytest.fixture
def semantic(monkeypatch):
    from arcaneum.retrieval import semantic as module

    client = MagicMock()
    client.collection_exists.side_effect = lambda name: name != "Gone"
    hits = {
        "A": [_hit(0.9, "A", "/a/1.pdf", "a1"), _hit(0.5, "A", "/a/2.pdf", "a2")],
        "B": [_hit(0.7, "B", "/b/1.pdf", "b1")],
    }
    calls = []

    def search_collection(**kwargs):
        calls.append(kwargs)
        return hits[kwargs["collection_name"]]

    monkeypatch.setattr(module, "search_collection", search_collection)
    monkeypatch.setattr(module, "acquire_embedder_slot", nullcontext)
    return SimpleNamespace(module=module, client=client, calls=calls, embedder=MagicMock())


class TestSearchSemantic:
    def test_hits_merge_by_score_and_paginate(self, semantic):
        result = semantic.module.search_semantic(
            "query",
            ["A", "B"],
            limit=2,
            offset=1,
            client=semantic.client,
            embedder=semantic.embedder,
        )

        assert [hit.point_id for hit in result.results] == ["b1", "a2"]
        assert result.missing_corpora == []
        data = result.to_data()
        assert data["query"] == "query"
        assert data["collection"] == "[A, B]"
        assert data["missing_corpora"] == []
        assert [hit["point_id"] for hit in data["results"]] == ["b1", "a2"]

    def test_missing_corpus_in_multi_search_is_reported_not_fatal(self, semantic):
        result = semantic.module.search_semantic(
            "query", ["A", "Gone"], client=semantic.client, embedder=semantic.embedder
        )

        assert result.missing_corpora == ["Gone"]
        assert result.to_data()["missing_corpora"] == ["Gone"]
        assert [hit.point_id for hit in result.results] == ["a1", "a2"]

    def test_single_missing_corpus_is_unknown_corpus(self, semantic):
        with pytest.raises(errors.UnknownCorpusError) as exc_info:
            semantic.module.search_semantic(
                "query", ["Gone"], client=semantic.client, embedder=semantic.embedder
            )

        assert exc_info.value.code == "unknown-corpus"

    def test_unparseable_filter_is_bad_filter(self, semantic):
        with pytest.raises(errors.BadFilterError) as exc_info:
            semantic.module.search_semantic(
                "query",
                ["A"],
                filter='{"filename": "db(31).pdf"}',
                client=semantic.client,
                embedder=semantic.embedder,
            )

        assert exc_info.value.code == "bad-filter"
        assert semantic.calls == []

    def test_unreachable_qdrant_is_service_down(self, semantic):
        semantic.client.collection_exists.side_effect = ResponseHandlingException(
            httpx.ConnectError("connection refused")
        )

        with pytest.raises(errors.ServiceDownError) as exc_info:
            semantic.module.search_semantic(
                "query", ["A"], client=semantic.client, embedder=semantic.embedder
            )

        assert exc_info.value.code == "service-down"
        assert "Qdrant" in str(exc_info.value)

    def test_busy_slot_is_semantic_unavailable(self, semantic, monkeypatch):
        @contextmanager
        def busy():
            raise errors.SearchSlotUnavailable("All 2 embedder slots busy")
            yield

        monkeypatch.setattr(semantic.module, "acquire_embedder_slot", busy)

        with pytest.raises(errors.SemanticUnavailableError) as exc_info:
            semantic.module.search_semantic(
                "query", ["A"], client=semantic.client, embedder=semantic.embedder
            )

        assert exc_info.value.code == "semantic-unavailable"

    def test_uncached_model_without_download_is_semantic_unavailable(self, semantic, monkeypatch):
        from arcaneum.search.embedder import ModelNotCachedError

        def uncached(**_kwargs):
            raise ModelNotCachedError("arctic-m")

        monkeypatch.setattr(semantic.module, "search_collection", uncached)

        with pytest.raises(errors.SemanticUnavailableError) as exc_info:
            semantic.module.search_semantic(
                "query",
                ["A"],
                allow_download=False,
                client=semantic.client,
                embedder=semantic.embedder,
            )

        assert "arctic-m" in str(exc_info.value)

    def test_unreachable_model_host_is_not_qdrant_service_down(self, semantic, monkeypatch):
        from arcaneum.search.embedder import QueryEmbeddingError

        def download_fails(**_kwargs):
            try:
                raise httpx.ConnectError("huggingface.co unreachable")
            except httpx.ConnectError as e:
                raise QueryEmbeddingError("arctic-m", e) from e

        monkeypatch.setattr(semantic.module, "search_collection", download_fails)

        with pytest.raises(errors.ArcaneumError) as exc_info:
            semantic.module.search_semantic(
                "query", ["A"], client=semantic.client, embedder=semantic.embedder
            )

        assert not isinstance(exc_info.value, errors.ServiceDownError)
        assert exc_info.value.code == "error"
        assert "Qdrant" not in str(exc_info.value)
        assert "arctic-m" in str(exc_info.value)

    @pytest.mark.parametrize("allow_download", [True, False])
    def test_allow_download_reaches_the_embedder(self, semantic, monkeypatch, allow_download):
        built = []
        monkeypatch.setattr(
            semantic.module,
            "SearchEmbedder",
            lambda **kwargs: built.append(kwargs) or MagicMock(),
        )

        semantic.module.search_semantic(
            "query", ["A"], allow_download=allow_download, client=semantic.client
        )

        assert built[0]["allow_download"] is allow_download

    def test_writes_nothing_to_stdout(self, semantic, capsys):
        semantic.module.search_semantic(
            "query", ["A", "Gone"], client=semantic.client, embedder=semantic.embedder, verbose=True
        )
        with pytest.raises(errors.UnknownCorpusError):
            semantic.module.search_semantic(
                "query", ["Gone"], client=semantic.client, embedder=semantic.embedder
            )

        assert capsys.readouterr().out == ""


class TestSearchEmbedderDownloadPolicy:
    """allow_download=False refuses a model missing from the local cache."""

    def _embedder(self, monkeypatch, cached):
        from arcaneum.search import embedder as module

        embedding_client = MagicMock()
        embedding_client.is_model_cached.return_value = cached
        monkeypatch.setattr(module, "EmbeddingClient", lambda **_kwargs: embedding_client)
        monkeypatch.setattr(module, "get_collection_metadata", lambda *_args: {})
        monkeypatch.setattr(module, "prompt_policy_issues", lambda *_args: [])
        return module, embedding_client

    def _client(self):
        client = MagicMock()
        client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors={"arctic-m": object()}))
        )
        return client

    def test_uncached_model_raises_before_loading(self, monkeypatch):
        module, embedding_client = self._embedder(monkeypatch, cached=False)
        embedder = module.SearchEmbedder(cache_dir="/tmp/models", allow_download=False)

        with pytest.raises(module.ModelNotCachedError) as exc_info:
            embedder.generate_query_embedding("query", "Docs", self._client())

        assert exc_info.value.model_key == "arctic-m"
        embedding_client.get_model.assert_not_called()

    def test_download_allowed_by_default(self, monkeypatch):
        module, embedding_client = self._embedder(monkeypatch, cached=False)
        embedding_client.get_model.return_value.query_embed.return_value = iter([[0.1, 0.2]])
        embedder = module.SearchEmbedder(cache_dir="/tmp/models")

        model_key, vector = embedder.generate_query_embedding("query", "Docs", self._client())

        assert model_key == "arctic-m"
        embedding_client.get_model.assert_called_once_with("arctic-m")

    def test_model_load_failure_is_query_embedding_error(self, monkeypatch):
        module, embedding_client = self._embedder(monkeypatch, cached=False)
        cause = httpx.ConnectError("huggingface.co unreachable")
        embedding_client.get_model.side_effect = RuntimeError("Network connection failed")
        embedding_client.get_model.side_effect.__cause__ = cause
        embedder = module.SearchEmbedder(cache_dir="/tmp/models")

        with pytest.raises(module.QueryEmbeddingError) as exc_info:
            embedder.generate_query_embedding("query", "Docs", self._client())

        assert exc_info.value.model_key == "arctic-m"
        assert "Network connection failed" in str(exc_info.value)

    def test_qdrant_failure_before_loading_is_not_query_embedding_error(self, monkeypatch):
        module, _embedding_client = self._embedder(monkeypatch, cached=True)
        client = self._client()
        client.get_collection.side_effect = ResponseHandlingException(
            httpx.ConnectError("connection refused")
        )
        embedder = module.SearchEmbedder(cache_dir="/tmp/models")

        with pytest.raises(ResponseHandlingException):
            embedder.generate_query_embedding("query", "Docs", client)


# ---------------------------------------------------------------------------
# search_text
# ---------------------------------------------------------------------------


@pytest.fixture
def text_client():
    client = MagicMock()
    client.health_check.return_value = True
    client.index_exists.side_effect = lambda name: name != "Gone"

    def search(index, query, **kwargs):
        hits = {
            "A": [{"id": "a1", "content": "a1"}, {"id": "a2", "content": "a2"}],
            "B": [{"id": "b1", "content": "b1"}],
        }[index]
        return {
            "hits": [dict(hit) for hit in hits],
            "estimatedTotalHits": len(hits),
            "processingTimeMs": 3,
        }

    client.search.side_effect = search
    client.get_index_settings.return_value = {"filterableAttributes": ["file_path", "page_number"]}
    return client


class TestSearchText:
    def test_hits_interleave_round_robin(self, text_client):
        from arcaneum.retrieval import search_text

        result = search_text("query", ["A", "B"], limit=10, client=text_client)

        assert [hit["id"] for hit in result.hits] == ["a1", "b1", "a2"]
        assert [hit["_corpus"] for hit in result.hits] == ["A", "B", "A"]
        data = result.to_data()
        assert data["query"] == "query"
        assert data["index"] == "[A, B]"
        assert data["corpora"] == ["A", "B"]
        assert data["estimatedTotalHits"] == 3
        assert data["processingTimeMs"] == 6
        assert data["missing_corpora"] == []

    def test_missing_corpus_in_multi_search_is_reported_not_fatal(self, text_client):
        from arcaneum.retrieval import search_text

        result = search_text("query", ["A", "Gone"], client=text_client)

        assert result.missing_corpora == ["Gone"]
        assert result.to_data()["missing_corpora"] == ["Gone"]

    def test_single_missing_corpus_is_unknown_corpus(self, text_client):
        from arcaneum.retrieval import search_text

        with pytest.raises(errors.UnknownCorpusError):
            search_text("query", ["Gone"], client=text_client)

    def test_unhealthy_meilisearch_is_service_down(self, text_client):
        from arcaneum.retrieval import search_text

        text_client.health_check.return_value = False

        with pytest.raises(errors.ServiceDownError) as exc_info:
            search_text("query", ["A"], client=text_client)

        assert "MeiliSearch" in str(exc_info.value)

    def test_lost_connection_is_service_down(self, text_client):
        from arcaneum.retrieval import search_text

        text_client.search.side_effect = MeilisearchCommunicationError("connection refused")

        with pytest.raises(errors.ServiceDownError):
            search_text("query", ["A"], client=text_client)

    def test_unfilterable_attribute_is_bad_filter_naming_filterable_fields(self, text_client):
        from arcaneum.retrieval import search_text

        text_client.search.side_effect = _meili_filter_error("filename")

        with pytest.raises(errors.BadFilterError) as exc_info:
            search_text("query", ["A"], filter='filename = "db(31).pdf"', client=text_client)

        message = str(exc_info.value)
        assert "filename" in message
        assert "file_path" in message
        assert "page_number" in message
        text_client.get_index_settings.assert_called_once_with("A")

    def test_unexpected_failure_keeps_its_message(self, text_client):
        from arcaneum.retrieval import search_text

        text_client.search.side_effect = RuntimeError("index exploded")

        with pytest.raises(errors.ArcaneumError) as exc_info:
            search_text("query", ["A"], client=text_client)

        assert "index exploded" in str(exc_info.value)

    def test_writes_nothing_to_stdout(self, text_client, capsys):
        from arcaneum.retrieval import search_text

        search_text("query", ["A", "Gone"], client=text_client, verbose=True)
        text_client.search.side_effect = _meili_filter_error()
        with pytest.raises(errors.BadFilterError):
            search_text("query", ["A"], filter="filename = x", client=text_client)

        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# list_corpora
# ---------------------------------------------------------------------------

_QDRANT_WRITES = (
    "upsert",
    "set_payload",
    "overwrite_payload",
    "delete_payload",
    "delete",
    "update_collection",
    "create_payload_index",
    "create_collection",
)


class _ListQdrant:
    """Read-only Qdrant double that records any write it receives."""

    def __init__(self, names, points_count=4):
        self.names = names
        self.points_count = points_count
        self.writes = []

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.names])

    def get_collection(self, _name):
        return SimpleNamespace(
            points_count=self.points_count,
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=768, distance="Cosine"))
            ),
        )

    def facet(self, collection_name, key, **_kwargs):
        return SimpleNamespace(
            hits=[SimpleNamespace(value=f"/{collection_name}/{i}.pdf", count=1) for i in range(2)]
        )

    def scroll(self, **_kwargs):
        return [], None

    def __getattr__(self, name):
        if name in _QDRANT_WRITES:
            return lambda *args, **kwargs: self.writes.append((name, args, kwargs))
        raise AttributeError(name)


class _ListMeili:
    def __init__(self, chunks, healthy=True):
        self.chunks = chunks
        self.healthy = healthy

    def health_check(self):
        return self.healthy

    def list_indexes(self):
        return [{"uid": name} for name in self.chunks]

    def get_stats(self):
        return {"indexes": {name: {"numberOfDocuments": n} for name, n in self.chunks.items()}}


@pytest.fixture
def list_metadata(monkeypatch):
    from arcaneum.retrieval import corpora as module

    metadata = {"Docs": {"collection_type": "pdf", "model": "arctic-m"}}
    monkeypatch.setattr(module, "get_collection_metadata", lambda _c, name: metadata.get(name, {}))
    return metadata


class TestListCorpora:
    @pytest.mark.parametrize("details", [False, True])
    def test_lists_names_and_counts_without_writing(self, list_metadata, details):
        from arcaneum.retrieval import list_corpora

        qdrant = _ListQdrant(["Docs"])
        listing = list_corpora(
            details=details, qdrant_client=qdrant, meili_client=_ListMeili({"Docs": 3})
        )

        data = listing.to_data()
        assert [corpus["name"] for corpus in data["corpora"]] == ["Docs"]
        corpus = data["corpora"][0]
        assert corpus["type"] == "pdf"
        assert corpus["status"] == "synced"
        assert corpus["item_count"] == 2
        assert corpus["item_count_status"] == "facet"
        assert qdrant.writes == []

    def test_one_service_down_lists_the_other_with_a_service_error(self, list_metadata):
        from arcaneum.retrieval import list_corpora

        listing = list_corpora(
            qdrant_client=_ListQdrant(["Docs"]), meili_client=_ListMeili({}, healthy=False)
        )

        data = listing.to_data()
        assert [corpus["name"] for corpus in data["corpora"]] == ["Docs"]
        assert data["corpora"][0]["status"] == "qdrant_only"
        assert data["service_errors"] == ["MeiliSearch: server not available"]

    def test_both_services_down_is_service_down(self, list_metadata):
        from arcaneum.retrieval import list_corpora

        class DownQdrant:
            def get_collections(self):
                raise ResponseHandlingException(httpx.ConnectError("connection refused"))

        with pytest.raises(errors.ServiceDownError) as exc_info:
            list_corpora(qdrant_client=DownQdrant(), meili_client=_ListMeili({}, healthy=False))

        assert "Qdrant" in str(exc_info.value)
        assert "MeiliSearch" in str(exc_info.value)

    def test_qdrant_failing_midway_keeps_the_collections_already_read(self, list_metadata):
        from arcaneum.retrieval import list_corpora

        class FailsOnSecond(_ListQdrant):
            def get_collection(self, name):
                if name == "Later":
                    raise RuntimeError("read timed out")
                return super().get_collection(name)

        listing = list_corpora(
            qdrant_client=FailsOnSecond(["Docs", "Later"]),
            meili_client=_ListMeili({"Docs": 3, "Later": 1}),
        )

        data = listing.to_data()
        by_name = {corpus["name"]: corpus for corpus in data["corpora"]}
        assert by_name["Docs"]["status"] == "synced"
        assert by_name["Docs"]["qdrant_chunks"] == 3
        assert data["service_errors"] == ["Qdrant: read timed out"]

    def test_qdrant_failing_midway_with_meili_down_still_lists(self, list_metadata):
        from arcaneum.retrieval import list_corpora

        class FailsOnSecond(_ListQdrant):
            def get_collection(self, name):
                if name == "Later":
                    raise RuntimeError("read timed out")
                return super().get_collection(name)

        listing = list_corpora(
            qdrant_client=FailsOnSecond(["Docs", "Later"]),
            meili_client=_ListMeili({}, healthy=False),
        )

        assert [corpus["name"] for corpus in listing.corpora] == ["Docs"]
        assert len(listing.service_errors) == 2

    def test_client_factory_failure_is_a_service_error_not_fatal(self, list_metadata):
        from arcaneum.retrieval import list_corpora

        def unreadable_key():
            raise PermissionError("meili key file unreadable")

        listing = list_corpora(
            qdrant_factory=lambda: _ListQdrant(["Docs"]), meili_factory=unreadable_key
        )

        data = listing.to_data()
        assert [corpus["name"] for corpus in data["corpora"]] == ["Docs"]
        assert data["service_errors"] == ["MeiliSearch: meili key file unreadable"]

    def test_writes_nothing_to_stdout(self, list_metadata, capsys):
        from arcaneum.retrieval import list_corpora

        list_corpora(qdrant_client=_ListQdrant(["Docs"]), meili_client=_ListMeili({"Docs": 3}))
        list_corpora(
            qdrant_client=_ListQdrant(["Docs"]), meili_client=_ListMeili({}, healthy=False)
        )

        assert capsys.readouterr().out == ""

"""``search_text`` serves each corpus from its own text backend (RDR-027 Q2, Q5).

The stores are ``fulltext.conftest``'s in-memory MeiliSearch and local-mode
Qdrant, written through the builders sync uses (``to_meilisearch_doc`` and
``to_qdrant_point``), so a chunk is the same chunk in both.
"""

from __future__ import annotations

import pytest
from fulltext.conftest import FakeFullTextClient, LocalQdrant

from arcaneum.cli import errors
from arcaneum.fulltext.backend import MeiliTextBackend
from arcaneum.indexing.collection_metadata import (
    set_collection_metadata,
    update_collection_metadata,
)
from arcaneum.retrieval import search_text
from arcaneum.retrieval.qdrant_text import QdrantTextBackend
from arcaneum.schema.document import DualIndexDocument


def _chunk(chunk_id: str, path: str, content: str, **fields) -> DualIndexDocument:
    return DualIndexDocument(
        id=chunk_id,
        content=content,
        file_path=path,
        filename=path.rsplit("/", 1)[-1],
        file_extension=".py",
        language="python",
        project="app",
        branch="main",
        git_project_identifier="app#main",
        chunk_count=2,
        vectors={"stub": [1.0, 0.0]},
        **fields,
    )


def _uuid(n: int) -> str:
    # Clear of METADATA_POINT_ID (...0001), which a chunk must never overwrite.
    return f"00000000-0000-0000-0000-{0xC000 + n:012x}"


class Stores:
    def __init__(self, meili_healthy: bool = True):
        self.qdrant = LocalQdrant()
        self.meili_client = FakeFullTextClient(healthy=meili_healthy)
        self.meili = MeiliTextBackend(self.meili_client)

    def corpus(self, name, text_backend, docs, corpus_type="code", roots=("/src",), metadata=True):
        """A corpus as create and sync leave it: a collection, and its text store."""
        self.qdrant.create_collection_with_vector(name)
        if metadata:
            extra = {"text_backend": text_backend} if text_backend else {}
            set_collection_metadata(
                self.qdrant, name, corpus_type, "jina-code", source_roots=list(roots), **extra
            )
        QdrantTextBackend(self.qdrant).add(name, docs)  # sync's point upsert
        if text_backend == "qdrant":
            QdrantTextBackend(self.qdrant).ensure(name, corpus_type)
        else:
            self.meili.ensure(name, corpus_type)
            self.meili.add(name, docs)

    def search(self, query, corpora, **kwargs):
        return search_text(
            query, corpora, text_backend=self.meili, qdrant_client=self.qdrant, **kwargs
        )


@pytest.fixture
def stores():
    return Stores()


def test_a_corpus_recording_no_backend_stays_on_meilisearch(stores):
    stores.corpus("Legacy", None, [_chunk(_uuid(1), "/src/app/a.py", "def scan(): pass")])
    stores.qdrant.query_calls.clear()

    result = stores.search("scan", ["Legacy"])

    assert [hit["chunk_id"] for hit in result.to_data()["hits"]] == [_uuid(1)]
    assert stores.qdrant.query_calls == []
    assert not hasattr(result, "estimated_total_hits")
    assert not hasattr(result, "processing_time_ms")


def test_a_qdrant_corpus_is_served_by_qdrant(stores):
    stores.corpus("Code", "qdrant", [_chunk(_uuid(1), "/src/app/a.py", "def scan(): pass")])

    data = stores.search("scan", ["Code"]).to_data()

    assert [hit["chunk_id"] for hit in data["hits"]] == [_uuid(1)]
    assert data["hits"][0]["cite"] == "Code:app/a.py"
    assert stores.meili_client.indexes == {}  # no MeiliSearch index was ever made
    assert stores.qdrant.query_calls


def test_mixed_corpora_merge_round_robin_with_one_hit_schema(stores):
    stores.corpus(
        "Meili",
        "meili",
        [
            _chunk(_uuid(1), "/src/m/a.py", "scan one"),
            _chunk(_uuid(2), "/src/m/b.py", "scan two"),
        ],
    )
    stores.corpus("Qdrant", "qdrant", [_chunk(_uuid(3), "/src/q/c.py", "scan three")])

    hits = stores.search("scan", ["Meili", "Qdrant"]).to_data(verbose=True)["hits"]

    assert [(hit["corpus"], hit["chunk_id"]) for hit in hits] == [
        ("Meili", _uuid(1)),
        ("Qdrant", _uuid(3)),
        ("Meili", _uuid(2)),
    ]
    assert [hit["cite"] for hit in hits] == ["Meili:m/a.py", "Qdrant:q/c.py", "Meili:m/b.py"]
    assert len({tuple(sorted(hit)) for hit in hits}) == 1
    assert len({tuple(sorted(hit["metadata"])) for hit in hits}) == 1


def test_the_same_chunk_gives_the_same_hit_on_either_backend(stores):
    """Scenario 7: only the score differs once a corpus moves to Qdrant."""
    chunk = _chunk(
        _uuid(9), "/src/app/scan.py", "def scan(paths):\n    return sorted(paths)", line_number=3
    )
    stores.corpus("Same", "meili", [chunk])
    before = stores.search("scan", ["Same"]).to_data(verbose=True)["hits"]

    QdrantTextBackend(stores.qdrant).ensure("Same", "code")
    update_collection_metadata(stores.qdrant, "Same", text_backend="qdrant")
    after = stores.search("scan", ["Same"]).to_data(verbose=True)["hits"]

    assert len(before) == len(after) == 1
    (meili_hit,), (qdrant_hit,) = before, after
    assert meili_hit["score"] != qdrant_hit["score"]
    assert {k: v for k, v in meili_hit.items() if k != "score"} == {
        k: v for k, v in qdrant_hit.items() if k != "score"
    }


def test_a_qdrant_only_search_runs_with_meilisearch_down():
    stores = Stores(meili_healthy=False)
    stores.corpus("Code", "qdrant", [_chunk(_uuid(1), "/src/app/a.py", "def scan(): pass")])

    assert len(stores.search("scan", ["Code"]).hits) == 1


def test_a_set_with_a_meili_corpus_still_needs_meilisearch():
    stores = Stores(meili_healthy=False)
    stores.corpus("Code", "qdrant", [_chunk(_uuid(1), "/src/app/a.py", "def scan(): pass")])
    stores.meili_client.create_index("Meili")

    with pytest.raises(errors.ServiceDownError) as exc_info:
        stores.search("scan", ["Code", "Meili"])
    assert "MeiliSearch" in str(exc_info.value)


def test_a_qdrant_corpus_without_its_text_index_names_repair(stores):
    stores.corpus("Code", "qdrant", [_chunk(_uuid(1), "/src/app/a.py", "def scan(): pass")])
    QdrantTextBackend(stores.qdrant).drop("Code")

    with pytest.raises(errors.ResourceNotFoundError) as exc_info:
        stores.search("scan", ["Code"])
    assert "arc corpus repair Code" in str(exc_info.value)


def test_a_meilisearch_only_corpus_beside_a_qdrant_one(stores):
    """A corpus Qdrant does not hold (`arc index text`) is MeiliSearch's, and
    does not stop the next corpus's metadata read."""
    stores.meili.ensure("TextOnly", "code")
    stores.meili.add("TextOnly", [_chunk(_uuid(1), "/src/t/a.py", "scan text")])
    stores.corpus("Code", "qdrant", [_chunk(_uuid(2), "/src/q/b.py", "scan code")])

    hits = stores.search("scan", ["TextOnly", "Code"]).to_data()["hits"]

    assert [(hit["corpus"], hit["cite"]) for hit in hits] == [
        ("TextOnly", "TextOnly:/src/t/a.py"),
        ("Code", "Code:q/b.py"),
    ]


def test_a_bad_filter_for_a_qdrant_corpus_is_refused_before_any_query(stores):
    stores.corpus("Code", "qdrant", [_chunk(_uuid(1), "/src/app/a.py", "def scan(): pass")])
    stores.qdrant.query_calls.clear()

    with pytest.raises(errors.BadFilterError):
        stores.search("scan", ["Code"], filter='{"must": [{"key": "x"}], "bogus": 1}')
    assert stores.qdrant.query_calls == []


def test_the_mcp_tool_serves_a_mixed_set(monkeypatch, stores):
    """Scenario 6, through the MCP server's search_text tool."""
    import arcaneum.mcp_server as mcp_server

    stores.corpus("Meili", "meili", [_chunk(_uuid(1), "/src/m/a.py", "scan one")])
    stores.corpus("Qdrant", "qdrant", [_chunk(_uuid(2), "/src/q/b.py", "scan two")])
    monkeypatch.setattr(mcp_server, "_meili_client", lambda: stores.meili_client)
    monkeypatch.setattr(mcp_server, "_source_roots_client", lambda: stores.qdrant)

    data = mcp_server.ArcTools(allowlist=None).search_text({"query": "scan", "corpora": ["Meili", "Qdrant"]})

    assert [(hit["corpus"], hit["cite"]) for hit in data["hits"]] == [
        ("Meili", "Meili:m/a.py"),
        ("Qdrant", "Qdrant:q/b.py"),
    ]
    assert len({tuple(sorted(hit)) for hit in data["hits"]}) == 1

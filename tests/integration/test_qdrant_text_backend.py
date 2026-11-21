"""``QdrantTextBackend`` against a real Qdrant 1.20 server (RDR-027 Q3-Q5).

Local mode builds no payload index, so the unit tests read the index
parameters the backend asks for; here the server builds Q3's index and runs
Q4's queries. Scenario 7 also needs MeiliSearch.

Requires QDRANT_URL (default http://localhost:6333) at 1.20 or later, and
MEILISEARCH_URL (default http://localhost:7700) for scenario 7; skips
otherwise. Each test uses collections and indexes of its own and removes them.
"""

from __future__ import annotations

import os
import uuid

import pytest

from arcaneum.retrieval.qdrant_text import QdrantTextBackend, tokenize
from arcaneum.schema.document import DualIndexDocument

pytestmark = pytest.mark.integration

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
MEILISEARCH_URL = os.environ.get("MEILISEARCH_URL", "http://localhost:7700")
PDF_STOP_WORDS = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"]

TEXTS = [
    "The write-ahead log flushes before the checkpoint completes.",
    "A checkpoint flushes dirty pages; the log is truncated afterwards.",
    "Café owners log their sales in a ledger.",
    "Schema evolution in the write-ahead log of a database.",
    "### ** `` __",
]


def _id(n: int) -> str:
    return str(uuid.UUID(int=0xA000 + n))


def _doc(n: int, content: str, path: str = None) -> DualIndexDocument:
    path = path or f"/docs/{n}.pdf"
    return DualIndexDocument(
        id=_id(n),
        content=content,
        file_path=path,
        filename=path.rsplit("/", 1)[-1],
        file_extension=".pdf",
        chunk_index=n,
        page_number=n + 1,
        vectors={"stub": [1.0, float(n)]},
    )


@pytest.fixture
def qdrant():
    from qdrant_client import QdrantClient

    client = QdrantClient(url=QDRANT_URL, timeout=30)
    try:
        version = client.info().version
    except Exception as e:
        pytest.skip(f"Qdrant not available at {QDRANT_URL}: {e}")
    if tuple(int(p) for p in version.split(".")[:2]) < (1, 20):
        pytest.skip(f"Qdrant {version} has no BM25 text index; 1.20 is required")
    return client


@pytest.fixture
def collection(qdrant):
    from qdrant_client.http import models

    names = []

    def make(corpus_type: str = "pdf", docs=None) -> str:
        name = f"arc_text_it_{uuid.uuid4().hex[:10]}"
        names.append(name)
        qdrant.create_collection(
            name,
            vectors_config={"stub": models.VectorParams(size=2, distance=models.Distance.COSINE)},
        )
        backend = QdrantTextBackend(qdrant)
        backend.ensure(name, corpus_type)
        backend.add(name, docs if docs is not None else [_doc(i, t) for i, t in enumerate(TEXTS)])
        return name

    yield make
    for name in names:
        qdrant.delete_collection(name)


def _ids(page):
    return [hit["id"] for hit in page.hits]


def test_the_server_holds_q3s_index(qdrant, collection):
    name = collection()
    QdrantTextBackend(qdrant).ensure(name, "pdf")  # idempotent

    entry = qdrant.get_collection(name).payload_schema["text"]
    params = entry.params
    assert str(entry.data_type.value) == "text"
    assert (params.tokenizer.value, params.lowercase, params.ascii_folding) == ("word", True, True)
    assert params.phrase_matching is True
    assert sorted(params.stopwords.custom) == sorted(PDF_STOP_WORDS)
    assert params.memory.value == "cold"
    assert params.scoring.type.value == "bm25"


def test_q4_on_the_server(qdrant, collection):
    name = collection()
    backend = QdrantTextBackend(qdrant, corpus_types={name: "pdf"})

    assert _ids(backend.search(name, '"write-ahead log" checkpoint')) == [_id(0)]
    assert set(_ids(backend.search(name, "log checkpoint"))) == {_id(0), _id(1)}
    assert set(_ids(backend.search(name, "ledger checkpoint"))) == {_id(0), _id(1), _id(2)}
    assert _ids(backend.search(name, '"schema evolution" checkpoint')) == [_id(3)]
    assert _ids(backend.search(name, "cafe")) == [_id(2)]
    assert set(_ids(backend.search(name, '"the write-ahead log"'))) == {_id(0), _id(3)}
    assert len(backend.search(name, "", limit=10).hits) == len(TEXTS)
    hit = backend.search(name, "ledger").hits[0]
    assert hit["content"] == TEXTS[2] and hit["page_number"] == 3
    assert isinstance(hit["_rankingScore"], float)


def test_a_code_corpus_keeps_stop_words_searchable(qdrant, collection):
    name = collection("code", [_doc(0, "for item in items"), _doc(1, "while true")])

    assert _ids(QdrantTextBackend(qdrant).search(name, "for in")) == [_id(0)]


def test_the_python_tokenizer_matches_the_index(qdrant, collection):
    """Every token ``tokenize`` yields matches its own chunk, and a chunk with
    no token matches nothing (RDR-027 Q8's flag rests on this)."""
    from qdrant_client.http import models

    name = collection()
    for n, text in enumerate(TEXTS):
        tokens = tokenize(text, "pdf")
        for token in tokens:
            count = qdrant.count(
                name,
                count_filter=models.Filter(
                    must=[
                        models.HasIdCondition(has_id=[_id(n)]),
                        models.FieldCondition(key="text", match=models.MatchText(text=token)),
                    ]
                ),
                exact=True,
            ).count
            assert count == 1, (text, token)
    assert tokenize(TEXTS[4], "pdf") == []
    assert qdrant.get_collection(name).payload_schema["text"].points == len(TEXTS) - 1


def test_writes_and_reads_on_the_server(qdrant, collection):
    name = collection()
    backend = QdrantTextBackend(qdrant, corpus_types={name: "pdf"})

    assert backend.exists(name)
    assert backend.get_document(name, _id(1))["content"] == TEXTS[1]
    assert backend.has_file(name, "/docs/1.pdf")
    backend.rename(name, {("/docs/1.pdf", "/moved/x.pdf"): [_id(1)]})
    assert backend.get_document(name, _id(1))["filename"] == "x.pdf"
    assert backend.file_documents(name, "/moved/x.pdf", ["id"]) == ([{"id": _id(1)}], 1)
    backend.delete_files(name, ["/moved/x.pdf"])
    assert not backend.has_file(name, "/moved/x.pdf")
    assert backend.coverage(name).chunks == len(TEXTS) - 1
    assert backend.corpora()[name] == len(TEXTS) - 1
    backend.drop(name)
    assert not backend.exists(name)


def test_scenario_7_the_same_chunk_from_either_backend(qdrant, collection):
    """A chunk written by sync's DualIndexer gives one hit on MeiliSearch and
    on Qdrant: only the score differs."""
    from arcaneum.fulltext.backend import MeiliTextBackend
    from arcaneum.fulltext.client import FullTextClient
    from arcaneum.indexing.collection_metadata import (
        set_collection_metadata,
        update_collection_metadata,
    )
    from arcaneum.indexing.dual_indexer import DualIndexer
    from arcaneum.retrieval import search_text

    meili_client = FullTextClient(url=MEILISEARCH_URL, api_key=os.environ.get("MEILI_MASTER_KEY"))
    if not meili_client.health_check():
        pytest.skip(f"MeiliSearch not available at {MEILISEARCH_URL}")

    name = collection(docs=[])
    meili = MeiliTextBackend(meili_client)
    meili.ensure(name, "pdf")
    try:
        set_collection_metadata(qdrant, name, "pdf", "stella", source_roots=["/docs"])
        doc = _doc(7, "The write-ahead log flushes before the checkpoint.", "/docs/wal.pdf")
        doc.title = "WAL notes"
        DualIndexer(qdrant, meili_client, name, name).index_batch([doc])

        before = search_text(
            '"write-ahead log"', [name], text_backend=meili, qdrant_client=qdrant
        ).to_data(verbose=True)["hits"]
        update_collection_metadata(qdrant, name, text_backend="qdrant")
        after = search_text(
            '"write-ahead log"', [name], text_backend=meili, qdrant_client=qdrant
        ).to_data(verbose=True)["hits"]
    finally:
        meili.drop(name)

    assert len(before) == len(after) == 1
    assert before[0]["cite"] == f"{name}:wal.pdf#page=8"
    assert {k: v for k, v in before[0].items() if k != "score"} == {
        k: v for k, v in after[0].items() if k != "score"
    }

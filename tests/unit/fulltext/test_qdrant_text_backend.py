"""``QdrantTextBackend`` beyond the shared contract (RDR-027 Q3-Q5).

The index parameters are read from what the backend asked for, since local
mode builds no index of its own (``conftest.LocalQdrant``); the query shapes
are read from the ``query_points`` calls.
"""

from __future__ import annotations

import pytest

from arcaneum.cli.errors import BadFilterError
from arcaneum.indexing.collection_metadata import (
    FILE_MANIFEST_PAYLOAD_KEY,
    FILE_MANIFEST_PAYLOAD_VALUE,
    METADATA_PAYLOAD_KEY,
    METADATA_POINT_ID,
)
from arcaneum.retrieval.qdrant_text import QdrantTextBackend, text_index_params, tokenize
from arcaneum.schema.document import DualIndexDocument
from arcaneum.search.filters import parse_filter

from .conftest import LocalQdrant

PDF_STOP_WORDS = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"]

A0 = "00000000-0000-0000-0000-0000000000a0"
A1 = "00000000-0000-0000-0000-0000000000a1"
B0 = "00000000-0000-0000-0000-0000000000b0"
MANIFEST = "00000000-0000-0000-0000-0000000000f0"


def _doc(chunk_id, path, index, content):
    return DualIndexDocument(
        id=chunk_id,
        content=content,
        file_path=path,
        filename=path.rsplit("/", 1)[-1],
        file_extension=".pdf",
        chunk_index=index,
        vectors={"stub": [1.0, 0.0]},
    )


@pytest.fixture
def client():
    client = LocalQdrant()
    client.create_collection_with_vector("Docs")
    return client


@pytest.fixture
def backend(client):
    backend = QdrantTextBackend(client)
    backend.ensure("Docs", "pdf")
    backend.add(
        "Docs",
        [
            _doc(A0, "/docs/a.pdf", 0, "the quick brown fox"),
            _doc(A1, "/docs/a.pdf", 1, "jumps over the lazy dog"),
            _doc(B0, "/docs/b.pdf", 0, "a quick note on foxes and a ledger"),
        ],
    )
    return backend


def _reserved_points(client):
    """The collection metadata point and a file manifest, as sync writes them."""
    from qdrant_client.http import models

    client.upsert(
        "Docs",
        points=[
            models.PointStruct(
                id=METADATA_POINT_ID,
                vector={"stub": [1.0, 0.0]},
                payload={METADATA_PAYLOAD_KEY: True, "text": "quick", "collection_type": "pdf"},
            ),
            models.PointStruct(
                id=MANIFEST,
                vector={"stub": [1.0, 0.0]},
                payload={
                    FILE_MANIFEST_PAYLOAD_KEY: FILE_MANIFEST_PAYLOAD_VALUE,
                    "file_path": "/docs/a.pdf",
                    "text": "quick",
                },
            ),
        ],
    )


def _ids(page):
    return [str(hit["id"]) for hit in page.hits]


def _text_conditions(call):
    """The ``text`` conditions of a query's filter, by match type."""
    found = []
    pending = list(call["query_filter"].must or [])
    while pending:
        condition = pending.pop(0)
        nested = getattr(condition, "must", None)
        if nested is not None:
            pending.extend(nested)
        elif getattr(condition, "key", None) == "text":
            found.append(condition.match)
    return found


class TestTextIndex:
    def test_ensure_asks_for_q3s_index(self, client):
        from qdrant_client.http import models

        QdrantTextBackend(client).ensure("Docs", "pdf")

        params = client.requested[("Docs", "text")]
        assert params.type == models.TextIndexType.TEXT
        assert params.tokenizer == models.TokenizerType.WORD
        assert params.lowercase is True
        assert params.ascii_folding is True
        assert params.phrase_matching is True
        assert params.scoring == models.TextScoringParams(type=models.TextScoringType.BM25)
        assert params.memory == models.Memory.COLD
        assert params.stemmer is None
        assert params.stopwords == models.StopwordsSet(custom=PDF_STOP_WORDS)

    @pytest.mark.parametrize("corpus_type, stop_words", [("markdown", PDF_STOP_WORDS), ("code", None)])
    def test_stop_words_follow_the_corpus_type(self, corpus_type, stop_words):
        params = text_index_params(corpus_type)
        assert (params.stopwords.custom if params.stopwords else None) == stop_words

    def test_ensure_creates_the_index_once(self, client):
        backend = QdrantTextBackend(client)
        backend.ensure("Docs", "pdf")
        backend.ensure("Docs", "pdf")

        assert client.index_calls == [("Docs", "text")]

    def test_reads_create_no_index(self, backend, client):
        client.index_calls.clear()
        reader = QdrantTextBackend(client)

        reader.health()
        reader.exists("Docs")
        reader.search("Docs", '"quick brown" fox', limit=5)
        reader.search("Docs", "", limit=5)
        reader.get_document("Docs", A0)
        reader.file_documents("Docs", "/docs/a.pdf", ["id"])
        reader.has_file("Docs", "/docs/a.pdf")
        reader.coverage("Docs")
        reader.corpora()

        assert client.index_calls == []

    def test_a_collection_without_the_text_index_is_not_a_text_corpus(self, client):
        backend = QdrantTextBackend(client)

        assert not backend.exists("Docs")
        assert not backend.exists("Absent")
        assert backend.corpora() == {}


class TestQueryTranslation:
    def test_phrases_are_match_phrase_and_the_rest_one_match_text(self, backend, client):
        from qdrant_client.http import models

        client.query_calls.clear()
        page = backend.search("Docs", '"quick brown" fox "lazy dog" jumps', limit=5)

        call = client.query_calls[0]
        assert call["using"] == "text"
        assert isinstance(call["query"], models.TextQuery)
        assert call["query"].text == "quick brown fox lazy dog jumps"
        assert _text_conditions(call) == [
            models.MatchPhrase(phrase="quick brown"),
            models.MatchPhrase(phrase="lazy dog"),
            models.MatchText(text="fox jumps"),
        ]
        assert page.hits == []  # no chunk holds every term: the fallback ran

    def test_every_term_is_required_before_the_fallback(self, backend):
        assert _ids(backend.search("Docs", "quick fox", limit=5)) == [A0]

    def test_with_no_chunk_holding_every_term_bm25_ranks_any_term(self, backend, client):
        client.query_calls.clear()
        page = backend.search("Docs", "fox ledger", limit=5)

        assert set(_ids(page)) == {A0, B0}
        fallback = client.query_calls[-1]
        assert fallback["query"].text == "fox ledger"
        assert _text_conditions(fallback) == []

    def test_the_fallback_keeps_the_phrases(self, backend, client):
        from qdrant_client.http import models

        client.query_calls.clear()
        page = backend.search("Docs", '"quick brown" ledger', limit=5)

        assert _ids(page) == [A0]
        assert _text_conditions(client.query_calls[-1]) == [
            models.MatchPhrase(phrase="quick brown")
        ]

    def test_a_query_of_only_stop_words_sends_no_match_text(self, backend, client):
        client.query_calls.clear()
        backend.search("Docs", "the and", limit=5)

        assert all(_text_conditions(call) == [] for call in client.query_calls)

    def test_a_code_corpus_keeps_stop_words_searchable(self, client):
        from qdrant_client.http import models

        backend = QdrantTextBackend(client)
        backend.ensure("Docs", "code")
        backend.add("Docs", [_doc(A0, "/src/a.py", 0, "for item in items")])
        client.query_calls.clear()

        page = backend.search("Docs", "for in", limit=5)

        assert _text_conditions(client.query_calls[0]) == [models.MatchText(text="for in")]
        assert _ids(page) == [A0]

    def test_an_empty_query_lists_chunks_in_id_order(self, backend, client):
        client.query_calls.clear()
        page = backend.search("Docs", "", limit=2, offset=1)

        assert client.query_calls[0]["query"] is None
        assert _ids(page) == [A1, B0]

    def test_a_page_past_the_last_strict_hit_does_not_fall_back(self, backend):
        assert backend.search("Docs", "quick", limit=5, offset=2).hits == []

    def test_a_neutral_filter_is_anded_with_the_text_conditions(self, backend):
        page = backend.search("Docs", "quick", filter=parse_filter("file_path=/docs/b.pdf"))
        assert _ids(page) == [B0]

    def test_a_meilisearch_expression_is_bad_filter(self, backend):
        with pytest.raises(BadFilterError):
            backend.check_filter(parse_filter('filename = "a.pdf" OR page_number > 3'))


class TestHits:
    def test_reserved_points_never_hit(self, backend, client):
        _reserved_points(client)

        for query in ("quick", ""):
            ids = _ids(backend.search("Docs", query, limit=10))
            assert METADATA_POINT_ID not in ids
            assert MANIFEST not in ids
        assert backend.get_document("Docs", METADATA_POINT_ID) is None
        assert backend.get_document("Docs", MANIFEST) is None
        assert backend.coverage("Docs").chunks == 3
        assert backend.coverage("Docs").files == {"/docs/a.pdf": 2, "/docs/b.pdf": 1}
        assert backend.file_documents("Docs", "/docs/a.pdf", ["id"])[1] == 2

    def test_deleting_a_file_keeps_its_manifest(self, backend, client):
        _reserved_points(client)

        backend.delete_files("Docs", ["/docs/a.pdf"])

        assert [str(p.id) for p in client.retrieve("Docs", ids=[MANIFEST])] == [MANIFEST]

    def test_a_hit_carries_the_meilisearch_field_names(self, client):
        backend = QdrantTextBackend(client)
        backend.ensure("Docs", "code")
        backend.add(
            "Docs",
            [
                DualIndexDocument(
                    id=A0,
                    content="def scan(): pass",
                    file_path="/src/scan.py",
                    filename="scan.py",
                    language="python",
                    project="app",
                    branch="main",
                    chunk_count=4,
                    vectors={"stub": [1.0, 0.0]},
                )
            ],
        )

        (hit,) = backend.search("Docs", "scan").hits

        assert (hit["content"], hit["language"], hit["project"], hit["branch"]) == (
            "def scan(): pass",
            "python",
            "app",
            "main",
        )
        assert "text" not in hit and "programming_language" not in hit and "chunk_count" not in hit

    def test_a_string_that_is_no_point_id_is_no_document(self, backend):
        assert backend.get_document("Docs", "not-a-uuid") is None
        assert backend.get_document("Docs", "00000000-0000-0000-0000-00000000dead") is None


class TestTokenize:
    def test_words_are_lowercased_and_ascii_folded(self):
        assert tokenize("Café ÜBER naïve", "code") == ["cafe", "uber", "naive"]

    def test_punctuation_and_underscores_split_and_drop(self):
        assert tokenize("### **write_ahead** `log`", "code") == ["write", "ahead", "log"]
        assert tokenize("### ** `` __", "pdf") == []

    def test_numbers_are_tokens(self):
        assert tokenize("error 404", "pdf") == ["error", "404"]

    def test_stop_words_follow_the_corpus_type(self):
        assert tokenize("the and or to", "pdf") == []
        assert tokenize("the and or to", "code") == ["the", "and", "or", "to"]

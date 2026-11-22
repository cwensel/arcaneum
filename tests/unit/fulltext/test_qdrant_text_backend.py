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
from arcaneum.retrieval.ascii_folding import fold_to_ascii
from arcaneum.retrieval.qdrant_text import QdrantTextBackend, text_index_params, tokenize, zero_token
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

        assert client.index_calls == [("Docs", "text"), ("Docs", "text_tokens")]

    def test_ensure_indexes_the_zero_token_flag_as_a_bool(self, client):
        from qdrant_client.http import models

        QdrantTextBackend(client).ensure("Docs", "pdf")

        assert client.requested[("Docs", "text_tokens")] == models.PayloadSchemaType.BOOL

    def test_ensure_adds_the_flag_index_beside_an_existing_text_index(self, client):
        client.create_payload_index("Docs", "text", field_schema=text_index_params("pdf"))
        client.index_calls.clear()

        QdrantTextBackend(client).ensure("Docs", "pdf")

        assert client.index_calls == [("Docs", "text_tokens")]

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

    # Each case is what a Qdrant 1.20.2 text index (word tokenizer, lowercase,
    # ascii_folding) held for the word, probed against a scratch collection on
    # 2026-11-21 (arcaneum#181): Qdrant folds through Lucene's table, then
    # lowercases, and keeps every character the table lacks.
    @pytest.mark.parametrize(
        "word, token",
        [
            ("Café", "cafe"),
            ("ÉCOLE", "ecole"),
            ("ﬁnance", "finance"),
            ("ﬄ", "ffl"),
            ("ＡＢｃ", "abc"),
            ("x²", "x2"),
            ("Straße", "strasse"),
            ("Øre", "ore"),
            ("Æon", "aeon"),
            ("Łódź", "lodz"),
            ("Þing", "thing"),
            ("ıstanbul", "istanbul"),
            ("Đak", "dak"),
            ("ħa", "ha"),
            ("ðe", "de"),
            ("ſun", "sun"),
            ("ĳs", "ijs"),
            ("œuvre", "oeuvre"),
            ("ᴀbc", "abc"),
            ("ʀx", "rx"),
            ("ǅem", "dzem"),
            # Kept: no entry in the fold table, so only lowercase applies.
            ("𝑦𝑠", "𝑦𝑠"),  # MATHEMATICAL ITALIC SMALL Y, S
            ("𝐂", "𝐂"),  # MATHEMATICAL BOLD CAPITAL C: no lowercase mapping
            ("Ωμέγα", "ωμέγα"),
            ("ά", "ά"),
            ("Ёлка", "ёлка"),
            ("Ⅳ", "ⅳ"),
            ("µm", "µm"),
            # Order witnesses: folding first leaves these to lowercase alone.
            ("Å", "å"),  # ANGSTROM SIGN
            ("K", "k"),  # KELVIN SIGN
        ],
    )
    def test_words_fold_as_the_qdrant_index_does(self, word, token):
        assert tokenize(f"technology {word} min", "code") == ["technology", token, "min"]

    # Qdrant's word tokenizer splits on ``!char::is_alphanumeric`` (Alphabetic,
    # or general category N), which keeps characters Python's ``\w`` drops.
    # Each case is what a Qdrant 1.20.2 text index held, probed against a
    # scratch collection (arcaneum#182).
    @pytest.mark.parametrize(
        "text, tokens",
        [
            ("technology Ⓐⓐ min", ["technology", "aa", "min"]),  # So, Other_Alphabetic
            ("xⓑy", ["xby"]),
            # The vowel sign U+0947 (Mn, Other_Alphabetic) stays in its word; the
            # virama U+094D is not Alphabetic, so it splits.
            ("नमस्ते दुनिया", ["नमस", "ते", "दुनिया"]),
            # Still split: a decomposed combining acute U+0301 is not Alphabetic.
            ("café bar", ["cafe", "bar"]),
            ("write_ahead", ["write", "ahead"]),
        ],
    )
    def test_words_split_as_the_qdrant_index_does(self, text, tokens):
        assert tokenize(text, "code") == tokens

    def test_a_circled_letter_is_a_token(self):
        assert zero_token("ⓐ", "code") is False


class TestAsciiFolding:
    def test_the_table_matches_qdrants_own_examples(self):
        # ascii_folding.rs::tests::basic_examples, Qdrant v1.20.2
        assert fold_to_ascii("ação café jalapeño Über") == "acao cafe jalapeno Uber"
        assert fold_to_ascii("Æsop and Œuvre") == "AEsop and OEuvre"
        assert fold_to_ascii("straße") == "strasse"
        assert fold_to_ascii("Łódź") == "Lodz"
        assert fold_to_ascii("“quote” — test １②⒊") == '"quote" - test 123.'
        assert fold_to_ascii("日本語のテキストです。") == "日本語のテキストです。"

    def test_circled_letters_fold(self):
        assert fold_to_ascii("Ⓐ") == "A"


class TestCoverage:
    """Q8: coverage is complete when the index holds every chunk not flagged as zero-token."""

    Z0 = "00000000-0000-0000-0000-0000000000c0"

    def _add_flagged(self, client, flag=True):
        from qdrant_client.http import models

        payload = {"text": "### ** ``", "file_path": "/docs/a.pdf", "chunk_index": 2}
        if flag:
            payload["text_tokens"] = False
        client.upsert(
            "Docs",
            points=[models.PointStruct(id=self.Z0, vector={"stub": [1.0, 0.0]}, payload=payload)],
        )

    def test_a_flagged_chunk_is_not_a_gap(self, backend, client):
        self._add_flagged(client)

        for files in (True, False):
            coverage = backend.coverage("Docs", files=files)
            assert (coverage.chunks, coverage.searchable_chunks, coverage.indexed) == (4, 3, 3)
            assert coverage.zero_token_chunks == 1
            assert coverage.flagged == [("/docs/a.pdf", 2)]
            assert coverage.status == "synced"
        coverage = backend.coverage("Docs")
        assert coverage.files == {"/docs/a.pdf": 3, "/docs/b.pdf": 1}
        assert coverage.searchable_files == {"/docs/a.pdf": 2, "/docs/b.pdf": 1}

    def test_an_unflagged_zero_token_chunk_is_a_gap(self, backend, client):
        self._add_flagged(client, flag=False)

        coverage = backend.coverage("Docs", files=False)
        assert (coverage.searchable_chunks, coverage.indexed) == (4, 3)
        assert coverage.status == "coverage_gap"

    def test_a_dropped_index_is_missing(self, backend, client):
        client.delete_payload_index("Docs", "text")

        coverage = backend.coverage("Docs", files=False)
        assert coverage.indexed is None
        assert coverage.status == "text_index_missing"
        assert coverage.chunks == 3

    def test_the_flagged_sample_is_bounded(self, backend, client, monkeypatch):
        from arcaneum.retrieval import qdrant_text

        monkeypatch.setattr(qdrant_text, "FLAGGED_SAMPLE_LIMIT", 1)
        self._add_flagged(client)
        self.Z0, saved = "00000000-0000-0000-0000-0000000000c1", self.Z0
        try:
            self._add_flagged(client)
        finally:
            self.Z0 = saved

        for files in (True, False):
            coverage = backend.coverage("Docs", files=files)
            assert coverage.zero_token_chunks == 2
            assert len(coverage.flagged) == 1

    def test_a_given_chunk_count_replaces_the_exact_count(self, backend, client, monkeypatch):
        from arcaneum.retrieval.qdrant_text import _chunk_filter

        self._add_flagged(client)
        filters = []
        real = client._client.count

        def count(collection_name, count_filter=None, **kwargs):
            filters.append(count_filter)
            return real(collection_name, count_filter=count_filter, **kwargs)

        monkeypatch.setattr(client, "count", count, raising=False)

        coverage = backend.coverage("Docs", files=False, chunks=4)
        assert (coverage.chunks, coverage.searchable_chunks, coverage.indexed) == (4, 3, 3)
        assert coverage.flagged == [("/docs/a.pdf", 2)]
        assert _chunk_filter() not in filters

        filters.clear()
        assert backend.coverage("Docs", files=False).chunks == 4
        assert _chunk_filter() in filters

    def test_coverage_creates_no_index(self, backend, client):
        client.delete_payload_index("Docs", "text")
        client.index_calls.clear()

        backend.coverage("Docs")
        backend.coverage("Docs", files=False)

        assert client.index_calls == []

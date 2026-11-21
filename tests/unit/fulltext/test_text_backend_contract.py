"""The ``TextBackend`` contract (RDR-027 Q1), run against every backend in
``conftest.TEXT_BACKENDS``.

Each test states behaviour a caller relies on through the protocol alone, so a
second backend passes the same tests without a backend-specific branch here.
"""

from __future__ import annotations

import pytest

from arcaneum.cli.errors import BadFilterError, ServiceDownError
from arcaneum.retrieval.text_backend import Coverage, TextBackend, TextPage
from arcaneum.schema.document import DualIndexDocument
from arcaneum.search.filters import parse_filter

CORPUS = "Docs"


def _doc(chunk_id: str, path: str, index: int, content: str) -> DualIndexDocument:
    return DualIndexDocument(
        id=chunk_id,
        content=content,
        file_path=path,
        filename=path.rsplit("/", 1)[-1],
        file_extension=".pdf",
        chunk_index=index,
        page_number=index + 1,
    )


@pytest.fixture
def corpus(text_backend):
    text_backend.ensure(CORPUS, "pdf")
    text_backend.add(
        CORPUS,
        [
            _doc("a0", "/docs/a.pdf", 0, "the quick brown fox"),
            _doc("a1", "/docs/a.pdf", 1, "jumps over the lazy dog"),
            _doc("b0", "/docs/b.pdf", 0, "a quick note on foxes"),
        ],
    )
    return text_backend


def _ids(page: TextPage) -> set:
    return {str(hit["id"]) for hit in page.hits}


def test_the_backend_satisfies_the_protocol(text_backend):
    assert isinstance(text_backend, TextBackend)


def test_ensure_creates_the_corpus_once(text_backend):
    assert not text_backend.exists(CORPUS)
    text_backend.ensure(CORPUS, "pdf")
    text_backend.ensure(CORPUS, "pdf")
    assert text_backend.exists(CORPUS)


def test_search_returns_hits_with_id_text_and_score(corpus):
    page = corpus.search(CORPUS, "quick", limit=10)

    assert _ids(page) == {"a0", "b0"}
    for hit in page.hits:
        assert hit["content"]
        assert hit["file_path"].startswith("/docs/")
        assert isinstance(hit["_rankingScore"], float)
    assert page.estimated_total_hits == 2


def test_search_applies_a_neutral_filter(corpus):
    page = corpus.search(CORPUS, "quick", filter=parse_filter("file_path=/docs/b.pdf"), limit=10)
    assert _ids(page) == {"b0"}


def test_search_honours_limit_and_offset(corpus):
    first = corpus.search(CORPUS, "", limit=2, offset=0)
    rest = corpus.search(CORPUS, "", limit=2, offset=2)
    assert len(first.hits) == 2
    assert len(rest.hits) == 1
    assert _ids(first) | _ids(rest) == {"a0", "a1", "b0"}


def test_an_unfilterable_field_is_bad_filter(corpus):
    with pytest.raises(BadFilterError) as exc_info:
        corpus.search(CORPUS, "quick", filter=parse_filter("nonexistent=1"))
    assert exc_info.value.code == "bad-filter"
    assert "nonexistent" in str(exc_info.value)


def test_get_document_reads_one_chunk_or_none(corpus):
    assert corpus.get_document(CORPUS, "a1")["content"] == "jumps over the lazy dog"
    assert corpus.get_document(CORPUS, "missing") is None


def test_file_documents_lists_a_files_chunks(corpus):
    documents, total = corpus.file_documents(CORPUS, "/docs/a.pdf", ["id", "chunk_index"])
    assert sorted(d["id"] for d in documents) == ["a0", "a1"]
    assert total == 2


def test_has_file_reads_back_whether_a_path_has_chunks(corpus):
    assert corpus.has_file(CORPUS, "/docs/a.pdf")
    assert not corpus.has_file(CORPUS, "/docs/none.pdf")


def test_delete_files_removes_every_chunk_of_a_file(corpus):
    corpus.delete_files(CORPUS, ["/docs/a.pdf"])

    assert _ids(corpus.search(CORPUS, "", limit=10)) == {"b0"}
    assert corpus.coverage(CORPUS).files == {"/docs/b.pdf": 1}


def test_rename_moves_chunks_to_the_new_path(corpus):
    corpus.rename(CORPUS, {("/docs/a.pdf", "/moved/c.pdf"): ["a0", "a1"]})

    moved = corpus.get_document(CORPUS, "a0")
    assert moved["file_path"] == "/moved/c.pdf"
    assert moved["filename"] == "c.pdf"
    assert moved["content"] == "the quick brown fox"
    assert corpus.coverage(CORPUS).files == {"/moved/c.pdf": 2, "/docs/b.pdf": 1}


def test_coverage_counts_chunks_and_files(corpus):
    coverage = corpus.coverage(CORPUS)
    assert coverage == Coverage(
        chunks=3, searchable_chunks=3, files={"/docs/a.pdf": 2, "/docs/b.pdf": 1}
    )


def test_corpora_counts_each_corpus(corpus):
    assert corpus.corpora() == {CORPUS: 3}


def test_drop_removes_the_corpus(corpus):
    corpus.drop(CORPUS)
    corpus.drop(CORPUS)
    assert not corpus.exists(CORPUS)
    assert CORPUS not in corpus.corpora()


def test_health_passes_when_the_service_answers(text_backend):
    text_backend.health()


def test_a_down_service_is_service_down(down_text_backend):
    with pytest.raises(ServiceDownError) as exc_info:
        down_text_backend.health()
    assert exc_info.value.code == "service-down"


def test_check_filter_accepts_the_neutral_dialect(text_backend):
    text_backend.check_filter(None)
    text_backend.check_filter(parse_filter("filename=a.pdf"))

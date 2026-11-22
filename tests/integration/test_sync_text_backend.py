"""Sync writes only to the corpus's text backend (RDR-027 Q7), on a real Qdrant 1.20.

A qdrant corpus's text lives in its chunk points, so a file changed,
renamed and removed moves its chunks with no MeiliSearch call, and the text
index still covers every chunk. The MeiliSearch client sync could reach
fails the test on any use.

Requires QDRANT_URL (default http://localhost:6333) at 1.20 or later; skips
otherwise. Each test makes a corpus of its own and removes it.
"""

from __future__ import annotations

import os
import uuid
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from arcaneum.cli import corpus as corpus_mod
from arcaneum.cli import sync as sync_mod
from arcaneum.indexing.collection_metadata import metadata_exclusion_filter
from arcaneum.retrieval.qdrant_text import QdrantTextBackend

pytestmark = pytest.mark.integration

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


class _NoMeili:
    """A MeiliSearch client that fails the test on any attribute access."""

    def __getattr__(self, name):
        raise AssertionError(f"MeiliSearch was used on a qdrant corpus: {name}")


class _FakeEmbeddingClient:
    def __init__(self, dims):
        self.dims = dims

    def embed(self, texts, model, max_internal_batch=None):
        return [[1.0] + [0.5] * (self.dims - 1) for _ in texts]


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
def qdrant_corpus(qdrant):
    names = []

    def make(corpus_type: str) -> str:
        name = f"arc_sync_it_{uuid.uuid4().hex[:10]}"
        names.append(name)
        with patch.object(corpus_mod, "create_qdrant_client", return_value=qdrant):
            corpus_mod.create_corpus_command(
                name, corpus_type, "arctic-m", None, True, text_backend="qdrant"
            )
        return name

    yield make
    for name in names:
        if qdrant.collection_exists(name):
            qdrant.delete_collection(name)


def _sync(qdrant, corpus, source_dir, *, parity=False, output_json=True):
    """Run sync with MeiliSearch unreachable by construction; return get_meili_client's mock."""
    from arcaneum.embeddings.client import EMBEDDING_MODELS

    dims = EMBEDDING_MODELS["arctic-m"]["dimensions"]
    get_meili_client = MagicMock(return_value=_NoMeili())
    targets = {
        "create_qdrant_client": MagicMock(return_value=qdrant),
        "get_meili_client": get_meili_client,
        "EmbeddingClient": MagicMock(return_value=_FakeEmbeddingClient(dims)),
        "_report_embedding_backend": MagicMock(),
        "_finalize_embedding_clients": MagicMock(),
        "interaction_logger": MagicMock(),
    }
    with ExitStack() as stack:
        for name, value in targets.items():
            stack.enter_context(patch.object(sync_mod, name, value))
        sync_mod._sync_directory_locked(
            corpus,
            (str(source_dir),),
            None,
            None,
            None,
            False,  # force
            False,
            None,
            None,
            True,
            None,
            False,
            output_json,
            parity=parity,
        )
    return get_meili_client


def _chunks_by_path(qdrant, corpus):
    """Each file's chunk points: path -> {id: text}."""
    by_path = {}
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=corpus,
            scroll_filter=metadata_exclusion_filter(),
            limit=256,
            offset=offset,
            with_payload=["file_path", "text"],
            with_vectors=False,
        )
        for p in points:
            if "text" in (p.payload or {}):
                by_path.setdefault(p.payload["file_path"], {})[str(p.id)] = p.payload["text"]
        if offset is None:
            return by_path


def _assert_covered(qdrant, corpus):
    coverage = QdrantTextBackend(qdrant).coverage(corpus)
    assert coverage.complete, coverage.to_data()


def test_change_rename_and_remove_make_no_meilisearch_call(qdrant, qdrant_corpus, tmp_path):
    corpus = qdrant_corpus("markdown")
    source = tmp_path / "notes"
    source.mkdir()
    a = source / "a.md"
    b = source / "b.md"
    a.write_text("# Alpha\n\nThe write-ahead log flushes before the checkpoint.\n")
    b.write_text("# Beta\n\nA checkpoint truncates the log afterwards.\n")
    a_path, b_path = str(a.absolute()), str(b.absolute())

    assert not _sync(qdrant, corpus, source).called
    first = _chunks_by_path(qdrant, corpus)
    assert set(first) == {a_path, b_path}
    _assert_covered(qdrant, corpus)

    # Changed: the old chunks are replaced, not kept beside the new ones.
    a.write_text("# Alpha\n\nSchema evolution rewrites the ledger entirely.\n")
    os.utime(a, (a.stat().st_atime, a.stat().st_mtime + 10))
    assert not _sync(qdrant, corpus, source).called
    changed = _chunks_by_path(qdrant, corpus)
    assert any("Schema evolution" in t for t in changed[a_path].values())
    assert not any("write-ahead" in t for t in changed[a_path].values())
    _assert_covered(qdrant, corpus)

    # Renamed: the same chunks move to the new path.
    c = source / "c.md"
    b.rename(c)
    c_path = str(c.absolute())
    assert not _sync(qdrant, corpus, source).called
    renamed = _chunks_by_path(qdrant, corpus)
    assert b_path not in renamed
    assert set(renamed[c_path]) == set(first[b_path])
    _assert_covered(qdrant, corpus)

    # Removed: parity drops its chunks.
    a.unlink()
    assert not _sync(qdrant, corpus, source, parity=True).called
    removed = _chunks_by_path(qdrant, corpus)
    assert set(removed) == {c_path}
    _assert_covered(qdrant, corpus)


def test_sync_and_parity_of_a_qdrant_corpus_print_no_meilisearch(
    qdrant, qdrant_corpus, tmp_path, capsys
):
    corpus = qdrant_corpus("markdown")
    source = tmp_path / "notes"
    source.mkdir()
    (source / "a.md").write_text("# Alpha\n\nThe write-ahead log flushes before the checkpoint.\n")

    for parity in (False, True):
        assert not _sync(qdrant, corpus, source, parity=parity, output_json=False).called
        out = capsys.readouterr().out
        assert f"Qdrant collection: {corpus}" in out
        assert "MeiliSearch" not in out


def _write_pdf(path, text):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_removing_a_canonical_pdf_promotes_its_alias_with_no_meilisearch_call(
    qdrant, qdrant_corpus, tmp_path
):
    corpus = qdrant_corpus("pdf")
    source = tmp_path / "papers"
    source.mkdir()
    canonical = source / "a.pdf"
    _write_pdf(canonical, "The write-ahead log flushes before the checkpoint completes.")
    alias = source / "b.pdf"
    alias.write_bytes(canonical.read_bytes())
    canonical_path, alias_path = str(canonical.absolute()), str(alias.absolute())

    assert not _sync(qdrant, corpus, source).called
    first = _chunks_by_path(qdrant, corpus)
    assert set(first) == {canonical_path}
    _assert_covered(qdrant, corpus)

    canonical.unlink()
    assert not _sync(qdrant, corpus, source, parity=True).called
    promoted = _chunks_by_path(qdrant, corpus)
    assert set(promoted) == {alias_path}
    assert set(promoted[alias_path]) == set(first[canonical_path])
    _assert_covered(qdrant, corpus)

"""Sync writes only to the corpus's text backend (RDR-027 Q7).

A qdrant corpus's text lives in its chunk points, so sync builds no
MeiliSearch client for it; a meili corpus still writes both stores and is
refused when MeiliSearch is down. tests/integration/test_sync_text_backend.py
runs the same against a real Qdrant.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from arcaneum.cli import sync as sync_mod
from arcaneum.cli.errors import ResourceNotFoundError
from arcaneum.indexing.dual_indexer import DualIndexer
from arcaneum.schema.document import DualIndexDocument


class _NoMeili:
    """A MeiliSearch client that fails the test on any attribute access."""

    def __getattr__(self, name):
        raise AssertionError(f"MeiliSearch was used on a qdrant corpus: {name}")


class _FakeEmbeddingClient:
    def embed(self, texts, model, max_internal_batch=None):
        return [[0.0, 0.0] for _ in texts]


def _doc() -> DualIndexDocument:
    return DualIndexDocument(
        id="00000000-0000-0000-0000-000000000001",
        content="The write-ahead log flushes before the checkpoint.",
        file_path="/notes/a.md",
        filename="a.md",
        file_extension=".md",
        chunk_index=0,
        vectors={"stub": [1.0, 0.0]},
    )


def test_index_batch_writes_only_qdrant_for_a_qdrant_corpus():
    qdrant = MagicMock()
    indexer = DualIndexer(qdrant, _NoMeili(), "Notes", "Notes", text_backend="qdrant")

    assert indexer.index_batch([_doc()]) == (1, 0)
    qdrant.upsert.assert_called()


def test_index_batch_still_writes_meilisearch_for_a_meili_corpus():
    meili = MagicMock()
    indexer = DualIndexer(MagicMock(), meili, "Notes", "Notes", text_backend="meili")

    assert indexer.index_batch([_doc()]) == (1, 1)
    meili.add_documents_sync.assert_called_once()


def _sync(tmp_path, text_backend, *, meili, already_indexed=False, parity=False, output_json=True):
    """Sync one markdown file; return (get_meili_client mock, _plan_parity_backfill mock, qdrant)."""
    source = tmp_path / "note.md"
    source.write_text("# Note\n\nThe write-ahead log flushes before the checkpoint.\n")
    source_path = str(source.resolve())
    qdrant = MagicMock()
    manifests = MagicMock()
    manifests.get_unindexed_files.return_value = ([source.resolve()], [])
    manifests._get_indexed_file_paths_set.return_value = {source_path} if already_indexed else set()
    manifests.find_file_by_content_hash.return_value = []
    manifests.last_stale_policy_paths = []
    metadata = {"model": "arctic-m"}
    if text_backend is not None:
        metadata["text_backend"] = text_backend

    get_meili_client = MagicMock(return_value=meili)
    plan_backfill = MagicMock(return_value=([], []))
    targets = {
        "create_qdrant_client": MagicMock(return_value=qdrant),
        "get_meili_client": get_meili_client,
        "get_collection_type": MagicMock(return_value="markdown"),
        "get_collection_metadata": MagicMock(return_value=metadata),
        "_maybe_backfill_legacy_prompt_policy": MagicMock(
            side_effect=lambda q, c, t, m, metadata, o: (metadata, [])
        ),
        "MetadataBasedSync": MagicMock(return_value=manifests),
        "_ensure_file_manifests": MagicMock(),
        "_plan_parity_backfill": plan_backfill,
        "EmbeddingClient": MagicMock(return_value=_FakeEmbeddingClient()),
        "_report_embedding_backend": MagicMock(),
        "_finalize_embedding_clients": MagicMock(),
        "chunk_markdown_file": MagicMock(
            return_value=[{"text": "The write-ahead log flushes.", "metadata": {}}]
        ),
        "interaction_logger": MagicMock(),
    }
    with ExitStack() as stack:
        for name, value in targets.items():
            stack.enter_context(patch.object(sync_mod, name, value))
        sync_mod._sync_directory_locked(
            "Notes",
            (str(source),),
            None,
            "arctic-m",
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
    return get_meili_client, plan_backfill, qdrant


def test_a_qdrant_corpus_replaces_an_edited_file_with_no_meilisearch_client(tmp_path):
    get_meili_client, _plan, qdrant = _sync(
        tmp_path, "qdrant", meili=_NoMeili(), already_indexed=True
    )

    get_meili_client.assert_not_called()
    qdrant.delete.assert_called()  # the replaced chunks
    qdrant.upsert.assert_called()


def test_a_qdrant_corpus_parity_plans_no_backfill(tmp_path):
    get_meili_client, plan_backfill, _qdrant = _sync(
        tmp_path, "qdrant", meili=_NoMeili(), parity=True
    )

    get_meili_client.assert_not_called()
    plan_backfill.assert_not_called()


def _meili_up():
    meili = MagicMock()
    meili.health_check.return_value = True
    meili.index_exists.return_value = True
    meili.get_all_file_paths.return_value = set()
    return meili


def test_a_meili_corpus_still_writes_both_stores(tmp_path):
    for text_backend in ("meili", None):
        meili = _meili_up()
        _sync(tmp_path, text_backend, meili=meili, already_indexed=True)

        meili.delete_documents_by_file_paths.assert_called_once()
        meili.add_documents_sync.assert_called_once()


def test_a_meili_corpus_parity_still_plans_its_backfills(tmp_path):
    _get, plan_backfill, _qdrant = _sync(tmp_path, "meili", meili=_meili_up(), parity=True)

    plan_backfill.assert_called_once()


def test_the_index_banner_names_only_qdrant_for_a_qdrant_corpus():
    lines = sync_mod._index_banner_lines("Notes", "qdrant")

    assert not any("MeiliSearch" in line for line in lines)
    assert any("Qdrant collection: Notes" in line for line in lines)


def test_the_index_banner_still_names_both_stores_for_a_meili_corpus():
    lines = sync_mod._index_banner_lines("Notes", "meili")

    assert any("Qdrant collection: Notes" in line for line in lines)
    assert any("MeiliSearch index: Notes" in line for line in lines)


@pytest.mark.parametrize("parity", [False, True])
def test_a_qdrant_corpus_sync_prints_no_meilisearch(tmp_path, capsys, parity):
    _sync(tmp_path, "qdrant", meili=_NoMeili(), parity=parity, output_json=False)

    out = capsys.readouterr().out
    assert "Qdrant collection: Notes" in out
    assert "MeiliSearch" not in out


def test_a_meili_corpus_sync_still_names_meilisearch(tmp_path, capsys):
    _sync(tmp_path, "meili", meili=_meili_up(), parity=True, output_json=False)

    out = capsys.readouterr().out
    assert "MeiliSearch index: Notes" in out
    assert "Checking index parity between Qdrant and MeiliSearch" in out
    assert "Total in MeiliSearch" in out


def test_a_meili_corpus_is_refused_when_meilisearch_is_down(tmp_path):
    meili = _meili_up()
    meili.health_check.return_value = False

    with pytest.raises(ResourceNotFoundError, match="MeiliSearch server not available"):
        _sync(tmp_path, "meili", meili=meili)


def test_removing_paths_of_a_qdrant_corpus_deletes_only_qdrant():
    qdrant = MagicMock()
    manager = MagicMock()

    removed = sync_mod._remove_indexed_paths(qdrant, None, manager, "Notes", "markdown", ["/a.md"])

    assert removed == 1
    qdrant.delete.assert_called_once()
    manager.delete_file_manifest.assert_called_once_with("Notes", "/a.md")


def test_renames_of_a_qdrant_corpus_are_confirmed_from_qdrant_alone():
    qdrant = MagicMock()
    qdrant.scroll.return_value = ([SimpleNamespace(id="c1"), SimpleNamespace(id="c2")], None)

    moved, confirmed = sync_mod._handle_renames_meili([("/a.md", "/b.md")], qdrant, None, "Notes")

    assert (moved, confirmed) == (2, [("/a.md", "/b.md")])

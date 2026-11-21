"""Sync flags a qdrant corpus's zero-token chunks with ``text_tokens: False`` (RDR-027 Q8).

A real ``DualIndexer`` writes to a recording Qdrant mock, so the payload read
here is the one sync upserts.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from arcaneum.cli import sync as sync_mod

_MARKUP = "### ** `` ``` __ ___"
_TOKENFUL = "## Write-ahead log\n\nThe log is flushed before the page."


class _FakeEmbeddingClient:
    def embed(self, texts, model, max_internal_batch=None):
        return [[0.0, 0.0] for _ in texts]


def _sync_chunks(tmp_path, text_backend):
    """Sync one markdown file of a markup chunk and a tokenful chunk; return text payloads."""
    source = tmp_path / "note.md"
    source.write_text(f"{_MARKUP}\n\n{_TOKENFUL}\n")
    qdrant = MagicMock()
    meili = MagicMock()
    meili.health_check.return_value = True
    meili.index_exists.return_value = True
    manifests = MagicMock()
    manifests.get_unindexed_files.return_value = ([source.resolve()], [])
    manifests._get_indexed_file_paths_set.return_value = set()
    manifests.find_file_by_content_hash.return_value = []
    metadata = {"model": "arctic-m"}
    if text_backend is not None:
        metadata["text_backend"] = text_backend

    targets = {
        "create_qdrant_client": MagicMock(return_value=qdrant),
        "get_meili_client": MagicMock(return_value=meili),
        "get_collection_type": MagicMock(return_value="markdown"),
        "get_collection_metadata": MagicMock(return_value=metadata),
        "_maybe_backfill_legacy_prompt_policy": MagicMock(
            side_effect=lambda q, c, t, m, metadata, o: (metadata, [])
        ),
        "MetadataBasedSync": MagicMock(return_value=manifests),
        "_ensure_file_manifests": MagicMock(),
        "EmbeddingClient": MagicMock(return_value=_FakeEmbeddingClient()),
        "_report_embedding_backend": MagicMock(),
        "_finalize_embedding_clients": MagicMock(),
        "chunk_markdown_file": MagicMock(
            return_value=[{"text": _MARKUP, "metadata": {}}, {"text": _TOKENFUL, "metadata": {}}]
        ),
        "interaction_logger": MagicMock(),
    }
    with ExitStack() as stack:
        for name, value in targets.items():
            stack.enter_context(patch.object(sync_mod, name, value))
        try:
            sync_mod._sync_directory_locked(
                "Notes",
                (str(source),),
                None,
                "arctic-m",
                None,
                True,  # force
                False,
                None,
                None,
                True,
                None,
                False,
                True,  # output_json
            )
        except SystemExit:
            pass
    return {
        p.payload["text"]: p.payload
        for c in qdrant.upsert.call_args_list
        for p in c.kwargs.get("points", [])
        if "text" in (p.payload or {})
    }


def test_a_qdrant_corpus_flags_only_its_markup_chunk(tmp_path):
    payloads = _sync_chunks(tmp_path, "qdrant")

    assert set(payloads) == {_MARKUP, _TOKENFUL}
    assert payloads[_MARKUP]["text_tokens"] is False
    assert "text_tokens" not in payloads[_TOKENFUL]


def test_a_meili_corpus_carries_no_flag(tmp_path):
    for text_backend in ("meili", None):
        payloads = _sync_chunks(tmp_path, text_backend)

        assert set(payloads) == {_MARKUP, _TOKENFUL}
        assert all("text_tokens" not in p for p in payloads.values())

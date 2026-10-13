"""A forced re-index keeps a file's old chunks until its new ones are ready.

Deleting first left an interrupted or failed file with no chunks at all, and
with its stored metadata gone `--from-index` could no longer select it.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from arcaneum.cli import sync as sync_mod


class _Recorder:
    def __init__(self):
        self.events = []


class _FakeEmbeddingClient:
    def __init__(self, recorder, error=None):
        self.recorder = recorder
        self.error = error

    def embed(self, texts, model, max_internal_batch=None):
        self.recorder.events.append("embed")
        if self.error:
            raise self.error
        return [[0.0, 0.0] for _ in texts]


class _FakeDualIndexer:
    def __init__(self, recorder):
        self.recorder = recorder

    def delete_by_file_path(self, file_path):
        self.recorder.events.append("delete_qdrant")
        return 0, 0

    def index_batch(self, documents, wait=True):
        self.recorder.events.append("index_batch")
        return len(documents), len(documents)


def _force_sync(tmp_path, *, chunks, embed_error=None):
    source = tmp_path / "note.md"
    source.write_text("# Note\n\nbody\n")
    recorder = _Recorder()

    meili = MagicMock()
    meili.health_check.return_value = True
    meili.index_exists.return_value = True
    meili.delete_documents_by_file_paths.side_effect = lambda *a, **k: recorder.events.append(
        "delete_meili"
    )
    manifests = MagicMock()
    manifests.delete_file_manifest.side_effect = lambda *a, **k: recorder.events.append(
        "delete_manifest"
    )

    targets = {
        "create_qdrant_client": MagicMock(return_value=MagicMock()),
        "get_meili_client": MagicMock(return_value=meili),
        "get_collection_type": MagicMock(return_value="markdown"),
        "get_collection_metadata": MagicMock(return_value={"model": "arctic-m"}),
        "_maybe_backfill_legacy_prompt_policy": MagicMock(
            side_effect=lambda q, c, t, m, metadata, o: (metadata, [])
        ),
        "MetadataBasedSync": MagicMock(return_value=manifests),
        "_ensure_file_manifests": MagicMock(),
        "EmbeddingClient": MagicMock(
            return_value=_FakeEmbeddingClient(recorder, error=embed_error)
        ),
        "_report_embedding_backend": MagicMock(),
        "_finalize_embedding_clients": MagicMock(),
        "DualIndexer": MagicMock(return_value=_FakeDualIndexer(recorder)),
        "chunk_markdown_file": MagicMock(return_value=chunks),
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
    return recorder.events


def test_force_deletes_old_chunks_only_after_embedding(tmp_path):
    events = _force_sync(tmp_path, chunks=[{"text": "body", "metadata": {}}])

    assert "index_batch" in events
    first_delete = events.index("delete_qdrant")
    assert events.index("embed") < first_delete < events.index("index_batch")
    assert events.index("delete_meili") < events.index("index_batch")


def test_force_keeps_old_chunks_when_embedding_fails(tmp_path):
    events = _force_sync(
        tmp_path,
        chunks=[{"text": "body", "metadata": {}}],
        embed_error=RuntimeError("embedding failed"),
    )

    assert "embed" in events
    assert "delete_qdrant" not in events
    assert "delete_meili" not in events
    assert "delete_manifest" not in events


def test_force_still_clears_old_chunks_when_file_yields_no_text(tmp_path):
    # The file is recorded with zero chunks, so its old chunks must go too.
    events = _force_sync(tmp_path, chunks=[])

    assert "delete_qdrant" in events
    assert "delete_meili" in events

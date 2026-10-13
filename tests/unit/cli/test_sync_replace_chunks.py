"""Re-indexing a file replaces its old chunks once its new ones are ready.

Force deleted first, which left an interrupted or failed file with no chunks
and, with its stored metadata gone, unselectable by `--from-index`. Normal
sync never deleted at all, so an edited file kept its old chunks beside the
new ones.
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


def _sync(
    tmp_path,
    *,
    chunks=(),
    force=True,
    embed_error=None,
    already_indexed=False,
    corpus_type="markdown",
    content_paths=None,
    git_version=False,
):
    source = tmp_path / ("paper.pdf" if corpus_type == "pdf" else "note.md")
    source.write_text("# Note\n\nbody\n")
    source_path = str(source.resolve())
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
    manifests.get_unindexed_files.return_value = ([source.resolve()], [])
    manifests._get_indexed_file_paths_set.return_value = {source_path} if already_indexed else set()
    if content_paths is not None:
        # First lookup finds the content already indexed elsewhere; the second
        # confirms this path registered as its alias.
        manifests.find_file_by_content_hash.side_effect = [
            content_paths,
            [*content_paths, source_path],
        ]
    else:
        manifests.find_file_by_content_hash.return_value = []

    targets = {
        "create_qdrant_client": MagicMock(return_value=MagicMock()),
        "get_meili_client": MagicMock(return_value=meili),
        "get_collection_type": MagicMock(return_value=corpus_type),
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
        "chunk_markdown_file": MagicMock(return_value=list(chunks)),
        "chunk_pdf_file": MagicMock(return_value=list(chunks)),
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
                force,
                False,
                None,
                None,
                True,
                None,
                False,
                True,  # output_json
                git_version=git_version,
            )
        except SystemExit:
            pass
    return recorder.events


_BODY = [{"text": "body", "metadata": {}}]


def test_force_deletes_old_chunks_only_after_embedding(tmp_path):
    events = _sync(tmp_path, chunks=_BODY)

    assert "index_batch" in events
    first_delete = events.index("delete_qdrant")
    assert events.index("embed") < first_delete < events.index("index_batch")
    assert events.index("delete_meili") < events.index("index_batch")


def test_force_keeps_old_chunks_when_embedding_fails(tmp_path):
    events = _sync(tmp_path, chunks=_BODY, embed_error=RuntimeError("embedding failed"))

    assert "embed" in events
    assert "delete_qdrant" not in events
    assert "delete_meili" not in events
    assert "delete_manifest" not in events


def test_force_still_clears_old_chunks_when_file_yields_no_text(tmp_path):
    # The file is recorded with zero chunks, so its old chunks must go too.
    events = _sync(tmp_path, chunks=[])

    assert "delete_qdrant" in events
    assert "delete_meili" in events


def test_edited_file_replaces_its_old_chunks_after_embedding(tmp_path):
    events = _sync(tmp_path, chunks=_BODY, force=False, already_indexed=True)

    assert "index_batch" in events
    first_delete = events.index("delete_qdrant")
    assert events.index("embed") < first_delete < events.index("index_batch")
    assert events.index("delete_meili") < events.index("index_batch")


def test_new_file_has_no_old_chunks_to_delete(tmp_path):
    events = _sync(tmp_path, chunks=_BODY, force=False, already_indexed=False)

    assert "index_batch" in events
    assert "delete_qdrant" not in events
    assert "delete_meili" not in events


def test_edited_pdf_that_became_an_alias_drops_its_old_chunks(tmp_path):
    # Its own stale chunks must not outlive its registration as an alias.
    events = _sync(
        tmp_path,
        force=False,
        already_indexed=True,
        corpus_type="pdf",
        content_paths=["/papers/original.pdf"],
    )

    assert "embed" not in events
    assert "delete_qdrant" in events
    assert "delete_meili" in events


def test_git_version_keeps_other_versions_of_an_edited_file(tmp_path):
    # --git-version keeps every indexed version under the same file_path.
    events = _sync(tmp_path, chunks=_BODY, force=False, already_indexed=True, git_version=True)

    assert "index_batch" in events
    assert "delete_qdrant" not in events
    assert "delete_meili" not in events

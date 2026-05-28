"""Regression tests for the markdown indexing pipeline.

Includes C2 coverage: streaming mode must count successfully-uploaded files in
stats["files"]. The bug was that streaming `_process_single_markdown` returns an
empty points list (it uploads inside the embed callback), and `index_directory`
gated `stats["files"] += 1` on `if points:` — never true in streaming mode — so
the default streaming path always reported 0 files.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from arcaneum.indexing.markdown.chunker import SemanticMarkdownChunker
from arcaneum.indexing.markdown.pipeline import MarkdownIndexingPipeline


def _pipeline_with_sync(sync):
    pipeline = MarkdownIndexingPipeline(
        qdrant_client=Mock(),
        embedding_client=Mock(),
    )
    pipeline.sync = sync
    return pipeline


def test_markdown_pipeline_uses_default_embedding_batch_size_for_none():
    pipeline = MarkdownIndexingPipeline(
        qdrant_client=Mock(),
        embedding_client=Mock(),
        embedding_batch_size=None,
    )

    assert pipeline.embedding_batch_size == 128


def test_markdown_pipeline_preserves_explicit_embedding_batch_size():
    pipeline = MarkdownIndexingPipeline(
        qdrant_client=Mock(),
        embedding_client=Mock(),
        embedding_batch_size=64,
    )

    assert pipeline.embedding_batch_size == 64


def test_duplicate_markdown_content_returns_empty_result_tuple(tmp_path):
    markdown_file = tmp_path / "copy.md"
    markdown_file.write_text("# Same content\n", encoding="utf-8")
    existing_file = tmp_path / "original.md"
    existing_file.write_text("# Same content\n", encoding="utf-8")

    sync = Mock()
    sync.find_file_by_content_hash.return_value = [str(existing_file.absolute())]
    sync.filter_existing_paths.return_value = [str(existing_file.absolute())]
    sync.add_alternate_path.return_value = 1

    result = _pipeline_with_sync(sync)._process_single_markdown(
        markdown_file,
        "docs",
        "arctic-m",
        {},
        SemanticMarkdownChunker(),
        1,
        False,
        1,
        1,
    )

    assert result == ([], 0, None)
    sync.add_alternate_path.assert_called_once()


def test_renamed_markdown_content_returns_empty_result_tuple(tmp_path):
    markdown_file = tmp_path / "renamed.md"
    markdown_file.write_text("# Moved content\n", encoding="utf-8")
    old_path = str((tmp_path / "old.md").absolute())

    sync = Mock()
    sync.find_file_by_content_hash.return_value = [old_path]
    sync.filter_existing_paths.return_value = []
    sync.handle_renames.return_value = 2

    result = _pipeline_with_sync(sync)._process_single_markdown(
        markdown_file,
        "docs",
        "arctic-m",
        {},
        SemanticMarkdownChunker(),
        1,
        False,
        1,
        1,
    )

    assert result == ([], 0, None)
    sync.handle_renames.assert_called_once()


class FakeEmbeddingClient:
    """Embedding client that drives the real streaming callback path."""

    def __init__(self):
        self.use_gpu = False

    def is_model_cached(self, model_name):
        return True

    def get_model(self, model_name):
        return object()

    def _clear_gpu_cache(self):  # pragma: no cover - not used (use_gpu False)
        pass

    def embed_parallel(self, texts, model_name, max_workers=None, batch_size=None,
                       progress_callback=None, on_batch_complete=None,
                       accumulate=True):
        embeddings = [[0.1, 0.2] for _ in texts]
        if not accumulate and on_batch_complete is not None:
            # Stream every text in a single batch through the callback.
            on_batch_complete(0, 0, embeddings)
            return None
        return embeddings


class FakeQdrant:
    """Qdrant stand-in: empty collection, records upserts."""

    def __init__(self):
        self.upserted = []

    def get_collection(self, _name):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=2))
            ),
            points_count=0,
        )

    def scroll(self, **kwargs):
        return ([], None)

    def upsert(self, collection_name, points, **kwargs):
        self.upserted.extend(points)


@pytest.fixture
def md_file(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Heading\n\nSome markdown body content for chunking.\n")
    return f


def _make_pipeline(streaming):
    qdrant = FakeQdrant()
    pipeline = MarkdownIndexingPipeline(
        qdrant_client=qdrant,
        embedding_client=FakeEmbeddingClient(),
        streaming=streaming,
        file_workers=1,
    )
    return pipeline, qdrant


def test_streaming_index_directory_counts_uploaded_files(md_file):
    """Streaming mode must report files=1 after a successful upload."""
    pipeline, qdrant = _make_pipeline(streaming=True)

    model_config = {"vector_name": None, "chunk_size": 512, "chunk_overlap": 50}
    stats = pipeline.index_directory(
        markdown_dir=md_file.parent,
        collection_name="md",
        model_name="stella",
        model_config=model_config,
        force_reindex=True,
        file_list=[md_file],
    )

    assert qdrant.upserted, "streaming mode should have uploaded chunks"
    assert stats["files"] == 1
    assert stats["chunks"] > 0
    assert stats["errors"] == 0


def test_non_streaming_index_directory_still_counts_files(md_file):
    """Non-streaming behavior must remain intact."""
    pipeline, qdrant = _make_pipeline(streaming=False)

    model_config = {"vector_name": None, "chunk_size": 512, "chunk_overlap": 50}
    stats = pipeline.index_directory(
        markdown_dir=md_file.parent,
        collection_name="md",
        model_name="stella",
        model_config=model_config,
        force_reindex=True,
        file_list=[md_file],
    )

    assert stats["files"] == 1
    assert stats["chunks"] > 0
    assert stats["errors"] == 0

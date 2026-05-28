from unittest.mock import Mock

from arcaneum.indexing.markdown.chunker import SemanticMarkdownChunker
from arcaneum.indexing.markdown.pipeline import MarkdownIndexingPipeline


def _pipeline_with_sync(sync):
    pipeline = MarkdownIndexingPipeline(
        qdrant_client=Mock(),
        embedding_client=Mock(),
    )
    pipeline.sync = sync
    return pipeline


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

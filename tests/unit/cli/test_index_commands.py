"""CLI tests for index commands (arc index pdf/code/markdown).

These tests verify that the index command modules expose the expected
callables and signatures. Behavior-level coverage lives with the pipeline
modules themselves.
"""

import inspect

import pytest


def _param_names(fn):
    return set(inspect.signature(fn).parameters)


def test_index_pdfs_command_signature():
    from arcaneum.cli.index_pdfs import index_pdfs_command

    params = _param_names(index_pdfs_command)
    assert ({'path'} & params or {'paths'} & params), "expected path/paths param"
    assert 'collection' in params
    assert 'model' in params


def test_index_source_command_signature():
    from arcaneum.cli.index_source import index_source_command

    params = _param_names(index_source_command)
    assert ({'path'} & params or {'paths'} & params), "expected path/paths param"
    assert 'collection' in params
    assert 'model' in params


def test_index_markdown_command_signature():
    from arcaneum.cli.index_markdown import index_markdown_command

    params = _param_names(index_markdown_command)
    assert ({'path'} & params or {'paths'} & params), "expected path/paths param"
    assert 'collection' in params
    assert 'model' in params


def test_index_markdown_exposes_store_command():
    from arcaneum.cli.index_markdown import store_command

    params = _param_names(store_command)
    assert {'file', 'collection', 'model'}.issubset(params)


def test_index_text_module_exports_expected_entrypoint():
    """index_text module must expose one of the known entrypoints."""
    from arcaneum.cli import index_text

    assert any(
        hasattr(index_text, name)
        for name in ('index_text_pdf_command', 'index_text_group')
    ), "index_text module missing both known entrypoints"


def test_fulltext_module_exports_expected_entrypoint():
    from arcaneum.cli import fulltext

    assert any(
        hasattr(fulltext, name) for name in ('fulltext', 'list_indexes')
    ), "fulltext module missing both known entrypoints"


def test_interaction_logger_singleton_has_required_methods():
    from arcaneum.cli.interaction_logger import interaction_logger

    assert callable(getattr(interaction_logger, 'start', None))
    assert callable(getattr(interaction_logger, 'finish', None))


class TestIndexingPipelines:
    """The pipeline modules referenced by the CLI must expose their entry classes."""

    def test_pdf_modules_expose_their_classes(self):
        from arcaneum.indexing.pdf import chunker, extractor

        assert hasattr(chunker, 'PDFChunker')
        assert hasattr(extractor, 'PDFExtractor')

    def test_markdown_pipeline_exposes_its_class(self):
        from arcaneum.indexing.markdown import pipeline

        assert hasattr(pipeline, 'MarkdownIndexingPipeline')

    def test_source_code_pipeline_exposes_its_class(self):
        from arcaneum.indexing import source_code_pipeline

        assert hasattr(source_code_pipeline, 'SourceCodeIndexer')

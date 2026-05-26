"""CLI tests for corpus management commands.

Tests for 'arc corpus' subcommands: create, list, delete, info, items, verify, parity, sync.
"""

import pytest


def test_corpus_module_exports_expected_commands():
    """Every documented 'arc corpus' subcommand function is importable."""
    from arcaneum.cli import corpus

    expected = {
        'create_corpus_command',
        'list_corpora_command',
        'delete_corpus_command',
        'corpus_info_command',
        'corpus_verify_command',
    }
    missing = [name for name in expected if not callable(getattr(corpus, name, None))]
    assert missing == [], f"corpus module missing exported commands: {missing}"


def test_corpus_module_uses_interaction_logger():
    """corpus commands rely on the shared interaction_logger singleton."""
    from arcaneum.cli.corpus import interaction_logger
    from arcaneum.cli.interaction_logger import interaction_logger as canonical

    # Must be the same singleton, not some unrelated object with matching attrs
    assert interaction_logger is canonical


class TestCorpusCollectionMetadata:
    """CollectionType enum covers all corpus types the CLI advertises."""

    def test_collection_type_values_include_pdf_code_markdown(self):
        from arcaneum.indexing.collection_metadata import CollectionType

        values = set(CollectionType.values())
        assert {'pdf', 'code', 'markdown'}.issubset(values), \
            f"CollectionType missing required values: {values}"


class TestCorpusDefaultModels:
    """Stable model defaults are inferred from corpus type."""

    def test_document_corpora_default_to_arctic_m(self):
        from arcaneum.cli.corpus import default_models_for_type

        assert default_models_for_type("pdf") == "arctic-m"
        assert default_models_for_type("markdown") == "arctic-m"

    def test_code_corpora_default_to_lightweight_code_model(self):
        from arcaneum.cli.corpus import default_models_for_type

        assert default_models_for_type("code") == "jina-code"


class TestCorpusMarkdownChunking:
    """Corpus sync markdown chunking should avoid embedding-time truncation."""

    def test_chunk_markdown_file_applies_hard_max_chars(self, tmp_path):
        from arcaneum.cli.sync import chunk_markdown_file

        marker = "TAIL_SENTINEL"
        file_path = tmp_path / "large.md"
        file_path.write_text("# Big\n\n" + ("A" * 900) + marker, encoding="utf-8")

        chunks = chunk_markdown_file(
            file_path,
            chunk_size=1000,
            chunk_overlap=10,
            hard_max_chars=300,
        )

        assert len(chunks) > 1
        assert all(len(chunk["text"]) <= 300 for chunk in chunks)
        assert any(marker in chunk["text"] for chunk in chunks)
        assert any(chunk["metadata"].get("hard_split") is True for chunk in chunks)

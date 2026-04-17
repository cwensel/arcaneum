"""CLI tests for corpus management commands.

Tests for 'arc corpus' subcommands: create, list, delete, info, items, verify, parity, sync.
"""

import inspect

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


def test_create_corpus_command_accepts_required_parameters():
    """create_corpus_command must accept name, corpus_type, models, output_json."""
    from arcaneum.cli.corpus import create_corpus_command

    params = set(inspect.signature(create_corpus_command).parameters)
    required = {'name', 'corpus_type', 'models', 'output_json'}
    assert required.issubset(params), f"missing params: {required - params}"


def test_list_corpora_command_accepts_output_json():
    from arcaneum.cli.corpus import list_corpora_command

    params = set(inspect.signature(list_corpora_command).parameters)
    assert 'output_json' in params


def test_delete_corpus_command_requires_name_and_confirm():
    from arcaneum.cli.corpus import delete_corpus_command

    params = set(inspect.signature(delete_corpus_command).parameters)
    assert {'name', 'confirm'}.issubset(params)


def test_corpus_info_command_requires_name_and_output_json():
    from arcaneum.cli.corpus import corpus_info_command

    params = set(inspect.signature(corpus_info_command).parameters)
    assert {'name', 'output_json'}.issubset(params)


def test_corpus_verify_command_requires_name_and_output_json():
    from arcaneum.cli.corpus import corpus_verify_command

    params = set(inspect.signature(corpus_verify_command).parameters)
    assert {'name', 'output_json'}.issubset(params)


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

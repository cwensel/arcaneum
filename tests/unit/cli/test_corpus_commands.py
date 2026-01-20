"""CLI tests for corpus management commands.

Tests for 'arc corpus' subcommands: create, list, delete, info, items, verify, parity, sync.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestCorpusCommandImports:
    """Test corpus command imports."""

    def test_corpus_module_imports(self):
        """Test that corpus module can be imported."""
        from arcaneum.cli import corpus
        # Check for key functions
        assert hasattr(corpus, 'create_corpus_command') or hasattr(corpus, 'list_corpora_command')

    def test_create_corpus_command_exists(self):
        """Test that create_corpus_command exists."""
        from arcaneum.cli.corpus import create_corpus_command
        assert callable(create_corpus_command)

    def test_list_corpora_command_exists(self):
        """Test that list_corpora_command exists."""
        from arcaneum.cli.corpus import list_corpora_command
        assert callable(list_corpora_command)

    def test_delete_corpus_command_exists(self):
        """Test that delete_corpus_command exists."""
        from arcaneum.cli.corpus import delete_corpus_command
        assert callable(delete_corpus_command)

    def test_corpus_info_command_exists(self):
        """Test that corpus_info_command exists."""
        from arcaneum.cli.corpus import corpus_info_command
        assert callable(corpus_info_command)

    def test_corpus_verify_command_exists(self):
        """Test that corpus_verify_command exists."""
        from arcaneum.cli.corpus import corpus_verify_command
        assert callable(corpus_verify_command)


class TestCreateCorpusCommandSignature:
    """Test create corpus command signature."""

    def test_create_corpus_command_signature(self):
        """Test create_corpus_command has expected parameters."""
        from arcaneum.cli.corpus import create_corpus_command
        import inspect

        sig = inspect.signature(create_corpus_command)
        param_names = list(sig.parameters.keys())

        assert 'name' in param_names
        assert 'corpus_type' in param_names
        assert 'models' in param_names
        assert 'output_json' in param_names


class TestListCorpusCommand:
    """Test list corpus command."""

    def test_list_corpora_command_signature(self):
        """Test list_corpora_command has expected parameters."""
        from arcaneum.cli.corpus import list_corpora_command
        import inspect

        sig = inspect.signature(list_corpora_command)
        param_names = list(sig.parameters.keys())

        assert 'output_json' in param_names


class TestDeleteCorpusCommand:
    """Test delete corpus command."""

    def test_delete_corpus_command_signature(self):
        """Test delete_corpus_command has expected parameters."""
        from arcaneum.cli.corpus import delete_corpus_command
        import inspect

        sig = inspect.signature(delete_corpus_command)
        param_names = list(sig.parameters.keys())

        assert 'name' in param_names
        assert 'confirm' in param_names


class TestCorpusInfoCommand:
    """Test corpus info command."""

    def test_corpus_info_command_signature(self):
        """Test corpus_info_command has expected parameters."""
        from arcaneum.cli.corpus import corpus_info_command
        import inspect

        sig = inspect.signature(corpus_info_command)
        param_names = list(sig.parameters.keys())

        assert 'name' in param_names
        assert 'output_json' in param_names


class TestCorpusVerifyCommand:
    """Test corpus verify command."""

    def test_corpus_verify_command_signature(self):
        """Test corpus_verify_command has expected parameters."""
        from arcaneum.cli.corpus import corpus_verify_command
        import inspect

        sig = inspect.signature(corpus_verify_command)
        param_names = list(sig.parameters.keys())

        assert 'name' in param_names
        assert 'output_json' in param_names


class TestCorpusInteractionLogger:
    """Test that corpus commands use interaction logging."""

    def test_interaction_logger_imported(self):
        """Test that interaction_logger is available in corpus module."""
        from arcaneum.cli.corpus import interaction_logger
        assert hasattr(interaction_logger, 'start')
        assert hasattr(interaction_logger, 'finish')


class TestCorpusUtilities:
    """Test corpus-related utilities are available."""

    def test_build_vectors_config_available(self):
        """Test that build_vectors_config is available."""
        from arcaneum.cli.utils import build_vectors_config
        assert callable(build_vectors_config)

    def test_get_model_dimensions_available(self):
        """Test that get_model_dimensions is available."""
        from arcaneum.cli.utils import get_model_dimensions
        assert callable(get_model_dimensions)

    def test_create_qdrant_client_available(self):
        """Test that create_qdrant_client is available."""
        from arcaneum.cli.utils import create_qdrant_client
        assert callable(create_qdrant_client)

    def test_create_meili_client_available(self):
        """Test that create_meili_client is available."""
        from arcaneum.cli.utils import create_meili_client
        assert callable(create_meili_client)


class TestCorpusCollectionMetadata:
    """Test collection metadata utilities."""

    def test_set_collection_metadata_available(self):
        """Test that set_collection_metadata is available."""
        from arcaneum.indexing.collection_metadata import set_collection_metadata
        assert callable(set_collection_metadata)

    def test_get_collection_metadata_available(self):
        """Test that get_collection_metadata is available."""
        from arcaneum.indexing.collection_metadata import get_collection_metadata
        assert callable(get_collection_metadata)

    def test_collection_type_enum_available(self):
        """Test that CollectionType enum is available."""
        from arcaneum.indexing.collection_metadata import CollectionType
        assert hasattr(CollectionType, 'PDF')
        assert hasattr(CollectionType, 'CODE')


class TestCorpusVerifierModule:
    """Test the collection verifier module."""

    def test_collection_verifier_importable(self):
        """Test that CollectionVerifier can be imported."""
        from arcaneum.indexing.verify import CollectionVerifier
        assert CollectionVerifier is not None


class TestCorpusTypes:
    """Test valid corpus types."""

    def test_valid_corpus_types(self):
        """Test that valid corpus types are supported."""
        from arcaneum.indexing.collection_metadata import CollectionType

        # Should support pdf, code, and markdown
        valid_types = CollectionType.values()
        assert 'pdf' in valid_types
        assert 'code' in valid_types

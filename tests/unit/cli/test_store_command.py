"""CLI tests for store command.

Tests for 'arc store' command for storing agent-generated content.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from io import StringIO

import pytest


class TestStoreCommandImports:
    """Test store command surface."""

    def test_store_command_signature(self):
        """store_command must accept all documented parameters."""
        from arcaneum.cli.index_markdown import store_command
        import inspect

        params = set(inspect.signature(store_command).parameters)
        required = {
            'file', 'collection', 'model', 'title',
            'category', 'tags', 'metadata',
            'chunk_size', 'chunk_overlap',
            'verbose', 'output_json',
        }
        missing = required - params
        assert not missing, f"store_command missing required params: {missing}"


class TestStoreCommandValidation:
    """Test store command validation."""

    def test_validates_file_exists(self, temp_dir):
        """Test that store_command validates file exists."""
        from arcaneum.cli.index_markdown import store_command

        with patch('arcaneum.cli.index_markdown.interaction_logger'):
            with pytest.raises(SystemExit):
                store_command(
                    file='/nonexistent/file.md',
                    collection='TestCollection',
                    model='stella',
                    title=None,
                    category=None,
                    tags=None,
                    metadata=None,
                    chunk_size=None,
                    chunk_overlap=None,
                    verbose=False,
                    output_json=False
                )

    def test_validates_model_exists(self, temp_dir):
        """Test that store_command validates model exists."""
        from arcaneum.cli.index_markdown import store_command

        # Create a test file
        test_file = temp_dir / "test.md"
        test_file.write_text("# Test\n\nContent")

        with patch('arcaneum.cli.index_markdown.interaction_logger'):
            with pytest.raises(SystemExit):
                store_command(
                    file=str(test_file),
                    collection='TestCollection',
                    model='nonexistent_model_xyz',
                    title=None,
                    category=None,
                    tags=None,
                    metadata=None,
                    chunk_size=None,
                    chunk_overlap=None,
                    verbose=False,
                    output_json=False
                )

    def test_validates_json_metadata(self, temp_dir):
        """Test that store_command validates JSON metadata."""
        from arcaneum.cli.index_markdown import store_command

        # Create a test file
        test_file = temp_dir / "test.md"
        test_file.write_text("# Test\n\nContent")

        with patch('arcaneum.cli.index_markdown.interaction_logger'):
            with pytest.raises(SystemExit):
                store_command(
                    file=str(test_file),
                    collection='TestCollection',
                    model='stella',
                    title=None,
                    category=None,
                    tags=None,
                    metadata='invalid json {',  # Invalid JSON
                    chunk_size=None,
                    chunk_overlap=None,
                    verbose=False,
                    output_json=False
                )


class TestStoreInteractionLogging:
    """Test interaction logging for store command."""

    def test_interaction_logger_start_called(self, temp_dir):
        """Test that interaction logger start is called."""
        from arcaneum.cli.index_markdown import store_command

        test_file = temp_dir / "test.md"
        test_file.write_text("# Test\n\nContent")

        mock_logger = MagicMock()

        with patch('arcaneum.cli.index_markdown.interaction_logger', mock_logger):
            # This will fail later but should call start first
            try:
                store_command(
                    file=str(test_file),
                    collection='TestCollection',
                    model='stella',
                    title='Test Title',
                    category='test',
                    tags=None,
                    metadata=None,
                    chunk_size=None,
                    chunk_overlap=None,
                    verbose=False,
                    output_json=False
                )
            except SystemExit:
                pass

        # Verify logging was started
        mock_logger.start.assert_called_once()
        call_kwargs = mock_logger.start.call_args[1]
        assert call_kwargs['collection'] == 'TestCollection'
        assert call_kwargs['title'] == 'Test Title'


class TestStoreTagsParsing:
    """Test tag parsing for store command."""

    def test_parses_comma_separated_tags(self, temp_dir):
        """Test that comma-separated tags are split and stripped before use."""
        from arcaneum.cli.index_markdown import store_command

        test_file = temp_dir / "test.md"
        test_file.write_text("# Test\n\nContent")

        mock_pipeline = MagicMock()
        mock_pipeline.inject_content.return_value = {'chunks': 1, 'persisted': False}

        with patch('arcaneum.cli.index_markdown.interaction_logger'):
            with patch('arcaneum.cli.index_markdown.create_qdrant_client'):
                with patch('arcaneum.cli.index_markdown.get_cached_model'):
                    with patch('arcaneum.cli.index_markdown.validate_collection_type'):
                        with patch('arcaneum.cli.index_markdown.get_vector_names', return_value=None):
                            with patch('arcaneum.cli.index_markdown.MarkdownIndexingPipeline', return_value=mock_pipeline):
                                try:
                                    store_command(
                                        file=str(test_file),
                                        collection='TestCollection',
                                        model='stella',
                                        title=None,
                                        category=None,
                                        tags='tag1, tag2, tag3',  # With spaces
                                        metadata=None,
                                        chunk_size=None,
                                        chunk_overlap=None,
                                        verbose=False,
                                        output_json=False,
                                    )
                                except SystemExit:
                                    pass  # sys.exit(0) on success path

        # inject_content must receive metadata with tags split and stripped
        mock_pipeline.inject_content.assert_called_once()
        call_kwargs = mock_pipeline.inject_content.call_args.kwargs
        assert call_kwargs['metadata']['tags'] == ['tag1', 'tag2', 'tag3']


class TestStoreDefaultModels:
    """Test default models configuration."""

    def test_default_models_includes_stella(self):
        """DEFAULT_MODELS must include the documented 'stella' alias."""
        from arcaneum.config import DEFAULT_MODELS

        assert 'stella' in DEFAULT_MODELS, \
            f"'stella' missing from DEFAULT_MODELS: {list(DEFAULT_MODELS)}"

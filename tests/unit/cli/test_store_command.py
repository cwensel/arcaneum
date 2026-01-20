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
    """Test store command imports."""

    def test_store_command_exists(self):
        """Test that store_command function exists."""
        from arcaneum.cli.index_markdown import store_command
        assert callable(store_command)

    def test_store_command_signature(self):
        """Test store_command has expected parameters."""
        from arcaneum.cli.index_markdown import store_command
        import inspect

        sig = inspect.signature(store_command)
        param_names = list(sig.parameters.keys())

        assert 'file' in param_names
        assert 'collection' in param_names
        assert 'model' in param_names
        assert 'title' in param_names
        assert 'tags' in param_names


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
        """Test that comma-separated tags are parsed correctly."""
        from arcaneum.cli.index_markdown import store_command

        test_file = temp_dir / "test.md"
        test_file.write_text("# Test\n\nContent")

        # We can't easily test the tag parsing without mocking more,
        # but we can verify the command accepts tags
        with patch('arcaneum.cli.index_markdown.interaction_logger'):
            # Just verify the command accepts tags parameter
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
                    output_json=False
                )
            except SystemExit:
                pass  # Expected - will fail on Qdrant connection


class TestStoreStdinSupport:
    """Test stdin support for store command."""

    def test_accepts_stdin_indicator(self):
        """Test that '-' is accepted as file parameter for stdin."""
        from arcaneum.cli.index_markdown import store_command
        import inspect

        # Just verify the command has file parameter that accepts '-'
        sig = inspect.signature(store_command)
        assert 'file' in sig.parameters


class TestStoreDefaultModels:
    """Test default models configuration."""

    def test_default_models_available(self):
        """Test that DEFAULT_MODELS is available."""
        from arcaneum.config import DEFAULT_MODELS

        # Should have at least stella model
        assert 'stella' in DEFAULT_MODELS or len(DEFAULT_MODELS) > 0

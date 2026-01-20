"""CLI tests for config management commands.

Tests for 'arc config' subcommands: show-cache-dir, clear-cache.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestShowCacheDir:
    """Test 'arc config show-cache-dir' command."""

    def test_shows_directories(self, temp_dir, capsys):
        """Test that show-cache-dir displays directory locations."""
        from arcaneum.cli.config import show_cache_dir

        models_dir = temp_dir / "cache" / "models"
        data_dir = temp_dir / "data"
        models_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        with patch('arcaneum.cli.config.get_models_dir', return_value=models_dir):
            with patch('arcaneum.cli.config.get_data_dir', return_value=data_dir):
                with patch('arcaneum.cli.config.get_legacy_arcaneum_dir', return_value=temp_dir / "legacy"):
                    show_cache_dir()

        captured = capsys.readouterr()

        # Should show both directories
        assert 'Cache' in captured.out or 'models' in captured.out
        assert 'Data' in captured.out or 'data' in captured.out

    def test_shows_sizes(self, temp_dir, capsys):
        """Test that sizes are shown for existing directories."""
        from arcaneum.cli.config import show_cache_dir

        models_dir = temp_dir / "cache" / "models"
        data_dir = temp_dir / "data"
        models_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        # Create some test files
        (models_dir / "model.bin").write_bytes(b"x" * 1024)

        with patch('arcaneum.cli.config.get_models_dir', return_value=models_dir):
            with patch('arcaneum.cli.config.get_data_dir', return_value=data_dir):
                with patch('arcaneum.cli.config.get_legacy_arcaneum_dir', return_value=temp_dir / "legacy"):
                    show_cache_dir()

        captured = capsys.readouterr()

        # Should show cache size
        assert 'size' in captured.out.lower() or 'KB' in captured.out or 'MB' in captured.out

    def test_shows_legacy_directory_if_exists(self, temp_dir, capsys):
        """Test that legacy directory is shown if it exists."""
        from arcaneum.cli.config import show_cache_dir

        models_dir = temp_dir / "cache" / "models"
        data_dir = temp_dir / "data"
        legacy_dir = temp_dir / "legacy"

        models_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        legacy_dir.mkdir(parents=True)

        with patch('arcaneum.cli.config.get_models_dir', return_value=models_dir):
            with patch('arcaneum.cli.config.get_data_dir', return_value=data_dir):
                with patch('arcaneum.cli.config.get_legacy_arcaneum_dir', return_value=legacy_dir):
                    show_cache_dir()

        captured = capsys.readouterr()

        # Should mention legacy directory
        assert 'Legacy' in captured.out or 'legacy' in captured.out.lower()


class TestClearCache:
    """Test 'arc config clear-cache' command."""

    def test_confirm_required(self, temp_dir, capsys):
        """Test that --confirm flag is required."""
        from arcaneum.cli.config import clear_cache

        models_dir = temp_dir / "cache" / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "model.bin").write_bytes(b"x" * 1024)

        with patch('arcaneum.cli.config.get_models_dir', return_value=models_dir):
            clear_cache(confirm=False)

        captured = capsys.readouterr()
        # Output may go to stdout or stderr
        output = captured.out + captured.err

        # Should prompt for confirmation
        assert '--confirm' in output or 'confirm' in output.lower()

        # File should still exist
        assert (models_dir / "model.bin").exists()

    def test_clears_with_confirm(self, temp_dir, capsys):
        """Test that cache is cleared with --confirm flag."""
        from arcaneum.cli.config import clear_cache

        models_dir = temp_dir / "cache" / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "model.bin").write_bytes(b"x" * 1024)
        (models_dir / "subdir").mkdir()
        (models_dir / "subdir" / "another.bin").write_bytes(b"y" * 512)

        with patch('arcaneum.cli.config.get_models_dir', return_value=models_dir):
            clear_cache(confirm=True)

        captured = capsys.readouterr()

        # Should show success message
        assert 'Cleared' in captured.out or 'cleared' in captured.out.lower()

        # Directory should be empty (but still exist)
        assert models_dir.exists()
        assert list(models_dir.iterdir()) == []

    def test_handles_empty_cache(self, temp_dir, capsys):
        """Test graceful handling of already empty cache."""
        from arcaneum.cli.config import clear_cache

        models_dir = temp_dir / "cache" / "models"
        # Don't create the directory

        with patch('arcaneum.cli.config.get_models_dir', return_value=models_dir):
            clear_cache(confirm=True)

        captured = capsys.readouterr()

        # Should indicate cache is already empty
        assert 'empty' in captured.out.lower() or 'already' in captured.out.lower()


class TestConfigGroup:
    """Test the config command group."""

    def test_show_cache_dir_command(self, temp_dir):
        """Test show-cache-dir Click command."""
        from arcaneum.cli.config import config_group
        from click.testing import CliRunner

        models_dir = temp_dir / "cache" / "models"
        models_dir.mkdir(parents=True)

        runner = CliRunner()
        with patch('arcaneum.cli.config.get_models_dir', return_value=models_dir):
            with patch('arcaneum.cli.config.get_data_dir', return_value=temp_dir / "data"):
                with patch('arcaneum.cli.config.get_legacy_arcaneum_dir', return_value=temp_dir / "legacy"):
                    result = runner.invoke(config_group, ['show-cache-dir'])

        assert result.exit_code == 0

    def test_clear_cache_command_without_confirm(self, temp_dir):
        """Test clear-cache Click command without --confirm."""
        from arcaneum.cli.config import config_group
        from click.testing import CliRunner

        models_dir = temp_dir / "cache" / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "test.bin").write_bytes(b"x" * 100)

        runner = CliRunner()
        with patch('arcaneum.cli.config.get_models_dir', return_value=models_dir):
            result = runner.invoke(config_group, ['clear-cache'])

        # Should not delete without --confirm
        assert (models_dir / "test.bin").exists()

    def test_clear_cache_command_with_confirm(self, temp_dir):
        """Test clear-cache Click command with --confirm."""
        from arcaneum.cli.config import config_group
        from click.testing import CliRunner

        models_dir = temp_dir / "cache" / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "test.bin").write_bytes(b"x" * 100)

        runner = CliRunner()
        with patch('arcaneum.cli.config.get_models_dir', return_value=models_dir):
            result = runner.invoke(config_group, ['clear-cache', '--confirm'])

        assert result.exit_code == 0
        # File should be deleted
        assert not (models_dir / "test.bin").exists()


class TestGetDirSize:
    """Test directory size calculation helper."""

    def test_calculates_size(self, temp_dir):
        """Test that directory size is calculated correctly."""
        from arcaneum.cli.config import get_dir_size

        test_dir = temp_dir / "test"
        test_dir.mkdir()
        (test_dir / "file1.bin").write_bytes(b"x" * 1000)
        (test_dir / "file2.bin").write_bytes(b"y" * 500)

        size = get_dir_size(test_dir)

        assert size == 1500

    def test_handles_nested_directories(self, temp_dir):
        """Test that nested directories are included in size."""
        from arcaneum.cli.config import get_dir_size

        test_dir = temp_dir / "test"
        test_dir.mkdir()
        (test_dir / "file1.bin").write_bytes(b"x" * 1000)

        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file2.bin").write_bytes(b"y" * 500)

        size = get_dir_size(test_dir)

        assert size == 1500

    def test_returns_zero_for_empty_dir(self, temp_dir):
        """Test that empty directory returns 0."""
        from arcaneum.cli.config import get_dir_size

        test_dir = temp_dir / "empty"
        test_dir.mkdir()

        size = get_dir_size(test_dir)

        assert size == 0

    def test_returns_zero_for_nonexistent_dir(self, temp_dir):
        """Test that nonexistent directory returns 0."""
        from arcaneum.cli.config import get_dir_size

        size = get_dir_size(temp_dir / "nonexistent")

        assert size == 0

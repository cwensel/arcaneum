"""Unit tests for Arc CLI interaction logging (RDR-018)."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from arcaneum.cli.interaction_logger import InteractionLogger, interaction_logger


class TestClaudeEnvironmentDetection:
    """Test Claude Code environment detection."""

    def test_detects_claudecode_env_variable(self):
        """Test detection via CLAUDECODE environment variable."""
        with patch.dict(os.environ, {"CLAUDECODE": "1"}, clear=False):
            logger = InteractionLogger()
            assert logger._is_claude is True
            assert logger.execution_context == "claude"

    def test_detects_claude_plugin_root(self):
        """Test detection via CLAUDE_PLUGIN_ROOT environment variable."""
        with patch.dict(os.environ, {"CLAUDE_PLUGIN_ROOT": "/path/to/plugin"}, clear=False):
            logger = InteractionLogger()
            assert logger._is_claude is True
            assert logger.execution_context == "claude"

    def test_detects_terminal_without_claude_env(self):
        """Test terminal detection when Claude env variables are absent."""
        # Clear Claude-related env vars
        env_without_claude = {
            k: v for k, v in os.environ.items()
            if not k.startswith("CLAUDE")
        }
        with patch.dict(os.environ, env_without_claude, clear=True):
            logger = InteractionLogger()
            assert logger._is_claude is False
            assert logger.execution_context == "terminal"


class TestLogFileCreation:
    """Test log file creation and location."""

    def test_creates_log_directory(self, tmp_path):
        """Test that log directory is created if missing."""
        log_dir = tmp_path / ".arcaneum" / "logs"

        # Patch LOG_DIR to use temp directory
        with patch.object(InteractionLogger, "LOG_DIR", log_dir):
            logger = InteractionLogger()
            logger.start("test", "command")
            logger.finish()

            assert log_dir.exists()

    def test_log_file_uses_utc_date(self, tmp_path):
        """Test that log filename uses UTC date."""
        log_dir = tmp_path / ".arcaneum" / "logs"
        log_dir.mkdir(parents=True)

        with patch.object(InteractionLogger, "LOG_DIR", log_dir):
            logger = InteractionLogger()
            logger.start("test", "command")
            logger.finish()

            # Check for date-based log file
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            expected_file = log_dir / f"arc-interactions-{today}.log"
            assert expected_file.exists()

    def test_writes_jsonl_format(self, tmp_path):
        """Test that entries are written as JSONL."""
        log_dir = tmp_path / ".arcaneum" / "logs"
        log_dir.mkdir(parents=True)

        with patch.object(InteractionLogger, "LOG_DIR", log_dir):
            logger = InteractionLogger()
            logger.start("search", "semantic", query="test query")
            logger.finish(result_count=5)

            # Read log file
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_file = log_dir / f"arc-interactions-{today}.log"
            content = log_file.read_text()

            # Parse as JSON (should be valid JSONL)
            entry = json.loads(content.strip())
            assert entry["command"] == "search"
            assert entry["subcommand"] == "semantic"
            assert entry["query"] == "test query"
            assert entry["result_count"] == 5


class TestLogEntryFormat:
    """Test log entry schema and fields."""

    def test_entry_includes_required_fields(self, tmp_path):
        """Test that entries include all required fields from RDR-018."""
        log_dir = tmp_path / ".arcaneum" / "logs"
        log_dir.mkdir(parents=True)

        with patch.object(InteractionLogger, "LOG_DIR", log_dir):
            logger = InteractionLogger()
            logger.start("search", "semantic", collection="TestCol", query="test")
            logger.finish(result_count=10)

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_file = log_dir / f"arc-interactions-{today}.log"
            entry = json.loads(log_file.read_text().strip())

            # Check required fields
            assert "timestamp" in entry
            assert "context" in entry
            assert "project" in entry
            assert "command" in entry
            assert "subcommand" in entry
            assert "duration_ms" in entry
            assert "cwd" in entry
            assert entry["collection"] == "TestCol"
            assert entry["query"] == "test"
            assert entry["result_count"] == 10

    def test_project_field_is_cwd_folder_name(self, tmp_path):
        """Test that project field is the folder name from cwd."""
        log_dir = tmp_path / ".arcaneum" / "logs"
        log_dir.mkdir(parents=True)

        with patch.object(InteractionLogger, "LOG_DIR", log_dir):
            logger = InteractionLogger()
            logger.start("test", "command")
            logger.finish()

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_file = log_dir / f"arc-interactions-{today}.log"
            entry = json.loads(log_file.read_text().strip())

            # Project should be the folder name
            expected_project = Path.cwd().name
            assert entry["project"] == expected_project

    def test_error_field_captures_errors(self, tmp_path):
        """Test that error field is populated on failure."""
        log_dir = tmp_path / ".arcaneum" / "logs"
        log_dir.mkdir(parents=True)

        with patch.object(InteractionLogger, "LOG_DIR", log_dir):
            logger = InteractionLogger()
            logger.start("search", "semantic")
            logger.finish(error="Connection refused")

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_file = log_dir / f"arc-interactions-{today}.log"
            entry = json.loads(log_file.read_text().strip())

            assert entry["error"] == "Connection refused"

    def test_error_is_null_on_success(self, tmp_path):
        """Test that error field is null on successful operation."""
        log_dir = tmp_path / ".arcaneum" / "logs"
        log_dir.mkdir(parents=True)

        with patch.object(InteractionLogger, "LOG_DIR", log_dir):
            logger = InteractionLogger()
            logger.start("collection", "list")
            logger.finish(result_count=3)

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_file = log_dir / f"arc-interactions-{today}.log"
            entry = json.loads(log_file.read_text().strip())

            assert entry["error"] is None

    def test_claude_env_captured_when_under_claude(self, tmp_path):
        """Test that claude_env is populated when running under Claude."""
        log_dir = tmp_path / ".arcaneum" / "logs"
        log_dir.mkdir(parents=True)

        with patch.dict(os.environ, {"CLAUDECODE": "1"}, clear=False):
            with patch.object(InteractionLogger, "LOG_DIR", log_dir):
                logger = InteractionLogger()
                logger.start("search", "semantic")
                logger.finish(result_count=5)

                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                log_file = log_dir / f"arc-interactions-{today}.log"
                entry = json.loads(log_file.read_text().strip())

                assert "claude_env" in entry
                assert entry["context"] == "claude"


class TestDisablingLogging:
    """Test disabling logging via environment variable."""

    def test_arc_interaction_log_0_disables_logging(self, tmp_path):
        """Test that ARC_INTERACTION_LOG=0 disables logging."""
        log_dir = tmp_path / ".arcaneum" / "logs"
        log_dir.mkdir(parents=True)

        with patch.dict(os.environ, {"ARC_INTERACTION_LOG": "0"}, clear=False):
            with patch.object(InteractionLogger, "LOG_DIR", log_dir):
                logger = InteractionLogger()
                assert logger._disabled is True

                logger.start("search", "semantic")
                logger.finish(result_count=5)

                # No log file should be created
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                log_file = log_dir / f"arc-interactions-{today}.log"
                assert not log_file.exists()

    def test_logging_enabled_by_default(self):
        """Test that logging is enabled by default."""
        # Clear ARC_INTERACTION_LOG if set
        env = {k: v for k, v in os.environ.items() if k != "ARC_INTERACTION_LOG"}
        with patch.dict(os.environ, env, clear=True):
            logger = InteractionLogger()
            assert logger._disabled is False


class TestSilentFailure:
    """Test that logging failures don't break the CLI."""

    def test_write_failure_is_silent(self, tmp_path):
        """Test that write failures are silently ignored."""
        # Use a non-existent directory that can't be created
        log_dir = tmp_path / "nonexistent" / "path" / ".arcaneum" / "logs"

        with patch.object(InteractionLogger, "LOG_DIR", log_dir):
            logger = InteractionLogger()

            # Patch _get_log_dir to raise an exception
            def raise_on_get_log_dir():
                raise PermissionError("Cannot create log directory")

            with patch.object(logger, "_get_log_dir", raise_on_get_log_dir):
                # This should not raise an exception - errors are silently caught
                logger.start("search", "semantic")
                logger.finish(result_count=5)

    def test_file_write_failure_is_silent(self, tmp_path):
        """Test that actual file write failures are handled silently."""
        # Create log directory but make it read-only
        log_dir = tmp_path / ".arcaneum" / "logs"
        log_dir.mkdir(parents=True)

        # On some systems, we can test with a read-only directory
        # For portability, we'll test by patching open to raise
        with patch.object(InteractionLogger, "LOG_DIR", log_dir):
            logger = InteractionLogger()

            # Patch builtins.open to fail
            original_open = open

            def failing_open(*args, **kwargs):
                if "arc-interactions" in str(args[0]):
                    raise PermissionError("Cannot write to log file")
                return original_open(*args, **kwargs)

            with patch("builtins.open", failing_open):
                # This should not raise an exception
                logger.start("search", "semantic")
                logger.finish(result_count=5)


class TestContextManager:
    """Test the track() context manager."""

    def test_track_context_manager(self, tmp_path):
        """Test the track() context manager usage."""
        log_dir = tmp_path / ".arcaneum" / "logs"
        log_dir.mkdir(parents=True)

        with patch.object(InteractionLogger, "LOG_DIR", log_dir):
            logger = InteractionLogger()

            with logger.track("search", "semantic", query="test") as result:
                result["result_count"] = 10

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_file = log_dir / f"arc-interactions-{today}.log"
            entry = json.loads(log_file.read_text().strip())

            assert entry["command"] == "search"
            assert entry["query"] == "test"
            assert entry["result_count"] == 10

    def test_track_captures_exceptions(self, tmp_path):
        """Test that track() captures exceptions."""
        log_dir = tmp_path / ".arcaneum" / "logs"
        log_dir.mkdir(parents=True)

        with patch.object(InteractionLogger, "LOG_DIR", log_dir):
            logger = InteractionLogger()

            with pytest.raises(ValueError):
                with logger.track("search", "semantic") as result:
                    raise ValueError("Search failed")

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_file = log_dir / f"arc-interactions-{today}.log"
            entry = json.loads(log_file.read_text().strip())

            assert entry["error"] == "Search failed"


class TestGlobalLoggerInstance:
    """Test the global interaction_logger instance."""

    def test_global_instance_exists(self):
        """Test that a global logger instance is available."""
        assert interaction_logger is not None
        assert isinstance(interaction_logger, InteractionLogger)

    def test_global_instance_is_reusable(self, tmp_path):
        """Test that the global instance can be used multiple times."""
        log_dir = tmp_path / ".arcaneum" / "logs"
        log_dir.mkdir(parents=True)

        with patch.object(InteractionLogger, "LOG_DIR", log_dir):
            # Use global instance multiple times
            interaction_logger.start("search", "semantic")
            interaction_logger.finish(result_count=5)

            interaction_logger.start("collection", "list")
            interaction_logger.finish(result_count=3)

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_file = log_dir / f"arc-interactions-{today}.log"
            lines = log_file.read_text().strip().split("\n")

            assert len(lines) == 2
            assert json.loads(lines[0])["command"] == "search"
            assert json.loads(lines[1])["command"] == "collection"


class TestDurationTracking:
    """Test duration_ms tracking."""

    def test_duration_is_recorded(self, tmp_path):
        """Test that duration_ms is recorded in log entries."""
        log_dir = tmp_path / ".arcaneum" / "logs"
        log_dir.mkdir(parents=True)

        with patch.object(InteractionLogger, "LOG_DIR", log_dir):
            logger = InteractionLogger()
            logger.start("search", "semantic")
            logger.finish(result_count=5)

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_file = log_dir / f"arc-interactions-{today}.log"
            entry = json.loads(log_file.read_text().strip())

            assert "duration_ms" in entry
            assert isinstance(entry["duration_ms"], int)
            assert entry["duration_ms"] >= 0

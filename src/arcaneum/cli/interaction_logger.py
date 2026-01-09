"""Arc CLI interaction logging (RDR-018).

Logs all Arc CLI interactions to ~/.arcaneum/logs/ for:
- Debugging search patterns and query effectiveness
- Understanding search behavior over time (both agent and user)
- Auditing Arc usage across sessions
- Correlating search patterns with project work
"""

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class InteractionLogger:
    """Logs all Arc CLI interactions to ~/.arcaneum/logs/.

    All logs go to a single global location with project identification
    via the 'project' field (folder name from cwd).

    Can be disabled by setting ARC_INTERACTION_LOG=0 environment variable.
    """

    LOG_DIR = Path.home() / ".arcaneum" / "logs"
    LOG_FILENAME_TEMPLATE = "arc-interactions-{date}.log"

    def __init__(self):
        self._disabled = os.environ.get("ARC_INTERACTION_LOG", "1") == "0"
        self._is_claude = self._detect_claude_environment()
        self._start_time: Optional[float] = None
        self._context: dict[str, Any] = {}

    def _detect_claude_environment(self) -> bool:
        """Check if running under Claude Code."""
        # Primary indicator: CLAUDECODE is set (observed in actual environment)
        if os.environ.get("CLAUDECODE"):
            return True
        # Secondary: CLAUDE_PLUGIN_ROOT (when invoked via plugin)
        if os.environ.get("CLAUDE_PLUGIN_ROOT"):
            return True
        return False

    def _get_log_dir(self) -> Path:
        """Get or create the log directory."""
        if not self.LOG_DIR.exists():
            self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        return self.LOG_DIR

    def _get_project(self) -> str:
        """Get project identifier from current working directory."""
        return Path.cwd().name

    def _get_claude_env(self) -> dict[str, Any]:
        """Capture relevant Claude environment variables."""
        if not self._is_claude:
            return {}
        env = {}
        claude_vars = [
            "CLAUDECODE",           # Primary indicator (always set)
            "CLAUDE_CODE_ENTRYPOINT",  # How Claude was launched (cli, etc.)
            "CLAUDE_PLUGIN_ROOT",   # Plugin directory (when via plugin)
            "CLAUDE_SESSION_ID",    # Future-proofing
            "CLAUDE_AGENT_ID",      # Future-proofing
        ]
        for var in claude_vars:
            value = os.environ.get(var)
            if value:
                env[var.lower().replace("claude_", "")] = value
        return env

    @property
    def execution_context(self) -> str:
        """Return the execution context identifier."""
        return "claude" if self._is_claude else "terminal"

    def start(self, command: str, subcommand: Optional[str] = None, **kwargs):
        """Start tracking an interaction.

        Args:
            command: The main command (e.g., 'search', 'collection', 'index')
            subcommand: The subcommand (e.g., 'semantic', 'list', 'pdf')
            **kwargs: Additional context to log (query, collection, filters, etc.)
        """
        self._start_time = time.perf_counter()
        self._context = {
            "command": command,
            "subcommand": subcommand,
            **kwargs
        }

    def finish(
        self,
        result_count: Optional[int] = None,
        error: Optional[str] = None,
        **extra
    ):
        """Complete and write the interaction log entry.

        Args:
            result_count: Number of results returned (for searches)
            error: Error message if command failed, None on success
            **extra: Additional fields to include in log entry
        """
        if self._start_time is None:
            return

        duration_ms = int((time.perf_counter() - self._start_time) * 1000)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": self.execution_context,
            "project": self._get_project(),
            "duration_ms": duration_ms,
            "cwd": str(Path.cwd()),
            "error": error,
            **self._context,
            **extra,
        }

        # Add Claude env only when running under Claude
        if self._is_claude:
            entry["claude_env"] = self._get_claude_env()

        if result_count is not None:
            entry["result_count"] = result_count

        self._write_entry(entry)
        self._reset()

    def _get_log_filename(self) -> str:
        """Get date-based log filename (UTC)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.LOG_FILENAME_TEMPLATE.format(date=today)

    def _write_entry(self, entry: dict[str, Any]):
        """Append entry to log file."""
        if self._disabled:
            return
        try:
            log_file = self._get_log_dir() / self._get_log_filename()
            with open(log_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            # Silent failure - logging should never break the CLI
            pass

    def _reset(self):
        """Reset internal state."""
        self._start_time = None
        self._context = {}

    @contextmanager
    def track(self, command: str, subcommand: Optional[str] = None, **kwargs):
        """Context manager for tracking interactions.

        Usage:
            with interaction_logger.track("search", "semantic", query="test") as result:
                # ... perform operation ...
                result["result_count"] = 10

        Args:
            command: The main command
            subcommand: The subcommand
            **kwargs: Additional context to log
        """
        self.start(command, subcommand, **kwargs)
        result = {"result_count": None, "error": None}
        try:
            yield result
        except Exception as e:
            result["error"] = str(e)
            raise
        finally:
            self.finish(
                result_count=result.get("result_count"),
                error=result.get("error")
            )


# Global logger instance
interaction_logger = InteractionLogger()

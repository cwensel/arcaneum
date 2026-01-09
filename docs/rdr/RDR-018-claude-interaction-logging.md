# Recommendation 018: Arc CLI Interaction Logging for Claude Code

## Metadata

- **Date**: 2026-01-08
- **Status**: Recommendation
- **Type**: Feature
- **Priority**: High
- **Related Issues**: None
- **Related Tests**: `tests/cli/test_interaction_logger.py`

## Problem Statement

Arc CLI interactions leave no trace, making it difficult to:

1. Identify when Claude agents fail to use Arc when they should
2. Debug search patterns and query effectiveness
3. Understand search behavior over time (both agent and user)
4. Audit Arc usage across sessions
5. Correlate search patterns with project work

## Context

### Background

Claude Code agents can invoke Arc via slash commands (`/arc:search`, `/arc:collection`, etc.) or directly via bash. Currently,
these interactions leave no trace, making it impossible to review agent search behavior post-session.

The logging should:

- Be automatic for all Arc CLI invocations
- Not interfere with normal CLI output
- Capture enough metadata to understand the interaction
- Use context-aware log locations (project-local under Claude, global under terminal)

### Technical Environment

- **Framework**: Click 8.3.0+ CLI framework
- **Python**: 3.10+
- **Entry Point**: `src/arcaneum/cli/main.py`
- **Existing Logging**: `src/arcaneum/cli/logging_config.py`
- **Output Utils**: `src/arcaneum/cli/output.py`

## Research Findings

### Investigation Process

1. Analyzed Arc CLI structure and command flow
2. Reviewed existing logging infrastructure
3. Investigated Claude Code environment variables
4. Examined `.claude` directory conventions

### Key Discoveries

1. **Claude Environment Variables**: Limited built-in variables
   - `CLAUDE_PLUGIN_ROOT` - Plugin directory path
   - No session/agent ID exposed by default

2. **Detection Strategy**: Check for Claude-specific indicators
   - `CLAUDE_PLUGIN_ROOT` environment variable presence
   - Parent process checks (less reliable)

3. **Click Middleware**: Can intercept all commands via `@cli.result_callback()`

4. **Log Location Convention**: Global `~/.arcaneum/logs/` for all Arc interactions
   - Single location simplifies log management and analysis
   - Project identification via `project` field in each log entry

## Proposed Solution

### Approach

Implement a lightweight interaction logger that:

1. **Always logs** all Arc CLI invocations to a single global location (`~/.arcaneum/logs/`)
2. Uses **UTC date-based filenames** for natural rotation (`arc-interactions-YYYY-MM-DD.log`)
   - UTC ensures consistent filenames across timezones and avoids ambiguity
3. Uses JSONL format for easy parsing
4. Includes **project identifier** in each log entry (folder name from cwd)
5. Marks log entries with execution context (claude vs terminal)

### Technical Design

#### Log Entry Schema

```json
{
  "timestamp": "2026-01-08T14:32:15.123456+00:00",
  "context": "claude",
  "project": "my-project",
  "command": "search",
  "subcommand": "semantic",
  "collection": "Standards",
  "query": "WCAG accessibility requirements",
  "filters": {"type": "standard"},
  "offset": 0,
  "limit": 10,
  "result_count": 7,
  "error": null,
  "duration_ms": 234,
  "claude_env": {
    "code": "1",
    "code_entrypoint": "cli"
  },
  "cwd": "/Users/user/my-project"
}
```

Key fields:

- `context` - Execution environment: `"claude"` or `"terminal"`
- `project` - Folder name from cwd, enables filtering/grouping logs by project
- `cwd` - Full path for disambiguation when project names collide

#### Component Architecture

```text
src/arcaneum/cli/
├── interaction_logger.py   # New: logging infrastructure
├── main.py                 # Modified: integrate logger
└── ...
```

### Implementation Example

```python
# src/arcaneum/cli/interaction_logger.py
"""Arc CLI interaction logging."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager

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
        """Start tracking an interaction."""
        self._start_time = time.perf_counter()
        self._context = {
            "command": command,
            "subcommand": subcommand,
            **kwargs
        }

    def finish(self, result_count: Optional[int] = None, error: Optional[str] = None, **extra):
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
        """Get date-based log filename."""
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
        """Context manager for tracking interactions."""
        self.start(command, subcommand, **kwargs)
        result = {"result_count": None}
        try:
            yield result
        finally:
            self.finish(result_count=result.get("result_count"))


# Global logger instance
interaction_logger = InteractionLogger()
```

#### Integration in Search Command

```python
# src/arcaneum/cli/search.py (modifications)

from arcaneum.cli.interaction_logger import interaction_logger

@search.command()
@click.argument("query")
@click.option("--collection", "-c", required=True)
@click.option("--limit", "-n", default=10)
@click.option("--offset", default=0)
@click.option("--filter", "-f", "filters", multiple=True)
def semantic(query: str, collection: str, limit: int, offset: int, filters: tuple):
    """Perform semantic search."""
    with interaction_logger.track(
        "search", "semantic",
        collection=collection,
        query=query,
        limit=limit,
        offset=offset,
        filters=list(filters) if filters else None
    ) as result:
        # ... existing search logic ...
        results = perform_search(query, collection, limit, offset, filters)
        result["result_count"] = len(results)
        # ... display results ...
```

## Alternatives Considered

### Alternative: Single Log File

**Description**: Use a single `arc-interactions.log` file instead of date-based filenames.

**Pros**:

- Simpler file management (one file to track)
- All history in one place

**Cons**:

- File grows indefinitely without manual intervention
- Requires implementing rotation logic or manual cleanup
- Harder to delete old logs selectively

**Reason for rejection**: Date-based files (`arc-interactions-YYYY-MM-DD.log`) provide natural rotation
and easy cleanup of old logs without additional code complexity.

## Trade-offs and Consequences

### Positive Consequences

- Visibility into Arc usage by Claude agents
- Ability to identify missed search opportunities
- Foundation for usage analytics and optimization
- Debugging aid for search queries

### Negative Consequences

- Small performance overhead (~1-2ms per command)
- Disk space usage (minimal, ~100 bytes per entry)

### Risks and Mitigations

- **Risk**: Logging fails and breaks CLI
  **Mitigation**: Wrap all logging in try/except with silent failure

- **Risk**: Log files accumulate over time
  **Mitigation**: Date-based files allow easy cleanup of old logs; users can delete old files

- **Risk**: Sensitive data in queries logged
  **Mitigation**: Log file in user home directory (`~/.arcaneum/logs/`), not in project

## Implementation Plan

### Prerequisites

- [ ] Understand current Click command structure
- [ ] Identify all commands that should be logged

### Step-by-Step Implementation

#### Step 1: Create InteractionLogger Class

Create `src/arcaneum/cli/interaction_logger.py` with:

- Claude environment detection
- Dual-location directory management
- JSONL log writing
- Context manager for tracking

**Effort**: ~1 hour

#### Step 2: Integrate with Search Commands

Modify `src/arcaneum/cli/search.py`:

- Import interaction_logger
- Wrap search commands with tracking
- Capture query, collection, filters, result count

**Effort**: ~30 minutes

#### Step 3: Integrate with Index Commands

Modify index commands (`index_pdfs.py`, `index_source.py`, `index_markdown.py`):

- Track paths, options, item counts

**Effort**: ~45 minutes

#### Step 4: Integrate with Collection Commands

Modify `src/arcaneum/cli/collections.py`:

- Track list, info, create, delete operations

**Effort**: ~30 minutes

#### Step 5: Add Unit Tests

Create `tests/cli/test_interaction_logger.py`:

- Test Claude detection (CLAUDECODE and CLAUDE_PLUGIN_ROOT)
- Test log file creation in ~/.arcaneum/logs/
- Test entry format including project, error, and claude_env fields
- Test project field matches cwd folder name
- Test silent failure mode
- Test ARC_INTERACTION_LOG=0 disables logging

**Effort**: ~1 hour

### Files to Modify

- `src/arcaneum/cli/interaction_logger.py` - Create new logger module
- `src/arcaneum/cli/search.py` - Add logging to search commands
- `src/arcaneum/cli/collections.py` - Add logging to collection commands
- `src/arcaneum/cli/index_pdfs.py` - Add logging to PDF indexing
- `src/arcaneum/cli/index_source.py` - Add logging to source indexing
- `src/arcaneum/cli/index_markdown.py` - Add logging to markdown indexing
- `tests/cli/test_interaction_logger.py` - Create unit tests

### Dependencies

- No new dependencies required
- Uses standard library: json, os, time, datetime, pathlib

## Validation

### Testing Approach

1. Unit tests for InteractionLogger class
2. Integration tests verifying log output
3. Manual testing under Claude Code

### Test Scenarios

1. **Scenario**: Run search command under Claude Code
   **Expected Result**: Log entry in `~/.arcaneum/logs/arc-interactions-YYYY-MM-DD.log` with
   `context: "claude"`, `project` matching cwd folder name, and `claude_env` populated

2. **Scenario**: Run search command from terminal (no Claude env)
   **Expected Result**: Log entry in `~/.arcaneum/logs/arc-interactions-YYYY-MM-DD.log` with
   `context: "terminal"`, `project` field set, no `claude_env` field

3. **Scenario**: `~/.arcaneum/logs/` directory doesn't exist
   **Expected Result**: Directory created automatically, log written

4. **Scenario**: Log write fails (permissions, disk full)
   **Expected Result**: Command completes successfully, no error shown

5. **Scenario**: Run arc from different project directories
   **Expected Result**: Each log entry has correct `project` field (folder name) and `cwd` (full path)

### Performance Validation

- Measure command execution time with/without logging
- Target: <5ms overhead

### Security Validation

- Verify logs in `~/.arcaneum/logs/` are not exposed (user home directory)
- Verify no sensitive data beyond query strings is logged

## References

- [Click Documentation](https://click.palletsprojects.com/)
- [JSONL Format](https://jsonlines.org/)
- Claude Code Plugin Development

## Notes

### Future Enhancements

- Auto-cleanup of logs older than N days
- Analytics dashboard for interaction patterns
- Integration with beads for correlation with issues
- Export/import for cross-machine analysis
- CLI command to view/analyze interaction logs

### Known Limitations

- Query strings logged in plain text
- No retroactive logging for past sessions
- Logging captures when Arc IS used, but cannot detect when Arc SHOULD have been used but wasn't
  (requires external correlation with conversation logs)
- No file locking for concurrent writes (JSONL format minimizes corruption risk, but interleaved
  writes are theoretically possible)
- Project identification relies on cwd folder name, which may not be unique across different
  projects with the same name (use `cwd` field for disambiguation)

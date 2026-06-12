"""Log inspection commands for Arcaneum."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from arcaneum.cli.interaction_logger import InteractionLogger


def current_interaction_log_path(
    log_dir: Path | None = None,
    now: Callable[[], datetime] | None = None,
) -> Path:
    """Return the active interaction log path for the current UTC date."""
    log_dir = log_dir or InteractionLogger.LOG_DIR
    current_time = now() if now is not None else datetime.now(timezone.utc)
    date = current_time.astimezone(timezone.utc).strftime("%Y-%m-%d")
    return log_dir / InteractionLogger.LOG_FILENAME_TEMPLATE.format(date=date)


def _tail_bytes(path: Path, lines: int) -> tuple[bytes, int]:
    """Read the last N lines from a file and return bytes plus EOF offset."""
    size = path.stat().st_size
    if lines <= 0 or size == 0:
        return b"", size

    with path.open("rb") as f:
        data = f.read()
    return b"".join(data.splitlines(keepends=True)[-lines:]), size


def _interaction_log_paths_through(log_dir: Path, active_path: Path) -> list[Path]:
    """Return dated interaction logs up to and including the active log date."""
    if not log_dir.exists():
        return []

    prefix, suffix = InteractionLogger.LOG_FILENAME_TEMPLATE.split("{date}")
    # ISO date filenames sort chronologically with normal string ordering.
    paths = [
        path
        for path in log_dir.glob(f"{prefix}*{suffix}")
        if path.is_file() and path.name <= active_path.name
    ]
    return sorted(paths)


def _tail_bytes_across(paths: list[Path], lines: int) -> bytes:
    """Read the last N lines from a sequence of log files."""
    if lines <= 0:
        return b""

    chunks: list[bytes] = []
    remaining = lines
    for path in reversed(paths):
        data, _ = _tail_bytes(path, remaining)
        if data:
            file_lines = data.splitlines(keepends=True)
            chunks[:0] = file_lines
            remaining -= len(file_lines)
            if remaining <= 0:
                break

    return b"".join(chunks[-lines:])


class CurrentLogTailer:
    """Poll-based tailer for the current date-based interaction log."""

    def __init__(
        self,
        log_dir: Path | None = None,
        lines: int = 0,
        now: Callable[[], datetime] | None = None,
    ):
        self.log_dir = log_dir or InteractionLogger.LOG_DIR
        self.lines = lines
        self.now = now
        self._active_path: Path | None = None
        self._offset = 0
        self._started = False
        self._start_at_eof = True

    def poll(self) -> str:
        """Return newly available log text, switching files at UTC date cutover."""
        expected_path = current_interaction_log_path(self.log_dir, self.now)
        if expected_path != self._active_path:
            self._start_at_eof = self._active_path is None
            self._active_path = expected_path
            self._offset = 0
            self._started = False

        if not self._active_path.exists():
            if not self._started and self._start_at_eof:
                paths = _interaction_log_paths_through(self.log_dir, self._active_path)
                data = _tail_bytes_across(paths, self.lines)
                self._started = True
                return data.decode("utf-8", errors="replace")
            self._started = True
            return ""

        if not self._started:
            if self._start_at_eof:
                paths = _interaction_log_paths_through(self.log_dir, self._active_path)
                data = _tail_bytes_across(paths, self.lines)
                self._offset = self._active_path.stat().st_size
            else:
                with self._active_path.open("rb") as f:
                    data = f.read()
                self._offset = self._active_path.stat().st_size
            self._started = True
            return data.decode("utf-8", errors="replace")

        size = self._active_path.stat().st_size
        if size < self._offset:
            self._offset = 0

        with self._active_path.open("rb") as f:
            f.seek(self._offset)
            data = f.read()
            self._offset = f.tell()

        return data.decode("utf-8", errors="replace")


def tail_current_log(
    log_dir: Path | None = None,
    lines: int = 0,
    poll_interval: float = 1.0,
    stream: TextIO | None = None,
):
    """Continuously tail the current interaction log and follow UTC cutovers."""
    stream = stream or sys.stdout
    tailer = CurrentLogTailer(log_dir=log_dir, lines=lines)

    try:
        while True:
            output = tailer.poll()
            if output:
                stream.write(output)
                stream.flush()
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        return

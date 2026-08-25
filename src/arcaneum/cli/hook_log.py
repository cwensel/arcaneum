"""Shared record for hook-driven indexing (follow-up to kata vq0n).

A drain that fails leaves the same empty spool a drain that succeeded does,
once the batch is consumed. Without a line per batch there is no way to tell
those apart afterwards -- and a launchd-driven drain wrote nowhere at all,
because the plist captured no output.

One file, ``<state>/arcaneum/hook.log``, written by three producers: the hook
script (in shell), this module (from the drain worker), and launchd/systemd
(by redirecting the worker's stdout and stderr here). The path must stay in
sync with the shell literal in ``hooks.render_block``.

Logging is diagnostics. Every failure here is swallowed: losing a log line is
never a reason to fail an indexing run.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from ..paths import get_state_dir

logger = logging.getLogger(__name__)


def hook_log_path() -> Path:
    """The single log file every hook-driven producer appends to."""
    return get_state_dir() / "hook.log"


def write(message: str) -> None:
    """Append one timestamped line. Never raises."""
    try:
        path = hook_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {message}\n")
    except OSError as exc:
        logger.debug("Could not write hook log: %s", exc)

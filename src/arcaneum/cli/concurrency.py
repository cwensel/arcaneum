"""Cross-process concurrency caps for embedder-loading commands.

Background: every `arc search semantic` invocation cold-loads an embedding
model into a fresh Python process. When agents fan out N parallel searches,
N copies of the model end up resident at once, which on consumer hardware
thrashes RAM/swap (observed: 5 concurrent runs took 115-121s each vs. the
5-25s baseline, plus system-wide memory pressure).

This module provides a counted, kernel-managed semaphore implemented via
`fcntl.flock` over N lock files in ~/.cache/arcaneum/locks/. flock is held
for the lifetime of the holding process, so kernel automatically releases
slots if a holder is killed mid-search — no stale-PID cleanup needed.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Iterator

from .errors import SearchSlotUnavailable
from ..paths import get_data_dir

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_CONCURRENCY = 2
DEFAULT_SEARCH_WAIT_SECONDS = 60.0

# Threshold above which we surface a one-line "still waiting" notice to the
# user, so an interactive caller knows the delay is the cap and not a hang.
_WAIT_NOTICE_THRESHOLD_S = 1.0


def _locks_dir() -> Path:
    locks_dir = get_data_dir() / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    return locks_dir


def _read_int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default
    return max(value, minimum)


def _read_float_env(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %.1f", name, raw, default)
        return default
    return max(value, minimum)


def get_search_concurrency() -> int:
    return _read_int_env("ARCANEUM_SEARCH_CONCURRENCY", DEFAULT_SEARCH_CONCURRENCY, minimum=1)


def get_search_wait_seconds() -> float:
    return _read_float_env("ARCANEUM_SEARCH_WAIT_SECONDS", DEFAULT_SEARCH_WAIT_SECONDS, minimum=0.0)


@contextlib.contextmanager
def acquire_embedder_slot(
    *,
    slots: int | None = None,
    timeout: float | None = None,
    name: str = "embedder",
) -> Iterator[None]:
    """Acquire one of N flock-backed slots, blocking up to `timeout` seconds.

    Parameters are read from env vars when not passed explicitly so callers
    do not have to thread configuration through. Pass `name` to scope the
    semaphore to a different family of operations (currently only "embedder").

    Raises SearchSlotUnavailable when all slots remain busy past the timeout.

    Windows is not a supported target; on platforms without fcntl this is a
    no-op so library imports remain clean.
    """
    if platform.system() == "Windows":
        yield
        return

    try:
        import fcntl
    except ImportError:
        # Defensive: any platform missing fcntl gets a no-op semaphore rather
        # than a hard import error inside the search path.
        yield
        return

    slot_count = slots if slots is not None else get_search_concurrency()
    wait_budget = timeout if timeout is not None else get_search_wait_seconds()

    locks_dir = _locks_dir()
    slot_paths = [locks_dir / f"{name}-slot-{i}.lock" for i in range(slot_count)]

    deadline = time.monotonic() + wait_budget
    acquired_fd: int | None = None
    started = time.monotonic()
    notice_emitted = False

    while True:
        for path in slot_paths:
            fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            acquired_fd = fd
            break

        if acquired_fd is not None:
            break

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SearchSlotUnavailable(
                f"All {slot_count} embedder slots busy after waiting "
                f"{wait_budget:.0f}s. Another `arc search semantic` is loading "
                "the embedding model — retry shortly, or raise the cap with "
                "ARCANEUM_SEARCH_CONCURRENCY=N (default 2)."
            )

        if not notice_emitted and (time.monotonic() - started) > _WAIT_NOTICE_THRESHOLD_S:
            print(
                f"[INFO] waiting for embedder slot ({slot_count} in use, "
                f"cap=ARCANEUM_SEARCH_CONCURRENCY)",
                file=sys.stderr,
            )
            notice_emitted = True

        # Jittered backoff prevents N contenders from waking in lockstep.
        sleep_for = min(remaining, 0.2 + random.uniform(0, 0.3))
        time.sleep(sleep_for)

    try:
        yield
    finally:
        try:
            fcntl.flock(acquired_fd, fcntl.LOCK_UN)
        finally:
            os.close(acquired_fd)

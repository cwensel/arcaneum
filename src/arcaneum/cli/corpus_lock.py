"""Per-corpus advisory write lock (kata htmw).

`arc corpus sync` reads "what is already indexed" and then writes to both
Qdrant and MeiliSearch. Those two steps are not atomic with respect to another
run, so two concurrent syncs of the same corpus interleave and produce
duplicate points/documents, lost deletes under ``--parity``, and cross-system
parity drift — besides loading the embedding model twice for no benefit.

This module serializes them with an ``fcntl.flock`` on a per-corpus lock file
under ``get_data_dir() / "locks"``, the same kernel-managed mechanism used by
``concurrency.py`` for embedder slots. flock is held for the lifetime of the
holding process, so the OS releases the lock if a sync is killed mid-run and
there is no stale-lock cleanup to get wrong.

The lock key mixes the corpus name with the configured Qdrant/MeiliSearch
endpoints so the same corpus name pointed at different services does not
needlessly serialize.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .errors import CorpusLockUnavailable
from ..paths import get_data_dir

logger = logging.getLogger(__name__)

DEFAULT_LOCK_TIMEOUT_SECONDS = 600.0

# Threshold above which we tell the user we are waiting, so a blocked sync
# reads as contention rather than a hang.
_WAIT_NOTICE_THRESHOLD_S = 0.5

# Corpus names are user-supplied; keep only characters that are safe in a
# filename so a name like "../../etc/passwd" cannot escape the locks dir.
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]")

# Locks this process already holds, keyed by lock path. Sync calls into
# helpers that may re-acquire (e.g. repair invoking sync); flock on a second
# fd for the same file would self-deadlock, so nested acquires are no-ops.
_held: Dict[str, int] = {}


def _locks_dir() -> Path:
    locks_dir = get_data_dir() / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    return locks_dir


def _service_endpoints() -> str:
    """Identify the Qdrant/MeiliSearch targets this invocation will write to.

    Read from the environment only. Resolving the full config here would import
    the client factories (and their heavy dependencies) just to build a lock
    name; env overrides are what actually differ between concurrent runs on one
    machine, and any unset value falls back to the same default string for both
    contenders — which is exactly when they *should* share a lock.
    """
    qdrant = (
        os.environ.get("ARC_QDRANT_URL")
        or os.environ.get("QDRANT_URL")
        or "http://localhost:6333"
    )
    meili = os.environ.get("MEILISEARCH_URL") or "http://localhost:7700"
    return f"{qdrant}|{meili}"


def corpus_lock_path(corpus: str) -> Path:
    """Return the lock file path for `corpus` against the current services."""
    digest = hashlib.sha256(f"{corpus}\x00{_service_endpoints()}".encode()).hexdigest()[:12]
    safe_name = _UNSAFE_NAME_CHARS.sub("_", corpus)[:64] or "corpus"
    return _locks_dir() / f"corpus-{safe_name}-{digest}.lock"


def read_lock_holder(lock_path: Path) -> Optional[Dict[str, Any]]:
    """Read the pid/start-time record the holder wrote, or None if unreadable.

    Purely diagnostic: the flock itself is the authority on who holds the lock.
    The file may be empty if a holder is between `flock` and its write.
    """
    try:
        raw = lock_path.read_text().strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _describe_holder(lock_path: Path, corpus: str) -> str:
    holder = read_lock_holder(lock_path)
    if not holder:
        return f"Another process is syncing corpus '{corpus}'."
    pid = holder.get("pid", "?")
    started = holder.get("started")
    if isinstance(started, (int, float)):
        elapsed = max(0.0, time.time() - started)
        return f"Another sync of corpus '{corpus}' is running (pid {pid}, {elapsed:.0f}s ago)."
    return f"Another sync of corpus '{corpus}' is running (pid {pid})."


def _write_holder_record(fd: int, corpus: str) -> None:
    record = json.dumps({"pid": os.getpid(), "corpus": corpus, "started": time.time()})
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, record.encode())
    except OSError as exc:  # diagnostics only — never fail the sync over this
        logger.debug("Could not write lock holder record: %s", exc)


@contextlib.contextmanager
def acquire_corpus_lock(
    corpus: str,
    *,
    wait: bool = True,
    timeout: Optional[float] = None,
    quiet: bool = False,
) -> Iterator[None]:
    """Hold the write lock for `corpus` for the duration of the block.

    Args:
        corpus: Corpus name to lock.
        wait: If False, raise CorpusLockUnavailable immediately when held.
        timeout: Seconds to wait before giving up (default 600).
        quiet: Suppress the "waiting for…" notice (e.g. under --json).

    Raises:
        CorpusLockUnavailable: the lock stayed held past the wait budget.

    Windows is not a supported target; on platforms without fcntl this is a
    no-op so library imports stay clean.
    """
    if platform.system() == "Windows":
        yield
        return

    import fcntl

    lock_path = corpus_lock_path(corpus)
    key = str(lock_path)

    # Already held higher in this call stack — the outer scope owns release.
    if key in _held:
        yield
        return

    wait_budget = timeout if timeout is not None else DEFAULT_LOCK_TIMEOUT_SECONDS
    deadline = time.monotonic() + (wait_budget if wait else 0.0)
    started = time.monotonic()
    notice_emitted = False

    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = _describe_holder(lock_path, corpus)
                if wait:
                    raise CorpusLockUnavailable(
                        f"{detail} Timed out after {wait_budget:.0f}s waiting for the "
                        "corpus lock. Retry once it finishes, or raise the budget "
                        "with --lock-timeout <seconds>."
                    )
                raise CorpusLockUnavailable(
                    f"{detail} Refusing to run concurrently (--no-wait). "
                    "Retry once it finishes, or drop --no-wait to queue behind it."
                )

            if not quiet and not notice_emitted and (time.monotonic() - started) > _WAIT_NOTICE_THRESHOLD_S:
                print(f"[INFO] {_describe_holder(lock_path, corpus)} Waiting…", file=sys.stderr)
                notice_emitted = True

            # Jittered backoff so several waiters do not wake in lockstep.
            time.sleep(min(remaining, 0.2 + random.uniform(0, 0.3)))

        _write_holder_record(fd, corpus)
        _held[key] = fd
        try:
            yield
        finally:
            _held.pop(key, None)
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError as exc:
                logger.debug("Could not unlock %s: %s", lock_path, exc)
    finally:
        os.close(fd)

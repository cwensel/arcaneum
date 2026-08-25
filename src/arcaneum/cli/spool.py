"""Maildir-style spool for hook-driven corpus syncs (kata vq0n).

A git hook must never block or fail the git operation that fired it, and it
cannot afford a full sync per commit: each `arc corpus sync` pays a fixed cold
start (Python, embedding model load, service connects) that dwarfs the work of
re-indexing one file. Firing a detached sync per commit also races itself on
fast successive commits and on `post-rewrite` after a rebase.

So the hook does the cheap half only: it writes the paths git reported into a
spool directory and tries to hand off to a background worker. Writes are
tmp-then-rename, so a reader never observes a partial entry. The worker holds a
per-corpus lock and drains until the spool is empty, which coalesces a burst of
commits into a single model load; if a worker is already running, the hook
leaves the entry for it and exits.

Layout::

    <data>/spool/<corpus>/<repo-hash>/<timestamp>-<pid>-<n>.json
    <data>/spool/<corpus>/worker.lock
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import platform
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from ..paths import get_data_dir

logger = logging.getLogger(__name__)

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]")
_counter = itertools.count()


@dataclass(frozen=True)
class SpoolBatch:
    """The union of every spool entry drained in one pass."""

    changed: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.changed or self.removed)


def _safe_name(name: str) -> str:
    return _UNSAFE_NAME_CHARS.sub("_", name)[:64] or "corpus"


def spool_root() -> Path:
    root = get_data_dir() / "spool"
    root.mkdir(parents=True, exist_ok=True)
    return root


def corpus_spool_dir(corpus: str) -> Path:
    """Directory holding every pending entry for `corpus`, across repos."""
    return spool_root() / _safe_name(corpus)


def repo_spool_dir(corpus: str, repo_root: Union[str, Path]) -> Path:
    """Per-repo subdirectory, so one repo can feed several corpora and vice versa."""
    digest = hashlib.sha256(str(repo_root).encode()).hexdigest()[:12]
    return corpus_spool_dir(corpus) / digest


def worker_lock_path(corpus: str) -> Path:
    corpus_spool_dir(corpus).mkdir(parents=True, exist_ok=True)
    return corpus_spool_dir(corpus) / "worker.lock"


def write_entry(
    corpus: str,
    repo_root: Union[str, Path],
    *,
    changed: Sequence[str],
    removed: Sequence[str],
) -> Optional[Path]:
    """Record one commit's paths for `corpus`. Returns the entry path, or None.

    Written to a temp name and renamed into place: rename is atomic within a
    directory, so a concurrently draining worker sees either nothing or a
    complete entry — never a truncated one it would have to discard.
    """
    if not changed and not removed:
        return None

    target_dir = repo_spool_dir(corpus, repo_root)
    target_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "corpus": corpus,
        "repo": str(repo_root),
        "changed": list(changed),
        "removed": list(removed),
        "written": time.time(),
        "pid": os.getpid(),
    }
    stem = f"{time.time():.6f}-{os.getpid()}-{next(_counter)}"
    tmp_path = target_dir / f".{stem}.tmp"
    final_path = target_dir / f"{stem}.json"

    tmp_path.write_text(json.dumps(record))
    tmp_path.rename(final_path)
    return final_path


def _entry_paths(corpus: str) -> List[Path]:
    root = corpus_spool_dir(corpus)
    if not root.is_dir():
        return []
    # Sorted so entries replay in the order they were written: a later commit's
    # delete must override an earlier commit's add for the same path.
    return sorted(root.rglob("*.json"))


def has_pending(corpus: str) -> bool:
    return bool(_entry_paths(corpus))


def drain_batch(corpus: str, *, consume: bool = True):
    """Read every pending entry for `corpus` and return their union.

    A path touched by several commits collapses to a single decision, with the
    newest entry winning — so a file added then deleted is not indexed, and one
    deleted then restored is.

    Entries that cannot be parsed are always discarded rather than retried
    forever: a malformed entry can never succeed, so keeping it would wedge the
    corpus. The next commit re-reports those paths anyway.

    Args:
        corpus: Corpus whose spool to read.
        consume: When True (default) entries are removed as they are read.
            Pass False when the caller may fail — indexing the batch can throw
            (services down, model load failure, OOM), and removing the entries
            up front would drop that work permanently with no retry. The caller
            then passes the returned entry paths to `release_entries` only once
            the batch has been indexed successfully.

    Returns:
        A SpoolBatch when ``consume`` is True; otherwise a
        ``(SpoolBatch, list[Path])`` pair of the union and the entries still
        on disk.
    """
    verdicts: Dict[str, bool] = {}  # path -> is_removed
    pending: List[Path] = []

    for entry_path in _entry_paths(corpus):
        try:
            raw = entry_path.read_text()
        except OSError as exc:
            logger.debug("Could not read spool entry %s: %s", entry_path, exc)
            continue

        try:
            record = json.loads(raw)
            changed = record["changed"]
            removed = record["removed"]
            if not isinstance(changed, list) or not isinstance(removed, list):
                raise ValueError("changed/removed must be lists")
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Discarding malformed spool entry %s: %s", entry_path, exc)
            _unlink_quietly(entry_path)
            continue

        for path in changed:
            verdicts[str(path)] = False
        for path in removed:
            verdicts[str(path)] = True

        if consume:
            _unlink_quietly(entry_path)
        else:
            pending.append(entry_path)

    batch = SpoolBatch(
        changed=[p for p, removed in verdicts.items() if not removed],
        removed=[p for p, removed in verdicts.items() if removed],
    )
    return batch if consume else (batch, pending)


def release_entries(entries: Iterable[Path]) -> None:
    """Remove spool entries a caller has finished with. Never raises."""
    for entry_path in entries:
        _unlink_quietly(entry_path)


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError as exc:
        logger.debug("Could not remove spool entry %s: %s", path, exc)


def try_acquire_worker_lock(corpus: str) -> Optional[int]:
    """Take the single-flight worker lock for `corpus`, or None if held.

    Non-blocking on purpose: the caller is either a hook that must exit
    immediately, or a worker that should decline when another is already
    draining. flock is released by the OS if the worker dies, so a crashed
    drain never wedges the corpus.

    Returns an fd to pass to `release_worker_lock`.
    """
    if platform.system() == "Windows":
        return None

    import fcntl

    path = worker_lock_path(corpus)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None

    try:
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"pid": os.getpid(), "started": time.time()}).encode())
    except OSError as exc:
        logger.debug("Could not stamp worker lock: %s", exc)
    return fd


def release_worker_lock(fd: Optional[int]) -> None:
    if fd is None:
        return
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
    except (ImportError, OSError) as exc:
        logger.debug("Could not release worker lock: %s", exc)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def list_corpora_with_pending() -> List[str]:
    """Corpus names that currently have spooled work, for status reporting."""
    root = spool_root()
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and any(d.rglob("*.json")))

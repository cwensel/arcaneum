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

    <data>/spool/<corpus>/queue/<timestamp>-<pid>-<n>.<repo-id>.json
    <data>/spool/<corpus>/claimed/   entries a worker is indexing
    <data>/spool/<corpus>/deferred/  entries waiting for their branch
    <data>/spool/<corpus>/worker.lock
    <data>/hook-marks/<corpus>/<repo-id>/<branch>

`queue/` is what the `--service` watcher watches (kata 4j93). launchd's
QueueDirectories and systemd's DirectoryNotEmpty= relaunch the drain for as
long as it holds anything at all, so it holds entries and nothing else: temp
files, the lock, claimed and deferred entries all live beside it. It is empty
exactly when there is nothing a drain could do.

A mark records the last commit of a pinned branch whose changes were spooled
(kata vr9b). It lives outside the spool so nothing that scans for pending work
can mistake it for an entry.
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
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from ..paths import get_data_dir
from .git_changes import current_branch
from .git_changes import repo_root as git_repo_root

logger = logging.getLogger(__name__)

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]")
_counter = itertools.count()


@dataclass(frozen=True)
class SpoolBatch:
    """The union of every spool entry drained in one pass."""

    changed: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    # Repo -> branch for every pinned entry in the batch, so the worker can
    # check after indexing that each repo stayed on the branch it read.
    pins: Dict[str, str] = field(default_factory=dict)

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


def repo_id(repo_root: Union[str, Path]) -> str:
    """Short stable id for a repo, carried in entry names and mark paths."""
    return hashlib.sha256(str(repo_root).encode()).hexdigest()[:12]


_repo_digest = repo_id

_QUEUE = "queue"
_CLAIMED = "claimed"
_DEFERRED = "deferred"
# A pre-queue-directory spool kept one subdirectory per repo, named by repo id.
_LEGACY_REPO_DIR = re.compile(r"[0-9a-f]{12}")


def queue_relpath(corpus: str) -> Path:
    """`queue_dir` relative to the data dir, for the hook to rebuild at run time."""
    return Path("spool") / _safe_name(corpus) / _QUEUE


def deferred_relpath(corpus: str) -> Path:
    """`deferred_dir` relative to the data dir, for the hook to rebuild at run time."""
    return Path("spool") / _safe_name(corpus) / _DEFERRED


def queue_dir(corpus: str) -> Path:
    """Entries waiting to be drained, and nothing else. The watched directory."""
    return get_data_dir() / queue_relpath(corpus)


def claimed_dir(corpus: str) -> Path:
    return corpus_spool_dir(corpus) / _CLAIMED


def deferred_dir(corpus: str) -> Path:
    return get_data_dir() / deferred_relpath(corpus)


def worker_lock_path(corpus: str) -> Path:
    corpus_spool_dir(corpus).mkdir(parents=True, exist_ok=True)
    return corpus_spool_dir(corpus) / "worker.lock"


def write_entry(
    corpus: str,
    repo_root: Union[str, Path],
    *,
    changed: Sequence[str],
    removed: Sequence[str],
    branch: Optional[str] = None,
) -> Optional[Path]:
    """Record one commit's paths for `corpus`. Returns the entry path, or None.

    Written to a temp name and renamed into place: rename is atomic within a
    directory, so a concurrently draining worker sees either nothing or a
    complete entry — never a truncated one it would have to discard.

    `branch` pins the entry: the worker indexes it only while `repo_root` has
    that branch checked out, since indexing reads the working tree.
    """
    if not changed and not removed:
        return None

    target_dir = queue_dir(corpus)
    target_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "corpus": corpus,
        "repo": str(repo_root),
        "changed": list(changed),
        "removed": list(removed),
        "branch": branch,
        "written": time.time(),
        "pid": os.getpid(),
    }
    stem = f"{time.time():.6f}-{os.getpid()}-{next(_counter)}.{repo_id(repo_root)}"
    # The temp file sits beside the queue, not in it: the watcher would wake
    # for it. Same filesystem, so the rename is still atomic.
    tmp_path = target_dir.parent / f".{stem}.tmp"
    final_path = target_dir / f"{stem}.json"

    tmp_path.write_text(json.dumps(record))
    tmp_path.rename(final_path)
    return final_path


def _entries_in(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    # Sorted so entries replay in the order they were written: a later commit's
    # delete must override an earlier commit's add for the same path. Names
    # start with the write time, so the order holds across repos and survives
    # an entry being claimed, deferred, and returned.
    return sorted(directory.glob("*.json"))


def _entry_paths(corpus: str) -> List[Path]:
    return _entries_in(queue_dir(corpus))


def _has_entries(corpus_dir: Path) -> bool:
    if not corpus_dir.is_dir():
        return False
    for child in corpus_dir.iterdir():
        holds_entries = child.name in (_QUEUE, _CLAIMED, _DEFERRED) or bool(
            _LEGACY_REPO_DIR.fullmatch(child.name)
        )
        if holds_entries and child.is_dir() and any(child.glob("*.json")):
            return True
    return False


def has_pending(corpus: str) -> bool:
    """True while any entry is not yet indexed: queued, being indexed, or deferred."""
    return _has_entries(corpus_spool_dir(corpus))


def _move(entry: Path, directory: Path) -> Optional[Path]:
    """Rename `entry` into `directory`, keeping its name. None if it vanished."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / entry.name
    try:
        entry.rename(target)
    except FileNotFoundError:
        return None
    return target


def _sweep_queue(corpus: str) -> None:
    """Move old-layout entries into the queue and drop anything that is not an entry."""
    import shutil

    root = corpus_spool_dir(corpus)
    if not root.is_dir():
        return
    queue = queue_dir(corpus)
    for child in root.iterdir():
        if not (child.is_dir() and _LEGACY_REPO_DIR.fullmatch(child.name)):
            continue
        for entry in child.glob("*.json"):
            queue.mkdir(parents=True, exist_ok=True)
            entry.rename(queue / f"{entry.stem}.{child.name}.json")
        shutil.rmtree(child, ignore_errors=True)

    if not queue.is_dir():
        return
    for child in queue.iterdir():
        if child.is_file() and child.suffix == ".json":
            continue
        logger.warning("Removing %s from the spool queue: it is not an entry", child)
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            _unlink_quietly(child)


def recover_claimed(corpus: str) -> None:
    """Return entries a dead worker had claimed to the queue.

    Call only while holding the worker lock: then no live worker owns them.
    """
    for entry in _entries_in(claimed_dir(corpus)):
        _move(entry, queue_dir(corpus))


def return_entries(entries: Iterable[Path]) -> None:
    """Put claimed entries back in the queue, for a later drain to retry."""
    for entry in entries:
        # A claimed entry sits in <corpus>/claimed/; its queue is <corpus>/queue/.
        _move(entry, entry.parent.parent / _QUEUE)


def requeue_deferred(corpus: str, repo_root: Union[str, Path]) -> None:
    """Return one repo's deferred entries to the queue (its branch is back)."""
    suffix = f".{repo_id(repo_root)}.json"
    for entry in _entries_in(deferred_dir(corpus)):
        if entry.name.endswith(suffix):
            _move(entry, queue_dir(corpus))


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
            up front would drop that work permanently with no retry. The
            entries are then claimed (moved out of the queue, so the watcher
            does not relaunch a drain for work already in hand), and the caller
            passes them to `release_entries` once the batch is indexed, or to
            `return_entries` to retry it later.

    Entries pinned to a branch their repo does not have checked out are moved
    to the deferred directory; `requeue_deferred` brings them back.

    Returns:
        A SpoolBatch when ``consume`` is True; otherwise a
        ``(SpoolBatch, list[Path])`` pair of the union and the claimed entries.
    """
    verdicts: Dict[str, bool] = {}  # path -> is_removed
    pending: List[Path] = []
    pins: Dict[str, str] = {}
    on_branch: Dict[tuple, bool] = {}

    def _on_branch(repo: str, branch: str) -> bool:
        if (repo, branch) not in on_branch:
            on_branch[(repo, branch)] = repo_is_on_branch(repo, branch)
        return on_branch[(repo, branch)]

    _sweep_queue(corpus)
    # Deferred entries whose branch is back join this pass.
    for entry_path in _entries_in(deferred_dir(corpus)):
        record = _read_record(entry_path)
        # Anything unreadable or unpinned goes back too: the loop below
        # discards or drains it rather than leaving it parked forever.
        if record is None or not record.get("branch"):
            _move(entry_path, queue_dir(corpus))
        elif _on_branch(str(record.get("repo", "")), record["branch"]):
            _move(entry_path, queue_dir(corpus))

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

        branch = record.get("branch")
        repo = str(record.get("repo", ""))
        if branch:
            if not _on_branch(repo, branch):
                # Parked out of the queue so the watcher goes quiet; the hook
                # requeues it when the branch is checked out again.
                _move(entry_path, deferred_dir(corpus))
                continue
            pins[repo] = branch

        for path in changed:
            verdicts[str(path)] = False
        for path in removed:
            verdicts[str(path)] = True

        if consume:
            _unlink_quietly(entry_path)
        else:
            claimed = _move(entry_path, claimed_dir(corpus))
            if claimed is not None:
                pending.append(claimed)

    batch = SpoolBatch(
        changed=[p for p, removed in verdicts.items() if not removed],
        removed=[p for p, removed in verdicts.items() if removed],
        pins=pins,
    )
    return batch if consume else (batch, pending)


def _read_record(entry_path: Path) -> Optional[dict]:
    try:
        record = json.loads(entry_path.read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def repo_is_on_branch(repo: Union[str, Path], branch: str) -> bool:
    """True when `repo` has `branch` checked out, or when that cannot be known.

    A repo that is gone or no longer a repository answers True: its entries can
    never become indexable by waiting, so deferring them would wedge the spool.
    """
    if current_branch(repo) == branch:
        return True
    return not Path(repo).is_dir() or git_repo_root(repo) is None


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
    return sorted(d.name for d in root.iterdir() if d.is_dir() and _has_entries(d))


# --- marks: the last spooled commit of a pinned branch (kata vr9b) -----------

MARKS_DIRNAME = "hook-marks"


def _branch_filename(branch: str) -> str:
    # The digest keeps `a/b` and `a_b` apart once both flatten to `a_b`.
    return f"{_safe_name(branch)}-{hashlib.sha256(branch.encode()).hexdigest()[:8]}"


def _repo_marks_relpath(corpus: str, repo_root: Union[str, Path]) -> Path:
    return Path(MARKS_DIRNAME) / _safe_name(corpus) / _repo_digest(repo_root)


def mark_relpath(corpus: str, repo_root: Union[str, Path], branch: str) -> Path:
    """`mark_path` relative to the data dir, for the hook to rebuild at run time."""
    return _repo_marks_relpath(corpus, repo_root) / _branch_filename(branch)


def mark_path(corpus: str, repo_root: Union[str, Path], branch: str) -> Path:
    return get_data_dir() / mark_relpath(corpus, repo_root, branch)


def read_mark(corpus: str, repo_root: Union[str, Path], branch: str) -> Optional[str]:
    """The last commit spooled for this branch, or None if none was recorded."""
    try:
        value = mark_path(corpus, repo_root, branch).read_text().strip()
    except OSError:
        return None
    return value or None


def write_mark(corpus: str, repo_root: Union[str, Path], branch: str, commit: str) -> None:
    """Record `commit` as spooled. Atomic, since the hook reads it unlocked."""
    path = mark_path(corpus, repo_root, branch)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    # Newline-terminated: the hook reads it with `read`, which reports failure
    # on a final line without one.
    tmp_path.write_text(f"{commit}\n")
    tmp_path.rename(path)


def clear_marks(corpus: str, repo_root: Union[str, Path]) -> None:
    """Forget every branch's mark for this corpus and repo."""
    import shutil

    shutil.rmtree(get_data_dir() / _repo_marks_relpath(corpus, repo_root), ignore_errors=True)


@contextmanager
def mark_lock(corpus: str, repo_root: Union[str, Path], branch: str):
    """Serialize read-diff-spool-advance for one mark.

    Several hooks fire for one git operation (a commit runs reference-transaction
    and post-commit), so two spoolers can read the same mark at once. Holding
    this lock makes the second see the advanced mark and spool nothing.
    """
    path = mark_path(corpus, repo_root, branch)
    path.parent.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Windows":
        yield
        return

    import fcntl

    fd = os.open(path.with_name(f".{path.name}.lock"), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)

"""`arc corpus jobs`: which indexing work is running now, and what is queued.

Running work is read from the holder records that the corpus write lock
(sync, repair, delete, parity) and the drain worker lock stamp on acquire and
clear on release. A record names a pid; it counts as running only while that
pid is alive and was created no later than the record. A crashed holder leaves
its record behind, and its pid may since belong to an unrelated process.

Nothing here takes a lock, not even briefly. The drain worker's lock is
single-try: a probe holding it at the instant a hook spawns a worker makes that
worker decline, stranding the queued work until the next commit.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import click
import psutil

from ..paths import get_data_dir
from ..utils.formatting import format_duration
from . import spool

# Clock granularity between psutil's create_time and the record's time.time().
_CREATE_TIME_SLACK_S = 1.0


@dataclass(frozen=True)
class Job:
    corpus: str
    kind: str  # "write" (corpus lock) or "drain" (spool worker)
    pid: int
    started: float
    command: Optional[str]
    endpoints: Optional[str] = None

    @property
    def elapsed(self) -> float:
        return max(0.0, time.time() - self.started)


@dataclass(frozen=True)
class QueuedWork:
    corpus: str  # spool directory name
    queued: int
    claimed: int
    deferred: int
    # Claimed by a drain that is gone; the next drain reclaims them.
    orphaned: bool


def _read_record(path: Path) -> Optional[Dict[str, Any]]:
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(record, dict):
        return None
    if not isinstance(record.get("pid"), int) or not isinstance(
        record.get("started"), (int, float)
    ):
        return None
    return record


def _live_title(pid: int, started: float) -> Optional[str]:
    """The process title of `pid` if it is the record's holder, else None.

    Returns "" for a live holder whose title cannot be read.
    """
    try:
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return None
        if proc.create_time() > started + _CREATE_TIME_SLACK_S:
            return None
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None
    except psutil.AccessDenied:
        return ""
    try:
        return " ".join(proc.cmdline())
    except psutil.Error:
        return ""


def _job_from(path: Path, kind: str, default_corpus: str) -> Optional[Job]:
    record = _read_record(path)
    if record is None:
        return None
    title = _live_title(record["pid"], record["started"])
    if title is None:
        return None
    # A title set by setproctitle is live (drains report their batch); without
    # the extra, the cmdline is just the interpreter, so the record reads better.
    if title == "arc" or title.startswith("arc "):
        command = title
    else:
        command = record.get("command") or title or None
    return Job(
        corpus=record.get("corpus") or default_corpus,
        kind=kind,
        pid=record["pid"],
        started=float(record["started"]),
        command=command,
        endpoints=record.get("endpoints"),
    )


def running_jobs() -> List[Job]:
    """Every live holder of a corpus write lock or drain worker lock, oldest first."""
    data_dir = get_data_dir()
    found: List[Job] = []

    spool_root = data_dir / "spool"
    if spool_root.is_dir():
        for lock_path in sorted(spool_root.glob("*/worker.lock")):
            job = _job_from(lock_path, "drain", lock_path.parent.name)
            if job is not None:
                found.append(job)

    # A drain syncs under the corpus lock too; list it once, as the drain.
    drains = {(job.pid, spool.spool_dirname(job.corpus)) for job in found}
    locks_dir = data_dir / "locks"
    if locks_dir.is_dir():
        for lock_path in sorted(locks_dir.glob("corpus-*.lock")):
            job = _job_from(lock_path, "write", lock_path.stem)
            if job is not None and (job.pid, spool.spool_dirname(job.corpus)) not in drains:
                found.append(job)

    return sorted(found, key=lambda job: job.started)


def queued_work(running: Iterable[Job]) -> List[QueuedWork]:
    """Hook work waiting in the spool, per corpus."""
    draining = {spool.spool_dirname(job.corpus) for job in running if job.kind == "drain"}
    return [
        QueuedWork(
            corpus=name,
            queued=counts["queued"],
            claimed=counts["claimed"],
            deferred=counts["deferred"],
            orphaned=counts["claimed"] > 0 and name not in draining,
        )
        for name, counts in spool.pending_counts().items()
    ]


def _describe_queued(work: QueuedWork) -> str:
    parts = []
    if work.queued:
        parts.append(f"{work.queued} queued")
    if work.claimed:
        orphaned = " (orphaned; the next drain reclaims them)" if work.orphaned else ""
        parts.append(f"{work.claimed} claimed{orphaned}")
    if work.deferred:
        parts.append(f"{work.deferred} deferred until their branch is checked out")
    return f"  {work.corpus}: {', '.join(parts)}"


def jobs_command(output_json: bool) -> None:
    from .output import print_json

    running = running_jobs()
    queued = queued_work(running)

    if output_json:
        print_json(
            "success",
            f"{len(running)} job(s) running",
            data={
                "jobs": [
                    {**asdict(job), "elapsed_seconds": round(job.elapsed, 1)} for job in running
                ],
                "queued": [asdict(work) for work in queued],
            },
        )
        return

    if not running:
        click.echo("No indexing jobs running.")
    else:
        rows = [("CORPUS", "KIND", "PID", "ELAPSED", "COMMAND")] + [
            (job.corpus, job.kind, str(job.pid), format_duration(job.elapsed), job.command or "?")
            for job in running
        ]
        widths = [max(len(row[i]) for row in rows) for i in range(4)]
        for row in rows:
            cells = [cell.ljust(width) for cell, width in zip(row, widths)]
            click.echo("  ".join(cells + [row[4]]))

    if queued:
        click.echo("\nQueued hook work:")
        for work in queued:
            click.echo(_describe_queued(work))

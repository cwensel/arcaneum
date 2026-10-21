"""Stop a spool drain that has stopped making progress.

A drain holds the corpus's worker lock until the spool is empty, so a drain
that hangs wedges every hook-queued change behind it. One sat for 25h at zero
CPU with its memory swapped out while 70 entries queued. A drain cannot police
itself (a starved process starves its own watchdog thread too), so whoever next
finds the lock held observes the holder instead: the hook, a declined drain,
`arc corpus jobs`.

The signal is CPU time, the holder's plus its children's. A slow drain keeps
accruing it; a stalled one does not. Each observation is recorded beside the
lock, and a holder that gains less than MIN_PROGRESS_CPU_S over STALL_AFTER_S is
stalled. Every bounded wait a drain makes (corpus lock, remote embedding,
MeiliSearch tasks) times out well inside that window.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import psutil

from . import jobs, spool

STALL_AFTER_S = 15 * 60
# CPU seconds that count as progress; a stuck process still spends a little on
# signals and page faults.
MIN_PROGRESS_CPU_S = 1.0


@dataclass(frozen=True)
class Observation:
    pid: int
    started: float
    cpu_s: float
    # How long the holder has gone without progress.
    idle_s: float

    @property
    def stalled(self) -> bool:
        return self.idle_s >= STALL_AFTER_S


def _tree_cpu_seconds(proc: psutil.Process) -> float:
    times = proc.cpu_times()
    total = times.user + times.system
    for child in proc.children(recursive=True):
        try:
            child_times = child.cpu_times()
        except psutil.Error:
            continue  # exited since listed
        total += child_times.user + child_times.system
    return total


def _read_watch(corpus: str) -> Optional[dict]:
    try:
        record = json.loads(spool.worker_watch_path(corpus).read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _write_watch(corpus: str, record: dict) -> None:
    path = spool.worker_watch_path(corpus)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(path)


def observe(corpus: str, *, now: Optional[float] = None) -> Optional[Observation]:
    """Sample the drain holding `corpus`'s worker lock. None when no live drain holds it."""
    now = time.time() if now is None else now
    record = jobs.read_holder_record(spool.worker_lock_path(corpus))
    proc = None
    if record is not None:
        try:
            proc = jobs.holder_process(record["pid"], record["started"])
            cpu_s = _tree_cpu_seconds(proc) if proc is not None else 0.0
        except psutil.Error:
            proc = None
    if proc is None:
        spool.worker_watch_path(corpus).unlink(missing_ok=True)
        return None

    pid, started = record["pid"], float(record["started"])
    previous = _read_watch(corpus) or {}
    baseline = previous.get("cpu")
    if (
        previous.get("pid") == pid
        and previous.get("started") == started
        and isinstance(baseline, (int, float))
        and 0 <= cpu_s - baseline < MIN_PROGRESS_CPU_S
        and isinstance(previous.get("since"), (int, float))
    ):
        since = float(previous["since"])
    else:
        # First sight of this holder, or it progressed: restart the window.
        since, baseline = now, cpu_s
    _write_watch(corpus, {"pid": pid, "started": started, "cpu": baseline, "since": since})
    return Observation(pid=pid, started=started, cpu_s=cpu_s, idle_s=now - since)


def stop(observation: Observation, *, grace_s: float = 10.0) -> bool:
    """Stop the observed holder and its children. True once it is gone.

    SIGTERM first, SIGKILL after `grace_s`. A pid that no longer belongs to the
    holder counts as gone and is never signalled.
    """
    try:
        proc = jobs.holder_process(observation.pid, observation.started)
        if proc is None:
            return True
        family = [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return False
    for member in family:
        try:
            member.terminate()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            return False
    _, alive = psutil.wait_procs(family, timeout=grace_s)
    for member in alive:
        try:
            member.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            return False
    _, alive = psutil.wait_procs(alive, timeout=grace_s)
    return proc not in alive

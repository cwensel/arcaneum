"""--background: run a sync or repair below interactive work on this machine.

macOS: the arc process drops to nice 10, which the worker processes it starts
inherit. CPU priority only: the darwin background band also throttles disk I/O,
swap page-ins included, so under memory pressure a drain in it can stall
indefinitely. The Docker VM is left alone; a renice would outlive the run and
slow every later search.

Linux: every arc thread takes SCHED_IDLE at nice 19. Docker is left alone: its
containers sit in system.slice, where a container weight does not rank them
against processes in a user session.

Nothing is restored on exit. An unprivileged process cannot raise a priority
again.
"""

from __future__ import annotations

import os
import sys

MACOS_NICE = 10
LINUX_NICE = 19


def enter_background(platform: str = sys.platform) -> list[str]:
    """Lower this run's priority; returns one note per change or skip."""
    if platform == "darwin":
        return [_darwin_nice()]
    if platform.startswith("linux"):
        return [_linux_idle()]
    return [f"--background is not supported on {platform}; priority unchanged"]


def _darwin_nice() -> str:
    current = os.getpriority(os.PRIO_PROCESS, 0)
    if current >= MACOS_NICE:
        return f"arc already at nice {current}"
    os.setpriority(os.PRIO_PROCESS, 0, MACOS_NICE)
    return f"arc runs at nice {MACOS_NICE}; Docker unchanged"


def _thread_ids() -> list[int]:
    return [int(tid) for tid in os.listdir("/proc/self/task")]


def _linux_idle() -> str:
    # Linux priorities are per thread; threads and processes started later inherit.
    for tid in _thread_ids():
        try:
            os.sched_setscheduler(tid, os.SCHED_IDLE, os.sched_param(0))
            if os.getpriority(os.PRIO_PROCESS, tid) < LINUX_NICE:
                os.setpriority(os.PRIO_PROCESS, tid, LINUX_NICE)
        except OSError:
            pass  # the thread exited
    return f"arc runs at SCHED_IDLE, nice {LINUX_NICE}; Docker unchanged"

"""--background: run a sync or repair below interactive work on this machine.

macOS: the arc process takes the darwin background policy, which the worker
processes it starts inherit: efficiency cores only, throttled disk I/O. The
Docker Desktop VM drops to nice 5, just below interactive work such as
compilers and tests, so Qdrant and MeiliSearch keep up with concurrent searches.

Linux: every arc thread takes SCHED_IDLE at nice 19. Docker is left alone: its
containers sit in system.slice, where a container weight does not rank them
against processes in a user session.

Nothing is restored on exit. An unprivileged process cannot raise a priority
again, and the VM returns to normal when Docker Desktop restarts.
"""

from __future__ import annotations

import os
import subprocess
import sys

PRIO_DARWIN_PROCESS = 4
PRIO_DARWIN_BG = 0x1000
DOCKER_VM_NICE = 5
DOCKER_VM_PROCESS = "com.apple.Virtualization.VirtualMachine"
DOCKER_DESKTOP_PROCESS = "com.docker.virtualization"
LINUX_NICE = 19


def enter_background(platform: str = sys.platform) -> list[str]:
    """Lower this run's priority; returns one note per change or skip."""
    if platform == "darwin":
        return [_darwin_background(), _lower_docker_vm()]
    if platform.startswith("linux"):
        return [_linux_idle()]
    return [f"--background is not supported on {platform}; priority unchanged"]


def _darwin_background() -> str:
    os.setpriority(PRIO_DARWIN_PROCESS, 0, PRIO_DARWIN_BG)
    return "arc runs at background priority (efficiency cores, throttled disk I/O)"


def _pids(pattern: str) -> list[int]:
    result = subprocess.run(
        ["pgrep", "-U", str(os.getuid()), "-f", pattern],
        capture_output=True,
        text=True,
        check=False,
    )
    return [int(line) for line in result.stdout.split()]


def _lower_docker_vm() -> str:
    if not _pids(DOCKER_DESKTOP_PROCESS):
        return "Docker VM unchanged: Docker Desktop is not running"
    vms = _pids(DOCKER_VM_PROCESS)
    if len(vms) != 1:
        # Another virtual machine may be running; renice none rather than guess.
        return f"Docker VM unchanged: found {len(vms)} virtual machines, expected one"
    (pid,) = vms
    try:
        current = os.getpriority(os.PRIO_PROCESS, pid)
        if current >= DOCKER_VM_NICE:
            return f"Docker VM already at nice {current}"
        os.setpriority(os.PRIO_PROCESS, pid, DOCKER_VM_NICE)
    except OSError as exc:
        return f"Docker VM unchanged: {exc}"
    return f"Docker VM lowered to nice {DOCKER_VM_NICE} until Docker Desktop restarts"


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

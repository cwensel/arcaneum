"""A drain that makes no CPU progress is stopped, so the spool cannot wedge.

Seen in the wild: a hook drain sat for 25h at zero CPU, its memory swapped out,
holding the worker lock while 70 entries queued behind it. A slow drain keeps
accruing CPU time; a stalled one does not, so CPU progress between observations
is the signal.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time

import psutil
import pytest

from arcaneum.cli import drain_watchdog, spool

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="flock-based spool worker is not supported on Windows",
)

WINDOW = drain_watchdog.STALL_AFTER_S


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    yield tmp_path


@pytest.fixture
def holder(isolated):
    """A live process recorded as the Docs worker-lock holder."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    lock = spool.worker_lock_path("Docs")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": proc.pid, "corpus": "Docs", "started": time.time()}))
    yield proc
    proc.kill()
    proc.wait()


@pytest.fixture
def cpu(monkeypatch):
    """Controls the CPU seconds the watchdog reads for any holder."""
    reading = {"seconds": 100.0}
    monkeypatch.setattr(drain_watchdog, "_tree_cpu_seconds", lambda proc: reading["seconds"])
    return reading


def test_first_sight_of_a_holder_is_not_a_stall(holder, cpu):
    observation = drain_watchdog.observe("Docs", now=1000.0)

    assert observation.pid == holder.pid
    assert observation.idle_s == 0
    assert not observation.stalled


def test_a_holder_without_cpu_progress_for_the_window_is_stalled(holder, cpu):
    drain_watchdog.observe("Docs", now=1000.0)

    observation = drain_watchdog.observe("Docs", now=1000.0 + WINDOW)

    assert observation.idle_s == WINDOW
    assert observation.stalled


def test_a_holder_short_of_the_window_is_not_stalled(holder, cpu):
    drain_watchdog.observe("Docs", now=1000.0)

    assert not drain_watchdog.observe("Docs", now=1000.0 + WINDOW - 1).stalled


def test_cpu_progress_restarts_the_window(holder, cpu):
    drain_watchdog.observe("Docs", now=1000.0)
    cpu["seconds"] += 30

    observation = drain_watchdog.observe("Docs", now=1000.0 + WINDOW)

    assert observation.idle_s == 0
    assert not observation.stalled


def test_a_trickle_of_cpu_does_not_hide_a_stall(holder, cpu):
    # Signal delivery and the odd page fault cost a little CPU even when stuck.
    drain_watchdog.observe("Docs", now=1000.0)
    for step in range(1, 4):
        cpu["seconds"] += drain_watchdog.MIN_PROGRESS_CPU_S / 4
        observation = drain_watchdog.observe("Docs", now=1000.0 + step * WINDOW / 3)

    assert observation.stalled


def test_cpu_going_backwards_restarts_the_window(holder, cpu):
    # A worker child exiting takes its CPU time out of the tree total.
    drain_watchdog.observe("Docs", now=1000.0)
    cpu["seconds"] -= 50

    assert drain_watchdog.observe("Docs", now=1000.0 + WINDOW).idle_s == 0


def test_a_new_holder_restarts_the_window(holder, cpu, isolated):
    drain_watchdog.observe("Docs", now=1000.0)
    successor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    try:
        spool.worker_lock_path("Docs").write_text(
            json.dumps({"pid": successor.pid, "corpus": "Docs", "started": time.time()})
        )

        observation = drain_watchdog.observe("Docs", now=1000.0 + WINDOW)
    finally:
        successor.kill()
        successor.wait()

    assert observation.pid == successor.pid
    assert not observation.stalled


def test_no_live_holder_is_no_observation_and_clears_the_record(holder, cpu):
    drain_watchdog.observe("Docs", now=1000.0)
    holder.kill()
    holder.wait()

    assert drain_watchdog.observe("Docs", now=1000.0 + WINDOW) is None
    assert not spool.worker_watch_path("Docs").exists()


def test_an_unstamped_lock_is_no_observation(isolated, cpu):
    assert drain_watchdog.observe("Docs", now=1000.0) is None


def test_a_watch_file_the_hook_touched_counts_as_first_sight(holder, cpu):
    # The hook touches the watch file to rate-limit its checks; it may be empty.
    spool.worker_watch_path("Docs").touch()

    observation = drain_watchdog.observe("Docs", now=1000.0)

    assert observation.idle_s == 0


def _alive(pid: int) -> bool:
    # stop() reaps the test's own child, so Popen cannot report how it ended.
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _holder_process(script: str) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    lock = spool.worker_lock_path("Docs")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": proc.pid, "corpus": "Docs", "started": time.time()}))
    return proc


def test_stop_terminates_a_stalled_holder(isolated, cpu):
    proc = _holder_process("import time; print('ready', flush=True); time.sleep(600)")
    observation = drain_watchdog.observe("Docs", now=1000.0)

    assert drain_watchdog.stop(observation, grace_s=5)
    assert not _alive(proc.pid)


def test_stop_kills_a_holder_that_ignores_sigterm(isolated, cpu):
    proc = _holder_process(
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); time.sleep(600)"
    )
    observation = drain_watchdog.observe("Docs", now=1000.0)

    # Gone despite ignoring SIGTERM: only SIGKILL could have done it.
    assert drain_watchdog.stop(observation, grace_s=0.5)
    assert not _alive(proc.pid)


def test_stop_also_stops_the_holders_children(isolated, cpu):
    proc = _holder_process(
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']); "
        "print('ready', flush=True); time.sleep(600)"
    )
    observation = drain_watchdog.observe("Docs", now=1000.0)
    children = psutil.Process(proc.pid).children()
    assert children

    assert drain_watchdog.stop(observation, grace_s=5)
    _, alive = psutil.wait_procs(children, timeout=5)
    assert not alive


def test_stop_never_signals_a_process_that_reused_the_holders_pid(isolated, cpu):
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    try:
        # A record stamped before the process existed: the pid was recycled.
        stale = drain_watchdog.Observation(
            pid=bystander.pid, started=time.time() - 3600, cpu_s=0.0, idle_s=WINDOW
        )

        assert drain_watchdog.stop(stale, grace_s=0.5)
        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait()

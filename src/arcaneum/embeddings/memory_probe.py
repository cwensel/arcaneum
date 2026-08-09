"""Lightweight memory/thread diagnostics for the embedding sync loop.

The corpus-sync process can grow monotonically on long unattended runs —
usually a combination of the MPS allocator not releasing buffers between
files, the GPU-poisoning daemon thread holding a closure ref to the GPU
model, and accumulated Python refs across many files. When it finally hits
the macOS memory pressure ceiling the process gets SIGKILLed by jetsam,
or (worse) the system pages itself into a kernel watchdog panic.

This module gives us observability without changing any behavior:

- `snapshot()` captures parent/worker RSS, cached child MPS allocator state,
  system/swap pressure, scratch-disk capacity, thread count, and Python objects.
- `format_snapshot_delta()` renders a compact one-line delta suitable for
  per-file verbose logging.
- `install_dump_handler()` wires SIGUSR1 to dump a full snapshot plus
  thread stack traces to stderr. Use this during a suspected hang:
  `kill -USR1 <pid>`.

Nothing here blocks, raises on the happy path, or allocates significant
memory of its own — so it's safe to call in the hot loop.
"""

from __future__ import annotations

import datetime
import faulthandler
import gc
import json
import os
import shutil
import signal
import sys
import tempfile
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import psutil

_BYTES_PER_GB = 1024**3
_DISK_WARNING_FLOOR = 10 * _BYTES_PER_GB
_DISK_CRITICAL_FLOOR = 1 * _BYTES_PER_GB
_MEMORY_WARNING_FLOOR = 2 * _BYTES_PER_GB


@dataclass
class MemorySnapshot:
    rss_bytes: int
    vsz_bytes: int
    thread_count: int
    gc_objects: int
    mps_current_bytes: Optional[int] = None
    mps_driver_bytes: Optional[int] = None
    mps_recommended_max_bytes: Optional[int] = None
    system_available_bytes: int = 0
    system_total_bytes: int = 0
    active_accelerator_workers: int = 0  # from EmbeddingClient, if passed in
    accelerator_worker_pids: tuple[int, ...] = ()
    accelerator_rss_bytes: int = 0
    swap_used_bytes: int = 0
    swap_total_bytes: int = 0
    disk_free_bytes: int = 0
    disk_total_bytes: int = 0
    disk_path: str = ""

    def delta(self, prev: "MemorySnapshot") -> dict:
        # Δdrv/Δsys_used are the actual leak signal on Apple Silicon — Metal
        # driver allocations don't show up in rss, and sys_used jumping while
        # rss is flat is the fingerprint of unified-memory pressure that
        # culminates in jetsam SIGKILL.
        sys_used_now = self.system_total_bytes - self.system_available_bytes
        sys_used_prev = prev.system_total_bytes - prev.system_available_bytes
        return {
            "rss": self.rss_bytes - prev.rss_bytes,
            "vsz": self.vsz_bytes - prev.vsz_bytes,
            "threads": self.thread_count - prev.thread_count,
            "gc": self.gc_objects - prev.gc_objects,
            "accelerator_rss": self.accelerator_rss_bytes - prev.accelerator_rss_bytes,
            "swap_used": self.swap_used_bytes - prev.swap_used_bytes,
            "mps_current": (
                None
                if self.mps_current_bytes is None or prev.mps_current_bytes is None
                else self.mps_current_bytes - prev.mps_current_bytes
            ),
            "mps_driver": (
                None
                if self.mps_driver_bytes is None or prev.mps_driver_bytes is None
                else self.mps_driver_bytes - prev.mps_driver_bytes
            ),
            "sys_used": sys_used_now - sys_used_prev,
            "disk_free": self.disk_free_bytes - prev.disk_free_bytes,
        }


def _mps_memory() -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Parent probes are runtime-neutral; native memory belongs to worker health."""
    return (None, None, None)


_PROC = psutil.Process(os.getpid())


def snapshot(embedding_client=None) -> MemorySnapshot:
    """Capture parent, accelerator, system-memory, and scratch-disk state."""
    mi = _PROC.memory_info()
    vm = psutil.virtual_memory()
    try:
        swap = psutil.swap_memory()
        swap_used = swap.used
        swap_total = swap.total
    except (OSError, psutil.Error):
        # macOS queries swap through sysctl, which may be denied in a sandbox.
        swap_used = 0
        swap_total = 0
    mps_current, mps_driver, mps_rec_max = _mps_memory()

    active_workers = 0
    worker_pids: list[int] = []
    accelerator_rss = 0
    worker_mps_current = 0
    worker_mps_driver = 0
    worker_mps_rec_max = 0
    has_mps_current = False
    has_mps_driver = False
    has_mps_rec_max = False
    if embedding_client is not None:
        workers = (getattr(embedding_client, "_accelerator_workers", {}) or {}).values()
        for worker in workers:
            if not worker.is_alive:
                continue
            active_workers += 1
            try:
                if worker.pid is not None:
                    worker_pids.append(worker.pid)
                    accelerator_rss += psutil.Process(worker.pid).memory_info().rss
            except (OSError, psutil.Error):
                pass

            # This property is populated by initialize/encode replies. Reading it
            # never takes the worker request lock, so the periodic probe remains
            # live while a long encode owns the synchronous protocol.
            health = getattr(worker, "last_backend_health", {})
            if not isinstance(health, dict):
                continue
            current = health.get("mps_current_allocated_bytes")
            driver = health.get("mps_driver_allocated_bytes")
            recommended = health.get("mps_recommended_max_bytes")
            if isinstance(current, int):
                worker_mps_current += current
                has_mps_current = True
            if isinstance(driver, int):
                worker_mps_driver += driver
                has_mps_driver = True
            if isinstance(recommended, int):
                worker_mps_rec_max = max(worker_mps_rec_max, recommended)
                has_mps_rec_max = True

    if has_mps_current:
        mps_current = worker_mps_current
    if has_mps_driver:
        mps_driver = worker_mps_driver
    if has_mps_rec_max:
        mps_rec_max = worker_mps_rec_max

    disk_path = tempfile.gettempdir()
    try:
        disk = shutil.disk_usage(disk_path)
        disk_free = disk.free
        disk_total = disk.total
    except OSError:
        disk_free = 0
        disk_total = 0

    return MemorySnapshot(
        rss_bytes=mi.rss,
        vsz_bytes=mi.vms,
        thread_count=threading.active_count(),
        gc_objects=len(gc.get_objects()),
        mps_current_bytes=mps_current,
        mps_driver_bytes=mps_driver,
        mps_recommended_max_bytes=mps_rec_max,
        system_available_bytes=vm.available,
        system_total_bytes=vm.total,
        active_accelerator_workers=active_workers,
        accelerator_worker_pids=tuple(worker_pids),
        accelerator_rss_bytes=accelerator_rss,
        swap_used_bytes=swap_used,
        swap_total_bytes=swap_total,
        disk_free_bytes=disk_free,
        disk_total_bytes=disk_total,
        disk_path=disk_path,
    )


def _fmt_gb(b: Optional[int]) -> str:
    if b is None:
        return "n/a"
    return f"{b / _BYTES_PER_GB:.2f}GB"


def _fmt_signed_mb(b: Optional[int]) -> str:
    if b is None:
        return "n/a"
    mb = b / (1024**2)
    return f"{mb:+.1f}MB"


def format_snapshot(snap: MemorySnapshot) -> str:
    """One-line absolute snapshot suitable for verbose log lines."""
    parts = [
        f"rss={_fmt_gb(snap.rss_bytes)}",
        f"worker_rss={_fmt_gb(snap.accelerator_rss_bytes)}",
        f"mps={_fmt_gb(snap.mps_current_bytes)}",
        f"drv={_fmt_gb(snap.mps_driver_bytes)}",
        f"threads={snap.thread_count}",
        f"gc_objs={snap.gc_objects}",
    ]
    if snap.active_accelerator_workers:
        parts.append(f"accelerator_workers={snap.active_accelerator_workers}")
        if snap.accelerator_worker_pids:
            parts.append(f"worker_pids={','.join(map(str, snap.accelerator_worker_pids))}")
    sys_pct = 0.0
    if snap.system_total_bytes:
        used = snap.system_total_bytes - snap.system_available_bytes
        sys_pct = 100.0 * used / snap.system_total_bytes
    parts.append(f"sys={sys_pct:.0f}%")
    if snap.swap_total_bytes:
        parts.append(f"swap={_fmt_gb(snap.swap_used_bytes)}")
    if snap.disk_total_bytes:
        parts.append(f"disk_free={_fmt_gb(snap.disk_free_bytes)}")
    return " ".join(parts)


def format_snapshot_delta(snap: MemorySnapshot, prev: MemorySnapshot) -> str:
    """One-line delta against a prior snapshot."""
    d = snap.delta(prev)
    parts = [
        f"Δrss={_fmt_signed_mb(d['rss'])}",
        f"Δworker_rss={_fmt_signed_mb(d['accelerator_rss'])}",
        f"Δswap={_fmt_signed_mb(d['swap_used'])}",
        f"Δmps={_fmt_signed_mb(d['mps_current'])}",
        f"Δdrv={_fmt_signed_mb(d['mps_driver'])}",
        f"Δsys={_fmt_signed_mb(d['sys_used'])}",
        f"Δdisk={_fmt_signed_mb(d['disk_free'])}",
        f"Δthreads={d['threads']:+d}",
        f"Δgc={d['gc']:+d}",
    ]
    return " ".join(parts)


def resource_warnings(snap: MemorySnapshot) -> dict[str, str]:
    """Return stable warning keys and actionable messages for resource pressure."""
    warnings: dict[str, str] = {}

    if snap.disk_total_bytes:
        critical_disk = max(_DISK_CRITICAL_FLOOR, int(snap.disk_total_bytes * 0.01))
        warning_disk = max(_DISK_WARNING_FLOOR, int(snap.disk_total_bytes * 0.05))
        if snap.disk_free_bytes <= critical_disk:
            warnings["disk_critical"] = (
                f"scratch disk critically low: {_fmt_gb(snap.disk_free_bytes)} free at "
                f"{snap.disk_path}; MPSGraph and index writes may fail"
            )
        elif snap.disk_free_bytes <= warning_disk:
            warnings["disk_low"] = (
                f"scratch disk low: {_fmt_gb(snap.disk_free_bytes)} free at "
                f"{snap.disk_path}; free space before continuing large accelerator runs"
            )

    if snap.system_total_bytes:
        warning_memory = max(_MEMORY_WARNING_FLOOR, int(snap.system_total_bytes * 0.10))
        if snap.system_available_bytes <= warning_memory:
            warnings["memory_low"] = (
                f"system memory low: {_fmt_gb(snap.system_available_bytes)} available; "
                "accelerator pressure may spill into swap"
            )

    if snap.mps_driver_bytes is not None and snap.mps_recommended_max_bytes:
        ratio = snap.mps_driver_bytes / snap.mps_recommended_max_bytes
        if ratio >= 0.85:
            warnings["mps_pressure"] = (
                f"MPS driver memory is {ratio:.0%} of the recommended maximum "
                f"({_fmt_gb(snap.mps_driver_bytes)} / "
                f"{_fmt_gb(snap.mps_recommended_max_bytes)})"
            )

    return warnings


# Module-level phase tracker. Single string assignment is atomic in CPython
# under the GIL — racy reads from the probe thread are fine for telemetry.
_current_phase: str = "startup"


def set_phase(name: str) -> None:
    """Update the current sync phase. Read by the periodic probe thread."""
    global _current_phase
    _current_phase = name


def get_phase() -> str:
    """Return the current sync phase."""
    return _current_phase


def format_snapshot_jsonl(snap: MemorySnapshot, phase: str = "") -> str:
    """Render a snapshot as a single JSONL line for machine consumption.

    Includes wallclock timestamp (UTC, ISO-8601 with Z suffix), phase string,
    and all snapshot fields. Designed for grep/jq/plot post-mortem analysis
    of memory growth across a sync run.
    """
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    sys_used_pct = 0.0
    if snap.system_total_bytes:
        used = snap.system_total_bytes - snap.system_available_bytes
        sys_used_pct = round(100.0 * used / snap.system_total_bytes, 2)
    obj = {
        "ts": ts,
        "phase": phase,
        "rss": snap.rss_bytes,
        "vsz": snap.vsz_bytes,
        "threads": snap.thread_count,
        "gc_objs": snap.gc_objects,
        "worker_rss": snap.accelerator_rss_bytes,
        "mps_current": snap.mps_current_bytes,
        "mps_driver": snap.mps_driver_bytes,
        "mps_recommended_max": snap.mps_recommended_max_bytes,
        "sys_used_pct": sys_used_pct,
        "sys_available": snap.system_available_bytes,
        "sys_total": snap.system_total_bytes,
        "active_accelerator_workers": snap.active_accelerator_workers,
        "accelerator_worker_pids": list(snap.accelerator_worker_pids),
        "swap_used": snap.swap_used_bytes,
        "swap_total": snap.swap_total_bytes,
        "disk_free": snap.disk_free_bytes,
        "disk_total": snap.disk_total_bytes,
        "disk_used_pct": (
            round(100.0 * (snap.disk_total_bytes - snap.disk_free_bytes) / snap.disk_total_bytes, 2)
            if snap.disk_total_bytes
            else 0.0
        ),
        "disk_path": snap.disk_path,
    }
    return json.dumps(obj, separators=(",", ":"))


def default_mem_probe_log_path() -> str:
    """Default JSONL output path: ~/.arcaneum/logs/arc-mem-<utc>-<pid>.jsonl.

    Mirrors the convention used by interaction_logger (~/.arcaneum/logs/) and
    namespaces by start time + pid so concurrent or repeated syncs don't
    overwrite each other.
    """
    from pathlib import Path

    log_dir = Path.home() / ".arcaneum" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(log_dir / f"arc-mem-{ts}-{os.getpid()}.jsonl")


def start_probe_thread(
    interval: float,
    log_path: Optional[str] = None,
    embedding_client=None,
) -> Callable[[], None]:
    """Start a daemon thread that periodically writes JSONL snapshots.

    Returns a stop() callable. The thread survives hostile encode loops that
    don't yield to Python — that's the whole point. JSONL is line-buffered so
    the file stays readable even if the process gets SIGKILLed.

    Args:
        interval: Seconds between snapshots. 0 disables the probe and returns
            a no-op stop callable.
        log_path: File path for JSONL output. None falls back to
            ~/.arcaneum/logs/arc-mem-<utc>-<pid>.jsonl. Pass "-" to write to
            stderr instead.
        embedding_client: Optional client to surface worker PIDs/RSS and cached
            child-reported accelerator allocator telemetry.

    Returns:
        stop() callable that signals the thread to exit. Idempotent.
    """
    if interval <= 0:

        def _noop_stop():
            return None

        return _noop_stop

    stop_event = threading.Event()
    active_warning_keys: set[str] = set()

    if log_path == "-":
        sink = sys.stderr
        owns_sink = False
    else:
        if not log_path:
            log_path = default_mem_probe_log_path()
        # Line-buffered so partial output survives a SIGKILL during a hang.
        sink = open(log_path, "a", buffering=1, encoding="utf-8")
        owns_sink = True
        try:
            sys.stderr.write(f"mem-probe: writing JSONL to {log_path}\n")
        except Exception:
            pass

    def _run():
        try:
            while not stop_event.is_set():
                try:
                    snap = snapshot(embedding_client)
                    line = format_snapshot_jsonl(snap, phase=get_phase())
                    sink.write(line + "\n")
                    if not owns_sink:
                        sink.flush()
                    warnings = resource_warnings(snap)
                    for key in warnings.keys() - active_warning_keys:
                        sys.stderr.write(f"resource-warning: {warnings[key]}\n")
                    active_warning_keys.clear()
                    active_warning_keys.update(warnings)
                except Exception as e:
                    # Telemetry must never crash the run.
                    try:
                        sys.stderr.write(f"mem-probe error: {e}\n")
                    except Exception:
                        pass
                stop_event.wait(interval)
        finally:
            if owns_sink:
                try:
                    sink.close()
                except Exception:
                    pass

    thread = threading.Thread(target=_run, daemon=True, name="arc-mem-probe")
    thread.start()

    def _stop():
        stop_event.set()

    return _stop


def install_dump_handler(embedding_client=None) -> None:
    """Install SIGUSR1 → dump handler. Use `kill -USR1 <pid>` during a hang.

    Also enables faulthandler's SIGABRT/segv traceback so a hard crash
    leaves a Python stack behind.

    Safe to call multiple times; re-registers the same handler.
    """
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:
        # faulthandler can fail if stderr is already redirected oddly
        pass

    def _handler(signum, frame):
        try:
            snap = snapshot(embedding_client)
            print("\n=== SIGUSR1 memory dump ===", file=sys.stderr, flush=True)
            print(f"  {format_snapshot(snap)}", file=sys.stderr, flush=True)
            print(
                f"  rss={snap.rss_bytes} vsz={snap.vsz_bytes} "
                f"worker_rss={snap.accelerator_rss_bytes} "
                f"worker_pids={snap.accelerator_worker_pids} "
                f"mps_current={snap.mps_current_bytes} "
                f"mps_driver={snap.mps_driver_bytes} "
                f"mps_rec_max={snap.mps_recommended_max_bytes} "
                f"sys_avail={snap.system_available_bytes} "
                f"sys_total={snap.system_total_bytes} "
                f"swap_used={snap.swap_used_bytes} "
                f"swap_total={snap.swap_total_bytes} "
                f"disk_free={snap.disk_free_bytes} "
                f"disk_total={snap.disk_total_bytes} "
                f"disk_path={snap.disk_path}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as e:
            print(f"  snapshot failed: {e}", file=sys.stderr, flush=True)

        print("=== thread stacks ===", file=sys.stderr, flush=True)
        try:
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        except Exception as e:
            print(f"  traceback dump failed: {e}", file=sys.stderr, flush=True)
        print("=== end dump ===", file=sys.stderr, flush=True)

    try:
        signal.signal(signal.SIGUSR1, _handler)
    except (ValueError, OSError):
        # Not in main thread, or SIGUSR1 unavailable on this platform.
        pass

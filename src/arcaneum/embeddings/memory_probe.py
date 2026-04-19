"""Lightweight memory/thread diagnostics for the embedding sync loop.

The corpus-sync process can grow monotonically on long unattended runs —
usually a combination of the MPS allocator not releasing buffers between
files, the GPU-poisoning daemon thread holding a closure ref to the GPU
model, and accumulated Python refs across many files. When it finally hits
the macOS memory pressure ceiling the process gets SIGKILLed by jetsam,
or (worse) the system pages itself into a kernel watchdog panic.

This module gives us observability without changing any behavior:

- `snapshot()` captures RSS/VSZ, MPS allocator state, thread count, and
  Python object count in ~0.2ms.
- `format_snapshot_delta()` renders a compact one-line delta suitable for
  per-file verbose logging.
- `install_dump_handler()` wires SIGUSR1 to dump a full snapshot plus
  thread stack traces to stderr. Use this during a suspected hang:
  `kill -USR1 <pid>`.

Nothing here blocks, raises on the happy path, or allocates significant
memory of its own — so it's safe to call in the hot loop.
"""

from __future__ import annotations

import faulthandler
import gc
import os
import signal
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional

import psutil


_BYTES_PER_GB = 1024 ** 3


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
    pending_gpu_cleanup: int = 0  # from EmbeddingClient, if passed in

    def delta(self, prev: "MemorySnapshot") -> dict:
        return {
            "rss": self.rss_bytes - prev.rss_bytes,
            "vsz": self.vsz_bytes - prev.vsz_bytes,
            "threads": self.thread_count - prev.thread_count,
            "gc": self.gc_objects - prev.gc_objects,
            "mps_current": (
                None if self.mps_current_bytes is None or prev.mps_current_bytes is None
                else self.mps_current_bytes - prev.mps_current_bytes
            ),
        }


def _mps_memory() -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (current, driver, recommended_max) bytes for MPS, or Nones.

    Each call is independent — if MPS isn't present we return (None, None, None)
    and silently skip. torch.mps accessors don't allocate.
    """
    try:
        import torch
        if not torch.backends.mps.is_available():
            return (None, None, None)
        current = torch.mps.current_allocated_memory()
        driver = torch.mps.driver_allocated_memory()
        try:
            rec_max = torch.mps.recommended_max_memory()
        except Exception:
            rec_max = None
        return (current, driver, rec_max)
    except Exception:
        return (None, None, None)


_PROC = psutil.Process(os.getpid())


def snapshot(embedding_client=None) -> MemorySnapshot:
    """Capture current memory + thread state. ~0.2ms on macOS."""
    mi = _PROC.memory_info()
    vm = psutil.virtual_memory()
    mps_current, mps_driver, mps_rec_max = _mps_memory()

    pending = 0
    if embedding_client is not None:
        pending = len(getattr(embedding_client, "_pending_gpu_cleanup", {}) or {})

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
        pending_gpu_cleanup=pending,
    )


def _fmt_gb(b: Optional[int]) -> str:
    if b is None:
        return "n/a"
    return f"{b / _BYTES_PER_GB:.2f}GB"


def _fmt_signed_mb(b: Optional[int]) -> str:
    if b is None:
        return "n/a"
    mb = b / (1024 ** 2)
    return f"{mb:+.1f}MB"


def format_snapshot(snap: MemorySnapshot) -> str:
    """One-line absolute snapshot suitable for verbose log lines."""
    parts = [
        f"rss={_fmt_gb(snap.rss_bytes)}",
        f"mps={_fmt_gb(snap.mps_current_bytes)}",
        f"drv={_fmt_gb(snap.mps_driver_bytes)}",
        f"threads={snap.thread_count}",
        f"gc_objs={snap.gc_objects}",
    ]
    if snap.pending_gpu_cleanup:
        parts.append(f"pending_cleanup={snap.pending_gpu_cleanup}")
    sys_pct = 0.0
    if snap.system_total_bytes:
        used = snap.system_total_bytes - snap.system_available_bytes
        sys_pct = 100.0 * used / snap.system_total_bytes
    parts.append(f"sys={sys_pct:.0f}%")
    return " ".join(parts)


def format_snapshot_delta(snap: MemorySnapshot, prev: MemorySnapshot) -> str:
    """One-line delta against a prior snapshot."""
    d = snap.delta(prev)
    parts = [
        f"Δrss={_fmt_signed_mb(d['rss'])}",
        f"Δmps={_fmt_signed_mb(d['mps_current'])}",
        f"Δthreads={d['threads']:+d}",
        f"Δgc={d['gc']:+d}",
    ]
    return " ".join(parts)


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
                f"mps_current={snap.mps_current_bytes} "
                f"mps_driver={snap.mps_driver_bytes} "
                f"mps_rec_max={snap.mps_recommended_max_bytes} "
                f"sys_avail={snap.system_available_bytes} "
                f"sys_total={snap.system_total_bytes}",
                file=sys.stderr, flush=True,
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

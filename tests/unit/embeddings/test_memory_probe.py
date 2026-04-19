"""Unit tests for the memory probe diagnostics."""

import signal
import sys

from arcaneum.embeddings.memory_probe import (
    MemorySnapshot,
    format_snapshot,
    format_snapshot_delta,
    install_dump_handler,
    snapshot,
)


def test_snapshot_has_core_fields():
    snap = snapshot()
    assert snap.rss_bytes > 0
    assert snap.vsz_bytes > 0
    assert snap.thread_count >= 1
    assert snap.gc_objects > 0
    assert snap.system_total_bytes > 0


def test_format_snapshot_is_one_line():
    s = format_snapshot(snapshot())
    assert "\n" not in s
    assert "rss=" in s and "threads=" in s


def test_delta_signed_mb_and_handles_none():
    a = MemorySnapshot(
        rss_bytes=100 * 1024 * 1024,
        vsz_bytes=0,
        thread_count=1,
        gc_objects=100,
        mps_current_bytes=None,
    )
    b = MemorySnapshot(
        rss_bytes=400 * 1024 * 1024,
        vsz_bytes=0,
        thread_count=3,
        gc_objects=150,
        mps_current_bytes=None,
    )
    delta_str = format_snapshot_delta(b, a)
    assert "Δrss=+300.0MB" in delta_str
    assert "Δthreads=+2" in delta_str
    assert "Δgc=+50" in delta_str
    # MPS delta should render as n/a when either side is None
    assert "Δmps=n/a" in delta_str


def test_install_dump_handler_registers_sigusr1():
    # Save existing handler to restore
    prev = signal.getsignal(signal.SIGUSR1)
    try:
        install_dump_handler()
        current = signal.getsignal(signal.SIGUSR1)
        assert callable(current)
        assert current is not prev or prev is None
    finally:
        if prev is not None:
            try:
                signal.signal(signal.SIGUSR1, prev)
            except Exception:
                pass


def test_install_dump_handler_tolerates_missing_client():
    # Should not raise even if embedding_client is None
    install_dump_handler(embedding_client=None)


def test_snapshot_accepts_embedding_client_with_pending_cleanup():
    class Stub:
        _pending_gpu_cleanup = {"jina-code": (object(), object())}
    snap = snapshot(embedding_client=Stub())
    assert snap.pending_gpu_cleanup == 1

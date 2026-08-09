"""Unit tests for the memory probe diagnostics."""

import json
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import arcaneum.embeddings.memory_probe as memory_probe
from arcaneum.embeddings.memory_probe import (
    MemorySnapshot,
    format_snapshot,
    format_snapshot_delta,
    format_snapshot_jsonl,
    get_phase,
    install_dump_handler,
    resource_warnings,
    set_phase,
    snapshot,
    start_probe_thread,
)


def test_snapshot_has_core_fields():
    snap = snapshot()
    assert snap.rss_bytes > 0
    assert snap.vsz_bytes > 0
    assert snap.thread_count >= 1
    assert snap.gc_objects > 0
    assert snap.system_total_bytes > 0
    assert snap.swap_total_bytes >= snap.swap_used_bytes >= 0
    assert snap.disk_total_bytes > 0
    assert snap.disk_free_bytes > 0
    assert snap.disk_path


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
    # MPS / driver deltas should render as n/a when either side is None
    assert "Δmps=n/a" in delta_str
    assert "Δdrv=n/a" in delta_str


def test_delta_includes_driver_and_sys_used():
    # Driver bytes grew 200MB, system used grew 1500MB — these are the
    # signals the per-file probe must surface so unified-memory leaks
    # don't stay invisible on Apple Silicon.
    a = MemorySnapshot(
        rss_bytes=0,
        vsz_bytes=0,
        thread_count=1,
        gc_objects=0,
        mps_current_bytes=10 * 1024 * 1024,
        mps_driver_bytes=1024 * 1024 * 1024,
        system_total_bytes=64 * 1024 * 1024 * 1024,
        system_available_bytes=32 * 1024 * 1024 * 1024,
    )
    b = MemorySnapshot(
        rss_bytes=0,
        vsz_bytes=0,
        thread_count=1,
        gc_objects=0,
        mps_current_bytes=10 * 1024 * 1024,
        mps_driver_bytes=(1024 + 200) * 1024 * 1024,
        system_total_bytes=64 * 1024 * 1024 * 1024,
        system_available_bytes=32 * 1024**3 - 1536 * 1024**2,
    )
    d = b.delta(a)
    assert d["mps_driver"] == 200 * 1024 * 1024
    assert d["sys_used"] == 1536 * 1024 * 1024
    assert isinstance(d["sys_used"], int)
    delta_str = format_snapshot_delta(b, a)
    assert "Δdrv=+200.0MB" in delta_str
    assert "Δsys=+1536.0MB" in delta_str


def test_install_dump_handler_registers_sigusr1():
    # Save existing handler to restore
    prev = signal.getsignal(signal.SIGUSR1)
    try:
        install_dump_handler()
        current = signal.getsignal(signal.SIGUSR1)
        assert callable(current)
        assert current is not prev or prev is None
    finally:
        try:
            signal.signal(signal.SIGUSR1, prev)
        except Exception:
            pass


def test_snapshot_accepts_embedding_client_with_pending_cleanup():
    class Stub:
        _accelerator_workers = {
            "jina-code": SimpleNamespace(
                is_alive=True,
                pid=os.getpid(),
                last_backend_health={
                    "mps_current_allocated_bytes": 100,
                    "mps_driver_allocated_bytes": 200,
                    "mps_recommended_max_bytes": 1000,
                },
            )
        }

    snap = snapshot(embedding_client=Stub())
    assert snap.active_accelerator_workers == 1
    assert snap.accelerator_worker_pids == (os.getpid(),)
    assert snap.accelerator_rss_bytes > 0
    assert snap.mps_current_bytes == 100
    assert snap.mps_driver_bytes == 200
    assert snap.mps_recommended_max_bytes == 1000


def test_resource_warnings_cover_disk_memory_and_mps_pressure():
    snap = MemorySnapshot(
        rss_bytes=1,
        vsz_bytes=1,
        thread_count=1,
        gc_objects=1,
        mps_driver_bytes=90,
        mps_recommended_max_bytes=100,
        system_available_bytes=1024**3,
        system_total_bytes=32 * 1024**3,
        disk_free_bytes=512 * 1024**2,
        disk_total_bytes=500 * 1024**3,
        disk_path="/tmp",
    )

    warnings = resource_warnings(snap)

    assert set(warnings) == {"disk_critical", "memory_low", "mps_pressure"}
    assert "MPSGraph" in warnings["disk_critical"]


def test_set_phase_and_get_phase_roundtrip():
    set_phase("startup")
    assert get_phase() == "startup"
    set_phase("encoding:foo.pdf")
    assert get_phase() == "encoding:foo.pdf"


def test_format_snapshot_jsonl_is_parseable_with_expected_keys():
    set_phase("encoding:test.pdf")
    snap = snapshot()
    line = format_snapshot_jsonl(snap, phase=get_phase())
    # Must be exactly one line, parseable as JSON
    assert "\n" not in line
    obj = json.loads(line)
    # Required keys for downstream analysis
    for key in (
        "ts",
        "phase",
        "rss",
        "vsz",
        "threads",
        "gc_objs",
        "worker_rss",
        "mps_current",
        "mps_driver",
        "sys_used_pct",
        "sys_available",
        "sys_total",
        "active_accelerator_workers",
        "accelerator_worker_pids",
        "swap_used",
        "swap_total",
        "disk_free",
        "disk_total",
        "disk_used_pct",
        "disk_path",
    ):
        assert key in obj, f"missing {key} in {obj}"
    assert obj["phase"] == "encoding:test.pdf"
    assert obj["rss"] > 0
    assert 0.0 <= obj["sys_used_pct"] <= 100.0
    # Timestamp must be ISO-8601 with timezone offset (Z or +HH:MM)
    assert obj["ts"].endswith("Z") or "+" in obj["ts"][10:]


def test_format_snapshot_jsonl_handles_none_mps_fields():
    snap = MemorySnapshot(
        rss_bytes=1024,
        vsz_bytes=2048,
        thread_count=1,
        gc_objects=10,
        mps_current_bytes=None,
        mps_driver_bytes=None,
        system_total_bytes=64 * 1024**3,
        system_available_bytes=32 * 1024**3,
    )
    obj = json.loads(format_snapshot_jsonl(snap, phase="idle"))
    assert obj["mps_current"] is None
    assert obj["mps_driver"] is None


def test_start_probe_thread_writes_periodically_then_stops(tmp_path):
    log_path = tmp_path / "mem.jsonl"
    set_phase("test-phase")
    stop = start_probe_thread(interval=0.1, log_path=str(log_path))
    try:
        time.sleep(0.45)  # Expect ~4 ticks
    finally:
        stop()
    # Give the thread a moment to drain after stop()
    time.sleep(0.2)
    lines = [line for line in log_path.read_text().splitlines() if line.strip()]
    assert len(lines) >= 2, f"expected ≥2 ticks, got {len(lines)}: {lines}"
    # Each line is JSON with the current phase
    for line in lines:
        obj = json.loads(line)
        assert obj["phase"] == "test-phase"


def test_start_probe_thread_zero_interval_returns_noop_stop(tmp_path):
    # Interval=0 means "off" — no thread, no file writes, stop is callable
    log_path = tmp_path / "mem.jsonl"
    stop = start_probe_thread(interval=0, log_path=str(log_path))
    time.sleep(0.2)
    stop()  # Must not raise
    assert not log_path.exists() or log_path.read_text() == ""


def test_start_probe_thread_writes_to_stderr_with_dash_sentinel(capsys):
    set_phase("stderr-test")
    stop = start_probe_thread(interval=0.1, log_path="-")
    try:
        time.sleep(0.25)
    finally:
        stop()
    time.sleep(0.15)
    captured = capsys.readouterr()
    # At least one JSONL line on stderr
    stderr_lines = [line for line in captured.err.splitlines() if line.strip().startswith("{")]
    assert len(stderr_lines) >= 1
    obj = json.loads(stderr_lines[0])
    assert obj["phase"] == "stderr-test"


def test_start_probe_thread_emits_each_active_resource_warning_once(monkeypatch, tmp_path, capsys):
    pressured = MemorySnapshot(
        rss_bytes=1,
        vsz_bytes=1,
        thread_count=1,
        gc_objects=1,
        disk_free_bytes=512 * 1024**2,
        disk_total_bytes=500 * 1024**3,
        disk_path="/tmp",
    )
    monkeypatch.setattr(memory_probe, "snapshot", lambda embedding_client=None: pressured)

    stop = start_probe_thread(interval=0.05, log_path=str(tmp_path / "mem.jsonl"))
    try:
        time.sleep(0.18)
    finally:
        stop()
    time.sleep(0.1)

    warning_lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("resource-warning:")
    ]
    assert len(warning_lines) == 1
    assert "scratch disk critically low" in warning_lines[0]


def test_start_probe_thread_default_path_under_arcaneum_logs(monkeypatch, tmp_path):
    # Redirect HOME so the default path lands in tmp_path/.arcaneum/logs/
    monkeypatch.setenv("HOME", str(tmp_path))
    # On macOS Path.home() honors HOME. Belt-and-suspenders: also patch
    # pathlib.Path.home in case some envs prefer pwd lookup.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    set_phase("default-path-test")
    stop = start_probe_thread(interval=0.1, log_path=None)
    try:
        time.sleep(0.25)
    finally:
        stop()
    time.sleep(0.15)

    log_dir = tmp_path / ".arcaneum" / "logs"
    assert log_dir.is_dir()
    files = list(log_dir.glob("arc-mem-*.jsonl"))
    assert files, f"expected an arc-mem-*.jsonl in {log_dir}"
    contents = files[0].read_text()
    lines = [line for line in contents.splitlines() if line.strip()]
    assert len(lines) >= 1
    obj = json.loads(lines[0])
    assert obj["phase"] == "default-path-test"

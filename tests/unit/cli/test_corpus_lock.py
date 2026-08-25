"""Tests for the per-corpus write lock (kata htmw).

`arc corpus sync` serializes writes to a corpus so two concurrent runs cannot
interleave their "what is indexed" reads with their Qdrant/MeiliSearch writes.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from arcaneum.cli import corpus_lock
from arcaneum.cli.errors import CorpusLockUnavailable


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="flock-based corpus lock is a no-op on Windows",
)


@pytest.fixture
def isolated_locks(tmp_path, monkeypatch):
    """Redirect ~/.local/share/arcaneum to a tmp dir so each test gets fresh locks."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    yield tmp_path / "arcaneum" / "locks"


def test_acquire_and_release_roundtrip(isolated_locks):
    with corpus_lock.acquire_corpus_lock("MyCorpus"):
        pass
    # A second acquire proves the first was released.
    with corpus_lock.acquire_corpus_lock("MyCorpus"):
        pass


def test_lock_file_records_pid(isolated_locks):
    with corpus_lock.acquire_corpus_lock("MyCorpus"):
        lock_path = corpus_lock.corpus_lock_path("MyCorpus")
        holder = corpus_lock.read_lock_holder(lock_path)
        assert holder is not None
        assert holder["pid"] == os.getpid()
        assert holder["corpus"] == "MyCorpus"
        assert holder["started"] > 0


def test_released_on_exception(isolated_locks):
    with pytest.raises(RuntimeError):
        with corpus_lock.acquire_corpus_lock("MyCorpus"):
            raise RuntimeError("boom")

    # If the lock leaked, --no-wait below would fail.
    with corpus_lock.acquire_corpus_lock("MyCorpus", wait=False):
        pass


def test_released_on_keyboard_interrupt(isolated_locks):
    with pytest.raises(KeyboardInterrupt):
        with corpus_lock.acquire_corpus_lock("MyCorpus"):
            raise KeyboardInterrupt()

    with corpus_lock.acquire_corpus_lock("MyCorpus", wait=False):
        pass


def test_reentrant_acquire_in_same_process(isolated_locks):
    """Nested acquires of the same corpus in one process must not self-deadlock."""
    with corpus_lock.acquire_corpus_lock("MyCorpus"):
        with corpus_lock.acquire_corpus_lock("MyCorpus", wait=False):
            pass
        # Still held by the outer scope.
        assert corpus_lock.read_lock_holder(corpus_lock.corpus_lock_path("MyCorpus"))


def test_different_corpora_do_not_block(isolated_locks):
    with corpus_lock.acquire_corpus_lock("CorpusA"):
        with corpus_lock.acquire_corpus_lock("CorpusB", wait=False):
            pass


def test_lock_key_isolates_service_endpoints(isolated_locks, monkeypatch):
    """Same corpus name against different Qdrant/Meili targets uses different locks."""
    monkeypatch.setenv("ARC_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("MEILISEARCH_URL", "http://localhost:7700")
    path_a = corpus_lock.corpus_lock_path("MyCorpus")

    monkeypatch.setenv("ARC_QDRANT_URL", "http://otherhost:6333")
    path_b = corpus_lock.corpus_lock_path("MyCorpus")

    assert path_a != path_b


def test_lock_key_stable_for_same_endpoints(isolated_locks, monkeypatch):
    monkeypatch.setenv("ARC_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("MEILISEARCH_URL", "http://localhost:7700")
    assert corpus_lock.corpus_lock_path("MyCorpus") == corpus_lock.corpus_lock_path("MyCorpus")


def test_corpus_name_is_visible_in_lock_filename(isolated_locks):
    assert "MyCorpus" in corpus_lock.corpus_lock_path("MyCorpus").name


def test_unsafe_corpus_name_is_sanitized(isolated_locks):
    """A corpus name with path separators must not escape the locks dir."""
    path = corpus_lock.corpus_lock_path("../../etc/passwd")
    assert path.parent == corpus_lock._locks_dir()


# --- Cross-process contention -------------------------------------------------

_HOLDER_SCRIPT = textwrap.dedent(
    """
    import sys, time
    from arcaneum.cli import corpus_lock

    with corpus_lock.acquire_corpus_lock(sys.argv[1]):
        print("held", flush=True)
        time.sleep(float(sys.argv[2]))
    """
)


def _spawn_holder(corpus: str, hold_seconds: float, env: dict) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SCRIPT, corpus, str(hold_seconds)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    # Wait until the child confirms it holds the lock.
    assert proc.stdout is not None
    line = proc.stdout.readline()
    assert line.strip() == "held", f"holder failed to start: {proc.stderr.read()}"
    return proc


@pytest.fixture
def holder_env(tmp_path, monkeypatch):
    env = dict(os.environ)
    env["XDG_DATA_HOME"] = str(tmp_path)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    return env


def test_no_wait_fails_fast_naming_the_holder(holder_env):
    proc = _spawn_holder("MyCorpus", 5.0, holder_env)
    try:
        t0 = time.monotonic()
        with pytest.raises(CorpusLockUnavailable) as exc_info:
            with corpus_lock.acquire_corpus_lock("MyCorpus", wait=False):
                pytest.fail("should not have acquired a held lock")
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, "--no-wait must not block"
        message = str(exc_info.value)
        assert "MyCorpus" in message
        assert str(proc.pid) in message
    finally:
        proc.kill()
        proc.wait()


def test_wait_times_out_with_holder_pid(holder_env):
    proc = _spawn_holder("MyCorpus", 5.0, holder_env)
    try:
        t0 = time.monotonic()
        with pytest.raises(CorpusLockUnavailable) as exc_info:
            with corpus_lock.acquire_corpus_lock("MyCorpus", timeout=0.5):
                pytest.fail("should not have acquired a held lock")
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.4, "should have waited for the timeout"
        assert str(proc.pid) in str(exc_info.value)
    finally:
        proc.kill()
        proc.wait()


def test_second_waiter_acquires_after_holder_exits(holder_env):
    proc = _spawn_holder("MyCorpus", 0.5, holder_env)
    try:
        with corpus_lock.acquire_corpus_lock("MyCorpus", timeout=10.0):
            pass
    finally:
        proc.kill()
        proc.wait()


def test_lock_released_when_holder_is_killed(holder_env):
    """flock is released by the OS on process death — no stale-lock cleanup."""
    proc = _spawn_holder("MyCorpus", 30.0, holder_env)
    proc.kill()
    proc.wait()

    with corpus_lock.acquire_corpus_lock("MyCorpus", wait=False):
        pass

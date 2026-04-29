"""Tests for the embedder-slot file-lock semaphore (RDR-019 follow-on)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from arcaneum.cli import concurrency
from arcaneum.cli.errors import SearchSlotUnavailable


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="flock-based semaphore is a no-op on Windows",
)


@pytest.fixture
def isolated_locks(tmp_path, monkeypatch):
    """Redirect ~/.local/share/arcaneum to a tmp dir so each test gets fresh locks."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    yield tmp_path / "arcaneum" / "locks"


def test_acquire_when_free(isolated_locks, monkeypatch):
    monkeypatch.setenv("ARCANEUM_SEARCH_CONCURRENCY", "2")
    monkeypatch.setenv("ARCANEUM_SEARCH_WAIT_SECONDS", "1")
    with concurrency.acquire_embedder_slot():
        pass


def test_two_holders_at_cap_2(isolated_locks, monkeypatch):
    """Two simultaneous acquirers fit when cap is 2."""
    monkeypatch.setenv("ARCANEUM_SEARCH_CONCURRENCY", "2")
    monkeypatch.setenv("ARCANEUM_SEARCH_WAIT_SECONDS", "1")
    with concurrency.acquire_embedder_slot():
        with concurrency.acquire_embedder_slot():
            pass


def test_third_acquirer_times_out_when_cap_2(isolated_locks, monkeypatch):
    monkeypatch.setenv("ARCANEUM_SEARCH_CONCURRENCY", "2")
    monkeypatch.setenv("ARCANEUM_SEARCH_WAIT_SECONDS", "0.5")
    with concurrency.acquire_embedder_slot():
        with concurrency.acquire_embedder_slot():
            t0 = time.monotonic()
            with pytest.raises(SearchSlotUnavailable) as exc_info:
                with concurrency.acquire_embedder_slot():
                    pytest.fail("should not have acquired")
            elapsed = time.monotonic() - t0
            assert "ARCANEUM_SEARCH_CONCURRENCY" in str(exc_info.value)
            # Should have waited ~the timeout, not returned instantly.
            assert elapsed >= 0.4


def test_slot_released_on_exception(isolated_locks, monkeypatch):
    """An exception inside the `with` must still release the slot."""
    monkeypatch.setenv("ARCANEUM_SEARCH_CONCURRENCY", "1")
    monkeypatch.setenv("ARCANEUM_SEARCH_WAIT_SECONDS", "0.5")

    with pytest.raises(RuntimeError):
        with concurrency.acquire_embedder_slot():
            raise RuntimeError("boom")

    # If the slot leaked, this acquire would time out.
    with concurrency.acquire_embedder_slot():
        pass


def test_cross_process_blocking(isolated_locks, monkeypatch, tmp_path):
    """A holder in another process actually blocks acquirers in this process."""
    monkeypatch.setenv("ARCANEUM_SEARCH_CONCURRENCY", "1")
    monkeypatch.setenv("ARCANEUM_SEARCH_WAIT_SECONDS", "0.4")

    ready_marker = tmp_path / "ready"
    release_marker = tmp_path / "release"

    holder_script = textwrap.dedent(
        f"""
        import os, sys, time
        os.environ["XDG_DATA_HOME"] = {str(isolated_locks.parent.parent)!r}
        os.environ["ARCANEUM_SEARCH_CONCURRENCY"] = "1"
        os.environ["ARCANEUM_SEARCH_WAIT_SECONDS"] = "5"
        sys.path.insert(0, {str(_repo_src())!r})
        from arcaneum.cli import concurrency
        with concurrency.acquire_embedder_slot():
            open({str(ready_marker)!r}, "w").close()
            # Hold until parent signals release, with a hard cap so the test
            # cannot hang forever if the parent fails.
            deadline = time.monotonic() + 10
            while not os.path.exists({str(release_marker)!r}):
                if time.monotonic() > deadline:
                    break
                time.sleep(0.05)
        """
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", holder_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait for holder to actually hold the slot.
        for _ in range(100):
            if ready_marker.exists():
                break
            time.sleep(0.05)
        else:
            stdout, stderr = proc.communicate(timeout=2)
            pytest.fail(
                f"holder process never reported ready. "
                f"stdout={stdout!r} stderr={stderr!r}"
            )

        with pytest.raises(SearchSlotUnavailable):
            with concurrency.acquire_embedder_slot():
                pytest.fail("should have been blocked by other process")
    finally:
        release_marker.touch()
        proc.wait(timeout=5)


def test_invalid_env_falls_back_to_default(isolated_locks, monkeypatch):
    monkeypatch.setenv("ARCANEUM_SEARCH_CONCURRENCY", "not-a-number")
    assert concurrency.get_search_concurrency() == concurrency.DEFAULT_SEARCH_CONCURRENCY


def test_minimum_concurrency_is_one(isolated_locks, monkeypatch):
    monkeypatch.setenv("ARCANEUM_SEARCH_CONCURRENCY", "0")
    assert concurrency.get_search_concurrency() == 1


def _repo_src() -> Path:
    """Return the absolute path of the project src/ dir for subprocess sys.path."""
    return Path(__file__).resolve().parents[3] / "src"

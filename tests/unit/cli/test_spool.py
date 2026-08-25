"""Tests for the maildir-style sync spool (kata vq0n).

The git hook must never block or fail a git operation, so it writes the paths a
commit touched into a spool directory and hands off to a background worker that
coalesces bursts of commits into one model load.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from arcaneum.cli import spool


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="flock-based spool worker is not supported on Windows",
)


@pytest.fixture
def isolated_spool(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    yield tmp_path / "arcaneum" / "spool"


# --- writing entries ----------------------------------------------------------


def test_write_entry_creates_a_readable_record(isolated_spool):
    path = spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    assert path.exists()
    record = json.loads(path.read_text())
    assert record["corpus"] == "Docs"
    assert record["changed"] == ["/repo/a.py"]
    assert record["removed"] == []


def test_write_entry_is_scoped_per_corpus_and_repo(isolated_spool):
    a = spool.write_entry("Docs", "/repo-one", changed=["/repo-one/a.py"], removed=[])
    b = spool.write_entry("Docs", "/repo-two", changed=["/repo-two/a.py"], removed=[])
    c = spool.write_entry("Other", "/repo-one", changed=["/repo-one/a.py"], removed=[])
    assert a.parent != b.parent
    assert a.parent != c.parent


def test_entries_land_atomically_with_no_partial_files(isolated_spool):
    """A reader must never see a half-written entry, so write tmp then rename."""
    path = spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    siblings = list(path.parent.iterdir())
    assert siblings == [path], "no temp file should be left behind"
    assert path.suffix == ".json"


def test_write_entry_with_nothing_to_do_writes_nothing(isolated_spool):
    assert spool.write_entry("Docs", "/repo", changed=[], removed=[]) is None


def test_multiple_entries_accumulate(isolated_spool):
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    spool.write_entry("Docs", "/repo", changed=["/repo/b.py"], removed=[])
    assert len(list(spool.corpus_spool_dir("Docs").rglob("*.json"))) == 2


# --- draining -----------------------------------------------------------------


def test_drain_batch_unions_entries_and_removes_them(isolated_spool):
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    spool.write_entry("Docs", "/repo", changed=["/repo/b.py"], removed=["/repo/c.py"])

    batch = spool.drain_batch("Docs")

    assert sorted(batch.changed) == ["/repo/a.py", "/repo/b.py"]
    assert batch.removed == ["/repo/c.py"]
    assert list(spool.corpus_spool_dir("Docs").rglob("*.json")) == []


def test_drain_batch_dedupes_a_path_touched_by_several_commits(isolated_spool):
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])

    batch = spool.drain_batch("Docs")
    assert batch.changed == ["/repo/a.py"]


def test_a_later_delete_wins_over_an_earlier_change(isolated_spool):
    """Commit adds a file, next commit deletes it: the file must not be indexed."""
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    spool.write_entry("Docs", "/repo", changed=[], removed=["/repo/a.py"])

    batch = spool.drain_batch("Docs")
    assert batch.changed == []
    assert batch.removed == ["/repo/a.py"]


def test_a_later_change_wins_over_an_earlier_delete(isolated_spool):
    """Deleted then restored: the file must be re-indexed, not dropped."""
    spool.write_entry("Docs", "/repo", changed=[], removed=["/repo/a.py"])
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])

    batch = spool.drain_batch("Docs")
    assert batch.changed == ["/repo/a.py"]
    assert batch.removed == []


def test_drain_batch_on_an_empty_spool_is_empty(isolated_spool):
    batch = spool.drain_batch("Docs")
    assert not batch


def test_a_corrupt_entry_is_discarded_not_fatal(isolated_spool):
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    bad = spool.corpus_spool_dir("Docs") / "bad.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not json")

    batch = spool.drain_batch("Docs")

    assert batch.changed == ["/repo/a.py"]
    assert not bad.exists(), "an unparseable entry must be dropped, not retried forever"


def test_entries_written_during_a_drain_survive_for_the_next_pass(isolated_spool):
    """The worker loops until empty; a commit landing mid-drain must not be lost."""
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    first = spool.drain_batch("Docs")
    spool.write_entry("Docs", "/repo", changed=["/repo/b.py"], removed=[])
    second = spool.drain_batch("Docs")

    assert first.changed == ["/repo/a.py"]
    assert second.changed == ["/repo/b.py"]


def test_pending_reports_whether_work_remains(isolated_spool):
    assert not spool.has_pending("Docs")
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    assert spool.has_pending("Docs")
    spool.drain_batch("Docs")
    assert not spool.has_pending("Docs")


# --- worker single-flight -----------------------------------------------------


_HOLDER = textwrap.dedent(
    """
    import sys, time
    from arcaneum.cli import spool

    lock = spool.try_acquire_worker_lock(sys.argv[1])
    if lock is None:
        print("nolock", flush=True)
        sys.exit(1)
    print("held", flush=True)
    time.sleep(float(sys.argv[2]))
    """
)


def test_worker_lock_is_single_flight(isolated_spool, tmp_path):
    env = dict(os.environ)
    env["XDG_DATA_HOME"] = str(tmp_path)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")

    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, "Docs", "5"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )
    try:
        assert proc.stdout.readline().strip() == "held"
        # A second worker for the same corpus must decline rather than race.
        assert spool.try_acquire_worker_lock("Docs") is None
    finally:
        proc.kill()
        proc.wait()

    # Released when the holder dies — the next commit can spawn a fresh worker.
    lock = spool.try_acquire_worker_lock("Docs")
    assert lock is not None
    spool.release_worker_lock(lock)


def test_worker_lock_is_per_corpus(isolated_spool):
    a = spool.try_acquire_worker_lock("Docs")
    b = spool.try_acquire_worker_lock("Other")
    assert a is not None and b is not None
    spool.release_worker_lock(a)
    spool.release_worker_lock(b)


def test_unsafe_corpus_name_cannot_escape_the_spool_root(isolated_spool):
    target = spool.corpus_spool_dir("../../etc")
    assert spool.spool_root() in target.parents

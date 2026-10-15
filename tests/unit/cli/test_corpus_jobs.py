"""Tests for `arc corpus jobs`: what is indexing right now, and what is queued.

The listing reads the holder records that the corpus write lock and the drain
worker lock already stamp, and checks each pid with psutil. It must never take
a lock itself: the drain worker's lock is single-try, so a probe that held it
for an instant could make a hook-spawned worker decline and strand its work.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

import pytest
from click.testing import CliRunner

from arcaneum.cli import corpus_lock, jobs, spool
from arcaneum.cli.main import cli

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="flock-based locks are not supported on Windows",
)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    yield tmp_path / "arcaneum"


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _write_corpus_record(corpus: str, **record) -> None:
    path = corpus_lock.corpus_lock_path(corpus)
    path.write_text(json.dumps({"corpus": corpus, **record}))


def _write_worker_record(corpus: str, **record) -> None:
    spool.worker_lock_path(corpus).write_text(json.dumps(record))


# --- running jobs ---------------------------------------------------------------


def test_held_corpus_lock_is_listed_as_a_write_job(data_dir):
    with corpus_lock.acquire_corpus_lock("Docs", quiet=True):
        listed = jobs.running_jobs()

    assert len(listed) == 1
    job = listed[0]
    assert job.corpus == "Docs"
    assert job.kind == "write"
    assert job.pid == os.getpid()
    assert job.command
    assert job.endpoints


def test_released_corpus_lock_is_not_listed(data_dir):
    """The holder is still alive after releasing, so the record must be cleared."""
    with corpus_lock.acquire_corpus_lock("Docs", quiet=True):
        pass
    assert jobs.running_jobs() == []


def test_held_worker_lock_is_listed_as_a_drain_job(data_dir):
    fd = spool.try_acquire_worker_lock("Docs")
    try:
        listed = jobs.running_jobs()
    finally:
        spool.release_worker_lock(fd)

    assert [(j.corpus, j.kind, j.pid) for j in listed] == [("Docs", "drain", os.getpid())]
    assert jobs.running_jobs() == [], "a released worker lock is not listed"


def test_drain_holding_both_locks_is_listed_once(data_dir):
    fd = spool.try_acquire_worker_lock("Docs")
    try:
        with corpus_lock.acquire_corpus_lock("Docs", quiet=True):
            listed = jobs.running_jobs()
    finally:
        spool.release_worker_lock(fd)

    assert [(j.corpus, j.kind) for j in listed] == [("Docs", "drain")]


def test_record_naming_a_dead_pid_is_hidden(data_dir):
    _write_corpus_record("Docs", pid=_dead_pid(), started=time.time())
    _write_worker_record("Docs", pid=_dead_pid(), started=time.time())
    assert jobs.running_jobs() == []


def test_record_naming_a_reused_pid_is_hidden(data_dir):
    """A live pid that started after the record was written is a different process."""
    long_ago = time.time() - 10 * 365 * 24 * 3600
    _write_corpus_record("Docs", pid=os.getpid(), started=long_ago)
    assert jobs.running_jobs() == []


def test_unreadable_records_are_skipped(data_dir):
    corpus_lock.corpus_lock_path("Empty").write_text("")
    corpus_lock.corpus_lock_path("Garbage").write_text("not json")
    _write_corpus_record("NoPid", started=time.time())
    assert jobs.running_jobs() == []


def test_record_without_command_falls_back_to_the_process_cmdline(data_dir):
    _write_corpus_record("Docs", pid=os.getpid(), started=time.time())
    [job] = jobs.running_jobs()
    assert job.command


def test_embedder_slot_locks_are_not_jobs(data_dir):
    locks = data_dir / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    (locks / "embedder-slot-0.lock").write_text(
        json.dumps({"pid": os.getpid(), "started": time.time()})
    )
    assert jobs.running_jobs() == []


# --- queued work ------------------------------------------------------------------


def _claim_one(corpus: str) -> None:
    entry = spool.write_entry(corpus, "/repo", changed=["/repo/a.py"], removed=[])
    claimed = spool.claimed_dir(corpus)
    claimed.mkdir(parents=True, exist_ok=True)
    entry.rename(claimed / entry.name)


def test_queued_work_counts_entries_per_corpus(data_dir):
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    spool.write_entry("Docs", "/repo", changed=["/repo/b.py"], removed=[])
    spool.write_entry("Other", "/repo", changed=["/repo/c.py"], removed=[])

    queued = {q.corpus: q for q in jobs.queued_work(jobs.running_jobs())}
    assert set(queued) == {"Docs", "Other"}
    assert (queued["Docs"].queued, queued["Docs"].claimed, queued["Docs"].deferred) == (2, 0, 0)
    assert queued["Other"].queued == 1


def test_claimed_entries_without_a_live_drain_are_orphaned(data_dir):
    _claim_one("Docs")
    [work] = jobs.queued_work(jobs.running_jobs())
    assert work.claimed == 1
    assert work.orphaned


def test_claimed_entries_with_a_live_drain_are_not_orphaned(data_dir):
    _claim_one("Docs")
    fd = spool.try_acquire_worker_lock("Docs")
    try:
        [work] = jobs.queued_work(jobs.running_jobs())
    finally:
        spool.release_worker_lock(fd)
    assert not work.orphaned


def test_corpora_with_no_entries_are_not_listed(data_dir):
    spool.worker_lock_path("Idle")
    assert jobs.queued_work([]) == []


# --- never in the way -----------------------------------------------------------


def test_listing_takes_no_locks(data_dir, monkeypatch):
    import fcntl

    fd = spool.try_acquire_worker_lock("Docs")
    try:
        with corpus_lock.acquire_corpus_lock("Other", quiet=True):

            def forbidden(*_args, **_kwargs):
                raise AssertionError("jobs must not call flock")

            with monkeypatch.context() as patch:
                patch.setattr(fcntl, "flock", forbidden)
                listed = jobs.running_jobs()
                jobs.queued_work(listed)
    finally:
        spool.release_worker_lock(fd)

    assert {j.corpus for j in listed} == {"Docs", "Other"}


def test_listing_an_empty_data_dir_creates_nothing(data_dir):
    assert jobs.running_jobs() == []
    assert jobs.queued_work([]) == []
    assert not (data_dir / "locks").exists()
    assert not (data_dir / "spool").exists()


# --- CLI ----------------------------------------------------------------------------


def test_cli_reports_nothing_running(data_dir):
    result = CliRunner().invoke(cli, ["corpus", "jobs"])
    assert result.exit_code == 0, result.output
    assert "No indexing jobs running" in result.output


def test_cli_lists_jobs_and_queued_work(data_dir):
    spool.write_entry("Queued", "/repo", changed=["/repo/a.py"], removed=[])
    with corpus_lock.acquire_corpus_lock("Docs", quiet=True):
        result = CliRunner().invoke(cli, ["corpus", "jobs"])
    assert result.exit_code == 0, result.output
    assert "Docs" in result.output
    assert str(os.getpid()) in result.output
    assert "Queued" in result.output


def test_cli_json(data_dir):
    _claim_one("Queued")
    with corpus_lock.acquire_corpus_lock("Docs", quiet=True):
        result = CliRunner().invoke(cli, ["corpus", "jobs", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "success"
    [job] = payload["data"]["jobs"]
    assert (job["corpus"], job["kind"], job["pid"]) == ("Docs", "write", os.getpid())
    assert job["elapsed_seconds"] >= 0
    [work] = payload["data"]["queued"]
    assert (work["corpus"], work["claimed"], work["orphaned"]) == ("Queued", 1, True)

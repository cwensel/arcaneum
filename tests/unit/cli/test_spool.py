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
    assert a.name.split(".")[-2] != b.name.split(".")[-2], "the repo is in the entry name"
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
    bad = spool.queue_dir("Docs") / "bad.json"
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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
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


# --- entries survive a failed drain (roborev 6104) ----------------------------


def test_drain_batch_can_defer_removing_entries(isolated_spool):
    """A caller that may fail needs the entries left on disk until it succeeds."""
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])

    batch, entries = spool.drain_batch("Docs", consume=False)

    assert batch.changed == ["/repo/a.py"]
    assert entries, "the caller needs the paths to release later"
    assert spool.has_pending("Docs"), "entries must survive until explicitly released"


def test_release_entries_removes_them(isolated_spool):
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    _, entries = spool.drain_batch("Docs", consume=False)

    spool.release_entries(entries)

    assert not spool.has_pending("Docs")


def test_release_entries_tolerates_an_already_removed_file(isolated_spool):
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    _, entries = spool.drain_batch("Docs", consume=False)
    entries[0].unlink()

    spool.release_entries(entries)  # must not raise


def test_deferred_drain_still_discards_malformed_entries(isolated_spool):
    """A malformed entry can never succeed; keeping it would loop forever."""
    bad = spool.queue_dir("Docs") / "bad.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not json")

    batch, entries = spool.drain_batch("Docs", consume=False)

    assert not batch
    assert not bad.exists()
    assert entries == []


# --- branch pinning (kata vr9b) -----------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture
def git_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@e.com")
    _git(root, "config", "user.name", "T")
    (root / "a.md").write_text("a\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


def test_write_entry_records_the_branch_it_was_spooled_for(isolated_spool):
    path = spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[], branch="main")
    assert json.loads(path.read_text())["branch"] == "main"


def test_drain_defers_entries_whose_repo_left_the_branch(isolated_spool, git_repo):
    """Indexing reads the working tree, which now holds another branch's files."""
    spool.write_entry("Docs", git_repo, changed=[str(git_repo / "a.md")], removed=[], branch="main")
    _git(git_repo, "checkout", "-q", "-b", "feature")

    batch, entries = spool.drain_batch("Docs", consume=False)

    assert not batch and entries == []
    assert spool.has_pending("Docs"), "the entry waits for the branch to come back"

    _git(git_repo, "checkout", "-q", "main")
    batch = spool.drain_batch("Docs")
    assert [Path(p).name for p in batch.changed] == ["a.md"]


def test_drain_defers_entries_while_head_is_detached(isolated_spool, git_repo):
    """A rebase in progress detaches HEAD; its working tree is in flux."""
    spool.write_entry("Docs", git_repo, changed=[str(git_repo / "a.md")], removed=[], branch="main")
    _git(git_repo, "checkout", "-q", "--detach")

    assert not spool.drain_batch("Docs")
    assert spool.has_pending("Docs")


def test_drain_does_not_wedge_on_a_repo_that_no_longer_exists(isolated_spool, tmp_path):
    """Branch state is unknowable there; deferring would keep the entry forever."""
    gone = tmp_path / "gone-repo"
    spool.write_entry("Docs", gone, changed=[], removed=[str(gone / "a.md")], branch="main")

    batch = spool.drain_batch("Docs")

    assert batch.removed == [str(gone / "a.md")]


def test_unpinned_entries_drain_regardless_of_branch(isolated_spool, git_repo):
    """Entries from hooks installed before pinning carry no branch."""
    spool.write_entry("Docs", git_repo, changed=[str(git_repo / "a.md")], removed=[])
    _git(git_repo, "checkout", "-q", "-b", "feature")

    assert spool.drain_batch("Docs").changed == [str(git_repo / "a.md")]


def test_batch_reports_the_branch_each_repo_was_pinned_to(isolated_spool, git_repo):
    spool.write_entry("Docs", git_repo, changed=[str(git_repo / "a.md")], removed=[], branch="main")
    assert spool.drain_batch("Docs").pins == {str(git_repo): "main"}


def test_mark_round_trips(isolated_spool):
    assert spool.read_mark("Docs", "/repo", "main") is None
    spool.write_mark("Docs", "/repo", "main", "abc123")
    assert spool.read_mark("Docs", "/repo", "main") == "abc123"


def test_marks_are_scoped_per_corpus_repo_and_branch(isolated_spool):
    spool.write_mark("Docs", "/repo", "main", "one")
    assert spool.read_mark("Code", "/repo", "main") is None
    assert spool.read_mark("Docs", "/other", "main") is None
    assert spool.read_mark("Docs", "/repo", "dev") is None


def test_branch_names_that_flatten_alike_get_separate_marks(isolated_spool):
    spool.write_mark("Docs", "/repo", "a/b", "slash")
    spool.write_mark("Docs", "/repo", "a_b", "underscore")
    assert spool.read_mark("Docs", "/repo", "a/b") == "slash"


def test_marks_live_outside_the_spool(isolated_spool):
    """Nothing that scans for pending work may ever see a mark."""
    spool.write_mark("Docs", "/repo", "main", "abc123")
    assert not spool.has_pending("Docs")
    assert spool.list_corpora_with_pending() == []
    assert isolated_spool not in spool.mark_path("Docs", "/repo", "main").parents


def test_clear_marks_removes_every_branch_for_the_repo(isolated_spool):
    spool.write_mark("Docs", "/repo", "main", "one")
    spool.write_mark("Docs", "/repo", "dev", "two")
    spool.write_mark("Docs", "/other", "main", "three")

    spool.clear_marks("Docs", "/repo")

    assert spool.read_mark("Docs", "/repo", "main") is None
    assert spool.read_mark("Docs", "/repo", "dev") is None
    assert spool.read_mark("Docs", "/other", "main") == "three"


def test_mark_path_is_data_dir_relative(isolated_spool, tmp_path):
    """The hook rebuilds this path from $XDG_DATA_HOME at run time."""
    relative = spool.mark_relpath("Docs", "/repo", "main")
    assert not relative.is_absolute()
    assert spool.mark_path("Docs", "/repo", "main") == tmp_path / "arcaneum" / relative


# --- the queue directory is empty exactly when nothing is drainable (kata 4j93)
#
# launchd QueueDirectories and systemd DirectoryNotEmpty= relaunch the drain for
# as long as the watched directory has any entry at all. A lock file, a temp
# file, a subdirectory or a parked entry left there relaunches it every ~10s
# forever: one real watcher had run 64737 times.


def test_the_queue_holds_nothing_but_entries(isolated_spool):
    path = spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    lock = spool.try_acquire_worker_lock("Docs")
    spool.release_worker_lock(lock)

    assert list(spool.queue_dir("Docs").iterdir()) == [path]
    assert spool.queue_dir("Docs") not in spool.worker_lock_path("Docs").parents


def test_entry_names_carry_the_repo_id(isolated_spool):
    """The hook finds its own repo's entries by name, without parsing them."""
    path = spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    assert path.name.endswith(f".{spool.repo_id('/repo')}.json")
    assert spool.repo_id("/repo") != spool.repo_id("/other")


def test_the_queue_is_empty_after_a_drain(isolated_spool):
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    spool.drain_batch("Docs")
    assert spool.queue_dir("Docs").is_dir()
    assert list(spool.queue_dir("Docs").iterdir()) == []


def test_claimed_entries_leave_the_queue_while_they_are_indexed(isolated_spool):
    """Otherwise the watcher relaunches a doomed drain every 10s of a long batch."""
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])

    batch, entries = spool.drain_batch("Docs", consume=False)

    assert batch.changed == ["/repo/a.py"]
    assert list(spool.queue_dir("Docs").iterdir()) == []
    assert all(entry.exists() for entry in entries)


def test_returned_entries_go_back_to_the_queue(isolated_spool):
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    _, entries = spool.drain_batch("Docs", consume=False)

    spool.return_entries(entries)

    assert spool.drain_batch("Docs").changed == ["/repo/a.py"]


def test_entries_a_dead_worker_claimed_are_recovered(isolated_spool):
    spool.write_entry("Docs", "/repo", changed=["/repo/a.py"], removed=[])
    spool.drain_batch("Docs", consume=False)  # the worker dies here

    spool.recover_claimed("Docs")

    assert spool.drain_batch("Docs").changed == ["/repo/a.py"]


def test_deferred_entries_leave_the_queue(isolated_spool, git_repo):
    spool.write_entry("Docs", git_repo, changed=[str(git_repo / "a.md")], removed=[], branch="main")
    _git(git_repo, "checkout", "-q", "-b", "feature")

    spool.drain_batch("Docs")

    assert list(spool.queue_dir("Docs").iterdir()) == []
    assert spool.has_pending("Docs")


def test_requeue_returns_a_repos_deferred_entries(isolated_spool, git_repo):
    spool.write_entry("Docs", git_repo, changed=[str(git_repo / "a.md")], removed=[], branch="main")
    spool.write_entry("Docs", "/other", changed=["/other/b.md"], removed=[], branch="main")
    _git(git_repo, "checkout", "-q", "-b", "feature")
    spool.drain_batch("Docs")  # /other is not a repo, so it drains; git_repo defers

    spool.requeue_deferred("Docs", git_repo)

    assert [p.name.split(".")[-2] for p in spool.queue_dir("Docs").iterdir()] == [
        spool.repo_id(git_repo)
    ]


def test_a_drain_requeues_deferred_entries_whose_branch_is_back(isolated_spool, git_repo):
    spool.write_entry("Docs", git_repo, changed=[str(git_repo / "a.md")], removed=[], branch="main")
    _git(git_repo, "checkout", "-q", "-b", "feature")
    spool.drain_batch("Docs")
    _git(git_repo, "checkout", "-q", "main")

    assert spool.drain_batch("Docs").changed == [str(git_repo / "a.md")]


def test_entries_in_the_old_layout_are_migrated(isolated_spool):
    """Spools written before the queue directory existed must still drain."""
    legacy = spool.corpus_spool_dir("Docs") / "0123456789ab"
    legacy.mkdir(parents=True)
    (legacy / "1.000000-1-0.json").write_text(
        json.dumps({"corpus": "Docs", "repo": "/repo", "changed": ["/repo/a.py"], "removed": []})
    )

    assert spool.drain_batch("Docs").changed == ["/repo/a.py"]
    assert not legacy.exists()
    assert sorted(p.name for p in spool.corpus_spool_dir("Docs").iterdir()) == ["queue"]


def test_stray_files_in_the_queue_are_removed(isolated_spool):
    """Anything that is not an entry would keep the watcher firing forever."""
    spool.queue_dir("Docs").mkdir(parents=True)
    (spool.queue_dir("Docs") / ".DS_Store").write_text("x")
    (spool.queue_dir("Docs") / "subdir").mkdir()

    spool.drain_batch("Docs")

    assert list(spool.queue_dir("Docs").iterdir()) == []

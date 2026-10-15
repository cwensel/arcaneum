"""`arc corpus sync --drain-spool`: the background worker (kata vq0n)."""

from __future__ import annotations

import platform
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from arcaneum.cli import spool
from arcaneum.cli.main import cli


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="flock-based spool worker is not supported on Windows",
)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    yield tmp_path


@pytest.fixture
def repo(tmp_path):
    """A directory of real files: the worker skips paths that no longer exist."""
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("a.py", "b.py", "1.py", "2.py", "3.py", "4.py"):
        (root / name).write_text("x\n")
    return root


def _run(args, impl):
    with patch("arcaneum.cli.sync.sync_directory_command", impl):
        return CliRunner().invoke(cli, args, catch_exceptions=False)


def test_drain_syncs_the_spooled_paths(isolated, repo):
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])
    calls = []

    result = _run(
        ["corpus", "sync", "Docs", "--drain-spool"],
        lambda corpus, paths, *a, **k: calls.append(
            {"corpus": corpus, "paths": list(paths), "removed": k.get("removed_paths")}
        ),
    )

    assert result.exit_code == 0, result.output
    assert calls == [{"corpus": "Docs", "paths": [str(repo / "a.py")], "removed": []}]


def test_drain_skips_paths_that_vanished_before_the_worker_ran(isolated, repo):
    """A file spooled then deleted must not be handed to the indexer."""
    gone = repo / "gone.py"
    spool.write_entry("Docs", repo, changed=[str(gone)], removed=[])
    calls = []

    result = _run(
        ["corpus", "sync", "Docs", "--drain-spool"],
        lambda *a, **k: calls.append(True),
    )

    assert result.exit_code == 0, result.output
    assert not calls


def test_drain_empties_the_spool(isolated, repo):
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])
    _run(["corpus", "sync", "Docs", "--drain-spool"], lambda *a, **k: None)
    assert not spool.has_pending("Docs")


def test_drain_loops_until_the_spool_is_empty(isolated, repo):
    """A commit landing mid-drain must be picked up without a second spawn."""
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])
    batches = []

    def impl(corpus, paths, *args, **kwargs):
        batches.append(list(paths))
        if len(batches) == 1:
            # Simulate a commit arriving while the first batch is indexing.
            spool.write_entry("Docs", repo, changed=[str(repo / "b.py")], removed=[])

    result = _run(["corpus", "sync", "Docs", "--drain-spool"], impl)

    assert result.exit_code == 0, result.output
    assert batches == [[str(repo / "a.py")], [str(repo / "b.py")]]
    assert not spool.has_pending("Docs")


def test_drain_on_an_empty_spool_does_nothing(isolated):
    called = []
    result = _run(["corpus", "sync", "Docs", "--drain-spool"], lambda *a, **k: called.append(True))
    assert result.exit_code == 0, result.output
    assert not called


def test_drain_declines_when_another_worker_holds_the_lock(isolated, repo):
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])
    called = []

    lock = spool.try_acquire_worker_lock("Docs")
    try:
        result = _run(
            ["corpus", "sync", "Docs", "--drain-spool"], lambda *a, **k: called.append(True)
        )
    finally:
        spool.release_worker_lock(lock)

    assert result.exit_code == 0, "declining is not an error"
    assert not called
    assert spool.has_pending("Docs"), "work must be left for the running worker"


def test_drain_releases_the_worker_lock_when_done(isolated, repo):
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])
    _run(["corpus", "sync", "Docs", "--drain-spool"], lambda *a, **k: None)

    lock = spool.try_acquire_worker_lock("Docs")
    assert lock is not None
    spool.release_worker_lock(lock)


def test_drain_releases_the_lock_when_a_batch_fails(isolated, repo):
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])

    def boom(*args, **kwargs):
        raise RuntimeError("indexing failed")

    with pytest.raises(RuntimeError):
        _run(["corpus", "sync", "Docs", "--drain-spool"], boom)

    lock = spool.try_acquire_worker_lock("Docs")
    assert lock is not None, "a failed drain must not wedge the corpus"
    spool.release_worker_lock(lock)


def test_drain_forwards_removed_paths(isolated, repo):
    spool.write_entry("Docs", repo, changed=[], removed=[str(repo / "gone.py")])
    calls = []

    _run(
        ["corpus", "sync", "Docs", "--drain-spool"],
        lambda corpus, paths, *a, **k: calls.append(k.get("removed_paths")),
    )

    assert calls == [[str(repo / "gone.py")]]


def test_drain_rejects_explicit_paths(isolated, tmp_path):
    result = CliRunner().invoke(cli, ["corpus", "sync", "Docs", str(tmp_path), "--drain-spool"])
    assert result.exit_code != 0
    assert "drain-spool" in result.output


def test_drain_stops_after_the_batch_limit(isolated, repo):
    """A pathological producer must not keep one worker looping forever."""
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])
    batches = []

    def impl(corpus, paths, *args, **kwargs):
        batches.append(list(paths))
        # Always leave more work behind.
        spool.write_entry("Docs", repo, changed=[str(repo / f"{len(batches)}.py")], removed=[])

    result = _run(["corpus", "sync", "Docs", "--drain-spool", "--max-batches", "3"], impl)

    assert result.exit_code == 0, result.output
    assert len(batches) == 3
    assert spool.has_pending("Docs"), "leftover work stays spooled for the next run"


def test_drain_skips_file_types_the_corpus_cannot_index(isolated, repo):
    """The hook spools every touched file; one it cannot index must not fail the batch.

    A commit touching `bin/rfd` or `.gitignore` next to markdown wedged a
    markdown corpus for days (kata 7g0v): the whole batch raised, the entry was
    kept, and launchd re-ran the drain every few seconds.
    """
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])
    seen = {}

    result = _run(
        ["corpus", "sync", "Docs", "--drain-spool"],
        lambda corpus, paths, *a, **k: seen.update(k),
    )

    assert result.exit_code == 0, result.output
    assert seen["skip_unsupported"] is True


# --- branch pinning (kata vr9b) -----------------------------------------------


def _git(repo, *args):
    import subprocess

    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture
def git_repo(repo):
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "T")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_drain_keeps_a_batch_when_the_repo_left_the_branch_mid_index(isolated, git_repo):
    """The files were read from another branch's working tree; redo them later."""
    spool.write_entry("Docs", git_repo, changed=[str(git_repo / "a.py")], removed=[], branch="main")

    def switch_branch_while_indexing(*a, **k):
        _git(git_repo, "checkout", "-q", "-b", "feature")

    result = _run(["corpus", "sync", "Docs", "--drain-spool"], switch_branch_while_indexing)

    assert result.exit_code == 0, result.output
    assert spool.has_pending("Docs"), "the entry must survive for a re-index on main"


def test_drain_releases_a_batch_when_the_repo_stayed_on_the_branch(isolated, git_repo):
    spool.write_entry("Docs", git_repo, changed=[str(git_repo / "a.py")], removed=[], branch="main")

    result = _run(["corpus", "sync", "Docs", "--drain-spool"], lambda *a, **k: None)

    assert result.exit_code == 0, result.output
    assert not spool.has_pending("Docs")


# --- the watched queue ends empty (kata 4j93) ---------------------------------


def _queued():
    queue = spool.queue_dir("Docs")
    return sorted(p.name for p in queue.iterdir()) if queue.is_dir() else []


def test_a_successful_drain_leaves_the_queue_empty(isolated, repo):
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])
    _run(["corpus", "sync", "Docs", "--drain-spool"], lambda *a, **k: None)
    assert _queued() == []


def test_a_failed_batch_goes_back_to_the_queue(isolated, repo):
    """Claimed while indexing; a failure must return it, or the work is stranded."""
    entry = spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])

    def boom(*a, **k):
        raise RuntimeError("services down")

    with pytest.raises(RuntimeError):
        _run(["corpus", "sync", "Docs", "--drain-spool"], boom)

    assert _queued() == [entry.name]
    assert not any(spool.claimed_dir("Docs").glob("*.json"))


def test_a_drain_recovers_entries_a_dead_worker_claimed(isolated, repo):
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])
    spool.drain_batch("Docs", consume=False)  # claimed, then the worker died
    calls = []

    _run(["corpus", "sync", "Docs", "--drain-spool"], lambda c, paths, *a, **k: calls.append(paths))

    assert calls == [(str(repo / "a.py"),)]
    assert not spool.has_pending("Docs")


def test_a_batch_whose_repo_left_the_branch_ends_deferred_not_queued(isolated, git_repo):
    spool.write_entry("Docs", git_repo, changed=[str(git_repo / "a.py")], removed=[], branch="main")

    def switch_branch_while_indexing(*a, **k):
        _git(git_repo, "checkout", "-q", "-b", "feature")

    _run(["corpus", "sync", "Docs", "--drain-spool"], switch_branch_while_indexing)

    assert _queued() == [], "a parked entry in the queue would keep the watcher firing"
    assert spool.has_pending("Docs")

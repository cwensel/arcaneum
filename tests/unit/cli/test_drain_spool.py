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
    assert calls == [
        {"corpus": "Docs", "paths": [str(repo / "a.py")], "removed": []}
    ]


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
    result = CliRunner().invoke(
        cli, ["corpus", "sync", "Docs", str(tmp_path), "--drain-spool"]
    )
    assert result.exit_code != 0
    assert "drain-spool" in result.output


def test_drain_stops_after_the_batch_limit(isolated, repo):
    """A pathological producer must not keep one worker looping forever."""
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])
    batches = []

    def impl(corpus, paths, *args, **kwargs):
        batches.append(list(paths))
        # Always leave more work behind.
        spool.write_entry(
            "Docs", repo, changed=[str(repo / f"{len(batches)}.py")], removed=[]
        )

    result = _run(["corpus", "sync", "Docs", "--drain-spool", "--max-batches", "3"], impl)

    assert result.exit_code == 0, result.output
    assert len(batches) == 3
    assert spool.has_pending("Docs"), "leftover work stays spooled for the next run"

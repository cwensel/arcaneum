"""`arc corpus sync --changed-since <rev>` (kata vq0n)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from arcaneum.cli import sync as sync_mod
from arcaneum.cli.main import cli


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / "a.py").write_text("a\n")
    (root / "b.py").write_text("b\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")
    return root


@pytest.fixture
def isolated_locks(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))


def _run(args, impl):
    with patch("arcaneum.cli.sync._sync_directory_locked", impl):
        return CliRunner().invoke(cli, args, catch_exceptions=False)


def test_changed_since_passes_only_touched_paths_to_sync(repo, isolated_locks):
    (repo / "b.py").write_text("b changed\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "touch b")

    seen = {}

    def impl(corpus, paths, *args, **kwargs):
        seen["paths"] = [Path(p).name for p in paths]

    result = _run(
        ["corpus", "sync", "Docs", str(repo), "--changed-since", "HEAD"], impl
    )

    assert result.exit_code == 0, result.output
    assert seen["paths"] == ["b.py"], "unchanged a.py must not be re-synced"


def test_changed_since_forwards_removed_paths(repo, isolated_locks):
    (repo / "a.py").unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "drop a")

    seen = {}

    def impl(corpus, paths, *args, **kwargs):
        seen["paths"] = list(paths)
        seen["removed"] = [Path(p).name for p in kwargs.get("removed_paths") or []]

    result = _run(
        ["corpus", "sync", "Docs", str(repo), "--changed-since", "HEAD"], impl
    )

    assert result.exit_code == 0, result.output
    assert seen["removed"] == ["a.py"]
    assert seen["paths"] == []


def test_changed_since_handles_a_rename(repo, isolated_locks):
    _git(repo, "mv", "a.py", "renamed.py")
    _git(repo, "commit", "-q", "-m", "rename")

    seen = {}

    def impl(corpus, paths, *args, **kwargs):
        seen["paths"] = [Path(p).name for p in paths]
        seen["removed"] = [Path(p).name for p in kwargs.get("removed_paths") or []]

    result = _run(["corpus", "sync", "Docs", str(repo), "--changed-since", "HEAD"], impl)

    assert result.exit_code == 0, result.output
    assert seen["paths"] == ["renamed.py"]
    assert seen["removed"] == ["a.py"]


def test_changed_since_accepts_a_range(repo, isolated_locks):
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "c.py").write_text("c\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c")
    (repo / "d.py").write_text("d\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "d")

    seen = {}

    result = _run(
        ["corpus", "sync", "Docs", str(repo), "--changed-since", f"{base}..HEAD"],
        lambda corpus, paths, *a, **k: seen.setdefault(
            "paths", sorted(Path(p).name for p in paths)
        ),
    )

    assert result.exit_code == 0, result.output
    assert seen["paths"] == ["c.py", "d.py"]


def test_changed_since_defaults_to_cwd_when_no_path_given(repo, isolated_locks, monkeypatch):
    (repo / "b.py").write_text("b2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "touch b")
    monkeypatch.chdir(repo)

    seen = {}
    result = _run(
        ["corpus", "sync", "Docs", "--changed-since", "HEAD"],
        lambda corpus, paths, *a, **k: seen.setdefault(
            "paths", [Path(p).name for p in paths]
        ),
    )

    assert result.exit_code == 0, result.output
    assert seen["paths"] == ["b.py"]


def test_changed_since_on_a_non_repo_errors(tmp_path, isolated_locks):
    result = CliRunner().invoke(
        cli, ["corpus", "sync", "Docs", str(tmp_path), "--changed-since", "HEAD"]
    )
    assert result.exit_code != 0


def test_changed_since_with_nothing_to_do_exits_clean(repo, isolated_locks):
    """An empty commit touches no files; that is success, not an error."""
    _git(repo, "commit", "-q", "--allow-empty", "-m", "empty")

    called = []
    result = _run(
        ["corpus", "sync", "Docs", str(repo), "--changed-since", "HEAD"],
        lambda *a, **k: called.append(True),
    )

    assert result.exit_code == 0, result.output
    assert not called, "no work means the heavy sync body should not run"


def test_changed_since_is_mutually_exclusive_with_from_file(repo, isolated_locks, tmp_path):
    listing = tmp_path / "paths.txt"
    listing.write_text(str(repo / "a.py") + "\n")
    result = CliRunner().invoke(
        cli,
        [
            "corpus", "sync", "Docs", str(repo),
            "--changed-since", "HEAD",
            "--from-file", str(listing),
        ],
    )
    assert result.exit_code != 0
    assert "changed-since" in result.output


def test_changed_since_skips_paths_deleted_from_the_worktree(repo, isolated_locks):
    """A path git reports as changed but that is gone now must not be indexed."""
    (repo / "c.py").write_text("c\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add c")
    (repo / "c.py").unlink()  # removed after the commit, before the sync runs

    seen = {}
    result = _run(
        ["corpus", "sync", "Docs", str(repo), "--changed-since", "HEAD"],
        lambda corpus, paths, *a, **k: seen.setdefault("paths", list(paths)),
    )

    assert result.exit_code == 0, result.output
    assert seen.get("paths", []) == []

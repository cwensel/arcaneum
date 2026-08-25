"""Tests for git diff parsing behind `--changed-since` and the sync hook (kata vq0n)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from arcaneum.cli import git_changes
from arcaneum.cli.errors import InvalidArgumentError


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    (root / "a.py").write_text("print('a')\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")
    return root


# --- parse_name_status --------------------------------------------------------


def test_parses_added_and_modified():
    changes = git_changes.parse_name_status("A\0new.py\0M\0old.py\0")
    assert changes.changed == ["new.py", "old.py"]
    assert changes.removed == []


def test_parses_deleted():
    changes = git_changes.parse_name_status("D\0gone.py\0")
    assert changes.changed == []
    assert changes.removed == ["gone.py"]


def test_rename_yields_new_path_as_changed_and_old_as_removed():
    changes = git_changes.parse_name_status("R100\0old.py\0new.py\0")
    assert changes.changed == ["new.py"]
    assert changes.removed == ["old.py"]


def test_copy_yields_only_the_new_path():
    """C is a copy: the source still exists, so it must not be removed."""
    changes = git_changes.parse_name_status("C75\0src.py\0copy.py\0")
    assert changes.changed == ["copy.py"]
    assert changes.removed == []


def test_type_change_is_treated_as_changed():
    changes = git_changes.parse_name_status("T\0link.py\0")
    assert changes.changed == ["link.py"]
    assert changes.removed == []


def test_unmerged_and_unknown_statuses_are_ignored():
    changes = git_changes.parse_name_status("U\0conflict.py\0X\0weird.py\0")
    assert changes.changed == []
    assert changes.removed == []


def test_empty_input_yields_nothing():
    assert git_changes.parse_name_status("") == git_changes.GitChanges([], [])
    assert git_changes.parse_name_status("\0") == git_changes.GitChanges([], [])


def test_paths_with_spaces_survive_nul_parsing():
    changes = git_changes.parse_name_status("M\0dir with spaces/a b.py\0")
    assert changes.changed == ["dir with spaces/a b.py"]


def test_duplicate_paths_are_deduped_preserving_order():
    changes = git_changes.parse_name_status("M\0a.py\0M\0a.py\0A\0b.py\0")
    assert changes.changed == ["a.py", "b.py"]


def test_path_deleted_then_readded_is_changed_not_removed():
    """Across a range a file can appear as both; the final state wins."""
    changes = git_changes.parse_name_status("D\0a.py\0A\0a.py\0")
    assert changes.changed == ["a.py"]
    assert changes.removed == []


def test_path_added_then_deleted_is_removed_not_changed():
    changes = git_changes.parse_name_status("A\0a.py\0D\0a.py\0")
    assert changes.changed == []
    assert changes.removed == ["a.py"]


# --- changes_since (real repo) ------------------------------------------------


def test_changes_since_head_lists_the_last_commit(repo):
    (repo / "b.py").write_text("print('b')\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add b")

    changes = git_changes.changes_since(repo, "HEAD")
    assert [Path(p).name for p in changes.changed] == ["b.py"]
    assert changes.removed == []


def test_changes_since_returns_absolute_paths(repo):
    (repo / "b.py").write_text("print('b')\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add b")

    changes = git_changes.changes_since(repo, "HEAD")
    assert all(Path(p).is_absolute() for p in changes.changed)
    assert Path(changes.changed[0]).resolve() == (repo / "b.py").resolve()


def test_changes_since_range_spans_multiple_commits(repo):
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "b.py").write_text("b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "b")
    (repo / "c.py").write_text("c\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c")

    changes = git_changes.changes_since(repo, f"{base}..HEAD")
    assert sorted(Path(p).name for p in changes.changed) == ["b.py", "c.py"]


def test_changes_since_detects_deletes(repo):
    (repo / "a.py").unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "remove a")

    changes = git_changes.changes_since(repo, "HEAD")
    assert changes.changed == []
    assert [Path(p).name for p in changes.removed] == ["a.py"]


def test_changes_since_detects_renames(repo):
    _git(repo, "mv", "a.py", "renamed.py")
    _git(repo, "commit", "-q", "-m", "rename")

    changes = git_changes.changes_since(repo, "HEAD")
    assert [Path(p).name for p in changes.changed] == ["renamed.py"]
    assert [Path(p).name for p in changes.removed] == ["a.py"]


def test_changes_since_root_commit_lists_all_files(repo):
    """HEAD on a repo's first commit has no parent; treat every file as added."""
    changes = git_changes.changes_since(repo, "HEAD")
    assert [Path(p).name for p in changes.changed] == ["a.py"]


def test_changes_since_rejects_a_non_repo(tmp_path):
    with pytest.raises(InvalidArgumentError):
        git_changes.changes_since(tmp_path, "HEAD")


def test_changes_since_rejects_an_unknown_revision(repo):
    with pytest.raises(InvalidArgumentError) as exc_info:
        git_changes.changes_since(repo, "nope-not-a-rev")
    assert "nope-not-a-rev" in str(exc_info.value)


def test_changes_since_rejects_a_revision_that_looks_like_a_flag(repo):
    """A leading dash must not be smuggled through as a git option."""
    with pytest.raises(InvalidArgumentError):
        git_changes.changes_since(repo, "--output=/tmp/pwned")


def test_repo_root_resolves_from_a_subdirectory(repo):
    sub = repo / "pkg" / "deep"
    sub.mkdir(parents=True)
    assert git_changes.repo_root(sub).resolve() == repo.resolve()


def test_repo_root_returns_none_outside_a_repo(tmp_path):
    assert git_changes.repo_root(tmp_path) is None

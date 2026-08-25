"""Each hook point must diff the range git actually gave it (kata vq0n follow-up).

Empirically, git passes:
  post-checkout  $1=old-sha  $2=new-sha  $3=1 for a branch switch, 0 for a file checkout
  post-rewrite   $1=reason,  old/new sha pairs on stdin
  post-merge     $1=squash-flag; ORIG_HEAD..HEAD is the moved range
  post-commit    (no args);  HEAD is the new commit

Using ORIG_HEAD..HEAD for the first two under-reports or misreports what moved.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

import pytest

from arcaneum.cli import hooks, spool


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="git hooks are shell scripts; not a supported target",
)


def _git(repo: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=check
    ).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@e.com")
    _git(root, "config", "user.name", "T")
    (root / "base.md").write_text("base\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    return root


def _env(repo: Path) -> dict:
    env = dict(os.environ)
    env["XDG_DATA_HOME"] = os.environ["XDG_DATA_HOME"]
    env["PATH"] = f"{Path(__file__).resolve().parents[3] / '.venv' / 'bin'}:{env['PATH']}"
    return env


def _drain_names(corpus: str) -> tuple[list[str], list[str]]:
    batch = spool.drain_batch(corpus)
    return (
        sorted(Path(p).name for p in batch.changed),
        sorted(Path(p).name for p in batch.removed),
    )


# --- the generated script uses the right revision expression ------------------


def test_post_checkout_block_uses_the_passed_shas(tmp_path):
    body = hooks.render_block("Docs", "post-checkout", tmp_path, spawn=False)
    assert "$1" in body and "$2" in body
    assert "ORIG_HEAD" not in body, "post-checkout is told the range; it must not guess"


def test_post_checkout_block_skips_file_checkouts(tmp_path):
    """$3=0 means `git checkout -- file`: content moved, but no commit range."""
    body = hooks.render_block("Docs", "post-checkout", tmp_path, spawn=False)
    assert "$3" in body


def test_post_rewrite_block_reads_stdin_pairs(tmp_path):
    body = hooks.render_block("Docs", "post-rewrite", tmp_path, spawn=False)
    assert "read" in body, "the SHAs arrive on stdin"
    # It compares two trees rather than walking the commits between them, so a
    # dropped commit's file is still reported (see test_post_rewrite_range).
    assert "--between" in body


def test_post_merge_still_uses_orig_head(tmp_path):
    """post-merge is not handed SHAs, so ORIG_HEAD..HEAD remains correct."""
    body = hooks.render_block("Docs", "post-merge", tmp_path, spawn=False)
    assert "ORIG_HEAD..HEAD" in body


def test_post_commit_still_uses_head(tmp_path):
    body = hooks.render_block("Docs", "post-commit", tmp_path, spawn=False)
    assert "HEAD" in body
    assert "ORIG_HEAD" not in body


@pytest.mark.parametrize("hook", hooks.SUPPORTED_HOOKS)
def test_every_generated_hook_is_valid_shell(hook, tmp_path):
    body = hooks.render_block("Docs", hook, tmp_path, spawn=False)
    script = tmp_path / f"{hook}.sh"
    script.write_text(f"#!/bin/sh\n{body}")
    check = subprocess.run(["/bin/sh", "-n", str(script)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


# --- end to end against real git operations -----------------------------------


def test_branch_switch_spanning_commits_reports_every_file(repo):
    """The bug: ORIG_HEAD..HEAD under-reported a switch spanning many commits."""
    hooks.install("Docs", repo, "post-checkout", spawn=False)

    _git(repo, "checkout", "-q", "-b", "feature")
    for name in ("one.md", "two.md", "three.md"):
        (repo / name).write_text(f"{name}\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", name)

    spool.drain_batch("Docs")  # discard anything queued so far

    # Switching back drops all three files relative to the new HEAD.
    result = subprocess.run(
        ["git", "checkout", "main"], cwd=repo, capture_output=True, text=True, env=_env(repo)
    )
    assert result.returncode == 0, result.stderr

    changed, removed = _drain_names("Docs")
    assert removed == ["one.md", "three.md", "two.md"]


def test_switching_forward_reports_the_added_files(repo):
    hooks.install("Docs", repo, "post-checkout", spawn=False)
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "added.md").write_text("added\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add")
    _git(repo, "checkout", "-q", "main")
    spool.drain_batch("Docs")

    result = subprocess.run(
        ["git", "checkout", "feature"], cwd=repo, capture_output=True, text=True, env=_env(repo)
    )
    assert result.returncode == 0, result.stderr

    changed, removed = _drain_names("Docs")
    assert changed == ["added.md"]


def test_file_checkout_queues_nothing(repo):
    """`git checkout -- file` has no commit range; it must not spool a bogus one."""
    hooks.install("Docs", repo, "post-checkout", spawn=False)
    (repo / "base.md").write_text("dirty\n")

    result = subprocess.run(
        ["git", "checkout", "--", "base.md"],
        cwd=repo, capture_output=True, text=True, env=_env(repo),
    )
    assert result.returncode == 0, result.stderr

    changed, removed = _drain_names("Docs")
    assert changed == [] and removed == []


def test_amend_reports_the_amended_files(repo):
    hooks.install("Docs", repo, "post-rewrite", spawn=False)
    (repo / "work.md").write_text("work\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "work")
    spool.drain_batch("Docs")

    (repo / "extra.md").write_text("extra\n")
    _git(repo, "add", "-A")
    result = subprocess.run(
        ["git", "commit", "--amend", "-m", "work amended"],
        cwd=repo, capture_output=True, text=True, env=_env(repo),
    )
    assert result.returncode == 0, result.stderr

    changed, removed = _drain_names("Docs")
    assert "extra.md" in changed


def test_rebase_reports_what_changed_on_disk(repo):
    """A rebase must report the files whose working-tree content actually moved.

    Commits merely replayed onto a new base keep identical blobs, so they need
    no re-index; the file the rebase brings in from the new base does.
    """
    hooks.install("Docs", repo, "post-rewrite", spawn=False)

    _git(repo, "checkout", "-q", "-b", "feature")
    for name in ("f1.md", "f2.md"):
        (repo / name).write_text(f"{name}\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", name)

    _git(repo, "checkout", "-q", "main")
    (repo / "on-main.md").write_text("main\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main work")

    _git(repo, "checkout", "-q", "feature")
    spool.drain_batch("Docs")

    result = subprocess.run(
        ["git", "rebase", "main"], cwd=repo, capture_output=True, text=True, env=_env(repo)
    )
    assert result.returncode == 0, result.stderr

    changed, removed = _drain_names("Docs")
    assert "on-main.md" in changed, f"got changed={changed}"


def test_hooks_never_break_the_git_command_they_run_under(repo):
    """Every rewritten hook must still exit 0 even with arc absent."""
    for hook in hooks.SUPPORTED_HOOKS:
        hooks.install("Docs", repo, hook, spawn=False)

    env = dict(os.environ)
    env["PATH"] = "/nonexistent"
    env["HOME"] = str(repo)
    for hook in hooks.SUPPORTED_HOOKS:
        script = repo / ".git" / "hooks" / hook
        result = subprocess.run(
            ["/bin/sh", str(script), "arg1", "arg2", "1"],
            cwd=repo, capture_output=True, text=True, env=env, input="",
        )
        assert result.returncode == 0, f"{hook}: {result.stderr}"

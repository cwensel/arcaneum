"""post-rewrite should spool one range, not one entry per rewritten commit.

Measured on a real rebase: 1057 spool entries for 198 distinct files -- ~5x
redundancy, because the hook fired once per rewritten commit and each spooled
that commit's diff. Indexing reads the working tree, so per-commit granularity
buys nothing: only the final tree state matters.

A range from the first rewrite's old parent to the last rewrite's new SHA is
also strictly more correct. Verified against real git: a rebase that drops a
commit reports only the surviving pairs on stdin, so files touched solely by
the dropped commit appear in no pair -- but they do appear in the range.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import pytest

from arcaneum.cli import hooks, spool


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="git hooks are shell scripts; not a supported target",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    yield tmp_path


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@e.com")
    _git(root, "config", "user.name", "T")
    (root / "base.md").write_text("base\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


# --- the generated shell ------------------------------------------------------


def test_post_rewrite_spools_one_range(tmp_path):
    body = hooks.render_block("Docs", "post-rewrite", tmp_path, spawn=False)
    # It must still read stdin (that is where the SHAs are) but collapse to one
    # spool call rather than one per line.
    assert "read" in body
    assert "_arc_first_old" in body and "_arc_last_new" in body


def test_post_rewrite_shell_is_valid(tmp_path):
    body = hooks.render_block("Docs", "post-rewrite", tmp_path, spawn=False)
    script = tmp_path / "h.sh"
    script.write_text(f"#!/bin/sh\n{body}")
    r = subprocess.run(["/bin/sh", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# --- end to end ---------------------------------------------------------------


def _drain_names(corpus: str):
    batch = spool.drain_batch(corpus)
    return sorted(Path(p).name for p in batch.changed)


def test_a_multi_commit_rebase_spools_one_entry(isolated, repo):
    """The redundancy fix: N rewritten commits must not mean N spool entries."""
    hooks.install("Docs", repo, "post-rewrite", spawn=False)

    _git(repo, "checkout", "-q", "-b", "feature")
    for name in ("f1.md", "f2.md", "f3.md"):
        (repo / name).write_text(f"{name}\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", name)

    _git(repo, "checkout", "-q", "main")
    (repo / "on-main.md").write_text("main\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main work")
    _git(repo, "checkout", "-q", "feature")
    spool.drain_batch("Docs")

    _git(repo, "rebase", "main")

    entries = list(spool.corpus_spool_dir("Docs").rglob("*.json"))
    assert len(entries) == 1, f"expected one entry for the whole rebase, got {len(entries)}"


def test_the_single_entry_still_covers_every_rewritten_file(isolated, repo):
    hooks.install("Docs", repo, "post-rewrite", spawn=False)

    _git(repo, "checkout", "-q", "-b", "feature")
    for name in ("f1.md", "f2.md", "f3.md"):
        (repo / name).write_text(f"{name}\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", name)

    _git(repo, "checkout", "-q", "main")
    (repo / "on-main.md").write_text("main\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main work")
    _git(repo, "checkout", "-q", "feature")
    spool.drain_batch("Docs")

    _git(repo, "rebase", "main")

    names = _drain_names("Docs")
    for expected in ("f1.md", "f2.md", "f3.md"):
        assert expected in names, f"{expected} missing from {names}"


def test_an_amend_still_reports_its_files(isolated, repo):
    hooks.install("Docs", repo, "post-rewrite", spawn=False)
    (repo / "work.md").write_text("work\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "work")
    spool.drain_batch("Docs")

    (repo / "extra.md").write_text("extra\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--amend", "-m", "work amended")

    assert "extra.md" in _drain_names("Docs")


def test_a_dropped_commit_is_still_covered(isolated, repo):
    """Per-commit pairs miss files touched only by a dropped commit; a range does not."""
    hooks.install("Docs", repo, "post-rewrite", spawn=False)

    for name in ("keep1.md", "dropme.md", "keep2.md"):
        (repo / name).write_text(f"{name}\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", name)
    spool.drain_batch("Docs")

    # Drop the middle commit: HEAD~2 (dropme) is skipped, HEAD~1..HEAD replayed.
    _git(repo, "rebase", "--onto", "HEAD~3", "HEAD~2", "main")

    names = _drain_names("Docs")
    assert "dropme.md" in names, (
        "a file removed by the dropped commit must be re-checked; "
        f"got {names}"
    )


def test_post_rewrite_never_fails_the_git_command(isolated, repo):
    hooks.install("Docs", repo, "post-rewrite", spawn=False)
    script = repo / ".git" / "hooks" / "post-rewrite"
    import os

    env = dict(os.environ)
    env["PATH"] = "/nonexistent"
    env["HOME"] = str(repo)
    r = subprocess.run(
        ["/bin/sh", str(script), "rebase"],
        cwd=repo, capture_output=True, text=True, env=env,
        input="aaa bbb\nccc ddd\n",
    )
    assert r.returncode == 0, r.stderr


def test_empty_stdin_spools_nothing(isolated, repo):
    """A post-rewrite with no pairs must not spool a bogus range."""
    hooks.install("Docs", repo, "post-rewrite", spawn=False)
    script = repo / ".git" / "hooks" / "post-rewrite"
    import os

    subprocess.run(
        ["/bin/sh", str(script), "amend"],
        cwd=repo, capture_output=True, text=True, env=dict(os.environ), input="",
    )
    assert list(spool.corpus_spool_dir("Docs").rglob("*.json")) == []

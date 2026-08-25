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


def test_post_rewrite_compares_endpoints_not_a_walked_range(tmp_path):
    body = hooks.render_block("Docs", "post-rewrite", tmp_path, spawn=False)
    # A walked range would miss a dropped commit's file; endpoints do not.
    assert "--between" in body
    assert ".." not in body.split("--between")[1].split("\n")[0]


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


def test_the_single_entry_covers_what_actually_changed_on_disk(isolated, repo):
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
    # f1-f3 were replayed with identical blobs, so their on-disk content did
    # not change and re-indexing them would be waste. What did change is the
    # file the rebase brought in from the new base.
    assert "on-main.md" in names, f"the rebase's real disk change is missing: {names}"


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
    """A file whose only commit was dropped must be de-indexed.

    History base -> keep1 -> dropme -> keep2. The rebase replays only keep2
    onto keep1, so `dropme` is genuinely dropped and dropme.md disappears from
    the working tree. A per-commit walk of the surviving rewrites never
    mentions it; comparing the two tip trees reports it as a deletion.
    """
    hooks.install("Docs", repo, "post-rewrite", spawn=False)

    for name in ("keep1.md", "dropme.md", "keep2.md"):
        (repo / name).write_text(f"{name}\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", name)
    spool.drain_batch("Docs")

    _git(repo, "rebase", "--onto", "HEAD~2", "HEAD~1", "main")

    assert not (repo / "dropme.md").exists(), "precondition: the file is gone from disk"

    batch = spool.drain_batch("Docs")
    removed = sorted(Path(p).name for p in batch.removed)
    assert "dropme.md" in removed, (
        f"the dropped commit's file must be de-indexed; got removed={removed} "
        f"changed={sorted(Path(p).name for p in batch.changed)}"
    )


def test_a_pure_resign_rewrite_spools_nothing(isolated, repo):
    """Re-signing rewrites every SHA but no tree: there is nothing to re-index."""
    hooks.install("Docs", repo, "post-rewrite", spawn=False)
    (repo / "work.md").write_text("work\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "work")
    spool.drain_batch("Docs")

    _git(repo, "commit", "-q", "--amend", "-m", "same content, new sha")

    batch = spool.drain_batch("Docs")
    assert not batch, f"expected no work; got {batch}"


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


# --- amend must not use a stale ORIG_HEAD (roborev 6167) ---------------------


def test_amend_ignores_a_stale_orig_head(isolated, repo):
    """`git commit --amend` does not update ORIG_HEAD; a prior merge left it set.

    Using it as the base makes every amend re-index everything changed since
    that stale ref -- and if the amend reverts a file to its ORIG_HEAD value,
    the diff reports nothing and the index is left stale.
    """
    hooks.install("Docs", repo, "post-rewrite", spawn=False)

    # A merge sets ORIG_HEAD to the pre-merge tip and brings in sidefile.md.
    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "sidefile.md").write_text("side\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "side")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "side", "-m", "merge")
    assert _git(repo, "rev-parse", "ORIG_HEAD").strip()
    spool.drain_batch("Docs")

    # Now amend, touching only newfile.md.
    (repo / "newfile.md").write_text("new\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--amend", "-m", "amended")

    names = _drain_names("Docs")
    assert names == ["newfile.md"], (
        f"the amend touched one file; a stale ORIG_HEAD base over-reports: {names}"
    )


def test_amend_that_reverts_to_the_orig_head_value_is_still_reported(isolated, repo):
    """The correctness half: a revert-to-ORIG_HEAD must not vanish from the diff."""
    hooks.install("Docs", repo, "post-rewrite", spawn=False)

    (repo / "f.md").write_text("v1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v1")
    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "other.md").write_text("o\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "other")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "side", "-m", "merge")

    # A commit changes f.md, then the amend puts it back to its merge-time value.
    (repo / "f.md").write_text("v2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v2")
    spool.drain_batch("Docs")

    (repo / "f.md").write_text("v1\n")
    (repo / "marker.md").write_text("marker\n")  # keep the amend non-empty
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--amend", "-m", "reverted")

    assert "f.md" in _drain_names("Docs"), (
        "the working tree changed back to v1; the index still holds v2"
    )


def test_rebase_still_uses_the_pre_rewrite_tip(isolated, repo):
    """rebase does set ORIG_HEAD, and it is the right base there."""
    hooks.install("Docs", repo, "post-rewrite", spawn=False)

    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "feat.md").write_text("feat\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feat")
    _git(repo, "checkout", "-q", "main")
    (repo / "on-main.md").write_text("m\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "mainwork")
    _git(repo, "checkout", "-q", "feature")
    spool.drain_batch("Docs")

    _git(repo, "rebase", "main")

    assert "on-main.md" in _drain_names("Docs")

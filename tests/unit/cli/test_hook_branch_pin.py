"""Hooks pinned to one branch, caught up from the last spooled commit (kata vr9b).

Every installed hook point runs the same gate: when the repo has the pinned
branch checked out and its tip differs from the recorded mark, spool the tree
diff mark..tip and advance the mark. These tests drive real git operations,
including the ones no post-* hook sees (plumbing, reset), and the ones that must
not index anything (other branches, linked worktrees).
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from arcaneum.cli import hooks, spool
from arcaneum.cli.main import cli

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="git hooks are shell scripts; not a supported target",
)

VENV_BIN = Path(__file__).resolve().parents[3] / ".venv" / "bin"


def _env() -> dict:
    env = dict(os.environ)
    env["PATH"] = f"{VENV_BIN}:{env['PATH']}"
    return env


def _git(repo: Path, *args: str) -> str:
    """Run git with the installed hooks firing, as a user's command would."""
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, env=_env())
    assert result.returncode == 0, f"git {' '.join(args)}: {result.stderr}"
    return result.stdout


def _quiet_git(repo: Path, *args: str) -> str:
    """Run git with every hook disabled: an event the hooks never saw."""
    return _git(repo, "-c", "core.hooksPath=/dev/null", *args)


def _commit(repo: Path, name: str, *, hooked: bool = True) -> None:
    run = _git if hooked else _quiet_git
    (repo / name).write_text(f"{name}\n")
    run(repo, "add", "-A")
    run(repo, "commit", "-q", "-m", name)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    _quiet_git(root, "init", "-q", "-b", "main")
    _quiet_git(root, "config", "user.email", "t@e.com")
    _quiet_git(root, "config", "user.name", "T")
    _commit(root, "base.md", hooked=False)
    return root.resolve()


def _install(repo: Path, **kwargs) -> None:
    for hook in hooks.default_hooks(repo):
        hooks.install("Docs", repo, hook, spawn=False, **kwargs)


def _drained(corpus: str = "Docs") -> tuple[list[str], list[str]]:
    batch = spool.drain_batch(corpus)
    return (
        sorted(Path(p).name for p in batch.changed),
        sorted(Path(p).name for p in batch.removed),
    )


def _head(repo: Path, rev: str = "HEAD") -> str:
    return _quiet_git(repo, "rev-parse", rev).strip()


# --- what reaches the pinned branch is spooled ---------------------------------


def test_commit_on_the_pinned_branch_is_spooled(repo):
    _install(repo)
    _commit(repo, "new.md")
    assert _drained() == (["new.md"], [])


def test_a_commit_is_spooled_once_though_several_hooks_fire(repo):
    _install(repo)
    _commit(repo, "new.md")
    assert len(list(spool.corpus_spool_dir("Docs").rglob("*.json"))) == 1


def test_plumbing_commit_on_the_pinned_branch_is_spooled(repo):
    """commit-tree + update-ref fires no post-* hook; tools write main this way."""
    _install(repo)
    (repo / "tool.md").write_text("tool\n")
    _git(repo, "add", "tool.md")
    tree = _git(repo, "write-tree").strip()
    commit = _git(repo, "commit-tree", tree, "-p", "HEAD", "-m", "tool").strip()
    _git(repo, "update-ref", "refs/heads/main", commit)

    assert _drained() == (["tool.md"], [])


def test_reset_to_an_earlier_commit_reports_the_dropped_file(repo):
    _install(repo)
    _commit(repo, "dropped.md")
    spool.drain_batch("Docs")

    _git(repo, "reset", "-q", "--hard", "HEAD~1")

    assert _drained() == ([], ["dropped.md"])


def test_amend_reports_the_amended_files(repo):
    _install(repo)
    _commit(repo, "work.md")
    spool.drain_batch("Docs")

    (repo / "extra.md").write_text("extra\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--amend", "-m", "work amended")

    assert _drained() == (["extra.md"], [])


def test_no_ff_merge_reports_what_the_merge_brought_in(repo):
    _install(repo)
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "merged.md")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "--no-edit", "feature")

    assert _drained() == (["merged.md"], [])


def test_conflicted_merge_finished_by_commit_reports_the_merged_files(repo):
    """`git commit` concluding a merge fires post-commit on a merge commit."""
    _install(repo)
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "only-on-feature.md")
    (repo / "conflict.md").write_text("feature\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feature side")
    _git(repo, "checkout", "-q", "main")
    (repo / "conflict.md").write_text("main\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main side")
    spool.drain_batch("Docs")

    merge = subprocess.run(
        ["git", "merge", "feature"], cwd=repo, capture_output=True, text=True, env=_env()
    )
    assert merge.returncode != 0, "the merge must stop on the conflict"
    (repo / "conflict.md").write_text("resolved\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--no-edit")

    assert _drained() == (["conflict.md", "only-on-feature.md"], [])


def test_pull_with_rebase_reports_the_upstream_files(repo, tmp_path):
    clone = tmp_path / "clone"
    _quiet_git(tmp_path, "clone", "-q", str(repo), str(clone))
    _quiet_git(clone, "config", "user.email", "t@e.com")
    _quiet_git(clone, "config", "user.name", "T")
    _install(clone)
    _commit(clone, "local.md")
    _commit(repo, "upstream.md", hooked=False)
    spool.drain_batch("Docs")

    _git(clone, "-c", "pull.rebase=true", "pull", "-q")

    changed, removed = _drained()
    assert "upstream.md" in changed and removed == []


def test_missed_events_are_caught_up_by_the_next_one(repo):
    """The mark makes the gate self-healing: nothing slips between hook runs."""
    _install(repo)
    _commit(repo, "unseen.md", hooked=False)
    assert _drained() == ([], [])

    _commit(repo, "seen.md")

    assert _drained() == (["seen.md", "unseen.md"], [])


# --- what does not reach the pinned branch is not -------------------------------


def test_commits_on_another_branch_are_not_spooled(repo):
    _install(repo)
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "feature.md")
    assert _drained() == ([], [])


def test_commit_in_a_linked_worktree_is_not_spooled(repo, tmp_path):
    """Worktrees share the hooks directory, so their commits fire our hooks."""
    _install(repo)
    worktree = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", str(worktree), "-b", "wt-branch")
    _commit(worktree, "in-worktree.md")

    assert _drained() == ([], [])


def test_merging_a_worktree_branch_reports_its_files(repo, tmp_path):
    _install(repo)
    worktree = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", str(worktree), "-b", "wt-branch")
    _commit(worktree, "in-worktree.md")

    _git(repo, "merge", "-q", "--ff-only", "wt-branch")

    assert _drained() == (["in-worktree.md"], [])


def test_branch_moved_while_checked_out_elsewhere_is_caught_up_on_return(repo):
    """Indexing reads the working tree, so the move waits for main to be checked out."""
    _install(repo)
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "feature.md")
    _git(repo, "branch", "-f", "main", "feature")
    assert _drained() == ([], [])

    _git(repo, "checkout", "-q", "main")

    assert _drained() == (["feature.md"], [])


def test_file_checkout_queues_nothing(repo):
    _install(repo)
    (repo / "base.md").write_text("dirty\n")
    _git(repo, "checkout", "--", "base.md")
    assert _drained() == ([], [])


def test_entries_are_pinned_to_the_branch(repo):
    import json

    _install(repo)
    _commit(repo, "new.md")
    (entry,) = spool.corpus_spool_dir("Docs").rglob("*.json")
    assert json.loads(entry.read_text())["branch"] == "main"


def test_checking_the_branch_out_again_requeues_parked_entries(repo):
    """Parked entries sit outside the watched queue, so the hook must bring them back."""
    _install(repo)
    _commit(repo, "new.md")
    _git(repo, "checkout", "-q", "-b", "feature")
    assert not spool.drain_batch("Docs"), "off the branch, the entry is parked"
    assert list(spool.queue_dir("Docs").iterdir()) == []

    _git(repo, "checkout", "-q", "main")

    # In the queue before any drain runs: that is what wakes the watcher.
    assert len(list(spool.queue_dir("Docs").iterdir())) == 1
    assert _drained() == (["new.md"], [])


# --- the gate itself -----------------------------------------------------------


def test_hook_targets_the_installed_repo_even_with_git_dir_exported(repo, tmp_path):
    """git exports GIT_DIR to some hooks; it must not redirect our commands."""
    other = tmp_path / "other"
    other.mkdir()
    _quiet_git(other, "init", "-q", "-b", "main")
    hooks.install("Docs", repo, "post-commit", spawn=False)
    _commit(repo, "new.md", hooked=False)

    env = _env()
    env["GIT_DIR"] = str(other / ".git")
    result = subprocess.run(
        ["/bin/sh", str(repo / ".git" / "hooks" / "post-commit")],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert _drained() == (["new.md"], [])


def test_reference_transaction_acts_only_once_committed(repo):
    hooks.install("Docs", repo, "reference-transaction", spawn=False)
    _commit(repo, "new.md", hooked=False)
    script = repo / ".git" / "hooks" / "reference-transaction"

    for state in ("preparing", "prepared", "aborted"):
        subprocess.run(["/bin/sh", str(script), state], cwd=repo, env=_env(), input="")
    assert _drained() == ([], [])

    subprocess.run(["/bin/sh", str(script), "committed"], cwd=repo, env=_env(), input="")
    assert _drained() == (["new.md"], [])


def test_hooks_never_break_the_git_command_they_run_under(repo):
    for hook in hooks.SUPPORTED_HOOKS:
        hooks.install("Docs", repo, hook, spawn=False)

    env = dict(os.environ)
    env["PATH"] = "/nonexistent"
    env["HOME"] = str(repo)
    for hook in hooks.SUPPORTED_HOOKS:
        result = subprocess.run(
            ["/bin/sh", str(repo / ".git" / "hooks" / hook), "committed", "x", "1"],
            cwd=repo,
            capture_output=True,
            text=True,
            env=env,
            input="",
        )
        assert result.returncode == 0, f"{hook}: {result.stderr}"


@pytest.mark.parametrize("hook", hooks.SUPPORTED_HOOKS)
@pytest.mark.parametrize("spawn", [True, False])
def test_every_generated_hook_is_valid_shell(hook, spawn, tmp_path):
    body = hooks.render_block("Docs", hook, tmp_path, spawn=spawn, branch="main")
    script = tmp_path / f"{hook}.sh"
    script.write_text(f"#!/bin/sh\n{body}")
    check = subprocess.run(["/bin/sh", "-n", str(script)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_a_quote_in_the_branch_name_cannot_break_the_script(tmp_path):
    body = hooks.render_block("Docs", "post-commit", tmp_path, spawn=False, branch="it's")
    script = tmp_path / "hook.sh"
    script.write_text(f"#!/bin/sh\n{body}")
    check = subprocess.run(["/bin/sh", "-n", str(script)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


# --- catching up -----------------------------------------------------------------


def test_catch_up_without_a_mark_spools_the_tip_commit(repo):
    _commit(repo, "tip.md", hooked=False)
    changes = hooks.catch_up("Docs", repo, "main")
    assert [Path(p).name for p in changes.changed] == ["tip.md"]
    assert spool.read_mark("Docs", repo, "main") == _head(repo)


def test_catch_up_with_an_unknown_mark_falls_back_to_the_tip_commit(repo):
    """A mark can name a commit that gc has since dropped."""
    spool.write_mark("Docs", repo, "main", "0" * 40)
    _commit(repo, "tip.md", hooked=False)
    changes = hooks.catch_up("Docs", repo, "main")
    assert [Path(p).name for p in changes.changed] == ["tip.md"]


def test_catch_up_off_the_branch_spools_nothing_and_keeps_the_mark(repo):
    spool.write_mark("Docs", repo, "main", _head(repo))
    _quiet_git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "feature.md", hooked=False)
    _quiet_git(repo, "branch", "-f", "main", "feature")

    assert hooks.catch_up("Docs", repo, "main") is None
    assert spool.read_mark("Docs", repo, "main") == _head(repo, "main~1")


# --- installing ------------------------------------------------------------------


def test_default_branch_follows_the_remote_head(repo):
    _quiet_git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/trunk")
    assert hooks.default_branch(repo) == "trunk"


def test_default_branch_falls_back_to_the_current_branch(repo):
    _quiet_git(repo, "checkout", "-q", "-b", "develop")
    assert hooks.default_branch(repo) == "develop"


def test_default_branch_on_a_detached_head_without_a_remote_errors(repo):
    from arcaneum.cli.errors import InvalidArgumentError

    _quiet_git(repo, "checkout", "-q", "--detach")
    with pytest.raises(InvalidArgumentError, match="--branch"):
        hooks.default_branch(repo)


def test_default_hooks_prefer_reference_transaction(repo):
    assert hooks.default_hooks(repo) == ["reference-transaction", "post-checkout", "post-rewrite"]


def test_default_hooks_without_reference_transaction_cover_every_post_hook(repo):
    with patch("arcaneum.cli.hooks.git_version", return_value=(2, 27, 0)):
        assert hooks.default_hooks(repo) == [
            "post-commit",
            "post-merge",
            "post-checkout",
            "post-rewrite",
        ]


def test_install_records_the_tip_so_existing_history_is_not_spooled(repo):
    _install(repo)
    assert spool.read_mark("Docs", repo, "main") == _head(repo)
    assert not spool.has_pending("Docs")


def test_reinstall_keeps_the_mark_so_pending_changes_are_not_lost(repo):
    _install(repo)
    mark = spool.read_mark("Docs", repo, "main")
    _commit(repo, "unseen.md", hooked=False)

    _install(repo)

    assert spool.read_mark("Docs", repo, "main") == mark


def test_install_writes_the_branch_into_the_block(repo):
    hooks.install("Docs", repo, "post-commit", spawn=False, branch="release/1.0")
    assert "_arc_branch='release/1.0'" in (repo / ".git" / "hooks" / "post-commit").read_text()


def test_install_refreshes_the_corpus_blocks_at_other_hook_points(repo):
    """One corpus, one repo, one branch: blocks must never disagree."""
    hooks.install("Docs", repo, "post-commit", spawn=False, branch="main")
    hooks.install("Code", repo, "post-commit", spawn=False, branch="main")
    hooks.install("Docs", repo, "reference-transaction", spawn=False, branch="dev")

    installed = {(h.corpus, h.hook): h.branch for h in hooks.list_installed(repo)}
    assert installed[("Docs", "post-commit")] == "dev"
    assert installed[("Docs", "reference-transaction")] == "dev"
    assert installed[("Code", "post-commit")] == "main", "other corpora are untouched"


def test_uninstall_forgets_the_mark(repo):
    _install(repo)
    hooks.uninstall("Docs", repo)
    assert spool.read_mark("Docs", repo, "main") is None


def test_partial_uninstall_keeps_the_mark(repo):
    _install(repo)
    hooks.uninstall("Docs", repo, "post-checkout")
    assert spool.read_mark("Docs", repo, "main") == _head(repo)


# --- the CLI -----------------------------------------------------------------------


def _run(*args):
    return CliRunner().invoke(cli, list(args), catch_exceptions=False)


def test_cli_install_defaults_to_the_default_hook_set(repo, monkeypatch):
    monkeypatch.chdir(repo)
    result = _run("corpus", "hook", "install", "Docs", "--no-spawn")

    assert result.exit_code == 0, result.output
    for hook in hooks.default_hooks(repo):
        assert (repo / ".git" / "hooks" / hook).exists()
    assert "main" in result.output


def test_cli_install_takes_a_branch(repo, monkeypatch):
    monkeypatch.chdir(repo)
    result = _run("corpus", "hook", "install", "Docs", "--branch", "dev", "--no-spawn")

    assert result.exit_code == 0, result.output
    assert {h.branch for h in hooks.list_installed(repo)} == {"dev"}


def test_cli_install_takes_several_hook_points(repo, monkeypatch):
    monkeypatch.chdir(repo)
    result = _run(
        "corpus", "hook", "install", "Docs", "--hook", "post-commit", "--hook", "post-merge"
    )

    assert result.exit_code == 0, result.output
    assert sorted(h.hook for h in hooks.list_installed(repo)) == ["post-commit", "post-merge"]


def test_cli_status_reports_the_pinned_branch(repo, monkeypatch):
    import json

    monkeypatch.chdir(repo)
    _run("corpus", "hook", "install", "Docs", "--branch", "dev", "--no-spawn")

    assert "dev" in _run("corpus", "hook", "status").output
    payload = json.loads(_run("corpus", "hook", "status", "--json").output)
    assert {entry["branch"] for entry in payload["data"]["installed"]} == {"dev"}


def test_cli_spool_with_a_branch_catches_up_from_the_mark(repo, monkeypatch):
    monkeypatch.chdir(repo)
    spool.write_mark("Docs", repo, "main", _head(repo))
    _commit(repo, "one.md", hooked=False)
    _commit(repo, "two.md", hooked=False)

    result = _run("corpus", "hook", "spool", "Docs", "--repo", str(repo), "--branch", "main")

    assert result.exit_code == 0, result.output
    assert _drained() == (["one.md", "two.md"], [])

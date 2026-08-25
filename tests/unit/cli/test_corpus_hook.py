"""`arc corpus hook install/uninstall/status` (kata vq0n)."""

from __future__ import annotations

import os
import platform
import stat
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from arcaneum.cli import hooks
from arcaneum.cli.main import cli


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="git hooks are shell scripts; not a supported target",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / "a.py").write_text("a\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(root)
    return root


def _run(*args):
    return CliRunner().invoke(cli, list(args), catch_exceptions=False)


def _hook(repo: Path, name: str = "post-commit") -> Path:
    return repo / ".git" / "hooks" / name


# --- install ------------------------------------------------------------------


def test_install_writes_an_executable_post_commit_hook(repo):
    result = _run("corpus", "hook", "install", "Docs")
    assert result.exit_code == 0, result.output

    hook = _hook(repo)
    assert hook.exists()
    assert hook.stat().st_mode & stat.S_IXUSR, "hook must be executable or git ignores it"
    assert "Docs" in hook.read_text()


def test_installed_hook_names_the_corpus_and_repo(repo):
    _run("corpus", "hook", "install", "Docs")
    body = _hook(repo).read_text()
    assert "Docs" in body
    # The repo root is recorded so one repo can feed several corpora and
    # uninstall removes only its own block.
    assert str(repo.resolve()) in body or "rev-parse" in body


def test_install_is_idempotent(repo):
    _run("corpus", "hook", "install", "Docs")
    first = _hook(repo).read_text()
    result = _run("corpus", "hook", "install", "Docs")
    assert result.exit_code == 0, result.output
    assert _hook(repo).read_text() == first
    assert first.count(hooks.BLOCK_START.format(corpus="Docs")) == 1


def test_install_preserves_an_existing_user_hook(repo):
    hook = _hook(repo)
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/sh\necho 'my own hook'\n")
    hook.chmod(0o755)

    _run("corpus", "hook", "install", "Docs")

    body = hook.read_text()
    assert "echo 'my own hook'" in body
    assert "Docs" in body


def test_two_corpora_coexist_in_one_hook(repo):
    _run("corpus", "hook", "install", "Docs")
    _run("corpus", "hook", "install", "Code")

    body = _hook(repo).read_text()
    assert hooks.BLOCK_START.format(corpus="Docs") in body
    assert hooks.BLOCK_START.format(corpus="Code") in body


def test_install_accepts_other_hook_points(repo):
    result = _run("corpus", "hook", "install", "Docs", "--hook", "post-merge")
    assert result.exit_code == 0, result.output
    assert _hook(repo, "post-merge").exists()
    assert not _hook(repo, "post-commit").exists()


def test_install_rejects_an_unsupported_hook_point(repo):
    result = CliRunner().invoke(cli, ["corpus", "hook", "install", "Docs", "--hook", "pre-commit"])
    assert result.exit_code != 0


def test_install_outside_a_repo_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["corpus", "hook", "install", "Docs"])
    assert result.exit_code != 0


def test_install_respects_core_hookspath(repo):
    custom = repo / "my-hooks"
    custom.mkdir()
    _git(repo, "config", "core.hooksPath", "my-hooks")

    result = _run("corpus", "hook", "install", "Docs")
    assert result.exit_code == 0, result.output
    assert (custom / "post-commit").exists()
    assert not _hook(repo).exists()


# --- uninstall ----------------------------------------------------------------


def test_uninstall_removes_only_its_own_block(repo):
    hook = _hook(repo)
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/sh\necho 'my own hook'\n")
    hook.chmod(0o755)

    _run("corpus", "hook", "install", "Docs")
    _run("corpus", "hook", "install", "Code")
    result = _run("corpus", "hook", "uninstall", "Docs")

    assert result.exit_code == 0, result.output
    body = hook.read_text()
    assert "echo 'my own hook'" in body, "pre-existing user hook must survive"
    assert hooks.BLOCK_START.format(corpus="Docs") not in body
    assert hooks.BLOCK_START.format(corpus="Code") in body, "other corpora untouched"


def test_uninstalling_the_only_block_leaves_no_arcaneum_content(repo):
    _run("corpus", "hook", "install", "Docs")
    _run("corpus", "hook", "uninstall", "Docs")

    hook = _hook(repo)
    if hook.exists():
        assert "arcaneum" not in hook.read_text().lower()


def test_uninstall_when_not_installed_is_not_an_error(repo):
    result = _run("corpus", "hook", "uninstall", "Docs")
    assert result.exit_code == 0, result.output


# --- status -------------------------------------------------------------------


def test_status_reports_an_installed_hook(repo):
    _run("corpus", "hook", "install", "Docs")
    result = _run("corpus", "hook", "status")
    assert result.exit_code == 0, result.output
    assert "Docs" in result.output
    assert "post-commit" in result.output


def test_status_reports_nothing_installed(repo):
    result = _run("corpus", "hook", "status")
    assert result.exit_code == 0, result.output
    assert "Docs" not in result.output


def test_status_json_is_machine_readable(repo):
    import json

    _run("corpus", "hook", "install", "Docs")
    result = _run("corpus", "hook", "status", "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    installed = payload["data"]["installed"]
    assert any(entry["corpus"] == "Docs" for entry in installed)


# --- the generated script -----------------------------------------------------


def test_generated_hook_is_valid_shell(repo):
    _run("corpus", "hook", "install", "Docs")
    check = subprocess.run(["sh", "-n", str(_hook(repo))], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_hook_exits_zero_even_when_arc_is_missing(repo):
    """The hook must never make a git command fail (acceptance criterion)."""
    _run("corpus", "hook", "install", "Docs")

    env = dict(os.environ)
    env["PATH"] = "/nonexistent"  # `arc` cannot be found
    env["HOME"] = str(repo)
    # Absolute sh, since PATH above hides the shell itself too.
    result = subprocess.run(
        ["/bin/sh", str(_hook(repo))], cwd=repo, capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr


def test_committing_with_the_hook_installed_succeeds_and_spools(repo):
    """End to end: a real commit runs the hook, which queues the touched file."""
    _run("corpus", "hook", "install", "Docs", "--no-spawn")

    (repo / "b.py").write_text("b\n")
    _git(repo, "add", "-A")
    commit = subprocess.run(
        ["git", "commit", "-m", "add b"], cwd=repo, capture_output=True, text=True,
        env={**os.environ, "XDG_DATA_HOME": str(repo.parent / "xdg")},
    )
    assert commit.returncode == 0, commit.stderr

    from arcaneum.cli import spool

    pending = list(spool.corpus_spool_dir("Docs").rglob("*.json"))
    assert pending, "the hook should have spooled the touched file"
    batch = spool.drain_batch("Docs")
    assert [Path(p).name for p in batch.changed] == ["b.py"]

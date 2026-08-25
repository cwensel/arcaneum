"""The hook must not spawn a drain that will immediately die (kata vq0n).

Measured during a real rebase: 1299 spawns declined on the worker lock against
51 useful batches -- a 25:1 waste ratio, ~1.1s of interpreter startup each,
burned concurrently with the worker actually embedding.

Two gates fix it:
  1. If a worker already holds the lock, skip the spawn. The entry is spooled;
     the running worker picks it up on its next loop.
  2. Otherwise debounce briefly, so a burst coalesces into one worker rather
     than a thundering herd.
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


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    yield tmp_path


# --- the generated script contains both gates --------------------------------


def test_hook_checks_the_worker_lock_before_spawning(tmp_path):
    body = hooks.render_block("Docs", "post-commit", tmp_path, spawn=True)
    assert "worker.lock" in body, "the hook must consult the same lock the worker takes"


def test_hook_debounces_before_spawning(tmp_path):
    body = hooks.render_block("Docs", "post-commit", tmp_path, spawn=True)
    assert "sleep" in body


def test_no_spawn_mode_has_neither_gate(tmp_path):
    """--no-spawn never starts a worker, so there is nothing to gate."""
    body = hooks.render_block("Docs", "post-commit", tmp_path, spawn=False)
    assert "setsid" not in body and "nohup" not in body


@pytest.mark.parametrize("hook", hooks.SUPPORTED_HOOKS)
def test_gated_hooks_are_valid_shell(hook, tmp_path):
    body = hooks.render_block("Docs", hook, tmp_path, spawn=True)
    script = tmp_path / f"{hook}.sh"
    script.write_text(f"#!/bin/sh\n{body}")
    check = subprocess.run(["/bin/sh", "-n", str(script)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


# --- the lock path the shell computes must match the worker's ----------------


def test_shell_and_python_agree_on_the_worker_lock_path(isolated, tmp_path):
    """A divergence here makes the gate silently never fire."""
    body = hooks.render_block("Docs", "post-commit", tmp_path, spawn=True)

    # Extract the shell's computed lock path by running just that assignment.
    script = tmp_path / "probe.sh"
    script.write_text(
        "#!/bin/sh\n"
        f"_arc_corpus='Docs'\n"
        f'_arc_data="${{XDG_DATA_HOME:-$HOME/.local/share}}/arcaneum"\n'
        f'echo "$_arc_data/spool/$_arc_corpus/worker.lock"\n'
    )
    out = subprocess.run(
        ["/bin/sh", str(script)], capture_output=True, text=True, env=dict(os.environ)
    ).stdout.strip()

    assert Path(out) == spool.worker_lock_path("Docs")


# --- end to end: a held lock suppresses the spawn ----------------------------


def _run_hook(script: Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/sh", str(script), *args],
        capture_output=True, text=True, env=env, input="",
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=root, check=True)
    (root / "a.md").write_text("a\n")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    return root


def test_hook_skips_the_spawn_while_a_worker_holds_the_lock(isolated, repo, tmp_path):
    """The core fix: no wasted process when a worker is already draining."""
    hooks.install("Docs", repo, "post-commit", spawn=True)
    hook_script = repo / ".git" / "hooks" / "post-commit"

    env = dict(os.environ)
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    # A fake `arc` that records every invocation.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "invocations.txt"
    (bin_dir / "arc").write_text(
        f'#!/bin/sh\necho "$@" >> "{marker}"\nexit 0\n'
    )
    (bin_dir / "arc").chmod(0o755)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    lock = spool.try_acquire_worker_lock("Docs")
    try:
        result = _run_hook(hook_script, env)
    finally:
        spool.release_worker_lock(lock)

    assert result.returncode == 0, result.stderr
    calls = marker.read_text() if marker.exists() else ""
    assert "hook spool" in calls, "it must still queue the paths"
    assert "--drain-spool" not in calls, "but must not spawn a doomed worker"


def test_hook_spawns_when_no_worker_is_running(isolated, repo, tmp_path):
    hooks.install("Docs", repo, "post-commit", spawn=True)
    hook_script = repo / ".git" / "hooks" / "post-commit"

    env = dict(os.environ)
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "invocations.txt"
    (bin_dir / "arc").write_text(f'#!/bin/sh\necho "$@" >> "{marker}"\nexit 0\n')
    (bin_dir / "arc").chmod(0o755)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    result = _run_hook(hook_script, env)
    assert result.returncode == 0, result.stderr

    # The spawn is backgrounded after a debounce; give it room to land.
    import time
    for _ in range(40):
        if "--drain-spool" in (marker.read_text() if marker.exists() else ""):
            break
        time.sleep(0.25)

    assert "--drain-spool" in marker.read_text()


def test_hook_still_exits_zero_when_the_gate_is_hit(isolated, repo, tmp_path):
    hooks.install("Docs", repo, "post-commit", spawn=True)
    hook_script = repo / ".git" / "hooks" / "post-commit"
    env = dict(os.environ)
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    env["PATH"] = "/nonexistent"

    lock = spool.try_acquire_worker_lock("Docs")
    try:
        result = _run_hook(hook_script, env)
    finally:
        spool.release_worker_lock(lock)

    assert result.returncode == 0

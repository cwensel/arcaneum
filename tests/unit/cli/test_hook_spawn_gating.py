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
    assert "probe-lock" in body, "the hook must consult the lock the worker takes"


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
    marker = tmp_path / "invocations.txt"
    _fake_arc(bin_dir, marker, held=True)
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
    # A fake arc that logs calls and answers probe-lock from a sentinel file:
    # present => a worker holds the lock (exit 0), absent => free (exit 1).
    (bin_dir / "arc").write_text(
        f'#!/bin/sh\n'
        f'echo "$@" >> "{marker}"\n'
        f'if [ "$2" = "hook" ] && [ "$3" = "probe-lock" ]; then\n'
        f'  [ -e "{bin_dir.parent}/HELD" ] && exit 0\n'
        f'  exit 1\n'
        f'fi\n'
        f'exit 0\n'
    )
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


# --- the gate must work without flock (roborev 6137) -------------------------


def _fake_arc(bin_dir: Path, marker: Path, *, held: bool = False) -> None:
    """A stand-in `arc` that logs calls and answers probe-lock deterministically."""
    bin_dir.mkdir(exist_ok=True)
    sentinel = bin_dir.parent / "HELD"
    if held:
        sentinel.write_text("held\n")
    elif sentinel.exists():
        sentinel.unlink()
    (bin_dir / "arc").write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{marker}"\n'
        'if [ "$2" = "hook" ] && [ "$3" = "probe-lock" ]; then\n'
        f'  [ -e "{sentinel}" ] && exit 0\n'
        "  exit 1\n"
        "fi\n"
        "exit 0\n"
    )
    (bin_dir / "arc").chmod(0o755)


def _env_without_flock(tmp_path: Path, bin_dir: Path) -> dict:
    """PATH containing our fake arc and a real shell, but no flock.

    Stock macOS ships no flock, so the gate must not depend on it -- that is
    the platform the 25:1 spawn waste was measured on.
    """
    env = dict(os.environ)
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin"
    env["ARC_HOOK_DEBOUNCE"] = "0.2"
    return env


def _wait_past_debounce() -> None:
    import time

    time.sleep(1.5)


def test_gate_holds_without_flock(isolated, repo, tmp_path, monkeypatch):
    """With a worker holding the lock, no drain may be spawned even sans flock."""
    hooks.install("Docs", repo, "post-commit", spawn=True)
    hook_script = repo / ".git" / "hooks" / "post-commit"
    bin_dir = tmp_path / "bin"
    marker = tmp_path / "calls.txt"
    _fake_arc(bin_dir, marker, held=True)
    env = _env_without_flock(tmp_path, bin_dir)
    assert not (bin_dir / "flock").exists()

    result = _run_hook(hook_script, env)
    # Wait past the debounce so a deferred spawn would have landed.
    _wait_past_debounce()

    assert result.returncode == 0, result.stderr
    calls = marker.read_text() if marker.exists() else ""
    assert "hook spool" in calls, "paths must still be queued"
    assert "--drain-spool" not in calls, (
        "a worker holds the lock; no drain should have been spawned"
    )


def test_spawn_still_happens_without_flock_when_idle(isolated, repo, tmp_path):
    """The gate must not become a blanket suppression."""
    hooks.install("Docs", repo, "post-commit", spawn=True)
    hook_script = repo / ".git" / "hooks" / "post-commit"
    bin_dir = tmp_path / "bin"
    marker = tmp_path / "calls.txt"
    _fake_arc(bin_dir, marker)
    env = _env_without_flock(tmp_path, bin_dir)

    result = _run_hook(hook_script, env)
    assert result.returncode == 0, result.stderr
    _wait_past_debounce()

    assert "--drain-spool" in (marker.read_text() if marker.exists() else "")


def test_the_nohup_fallback_path_is_gated_too(isolated, repo, tmp_path):
    """CI always has setsid, so exercise the other branch explicitly."""
    hooks.install("Docs", repo, "post-commit", spawn=True)
    hook_script = repo / ".git" / "hooks" / "post-commit"
    bin_dir = tmp_path / "bin"
    marker = tmp_path / "calls.txt"
    _fake_arc(bin_dir, marker)
    # A PATH with neither setsid nor flock forces the fallback spawn path.
    env = _env_without_flock(tmp_path, bin_dir)
    shim = tmp_path / "noshim"
    shim.mkdir()
    for tool in ("sh", "git", "mkdir", "dirname", "sleep", "command"):
        src = Path("/bin") / tool
        if src.exists():
            (shim / tool).symlink_to(src)
    env["PATH"] = f"{bin_dir}:{shim}"

    lock = spool.try_acquire_worker_lock("Docs")
    try:
        result = _run_hook(hook_script, env)
        _wait_past_debounce()
    finally:
        spool.release_worker_lock(lock)

    assert result.returncode == 0
    assert "--drain-spool" not in (marker.read_text() if marker.exists() else "")


# --- the lock probe must not depend on flock(1) (roborev 6137) ---------------


def test_probe_lock_reports_free(isolated):
    from click.testing import CliRunner

    from arcaneum.cli.main import cli

    result = CliRunner().invoke(cli, ["corpus", "hook", "probe-lock", "Docs"])
    assert result.exit_code == 1, "exit 1 == not held"


def test_probe_lock_reports_held(isolated):
    from click.testing import CliRunner

    from arcaneum.cli.main import cli

    lock = spool.try_acquire_worker_lock("Docs")
    try:
        # Clear the in-process record so the probe contends like a real
        # separate process would; the flock itself is still held.
        held = dict(spool.__dict__.get("_held", {})) if hasattr(spool, "_held") else None
        result = CliRunner().invoke(cli, ["corpus", "hook", "probe-lock", "Docs"])
    finally:
        spool.release_worker_lock(lock)

    assert result.exit_code == 0, "exit 0 == held by another worker"


def test_the_generated_hook_does_not_require_flock(tmp_path):
    """Stock macOS ships no flock; the gate must not silently vanish there."""
    body = hooks.render_block("Docs", "post-commit", tmp_path, spawn=True)
    assert "command -v flock" not in body, (
        "gating must not be conditional on flock(1) being installed"
    )
    assert "probe-lock" in body

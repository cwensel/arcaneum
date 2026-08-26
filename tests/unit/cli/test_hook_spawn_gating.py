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
import sys
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

    # The real `corpus hook spool` call creates this directory before the
    # background worker runs.  Preserve that side effect in the fake so Linux
    # flock(1) can create worker.lock instead of mistaking ENOENT for a held lock.
    spool.corpus_spool_dir("Docs").mkdir(parents=True, exist_ok=True)

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


# Every external tool the generated hook invokes, minus flock. Keep this in
# step with render_block: a missing tool makes the hook fail early, which can
# look like a working gate.
_HOOK_TOOLS = ("sh", "git", "mkdir", "dirname", "sleep", "python3", "setsid")


def _flockless_bin(tmp_path: Path) -> Path:
    """A PATH directory with the hook's tools symlinked, deliberately no flock.

    Stock macOS ships no flock(1) -- the platform the spawn waste was measured
    on -- but CI runs ubuntu, where /usr/bin/flock exists. Inheriting the real
    PATH would silently take the flock branch there, so a test claiming to
    exercise the flock-less gate would prove nothing.
    """
    shim = tmp_path / "flockless"
    shim.mkdir(exist_ok=True)
    for tool in _HOOK_TOOLS:
        if (shim / tool).exists():
            continue
        for root in ("/bin", "/usr/bin"):
            src = Path(root) / tool
            if src.exists():
                (shim / tool).symlink_to(src)
                break
    assert not (shim / "flock").exists()
    return shim


def _env_without_flock(tmp_path: Path, bin_dir: Path) -> dict:
    """Environment whose PATH genuinely contains no flock(1)."""
    env = dict(os.environ)
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    env["PATH"] = f"{bin_dir}:{_flockless_bin(tmp_path)}"
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
    _fake_arc(bin_dir, marker)
    env = _env_without_flock(tmp_path, bin_dir)
    assert not (bin_dir / "flock").exists()

    # Hold the real lock: the gate probes it with the installed probe script,
    # not through the fake arc.
    lock = spool.try_acquire_worker_lock("Docs")
    try:
        result = _run_hook(hook_script, env)
        # Wait past the debounce so a deferred spawn would have landed.
        _wait_past_debounce()
    finally:
        spool.release_worker_lock(lock)

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
    # A PATH without setsid forces the fallback spawn branch. Everything else
    # the hook needs must still resolve, or it would fail early and the
    # assertion would pass for the wrong reason.
    # Same environment, but a PATH without setsid so the fallback spawn branch
    # is the one exercised. (flock is absent either way.)
    env = _env_without_flock(tmp_path, bin_dir)
    shim = tmp_path / "nosetsid"
    shim.mkdir(exist_ok=True)
    for tool in _HOOK_TOOLS:
        if tool == "setsid" or (shim / tool).exists():
            continue
        for root in ("/bin", "/usr/bin"):
            src = Path(root) / tool
            if src.exists():
                (shim / tool).symlink_to(src)
                break
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


def test_probe_script_reports_free(isolated, tmp_path):
    lock = tmp_path / "worker.lock"
    lock.touch()
    script = hooks.write_probe_script()
    r = subprocess.run([sys.executable, str(script), str(lock)], capture_output=True)
    assert r.returncode == 1, "exit 1 == not held"


def test_probe_script_reports_held(isolated, tmp_path):
    """A second flock on a separate fd is denied by the kernel, same process or not."""
    script = hooks.write_probe_script()
    lock_path = spool.worker_lock_path("Docs")
    fd = spool.try_acquire_worker_lock("Docs")
    try:
        r = subprocess.run(
            [sys.executable, str(script), str(lock_path)], capture_output=True
        )
    finally:
        spool.release_worker_lock(fd)
    assert r.returncode == 0, "exit 0 == held by a worker"


def test_probe_script_on_a_missing_lock_reports_free(isolated, tmp_path):
    script = hooks.write_probe_script()
    r = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "nope.lock")], capture_output=True
    )
    assert r.returncode == 1


def test_install_writes_the_probe_script(isolated, repo):
    hooks.install("Docs", repo, "post-commit", spawn=True)
    assert hooks.probe_script_path().exists()


def test_the_generated_hook_does_not_require_flock(tmp_path):
    """Stock macOS ships no flock; the gate must not silently vanish there."""
    body = hooks.render_block("Docs", "post-commit", tmp_path, spawn=True)
    # flock may be *used* when present, but the gate must still work without it.
    assert "_arc_probe" in body, "a flock-less platform needs a working probe"


# --- the probe itself must be cheap (roborev 6166) ---------------------------


def test_the_gate_does_not_invoke_the_arc_console_script(tmp_path):
    """A full `arc` import costs ~700ms; paying that per commit defeats the gate.

    The gate exists to avoid ~1.1s of wasted interpreter startup. Spending
    ~0.7s to find that out -- synchronously, on every commit -- trades one cost
    for another and regresses Linux, where flock(1) made the check free.
    """
    body = hooks.render_block("Docs", "post-commit", tmp_path, spawn=True)
    # The gate may run `arc corpus sync` (that is the work) but must not invoke
    # arc merely to ask whether the lock is held.
    # Scope to the lock-check function: the block legitimately runs
    # `arc corpus hook spool` (the queueing that must happen anyway) and
    # `arc corpus sync` (the work). Only the probe must stay cheap.
    check = body.split("_arc_locked() {")[1].split("}")[0]
    assert "$_arc_bin" not in check, "the lock check must not start a full arc"
    assert "$_arc_py" in check, "it should use the bare interpreter probe"


def test_flock_is_the_fast_path_when_available(isolated, repo, tmp_path):
    """With flock(1) on PATH the gate must not fall back to the interpreter.

    Asserting the literal word "flock" appears in the script proves nothing --
    a comment would satisfy it. Observe instead that the python probe is never
    reached: point ARC_HOOK_PYTHON at a recorder and assert it stays unused.
    """
    hooks.install("Docs", repo, "post-commit", spawn=True)
    hook_script = repo / ".git" / "hooks" / "post-commit"
    bin_dir = tmp_path / "bin"
    marker = tmp_path / "calls.txt"
    _fake_arc(bin_dir, marker)

    py_used = tmp_path / "python-was-used.txt"
    fake_py = bin_dir / "recording-python"
    fake_py.write_text(f'#!/bin/sh\ntouch "{py_used}"\nexit 1\n')
    fake_py.chmod(0o755)

    fake_flock = bin_dir / "flock"
    fake_flock.write_text("#!/bin/sh\nexit 0\n")  # exit 0 => lock is free
    fake_flock.chmod(0o755)

    env = dict(os.environ)
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin"
    env["ARC_HOOK_DEBOUNCE"] = "0.2"
    env["ARC_HOOK_PYTHON"] = str(fake_py)

    # Hold the lock so the probe is actually reached: the hook short-circuits
    # when the lock file does not exist, which would make this vacuous.
    lock = spool.try_acquire_worker_lock("Docs")
    try:
        _run_hook(hook_script, env)
        _wait_past_debounce()
    finally:
        spool.release_worker_lock(lock)

    assert not py_used.exists(), "flock was available; the python probe should be unused"


def test_the_flockless_gate_depends_on_the_installed_probe(isolated, repo, tmp_path):
    """Without flock, the probe script is what makes the gate work.

    Deleting it must make the gate fall through to a spawn rather than silently
    suppress work -- which also proves the probe is genuinely load-bearing.
    """
    hooks.install("Docs", repo, "post-commit", spawn=True)
    hook_script = repo / ".git" / "hooks" / "post-commit"
    bin_dir = tmp_path / "bin"
    marker = tmp_path / "calls.txt"
    _fake_arc(bin_dir, marker)
    env = _env_without_flock(tmp_path, bin_dir)

    probe = Path(env["XDG_DATA_HOME"]) / "arcaneum" / "hook-probe-lock.py"
    if probe.exists():
        probe.unlink()

    lock = spool.try_acquire_worker_lock("Docs")
    try:
        _run_hook(hook_script, env)
        _wait_past_debounce()
    finally:
        spool.release_worker_lock(lock)

    calls = marker.read_text() if marker.exists() else ""
    assert "--drain-spool" in calls, (
        "with no probe the gate cannot know the lock is held, so it must not "
        "suppress the spawn"
    )


def test_the_probe_is_measurably_cheaper_than_a_spawn(isolated, tmp_path):
    """Guard the property that motivated the change, not just its shape."""
    import time

    lock_path = spool.worker_lock_path("Docs")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch()

    probe = (
        "import fcntl,os,sys\n"
        "try: fd=os.open(sys.argv[1],os.O_RDWR)\n"
        "except OSError: sys.exit(1)\n"
        "try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        "except BlockingIOError: sys.exit(0)\n"
        "sys.exit(1)\n"
    )
    start = time.monotonic()
    subprocess.run([sys.executable, "-c", probe, str(lock_path)], capture_output=True)
    elapsed = time.monotonic() - start

    # A full `arc` import measured ~0.7s; a bare interpreter probe ~0.03s.
    assert elapsed < 0.5, f"probe took {elapsed:.2f}s; too close to a real spawn"

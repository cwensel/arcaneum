"""The drain worker must leave a durable record (follow-up to kata vq0n).

A drain that succeeds leaves an empty spool; a drain that fails leaves an empty
spool too, once the batch is consumed. Without a per-batch log line there is no
way to tell those apart after the fact, and launchd-driven drains wrote nowhere
at all because the plist captured no output.
"""

from __future__ import annotations

import platform
import plistlib
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from arcaneum.cli import hook_log, spool, spool_service
from arcaneum.cli.main import cli


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="flock-based spool worker is not supported on Windows",
)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("a.py", "b.py"):
        (root / name).write_text("x\n")
    return root


def _run(args, impl):
    with patch("arcaneum.cli.sync.sync_directory_command", impl):
        return CliRunner().invoke(cli, args, catch_exceptions=False)


# --- the shared log path ------------------------------------------------------


def test_log_path_lives_under_the_state_dir(isolated):
    path = hook_log.hook_log_path()
    assert path.name == "hook.log"
    assert "arcaneum" in str(path)


def test_log_path_honors_xdg_state_home(isolated, tmp_path):
    assert str(hook_log.hook_log_path()).startswith(str(tmp_path / "state"))


def test_the_shell_hook_and_python_agree_on_the_log_path(isolated, tmp_path):
    """A divergence here splits the record across two files silently."""
    from arcaneum.cli import hooks

    body = hooks.render_block("Docs", "post-commit", tmp_path, spawn=False)
    # The hook resolves the path in shell; both must land in the same place.
    assert "arcaneum/hook.log" in body


def test_writing_creates_the_parent_directory(isolated):
    hook_log.write("hello")
    assert hook_log.hook_log_path().read_text().strip().endswith("hello")


def test_entries_are_timestamped_and_appended(isolated):
    hook_log.write("first")
    hook_log.write("second")
    lines = hook_log.hook_log_path().read_text().strip().splitlines()
    assert len(lines) == 2
    assert "first" in lines[0] and "second" in lines[1]
    # A record with no time is not much of a record.
    assert lines[0][:4].isdigit()


def test_logging_never_raises_when_the_path_is_unwritable(isolated, monkeypatch):
    """Logging is diagnostics; it must never take down an indexing run."""
    monkeypatch.setattr(hook_log, "hook_log_path", lambda: Path("/nonexistent/x/hook.log"))
    hook_log.write("still fine")  # must not raise


# --- per-batch drain logging --------------------------------------------------


def test_successful_batch_is_logged(isolated, repo):
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])
    _run(["corpus", "sync", "Docs", "--drain-spool"], lambda *a, **k: None)

    body = hook_log.hook_log_path().read_text()
    assert "Docs" in body
    assert "ok" in body.lower()


def test_batch_log_records_the_counts(isolated, repo):
    spool.write_entry(
        "Docs", repo, changed=[str(repo / "a.py"), str(repo / "b.py")], removed=["/gone.py"]
    )
    _run(["corpus", "sync", "Docs", "--drain-spool"], lambda *a, **k: None)

    body = hook_log.hook_log_path().read_text()
    assert "2" in body and "1" in body


def test_a_failing_batch_is_logged_before_it_propagates(isolated, repo):
    """The whole point: a failure must leave a trace, not just an empty spool."""
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])

    def boom(*args, **kwargs):
        raise RuntimeError("embedding exploded")

    with pytest.raises(RuntimeError):
        _run(["corpus", "sync", "Docs", "--drain-spool"], boom)

    body = hook_log.hook_log_path().read_text()
    assert "embedding exploded" in body
    assert "fail" in body.lower() or "error" in body.lower()


def test_a_batch_that_exits_is_logged(isolated, repo):
    """sync_directory_command reports failure via sys.exit, not an exception.

    SystemExit is a BaseException, so an `except Exception` handler misses the
    single most common failure path -- observed live: a real sync failed on a
    missing embedding backend and wrote nothing to the log.
    """
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])

    def bail(*args, **kwargs):
        raise SystemExit(1)

    # CliRunner converts SystemExit into a non-zero result rather than
    # re-raising, so assert on the exit code the user would see.
    result = _run(["corpus", "sync", "Docs", "--drain-spool"], bail)
    assert result.exit_code != 0

    body = hook_log.hook_log_path().read_text()
    assert "fail" in body.lower()
    assert "Docs" in body


def test_a_keyboard_interrupt_mid_batch_is_logged(isolated, repo):
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    result = _run(["corpus", "sync", "Docs", "--drain-spool"], interrupted)
    assert result.exit_code != 0

    assert "fail" in hook_log.hook_log_path().read_text().lower()


def test_declining_because_another_worker_holds_the_lock_is_logged(isolated, repo):
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])
    lock = spool.try_acquire_worker_lock("Docs")
    try:
        _run(["corpus", "sync", "Docs", "--drain-spool"], lambda *a, **k: None)
    finally:
        spool.release_worker_lock(lock)

    assert "already" in hook_log.hook_log_path().read_text().lower()


def test_an_empty_spool_does_not_spam_the_log(isolated):
    """launchd may fire on a spool another worker already drained."""
    _run(["corpus", "sync", "Docs", "--drain-spool"], lambda *a, **k: None)
    path = hook_log.hook_log_path()
    body = path.read_text() if path.exists() else ""
    assert body.strip() == "", "nothing happened; nothing to say"


# --- launchd captures its output ----------------------------------------------


def test_plist_captures_stdout_and_stderr(isolated):
    parsed = plistlib.loads(spool_service.render_launchd_plist("Docs", arc_bin="/bin/arc"))
    assert parsed["StandardOutPath"] == str(hook_log.hook_log_path())
    assert parsed["StandardErrorPath"] == str(hook_log.hook_log_path())


def test_systemd_service_captures_output_too(isolated):
    unit = spool_service.render_systemd_service_unit("Docs", arc_bin="/bin/arc")
    # journald captures a systemd unit's output by default; be explicit so the
    # record lands in the same file a user is told to read.
    assert "hook.log" in unit


# --- a failed batch must not lose the queued work (roborev 6104) --------------


def test_a_failed_batch_leaves_the_work_spooled(isolated, repo):
    """Transient backend failures must not permanently drop queued paths."""
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])

    def boom(*args, **kwargs):
        raise RuntimeError("qdrant unreachable")

    with pytest.raises(RuntimeError):
        _run(["corpus", "sync", "Docs", "--drain-spool"], boom)

    assert spool.has_pending("Docs"), "the batch must survive for the next drain"


def test_the_retained_work_is_indexed_on_the_next_drain(isolated, repo):
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])

    def boom(*args, **kwargs):
        raise RuntimeError("transient")

    with pytest.raises(RuntimeError):
        _run(["corpus", "sync", "Docs", "--drain-spool"], boom)

    seen = []
    _run(["corpus", "sync", "Docs", "--drain-spool"], lambda c, p, *a, **k: seen.append(list(p)))

    assert seen == [[str(repo / "a.py")]]
    assert not spool.has_pending("Docs")


def test_a_batch_that_exits_also_retains_its_work(isolated, repo):
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])

    def bail(*args, **kwargs):
        raise SystemExit(1)

    result = _run(["corpus", "sync", "Docs", "--drain-spool"], bail)
    assert result.exit_code != 0
    assert spool.has_pending("Docs")


def test_a_successful_batch_still_clears_the_spool(isolated, repo):
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])
    _run(["corpus", "sync", "Docs", "--drain-spool"], lambda *a, **k: None)
    assert not spool.has_pending("Docs")

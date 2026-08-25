"""`arc corpus sync` holds the per-corpus write lock (kata htmw)."""

from __future__ import annotations

import platform
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from arcaneum.cli import corpus_lock
from arcaneum.cli.main import cli


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="flock-based corpus lock is a no-op on Windows",
)


@pytest.fixture
def isolated_locks(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    yield tmp_path / "arcaneum" / "locks"


def _run(args, impl):
    """Invoke `arc` with the heavy locked sync body replaced by `impl`."""
    runner = CliRunner()
    with patch("arcaneum.cli.sync._sync_directory_locked", impl):
        return runner.invoke(cli, args, catch_exceptions=False)


def test_sync_holds_the_corpus_lock_while_indexing(isolated_locks, tmp_path):
    """The lock must be held for the duration of the sync body, not just around it."""
    observed = {}

    def impl(corpus, *args, **kwargs):
        holder = corpus_lock.read_lock_holder(corpus_lock.corpus_lock_path(corpus))
        observed["holder"] = holder
        # A --no-wait acquire from "another process" would fail while held; in
        # this process the lock is reentrant, so assert on the holder record.
        return None

    result = _run(["corpus", "sync", "Docs", str(tmp_path)], impl)

    assert result.exit_code == 0, result.output
    assert observed["holder"] is not None
    assert observed["holder"]["corpus"] == "Docs"


def test_lock_released_after_sync_completes(isolated_locks, tmp_path):
    _run(["corpus", "sync", "Docs", str(tmp_path)], lambda *a, **k: None)

    # Would raise CorpusLockUnavailable if the lock had leaked.
    with corpus_lock.acquire_corpus_lock("Docs", wait=False):
        pass


def test_lock_released_when_sync_raises(isolated_locks, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("indexing blew up")

    with pytest.raises(RuntimeError):
        _run(["corpus", "sync", "Docs", str(tmp_path)], boom)

    with corpus_lock.acquire_corpus_lock("Docs", wait=False):
        pass


def test_lock_released_on_sigint(isolated_locks, tmp_path):
    """Ctrl-C during indexing must release the lock (CliRunner reports exit 1)."""

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    result = _run(["corpus", "sync", "Docs", str(tmp_path)], interrupted)
    assert result.exit_code != 0

    with corpus_lock.acquire_corpus_lock("Docs", wait=False):
        pass


def test_lock_released_when_sync_exits(isolated_locks, tmp_path):
    """sys.exit() inside the body must still release the lock."""

    def bail(*args, **kwargs):
        raise SystemExit(1)

    result = _run(["corpus", "sync", "Docs", str(tmp_path)], bail)
    assert result.exit_code == 1

    with corpus_lock.acquire_corpus_lock("Docs", wait=False):
        pass


def test_no_wait_exits_nonzero_when_corpus_is_locked(isolated_locks, tmp_path):
    called = []

    def impl(*args, **kwargs):
        called.append(True)

    with corpus_lock.acquire_corpus_lock("Docs"):
        # Clear the reentrancy record so the CLI run contends like a separate
        # process would; the flock itself is still held by this process.
        held = dict(corpus_lock._held)
        corpus_lock._held.clear()
        try:
            result = _run(["corpus", "sync", "Docs", str(tmp_path), "--no-wait"], impl)
        finally:
            corpus_lock._held.update(held)

    assert result.exit_code != 0
    assert "Docs" in result.output
    assert not called, "sync body must not run when the lock is unavailable"


def test_lock_timeout_flag_is_honored(isolated_locks, tmp_path):
    called = []

    with corpus_lock.acquire_corpus_lock("Docs"):
        held = dict(corpus_lock._held)
        corpus_lock._held.clear()
        try:
            result = _run(
                ["corpus", "sync", "Docs", str(tmp_path), "--lock-timeout", "0.3"],
                lambda *a, **k: called.append(True),
            )
        finally:
            corpus_lock._held.update(held)

    assert result.exit_code != 0
    assert not called


def test_dry_run_takes_no_lock(isolated_locks, tmp_path):
    """A read-only preview must not queue behind (or block) a real sync."""
    ran = []

    with corpus_lock.acquire_corpus_lock("Docs"):
        held = dict(corpus_lock._held)
        corpus_lock._held.clear()
        try:
            result = _run(
                ["corpus", "sync", "Docs", str(tmp_path), "--dry-run", "--no-wait"],
                lambda *a, **k: ran.append(True),
            )
        finally:
            corpus_lock._held.update(held)

    assert result.exit_code == 0, result.output
    assert ran == [True]


def test_different_corpora_do_not_serialize(isolated_locks, tmp_path):
    with corpus_lock.acquire_corpus_lock("Other"):
        held = dict(corpus_lock._held)
        corpus_lock._held.clear()
        try:
            result = _run(
                ["corpus", "sync", "Docs", str(tmp_path), "--no-wait"],
                lambda *a, **k: None,
            )
        finally:
            corpus_lock._held.update(held)

    assert result.exit_code == 0, result.output


def test_repair_holds_the_lock(isolated_locks):
    """`arc corpus repair` writes to the corpus, so it locks too."""
    observed = {}

    def impl(corpus, *args, **kwargs):
        observed["holder"] = corpus_lock.read_lock_holder(corpus_lock.corpus_lock_path(corpus))

    runner = CliRunner()
    with patch("arcaneum.cli.sync._sync_directory_locked", impl):
        with patch("arcaneum.cli.utils.create_qdrant_client", side_effect=RuntimeError("no server")):
            result = runner.invoke(cli, ["corpus", "repair", "Docs"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert observed["holder"] is not None
    assert observed["holder"]["corpus"] == "Docs"


def test_delete_holds_the_lock_during_destructive_work(isolated_locks):
    """`arc corpus delete` must not interleave with an in-flight sync."""
    observed = {}

    qdrant = MagicMock()
    qdrant.get_collection.return_value = object()

    def record_delete(name):
        observed["holder"] = corpus_lock.read_lock_holder(corpus_lock.corpus_lock_path(name))

    qdrant.delete_collection.side_effect = record_delete

    meili = MagicMock()
    meili.health_check.return_value = False

    with patch("arcaneum.cli.corpus.create_qdrant_client", return_value=qdrant):
        with patch("arcaneum.cli.corpus.create_meili_client", return_value=meili):
            result = CliRunner().invoke(
                cli, ["corpus", "delete", "Docs", "--confirm"], catch_exceptions=False
            )

    assert result.exit_code == 0, result.output
    assert observed["holder"] is not None
    assert observed["holder"]["corpus"] == "Docs"

    # Released afterwards.
    with corpus_lock.acquire_corpus_lock("Docs", wait=False):
        pass


def test_delete_does_not_hold_the_lock_across_the_confirm_prompt(isolated_locks):
    """Prompting while holding the lock would block a running sync on a human."""
    qdrant = MagicMock()
    qdrant.get_collection.return_value = object()
    meili = MagicMock()
    meili.health_check.return_value = False

    def answer_no(*args, **kwargs):
        # The lock must be free while the user is being asked.
        with corpus_lock.acquire_corpus_lock("Docs", wait=False):
            pass
        return False

    with patch("arcaneum.cli.corpus.create_qdrant_client", return_value=qdrant):
        with patch("arcaneum.cli.corpus.create_meili_client", return_value=meili):
            with patch("click.confirm", answer_no):
                result = CliRunner().invoke(
                    cli, ["corpus", "delete", "Docs"], catch_exceptions=False
                )

    assert result.exit_code == 0, result.output
    qdrant.delete_collection.assert_not_called()

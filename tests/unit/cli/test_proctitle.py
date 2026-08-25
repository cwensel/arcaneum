"""Readable process titles for long-running arc work.

A drain worker shows up in ps/top as the full interpreter path plus the script
path, which is unidentifiable at a glance -- and during a rebase burst there
can be several at once, with no way to tell the worker holding the lock from
the ones exiting immediately.

setproctitle is optional: when it is absent every helper is a no-op, so this
never becomes a hard dependency for a cosmetic feature.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from arcaneum.cli import proctitle


def test_set_title_is_a_noop_without_setproctitle():
    with patch.object(proctitle, "_load", return_value=None):
        proctitle.set_title("arc drain Docs")  # must not raise


def test_set_title_prefixes_with_arc():
    fake = MagicMock()
    with patch.object(proctitle, "_load", return_value=fake):
        proctitle.set_title("drain Docs")
    fake.setproctitle.assert_called_once_with("arc drain Docs")


def test_from_argv_keeps_the_subcommand_and_options():
    """`top` should show what this arc is actually doing, options included."""
    argv = [
        "/Users/x/Library/Python/3.12/bin/arc",
        "corpus", "sync", "RetrofitRDR", "--drain-spool", "--max-batches", "5",
    ]
    assert proctitle.title_from_argv(argv) == (
        "arc corpus sync RetrofitRDR --drain-spool --max-batches 5"
    )


def test_from_argv_drops_only_the_interpreter_path():
    argv = ["/usr/bin/arc", "corpus", "hook", "status"]
    assert proctitle.title_from_argv(argv) == "arc corpus hook status"


def test_from_argv_with_no_arguments_is_just_arc():
    assert proctitle.title_from_argv(["/usr/bin/arc"]) == "arc"


def test_from_argv_truncates_a_pathological_title():
    argv = ["/usr/bin/arc", "corpus", "sync", "C"] + [f"--flag-{i}" for i in range(200)]
    title = proctitle.title_from_argv(argv)
    assert len(title) <= proctitle.MAX_TITLE_CHARS
    assert title.startswith("arc corpus sync C")


def test_set_title_does_not_double_prefix():
    fake = MagicMock()
    with patch.object(proctitle, "_load", return_value=fake):
        proctitle.set_title("arc drain Docs")
    fake.setproctitle.assert_called_once_with("arc drain Docs")


def test_a_failing_backend_is_swallowed():
    """Cosmetics must never take down an indexing run."""
    fake = MagicMock()
    fake.setproctitle.side_effect = RuntimeError("nope")
    with patch.object(proctitle, "_load", return_value=fake):
        proctitle.set_title("drain Docs")  # must not raise


def test_drain_sets_a_title_naming_the_corpus(tmp_path, monkeypatch):
    """The whole point: tell one worker from another in top."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    from click.testing import CliRunner

    from arcaneum.cli import spool
    from arcaneum.cli.main import cli

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x\n")
    spool.write_entry("Docs", repo, changed=[str(repo / "a.py")], removed=[])

    titles = []
    with patch("arcaneum.cli.proctitle.set_title", lambda t: titles.append(t)):
        with patch("arcaneum.cli.sync.sync_directory_command", lambda *a, **k: None):
            CliRunner().invoke(
                cli, ["corpus", "sync", "Docs", "--drain-spool"], catch_exceptions=False
            )

    assert any("Docs" in t for t in titles)
    assert any("drain" in t.lower() for t in titles)


def test_main_sets_the_title_from_argv():
    """Every arc invocation should be readable in ps/top, not just drains."""
    from arcaneum.cli import main as main_mod

    titles = []
    with patch("arcaneum.cli.proctitle.set_title_from_argv", lambda: titles.append("set")):
        with patch.object(main_mod, "cli", lambda **k: None):
            with patch.object(main_mod, "configure_ssl_from_env", lambda: None):
                main_mod.main()

    assert titles == ["set"]


# --- the title must land before the heavy imports (not ~700ms in) -------------


def test_proctitle_module_does_not_pull_heavy_deps():
    """Importing proctitle must not drag in qdrant_client/fastembed.

    The whole point of titling early is to beat the ~700ms import chain; if
    this module joined it, the title would land no sooner than before.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from arcaneum.cli import proctitle\n"
        "heavy = [m for m in sys.modules if m.split('.')[0] in "
        "('qdrant_client', 'fastembed', 'torch', 'click')]\n"
        "print(heavy)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", f"proctitle pulled heavy modules: {out.stdout}"


def test_entrypoint_titles_before_importing_main():
    """`arc` should title the process first, then do the expensive import."""
    from arcaneum.cli import entrypoint

    order = []
    with patch("arcaneum.cli.proctitle.set_title_from_argv", lambda: order.append("title")):
        with patch.object(entrypoint, "_load_main", lambda: (order.append("import"), lambda: 0)[1]):
            entrypoint.run()

    assert order == ["title", "import"], "title must be set before the heavy import"


def test_entrypoint_returns_the_exit_code_from_main():
    from arcaneum.cli import entrypoint

    with patch("arcaneum.cli.proctitle.set_title_from_argv", lambda: None):
        with patch.object(entrypoint, "_load_main", lambda: (lambda: 42)):
            assert entrypoint.run() == 42


def test_entrypoint_still_works_without_setproctitle():
    from arcaneum.cli import entrypoint
    from arcaneum.cli import proctitle as pt

    with patch.object(pt, "_load", return_value=None):
        with patch.object(entrypoint, "_load_main", lambda: (lambda: 0)):
            assert entrypoint.run() == 0

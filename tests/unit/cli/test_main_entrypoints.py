"""Smoke tests for click CLI entrypoints in arcaneum.cli.main.

These tests invoke the click group the same way `arc` does, so they catch
import-time errors inside command handlers — for example, a handler that
imports a symbol from a module that no longer exports it. Pure-function
tests on the underlying *_command helpers bypass this layer and miss such
regressions.
"""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from arcaneum.cli.main import cli


def _run(args, **patches):
    """Invoke the cli with patches applied; return the CliRunner result."""
    runner = CliRunner()
    stack = []
    try:
        for target, value in patches.items():
            p = patch(target, value)
            p.start()
            stack.append(p)
        return runner.invoke(cli, args, catch_exceptions=False)
    finally:
        for p in reversed(stack):
            p.stop()


def _silent_logger():
    logger = MagicMock()
    track_ctx = MagicMock()
    track_ctx.__enter__.return_value = {}
    track_ctx.__exit__.return_value = False
    logger.track.return_value = track_ctx
    return logger


def test_search_semantic_entrypoint_dispatches():
    """`arc search semantic` must reach search_command without ImportError."""
    called = {}

    def fake_search_command(query, corpora, *args, **kwargs):
        called['query'] = query
        called['corpora'] = corpora
        raise SystemExit(0)

    result = _run(
        ['search', 'semantic', 'hello world', '--corpus', 'MyCorpus'],
        **{
            'arcaneum.cli.search.search_command': fake_search_command,
            'arcaneum.cli.search.interaction_logger': _silent_logger(),
        },
    )

    assert result.exit_code == 0, result.output
    assert called['query'] == 'hello world'
    assert called['corpora'] == ['MyCorpus']


def test_search_text_entrypoint_dispatches():
    """`arc search text` must reach search_text_command without ImportError."""
    called = {}

    def fake_search_text_command(query, corpora, *args, **kwargs):
        called['query'] = query
        called['corpora'] = corpora
        raise SystemExit(0)

    result = _run(
        ['search', 'text', 'hello world', '--corpus', 'MyCorpus'],
        **{
            'arcaneum.cli.fulltext.search_text_command': fake_search_text_command,
            'arcaneum.cli.fulltext.interaction_logger': _silent_logger(),
        },
    )

    assert result.exit_code == 0, result.output
    assert called['query'] == 'hello world'
    assert called['corpora'] == ['MyCorpus']

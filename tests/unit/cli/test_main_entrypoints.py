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


def test_corpus_create_models_default_is_inferred():
    """`arc corpus create` leaves model selection to corpus-type defaults."""
    called = {}

    def fake_create_corpus_command(name, corpus_type, models, output_json):
        called['name'] = name
        called['corpus_type'] = corpus_type
        called['models'] = models
        raise SystemExit(0)

    result = _run(
        ['corpus', 'create', 'Docs', '--type', 'markdown'],
        **{'arcaneum.cli.corpus.create_corpus_command': fake_create_corpus_command},
    )

    assert result.exit_code == 0, result.output
    assert called == {
        'name': 'Docs',
        'corpus_type': 'markdown',
        'models': None,
    }


def test_corpus_sync_defaults_to_cpu_and_gpu_is_opt_in():
    """`arc corpus sync` defaults to no_gpu=True but accepts --gpu."""
    calls = []

    def fake_sync_directory_command(
        corpus, paths, from_file, models, file_types, force, verify,
        text_workers, max_embedding_batch, no_gpu, *args, **kwargs
    ):
        calls.append({
            'corpus': corpus,
            'models': models,
            'no_gpu': no_gpu,
        })
        raise SystemExit(0)

    default_result = _run(
        ['corpus', 'sync', 'Docs', '.'],
        **{'arcaneum.cli.sync.sync_directory_command': fake_sync_directory_command},
    )
    gpu_result = _run(
        ['corpus', 'sync', 'Docs', '.', '--gpu'],
        **{'arcaneum.cli.sync.sync_directory_command': fake_sync_directory_command},
    )

    assert default_result.exit_code == 0, default_result.output
    assert gpu_result.exit_code == 0, gpu_result.output
    assert calls == [
        {'corpus': 'Docs', 'models': None, 'no_gpu': True},
        {'corpus': 'Docs', 'models': None, 'no_gpu': False},
    ]

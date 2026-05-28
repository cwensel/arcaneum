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

    def fake_create_corpus_command(name, corpus_type, models, description, output_json):
        called['name'] = name
        called['corpus_type'] = corpus_type
        called['models'] = models
        called['description'] = description
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
        'description': None,
    }


def test_corpus_create_description_dispatches():
    """`arc corpus create --description` forwards metadata to the handler."""
    called = {}

    def fake_create_corpus_command(name, corpus_type, models, description, output_json):
        called['description'] = description
        raise SystemExit(0)

    result = _run(
        [
            'corpus', 'create', 'Docs', '--type', 'markdown',
            '--description', 'Project design notes',
        ],
        **{'arcaneum.cli.corpus.create_corpus_command': fake_create_corpus_command},
    )

    assert result.exit_code == 0, result.output
    assert called['description'] == 'Project design notes'


def test_corpus_update_description_dispatches():
    """`arc corpus update --description` reaches the metadata update handler."""
    called = {}

    def fake_update_corpus_command(name, description, clear_description, output_json):
        called['name'] = name
        called['description'] = description
        called['clear_description'] = clear_description
        raise SystemExit(0)

    result = _run(
        ['corpus', 'update', 'Docs', '--description', 'Updated scope'],
        **{'arcaneum.cli.corpus.update_corpus_command': fake_update_corpus_command},
    )

    assert result.exit_code == 0, result.output
    assert called == {
        'name': 'Docs',
        'description': 'Updated scope',
        'clear_description': False,
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


def test_index_markdown_qdrant_url_defaults_to_shared_resolution(tmp_path):
    """`arc index markdown` must not force localhost before client creation."""
    called = {}
    docs = tmp_path / "docs"
    docs.mkdir()

    def fake_index_markdown_command(*args, **kwargs):
        called['qdrant_url'] = args[9]
        called['embedding_batch_size'] = args[4]
        raise SystemExit(0)

    result = _run(
        ['index', 'markdown', str(docs), '--collection', 'Docs'],
        **{'arcaneum.cli.index_markdown.index_markdown_command': fake_index_markdown_command},
    )

    assert result.exit_code == 0, result.output
    assert called['qdrant_url'] is None
    assert called['embedding_batch_size'] is None


def test_index_markdown_explicit_qdrant_url_is_preserved(tmp_path):
    """Explicit `--qdrant-url` remains an override."""
    called = {}
    docs = tmp_path / "docs"
    docs.mkdir()

    def fake_index_markdown_command(*args, **kwargs):
        called['qdrant_url'] = args[9]
        raise SystemExit(0)

    result = _run(
        [
            'index', 'markdown', str(docs), '--collection', 'Docs',
            '--qdrant-url', 'https://qdrant.example',
        ],
        **{'arcaneum.cli.index_markdown.index_markdown_command': fake_index_markdown_command},
    )

    assert result.exit_code == 0, result.output
    assert called['qdrant_url'] == 'https://qdrant.example'


def test_index_source_uses_shared_qdrant_resolution(tmp_path):
    """`arc index code` must not force localhost before client creation."""
    from arcaneum.cli.index_source import index_source_command

    with patch('arcaneum.cli.index_source.interaction_logger'):
        with patch('arcaneum.cli.index_source.create_qdrant_client') as mock_create:
            with patch(
                'arcaneum.indexing.collection_metadata.get_collection_metadata',
                return_value=None,
            ):
                result = None
                try:
                    index_source_command(
                        path=str(tmp_path),
                        from_file=None,
                        collection='Code',
                        model=None,
                        embedding_batch_size=None,
                        chunk_size=None,
                        chunk_overlap=None,
                        depth=None,
                        process_priority='normal',
                        not_nice=True,
                        force=False,
                        no_gpu=True,
                        verify=False,
                        streaming=True,
                        verbose=False,
                        debug=False,
                        profile=False,
                        output_json=True,
                    )
                except SystemExit as exc:
                    result = exc

    assert result is not None
    assert result.code == 1
    mock_create.assert_called_once_with()

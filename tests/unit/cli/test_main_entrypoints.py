"""Smoke tests for click CLI entrypoints in arcaneum.cli.main.

These tests invoke the click group the same way `arc` does, so they catch
import-time errors inside command handlers — for example, a handler that
imports a symbol from a module that no longer exports it. Pure-function
tests on the underlying *_command helpers bypass this layer and miss such
regressions.
"""

from unittest.mock import MagicMock, patch

import json
import click
import pytest
from click.testing import CliRunner

from arcaneum.cli.main import cli, main


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


def test_root_json_flag_propagates_to_subcommand(tmp_path):
    """`arc --json ...` should activate existing local output_json handlers."""
    runner = CliRunner()

    models_dir = tmp_path / "models"
    data_dir = tmp_path / "data"
    legacy_dir = tmp_path / "legacy"

    with patch('arcaneum.cli.config.get_models_dir', return_value=models_dir):
        with patch('arcaneum.cli.config.get_data_dir', return_value=data_dir):
            with patch('arcaneum.cli.config.get_legacy_arcaneum_dir', return_value=legacy_dir):
                result = runner.invoke(
                    cli,
                    ['--json', 'config', 'show-cache-dir'],
                    catch_exceptions=False,
                )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "success"
    assert "cache" in payload["data"]


def test_main_json_formats_click_errors(capsys):
    """Console entrypoint should format Click errors as JSON under --json."""
    with patch('sys.argv', ['arc', '--json', 'missing-command']):
        with patch('arcaneum.cli.main.configure_ssl_from_env'):
            exit_code = main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 2
    assert payload["status"] == "error"
    assert "missing-command" in payload["message"]


def test_main_json_formats_click_abort(capsys):
    """Console entrypoint should not classify Click aborts as unexpected."""
    with patch('sys.argv', ['arc', '--json']):
        with patch('arcaneum.cli.main.configure_ssl_from_env'):
            with patch('arcaneum.cli.main.cli', side_effect=click.Abort()):
                exit_code = main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["message"] == "Operation aborted"


def test_main_preserves_click_exit():
    """Console entrypoint should preserve successful Click exits."""
    with patch('sys.argv', ['arc', '--json']):
        with patch('arcaneum.cli.main.configure_ssl_from_env'):
            with patch('arcaneum.cli.main.cli', side_effect=click.exceptions.Exit(0)):
                exit_code = main()

    assert exit_code == 0


def test_main_json_formats_path_validation_errors(capsys):
    """Validation helpers should surface specific messages in JSON mode."""
    with patch('sys.argv', ['arc', '--json', 'index', 'markdown', '--collection', 'Docs']):
        with patch('arcaneum.cli.main.configure_ssl_from_env'):
            exit_code = main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 2
    assert payload["status"] == "error"
    assert "Either PATH or --from-file" in payload["message"]


def test_main_json_formats_corpus_sync_validation_errors(capsys):
    """Corpus sync validation should flow through JSON UsageError handling."""
    with patch('sys.argv', ['arc', '--json', 'corpus', 'sync', 'Docs']):
        with patch('arcaneum.cli.main.configure_ssl_from_env'):
            exit_code = main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 2
    assert payload["status"] == "error"
    assert "Either PATH(s) or --from-file" in payload["message"]


def test_main_json_formats_corpus_sync_git_flag_conflict(capsys):
    """Mutually exclusive corpus sync flags should return JSON errors."""
    with patch('sys.argv', [
        'arc', '--json', 'corpus', 'sync', 'Docs', '.',
        '--git-update', '--git-version',
    ]):
        with patch('arcaneum.cli.main.configure_ssl_from_env'):
            exit_code = main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 2
    assert payload["status"] == "error"
    assert "--git-update and --git-version" in payload["message"]


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


def test_corpus_list_details_dispatches():
    """`arc corpus list --details` enables extended list output."""
    called = {}

    def fake_list_corpora_command(details, output_json):
        called['details'] = details
        called['output_json'] = output_json
        raise SystemExit(0)

    result = _run(
        ['corpus', 'list', '--details', '--json'],
        **{'arcaneum.cli.corpus.list_corpora_command': fake_list_corpora_command},
    )

    assert result.exit_code == 0, result.output
    assert called == {'details': True, 'output_json': True}


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


def test_corpus_parity_passes_max_embedding_batch():
    called = {}

    def fake_parity_command(
        name, dry_run, verify, repair_metadata, text_workers,
        max_embedding_batch, timeout, create_missing, confirm, verbose, output_json
    ):
        called["name"] = name
        called["max_embedding_batch"] = max_embedding_batch
        raise SystemExit(0)

    result = _run(
        ['corpus', 'parity', 'Docs', '--max-embedding-batch', '4'],
        **{'arcaneum.cli.sync.parity_command': fake_parity_command},
    )

    assert result.exit_code == 0, result.output
    assert called == {"name": "Docs", "max_embedding_batch": 4}


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
                        prune=False,
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


def test_index_source_blocks_stale_prompt_policy_before_indexing(tmp_path):
    from types import SimpleNamespace

    from arcaneum.cli.index_source import index_source_command
    from arcaneum.embeddings.client import get_embedding_prompt_policy

    stale_policy = {
        **get_embedding_prompt_policy("jina-code-st"),
        "model": "jina-code",
    }
    mock_indexer = MagicMock()
    mock_indexer.collection_exists.return_value = True

    with patch('arcaneum.cli.index_source.interaction_logger'):
        with patch('arcaneum.cli.index_source.create_qdrant_client', return_value=object()):
            with patch('arcaneum.cli.index_source.QdrantIndexer', return_value=mock_indexer):
                with patch('arcaneum.cli.index_source.get_vector_names', return_value=["jina-code"]):
                    with patch(
                        'arcaneum.cli.index_source.get_cached_model',
                        return_value=SimpleNamespace(get_device_info=lambda: {"gpu_available": False}),
                    ):
                        with patch('arcaneum.cli.index_source.validate_collection_type'):
                            with patch(
                                'arcaneum.cli.index_source.get_collection_metadata',
                                return_value={
                                    "model": "jina-code",
                                    "embedding_prompt_policy": {"jina-code": stale_policy},
                                },
                            ):
                                with pytest.raises(SystemExit) as exc:
                                    index_source_command(
                                        path=str(tmp_path),
                                        from_file=None,
                                        collection='Code',
                                        model=None,
                                        embedding_batch_size=128,
                                        chunk_size=None,
                                        chunk_overlap=None,
                                        depth=None,
                                        process_priority='normal',
                                        not_nice=True,
                                        force=False,
                                        prune=False,
                                        no_gpu=True,
                                        verify=False,
                                        streaming=True,
                                        verbose=False,
                                        debug=False,
                                        profile=False,
                                        output_json=True,
                                    )

    assert exc.value.code == 1
    mock_indexer.index_directory.assert_not_called()


def test_index_source_force_requires_prompt_policy_certification(tmp_path):
    from types import SimpleNamespace

    from arcaneum.cli.index_source import index_source_command
    from arcaneum.embeddings.client import get_embedding_prompt_policy

    source_file = tmp_path / "app.py"
    source_file.write_text("print('ok')\n")
    stale_policy = {
        **get_embedding_prompt_policy("jina-code-st"),
        "model": "jina-code",
    }
    mock_qdrant_indexer = MagicMock()
    mock_qdrant_indexer.collection_exists.return_value = True
    mock_source_indexer = MagicMock()
    mock_source_indexer.index_directory.return_value = {
        "covered_paths": [str(source_file)],
        "files_processed": 1,
        "errors": 0,
        "projects_processed": 1,
        "chunks_uploaded": 1,
    }
    mock_sync = MagicMock()
    mock_sync._get_indexed_file_paths_set.return_value = {str(source_file)}
    mock_cached_model = MagicMock(
        return_value=SimpleNamespace(get_device_info=lambda: {"gpu_available": False})
    )

    with patch('arcaneum.cli.index_source.interaction_logger'):
        with patch('arcaneum.cli.index_source.create_qdrant_client', return_value=object()):
            with patch('arcaneum.cli.index_source.QdrantIndexer', return_value=mock_qdrant_indexer):
                with patch('arcaneum.cli.index_source.SourceCodeIndexer', return_value=mock_source_indexer):
                    with patch('arcaneum.cli.index_source.get_vector_names', return_value=["jina-code"]):
                        with patch(
                            'arcaneum.cli.index_source.get_cached_model',
                            mock_cached_model,
                        ):
                            with patch('arcaneum.cli.index_source.validate_collection_type'):
                                with patch(
                                    'arcaneum.cli.index_source.get_collection_metadata',
                                    return_value={
                                        "model": "jinaai/jina-embeddings-v2-base-code",
                                        "embedding_prompt_policy": {"jina-code": stale_policy},
                                    },
                                ):
                                    with patch(
                                        'arcaneum.indexing.common.sync.MetadataBasedSync',
                                        return_value=mock_sync,
                                    ):
                                        with patch(
                                            'arcaneum.indexing.collection_metadata.prune_orphans_and_stamp',
                                            return_value={"stamped": False},
                                        ):
                                            with pytest.raises(SystemExit) as exc:
                                                index_source_command(
                                                    path=str(tmp_path),
                                                    from_file=None,
                                                    collection='Code',
                                                    model=None,
                                                    embedding_batch_size=128,
                                                    chunk_size=None,
                                                    chunk_overlap=None,
                                                    depth=None,
                                                    process_priority='normal',
                                                    not_nice=True,
                                                    force=True,
                                                    prune=False,
                                                    no_gpu=True,
                                                    verify=False,
                                                    streaming=True,
                                                    verbose=False,
                                                    debug=False,
                                                    profile=False,
                                                    output_json=True,
                                                )

    assert exc.value.code == 1
    assert mock_cached_model.call_args.kwargs["model_name"] == "jina-code"
    mock_source_indexer.index_directory.assert_called_once()


def test_index_source_rejects_explicit_model_vector_mismatch(tmp_path):
    from types import SimpleNamespace

    from arcaneum.cli.index_source import index_source_command
    from arcaneum.embeddings.client import get_embedding_prompt_policy

    mock_qdrant_indexer = MagicMock()
    mock_qdrant_indexer.collection_exists.return_value = True

    with patch('arcaneum.cli.index_source.interaction_logger'):
        with patch('arcaneum.cli.index_source.create_qdrant_client', return_value=object()):
            with patch('arcaneum.cli.index_source.QdrantIndexer', return_value=mock_qdrant_indexer):
                with patch('arcaneum.cli.index_source.SourceCodeIndexer') as source_indexer:
                    with patch('arcaneum.cli.index_source.get_vector_names', return_value=["jina-code"]):
                        with patch(
                            'arcaneum.cli.index_source.get_cached_model',
                            return_value=SimpleNamespace(
                                get_device_info=lambda: {"gpu_available": False}
                            ),
                        ):
                            with patch('arcaneum.cli.index_source.validate_collection_type'):
                                with patch(
                                    'arcaneum.cli.index_source.get_collection_metadata',
                                    return_value={
                                        "model": "jina-code",
                                        "embedding_prompt_policy": {
                                            "jina-code": get_embedding_prompt_policy("jina-code")
                                        },
                                    },
                                ):
                                    with pytest.raises(SystemExit) as exc:
                                        index_source_command(
                                            path=str(tmp_path),
                                            from_file=None,
                                            collection='Code',
                                            model="jina-code-st",
                                            embedding_batch_size=128,
                                            chunk_size=None,
                                            chunk_overlap=None,
                                            depth=None,
                                            process_priority='normal',
                                            not_nice=True,
                                            force=True,
                                            prune=False,
                                            no_gpu=True,
                                            verify=False,
                                            streaming=True,
                                            verbose=False,
                                            debug=False,
                                            profile=False,
                                            output_json=True,
                                        )

    assert exc.value.code == 1
    source_indexer.assert_not_called()


def test_index_markdown_prune_flag_is_threaded(tmp_path):
    """`arc index markdown --prune` forwards prune=True to the command."""
    called = {}
    docs = tmp_path / "docs"
    docs.mkdir()

    def fake_cmd(*args, **kwargs):
        # force at index 12, prune at index 13
        called['force'] = args[12]
        called['prune'] = args[13]
        raise SystemExit(0)

    result = _run(
        ['index', 'markdown', str(docs), '--collection', 'Docs', '--force', '--prune'],
        **{'arcaneum.cli.index_markdown.index_markdown_command': fake_cmd},
    )

    assert result.exit_code == 0, result.output
    assert called['force'] is True
    assert called['prune'] is True


def test_index_markdown_prune_defaults_off(tmp_path):
    """Without --prune, prune must default to False."""
    called = {}
    docs = tmp_path / "docs"
    docs.mkdir()

    def fake_cmd(*args, **kwargs):
        called['prune'] = args[13]
        raise SystemExit(0)

    result = _run(
        ['index', 'markdown', str(docs), '--collection', 'Docs', '--force'],
        **{'arcaneum.cli.index_markdown.index_markdown_command': fake_cmd},
    )

    assert result.exit_code == 0, result.output
    assert called['prune'] is False


def test_index_pdf_prune_flag_is_threaded(tmp_path):
    """`arc index pdf --prune` forwards prune=True."""
    called = {}
    docs = tmp_path / "pdfs"
    docs.mkdir()

    def fake_cmd(*args, **kwargs):
        called['prune'] = args[13]
        raise SystemExit(0)

    result = _run(
        ['index', 'pdf', str(docs), '--collection', 'Docs', '--force', '--prune'],
        **{'arcaneum.cli.index_pdfs.index_pdfs_command': fake_cmd},
    )

    assert result.exit_code == 0, result.output
    assert called['prune'] is True


def test_index_code_prune_flag_is_threaded(tmp_path):
    """`arc index code --prune` forwards prune=True."""
    called = {}
    repo = tmp_path / "repo"
    repo.mkdir()

    def fake_cmd(*args, **kwargs):
        # force at index 10, prune at index 11
        called['force'] = args[10]
        called['prune'] = args[11]
        raise SystemExit(0)

    result = _run(
        ['index', 'code', str(repo), '--collection', 'Code', '--force', '--prune'],
        **{'arcaneum.cli.index_source.index_source_command': fake_cmd},
    )

    assert result.exit_code == 0, result.output
    assert called['force'] is True
    assert called['prune'] is True

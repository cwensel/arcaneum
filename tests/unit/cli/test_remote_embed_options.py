"""--remote-embed / --local-embed and per-command routing (RDR-025 Step 4)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from arcaneum.cli.main import cli
from arcaneum.embeddings.client import EmbeddingClient
from arcaneum.embeddings.remote.config import EndpointConfig, RemoteEmbeddingConfig
from arcaneum.embeddings.remote.routing import RemoteRouter, active_router, install_router


@pytest.fixture(autouse=True)
def no_router():
    install_router(None)
    yield
    install_router(None)


@pytest.mark.parametrize(
    ("flags", "mode"),
    [([], "default"), (["--remote-embed"], "remote"), (["--local-embed"], "local")],
)
def test_corpus_sync_passes_the_embed_mode(tmp_path, flags, mode):
    with patch("arcaneum.cli.sync.sync_directory_command") as sync:
        result = CliRunner().invoke(cli, ["corpus", "sync", "MyCorpus", str(tmp_path), *flags])

    assert result.exit_code == 0, result.output
    assert sync.call_args.kwargs["embed_mode"] == mode


@pytest.mark.parametrize(
    ("flags", "mode"),
    [([], "default"), (["--remote-embed"], "remote"), (["--local-embed"], "local")],
)
def test_corpus_repair_passes_the_embed_mode(flags, mode):
    with patch("arcaneum.cli.sync.repair_corpus_command") as repair:
        result = CliRunner().invoke(cli, ["corpus", "repair", "MyCorpus", *flags])

    assert result.exit_code == 0, result.output
    assert repair.call_args.kwargs["embed_mode"] == mode


def test_remote_and_local_embed_are_mutually_exclusive(tmp_path):
    with patch("arcaneum.cli.sync.sync_directory_command") as sync:
        result = CliRunner().invoke(
            cli, ["corpus", "sync", "MyCorpus", str(tmp_path), "--remote-embed", "--local-embed"]
        )

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
    sync.assert_not_called()


def test_sync_command_configures_routing_for_its_invocation():
    from arcaneum.cli import sync as sync_module

    with (
        patch.object(sync_module, "configure_remote_embedding") as configure,
        patch.object(sync_module, "_sync_directory_locked"),
        patch.object(sync_module, "acquire_corpus_lock"),
    ):
        sync_module.sync_directory_command(
            "MyCorpus", (), None, None, None, False, False, None, None, True, None, False,
            False, embed_mode="remote",
        )  # fmt: skip

    configure.assert_called_once_with("remote")


def test_repair_routes_both_passes_with_its_embed_mode():
    from arcaneum.cli import sync as sync_module

    with (
        patch.object(sync_module, "sync_directory_command") as sync,
        patch.object(sync_module, "parity_command") as parity,
        patch.object(sync_module, "acquire_corpus_lock"),
    ):
        sync_module.repair_corpus_command(
            "MyCorpus", None, 0.9, False, True, None, False, False, False, embed_mode="local"
        )

    assert sync.call_args.kwargs["embed_mode"] == "local"
    assert parity.call_args.kwargs["embed_mode"] == "local"


def test_invalid_configuration_stops_the_sync(capsys):
    from arcaneum.cli import sync as sync_module
    from arcaneum.embeddings.remote.config import RemoteEmbeddingConfigError

    error = RemoteEmbeddingConfigError("Model 'bge-small' cannot be routed 'remote'")
    with (
        patch.object(sync_module, "configure_remote_embedding", side_effect=error),
        patch.object(sync_module, "_sync_directory_locked") as locked,
        pytest.raises(SystemExit),
    ):
        sync_module.sync_directory_command(
            "MyCorpus", (), None, None, None, False, False, None, None, True, None, False, False
        )

    locked.assert_not_called()
    captured = capsys.readouterr()
    assert "bge-small" in captured.out + captured.err


@pytest.mark.parametrize("command", ["pdf", "code", "markdown"])
def test_index_commands_route_by_configuration_only(tmp_path, command):
    target = {
        "pdf": "arcaneum.cli.index_pdfs.index_pdfs_command",
        "code": "arcaneum.cli.index_source.index_source_command",
        "markdown": "arcaneum.cli.index_markdown.index_markdown_command",
    }[command]
    with (
        patch("arcaneum.embeddings.remote.routing.configure_remote_embedding") as configure,
        patch(target),
    ):
        result = CliRunner().invoke(
            cli, ["index", command, str(tmp_path), "--collection", "MyCollection"]
        )

    assert result.exit_code == 0, result.output
    configure.assert_called_once_with("default")


def test_cached_model_is_not_loaded_locally_when_routed_remote(tmp_path):
    from arcaneum.embeddings import model_cache

    router = RemoteRouter(
        RemoteEmbeddingConfig(endpoints=[EndpointConfig(host="dev-0")]), mode="default"
    )
    install_router(router)
    model_cache._model_cache.clear()
    with patch.object(EmbeddingClient, "get_model") as get_model:
        client = model_cache.get_cached_model("arctic-m", cache_dir=str(tmp_path))

    get_model.assert_not_called()
    assert client.local_models == []
    model_cache._model_cache.clear()


def test_prepare_model_loads_unrouted_models(tmp_path):
    client = EmbeddingClient(cache_dir=str(tmp_path))

    with patch.object(client, "get_model") as get_model:
        assert client.prepare_model("arctic-m") is True

    get_model.assert_called_once_with("arctic-m")


def test_sync_ends_the_run_when_remote_embedding_is_unavailable(tmp_path):
    """A `fail` policy error is not swallowed by sync's per-file error handling:
    the file being embedded gets no chunks and later files are not attempted."""
    from contextlib import ExitStack

    from arcaneum.cli import sync as sync_module
    from arcaneum.embeddings.remote.errors import RemoteEmbeddingUnavailable

    sources = []
    for name in ("a.md", "b.md"):
        source = tmp_path / name
        source.write_text("# Note\n\nbody\n")
        sources.append(source.resolve())
    embedded, indexed = [], []
    client = MagicMock()

    def embed(texts, model, max_internal_batch=None):
        embedded.append(texts)
        raise RemoteEmbeddingUnavailable("No remote embedding endpoint can serve arctic-m")

    client.embed.side_effect = embed
    indexer = MagicMock()
    indexer.index_batch.side_effect = lambda docs, wait=True: indexed.append(docs) or (1, 1)
    indexer.delete_by_file_path.return_value = (0, 0)
    manifests = MagicMock()
    manifests.get_unindexed_files.return_value = (sources, [])
    manifests._get_indexed_file_paths_set.return_value = set()
    manifests.find_file_by_content_hash.return_value = []
    meili = MagicMock()
    meili.health_check.return_value = True
    meili.index_exists.return_value = True
    targets = {
        "create_qdrant_client": MagicMock(),
        "get_meili_client": MagicMock(return_value=meili),
        "get_collection_type": MagicMock(return_value="markdown"),
        "get_collection_metadata": MagicMock(return_value={"model": "arctic-m"}),
        "_maybe_backfill_legacy_prompt_policy": MagicMock(
            side_effect=lambda q, c, t, m, metadata, o: (metadata, [])
        ),
        "MetadataBasedSync": MagicMock(return_value=manifests),
        "_ensure_file_manifests": MagicMock(),
        "EmbeddingClient": MagicMock(return_value=client),
        "_report_embedding_backend": MagicMock(),
        "_finalize_embedding_clients": MagicMock(),
        "DualIndexer": MagicMock(return_value=indexer),
        "chunk_markdown_file": MagicMock(return_value=[{"text": "body", "metadata": {}}]),
        "interaction_logger": MagicMock(),
    }
    with ExitStack() as stack:
        for name, value in targets.items():
            stack.enter_context(patch.object(sync_module, name, value))
        with pytest.raises(SystemExit) as exited:
            sync_module._sync_directory_locked(
                "Notes", tuple(map(str, sources)), None, "arctic-m", None, True, False,
                None, None, True, None, False, True,
            )  # fmt: skip

    assert exited.value.code == 1
    assert len(embedded) == 1
    assert indexed == []


def test_no_endpoints_installs_no_router(monkeypatch):
    from arcaneum.embeddings.remote.routing import configure_remote_embedding

    monkeypatch.setenv("ARC_EMBED_ENDPOINTS", "")
    assert configure_remote_embedding("remote") is None
    assert active_router() is None


def test_endpoints_install_a_router_with_the_invocation_mode(monkeypatch):
    from arcaneum.embeddings.remote.routing import configure_remote_embedding

    monkeypatch.setenv("ARC_EMBED_ENDPOINTS", "dev-0:7300")
    router = configure_remote_embedding("remote")

    assert active_router() is router
    assert router.mode == "remote"
    assert router.routes("qwen3-embed", "document")


@pytest.mark.parametrize(
    ("mode", "models"),
    [("default", "arctic-m"), ("remote", "qwen3-embed, arctic-m")],
)
def test_run_start_names_the_remote_models_and_workers_once(monkeypatch, capsys, mode, models):
    from arcaneum.embeddings.remote.routing import configure_remote_embedding

    monkeypatch.setenv("ARC_EMBED_ENDPOINTS", "dev-0:7300,dev-1:7300")
    configure_remote_embedding(mode)
    configure_remote_embedding(mode)  # repair's parity pass reuses the router

    captured = capsys.readouterr()
    assert captured.out == ""
    lines = [line for line in captured.err.splitlines() if "Remote embedding" in line]
    assert lines == [f"   Remote embedding: {models} on dev-0:7300, dev-1:7300"]


@pytest.mark.parametrize("environment", ["", "dev-0:7300"])
def test_runs_that_route_nothing_print_nothing(monkeypatch, capsys, environment):
    from arcaneum.embeddings.remote.routing import configure_remote_embedding

    monkeypatch.setenv("ARC_EMBED_ENDPOINTS", environment)
    configure_remote_embedding("local")

    assert "Remote embedding" not in capsys.readouterr().err


def test_router_embed_delegates_to_its_pool():
    router = RemoteRouter(RemoteEmbeddingConfig(endpoints=[EndpointConfig(host="dev-0")]))
    pool = MagicMock()
    pool.embed.return_value = np.ones((1, 768), dtype=np.float32)
    router._pool = pool

    router.embed(["x"], "arctic-m")

    pool.embed.assert_called_once_with(["x"], "arctic-m", validate=None)


def test_unchanged_configuration_reuses_the_router_and_its_connections(monkeypatch):
    from arcaneum.embeddings.remote.routing import configure_remote_embedding

    monkeypatch.setenv("ARC_EMBED_ENDPOINTS", "dev-0:7300")
    first = configure_remote_embedding("default")
    pool = first.pool

    assert configure_remote_embedding("default") is first
    assert first.pool is pool
    assert configure_remote_embedding("remote") is not first

"""`arc models serve` command (RDR-025 Step 3)."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from arcaneum.cli.main import cli
from arcaneum.cli.remote_embedding import parse_batch_sizes, parse_served_models


def test_parse_served_models_accepts_pinned_models():
    assert parse_served_models("arctic-m, qwen3-embed") == ("arctic-m", "qwen3-embed")
    assert parse_served_models(None) is None
    assert parse_served_models("") is None


def test_parse_served_models_rejects_unknown_and_unpinned_models():
    with pytest.raises(ValueError, match="nope"):
        parse_served_models("nope")
    with pytest.raises(ValueError, match="bge-small.*revision"):
        parse_served_models("bge-small")


def test_parse_batch_sizes():
    assert parse_batch_sizes("arctic-m=16, qwen3-embed=4") == {"arctic-m": 16, "qwen3-embed": 4}
    assert parse_batch_sizes(None) == {}
    with pytest.raises(ValueError, match="MODEL=N"):
        parse_batch_sizes("arctic-m")
    with pytest.raises(ValueError, match="positive"):
        parse_batch_sizes("arctic-m=0")


def test_serve_builds_settings_from_options_and_environment(monkeypatch):
    monkeypatch.setenv("ARC_EMBED_AUTHKEY", "shared-secret")
    monkeypatch.setenv("ARC_EMBED_BIND", "tailscale")
    monkeypatch.setenv("ARC_EMBED_MODELS", "arctic-m")
    monkeypatch.setenv("ARC_EMBED_BATCH_SIZES", "arctic-m=32")

    with patch("arcaneum.cli.remote_embedding.EmbeddingServer") as server:
        result = CliRunner().invoke(
            cli,
            ["models", "serve", "--port", "7311", "--idle-grace", "30", "--turn-length", "300"],
        )

    assert result.exit_code == 0, result.output
    settings = server.call_args.args[0]
    assert settings.port == 7311
    assert settings.bind == "tailscale"
    assert settings.authkey == b"shared-secret"
    assert settings.models == ("arctic-m",)
    assert settings.batch_sizes == {"arctic-m": 32}
    assert settings.idle_grace == 30
    assert settings.turn_length == 300
    server.return_value.serve_forever.assert_called_once_with()


def test_serve_rejects_an_invalid_model_list(monkeypatch):
    monkeypatch.delenv("ARC_EMBED_MODELS", raising=False)
    with patch("arcaneum.cli.remote_embedding.EmbeddingServer") as server:
        result = CliRunner().invoke(cli, ["models", "serve", "--models", "bge-small"])

    assert result.exit_code != 0
    assert "revision" in result.output
    server.assert_not_called()

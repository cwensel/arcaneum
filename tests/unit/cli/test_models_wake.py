"""`arc models wake` wiring, and the wake hint in `arc models workers`."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from arcaneum.cli.main import cli
from arcaneum.embeddings.remote.config import EndpointConfig, RemoteEmbeddingConfig, WakeConfig
from arcaneum.embeddings.remote.pool import EndpointStatus

WAKE = WakeConfig(mac="10:dd:b1:c9:7d:22")


@pytest.fixture
def config(monkeypatch):
    value = RemoteEmbeddingConfig(
        endpoints=[EndpointConfig(host="dev-0", wake=WAKE), EndpointConfig(host="dev-1")]
    )
    monkeypatch.setattr(
        "arcaneum.cli.remote_embedding._load_config_or_exit", lambda output_json: value
    )
    return value


class Calls(list):
    result = True


@pytest.fixture
def wake_calls(monkeypatch):
    recorded = Calls()
    recorded.result = True

    def fake(endpoints, **kwargs):
        recorded.append(([e.host for e in endpoints], kwargs))
        kwargs["out"]("wake done")
        return recorded.result

    monkeypatch.setattr("arcaneum.cli.remote_embedding.wake_endpoints", fake)
    return recorded


def test_wake_passes_named_hosts_and_options(config, wake_calls):
    result = CliRunner().invoke(
        cli, ["models", "wake", "dev-0", "--no-wait", "--dry-run", "--timeout", "30"]
    )

    assert result.exit_code == 0, result.output
    hosts, kwargs = wake_calls[0]
    assert hosts == ["dev-0"]
    assert kwargs["explicit"] is True
    assert kwargs["wait"] is False
    assert kwargs["dry_run"] is True
    assert kwargs["timeout"] == 30
    assert "wake done" in result.output


def test_wake_without_hosts_wakes_every_worker_with_wake_settings(config, wake_calls):
    result = CliRunner().invoke(cli, ["models", "wake"])

    assert result.exit_code == 0, result.output
    hosts, kwargs = wake_calls[0]
    assert hosts == ["dev-0", "dev-1"]
    assert kwargs["explicit"] is False
    assert kwargs["wait"] is True
    assert kwargs["timeout"] is None


def test_wake_exits_nonzero_when_a_host_fails(config, wake_calls):
    wake_calls.result = False

    assert CliRunner().invoke(cli, ["models", "wake"]).exit_code == 1


def test_wake_rejects_an_unknown_host(config, wake_calls):
    result = CliRunner().invoke(cli, ["models", "wake", "dev-9"])

    assert result.exit_code == 1
    assert "dev-9" in result.output
    assert wake_calls == []


def test_wake_without_endpoints_exits_nonzero(monkeypatch, wake_calls):
    monkeypatch.setattr(
        "arcaneum.cli.remote_embedding._load_config_or_exit",
        lambda output_json: RemoteEmbeddingConfig(),
    )

    result = CliRunner().invoke(cli, ["models", "wake"])

    assert result.exit_code == 1
    assert "No remote embedding endpoints" in result.output


def test_workers_suggests_waking_an_unreachable_worker_with_wake_settings(config, monkeypatch):
    monkeypatch.setattr(
        "arcaneum.cli.remote_embedding.probe_endpoint",
        lambda endpoint, config: EndpointStatus(endpoint.name, "unreachable", "timed out"),
    )

    result = CliRunner().invoke(cli, ["models", "workers"])

    assert "arc models wake" in result.output
    assert "dev-0:7300: " in result.output and "wake 10:dd:b1:c9:7d:22" in result.output


def test_workers_json_reports_wake_mac(config, monkeypatch):
    import json

    monkeypatch.setattr(
        "arcaneum.cli.remote_embedding.probe_endpoint",
        lambda endpoint, config: EndpointStatus(endpoint.name, "up"),
    )

    result = CliRunner().invoke(cli, ["models", "workers", "--json"])
    workers = json.loads(result.output)["data"]["workers"]

    assert [w["wake"] for w in workers] == ["10:dd:b1:c9:7d:22", None]

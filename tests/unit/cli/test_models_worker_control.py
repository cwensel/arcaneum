"""`arc models stop` and `arc models start` wiring to WorkerControl."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from arcaneum.cli.main import cli
from arcaneum.embeddings.remote.config import EndpointConfig, RemoteEmbeddingConfig
from arcaneum.embeddings.remote.deploy import DeployError


class FakeControl:
    calls: list[tuple] = []
    result = True
    error: Exception | None = None

    def __init__(self, config, *, dry_run, out):
        self.dry_run = dry_run
        self.out = out

    def _record(self, action, hosts):
        FakeControl.calls.append((action, tuple(hosts), self.dry_run))
        if FakeControl.error is not None:
            raise FakeControl.error
        self.out(f"{action} done")
        return FakeControl.result

    def stop(self, hosts=None):
        return self._record("stop", hosts)

    def start(self, hosts=None):
        return self._record("start", hosts)


@pytest.fixture
def control(monkeypatch):
    FakeControl.calls, FakeControl.result, FakeControl.error = [], True, None
    config = RemoteEmbeddingConfig(endpoints=[EndpointConfig(host="dev-0")])
    monkeypatch.setattr(
        "arcaneum.cli.remote_embedding._load_config_or_exit", lambda output_json: config
    )
    monkeypatch.setattr("arcaneum.cli.remote_embedding.WorkerControl", FakeControl)
    return FakeControl


@pytest.mark.parametrize("action", ["stop", "start"])
def test_command_passes_hosts_and_dry_run(control, action):
    result = CliRunner().invoke(cli, ["models", action, "dev-0", "dev-1", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert control.calls == [(action, ("dev-0", "dev-1"), True)]
    assert f"{action} done" in result.output


@pytest.mark.parametrize("action", ["stop", "start"])
def test_command_without_hosts_acts_on_every_worker(control, action):
    result = CliRunner().invoke(cli, ["models", action])

    assert result.exit_code == 0, result.output
    assert control.calls == [(action, (), False)]


@pytest.mark.parametrize("action", ["stop", "start"])
def test_a_failed_host_exits_nonzero(control, action):
    control.result = False

    assert CliRunner().invoke(cli, ["models", action]).exit_code == 1


def test_unknown_host_is_reported_as_an_error(control):
    control.error = DeployError("No configured endpoint matches 'dev-9'")

    result = CliRunner().invoke(cli, ["models", "stop", "dev-9"])

    assert result.exit_code == 1
    assert "dev-9" in result.output


def test_no_configured_endpoints_exits_nonzero(monkeypatch, control):
    monkeypatch.setattr(
        "arcaneum.cli.remote_embedding._load_config_or_exit",
        lambda output_json: RemoteEmbeddingConfig(),
    )

    result = CliRunner().invoke(cli, ["models", "stop"])

    assert result.exit_code == 1
    assert control.calls == []

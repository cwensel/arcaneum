"""Run-start check of remote workers and auto-deploy on a code mismatch (arcaneum#187)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from arcaneum.embeddings.remote import preflight, routing
from arcaneum.embeddings.remote.config import EndpointConfig, RemoteEmbeddingConfig
from arcaneum.embeddings.remote.deploy import AutoDeployFailures
from arcaneum.embeddings.remote.fingerprint import code_fingerprint
from arcaneum.embeddings.remote.pool import EndpointStatus
from arcaneum.embeddings.remote.routing import RemoteRouter, install_router

OLD = "0" * 64


class FakeProcess:
    def __init__(self, returncode=None) -> None:
        self.returncode = returncode

    def poll(self):
        return self.returncode


class Harness:
    """A router over fake endpoints whose probes, spawns and wakes are recorded."""

    def __init__(self, tmp_path, *hosts, auto_deploy=True, statuses=None, **config) -> None:
        self.config = RemoteEmbeddingConfig(
            endpoints=[EndpointConfig(host=host) for host in hosts], **config
        )
        self.router = RemoteRouter(self.config, auto_deploy=auto_deploy)
        self.statuses = statuses or {}
        self.spawned: list[tuple[str, str, bool]] = []
        self.messages: list[str] = []
        self.deploys: dict = {}
        self.failures = AutoDeployFailures(tmp_path / "failures")
        self.changed = False
        self.buildable = True

    def probe(self, endpoint, config):
        return self.statuses.get(endpoint.name) or mismatch(endpoint.name)

    def spawn(self, endpoint, fingerprint, *, build):
        self.spawned.append((endpoint.name, fingerprint, build))
        return preflight.DeployChild(endpoint.name, FakeProcess(), Path(f"/logs/{endpoint.host}"))

    def run(self, models=("arctic-m",)):
        preflight.run_preflight(
            self.router,
            list(models),
            probe=self.probe,
            spawn=self.spawn,
            wake=lambda *args, **kwargs: True,
            notify=self.messages.append,
            failures=self.failures,
            deploys=self.deploys,
            changed=lambda: self.changed,
            buildable=lambda: self.buildable,
        )


def mismatch(name, *, busy=False):
    return EndpointStatus(
        name,
        "mismatch",
        f"version mismatch: code_fingerprint local {code_fingerprint()[:12]}… remote 000000000000…",
        arcaneum_version="0.12.0",
        code_fingerprint=OLD,
        busy=busy,
        mismatches=["code_fingerprint"],
    )


def up(name):
    return EndpointStatus(
        name, "up", arcaneum_version="0.12.0", code_fingerprint=code_fingerprint()
    )


def test_mismatch_spawns_one_deploy_per_endpoint_with_the_runs_fingerprint(tmp_path):
    harness = Harness(tmp_path, "dev-0", "dev-1")

    harness.run()

    assert harness.spawned == [
        ("dev-0:7300", code_fingerprint(), True),
        ("dev-1:7300", code_fingerprint(), True),
    ]
    assert set(harness.deploys) == {"dev-0:7300", "dev-1:7300"}
    assert any("dev-0:7300" in m and "/logs/dev-0" in m for m in harness.messages)


def test_each_endpoints_state_is_reported_once_up_front(tmp_path):
    harness = Harness(tmp_path, "dev-0", "dev-1", statuses={"dev-1:7300": up("dev-1:7300")})

    harness.run()

    for name in ("dev-0:7300", "dev-1:7300"):
        assert (
            len([m for m in harness.messages if m.startswith(f"Remote embedding worker {name}")])
            == 1
        )
    assert harness.spawned == [("dev-0:7300", code_fingerprint(), True)]


def test_release_install_deploys_the_release_wheel(tmp_path):
    harness = Harness(tmp_path, "dev-0")
    harness.buildable = False

    harness.run()

    assert harness.spawned == [("dev-0:7300", code_fingerprint(), False)]


def test_auto_deploy_off_spawns_nothing_and_names_the_manual_command(tmp_path):
    harness = Harness(tmp_path, "dev-0", auto_deploy=False)

    harness.run()

    assert harness.spawned == []
    assert any("arc models deploy" in m and "auto-deploy" in m.lower() for m in harness.messages)


def test_matching_workers_spawn_nothing(tmp_path):
    harness = Harness(tmp_path, "dev-0", statuses={"dev-0:7300": up("dev-0:7300")})

    harness.run()

    assert harness.spawned == []


def test_unreachable_workers_spawn_nothing(tmp_path):
    status = EndpointStatus("dev-0:7300", "unreachable", "unreachable (timed out)")
    harness = Harness(tmp_path, "dev-0", statuses={"dev-0:7300": status})

    harness.run()

    assert harness.spawned == []
    assert any("dev-0:7300" in m and "timed out" in m for m in harness.messages)


def test_busy_worker_is_not_redeployed(tmp_path):
    harness = Harness(tmp_path, "dev-0", statuses={"dev-0:7300": mismatch("dev-0:7300", busy=True)})

    harness.run()

    assert harness.spawned == []
    assert any("busy" in m for m in harness.messages)


def test_changed_source_tree_skips_the_deploy_with_a_message(tmp_path):
    harness = Harness(tmp_path, "dev-0")
    harness.changed = True

    harness.run()

    assert harness.spawned == []
    assert any("changed" in m for m in harness.messages)


def test_recorded_failure_suppresses_the_retry(tmp_path):
    harness = Harness(tmp_path, "dev-0", "dev-1")
    harness.failures.record("dev-0:7300", code_fingerprint(), "image build failed")

    harness.run()

    assert harness.spawned == [("dev-1:7300", code_fingerprint(), True)]
    assert any("dev-0:7300" in m and "image build failed" in m for m in harness.messages)


def test_a_failure_for_other_code_does_not_suppress(tmp_path):
    harness = Harness(tmp_path, "dev-0")
    harness.failures.record("dev-0:7300", OLD, "image build failed")

    harness.run()

    assert [name for name, _, _ in harness.spawned] == ["dev-0:7300"]


def test_one_deploy_per_endpoint_per_process(tmp_path):
    harness = Harness(tmp_path, "dev-0")

    harness.run()
    harness.run()

    assert len(harness.spawned) == 1


def test_endpoints_serving_none_of_the_runs_models_are_not_checked(tmp_path):
    config = RemoteEmbeddingConfig(
        endpoints=[
            EndpointConfig(host="dev-0", models=["qwen3-embed"]),
            EndpointConfig(host="dev-1"),
        ]
    )
    harness = Harness(tmp_path)
    harness.config = config
    harness.router = RemoteRouter(config)

    harness.run(["arctic-m"])

    assert [name for name, _, _ in harness.spawned] == ["dev-1:7300"]


def test_sleeping_workers_are_woken_before_the_probe(tmp_path):
    from arcaneum.embeddings.remote.config import WakeConfig

    config = RemoteEmbeddingConfig(
        endpoints=[EndpointConfig(host="dev-0", wake=WakeConfig(mac="10:dd:b1:c9:7d:22"))]
    )
    order = []
    preflight.run_preflight(
        RemoteRouter(config),
        ["arctic-m"],
        probe=lambda endpoint, config: order.append("probe") or up(endpoint.name),
        spawn=None,
        wake=lambda endpoints, **kwargs: order.append(("wake", [e.name for e in endpoints])),
        notify=lambda message: None,
        failures=AutoDeployFailures(tmp_path),
        deploys={},
        changed=lambda: False,
        buildable=lambda: True,
    )

    assert order == [("wake", ["dev-0:7300"]), "probe"]


def test_spawned_deploy_is_detached_and_logs_to_a_file(tmp_path):
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    endpoint = EndpointConfig(host="dev-0")
    child = preflight.spawn_deploy(
        endpoint, "f" * 64, build=True, popen=popen, logs=tmp_path / "logs"
    )

    command, kwargs = calls[0]
    assert command[:3] == [sys.executable, "-m", "arcaneum.cli.main"]
    assert command[3:6] == ["models", "deploy", "dev-0:7300"]
    assert command[command.index("--expect-fingerprint") + 1] == "f" * 64
    assert "--build" in command
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert child.log.parent == tmp_path / "logs" and child.log.exists()
    assert child.running


def test_release_deploy_command_omits_build(tmp_path):
    calls = []
    preflight.spawn_deploy(
        EndpointConfig(host="dev-0"),
        "f" * 64,
        build=False,
        popen=lambda command, **kwargs: calls.append(command) or FakeProcess(),
        logs=tmp_path,
    )

    assert "--build" not in calls[0]


# --- configure_remote_embedding starts the check -------------------------------------------


@pytest.fixture
def started(monkeypatch):
    """Record each preflight configure_remote_embedding starts, instead of running it."""
    calls = []
    monkeypatch.setattr(
        routing, "start_preflight", lambda router, models: calls.append((router, models))
    )
    install_router(None)
    yield calls
    install_router(None)


def test_configure_starts_the_check_before_returning(monkeypatch, started):
    monkeypatch.setenv("ARC_EMBED_ENDPOINTS", "dev-0:7300")

    router = routing.configure_remote_embedding("default")

    assert started == [(router, ["arctic-m"])]


def test_reused_router_is_not_checked_again(monkeypatch, started):
    monkeypatch.setenv("ARC_EMBED_ENDPOINTS", "dev-0:7300")

    routing.configure_remote_embedding("default")
    routing.configure_remote_embedding("default")

    assert len(started) == 1


def test_preview_installs_the_router_without_checking_its_workers(monkeypatch, started):
    """A dry run changes nothing, so it must not wake workers or start a deploy."""
    monkeypatch.setenv("ARC_EMBED_ENDPOINTS", "dev-0:7300")

    router = routing.configure_remote_embedding("default", check_workers=False)

    assert routing.active_router() is router
    assert started == []


def test_real_run_after_a_preview_checks_the_reused_router_once(monkeypatch, started):
    monkeypatch.setenv("ARC_EMBED_ENDPOINTS", "dev-0:7300")

    preview = routing.configure_remote_embedding("default", check_workers=False)
    router = routing.configure_remote_embedding("default")
    routing.configure_remote_embedding("default")

    assert router is preview
    assert started == [(router, ["arctic-m"])]


@pytest.mark.parametrize(("mode", "environment"), [("local", "dev-0:7300"), ("default", "")])
def test_runs_that_route_nothing_check_nothing(monkeypatch, started, mode, environment):
    monkeypatch.setenv("ARC_EMBED_ENDPOINTS", environment)

    routing.configure_remote_embedding(mode)

    assert started == []


def test_auto_deploy_follows_configuration_unless_the_run_overrides_it(
    monkeypatch, started, tmp_path
):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("remote_embedding:\n  auto_deploy: false\n")
    monkeypatch.setenv("ARC_EMBED_ENDPOINTS", "dev-0:7300")
    monkeypatch.setattr(
        "arcaneum.embeddings.remote.config._default_config_path", lambda: config_file
    )

    assert routing.configure_remote_embedding("default").auto_deploy is False
    config_file.write_text("remote_embedding: {}\n")
    assert routing.configure_remote_embedding("default").auto_deploy is True
    assert routing.configure_remote_embedding("default", auto_deploy=False).auto_deploy is False


def test_start_preflight_runs_in_a_daemon_thread(monkeypatch):
    ran = []
    monkeypatch.setattr(preflight, "run_preflight", lambda router, models: ran.append(models))
    router = RemoteRouter(RemoteEmbeddingConfig(endpoints=[EndpointConfig(host="dev-0")]))

    thread = preflight.start_preflight(router, ["arctic-m"])
    thread.join(5)

    assert thread.daemon and ran == [["arctic-m"]]

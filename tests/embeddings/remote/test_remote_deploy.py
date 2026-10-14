"""`arc models deploy` against fake Docker hosts (RDR-025 Step 6)."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from arcaneum.embeddings.remote.config import EndpointConfig, RemoteEmbeddingConfig
from arcaneum.embeddings.remote.deploy import (
    CHANGING_COMMANDS,
    CommandResult,
    Deployer,
    DeployError,
    VerifyResult,
    WheelInfo,
    container_name,
    image_tag,
    render_dockerfile,
    resolve_wheel,
)
from arcaneum.embeddings.remote.fingerprint import code_fingerprint
from arcaneum.embeddings.remote.pool import EndpointStatus

FINGERPRINT = "c" * 64
OLD = "d" * 64


class FakeDockerHost:
    """Just enough of `docker` over ssh to exercise deploy."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.commands: list[list[str]] = []
        self.copies: list[tuple[str, str]] = []
        self.images: dict[str, dict] = {}  # id -> {"tags": set, "labels": dict}
        self.containers: dict[str, dict] = {}
        self.fail_build = False
        self.reachable = True
        self._ids = 0

    # -- helpers for tests --
    def add_image(self, fingerprint: str) -> str:
        self._ids += 1
        image_id = f"sha256:{self.name}{self._ids:04d}"
        self.images[image_id] = {
            "tags": {image_tag(fingerprint)},
            "labels": {"arcaneum.role": "embed-worker", "arcaneum.fingerprint": fingerprint},
            "created": self._created(),
        }
        return image_id

    def add_container(self, port, fingerprint, spec, running=True):
        image = image_tag(fingerprint)
        if not any(image in i["tags"] for i in self.images.values()):
            self.add_image(fingerprint)
        self.containers[container_name(port)] = {
            "running": running,
            "image": image,
            "labels": {"arcaneum.fingerprint": fingerprint, "arcaneum.spec": spec},
            "args": [],
        }

    def changing(self) -> list[list[str]]:
        return [
            c
            for c in self.commands
            if tuple(c[:2]) in CHANGING_COMMANDS
            or c[:3] == ["docker", "image", "rm"]
            or c[0] == "rm"
        ]

    # -- runner protocol --
    def run(self, args, *, timeout=None, check=True) -> CommandResult:
        self.commands.append(list(args))
        result = self._dispatch(list(args))
        if check and result.returncode != 0:
            raise DeployError(f"{' '.join(args[:3])} failed: {result.stderr}")
        return result

    def _created(self) -> str:
        return f"2026-10-14 16:{self._ids // 60:02d}:{self._ids % 60:02d} +0000 UTC"

    def copy(self, local: Path, remote: str) -> None:
        self.copies.append((Path(local).name, remote))

    def _dispatch(self, args) -> CommandResult:
        ok = CommandResult(0, "", "")
        if not self.reachable:
            return CommandResult(255, "", "ssh: connect to host: Connection refused")
        if args[:2] == ["docker", "version"]:
            return CommandResult(0, "29.8.1\n", "")
        if args[:3] == ["docker", "container", "inspect"]:
            container = self.containers.get(args[3])
            if container is None:
                return CommandResult(1, "", f"Error: No such container: {args[3]}")
            info = {
                "State": {"Running": container["running"], "Status": "running"},
                "Config": {"Labels": container["labels"], "Image": container["image"]},
            }
            return CommandResult(0, json.dumps([info]), "")
        if args[:3] == ["docker", "image", "ls"]:
            lines = []
            for image_id, image in self.images.items():
                tags = sorted(image["tags"]) or ["<none>:<none>"]
                for tag in tags:
                    repository, _, tag_name = tag.partition(":")
                    lines.append(
                        json.dumps(
                            {
                                "ID": image_id,
                                "Repository": repository,
                                "Tag": tag_name,
                                "CreatedAt": image["created"],
                            }
                        )
                    )
            return CommandResult(0, "\n".join(lines), "")
        if args[:2] == ["docker", "build"]:
            if self.fail_build:
                return CommandResult(1, "", "pip install failed")
            tag = args[args.index("-t") + 1]
            labels = dict(args[i + 1].split("=", 1) for i, a in enumerate(args) if a == "--label")
            for image in self.images.values():
                image["tags"].discard(tag)
            self._ids += 1
            self.images[f"sha256:{self.name}{self._ids:04d}"] = {
                "tags": {tag},
                "labels": labels,
                "created": self._created(),
            }
            return ok
        if args[:2] == ["docker", "rm"]:
            self.containers.pop(args[-1], None)
            return ok
        if args[:2] == ["docker", "run"]:
            name = args[args.index("--name") + 1]
            labels = dict(args[i + 1].split("=", 1) for i, a in enumerate(args) if a == "--label")
            self.containers[name] = {
                "running": True,
                "image": args[-1],
                "labels": labels,
                "args": list(args),
            }
            return ok
        if args[:2] == ["docker", "start"]:
            self.containers[args[2]]["running"] = True
            return ok
        if args[:2] == ["docker", "logs"]:
            return CommandResult(0, "Waiting for the Tailscale address\n", "")
        if args[:3] == ["docker", "image", "rm"]:
            self.images.pop(args[3], None)
            return ok
        if args[:2] == ["mktemp", "-d"]:
            return CommandResult(0, "/tmp/arcaneum-embed-build.abc123\n", "")
        if args[:2] == ["rm", "-rf"]:
            return ok
        raise AssertionError(f"unexpected command {args}")


def make_wheel(tmp_path: Path, fingerprint=FINGERPRINT) -> WheelInfo:
    path = tmp_path / "arcaneum-0.11.0-py3-none-any.whl"
    path.write_bytes(b"wheel")
    return WheelInfo(path, fingerprint, "0.11.0", hashlib.sha256(b"wheel").hexdigest())


class Harness:
    def __init__(self, tmp_path, endpoints, **options):
        self.hosts = {e.ssh_target: FakeDockerHost(e.ssh_target) for e in endpoints}
        self.config = RemoteEmbeddingConfig(endpoints=endpoints, authkey="secret")
        self.verified: list[tuple[str, list[str]]] = []
        self.verify_results: dict[str, list[VerifyResult]] = {}
        self.unresponsive: set[str] = set()
        self.output: list[str] = []
        self.wheel = options.pop("wheel", None) or make_wheel(tmp_path)
        self.deployer = Deployer(
            self.config,
            self.wheel,
            runner_factory=lambda target: self.hosts[target],
            probe=self.probe,
            verifier=self.verify,
            out=self.output.append,
            sleep=lambda seconds: None,
            health_timeout=5,
            **options,
        )

    def probe(self, endpoint, config):
        host = self.hosts[endpoint.ssh_target]
        container = host.containers.get(container_name(endpoint.port))
        if (
            not host.reachable
            or container is None
            or not container["running"]
            or endpoint.name in self.unresponsive
        ):
            return EndpointStatus(endpoint.name, "unreachable", "unreachable (refused)")
        return EndpointStatus(
            endpoint.name, "up", code_fingerprint=container["labels"]["arcaneum.fingerprint"]
        )

    def verify(self, endpoint, config, models):
        self.verified.append((endpoint.name, list(models)))
        return self.verify_results.get(
            endpoint.name, [VerifyResult(model, 1.0, {}, {}) for model in models]
        )


def endpoints(*hosts):
    return [EndpointConfig(host=host, bind="tailscale") for host in hosts]


def test_first_install_builds_the_image_and_creates_a_verified_container(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0"))
    host = harness.hosts["dev-0"]

    assert harness.deployer.deploy() is True

    builds = [c for c in host.commands if c[:2] == ["docker", "build"]]
    assert len(builds) == 1 and "--pull" not in builds[0]
    assert image_tag(FINGERPRINT) in builds[0]
    assert {name for name, _ in host.copies} == {"arcaneum-0.11.0-py3-none-any.whl", "Dockerfile"}
    container = host.containers[container_name(7300)]
    assert container["running"]
    assert container["labels"]["arcaneum.fingerprint"] == FINGERPRINT
    assert harness.verified == [("dev-0:7300", ["qwen3-embed", "arctic-m"])]


def test_container_uses_host_networking_and_publishes_no_ports(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0"))
    harness.deployer.deploy()

    args = harness.hosts["dev-0"].containers[container_name(7300)]["args"]
    assert args[args.index("--network") + 1] == "host"
    assert "-p" not in args and "--publish" not in args
    assert args[args.index("--restart") + 1] == "unless-stopped"
    env = [args[i + 1] for i, a in enumerate(args) if a == "-e"]
    assert "ARC_EMBED_BIND=tailscale" in env
    assert "ARC_EMBED_PORT=7300" in env
    assert "ARC_EMBED_AUTHKEY=secret" in env
    assert any(a.startswith("arcaneum-embed-models:") for a in args)


def test_second_deploy_reports_up_to_date_without_changes(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0"))
    harness.deployer.deploy()
    host = harness.hosts["dev-0"]
    host.commands.clear()
    harness.verified.clear()

    assert harness.deployer.deploy() is True

    assert host.changing() == []
    assert harness.verified == []
    assert any("up to date" in line for line in harness.output)


def test_reconfigure_replaces_the_container_and_reuses_the_image(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0"))
    harness.deployer.deploy()
    host = harness.hosts["dev-0"]
    host.commands.clear()
    harness.config.endpoints[0].batch_size = {"arctic-m": 16}

    assert harness.deployer.deploy() is True

    assert not [c for c in host.commands if c[:2] == ["docker", "build"]]
    assert ["docker", "rm", "-f", "-v", container_name(7300)] in host.commands
    args = host.containers[container_name(7300)]["args"]
    assert "ARC_EMBED_BATCH_SIZES=arctic-m=16" in args


def test_refresh_rebuilds_without_cache_and_prunes_the_superseded_image(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0"))
    harness.deployer.deploy()
    host = harness.hosts["dev-0"]
    first_ids = set(host.images)
    harness.deployer.refresh = True

    assert harness.deployer.deploy() is True

    build = [c for c in host.commands if c[:2] == ["docker", "build"]][-1]
    assert "--pull" in build and "--no-cache" in build
    assert not first_ids & set(host.images)
    assert len(host.images) == 1


def test_refresh_after_an_upgrade_keeps_the_previous_fingerprints_image(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0"))
    host = harness.hosts["dev-0"]
    host.add_image("e" * 64)
    host.add_container(7300, OLD, "old-spec")
    harness.deployer.deploy()
    harness.deployer.refresh = True

    assert harness.deployer.deploy() is True

    tags = {tag for image in host.images.values() for tag in image["tags"]}
    assert tags == {image_tag(FINGERPRINT), image_tag(OLD)}
    assert len(host.images) == 2


def test_upgrade_keeps_the_previous_image_and_removes_older_ones(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0"))
    host = harness.hosts["dev-0"]
    ancient = host.add_image("e" * 64)
    host.add_container(7300, OLD, "old-spec")

    assert harness.deployer.deploy() is True

    tags = {tag for image in host.images.values() for tag in image["tags"]}
    assert tags == {image_tag(FINGERPRINT), image_tag(OLD)}
    assert ancient not in host.images


def test_stopped_container_is_started_without_verification(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0"))
    harness.deployer.deploy()
    host = harness.hosts["dev-0"]
    host.containers[container_name(7300)]["running"] = False
    host.commands.clear()
    harness.verified.clear()

    assert harness.deployer.deploy() is True

    assert ["docker", "start", container_name(7300)] in host.commands
    assert harness.verified == []


def test_unresponsive_container_is_replaced(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0"))
    harness.deployer.deploy()
    harness.unresponsive.add("dev-0:7300")
    host = harness.hosts["dev-0"]
    host.commands.clear()
    original_run = host.run

    def run(args, **kwargs):
        if args[:2] == ["docker", "run"]:
            harness.unresponsive.discard("dev-0:7300")
        return original_run(args, **kwargs)

    host.run = run
    assert harness.deployer.deploy() is True
    assert ["docker", "rm", "-f", "-v", container_name(7300)] in host.commands


def test_dry_run_issues_no_changing_command(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0", "dev-1"), dry_run=True)
    harness.hosts["dev-1"].add_container(7300, OLD, "old-spec")

    harness.deployer.deploy()

    for host in harness.hosts.values():
        assert host.changing() == []
        assert host.copies == []
    text = "\n".join(harness.output)
    assert "dev-0:7300" in text and "dev-1:7300" in text
    assert "build image, create container" in text
    assert "build image, replace container" in text
    assert OLD[:12] in text and FINGERPRINT[:12] in text


def test_failed_build_leaves_that_hosts_worker_running_and_deploys_the_others(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0", "dev-1"))
    broken = harness.hosts["dev-0"]
    broken.add_container(7300, OLD, "old-spec")
    broken.fail_build = True

    assert harness.deployer.deploy() is False

    assert broken.containers[container_name(7300)]["labels"]["arcaneum.fingerprint"] == OLD
    assert not [c for c in broken.commands if c[:2] in (["docker", "rm"], ["docker", "run"])]
    healthy = harness.hosts["dev-1"].containers[container_name(7300)]
    assert healthy["labels"]["arcaneum.fingerprint"] == FINGERPRINT
    assert any("dev-0" in line and "build" in line for line in harness.output)


def test_failed_verification_removes_the_container_and_keeps_the_image(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0"))
    harness.verify_results["dev-0:7300"] = [
        VerifyResult("qwen3-embed", 1.0, {}, {}),
        VerifyResult(
            "arctic-m",
            0.99664,
            {"onnxruntime": "1.31.0"},
            {"onnxruntime": "1.28.0"},
        ),
    ]
    host = harness.hosts["dev-0"]

    assert harness.deployer.deploy() is False

    assert container_name(7300) not in host.containers
    assert any(image_tag(FINGERPRINT) in i["tags"] for i in host.images.values())
    text = "\n".join(harness.output)
    assert "arctic-m" in text and "0.99664" in text and "1.31.0" in text and "1.28.0" in text

    # The next deploy recreates and re-verifies rather than restarting.
    harness.verify_results.clear()
    harness.verified.clear()
    host.commands.clear()
    assert harness.deployer.deploy() is True
    assert not [c for c in host.commands if c[:2] == ["docker", "build"]]
    assert [c for c in host.commands if c[:2] == ["docker", "run"]]
    assert harness.verified == [("dev-0:7300", ["qwen3-embed", "arctic-m"])]


def test_successful_verification_reports_each_models_minimum_cosine(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0"))
    harness.verify_results["dev-0:7300"] = [
        VerifyResult("qwen3-embed", 0.9999987, {}, {}),
        VerifyResult("arctic-m", 1.0, {}, {}),
    ]

    assert harness.deployer.deploy() is True

    text = "\n".join(harness.output)
    assert "qwen3-embed min cosine 0.999999" in text
    assert "arctic-m min cosine 1.000000" in text


def test_no_verify_skips_verification(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0"), verify=False)

    assert harness.deployer.deploy() is True
    assert harness.verified == []


def test_health_wait_timeout_prints_the_container_log(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0"))
    harness.unresponsive.add("dev-0:7300")

    assert harness.deployer.deploy() is False

    assert ["docker", "logs", "--tail", "40", container_name(7300)] in harness.hosts[
        "dev-0"
    ].commands
    assert any("Waiting for the Tailscale address" in line for line in harness.output)


def test_unreachable_ssh_host_is_reported_and_skipped(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0", "dev-1"))
    harness.hosts["dev-0"].reachable = False

    assert harness.deployer.deploy() is False

    assert harness.hosts["dev-1"].containers
    assert any("dev-0" in line and "preflight" in line for line in harness.output)


def test_unresponsive_tailscale_worker_outside_the_tailnet_is_reported_not_replaced(
    tmp_path, monkeypatch
):
    harness = Harness(tmp_path, endpoints("dev-0"))
    harness.deployer.deploy()
    harness.unresponsive.add("dev-0:7300")
    host = harness.hosts["dev-0"]
    host.commands.clear()
    monkeypatch.setattr(
        "arcaneum.embeddings.remote.deploy.tailscale_resolution_problem",
        lambda endpoint: "dev-0 resolves to 192.168.1.5, outside the tailnet",
    )

    assert harness.deployer.deploy() is False

    assert host.changing() == []
    assert any("outside the tailnet" in line for line in harness.output)


def test_hosts_can_be_selected_by_name(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0", "dev-1"))

    assert harness.deployer.deploy(["dev-1"]) is True

    assert harness.hosts["dev-0"].commands == []
    assert harness.hosts["dev-1"].containers


def test_unknown_host_selection_is_an_error(tmp_path):
    harness = Harness(tmp_path, endpoints("dev-0"))

    with pytest.raises(DeployError, match="dev-9"):
        harness.deployer.deploy(["dev-9"])


def test_dockerfile_installs_cpu_torch_in_its_own_layer_before_the_wheel():
    text = render_dockerfile("arcaneum-0.11.0-py3-none-any.whl")

    lines = text.splitlines()
    assert lines[0] == "FROM python:3.12-slim"
    torch = next(i for i, line in enumerate(lines) if "download.pytorch.org/whl/cpu" in line)
    wheel = next(i for i, line in enumerate(lines) if "arcaneum-0.11.0-py3-none-any.whl[" in line)
    assert torch < wheel
    assert "sentence-transformers" in lines[wheel]
    assert lines[-1] == 'CMD ["arc", "models", "serve"]'


# --- Wheel resolution (Testing Strategy 14) -------------------------------------------


def wheel_with(tmp_path: Path, name: str, sources: dict[str, str], requires: list[str]) -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as wheel:
        for source, text in sources.items():
            wheel.writestr(f"arcaneum/{source}", text)
        meta = ["Metadata-Version: 2.4", "Name: arcaneum", "Version: 0.10.0"]
        meta += [f"Requires-Dist: {r}" for r in requires]
        wheel.writestr("arcaneum-0.10.0.dist-info/METADATA", "\n".join(meta) + "\n")
    return path


def test_release_wheel_of_a_modified_workstation_is_refused_naming_build(tmp_path):
    stale = wheel_with(tmp_path, "arcaneum-0.10.0-py3-none-any.whl", {"__init__.py": "x"}, [])

    with pytest.raises(DeployError, match="--build"):
        resolve_wheel(downloader=lambda version: stale)


def test_explicit_wheel_must_match_the_workstation(tmp_path):
    other = wheel_with(tmp_path, "arcaneum-0.10.0-py3-none-any.whl", {"__init__.py": "y"}, [])

    with pytest.raises(DeployError, match="fingerprint"):
        resolve_wheel(wheel=other)


def test_missing_release_wheel_names_build(tmp_path):
    def missing(version):
        raise DeployError(f"No release wheel for arcaneum {version}")

    with pytest.raises(DeployError, match="--build"):
        resolve_wheel(downloader=missing)


@pytest.mark.slow
def test_built_wheels_match_the_installed_tree_and_each_other(tmp_path):
    first = resolve_wheel(build=True, build_dir=tmp_path / "a")
    second = resolve_wheel(build=True, build_dir=tmp_path / "b")

    assert first.fingerprint == second.fingerprint == code_fingerprint()
    # Wheels built at different times may differ byte for byte; only the
    # fingerprint identifies the image.
    assert first.path.name == second.path.name

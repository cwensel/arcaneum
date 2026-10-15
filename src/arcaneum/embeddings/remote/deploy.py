"""`arc models deploy`: bring each worker to the workstation's code (RDR-025).

The desired state of a worker has three parts: a wheel with the workstation's code
fingerprint, an image built on the worker from that wheel (tagged and labeled by
fingerprint), and a container running ``arc models serve`` with host networking and
a specification hash label. Deploy compares that with the worker's current
container and images over SSH and takes the one action the lifecycle table names,
building before replacing so a failed build leaves the running worker untouched,
then waits for the new worker to answer and checks vector parity before the pool
can use it.
"""

from __future__ import annotations

import hashlib
import json
import math
import shlex
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from email.parser import HeaderParser
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

import arcaneum
from arcaneum.embeddings.remote.config import EndpointConfig, RemoteEmbeddingConfig
from arcaneum.embeddings.remote.fingerprint import code_fingerprint, wheel_fingerprint
from arcaneum.embeddings.remote.pool import (
    EndpointStatus,
    probe_endpoint,
    tailscale_resolution_problem,
)

IMAGE_REPOSITORY = "arcaneum-embed"
MODEL_VOLUME = "arcaneum-embed-models"
CACHE_MOUNT = "/cache"
ROLE_LABEL = "arcaneum.role=embed-worker"
RELEASE_URL = "https://github.com/cwensel/arcaneum/releases/download/v{version}/{name}"
PARITY_THRESHOLD = 0.9999
HEALTH_POLL = 2.0
BUILD_TIMEOUT = 1800.0

# Remote commands that change a worker; --dry-run issues none of them.
CHANGING_COMMANDS = frozenset(
    {
        ("docker", "build"),
        ("docker", "rm"),
        ("docker", "run"),
        ("docker", "start"),
        ("docker", "stop"),
        ("mktemp", "-d"),
    }
)


class DeployError(RuntimeError):
    """One step of deploying a worker failed; the message names the step."""


def container_name(port: int) -> str:
    return f"{IMAGE_REPOSITORY}-{port}"


def image_tag(fingerprint: str) -> str:
    return f"{IMAGE_REPOSITORY}:{fingerprint[:12]}"


# --- Wheels --------------------------------------------------------------------------


@dataclass(frozen=True)
class WheelInfo:
    path: Path
    fingerprint: str
    version: str
    sha256: str


def _wheel_version(path: Path) -> str:
    with zipfile.ZipFile(path) as wheel:
        names = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
        if names:
            headers = HeaderParser().parsestr(wheel.read(names[0]).decode())
            if headers.get("Version"):
                return headers["Version"]
    return path.name.split("-")[1]


def _wheel_cache_dir() -> Path:
    from arcaneum.paths import get_models_dir

    directory = get_models_dir().parent / "wheels"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def download_release_wheel(version: str) -> Path:
    """Fetch the GitHub release wheel for ``version`` into arcaneum's cache."""
    import requests

    name = f"arcaneum-{version}-py3-none-any.whl"
    target = _wheel_cache_dir() / name
    if target.exists():
        return target
    url = RELEASE_URL.format(version=version, name=name)
    try:
        response = requests.get(url, timeout=60)
    except requests.RequestException as exc:
        raise DeployError(f"Could not download the release wheel {url}: {exc}") from exc
    if response.status_code != 200:
        raise DeployError(
            f"No release wheel for arcaneum {version} at {url} (HTTP {response.status_code})"
        )
    partial = target.with_suffix(".partial")
    partial.write_bytes(response.content)
    partial.replace(target)
    return target


def source_tree() -> Path:
    """The source checkout the running arcaneum was imported from."""
    package = Path(arcaneum.__file__).resolve().parent
    root = package.parent.parent
    if package.parent.name != "src" or not (root / "pyproject.toml").is_file():
        raise DeployError(
            f"--build needs arcaneum imported from a source tree, but it was imported "
            f"from {package}; use --wheel PATH or deploy a release"
        )
    return root


def build_wheel(out_dir: Path) -> Path:
    """Build a wheel from the source tree with ``uv build --wheel``."""
    uv = shutil.which("uv")
    if uv is None:
        raise DeployError("--build needs `uv` on PATH")
    out_dir.mkdir(parents=True, exist_ok=True)
    root = source_tree()
    result = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(out_dir), str(root)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise DeployError(f"uv build failed: {result.stderr.strip()[-2000:]}")
    wheels = sorted(out_dir.glob("arcaneum-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise DeployError(f"uv build produced no wheel in {out_dir}")
    return wheels[-1]


def resolve_wheel(
    *,
    build: bool = False,
    wheel: Optional[Path] = None,
    downloader: Callable[[str], Path] = download_release_wheel,
    builder: Callable[[Path], Path] = build_wheel,
    build_dir: Optional[Path] = None,
) -> WheelInfo:
    """Pick the wheel to deploy and refuse one whose code differs from the workstation's."""
    local = code_fingerprint()
    if wheel is not None:
        path, source = Path(wheel), f"wheel {wheel}"
    elif build:
        path = builder(Path(build_dir or tempfile.mkdtemp(prefix="arcaneum-wheel-")))
        source = f"wheel built from {source_tree()}"
    else:
        try:
            path = downloader(arcaneum.__version__)
        except DeployError as exc:
            raise DeployError(
                f"{exc}. If this workstation runs an unreleased or modified build, "
                "deploy it with --build"
            ) from exc
        source = f"release wheel for arcaneum {arcaneum.__version__}"
    fingerprint = wheel_fingerprint(path)
    if fingerprint != local:
        hint = (
            "the workstation runs an unreleased or modified build; deploy its source tree "
            "with --build"
            if wheel is None and not build
            else (
                "workers must run exactly the workstation's code"
                if wheel is not None
                else "reinstall the source tree (e.g. `uv sync`) so the installed code matches it"
            )
        )
        raise DeployError(
            f"The {source} has code fingerprint {fingerprint[:12]}, but this workstation's "
            f"is {local[:12]}: {hint}"
        )
    return WheelInfo(
        path=path,
        fingerprint=fingerprint,
        version=_wheel_version(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


# --- Image and container ------------------------------------------------------------


def render_dockerfile(wheel_name: str) -> str:
    """Dockerfile for a worker image; CPU torch sits in its own cached layer."""
    return "\n".join(
        [
            "FROM python:3.12-slim",
            "ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 XDG_CACHE_HOME=" + CACHE_MOUNT,
            "RUN pip install --index-url https://download.pytorch.org/whl/cpu torch",
            f"COPY {wheel_name} /tmp/{wheel_name}",
            f'RUN pip install "/tmp/{wheel_name}[sentence-transformers]" '
            f"&& rm /tmp/{wheel_name}",
            'CMD ["arc", "models", "serve"]',
        ]
    )


def container_environment(
    endpoint: EndpointConfig, config: RemoteEmbeddingConfig
) -> dict[str, str]:
    env = {
        "ARC_EMBED_PORT": str(endpoint.port),
        "ARC_EMBED_BIND": endpoint.bind,
        "ARC_EMBED_MODELS": ",".join(config.endpoint_models(endpoint)),
        "ARC_EMBED_IDLE_GRACE": f"{config.idle_grace:g}",
        "ARC_EMBED_TURN_LENGTH": f"{config.turn_length:g}",
        "XDG_CACHE_HOME": CACHE_MOUNT,
    }
    if endpoint.batch_size:
        env["ARC_EMBED_BATCH_SIZES"] = ",".join(
            f"{model}={size}" for model, size in sorted(endpoint.batch_size.items())
        )
    if config.authkey:
        env["ARC_EMBED_AUTHKEY"] = config.authkey
    return env


def spec_hash(endpoint: EndpointConfig, config: RemoteEmbeddingConfig) -> str:
    """Hash of the full run specification apart from the image (fingerprint covers it)."""
    spec = {
        "name": container_name(endpoint.port),
        "network": "host",
        "restart": "unless-stopped",
        "cpu_shares": endpoint.cpu_shares,
        "volume": f"{MODEL_VOLUME}:{CACHE_MOUNT}",
        "env": container_environment(endpoint, config),
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def run_arguments(
    endpoint: EndpointConfig, config: RemoteEmbeddingConfig, wheel: WheelInfo
) -> list[str]:
    args = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name(endpoint.port),
        "--restart",
        "unless-stopped",
        "--cpu-shares",
        str(endpoint.cpu_shares),
        "--network",
        "host",
        "-v",
        f"{MODEL_VOLUME}:{CACHE_MOUNT}",
        "--label",
        ROLE_LABEL,
        "--label",
        f"arcaneum.version={wheel.version}",
        "--label",
        f"arcaneum.fingerprint={wheel.fingerprint}",
        "--label",
        f"arcaneum.spec={spec_hash(endpoint, config)}",
    ]
    for key, value in container_environment(endpoint, config).items():
        args += ["-e", f"{key}={value}"]
    args.append(image_tag(wheel.fingerprint))
    return args


# --- Planning ------------------------------------------------------------------------


@dataclass(frozen=True)
class Desired:
    fingerprint: str
    version: str
    spec_hash: str


@dataclass(frozen=True)
class ContainerState:
    running: bool
    fingerprint: Optional[str]
    spec_hash: Optional[str]
    image: Optional[str]
    version: Optional[str] = None


@dataclass
class Plan:
    build: bool
    container: str  # none | start | create | replace
    verify: bool
    refresh: bool = False
    report: Optional[str] = None

    def describe(self) -> str:
        if self.report:
            return f"report: {self.report}"
        if self.container == "none" and not self.build:
            return "none (up to date)"
        parts = []
        if self.build:
            parts.append("rebuild image (--pull --no-cache)" if self.refresh else "build image")
        elif self.container in ("create", "replace"):
            parts.append("reuse image")
        if self.container != "none":
            parts.append(f"{self.container} container")
        return ", ".join(parts)


def plan_host(
    container: Optional[ContainerState],
    images: set[str],
    status: Optional[EndpointStatus],
    desired: Desired,
    *,
    refresh: bool,
    resolution_problem: Optional[str] = None,
) -> Plan:
    """Map a worker's current state to the lifecycle table's action."""
    build = refresh or desired.fingerprint not in images
    if container is None:
        action = "create"
    elif (
        refresh
        or container.fingerprint != desired.fingerprint
        or container.spec_hash != desired.spec_hash
    ):
        action = "replace"
    elif not container.running:
        action = "start"
    elif status is None or status.state != "up" or status.code_fingerprint != desired.fingerprint:
        if resolution_problem:
            # The workstation cannot reach it by name; replacing would not help.
            return Plan(False, "none", False, report=resolution_problem)
        action = "replace"
    else:
        action = "none"
    return Plan(build, action, verify=action in ("create", "replace"), refresh=refresh)


# --- Remote execution ----------------------------------------------------------------


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class RemoteRunner(Protocol):
    def run(
        self, args: list[str], *, timeout: Optional[float] = None, check: bool = True
    ) -> CommandResult: ...

    def copy(self, local: Path, remote: str) -> None: ...


SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


class SshRunner:
    """Runs commands on a worker with the system ``ssh`` and ``scp`` in batch mode."""

    def __init__(self, target: str) -> None:
        self.target = target

    def run(
        self, args: list[str], *, timeout: Optional[float] = 120.0, check: bool = True
    ) -> CommandResult:
        command = ["ssh", *SSH_OPTIONS, self.target, shlex.join(args)]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise DeployError(f"`{shlex.join(args[:3])}` on {self.target} timed out") from exc
        result = CommandResult(completed.returncode, completed.stdout, completed.stderr)
        if check and result.returncode != 0:
            raise DeployError(
                f"`{shlex.join(args[:3])}` on {self.target} failed "
                f"(exit {result.returncode}): {result.stderr.strip()[-1500:]}"
            )
        return result

    def copy(self, local: Path, remote: str) -> None:
        completed = subprocess.run(
            ["scp", "-q", *SSH_OPTIONS, str(local), f"{self.target}:{remote}"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if completed.returncode != 0:
            raise DeployError(f"copying {local.name} to {self.target} failed: {completed.stderr}")


# --- Verification ----------------------------------------------------------------------


@dataclass
class VerifyResult:
    model: str
    min_cosine: float
    remote_dependencies: dict = field(default_factory=dict)
    local_dependencies: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.min_cosine >= PARITY_THRESHOLD


def _row_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    norms = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return (left * right).sum(axis=1) / norms


def verify_endpoint(
    endpoint: EndpointConfig, config: RemoteEmbeddingConfig, models: Iterable[str]
) -> list[VerifyResult]:
    """Embed the probe set remotely and on the workstation's CPU path, one model at a time.

    The first remote encode downloads and loads the pinned revision, warming the
    worker's cache; each model switch waits out the worker's idle grace period.
    """
    from arcaneum.embeddings.client import EmbeddingClient
    from arcaneum.embeddings.remote.identity import dependency_versions
    from arcaneum.embeddings.remote.pool import RemoteEmbeddingPool
    from arcaneum.embeddings.remote.probe_set import PROBE_TEXTS

    single = config.model_copy(update={"endpoints": [endpoint], "unavailable": "fail"})
    pool = RemoteEmbeddingPool(single)
    local = EmbeddingClient(use_gpu=False, remote_routing=False)
    texts = list(PROBE_TEXTS)
    results = []
    try:
        for model in models:
            remote_vectors = pool.embed(texts, model)
            local_vectors = np.asarray(local.embed(texts, model), dtype=np.float32)
            status = probe_endpoint(endpoint, config)
            results.append(
                VerifyResult(
                    model,
                    float(_row_cosines(remote_vectors, local_vectors).min()),
                    status.dependencies,
                    dependency_versions(),
                )
            )
            local.unload_model(model)
    finally:
        pool.close()
    return results


# --- Deploy --------------------------------------------------------------------------


def _parse_container(stdout: str) -> Optional[ContainerState]:
    try:
        info = json.loads(stdout)[0]
    except (ValueError, IndexError, TypeError):
        return None
    labels = (info.get("Config") or {}).get("Labels") or {}
    return ContainerState(
        running=bool((info.get("State") or {}).get("Running")),
        fingerprint=labels.get("arcaneum.fingerprint"),
        spec_hash=labels.get("arcaneum.spec"),
        image=(info.get("Config") or {}).get("Image"),
        version=labels.get("arcaneum.version"),
    )


def _parse_images(stdout: str) -> list[tuple[str, str, str]]:
    """(image ID, repository:tag, created) for every worker image, including untagged ones."""
    images = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        images.append(
            (
                entry.get("ID", ""),
                f"{entry.get('Repository')}:{entry.get('Tag')}",
                entry.get("CreatedAt", ""),
            )
        )
    return images


class Deployer:
    def __init__(
        self,
        config: RemoteEmbeddingConfig,
        wheel: WheelInfo,
        *,
        dry_run: bool = False,
        refresh: bool = False,
        verify: bool = True,
        runner_factory: Callable[[str], RemoteRunner] = SshRunner,
        probe: Callable[[EndpointConfig, RemoteEmbeddingConfig], EndpointStatus] = probe_endpoint,
        verifier: Callable[..., list[VerifyResult]] = verify_endpoint,
        out: Callable[[str], None] = print,
        sleep: Callable[[float], None] = None,
        health_timeout: float = 180.0,
    ) -> None:
        import time

        self.config = config
        self.wheel = wheel
        self.dry_run = dry_run
        self.refresh = refresh
        self.verify = verify
        self.runner_factory = runner_factory
        self.probe = probe
        self.verifier = verifier
        self.out = out
        self.sleep = sleep or time.sleep
        self.health_timeout = health_timeout

    def select(self, hosts: Optional[Iterable[str]]) -> list[EndpointConfig]:
        if not hosts:
            return list(self.config.endpoints)
        selected = []
        for name in hosts:
            matches = [e for e in self.config.endpoints if name in (e.host, e.name, e.ssh_target)]
            if not matches:
                known = ", ".join(e.name for e in self.config.endpoints) or "none"
                raise DeployError(f"No configured endpoint matches '{name}' (configured: {known})")
            selected.extend(e for e in matches if e not in selected)
        return selected

    def deploy(self, hosts: Optional[Iterable[str]] = None) -> bool:
        """Deploy each selected endpoint in turn; returns False if any host failed."""
        ok = True
        for endpoint in self.select(hosts):
            try:
                self._deploy_endpoint(endpoint)
            except DeployError as exc:
                ok = False
                self.out(f"✗ {endpoint.name}: {exc}")
        return ok

    def _deploy_endpoint(self, endpoint: EndpointConfig) -> None:
        runner = self.runner_factory(endpoint.ssh_target)
        name = container_name(endpoint.port)
        try:
            runner.run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=60)
        except DeployError as exc:
            raise DeployError(f"preflight failed on {endpoint.ssh_target}: {exc}") from exc
        inspected = runner.run(["docker", "container", "inspect", name], check=False)
        container = _parse_container(inspected.stdout) if inspected.returncode == 0 else None
        images = self._images(runner)
        tags = {tag for _, tag, _ in images}
        status = self.probe(endpoint, self.config) if container and container.running else None
        resolution = (
            tailscale_resolution_problem(endpoint)
            if status is not None and status.state != "up"
            else None
        )
        desired = Desired(
            self.wheel.fingerprint, self.wheel.version, spec_hash(endpoint, self.config)
        )
        available = {
            fp
            for fp in (desired.fingerprint, container.fingerprint if container else None)
            if fp and image_tag(fp) in tags
        }
        plan = plan_host(
            container,
            available,
            status,
            desired,
            refresh=self.refresh,
            resolution_problem=resolution,
        )
        current = (
            f"{container.version or '?'} {(container.fingerprint or '?')[:12]}"
            f"{'' if container.running else ' (stopped)'}"
            if container
            else "no container"
        )
        self.out(
            f"{endpoint.name}: current {current} → desired {desired.version} "
            f"{desired.fingerprint[:12]}: {plan.describe()}"
        )
        if plan.report:
            raise DeployError(plan.report)
        if self.dry_run:
            return
        if plan.container == "none" and not plan.build:
            self.out(f"✓ {endpoint.name}: up to date")
            return
        if plan.build:
            self._build(runner, endpoint)
        if plan.container in ("create", "replace"):
            runner.run(["docker", "rm", "-f", "-v", name], check=False)
            runner.run(run_arguments(endpoint, self.config, self.wheel))
        elif plan.container == "start":
            runner.run(["docker", "start", name])
        self._wait_healthy(runner, endpoint, desired)
        if plan.verify and self.verify:
            self._verify(runner, endpoint)
        if plan.build or plan.container in ("create", "replace"):
            previous = container.fingerprint if container else None
            self._prune(
                runner, image_tag(desired.fingerprint), image_tag(previous) if previous else None
            )
        self.out(
            f"✓ {endpoint.name}: running arcaneum {desired.version} {desired.fingerprint[:12]}"
        )

    def _images(self, runner: RemoteRunner) -> list[tuple[str, str, str]]:
        listed = runner.run(
            ["docker", "image", "ls", "--filter", f"label={ROLE_LABEL}", "--format", "{{json .}}"],
            check=False,
        )
        return _parse_images(listed.stdout) if listed.returncode == 0 else []

    def _build(self, runner: RemoteRunner, endpoint: EndpointConfig) -> None:
        self.out(
            f"  building {image_tag(self.wheel.fingerprint)} on {endpoint.ssh_target} "
            "(can take several minutes; a first or --refresh build downloads torch)"
        )
        try:
            directory = runner.run(
                ["mktemp", "-d", "-t", "arcaneum-embed-build.XXXXXX"]
            ).stdout.strip()
            with tempfile.TemporaryDirectory(prefix="arcaneum-deploy-") as local:
                dockerfile = Path(local) / "Dockerfile"
                dockerfile.write_text(render_dockerfile(self.wheel.path.name) + "\n")
                runner.copy(self.wheel.path, f"{directory}/{self.wheel.path.name}")
                runner.copy(dockerfile, f"{directory}/Dockerfile")
            refresh = ["--pull", "--no-cache"] if self.refresh else []
            labels = [
                ROLE_LABEL,
                f"arcaneum.version={self.wheel.version}",
                f"arcaneum.fingerprint={self.wheel.fingerprint}",
                f"arcaneum.wheel_sha256={self.wheel.sha256}",
            ]
            label_args = [part for label in labels for part in ("--label", label)]
            try:
                runner.run(
                    [
                        "docker",
                        "build",
                        *refresh,
                        "-t",
                        image_tag(self.wheel.fingerprint),
                        *label_args,
                        directory,
                    ],
                    timeout=BUILD_TIMEOUT,
                )
            finally:
                runner.run(["rm", "-rf", directory], check=False)
        except DeployError as exc:
            raise DeployError(
                f"image build failed; the running worker is untouched: {exc}"
            ) from exc

    def _wait_healthy(
        self, runner: RemoteRunner, endpoint: EndpointConfig, desired: Desired
    ) -> None:
        name = container_name(endpoint.port)
        attempts = max(1, math.ceil(self.health_timeout / HEALTH_POLL))
        status = None
        for _ in range(attempts):
            status = self.probe(endpoint, self.config)
            if status.state == "up" and status.code_fingerprint == desired.fingerprint:
                return
            inspected = runner.run(["docker", "container", "inspect", name], check=False)
            container = _parse_container(inspected.stdout) if inspected.returncode == 0 else None
            if container is None or not container.running:
                break
            self.sleep(HEALTH_POLL)
        logs = runner.run(["docker", "logs", "--tail", "40", name], check=False)
        self.out(f"  {name} log on {endpoint.ssh_target}:\n{(logs.stdout + logs.stderr).rstrip()}")
        detail = tailscale_resolution_problem(endpoint) or (status.reason if status else None)
        raise DeployError(
            f"the worker did not report fingerprint {desired.fingerprint[:12]} within "
            f"{self.health_timeout:.0f}s" + (f" ({detail})" if detail else "")
        )

    def _verify(self, runner: RemoteRunner, endpoint: EndpointConfig) -> None:
        models = self.config.endpoint_models(endpoint)
        self.out(f"  verifying vector parity on {endpoint.name}: {', '.join(models)}")
        name = container_name(endpoint.port)
        try:
            results = self.verifier(endpoint, self.config, models)
        except Exception as exc:
            # A worker that crashed or disconnected mid-check is unverified; Docker
            # would restart it and the pool would use it, so remove it.
            logs = runner.run(["docker", "logs", "--tail", "40", name], check=False)
            self.out(
                f"  {name} log on {endpoint.ssh_target}:\n{(logs.stdout + logs.stderr).rstrip()}"
            )
            runner.run(["docker", "rm", "-f", "-v", name], check=False)
            raise DeployError(
                "vector parity check could not complete "
                f"({type(exc).__name__}: {exc}); container removed (image kept)"
            ) from exc
        self.out("  " + "; ".join(f"{r.model} min cosine {r.min_cosine:.6f}" for r in results))
        failed = [result for result in results if not result.ok]
        if not failed:
            return
        runner.run(["docker", "rm", "-f", "-v", container_name(endpoint.port)], check=False)
        details = "; ".join(
            f"{r.model}: min cosine {r.min_cosine:.5f} < {PARITY_THRESHOLD} "
            f"(worker {self._versions(r.remote_dependencies)}; "
            f"workstation {self._versions(r.local_dependencies)})"
            for r in failed
        )
        raise DeployError(f"vector parity check failed, container removed (image kept): {details}")

    @staticmethod
    def _versions(dependencies: dict) -> str:
        return ", ".join(f"{k} {v}" for k, v in sorted(dependencies.items()) if v) or "unknown"

    def _prune(self, runner: RemoteRunner, current: str, previous: Optional[str]) -> None:
        """Keep the current and previous fingerprint's images; remove the rest.

        The previous fingerprint is the replaced container's; when that was already
        the current one (a refresh or reconfigure), it is the newest other tagged
        worker image, so a downgrade can still reuse it.
        """
        images = self._images(runner)
        if previous is None or previous == current:
            others = [
                (created, tag)
                for _, tag, created in images
                if tag != current and not tag.endswith(":<none>")
            ]
            previous = max(others)[1] if others else None
        keep = {current, previous}
        keep_ids = {image_id for image_id, tag, _ in images if tag in keep}
        for image_id in sorted({image_id for image_id, _, _ in images} - keep_ids):
            runner.run(["docker", "image", "rm", image_id], check=False)

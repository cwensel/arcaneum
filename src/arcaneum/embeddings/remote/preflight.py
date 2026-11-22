"""Run-start check of remote embedding workers, and auto-deploy (RDR-025, arcaneum#187).

A command that embeds documents installs a router at its start
(``configure_remote_embedding``), and the router checks its workers in the
background: waking a sleeping worker and probing it overlap extraction instead
of waiting for the first embed. Each endpoint's state is reported once.

A reachable worker whose code fingerprint differs from this process's is
updated by a detached ``arc models deploy`` child, one per endpoint
(``remote_embedding.auto_deploy``, on by default; ``--no-auto-deploy`` for one
run). The child runs in its own session and outlives this process, so a run
that ends mid-deploy never strands a worker between ``docker rm`` and ``docker
run``. It is passed this process's fingerprint and refuses to build if its own
differs. The pool keeps the endpoint out until the child exits, so deploy's
vector parity check still gates the rejoin; until then the run follows
``unavailable`` (embedding locally under ``local``, waiting under ``wait``).

Auto-deploy is skipped, with a message, when:

- the worker is busy with encodes: another workstation running other code is
  using it, and two installs with different code sharing workers would
  otherwise redeploy it back and forth;
- the source tree changed since this process computed its fingerprint (another
  commit landed mid-run), so a deploy would build code this run does not run;
- an auto-deploy of this fingerprint to the endpoint failed before; a manual
  ``arc models deploy`` clears that record;
- this process runs from a source tree but no wheel build tool (uv, or the
  ``build`` package) can be found, so the child could only fail (arcaneum#188).

Each child is watched from a daemon thread, which reports its outcome once: a
failure with its exit code, last log line, log path and the retry command. The
pool's "embedding locally" notice also names a child that failed.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from arcaneum.embeddings.remote.config import EndpointConfig, RemoteEmbeddingConfig
from arcaneum.embeddings.remote.deploy import (
    AutoDeployFailures,
    DeployError,
    build_command,
    build_tool_searched,
    safe_name,
    source_tree,
    wake_sleeping,
)
from arcaneum.embeddings.remote.fingerprint import (
    code_fingerprint,
    installed_requires,
    tree_fingerprint,
)
from arcaneum.embeddings.remote.pool import EndpointStatus, probe_endpoint
from arcaneum.embeddings.remote.wake import wake_endpoints

if TYPE_CHECKING:
    from arcaneum.embeddings.remote.routing import RemoteRouter

logger = logging.getLogger(__name__)


@dataclass
class DeployChild:
    """A detached `arc models deploy` this process started for one endpoint."""

    endpoint: str
    process: Any  # subprocess.Popen, or anything with poll()
    log: Path

    @property
    def running(self) -> bool:
        return self.process.poll() is None

    @property
    def returncode(self) -> Optional[int]:
        """The child's exit code, or None while it runs."""
        return self.process.poll()

    def last_error(self) -> Optional[str]:
        """The last non-empty line of the child's log: its error, when it failed."""
        try:
            with open(self.log, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - 8192))
                text = handle.read().decode(errors="replace")
        except OSError:
            return None
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else None

    @property
    def failure(self) -> Optional[str]:
        """How the deploy failed once it exited non-zero; None while running or after success."""
        code = self.returncode
        if code is None or code == 0:
            return None
        return f"auto-deploy failed (exit {code}: {self.last_error() or 'no output'}; log {self.log})"


# Endpoint name -> the deploy this process started for it: one per endpoint per
# process, whichever router started it. The pool reads it to hold the endpoint out.
DEPLOYS: dict[str, DeployChild] = {}


def log_dir() -> Path:
    return Path.home() / ".arcaneum" / "logs"


def deploy_arguments(endpoint: EndpointConfig, fingerprint: str, *, build: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "arcaneum.cli.main",
        "models",
        "deploy",
        endpoint.name,
        "--expect-fingerprint",
        fingerprint,
    ]
    if build:
        command.append("--build")
    return command


def spawn_deploy(
    endpoint: EndpointConfig,
    fingerprint: str,
    *,
    build: bool,
    popen: Callable[..., Any] = subprocess.Popen,
    logs: Optional[Path] = None,
) -> DeployChild:
    """Start `arc models deploy` for one endpoint in its own session, logging to a file."""
    directory = Path(logs) if logs is not None else log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log = directory / f"arc-deploy-{safe_name(endpoint.name)}-{stamp}-{os.getpid()}.log"
    with open(log, "ab") as handle:
        process = popen(
            deploy_arguments(endpoint, fingerprint, build=build),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    return DeployChild(endpoint.name, process, log)


def can_build() -> bool:
    """Whether this process runs from a source tree `arc models deploy --build` can build.

    False means a release install, which deploys the release wheel; whether a
    build tool exists is ``missing_build_tool``'s question.
    """
    try:
        source_tree()
    except DeployError:
        return False
    return True


def missing_build_tool() -> Optional[str]:
    """Where a wheel build tool was looked for when none was found; None when one was."""
    return None if build_command() is not None else build_tool_searched()


def report_deploy(child: DeployChild, retry: str, notify: Callable[[str], None]) -> None:
    """Wait for one deploy child and report its outcome once."""
    code = child.process.wait()
    if code == 0:
        notify(f"{child.endpoint} updated by the background deploy; it rejoins the pool.")
        return
    notify(
        f"The background deploy to {child.endpoint} failed (exit {code}): "
        f"{child.last_error() or 'no output'}. Log {child.log}. Later runs skip this code "
        f"for it until `{retry}` succeeds."
    )


def watch_deploy(
    child: DeployChild, retry: str, notify: Optional[Callable[[str], None]] = None
) -> threading.Thread:
    """Report a deploy child's outcome from a daemon thread, so the run never waits on it."""
    notify = notify or _stderr_notify

    def run() -> None:
        try:
            report_deploy(child, retry, notify)
        except Exception:
            logger.warning("Watching the deploy to %s failed", child.endpoint, exc_info=True)

    thread = threading.Thread(target=run, name=f"arc-deploy-watch-{child.endpoint}", daemon=True)
    thread.start()
    return thread


def tree_changed() -> bool:
    """Whether the package sources on disk differ from the code this process fingerprinted."""
    import arcaneum

    current = tree_fingerprint(Path(arcaneum.__file__).parent, installed_requires())
    return current != code_fingerprint()


def _stderr_notify(message: str) -> None:
    print(f"   {message}", file=sys.stderr, flush=True)


def describe(status: EndpointStatus) -> str:
    if status.state == "up":
        return (
            f"{status.endpoint}: up (arcaneum {status.arcaneum_version} "
            f"{(status.code_fingerprint or '?')[:12]})"
        )
    return f"{status.endpoint}: {status.state} ({status.reason})"


def run_preflight(
    router: "RemoteRouter",
    models: Sequence[str],
    *,
    probe: Callable[[EndpointConfig, RemoteEmbeddingConfig], EndpointStatus] = probe_endpoint,
    spawn: Callable[..., DeployChild] = spawn_deploy,
    wake: Callable[..., bool] = wake_endpoints,
    notify: Callable[[str], None] = _stderr_notify,
    failures: Optional[AutoDeployFailures] = None,
    deploys: Optional[dict[str, DeployChild]] = None,
    changed: Callable[[], bool] = tree_changed,
    buildable: Callable[[], bool] = can_build,
    missing_tool: Callable[[], Optional[str]] = missing_build_tool,
    watch: Callable[..., Any] = watch_deploy,
) -> None:
    """Wake and probe the endpoints serving ``models``; deploy to those running other code."""
    config = router.config
    deploys = DEPLOYS if deploys is None else deploys
    endpoints = [
        endpoint
        for endpoint in config.endpoints
        if any(model in config.endpoint_models(endpoint) for model in models)
    ]
    if not endpoints:
        return
    wake_sleeping(config, endpoints, wake, dry_run=False, out=notify)
    with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
        statuses = list(executor.map(lambda endpoint: probe(endpoint, config), endpoints))
    for status in statuses:
        notify(f"Remote embedding worker {describe(status)}")
    stale = [
        endpoint
        for endpoint, status in zip(endpoints, statuses)
        if status.state == "mismatch" and endpoint.name not in deploys
    ]
    if not stale:
        return
    names = ", ".join(endpoint.name for endpoint in stale)
    build = buildable()
    command = "arc models deploy" + (" --build" if build else "")
    if not router.auto_deploy:
        notify(
            f"Auto-deploy is off (remote_embedding.auto_deploy: false or --no-auto-deploy); "
            f"run `{command}` to update {names}."
        )
        return
    if changed():
        notify(
            "The source tree changed since this run started, so a deploy would build code "
            f"this run does not run; not updating {names}. Run `{command}` after the run."
        )
        return
    missing = missing_tool() if build else None
    if missing is not None:
        notify(
            f"Not updating {names}: this run's code is a source tree and auto-deploy builds "
            f"its wheel, but {missing}. Install uv, then run `{command}`."
        )
        return
    failures = failures or AutoDeployFailures()
    fingerprint = code_fingerprint()
    by_name = dict(zip((endpoint.name for endpoint in endpoints), statuses))
    for endpoint in stale:
        if by_name[endpoint.name].busy:
            notify(
                f"{endpoint.name} is busy with other encodes (another workstation running "
                f"other code?); not updating it. Run `{command}` when it is idle."
            )
            continue
        failed = failures.recorded(endpoint.name, fingerprint)
        if failed is not None:
            notify(
                f"An earlier auto-deploy of {fingerprint[:12]} to {endpoint.name} failed "
                f"({failed}); run `{command} {endpoint.host}` to retry."
            )
            continue
        try:
            child = spawn(endpoint, fingerprint, build=build)
        except OSError as exc:
            notify(f"Could not start a deploy to {endpoint.name}: {exc}")
            continue
        deploys[endpoint.name] = child
        notify(
            f"Updating {endpoint.name} to this workstation's code ({fingerprint[:12]}) with a "
            f"background `{command}` (log {child.log}); it rejoins when the deploy finishes."
        )
        watch(child, f"{command} {endpoint.host}", notify)


def start_preflight(router: "RemoteRouter", models: Sequence[str]) -> threading.Thread:
    """Run the preflight in a daemon thread: it only probes and starts detached children."""

    def run() -> None:
        try:
            run_preflight(router, list(models))
        except Exception:
            logger.warning("Remote embedding preflight failed", exc_info=True)

    thread = threading.Thread(target=run, name="arc-embed-preflight", daemon=True)
    thread.start()
    return thread

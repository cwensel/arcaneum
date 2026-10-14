"""Remote embedding worker commands: `arc models serve`, `workers`, `deploy` (RDR-025)."""

from __future__ import annotations

import logging
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from arcaneum.embeddings.client import EMBEDDING_MODELS, model_revision
from arcaneum.embeddings.remote.deploy import Deployer, DeployError, resolve_wheel
from arcaneum.embeddings.remote.pool import probe_endpoint
from arcaneum.embeddings.remote.server import EmbeddingServer, ServerSettings


def parse_served_models(value: Optional[str]) -> Optional[tuple[str, ...]]:
    """Parse a comma-separated model list; None or empty serves every pinned model."""
    if not value or not value.strip():
        return None
    models = tuple(part.strip() for part in value.split(",") if part.strip())
    for model in models:
        if model not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown model '{model}'")
        if model_revision(model) is None:
            raise ValueError(
                f"Model '{model}' cannot be served remotely: it has no pinned revision"
            )
    return models


def parse_batch_sizes(value: Optional[str]) -> dict[str, int]:
    """Parse ``MODEL=N[,MODEL=N]`` encode batch sizes."""
    sizes: dict[str, int] = {}
    if not value:
        return sizes
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        model, separator, raw = part.partition("=")
        if not separator or not model.strip() or not raw.strip():
            raise ValueError(f"Batch sizes must be MODEL=N, not {part!r}")
        size = int(raw)
        if size < 1:
            raise ValueError(f"Batch size for {model.strip()} must be positive")
        sizes[model.strip()] = size
    return sizes


def serve_command(
    *,
    port: int,
    bind: str,
    served_models: Optional[str],
    batch_sizes: Optional[str],
    idle_grace: float,
    turn_length: float,
    cache_dir: Optional[str],
) -> None:
    """Run the embedding server until SIGTERM or SIGINT."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    authkey = os.environ.get("ARC_EMBED_AUTHKEY") or None
    settings = ServerSettings(
        port=port,
        bind=bind,
        authkey=authkey.encode() if authkey else None,
        models=parse_served_models(served_models),
        batch_sizes=parse_batch_sizes(batch_sizes),
        idle_grace=idle_grace,
        turn_length=turn_length,
        cache_dir=cache_dir,
    )
    server = EmbeddingServer(settings)

    def stop(signum, frame):
        logging.getLogger(__name__).info("Received signal %d; stopping", signum)
        server.stop()

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        server.serve_forever()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


# --- arc models workers ------------------------------------------------------------

DEPENDENCY_COLUMNS = (
    ("torch", "torch"),
    ("onnxruntime", "onnxruntime"),
    ("fastembed", "fastembed"),
    ("sentence-transformers", "s-t"),
    ("transformers", "transformers"),
)


def _short_fingerprint(value: Optional[str]) -> str:
    return value[:12] if value else "-"


def _load_config_or_exit(output_json: bool):
    from arcaneum.cli.output import print_error
    from arcaneum.embeddings.remote.config import (
        RemoteEmbeddingConfigError,
        load_remote_embedding_config,
    )

    try:
        return load_remote_embedding_config()
    except RemoteEmbeddingConfigError as exc:
        print_error(f"Invalid remote_embedding configuration: {exc}", output_json)
        sys.exit(1)


NO_ENDPOINTS = (
    "No remote embedding endpoints are configured. Add them under "
    "remote_embedding.endpoints in ~/.config/arcaneum/config.yaml or set ARC_EMBED_ENDPOINTS."
)


def workers_command(output_json: bool) -> None:
    """Report each configured endpoint's state, versions and resident model."""
    from rich.console import Console
    from rich.table import Table

    from arcaneum.cli.output import print_json
    from arcaneum.embeddings.remote.identity import dependency_versions, local_identity

    config = _load_config_or_exit(output_json)
    identity = local_identity(None)
    local = {
        "arcaneum_version": identity["arcaneum_version"],
        "code_fingerprint": identity["code_fingerprint"],
        "dependencies": dependency_versions(),
    }
    if not config.endpoints:
        if output_json:
            print_json("success", NO_ENDPOINTS, data={"local": local, "workers": []})
        else:
            Console().print(NO_ENDPOINTS)
        return

    with ThreadPoolExecutor(max_workers=len(config.endpoints)) as executor:
        statuses = list(executor.map(lambda e: probe_endpoint(e, config), config.endpoints))

    workers = [
        {
            "endpoint": status.endpoint,
            "state": status.state,
            "reason": status.reason,
            "arcaneum_version": status.arcaneum_version,
            "code_fingerprint": status.code_fingerprint,
            "resident_model": status.resident_model,
            "loading_model": status.loading_model,
            "busy": status.busy,
            "dependencies": status.dependencies,
            "bind": endpoint.bind,
            "models": config.endpoint_models(endpoint),
        }
        for endpoint, status in zip(config.endpoints, statuses)
    ]
    healthy = all(worker["state"] == "up" for worker in workers)
    summary = f"{sum(w['state'] == 'up' for w in workers)} of {len(workers)} workers up"
    if output_json:
        status = "success" if healthy else "warning"
        print_json(status, summary, data={"local": local, "workers": workers})
        return

    console = Console()
    console.print(
        f"Workstation: arcaneum {local['arcaneum_version']} "
        f"fingerprint {_short_fingerprint(local['code_fingerprint'])}"
    )
    table = Table(title="Remote embedding workers")
    for column in ("Endpoint", "State", "Version", "Fingerprint", "Resident", "Busy"):
        table.add_column(column, no_wrap=column in ("Endpoint", "State"))
    styles = {"up": "green", "mismatch": "yellow", "unreachable": "red"}
    for worker in workers:
        state = worker["state"]
        resident = worker["resident_model"] or (
            f"loading {worker['loading_model']}" if worker["loading_model"] else "-"
        )
        busy = "-" if worker["busy"] is None else ("yes" if worker["busy"] else "no")
        table.add_row(
            worker["endpoint"],
            f"[{styles.get(state, 'white')}]{state}[/]",
            worker["arcaneum_version"] or "-",
            _short_fingerprint(worker["code_fingerprint"]),
            resident,
            busy,
        )
    console.print(table)
    for worker in workers:
        details = [f"models {','.join(worker['models'])}", f"bind {worker['bind']}"]
        if worker["dependencies"]:
            details.append(
                " ".join(
                    f"{label} {worker['dependencies'].get(name) or '-'}"
                    for name, label in DEPENDENCY_COLUMNS
                )
            )
        if worker["reason"]:
            details.append(worker["reason"])
        console.print(f"  {worker['endpoint']}: {'; '.join(details)}", soft_wrap=True)
    if not healthy:
        console.print(
            "Run `arc models deploy` to install or update workers "
            "(`arc models deploy --dry-run` shows what would change)."
        )


# --- arc models deploy -------------------------------------------------------------


def deploy_command(
    hosts: tuple[str, ...],
    *,
    build: bool,
    wheel_path: Optional[Path],
    refresh: bool,
    dry_run: bool,
    no_verify: bool,
) -> None:
    """Bring each configured worker to the workstation's code; exit 1 if any host fails."""
    import click

    config = _load_config_or_exit(False)
    if not config.endpoints:
        click.echo(NO_ENDPOINTS, err=True)
        sys.exit(1)
    if build and wheel_path is not None:
        raise click.UsageError("--build and --wheel are mutually exclusive")
    try:
        wheel = resolve_wheel(build=build, wheel=wheel_path)
    except DeployError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo(
        f"Deploying arcaneum {wheel.version} (fingerprint {wheel.fingerprint[:12]}) "
        f"from {wheel.path}" + (" [dry run]" if dry_run else "")
    )
    deployer = Deployer(
        config,
        wheel,
        dry_run=dry_run,
        refresh=refresh,
        verify=not no_verify,
        out=click.echo,
    )
    try:
        ok = deployer.deploy(hosts)
    except DeployError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    if not ok:
        sys.exit(1)

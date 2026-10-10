"""Container management commands for Arcaneum services."""

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import requests

from arcaneum.cli.errors import HelpfulGroup
from arcaneum.cli.output import print_error, print_info, print_json, print_success, print_warning
from arcaneum.cli.utils import resolve_config_path
from arcaneum.config import load_backup_config
from arcaneum.paths import get_data_dir
from arcaneum.utils.formatting import format_size

_TASK_UID_UNSET = object()


def _exit_on_error(code: int = 1):
    raise SystemExit(code)


def _resolve_backup_path(
    output: str | None, timestamp: str, output_json: bool = False, dry_run: bool = False
) -> Path:
    """Resolve the directory for a full backup.

    An explicit output path names the backup directory exactly. A configured
    backup path is a root under which timestamped backup directories are
    created. Relative configured paths are resolved from the config directory.
    """
    if output:
        return Path(output).expanduser()

    if dry_run:
        # Preview the post-migration destination without migrating legacy config.
        config_home = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        config_path = Path(config_home) / "arcaneum" / "config.yaml"
        config_source = config_path
        if not config_source.exists():
            config_source = Path.home() / ".arcaneum" / "config.yaml"
    else:
        config_source = config_path = resolve_config_path()
    if config_source.exists():
        try:
            configured_path = load_backup_config(config_source).path
        except (OSError, ValueError) as exc:
            print_warning(
                f"Ignoring invalid backup configuration in {config_path}: {exc}",
                output_json,
            )
            configured_path = None
        if configured_path is not None:
            backup_root = configured_path.expanduser()
            if not backup_root.is_absolute():
                backup_root = (config_path.parent / backup_root).resolve()
            return backup_root / timestamp

    if dry_run:
        data_home = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
        return Path(data_home) / "arcaneum" / "backups" / timestamp
    return get_data_dir() / "backups" / timestamp


def _backup_corpus_details(backup_path: Path, manifest: dict) -> list[dict]:
    """Describe each backed-up corpus and the sizes of its artifacts."""
    corpora: dict[str, dict] = {}

    def add_file(corpus_name: str, label: str, relative_path: str) -> None:
        size_bytes = (backup_path / relative_path).stat().st_size
        corpus = corpora.setdefault(
            corpus_name,
            {"name": corpus_name, "size_bytes": 0, "files": []},
        )
        corpus["size_bytes"] += size_bytes
        corpus["files"].append({"label": label, "file": relative_path, "size_bytes": size_bytes})

    for snapshot in manifest["qdrant"]:
        add_file(snapshot["collection"], "Qdrant", snapshot["file"])

    for index in manifest["meilisearch"]:
        add_file(index["index"], "MeiliSearch metadata", index["metadata_file"])
        add_file(index["index"], "MeiliSearch documents", index["documents_file"])

    return sorted(corpora.values(), key=lambda corpus: corpus["name"].casefold())


def check_docker_available(output_json: bool = False):
    """Check if Docker is installed and running."""
    if not shutil.which("docker"):
        print_error(
            "Docker is not installed. Please install Docker Desktop or Docker Engine.", output_json
        )
        print_info("Visit: https://docs.docker.com/get-docker/", output_json)
        return False

    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=5)
        return True
    except subprocess.CalledProcessError:
        print_error("Docker is installed but not running. Please start Docker.", output_json)
        return False
    except subprocess.TimeoutExpired:
        print_error("Docker is not responding. Please check Docker status.", output_json)
        return False


def get_compose_file(output_json: bool = False):
    """Get the path to docker-compose.yml."""
    # Find the repository root directory
    # This file is at: src/arcaneum/cli/docker.py
    # We need to go up 3 levels to get to repo root: cli/ -> arcaneum/ -> src/ -> root/
    repo_root = Path(__file__).parent.parent.parent.parent

    # Try repo deploy/ directory first, then current directory as fallback
    compose_paths = [
        repo_root / "deploy" / "docker-compose.yml",
        Path("docker-compose.yml"),
        Path("deploy/docker-compose.yml"),
    ]

    for path in compose_paths:
        if path.exists():
            return str(path.resolve())

    print_error("docker-compose.yml not found", output_json)
    print_info(
        f"Expected locations: {repo_root}/deploy/docker-compose.yml or ./docker-compose.yml",
        output_json,
    )
    return None


def run_compose_command(
    args, check=True, capture_output=False, env=None, output_json: bool = False
):
    """Run a docker compose command.

    Args:
        args: Command arguments to pass to docker compose
        check: Whether to raise on non-zero exit code
        capture_output: Whether to capture stdout/stderr
        env: Optional environment variables to add (merged with current env)
    """
    compose_file = get_compose_file(output_json)
    if not compose_file:
        return None

    cmd = ["docker", "compose", "-f", compose_file, "-p", "arcaneum"] + args

    # Merge environment variables
    import os

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        result = subprocess.run(
            cmd, check=check, capture_output=capture_output, text=True, env=run_env
        )
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Container command failed: {e}", output_json)
        if e.stderr and not output_json:
            print(e.stderr)
        return None


# Default MeiliSearch CPU limit in deploy/docker-compose.yml.
_MEILI_DEFAULT_CPUS = 8


def _docker_cpu_count() -> int | None:
    """CPUs available to the Docker daemon, or None if it cannot be asked."""
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.NCPU}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(result.stdout.strip()) if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def get_resource_limit_env():
    """Get compose CPU limits that fit the Docker daemon.

    Compose refuses to create a container whose CPU limit exceeds the daemon's
    CPUs, so MEILI_CPUS is capped on smaller hosts unless the user set it.
    """
    if "MEILI_CPUS" in os.environ:
        return {}
    cpus = _docker_cpu_count()
    if cpus and cpus < _MEILI_DEFAULT_CPUS:
        return {"MEILI_CPUS": str(cpus)}
    return {}


# MeiliSearch opens only a database written by its exact version, down to the
# patch, so deploy/docker-compose.yml pins an exact tag and data moves between
# versions through `arc container upgrade`.
_MEILI_IMAGE = "getmeili/meilisearch"
_MEILI_DATA_VOLUME = "arcaneum_meilisearch-arcaneum-data"
# Oldest database version MeiliSearch's --upgrade-db can migrate.
_MEILI_MIN_UPGRADABLE = (1, 12, 0)


def _parse_meili_version(text) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", str(text).strip())
    return tuple(int(part) for part in match.groups()) if match else None


def _format_meili_version(version: tuple[int, int, int]) -> str:
    return "v" + ".".join(str(part) for part in version)


def _compose_meili_image_tag(output_json: bool = False) -> str | None:
    """MeiliSearch image tag compose will run: MEILI_IMAGE_TAG or the pinned default."""
    if os.environ.get("MEILI_IMAGE_TAG"):
        return os.environ["MEILI_IMAGE_TAG"]
    compose_file = get_compose_file(output_json)
    if not compose_file:
        return None
    try:
        compose = Path(compose_file).read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"getmeili/meilisearch:\$\{MEILI_IMAGE_TAG:-([^}]+)\}", compose)
    return match.group(1) if match else None


def _meili_data_version(image_tag: str) -> tuple[int, int, int] | None:
    """Version of the database in the MeiliSearch volume, or None if unknown.

    Read from the volume rather than the server so it works while MeiliSearch
    is stopped, or crash-looping on data it refuses to open.
    """
    try:
        # `docker run -v <name>:...` silently creates a missing volume.
        inspect = subprocess.run(
            ["docker", "volume", "inspect", _MEILI_DATA_VOLUME],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if inspect.returncode != 0:
            return None
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "cat",
                "-v",
                f"{_MEILI_DATA_VOLUME}:/meili_data:ro",
                f"{_MEILI_IMAGE}:{image_tag}",
                "/meili_data/data.ms/VERSION",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _parse_meili_version(result.stdout) if result.returncode == 0 else None


def _meili_version_blocker(output_json: bool = False) -> str | None:
    """Why compose cannot start MeiliSearch on the existing data, if it cannot."""
    image_tag = _compose_meili_image_tag(output_json)
    target = _parse_meili_version(image_tag) if image_tag else None
    if target is None:
        return None
    current = _meili_data_version(image_tag)
    if current is None or current == target:
        return None
    if current < target:
        return (
            f"MeiliSearch data is {_format_meili_version(current)} but the configured image "
            f"is {image_tag}, and MeiliSearch will not open older data. "
            "Migrate it with: arc container upgrade"
        )
    return (
        f"MeiliSearch data is {_format_meili_version(current)}, newer than the configured "
        f"image {image_tag}; downgrades are not supported. Set MEILI_IMAGE_TAG to the "
        "data's version, or restore a backup into a fresh volume."
    )


def get_container_env():
    """Get environment variables for container startup.

    Returns dict with MEILISEARCH_API_KEY set to auto-generated key.
    """
    from arcaneum.paths import get_meilisearch_api_key

    return {
        "MEILISEARCH_API_KEY": get_meilisearch_api_key(),
        "MEILI_ENV": "production",
    }


def check_qdrant_health():
    """Check if Qdrant is healthy."""
    try:
        response = requests.get("http://localhost:6333/healthz", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


@click.group(
    name="container",
    cls=HelpfulGroup,
    usage_examples=[
        "arc container start",
        "arc container stop",
        "arc container status",
        "arc container logs -f",
    ],
)
def container_group():
    """Manage container services (Qdrant, MeiliSearch)"""
    pass


def check_meilisearch_health():
    """Check if MeiliSearch is healthy."""
    try:
        response = requests.get("http://localhost:7700/health", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


@container_group.command("start")
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
def start_command(output_json=False):
    """Start container services"""
    if not check_docker_available(output_json):
        _exit_on_error()

    blocker = _meili_version_blocker(output_json)
    if blocker:
        print_error(blocker, output_json)
        _exit_on_error()

    # Get container environment (auto-generates MeiliSearch key if needed)
    container_env = {**get_container_env(), **get_resource_limit_env()}

    print_info("Starting container services...", output_json)
    result = run_compose_command(
        ["up", "-d"],
        env=container_env,
        capture_output=output_json,
        output_json=output_json,
    )

    if result is None:
        _exit_on_error()

    # Wait for services to start
    time.sleep(3)

    qdrant_healthy = check_qdrant_health()
    meili_healthy = check_meilisearch_health()

    if output_json:
        print_json(
            "success",
            "Container services start requested",
            {
                "services": {
                    "qdrant": {
                        "healthy": qdrant_healthy,
                        "rest_api": "http://localhost:6333",
                        "dashboard": "http://localhost:6333/dashboard",
                    },
                    "meilisearch": {
                        "healthy": meili_healthy,
                        "http_api": "http://localhost:7700",
                    },
                },
                "data_directory": str(get_data_dir()),
            },
        )
        return

    # Check Qdrant
    if qdrant_healthy:
        print_success("Qdrant started successfully")
        print("  REST API: http://localhost:6333")
        print("  Dashboard: http://localhost:6333/dashboard")
    else:
        print_warning("Qdrant may not be ready yet. Check logs with: arc container logs")

    # Check MeiliSearch
    if meili_healthy:
        print_success("MeiliSearch started successfully")
        print("  HTTP API: http://localhost:7700")
    else:
        print_warning("MeiliSearch may not be ready yet. Check logs with: arc container logs")

    print()
    print_info(f"Data directory: {get_data_dir()}")


@container_group.command("stop")
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
def stop_command(output_json=False):
    """Stop container services"""
    if not check_docker_available(output_json):
        _exit_on_error()

    print_info("Stopping container services...", output_json)
    result = run_compose_command(["down"], capture_output=output_json, output_json=output_json)

    if result is not None:
        print_success("Container services stopped", json_output=output_json, data={"stopped": True})
    else:
        _exit_on_error()


@container_group.command("restart")
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
def restart_command(output_json=False):
    """Restart container services"""
    if not check_docker_available(output_json):
        _exit_on_error()

    print_info("Restarting container services...", output_json)
    result = run_compose_command(["restart"], capture_output=output_json, output_json=output_json)

    if result is None:
        _exit_on_error()

    time.sleep(2)

    qdrant_healthy = check_qdrant_health()
    meili_healthy = check_meilisearch_health()

    if output_json:
        print_json(
            "success",
            "Container services restart requested",
            {
                "services": {
                    "qdrant": {"healthy": qdrant_healthy},
                    "meilisearch": {"healthy": meili_healthy},
                }
            },
        )
    elif qdrant_healthy:
        print_success("Container services restarted")
    else:
        print_warning("Services may not be ready yet")


@container_group.command("status")
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
def status_command(output_json=False):
    """Show container services status"""
    if not check_docker_available(output_json):
        _exit_on_error()

    print_info("Container Services Status:", output_json)
    if not output_json:
        print()
    # Include stopped services so the status table does not look empty when the
    # Compose project exists but its containers are not currently running. Pass
    # the same interpolation environment used by ``start`` to avoid Compose
    # warning that MEILISEARCH_API_KEY is unset while reading the project.
    ps_result = run_compose_command(
        ["ps", "--all"],
        capture_output=output_json,
        env=get_container_env(),
        output_json=output_json,
    )
    if ps_result is None:
        _exit_on_error()

    qdrant_healthy = check_qdrant_health()
    meili_healthy = check_meilisearch_health()

    if not output_json:
        print()
        if qdrant_healthy:
            print_success("Qdrant: Healthy")
        else:
            print_error("Qdrant: Unhealthy or not running")

        if meili_healthy:
            print_success("MeiliSearch: Healthy")
        else:
            print_error("MeiliSearch: Unhealthy or not running")

    # Show Docker volume information
    if not output_json:
        print()
    print_info("Docker Volumes:", output_json)

    # Get volume information including sizes using docker system df
    try:
        import json

        df_result = subprocess.run(
            ["docker", "system", "df", "-v", "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        )

        df_data = json.loads(df_result.stdout)
        volumes_data = {v["Name"]: v for v in df_data.get("Volumes", [])}

        # List volumes for this project
        result = subprocess.run(
            ["docker", "volume", "ls", "--filter", "name=arcaneum", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            check=True,
        )

        volumes = result.stdout.strip().split("\n")
        volume_results = []
        for volume in volumes:
            if volume:
                # Get volume details
                volume_info = volumes_data.get(volume, {})
                size = volume_info.get("Size", "unknown")

                # Get mountpoint
                inspect_result = subprocess.run(
                    ["docker", "volume", "inspect", volume, "--format", "{{.Mountpoint}}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                volume_result = {"name": volume, "size": None if size == "unknown" else size}
                if inspect_result.returncode == 0:
                    mountpoint = inspect_result.stdout.strip()
                    volume_result["mountpoint"] = mountpoint
                volume_results.append(volume_result)

                if not output_json:
                    print(f"  {volume}")
                    if size != "unknown":
                        print(f"    Size: {size}")
                    if inspect_result.returncode == 0:
                        print(f"    Mountpoint: {mountpoint}")

        if output_json:
            print_json(
                "success",
                "Container services status",
                {
                    "compose": {
                        "stdout": ps_result.stdout if ps_result is not None else "",
                    },
                    "services": {
                        "qdrant": {"healthy": qdrant_healthy},
                        "meilisearch": {"healthy": meili_healthy},
                    },
                    "volumes": volume_results,
                },
            )
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        if output_json:
            print_json(
                "success",
                "Container services status",
                {
                    "compose": {
                        "stdout": ps_result.stdout if ps_result is not None else "",
                    },
                    "services": {
                        "qdrant": {"healthy": qdrant_healthy},
                        "meilisearch": {"healthy": meili_healthy},
                    },
                    "volumes": [],
                    "warnings": [f"Could not retrieve volume information: {e}"],
                },
            )
        else:
            print_warning(f"Could not retrieve volume information: {e}")


@container_group.command("logs")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--tail", type=int, default=100, help="Number of lines to show")
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
def logs_command(follow, tail, output_json=False):
    """Show container services logs"""
    if output_json and follow:
        print_error("--json cannot be combined with --follow; use a finite --tail", output_json)
        raise SystemExit(2)

    if not check_docker_available(output_json):
        _exit_on_error()

    args = ["logs", f"--tail={tail}"]
    if follow:
        args.append("-f")

    result = run_compose_command(args, capture_output=output_json, output_json=output_json)
    if result is None:
        _exit_on_error()

    if output_json and result is not None:
        print_json(
            "success",
            "Container logs",
            {
                "follow": follow,
                "tail": tail,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "lines": result.stdout.splitlines(),
            },
        )


def _request_json(method: str, url: str, **kwargs):
    response = requests.request(method, url, timeout=kwargs.pop("timeout", 30), **kwargs)
    response.raise_for_status()
    return response.json()


def _meilisearch_headers(read_only: bool = False):
    from arcaneum.paths import get_meilisearch_api_key

    if read_only:
        key = os.environ.get("MEILISEARCH_API_KEY", "")
        if len(key) < 16:
            config_home = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
            key_file = Path(config_home) / "arcaneum" / "meilisearch.key"
            key = key_file.read_text().strip() if key_file.exists() else ""
        if len(key) < 16:
            raise RuntimeError(
                "No existing MeiliSearch API key. Set MEILISEARCH_API_KEY or use "
                "--skip-meilisearch to preview Qdrant only."
            )
        return {"Authorization": f"Bearer {key}"}
    return {"Authorization": f"Bearer {get_meilisearch_api_key()}"}


def _copy_from_container(container_name: str, source: str, destination: Path) -> None:
    subprocess.run(
        ["docker", "cp", f"{container_name}:{source}", str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )


def _copy_to_container(source: Path, container_name: str, destination: str) -> None:
    subprocess.run(
        ["docker", "cp", str(source), f"{container_name}:{destination}"],
        check=True,
        capture_output=True,
        text=True,
    )


def _mkdir_in_container(container_name: str, path: str) -> None:
    subprocess.run(
        ["docker", "exec", container_name, "mkdir", "-p", path],
        check=True,
        capture_output=True,
        text=True,
    )


def _remove_from_container(container_name: str, path: str) -> None:
    subprocess.run(
        ["docker", "exec", container_name, "rm", "-f", path],
        check=True,
        capture_output=True,
        text=True,
    )


def _record_warning(message: str, output_json: bool, warnings: list | None = None) -> None:
    """Surface a non-fatal warning in whichever mode the caller is running.

    `print_warning` is text-only by design, so JSON callers would otherwise lose
    disk-reclamation signals entirely. Collect them for the final payload
    instead of printing mid-document, which would corrupt the JSON output.
    """
    if warnings is not None:
        warnings.append(message)
    print_warning(message, output_json)


def _warn_orphaned_qdrant_snapshots(
    qdrant_url: str,
    collection: str,
    output_json: bool = False,
    warnings: list | None = None,
) -> None:
    """Report snapshots an earlier interrupted backup left inside the container.

    Snapshots are full-size copies of the collection, so orphans quietly consume
    the Qdrant volume until someone deletes them.
    """
    try:
        response = _request_json("GET", f"{qdrant_url}/collections/{collection}/snapshots")
    except requests.RequestException:
        return

    orphans = response.get("result", []) or []
    if not orphans:
        return

    delete_commands = "\n".join(
        "curl -X DELETE "
        + shlex.quote(f"{qdrant_url}/collections/{collection}/snapshots/{snapshot['name']}")
        for snapshot in orphans
    )
    _record_warning(
        f"{collection}: {len(orphans)} orphaned snapshot"
        f"{'s' if len(orphans) != 1 else ''} left in the container from an earlier "
        f"backup; delete with:\n{delete_commands}",
        output_json,
        warnings,
    )


def _backup_qdrant(
    backup_path: Path,
    qdrant_url: str,
    container_name: str,
    timeout: int,
    output_json: bool = False,
    warnings: list | None = None,
) -> list[dict]:
    qdrant_dir = backup_path / "qdrant"
    qdrant_dir.mkdir(parents=True, exist_ok=True)

    collections_response = _request_json("GET", f"{qdrant_url}/collections")
    collections = collections_response.get("result", {}).get("collections", [])
    snapshots = []

    for collection in collections:
        name = collection["name"]
        _warn_orphaned_qdrant_snapshots(qdrant_url, name, output_json, warnings)
        snapshot_response = _request_json(
            "POST",
            f"{qdrant_url}/collections/{name}/snapshots",
            timeout=timeout,
        )
        snapshot_name = snapshot_response["result"]["name"]
        destination = qdrant_dir / snapshot_name
        try:
            _copy_from_container(
                container_name,
                f"/qdrant/snapshots/{name}/{snapshot_name}",
                destination,
            )
        finally:
            # Always reclaim the in-container snapshot, even when the copy
            # failed or the process is interrupted mid-backup. A cleanup
            # failure must not mask the error that caused it.
            try:
                _request_json(
                    "DELETE",
                    f"{qdrant_url}/collections/{name}/snapshots/{snapshot_name}",
                    timeout=timeout,
                )
            except requests.RequestException as exc:
                _record_warning(
                    f"Could not delete Qdrant snapshot {name}/{snapshot_name}: {exc}",
                    output_json,
                    warnings,
                )
        snapshots.append(
            {
                "collection": name,
                "snapshot": snapshot_name,
                "file": f"qdrant/{snapshot_name}",
            }
        )

    return snapshots


def _ensure_meilisearch_idle(meilisearch_url: str, headers: dict) -> None:
    tasks_response = _request_json(
        "GET",
        f"{meilisearch_url}/tasks",
        headers=headers,
        params={"statuses": "enqueued,processing", "limit": 1},
    )
    if tasks_response.get("results"):
        raise RuntimeError(
            "MeiliSearch has active tasks. Wait for indexing to finish before backup."
        )


def _latest_meilisearch_task_uid(meilisearch_url: str, headers: dict):
    tasks_response = _request_json(
        "GET",
        f"{meilisearch_url}/tasks",
        headers=headers,
        params={"limit": 1},
    )
    tasks = tasks_response.get("results", [])
    if not tasks:
        return None
    return tasks[0].get("uid")


def _list_meilisearch_indexes(meilisearch_url: str, headers: dict) -> list[dict]:
    indexes = []
    offset = 0
    limit = 100
    while True:
        response = _request_json(
            "GET",
            f"{meilisearch_url}/indexes",
            headers=headers,
            params={"limit": limit, "offset": offset},
        )
        batch = response.get("results", [])
        indexes.extend(batch)
        if len(batch) < limit:
            return indexes
        offset += limit


def _preview_backup(
    backup_path: Path,
    qdrant_url: str,
    meilisearch_url: str,
    skip_meilisearch: bool,
    output_json: bool,
) -> None:
    """Inspect backup inputs without creating artifacts or modifying services."""
    if backup_path.exists():
        raise FileExistsError(f"Backup directory already exists: {backup_path}")

    warnings: list[str] = []
    collections = _request_json("GET", f"{qdrant_url}/collections")
    names = [item["name"] for item in collections.get("result", {}).get("collections", [])]
    for name in names:
        _warn_orphaned_qdrant_snapshots(qdrant_url, name, output_json, warnings)

    indexes = []
    if not skip_meilisearch:
        headers = _meilisearch_headers(read_only=True)
        _ensure_meilisearch_idle(meilisearch_url, headers)
        indexes = [item["uid"] for item in _list_meilisearch_indexes(meilisearch_url, headers)]

    print_info(f"Would back up to {backup_path}", output_json)
    print_info(f"Qdrant collections: {', '.join(names) or '(none)'}", output_json)
    meili_summary = "(skipped)" if skip_meilisearch else ", ".join(indexes) or "(none)"
    print_info(
        f"MeiliSearch indexes: {meili_summary}",
        output_json,
    )
    print_success(
        "Backup dry run complete; no backup created",
        output_json,
        data={
            "dry_run": True,
            "path": str(backup_path),
            "qdrant_collections": names,
            "meilisearch_indexes": indexes,
            "skip_meilisearch": skip_meilisearch,
            "warnings": warnings,
        },
    )


def _backup_meilisearch(
    backup_path: Path,
    meilisearch_url: str,
    starting_task_uid=_TASK_UID_UNSET,
) -> list[dict]:
    meili_dir = backup_path / "meilisearch"
    meili_dir.mkdir(parents=True, exist_ok=True)
    headers = _meilisearch_headers()

    if starting_task_uid is _TASK_UID_UNSET:
        starting_task_uid = _latest_meilisearch_task_uid(meilisearch_url, headers)
        _ensure_meilisearch_idle(meilisearch_url, headers)

    exported = []
    indexes = _list_meilisearch_indexes(meilisearch_url, headers)

    for index in indexes:
        uid = index["uid"]
        settings = _request_json(
            "GET",
            f"{meilisearch_url}/indexes/{uid}/settings",
            headers=headers,
        )
        document_count = 0
        metadata_file = meili_dir / f"{uid}.metadata.json"
        documents_file = meili_dir / f"{uid}.documents.jsonl"
        offset = 0
        limit = 1000

        with documents_file.open("w", encoding="utf-8") as documents_handle:
            while True:
                response = _request_json(
                    "GET",
                    f"{meilisearch_url}/indexes/{uid}/documents",
                    headers=headers,
                    params={"limit": limit, "offset": offset, "fields": "*"},
                )
                batch = response.get("results", [])
                for document in batch:
                    documents_handle.write(json.dumps(document) + "\n")
                document_count += len(batch)
                if len(batch) < limit:
                    break
                offset += limit

        metadata = {
            "uid": uid,
            "primaryKey": index.get("primaryKey"),
            "settings": settings,
            "documents": document_count,
        }
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        exported.append(
            {
                "index": uid,
                "primaryKey": index.get("primaryKey"),
                "documents": document_count,
                "metadata_file": f"meilisearch/{uid}.metadata.json",
                "documents_file": f"meilisearch/{uid}.documents.jsonl",
            }
        )

    _ensure_meilisearch_idle(meilisearch_url, headers)
    ending_task_uid = _latest_meilisearch_task_uid(meilisearch_url, headers)
    if ending_task_uid != starting_task_uid:
        raise RuntimeError(
            "MeiliSearch task history changed during backup. Run backup while indexing is idle."
        )

    return exported


def _wait_for_meili_task(
    meilisearch_url: str,
    task_uid: int | str,
    headers: dict,
    timeout_seconds: int,
) -> None:
    for _ in range(timeout_seconds):
        task = _request_json(
            "GET",
            f"{meilisearch_url}/tasks/{task_uid}",
            headers=headers,
        )
        status = task.get("status")
        if status == "succeeded":
            return
        if status == "failed":
            error = task.get("error", {})
            raise RuntimeError(error.get("message", f"MeiliSearch task {task_uid} failed"))
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for MeiliSearch task {task_uid}")


def _delete_meilisearch_index_if_exists(
    meilisearch_url: str,
    uid: str,
    headers: dict,
    timeout_seconds: int,
) -> None:
    response = requests.request(
        "GET",
        f"{meilisearch_url}/indexes/{uid}",
        headers=headers,
        timeout=30,
    )
    if response.status_code == 404:
        return
    response.raise_for_status()

    delete_response = _request_json(
        "DELETE",
        f"{meilisearch_url}/indexes/{uid}",
        headers=headers,
    )
    _wait_for_meili_task(
        meilisearch_url,
        delete_response["taskUid"],
        headers,
        timeout_seconds,
    )


def _iter_document_batches(
    documents,
    max_count: int = 1000,
    max_bytes: int = 8 * 1024 * 1024,
):
    batch = []
    batch_bytes = 2

    for document in documents:
        document_bytes = len(json.dumps(document).encode("utf-8")) + 1
        if batch and (len(batch) >= max_count or batch_bytes + document_bytes > max_bytes):
            yield batch
            batch = []
            batch_bytes = 2

        batch.append(document)
        batch_bytes += document_bytes

    if batch:
        yield batch


def _read_jsonl_documents(path: Path):
    with path.open(encoding="utf-8") as documents_handle:
        for line in documents_handle:
            if line.strip():
                yield json.loads(line)


def _validate_jsonl_documents(path: Path, expected_count: int | None) -> None:
    count = 0
    for _ in _read_jsonl_documents(path):
        count += 1
    if expected_count is not None and count != expected_count:
        raise ValueError(
            f"MeiliSearch document count mismatch for {path}: "
            f"expected {expected_count}, found {count}"
        )


def _restore_qdrant(
    backup_path: Path,
    qdrant_url: str,
    container_name: str,
    snapshots: list[dict],
    timeout: int,
    output_json: bool = False,
    warnings: list | None = None,
) -> None:
    for snapshot in snapshots:
        collection = snapshot["collection"]
        snapshot_name = snapshot["snapshot"]
        snapshot_file = backup_path / snapshot["file"]
        if not snapshot_file.exists():
            raise FileNotFoundError(f"Missing Qdrant snapshot: {snapshot_file}")

        container_path = f"/qdrant/snapshots/{collection}/{snapshot_name}"
        _mkdir_in_container(container_name, f"/qdrant/snapshots/{collection}")
        try:
            # The copy is inside the try: an interrupted `docker cp` can leave a
            # partial file behind, which leaks exactly like a completed one.
            _copy_to_container(snapshot_file, container_name, container_path)
            _request_json(
                "PUT",
                f"{qdrant_url}/collections/{collection}/snapshots/recover",
                json={"location": f"file://{container_path}"},
                params={"wait": "true"},
                timeout=timeout,
            )
        finally:
            # The copied-in snapshot is only needed for recovery; leaving it
            # behind permanently doubles the collection's disk use. A cleanup
            # failure must not mask the error that caused it.
            try:
                _remove_from_container(container_name, container_path)
            except subprocess.CalledProcessError as exc:
                _record_warning(
                    f"Could not remove restored snapshot {container_path}: {exc}",
                    output_json,
                    warnings,
                )


def _load_meilisearch_restore_specs(backup_path: Path, indexes: list[dict]) -> list[dict]:
    specs = []
    for index_manifest in indexes:
        metadata_file = backup_path / index_manifest["metadata_file"]
        documents_file = backup_path / index_manifest["documents_file"]
        if not metadata_file.exists():
            raise FileNotFoundError(f"Missing MeiliSearch metadata: {metadata_file}")
        if not documents_file.exists():
            raise FileNotFoundError(f"Missing MeiliSearch documents: {documents_file}")

        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        uid = metadata["uid"]
        _validate_jsonl_documents(documents_file, metadata.get("documents"))
        specs.append(
            {
                "uid": uid,
                "primary_key": metadata.get("primaryKey"),
                "settings": metadata.get("settings", {}),
                "documents_file": documents_file,
            }
        )

    return specs


def _restore_meilisearch(
    backup_path: Path,
    meilisearch_url: str,
    indexes: list[dict],
    timeout_seconds: int,
) -> None:
    headers = _meilisearch_headers()
    specs = _load_meilisearch_restore_specs(backup_path, indexes)

    for spec in specs:
        uid = spec["uid"]
        primary_key = spec["primary_key"]

        _delete_meilisearch_index_if_exists(meilisearch_url, uid, headers, timeout_seconds)

        create_response = _request_json(
            "POST",
            f"{meilisearch_url}/indexes",
            headers=headers,
            json={"uid": uid, "primaryKey": primary_key},
        )
        _wait_for_meili_task(
            meilisearch_url,
            create_response["taskUid"],
            headers,
            timeout_seconds,
        )

        settings_response = _request_json(
            "PATCH",
            f"{meilisearch_url}/indexes/{uid}/settings",
            headers=headers,
            json=spec["settings"],
        )
        _wait_for_meili_task(
            meilisearch_url,
            settings_response["taskUid"],
            headers,
            timeout_seconds,
        )

        for document_batch in _iter_document_batches(_read_jsonl_documents(spec["documents_file"])):
            add_response = _request_json(
                "POST",
                f"{meilisearch_url}/indexes/{uid}/documents",
                headers=headers,
                json=document_batch,
            )
            _wait_for_meili_task(
                meilisearch_url,
                add_response["taskUid"],
                headers,
                timeout_seconds,
            )


@container_group.command("backup")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Backup directory to create (overrides configured backup.path)",
)
@click.option("--qdrant-url", default="http://localhost:6333", help="Qdrant URL")
@click.option("--meilisearch-url", default="http://localhost:7700", help="MeiliSearch URL")
@click.option("--qdrant-container", default="qdrant-arcaneum", help="Qdrant container name")
@click.option(
    "--qdrant-timeout",
    default=300,
    show_default=True,
    help="Qdrant operation timeout in seconds",
)
@click.option("--skip-meilisearch", is_flag=True, help="Only back up Qdrant snapshots")
@click.option("--dry-run", is_flag=True, help="Preview the backup using read-only service queries")
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
def backup_command(
    output,
    qdrant_url,
    meilisearch_url,
    qdrant_container,
    qdrant_timeout,
    skip_meilisearch,
    output_json,
    dry_run=False,
):
    """Back up Qdrant snapshots and MeiliSearch indexes."""
    if not check_docker_available(output_json):
        _exit_on_error()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = _resolve_backup_path(output, timestamp, output_json, dry_run=dry_run)
    if dry_run:
        _preview_backup(backup_path, qdrant_url, meilisearch_url, skip_meilisearch, output_json)
        return
    backup_path.mkdir(parents=True, exist_ok=False)
    print_info(f"Backing up to {backup_path}", output_json)

    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qdrant_url": qdrant_url,
        "meilisearch_url": None if skip_meilisearch else meilisearch_url,
        "protected": [
            "Qdrant collection snapshots",
            "MeiliSearch index settings and documents",
            "Arcaneum collection/corpus metadata stored inside indexed systems",
        ],
        "not_protected": [
            "Embedding model cache",
            "Docker images",
            "Local source files referenced by indexed metadata",
            "Configuration secrets outside this backup directory",
        ],
        "qdrant": [],
        "meilisearch": [],
    }

    warnings: list[str] = []
    meili_starting_task_uid = None
    if not skip_meilisearch:
        meili_headers = _meilisearch_headers()
        meili_starting_task_uid = _latest_meilisearch_task_uid(meilisearch_url, meili_headers)
        _ensure_meilisearch_idle(meilisearch_url, meili_headers)

    print_info("Creating Qdrant snapshots...", output_json)
    manifest["qdrant"] = _backup_qdrant(
        backup_path,
        qdrant_url,
        qdrant_container,
        timeout=qdrant_timeout,
        output_json=output_json,
        warnings=warnings,
    )
    if not skip_meilisearch:
        print_info("Exporting MeiliSearch indexes...", output_json)
        manifest["meilisearch"] = _backup_meilisearch(
            backup_path,
            meilisearch_url,
            starting_task_uid=meili_starting_task_uid,
        )

    from arcaneum.backup import backup_checksums

    print_info("Calculating backup checksums...", output_json)
    manifest["checksums"] = backup_checksums(backup_path, manifest)
    manifest_path = backup_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    corpora = _backup_corpus_details(backup_path, manifest)

    data = {
        "path": str(backup_path),
        "qdrant_snapshots": len(manifest["qdrant"]),
        "meilisearch_indexes": len(manifest["meilisearch"]),
        "corpora": corpora,
        "warnings": warnings,
    }
    qdrant_count = data["qdrant_snapshots"]
    meilisearch_count = data["meilisearch_indexes"]
    summary = (
        f"{qdrant_count} Qdrant snapshot{'s' if qdrant_count != 1 else ''}, "
        f"{meilisearch_count} MeiliSearch index{'es' if meilisearch_count != 1 else ''}"
    )
    if corpora:
        print_info("Corpora backed up:", output_json)
        for corpus in corpora:
            file_sizes = ", ".join(
                f"{artifact['label']} {format_size(artifact['size_bytes'])}"
                for artifact in corpus["files"]
            )
            print_info(
                f"  {corpus['name']}: {file_sizes} ({format_size(corpus['size_bytes'])} total)",
                output_json,
            )
    print_success(f"Backup complete: {backup_path} ({summary})", output_json, data=data)


@container_group.command("restore")
@click.argument("backup_directory", type=click.Path(exists=True, file_okay=False))
@click.option("--qdrant-url", default="http://localhost:6333", help="Qdrant URL")
@click.option("--meilisearch-url", default="http://localhost:7700", help="MeiliSearch URL")
@click.option("--qdrant-container", default="qdrant-arcaneum", help="Qdrant container name")
@click.option(
    "--qdrant-timeout",
    default=300,
    show_default=True,
    help="Qdrant operation timeout in seconds",
)
@click.option(
    "--meilisearch-timeout",
    default=1800,
    show_default=True,
    help="MeiliSearch task timeout in seconds",
)
@click.option("--skip-meilisearch", is_flag=True, help="Only restore Qdrant snapshots")
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
def restore_command(
    backup_directory,
    qdrant_url,
    meilisearch_url,
    qdrant_container,
    qdrant_timeout,
    meilisearch_timeout,
    skip_meilisearch,
    output_json,
):
    """Restore Qdrant snapshots and MeiliSearch indexes from a backup."""
    if not check_docker_available(output_json):
        _exit_on_error()

    backup_path = Path(backup_directory).expanduser()
    manifest_path = backup_path / "manifest.json"
    if not manifest_path.exists():
        print_error(f"Backup manifest not found: {manifest_path}", output_json)
        raise SystemExit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    warnings: list[str] = []
    _restore_qdrant(
        backup_path,
        qdrant_url,
        qdrant_container,
        manifest.get("qdrant", []),
        timeout=qdrant_timeout,
        output_json=output_json,
        warnings=warnings,
    )
    if not skip_meilisearch:
        _restore_meilisearch(
            backup_path,
            meilisearch_url,
            manifest.get("meilisearch", []),
            timeout_seconds=meilisearch_timeout,
        )

    data = {
        "path": str(backup_path),
        "qdrant_snapshots": len(manifest.get("qdrant", [])),
        "meilisearch_indexes": 0 if skip_meilisearch else len(manifest.get("meilisearch", [])),
        "warnings": warnings,
    }
    print_success(f"Restore complete: {backup_path}", output_json, data=data)


def _meili_server_version(meilisearch_url: str, headers: dict) -> tuple[int, int, int] | None:
    try:
        response = _request_json("GET", f"{meilisearch_url}/version", headers=headers)
    except requests.RequestException:
        return None
    return _parse_meili_version(response.get("pkgVersion", ""))


def _meili_index_counts(meilisearch_url: str, headers: dict) -> dict[str, int]:
    stats = _request_json("GET", f"{meilisearch_url}/stats", headers=headers, timeout=120)
    return {
        uid: index.get("numberOfDocuments", 0)
        for uid, index in sorted(stats.get("indexes", {}).items())
    }


def _wait_for_meili_health(meilisearch_url: str, timeout_seconds: int = 300) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{meilisearch_url}/health", timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def _wait_for_meili_upgrade_task(
    meilisearch_url: str,
    headers: dict,
    output_json: bool = False,
    unresponsive_seconds: int = 120,
) -> dict:
    """Wait for the upgradeDatabase task an --upgrade-db start enqueues.

    There is no overall timeout: restarting MeiliSearch mid-upgrade can corrupt
    the database, so the only safe stop is the task's own terminal state.
    """
    last_progress = time.monotonic()
    last_report = last_progress
    while True:
        try:
            tasks = _request_json(
                "GET",
                f"{meilisearch_url}/tasks",
                headers=headers,
                params={"types": "upgradeDatabase", "limit": 1},
            ).get("results", [])
        except requests.RequestException:
            tasks = None
        if tasks:
            last_progress = time.monotonic()
            if tasks[0].get("status") in ("succeeded", "failed", "canceled"):
                return tasks[0]
        elif time.monotonic() - last_progress > unresponsive_seconds:
            state = "stopped responding" if tasks is None else "never reported an upgrade task"
            raise RuntimeError(f"MeiliSearch {state} during the upgrade")
        if time.monotonic() - last_report >= 60:
            print_info("Still upgrading the MeiliSearch database...", output_json)
            last_report = time.monotonic()
        time.sleep(2)


def _all_cores_env() -> dict:
    """Let MeiliSearch use every Docker CPU while it serves no searches."""
    cpus = _docker_cpu_count()
    if not cpus:
        return {}
    return {
        name: str(cpus)
        for name in ("MEILI_MAX_INDEXING_THREADS", "MEILI_CPUS")
        if name not in os.environ
    }


def _meili_upgrade_recovery_message(reason, backup_path: Path) -> str:
    return (
        f"MeiliSearch upgrade did not complete: {reason}\n"
        "The database may be partially upgraded; MeiliSearch upgrades are not atomic.\n"
        "Check 'arc container logs' first. To rebuild MeiliSearch from the backup:\n"
        "  arc container stop\n"
        f"  docker volume rm {_MEILI_DATA_VOLUME}\n"
        "  arc container start\n"
        f"  arc container restore {backup_path}"
    )


@container_group.command("upgrade")
@click.option("--meilisearch-url", default="http://localhost:7700", help="MeiliSearch URL")
@click.option(
    "--dry-run", is_flag=True, help="Show the upgrade plan without backing up or restarting"
)
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
def upgrade_command(meilisearch_url, dry_run, output_json):
    """Migrate MeiliSearch data to the pinned image version.

    Exports every index to a JSONL backup, restarts MeiliSearch on the pinned
    image with --upgrade-db, and verifies per-index document counts. If the
    upgrade fails, 'arc container restore' rebuilds the indexes from the backup.
    """
    if not check_docker_available(output_json):
        _exit_on_error()

    image_tag = _compose_meili_image_tag(output_json)
    target = _parse_meili_version(image_tag) if image_tag else None
    if target is None:
        print_error(
            f"Cannot determine the pinned MeiliSearch version from {image_tag!r}", output_json
        )
        _exit_on_error()

    current = _meili_data_version(image_tag)
    if current is None or current == target:
        message = (
            "No MeiliSearch data to upgrade"
            if current is None
            else (f"MeiliSearch data is already {image_tag}")
        )
        print_success(message, output_json, data={"upgraded": False, "to": image_tag})
        return
    from_tag = _format_meili_version(current)
    if current > target:
        print_error(
            f"MeiliSearch data is {from_tag}, newer than the configured image {image_tag}; "
            "downgrades are not supported.",
            output_json,
        )
        _exit_on_error()
    if current < _MEILI_MIN_UPGRADABLE:
        print_error(
            f"MeiliSearch data is {from_tag}, too old to upgrade in place (needs v1.12+). "
            f"Run 'arc container backup' with MEILI_IMAGE_TAG={from_tag}, remove the "
            f"{_MEILI_DATA_VOLUME} volume, then 'arc container start' and "
            "'arc container restore <backup>'.",
            output_json,
        )
        _exit_on_error()

    headers = _meilisearch_headers()
    container_env = {**get_container_env(), **get_resource_limit_env()}
    target_env = {**container_env, "MEILI_IMAGE_TAG": image_tag}

    # The export and the baseline counts need the server on the data's own version.
    if _meili_server_version(meilisearch_url, headers) != current:
        if dry_run:
            print_success(
                f"Would start MeiliSearch {from_tag}, back up every index, "
                f"and upgrade to {image_tag}",
                output_json,
                data={"dry_run": True, "from": from_tag, "to": image_tag, "indexes": None},
            )
            return
        print_info(f"Starting MeiliSearch {from_tag} to export the current data...", output_json)
        started = run_compose_command(
            ["up", "-d", "meilisearch"],
            env={**container_env, "MEILI_IMAGE_TAG": from_tag, "MEILI_UPGRADE_DB": "false"},
            capture_output=output_json,
            output_json=output_json,
        )
        if started is None or not _wait_for_meili_health(meilisearch_url):
            print_error(
                f"MeiliSearch {from_tag} did not become healthy; check: arc container logs",
                output_json,
            )
            _exit_on_error()

    try:
        _ensure_meilisearch_idle(meilisearch_url, headers)
        counts = _meili_index_counts(meilisearch_url, headers)
    except (RuntimeError, requests.RequestException) as exc:
        print_error(f"Cannot upgrade MeiliSearch: {exc}", output_json)
        _exit_on_error()

    plan = {"from": from_tag, "to": image_tag, "indexes": counts}
    if dry_run:
        print_success(
            f"Would back up {len(counts)} indexes and upgrade {from_tag} -> {image_tag}",
            output_json,
            data={**plan, "dry_run": True},
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = _resolve_backup_path(None, timestamp, output_json)
    backup_path.mkdir(parents=True, exist_ok=False)
    print_info(f"Backing up MeiliSearch indexes to {backup_path}", output_json)
    try:
        exported = _backup_meilisearch(backup_path, meilisearch_url)
    except (RuntimeError, requests.RequestException) as exc:
        print_error(f"MeiliSearch backup failed; nothing was changed: {exc}", output_json)
        _exit_on_error()
    exported_counts = {entry["index"]: entry["documents"] for entry in exported}
    if exported_counts != counts:
        mismatched = sorted(
            uid
            for uid in set(counts) | set(exported_counts)
            if counts.get(uid) != exported_counts.get(uid)
        )
        print_error(
            "Backup does not match the live document counts for "
            f"{', '.join(mismatched)}; nothing was changed.",
            output_json,
        )
        _exit_on_error()
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qdrant_url": None,
        "meilisearch_url": meilisearch_url,
        "meilisearch_version": from_tag,
        "protected": ["MeiliSearch index settings and documents"],
        "not_protected": [
            "Qdrant collections (this backup precedes a MeiliSearch upgrade)",
            "Local source files referenced by indexed metadata",
        ],
        "qdrant": [],
        "meilisearch": exported,
    }
    (backup_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Pull before stopping anything, so a registry failure costs no downtime.
    if (
        run_compose_command(
            ["pull", "meilisearch"],
            env=target_env,
            capture_output=output_json,
            output_json=output_json,
        )
        is None
    ):
        print_error(f"Could not pull {_MEILI_IMAGE}:{image_tag}; nothing was changed.", output_json)
        _exit_on_error()

    print_warning(
        f"Upgrading MeiliSearch {from_tag} -> {image_tag} in place. "
        "Do not stop MeiliSearch until this command finishes.",
        output_json,
    )
    try:
        if (
            run_compose_command(
                ["up", "-d", "meilisearch"],
                env={**target_env, **_all_cores_env(), "MEILI_UPGRADE_DB": "true"},
                capture_output=output_json,
                output_json=output_json,
            )
            is None
        ):
            raise RuntimeError(f"docker compose could not start {image_tag}")
        if not _wait_for_meili_health(meilisearch_url):
            raise RuntimeError(f"MeiliSearch {image_tag} did not become healthy")
        task = _wait_for_meili_upgrade_task(meilisearch_url, headers, output_json)
        if task.get("status") != "succeeded":
            error = (task.get("error") or {}).get("message")
            raise RuntimeError(
                error or f"upgradeDatabase task {task.get('uid')} {task.get('status')}"
            )
        server_version = _meili_server_version(meilisearch_url, headers)
        if server_version != target:
            raise RuntimeError(f"server reports {server_version}, expected {image_tag}")
        after = _meili_index_counts(meilisearch_url, headers)
        if after != counts:
            changed = ", ".join(
                f"{uid} {counts.get(uid)} -> {after.get(uid)}"
                for uid in sorted(set(counts) | set(after))
                if counts.get(uid) != after.get(uid)
            )
            raise RuntimeError(f"document counts changed: {changed}")
    except (RuntimeError, requests.RequestException) as exc:
        print_error(_meili_upgrade_recovery_message(exc, backup_path), output_json)
        _exit_on_error()

    # Recreate without the flag, so a later image bump never upgrades unasked.
    restarted = run_compose_command(
        ["up", "-d", "meilisearch"],
        env={**target_env, "MEILI_UPGRADE_DB": "false"},
        capture_output=output_json,
        output_json=output_json,
    )
    if restarted is None or not _wait_for_meili_health(meilisearch_url):
        print_warning(
            "Upgrade succeeded, but MeiliSearch did not restart cleanly; run: arc container start",
            output_json,
        )

    print_success(
        f"MeiliSearch upgraded {from_tag} -> {image_tag} "
        f"({len(counts)} indexes, document counts verified)",
        output_json,
        data={**plan, "upgraded": True, "backup": str(backup_path)},
    )
    print_info("Next: check each corpus with 'arc corpus verify <corpus>'", output_json)
    print_info(f"Backup kept at {backup_path}; delete it once searches look right", output_json)


@container_group.command("reset")
@click.option("--confirm", is_flag=True, help="Confirm deletion of all data")
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
def reset_command(confirm, output_json=False):
    """Reset all container data (WARNING: deletes all collections)"""
    if not confirm:
        print_error("Use --confirm to delete ALL data including collections", output_json)
        _exit_on_error(code=2)

    if not check_docker_available(output_json):
        _exit_on_error()

    # Stop services first
    print_warning("Stopping services...", output_json)
    result = run_compose_command(["down"], capture_output=output_json, output_json=output_json)
    if result is None:
        _exit_on_error()

    # Delete data directories
    data_dir = get_data_dir()
    qdrant_dir = data_dir / "qdrant"
    snapshots_dir = data_dir / "qdrant_snapshots"

    try:
        deleted = []
        if qdrant_dir.exists():
            size = get_dir_size(qdrant_dir)
            print_warning(f"Deleting Qdrant data ({format_size(size)})...", output_json)
            shutil.rmtree(qdrant_dir)
            deleted.append({"path": str(qdrant_dir), "size_bytes": size})

        if snapshots_dir.exists():
            print_warning("Deleting Qdrant snapshots...", output_json)
            size = get_dir_size(snapshots_dir)
            shutil.rmtree(snapshots_dir)
            deleted.append({"path": str(snapshots_dir), "size_bytes": size})

        # Recreate empty directories
        qdrant_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        print_success(
            "Data reset complete",
            json_output=output_json,
            data={
                "deleted": deleted,
                "created": [str(qdrant_dir), str(snapshots_dir)],
                "next": "arc container start",
            },
        )
        print_info("Run 'arc container start' to restart services", output_json)

    except Exception as e:
        print_error(f"Failed to reset data: {e}", output_json)
        _exit_on_error()


def get_dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except (PermissionError, OSError):
        pass
    return total

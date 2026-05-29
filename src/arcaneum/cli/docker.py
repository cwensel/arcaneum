"""Container management commands for Arcaneum services."""

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import requests

from arcaneum.cli.errors import HelpfulGroup
from arcaneum.cli.output import print_error, print_info, print_success, print_warning, print_json
from arcaneum.paths import get_data_dir
from arcaneum.utils.formatting import format_size

_TASK_UID_UNSET = object()


def _exit_on_json_error(output_json: bool, code: int = 1):
    if output_json:
        raise SystemExit(code)


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
        _exit_on_json_error(output_json)
        return

    # Get container environment (auto-generates MeiliSearch key if needed)
    container_env = get_container_env()

    print_info("Starting container services...", output_json)
    result = run_compose_command(
        ["up", "-d"],
        env=container_env,
        capture_output=output_json,
        output_json=output_json,
    )

    if result is None:
        _exit_on_json_error(output_json)
        return

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
        _exit_on_json_error(output_json)
        return

    print_info("Stopping container services...", output_json)
    result = run_compose_command(["down"], capture_output=output_json, output_json=output_json)

    if result is not None:
        print_success("Container services stopped", json_output=output_json, data={"stopped": True})
    else:
        _exit_on_json_error(output_json)


@container_group.command("restart")
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
def restart_command(output_json=False):
    """Restart container services"""
    if not check_docker_available(output_json):
        _exit_on_json_error(output_json)
        return

    print_info("Restarting container services...", output_json)
    result = run_compose_command(["restart"], capture_output=output_json, output_json=output_json)

    if result is None:
        _exit_on_json_error(output_json)
        return

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
        _exit_on_json_error(output_json)
        return

    print_info("Container Services Status:", output_json)
    if not output_json:
        print()
    ps_result = run_compose_command(["ps"], capture_output=output_json, output_json=output_json)
    if ps_result is None:
        _exit_on_json_error(output_json)
        return

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
        _exit_on_json_error(output_json)
        return

    args = ["logs", f"--tail={tail}"]
    if follow:
        args.append("-f")

    result = run_compose_command(args, capture_output=output_json, output_json=output_json)
    if result is None:
        _exit_on_json_error(output_json)
        return

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


def _meilisearch_headers():
    from arcaneum.paths import get_meilisearch_api_key

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


def _backup_qdrant(
    backup_path: Path,
    qdrant_url: str,
    container_name: str,
    timeout: int,
) -> list[dict]:
    qdrant_dir = backup_path / "qdrant"
    qdrant_dir.mkdir(parents=True, exist_ok=True)

    collections_response = _request_json("GET", f"{qdrant_url}/collections")
    collections = collections_response.get("result", {}).get("collections", [])
    snapshots = []

    for collection in collections:
        name = collection["name"]
        snapshot_response = _request_json(
            "POST",
            f"{qdrant_url}/collections/{name}/snapshots",
            timeout=timeout,
        )
        snapshot_name = snapshot_response["result"]["name"]
        destination = qdrant_dir / snapshot_name
        _copy_from_container(
            container_name,
            f"/qdrant/snapshots/{name}/{snapshot_name}",
            destination,
        )
        _request_json(
            "DELETE",
            f"{qdrant_url}/collections/{name}/snapshots/{snapshot_name}",
            timeout=timeout,
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
    indexes = []
    offset = 0
    limit = 100

    while True:
        indexes_response = _request_json(
            "GET",
            f"{meilisearch_url}/indexes",
            headers=headers,
            params={"limit": limit, "offset": offset},
        )
        batch = indexes_response.get("results", [])
        indexes.extend(batch)
        if len(batch) < limit:
            break
        offset += limit

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
) -> None:
    for snapshot in snapshots:
        collection = snapshot["collection"]
        snapshot_name = snapshot["snapshot"]
        snapshot_file = backup_path / snapshot["file"]
        if not snapshot_file.exists():
            raise FileNotFoundError(f"Missing Qdrant snapshot: {snapshot_file}")

        container_path = f"/qdrant/snapshots/{collection}/{snapshot_name}"
        _mkdir_in_container(container_name, f"/qdrant/snapshots/{collection}")
        _copy_to_container(snapshot_file, container_name, container_path)
        _request_json(
            "PUT",
            f"{qdrant_url}/collections/{collection}/snapshots/recover",
            json={"location": f"file://{container_path}"},
            params={"wait": "true"},
            timeout=timeout,
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
@click.option("--output", "-o", type=click.Path(), help="Backup directory to create")
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
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
def backup_command(
    output,
    qdrant_url,
    meilisearch_url,
    qdrant_container,
    qdrant_timeout,
    skip_meilisearch,
    output_json,
):
    """Back up Qdrant snapshots and MeiliSearch indexes."""
    if not check_docker_available(output_json):
        _exit_on_json_error(output_json)
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = Path(output).expanduser() if output else get_data_dir() / "backups" / timestamp
    backup_path.mkdir(parents=True, exist_ok=False)

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

    meili_starting_task_uid = None
    if not skip_meilisearch:
        meili_headers = _meilisearch_headers()
        meili_starting_task_uid = _latest_meilisearch_task_uid(meilisearch_url, meili_headers)
        _ensure_meilisearch_idle(meilisearch_url, meili_headers)

    manifest["qdrant"] = _backup_qdrant(
        backup_path,
        qdrant_url,
        qdrant_container,
        timeout=qdrant_timeout,
    )
    if not skip_meilisearch:
        manifest["meilisearch"] = _backup_meilisearch(
            backup_path,
            meilisearch_url,
            starting_task_uid=meili_starting_task_uid,
        )

    manifest_path = backup_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    data = {
        "path": str(backup_path),
        "qdrant_snapshots": len(manifest["qdrant"]),
        "meilisearch_indexes": len(manifest["meilisearch"]),
    }
    print_success(f"Backup complete: {backup_path}", output_json, data=data)


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
        _exit_on_json_error(output_json)
        return

    backup_path = Path(backup_directory).expanduser()
    manifest_path = backup_path / "manifest.json"
    if not manifest_path.exists():
        print_error(f"Backup manifest not found: {manifest_path}", output_json)
        raise SystemExit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _restore_qdrant(
        backup_path,
        qdrant_url,
        qdrant_container,
        manifest.get("qdrant", []),
        timeout=qdrant_timeout,
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
    }
    print_success(f"Restore complete: {backup_path}", output_json, data=data)


@container_group.command("reset")
@click.option("--confirm", is_flag=True, help="Confirm deletion of all data")
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
def reset_command(confirm, output_json=False):
    """Reset all container data (WARNING: deletes all collections)"""
    if not confirm:
        print_error("Use --confirm to delete ALL data including collections", output_json)
        _exit_on_json_error(output_json, code=2)
        return

    if not check_docker_available(output_json):
        _exit_on_json_error(output_json)
        return

    # Stop services first
    print_warning("Stopping services...", output_json)
    result = run_compose_command(["down"], capture_output=output_json, output_json=output_json)
    if result is None:
        _exit_on_json_error(output_json)
        return

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
        _exit_on_json_error(output_json)


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

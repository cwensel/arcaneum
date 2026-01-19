"""Container management commands for Arcaneum services."""

import subprocess
import time
import shutil
import click
from pathlib import Path
import requests
from arcaneum.cli.output import print_info, print_success, print_error, print_warning
from arcaneum.paths import get_data_dir
from arcaneum.utils.formatting import format_size


def check_docker_available():
    """Check if Docker is installed and running."""
    if not shutil.which("docker"):
        print_error("Docker is not installed. Please install Docker Desktop or Docker Engine.")
        print_info("Visit: https://docs.docker.com/get-docker/")
        return False

    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            check=True,
            timeout=5
        )
        return True
    except subprocess.CalledProcessError:
        print_error("Docker is installed but not running. Please start Docker.")
        return False
    except subprocess.TimeoutExpired:
        print_error("Docker is not responding. Please check Docker status.")
        return False


def get_compose_file():
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

    print_error("docker-compose.yml not found")
    print_info(f"Expected locations: {repo_root}/deploy/docker-compose.yml or ./docker-compose.yml")
    return None


def run_compose_command(args, check=True, capture_output=False, env=None):
    """Run a docker compose command.

    Args:
        args: Command arguments to pass to docker compose
        check: Whether to raise on non-zero exit code
        capture_output: Whether to capture stdout/stderr
        env: Optional environment variables to add (merged with current env)
    """
    compose_file = get_compose_file()
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
            cmd,
            check=check,
            capture_output=capture_output,
            text=True,
            env=run_env
        )
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Container command failed: {e}")
        if e.stderr:
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


from arcaneum.cli.errors import HelpfulGroup


@click.group(name='container', cls=HelpfulGroup, usage_examples=[
    'arc container start',
    'arc container stop',
    'arc container status',
    'arc container logs -f',
])
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


@container_group.command('start')
def start_command():
    """Start container services"""
    if not check_docker_available():
        return

    # Get container environment (auto-generates MeiliSearch key if needed)
    container_env = get_container_env()

    print_info("Starting container services...")
    result = run_compose_command(["up", "-d"], env=container_env)

    if result is None:
        return

    # Wait for services to start
    time.sleep(3)

    # Check Qdrant
    if check_qdrant_health():
        print_success("Qdrant started successfully")
        print(f"  REST API: http://localhost:6333")
        print(f"  Dashboard: http://localhost:6333/dashboard")
    else:
        print_warning("Qdrant may not be ready yet. Check logs with: arc container logs")

    # Check MeiliSearch
    if check_meilisearch_health():
        print_success("MeiliSearch started successfully")
        print(f"  HTTP API: http://localhost:7700")
    else:
        print_warning("MeiliSearch may not be ready yet. Check logs with: arc container logs")

    print()
    print_info(f"Data directory: {get_data_dir()}")


@container_group.command('stop')
def stop_command():
    """Stop container services"""
    if not check_docker_available():
        return

    print_info("Stopping container services...")
    result = run_compose_command(["down"])

    if result is not None:
        print_success("Container services stopped")


@container_group.command('restart')
def restart_command():
    """Restart container services"""
    if not check_docker_available():
        return

    print_info("Restarting container services...")
    result = run_compose_command(["restart"])

    if result is None:
        return

    time.sleep(2)

    if check_qdrant_health():
        print_success("Container services restarted")
    else:
        print_warning("Services may not be ready yet")


@container_group.command('status')
def status_command():
    """Show container services status"""
    if not check_docker_available():
        return

    print_info("Container Services Status:")
    print()
    run_compose_command(["ps"])

    print()
    if check_qdrant_health():
        print_success("Qdrant: Healthy")
    else:
        print_error("Qdrant: Unhealthy or not running")

    if check_meilisearch_health():
        print_success("MeiliSearch: Healthy")
    else:
        print_error("MeiliSearch: Unhealthy or not running")

    # Show Docker volume information
    print()
    print_info("Docker Volumes:")

    # Get volume information including sizes using docker system df
    try:
        import json

        df_result = subprocess.run(
            ["docker", "system", "df", "-v", "--format", "json"],
            capture_output=True,
            text=True,
            check=True
        )

        df_data = json.loads(df_result.stdout)
        volumes_data = {v['Name']: v for v in df_data.get('Volumes', [])}

        # List volumes for this project
        result = subprocess.run(
            ["docker", "volume", "ls", "--filter", "name=arcaneum", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            check=True
        )

        volumes = result.stdout.strip().split('\n')
        for volume in volumes:
            if volume:
                # Get volume details
                volume_info = volumes_data.get(volume, {})
                size = volume_info.get('Size', 'unknown')

                # Get mountpoint
                inspect_result = subprocess.run(
                    ["docker", "volume", "inspect", volume, "--format", "{{.Mountpoint}}"],
                    capture_output=True,
                    text=True,
                    check=False
                )

                print(f"  {volume}")
                if size != 'unknown':
                    print(f"    Size: {size}")
                if inspect_result.returncode == 0:
                    mountpoint = inspect_result.stdout.strip()
                    print(f"    Mountpoint: {mountpoint}")
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print_warning(f"Could not retrieve volume information: {e}")


@container_group.command('logs')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.option('--tail', type=int, default=100, help='Number of lines to show')
def logs_command(follow, tail):
    """Show container services logs"""
    if not check_docker_available():
        return

    args = ["logs", f"--tail={tail}"]
    if follow:
        args.append("-f")

    run_compose_command(args)


@container_group.command('reset')
@click.option('--confirm', is_flag=True, help='Confirm deletion of all data')
def reset_command(confirm):
    """Reset all container data (WARNING: deletes all collections)"""
    if not confirm:
        print_error("This will delete ALL data including collections!")
        print_error("Use --confirm to proceed")
        return

    if not check_docker_available():
        return

    # Stop services first
    print_warning("Stopping services...")
    run_compose_command(["down"])

    # Delete data directories
    data_dir = get_data_dir()
    qdrant_dir = data_dir / "qdrant"
    snapshots_dir = data_dir / "qdrant_snapshots"

    try:
        if qdrant_dir.exists():
            size = get_dir_size(qdrant_dir)
            print_warning(f"Deleting Qdrant data ({format_size(size)})...")
            shutil.rmtree(qdrant_dir)

        if snapshots_dir.exists():
            print_warning(f"Deleting Qdrant snapshots...")
            shutil.rmtree(snapshots_dir)

        # Recreate empty directories
        qdrant_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        print_success("Data reset complete")
        print_info("Run 'arc container start' to restart services")

    except Exception as e:
        print_error(f"Failed to reset data: {e}")


def get_dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    try:
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
    except (PermissionError, OSError):
        pass
    return total

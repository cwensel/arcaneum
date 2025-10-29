"""Container management commands for Arcaneum services."""

import subprocess
import time
import shutil
import click
from pathlib import Path
import requests
from arcaneum.cli.output import print_info, print_success, print_error, print_warning
from arcaneum.paths import get_data_dir


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
    # Try current directory first, then deploy/
    compose_paths = [
        Path("docker-compose.yml"),
        Path("deploy/docker-compose.yml"),
    ]

    for path in compose_paths:
        if path.exists():
            return str(path)

    print_error("docker-compose.yml not found")
    print_info("Expected locations: ./docker-compose.yml or ./deploy/docker-compose.yml")
    return None


def run_compose_command(args, check=True, capture_output=False):
    """Run a docker compose command."""
    compose_file = get_compose_file()
    if not compose_file:
        return None

    cmd = ["docker", "compose", "-f", compose_file] + args

    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Container command failed: {e}")
        if e.stderr:
            print(e.stderr)
        return None


def check_qdrant_health():
    """Check if Qdrant is healthy."""
    try:
        response = requests.get("http://localhost:6333/healthz", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


@click.group(name='container')
def container_group():
    """Manage container services (Qdrant, MeiliSearch)"""
    pass


@container_group.command('start')
def start_command():
    """Start container services"""
    if not check_docker_available():
        return

    print_info("Starting container services...")
    result = run_compose_command(["up", "-d"])

    if result is None:
        return

    # Wait for services to start
    time.sleep(3)

    if check_qdrant_health():
        print_success("Qdrant started successfully")
        print(f"  REST API: http://localhost:6333")
        print(f"  Dashboard: http://localhost:6333/dashboard")
        print(f"  Data: {get_data_dir()}")
    else:
        print_warning("Qdrant may not be ready yet. Check logs with: arc container logs")


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

    # Show data directory sizes
    data_dir = get_data_dir()
    if data_dir.exists():
        print()
        print_info("Data Directory:")
        print(f"  Location: {data_dir}")

        qdrant_dir = data_dir / "qdrant"
        if qdrant_dir.exists():
            size = get_dir_size(qdrant_dir)
            print(f"  Qdrant: {format_size(size)}")


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


def format_size(bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} PB"

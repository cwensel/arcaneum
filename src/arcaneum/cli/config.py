"""Configuration and cache management commands."""

import shutil
import click
from pathlib import Path
from arcaneum.paths import get_models_dir, get_data_dir, get_legacy_arcaneum_dir
from arcaneum.cli.output import print_info, print_success, print_error


def show_cache_dir():
    """Show the cache directory location (XDG-compliant structure)."""
    models_dir = get_models_dir()
    data_dir = get_data_dir()
    legacy_dir = get_legacy_arcaneum_dir()

    print_info(f"Arcaneum directories (XDG-compliant):")
    print(f"  Cache (models): {models_dir}")
    print(f"  Data (databases): {data_dir}")

    # Show legacy directory if it exists
    if legacy_dir.exists():
        print(f"  Legacy (old): {legacy_dir} [Run any command to auto-migrate]")

    # Show sizes if directories exist
    if models_dir.exists():
        size = get_dir_size(models_dir)
        print(f"  Cache size: {format_size(size)}")

    if data_dir.exists():
        size = get_dir_size(data_dir)
        print(f"  Data size: {format_size(size)}")


def clear_cache(confirm: bool = False):
    """Clear the model cache directory."""
    models_dir = get_models_dir()

    if not models_dir.exists():
        print_info("Model cache is already empty")
        return

    size = get_dir_size(models_dir)
    size_str = format_size(size)

    if not confirm:
        print_error(f"This will delete {size_str} of cached models from {models_dir}")
        print_error("Use --confirm to proceed")
        return

    try:
        # Remove all contents but keep the directory
        for item in models_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

        print_success(f"Cleared {size_str} from model cache")
    except Exception as e:
        print_error(f"Failed to clear cache: {e}")


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


@click.group(name='config')
def config_group():
    """Configuration and cache management"""
    pass


@config_group.command('show-cache-dir')
def show_cache_dir_command():
    """Show cache directory location and sizes"""
    show_cache_dir()


@config_group.command('clear-cache')
@click.option('--confirm', is_flag=True, help='Confirm deletion')
def clear_cache_command(confirm):
    """Clear model cache directory"""
    clear_cache(confirm)

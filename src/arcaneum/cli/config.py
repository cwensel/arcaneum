"""Configuration and cache management commands."""

import shutil
import click
from pathlib import Path
from arcaneum.paths import get_models_dir, get_data_dir, get_legacy_arcaneum_dir
from arcaneum.cli.output import print_info, print_success, print_error, print_json
from arcaneum.utils.formatting import format_size


def _directory_info(path: Path) -> dict:
    exists = path.exists()
    size_bytes = get_dir_size(path) if exists else 0
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": size_bytes,
        "size": format_size(size_bytes),
    }


def show_cache_dir(output_json: bool = False):
    """Show the cache directory location (XDG-compliant structure)."""
    models_dir = get_models_dir()
    data_dir = get_data_dir()
    legacy_dir = get_legacy_arcaneum_dir()

    data = {
        "cache": _directory_info(models_dir),
        "data": _directory_info(data_dir),
        "legacy": _directory_info(legacy_dir),
    }
    if output_json:
        print_json("success", "Arcaneum directories", data)
        return

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


def clear_cache(confirm: bool = False, output_json: bool = False):
    """Clear the model cache directory."""
    models_dir = get_models_dir()

    if not models_dir.exists():
        if output_json:
            print_json(
                "success",
                "Model cache is already empty",
                {"path": str(models_dir), "cleared_bytes": 0},
            )
        else:
            print_info("Model cache is already empty")
        return

    size = get_dir_size(models_dir)
    size_str = format_size(size)

    if not confirm:
        message = f"Use --confirm to delete {size_str} of cached models from {models_dir}"
        print_error(message, output_json)
        if output_json:
            raise SystemExit(2)
        return

    try:
        # Remove all contents but keep the directory
        for item in models_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

        print_success(
            f"Cleared {size_str} from model cache",
            json_output=output_json,
            data={"path": str(models_dir), "cleared_bytes": size, "cleared_size": size_str},
        )
    except Exception as e:
        print_error(f"Failed to clear cache: {e}", output_json)
        if output_json:
            raise SystemExit(1)


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


from arcaneum.cli.errors import HelpfulGroup


@click.group(name='config', cls=HelpfulGroup, usage_examples=[
    'arc config show-cache-dir',
    'arc config clear-cache --confirm',
])
def config_group():
    """Configuration and cache management"""
    pass


@config_group.command('show-cache-dir')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def show_cache_dir_command(output_json):
    """Show cache directory location and sizes"""
    show_cache_dir(output_json)


@config_group.command('clear-cache')
@click.option('--confirm', is_flag=True, help='Confirm deletion')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def clear_cache_command(confirm, output_json):
    """Clear model cache directory"""
    clear_cache(confirm, output_json)

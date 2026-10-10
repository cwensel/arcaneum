"""Backup inventory, verification, retention, and Qdrant snapshot cleanup."""

import functools
import shutil
from pathlib import Path
from urllib.parse import quote

import click
import requests

from arcaneum.backup import list_backups, select_prunable, verify_backup
from arcaneum.cli.docker import _request_json, _resolve_backup_path
from arcaneum.cli.output import print_error, print_info, print_json, print_success, print_warning
from arcaneum.utils.formatting import format_size


def _report_errors(callback):
    @functools.wraps(callback)
    def wrapped(*args, **kwargs):
        try:
            return callback(*args, **kwargs)
        except (OSError, ValueError, requests.RequestException) as exc:
            print_error(str(exc), kwargs.get("output_json", False))
            raise click.exceptions.Exit(1) from exc

    return wrapped


def _root_path(root, output_json):
    return (
        Path(root).expanduser()
        if root
        else _resolve_backup_path(None, "", output_json, dry_run=True)
    )


def _finish(message, data, output_json):
    errors = data.get("errors", [])
    for warning in data.get("warnings", []):
        print_warning(warning, output_json)
    if errors:
        if output_json:
            print_json("error", message, data=data, errors=errors)
        else:
            for error in errors:
                print_error(error)
        raise click.exceptions.Exit(1)
    print_success(message, output_json, data=data)


@click.command("backup-list")
@click.option(
    "--root", type=click.Path(file_okay=False), help="Backup root (defaults to backup.path)"
)
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
@_report_errors
def backup_list_command(root, output_json):
    """List local backups, dates, artifact sizes, and corpora. No services required."""
    directory = _root_path(root, output_json)
    backups, skipped = list_backups(directory)
    for backup in backups:
        print_info(
            f"{backup['created_at']}  {backup['path']}  {format_size(backup['size_bytes'])}  "
            f"{', '.join(backup['corpora']) or '(empty)'}",
            output_json,
        )
    for item in skipped:
        print_warning(f"Skipped {item['path']}: {item['reason']}", output_json)
    print_success(
        f"Found {len(backups)} backups in {directory}",
        output_json,
        data={"root": str(directory), "backups": backups, "skipped": skipped},
    )


@click.command("backup-verify")
@click.argument("backup_directory", type=click.Path(exists=True, file_okay=False))
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
@_report_errors
def backup_verify_command(backup_directory, output_json):
    """Check manifest, artifact checksums, and MeiliSearch document counts locally."""
    report = verify_backup(Path(backup_directory).expanduser())
    _finish(
        "Backup verified" if report["valid"] else "Backup verification failed", report, output_json
    )


@click.command("backup-prune")
@click.option(
    "--root", type=click.Path(file_okay=False), help="Backup root (defaults to backup.path)"
)
@click.option(
    "--keep", type=click.IntRange(min=1), help="Retain at least the newest N complete backups"
)
@click.option(
    "--older-than", type=click.IntRange(min=1), help="Only remove backups older than N days"
)
@click.option("--dry-run", is_flag=True, help="List selected backups without deleting them")
@click.option("--confirm", is_flag=True, help="Confirm deletion of selected backup directories")
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
@_report_errors
def backup_prune_command(root, keep, older_than, dry_run, confirm, output_json):
    """Prune complete backups. With both filters, a backup must satisfy both."""
    if keep is None and older_than is None:
        raise ValueError("Specify --keep and/or --older-than")
    if not dry_run and not confirm:
        raise ValueError("Use --dry-run to preview or --confirm to delete backups")
    directory = _root_path(root, output_json)
    backups, skipped = list_backups(directory)
    verified = []
    identities = {}
    for backup in backups:
        path = Path(backup["path"])
        report = verify_backup(path)
        if not report["valid"]:
            skipped.append({"path": str(path), "reason": "; ".join(report["errors"])})
            continue
        stat = path.stat()
        identities[str(path)] = (stat.st_dev, stat.st_ino, (path / "manifest.json").read_bytes())
        verified.append(backup)
    selected = select_prunable(verified, keep, older_than)
    data = {
        "root": str(directory),
        "dry_run": dry_run,
        "selected": selected,
        "deleted": [],
        "skipped": skipped,
        "errors": [],
    }
    for backup in selected:
        path = Path(backup["path"])
        if dry_run:
            print_info(f"Would delete {path} ({format_size(backup['size_bytes'])})", output_json)
            continue
        try:
            if path.is_symlink() or path.resolve().parent != directory.resolve():
                raise ValueError(f"Backup path changed: {path}")
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino, (path / "manifest.json").read_bytes())
            if identity != identities[str(path)]:
                raise ValueError(f"Backup changed after selection: {path}")
            shutil.rmtree(path)
            data["deleted"].append(str(path))
            print_info(f"Deleted {path}", output_json)
        except (OSError, ValueError) as exc:
            data["errors"].append(str(exc))
    for item in skipped:
        print_warning(f"Skipped {item['path']}: {item['reason']}", output_json)
    message = (
        f"Would delete {len(selected)} backups"
        if dry_run
        else f"Deleted {len(data['deleted'])} backups"
    )
    _finish(message, data, output_json)


def _snapshot_url(qdrant_url, collection, snapshot=None):
    url = f"{qdrant_url.rstrip('/')}/collections/{quote(collection, safe='')}/snapshots"
    return url if snapshot is None else f"{url}/{quote(snapshot, safe='')}"


def _snapshots(qdrant_url, collections):
    if not collections:
        result = _request_json("GET", f"{qdrant_url.rstrip('/')}/collections")
        collections = [item["name"] for item in result["result"]["collections"]]
    snapshots = []
    for collection in dict.fromkeys(collections):
        result = _request_json("GET", _snapshot_url(qdrant_url, collection))
        snapshots.extend({**item, "collection": collection} for item in result["result"] or [])
    return snapshots


@click.command("snapshot-list")
@click.option("--collection", multiple=True, help="Limit to these collections (repeatable)")
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True)
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
@_report_errors
def snapshot_list_command(collection, qdrant_url, output_json):
    """List server snapshots that may have been left by interrupted backups."""
    snapshots = _snapshots(qdrant_url, collection)
    for snapshot in snapshots:
        print_info(
            f"{snapshot['collection']}/{snapshot['name']}  "
            f"{format_size(snapshot.get('size', 0))}  {snapshot.get('creation_time', '')}",
            output_json,
        )
    print_success(f"Found {len(snapshots)} snapshots", output_json, data={"snapshots": snapshots})


@click.command("snapshot-cleanup")
@click.option("--collection", help="Limit cleanup to one collection (default: all collections)")
@click.option("--snapshot", multiple=True, help="Limit to exact snapshot names (default: all)")
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True)
@click.option("--qdrant-timeout", type=click.IntRange(min=1), default=300, show_default=True)
@click.option("--dry-run", is_flag=True, help="Show selected snapshots without deleting them")
@click.option("--confirm", is_flag=True, help="Confirm deletion of the selected snapshots")
@click.option("--json", "output_json", is_flag=True, help="Output JSON format")
@_report_errors
def snapshot_cleanup_command(
    collection, snapshot, qdrant_url, qdrant_timeout, dry_run, confirm, output_json
):
    """Clean up server snapshots (all by default). Run when backup/restore is idle."""
    if not dry_run and not confirm:
        raise ValueError("Use --dry-run to preview or --confirm to delete snapshots")
    available = _snapshots(qdrant_url, [collection] if collection else [])
    available_names = {item["name"] for item in available}
    names = list(dict.fromkeys(snapshot))
    missing = [name for name in names if name not in available_names]
    if missing:
        scope = collection or "all collections"
        raise ValueError(f"Unknown snapshots in {scope}: {', '.join(missing)}")
    selected = [item for item in available if not names or item["name"] in names]
    data = {
        "collection": collection,
        "dry_run": dry_run,
        "selected": selected,
        "deleted": [],
        "deleted_snapshots": [],
        "errors": [],
    }
    for item in selected:
        name = item["name"]
        collection_name = item["collection"]
        if dry_run:
            print_info(f"Would delete {collection_name}/{name}", output_json)
            continue
        try:
            _request_json(
                "DELETE",
                _snapshot_url(qdrant_url, collection_name, name),
                params={"wait": "true"},
                timeout=qdrant_timeout,
            )
            data["deleted"].append(name)
            data["deleted_snapshots"].append({"collection": collection_name, "name": name})
            print_info(f"Deleted {collection_name}/{name}", output_json)
        except requests.RequestException as exc:
            data["errors"].append(f"{collection_name}/{name}: {exc}")
    message = (
        f"Would delete {len(selected)} snapshots"
        if dry_run
        else f"Deleted {len(data['deleted'])} snapshots"
    )
    _finish(message, data, output_json)

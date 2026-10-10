"""Local backup manifests, integrity checks, and retention selection."""

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


def backup_time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Manifest created_at must be an ISO timestamp")
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Manifest created_at must include a timezone")
    return timestamp.astimezone(timezone.utc)


def artifact_path(directory: Path, relative: str) -> Path:
    """Reject unsafe paths, including symlinks inside a backup."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("Artifact paths must be nonempty strings")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"Unsafe backup artifact path: {relative}")
    candidate = directory
    for part in path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"Symlink in backup artifact path: {relative}")
    if not candidate.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"Artifact escapes backup directory: {relative}")
    return candidate


def manifest_artifacts(manifest: dict) -> list[str]:
    """Validate the supported manifest structure and enumerate its artifacts."""
    if not isinstance(manifest, dict) or type(manifest.get("version")) is not int:
        raise ValueError("Invalid backup manifest")
    if manifest["version"] != 1:
        raise ValueError(f"Unsupported backup manifest version: {manifest['version']}")
    backup_time(manifest.get("created_at"))
    artifacts = []
    for service, fields in (
        ("qdrant", ("collection", "snapshot", "file")),
        ("meilisearch", ("index", "metadata_file", "documents_file")),
    ):
        entries = manifest.get(service)
        if not isinstance(entries, list):
            raise ValueError(f"Manifest {service} must be a list")
        names = set()
        for entry in entries:
            if not isinstance(entry, dict) or any(
                not isinstance(entry.get(field), str) or not entry[field] for field in fields
            ):
                raise ValueError(f"Invalid {service} manifest entry")
            name = entry[fields[0]]
            if name in names:
                raise ValueError(f"Duplicate {service} entry: {name}")
            names.add(name)
            artifacts.extend(entry[field] for field in fields if field.endswith("file"))
    if len(artifacts) != len(set(artifacts)):
        raise ValueError("Manifest references duplicate artifact paths")
    if any(Path(name) == Path("manifest.json") for name in artifacts):
        raise ValueError("Manifest cannot reference itself as an artifact")
    if "checksums" in manifest:
        checksums = manifest["checksums"]
        if not isinstance(checksums, dict) or set(checksums) != set(artifacts):
            raise ValueError("Checksums must cover exactly the manifest artifacts")
        for name, spec in checksums.items():
            if (
                not isinstance(spec, dict)
                or not isinstance(spec.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]) is None
                or type(spec.get("size_bytes")) is not int
                or spec["size_bytes"] < 0
            ):
                raise ValueError(f"Invalid checksum entry: {name}")
    return artifacts


def read_manifest(directory: Path) -> dict:
    path = artifact_path(directory, "manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for relative in manifest_artifacts(manifest):
        artifact_path(directory, relative)
    return manifest


def file_checksum(path: Path) -> dict:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"sha256": digest, "size_bytes": path.stat().st_size}


def backup_checksums(directory: Path, manifest: dict) -> dict:
    return {
        name: file_checksum(artifact_path(directory, name)) for name in manifest_artifacts(manifest)
    }


def verify_backup(directory: Path) -> dict:
    """Verify local artifacts without accessing either database service."""
    report = {
        "path": str(directory),
        "valid": False,
        "files_checked": 0,
        "checksums_verified": 0,
        "warnings": [],
        "errors": [],
    }
    try:
        manifest = read_manifest(directory)
    except (OSError, ValueError) as exc:
        report["errors"].append(str(exc))
        return report
    checksums = manifest.get("checksums")
    if checksums is None:
        report["warnings"].append(
            "Legacy backup has no checksums; content corruption cannot be ruled out."
        )
    for relative in manifest_artifacts(manifest):
        try:
            path = artifact_path(directory, relative)
            if not path.is_file():
                raise ValueError(f"Missing backup file: {relative}")
            if checksums is not None:
                if file_checksum(path) != checksums[relative]:
                    raise ValueError(f"Checksum or size mismatch: {relative}")
                report["checksums_verified"] += 1
            report["files_checked"] += 1
        except (OSError, ValueError) as exc:
            report["errors"].append(str(exc))
    for snapshot in manifest["qdrant"]:
        path = artifact_path(directory, snapshot["file"])
        if path.is_file() and path.stat().st_size == 0:
            report["errors"].append(f"Empty Qdrant snapshot: {snapshot['file']}")
    for index in manifest["meilisearch"]:
        try:
            metadata = json.loads(
                artifact_path(directory, index["metadata_file"]).read_text(encoding="utf-8")
            )
            if (
                not isinstance(metadata, dict)
                or metadata.get("uid") != index["index"]
                or not isinstance(metadata.get("settings"), dict)
                or metadata.get("primaryKey") != index.get("primaryKey")
            ):
                raise ValueError(f"Invalid MeiliSearch metadata: {index['metadata_file']}")
            count = 0
            with artifact_path(directory, index["documents_file"]).open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    if not isinstance(json.loads(line), dict):
                        raise ValueError(f"Document on line {line_number} must be an object")
                    count += 1
            for source in (metadata, index):
                expected = source.get("documents")
                if type(expected) is not int or expected != count:
                    raise ValueError(
                        f"Document count mismatch for {index['index']}: "
                        f"expected {expected}, found {count}"
                    )
        except (OSError, ValueError) as exc:
            report["errors"].append(f"{index['index']}: {exc}")
    report["valid"] = not report["errors"]
    return report


def list_backups(root: Path) -> tuple[list[dict], list[dict]]:
    """Inventory immediate children; do not hash large snapshot files."""
    backups, skipped = [], []
    if not root.exists():
        return backups, skipped
    for directory in sorted(root.iterdir()):
        if directory.is_symlink():
            skipped.append({"path": str(directory), "reason": "Symlink"})
            continue
        if not directory.is_dir():
            continue
        try:
            manifest = read_manifest(directory)
            size = 0
            for relative in manifest_artifacts(manifest):
                path = artifact_path(directory, relative)
                if not path.is_file():
                    raise ValueError(f"Missing backup file: {relative}")
                size += path.stat().st_size
            backups.append(
                {
                    "path": str(directory),
                    "created_at": manifest["created_at"],
                    "size_bytes": size,
                    "corpora": sorted(
                        {entry["collection"] for entry in manifest["qdrant"]}
                        | {entry["index"] for entry in manifest["meilisearch"]}
                    ),
                    "has_checksums": "checksums" in manifest,
                }
            )
        except (OSError, ValueError) as exc:
            skipped.append({"path": str(directory), "reason": str(exc)})
    backups.sort(
        key=lambda backup: (backup_time(backup["created_at"]), backup["path"]), reverse=True
    )
    return backups, skipped


def select_prunable(backups: list[dict], keep: int | None, older_than: int | None) -> list[dict]:
    """Keep the newest N; when supplied, the age cutoff must also be met."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than or 0)
    return [
        backup
        for backup in backups[keep or 0 :]
        if older_than is None or backup_time(backup["created_at"]) < cutoff
    ]

"""Backup integrity, retention, and snapshot maintenance through the full CLI."""

import json
import shutil
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
import requests
from click.testing import CliRunner

from arcaneum.backup import backup_checksums, verify_backup
from arcaneum.cli.main import cli


@pytest.fixture
def make_backup(tmp_path):
    def create(name="backup", days=0, checksums=True):
        path = tmp_path / name
        (path / "qdrant").mkdir(parents=True)
        (path / "meilisearch").mkdir()
        (path / "qdrant" / "Docs.snapshot").write_bytes(b"snapshot contents")
        (path / "meilisearch" / "Docs.metadata.json").write_text(
            json.dumps(
                {
                    "uid": "Docs",
                    "primaryKey": "id",
                    "settings": {},
                    "documents": 1,
                }
            )
        )
        (path / "meilisearch" / "Docs.documents.jsonl").write_text('{"id": "1"}\n')
        manifest = {
            "version": 1,
            "created_at": (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(),
            "qdrant": [
                {"collection": "Docs", "snapshot": "Docs.snapshot", "file": "qdrant/Docs.snapshot"}
            ],
            "meilisearch": [
                {
                    "index": "Docs",
                    "primaryKey": "id",
                    "documents": 1,
                    "metadata_file": "meilisearch/Docs.metadata.json",
                    "documents_file": "meilisearch/Docs.documents.jsonl",
                }
            ],
        }
        if checksums:
            manifest["checksums"] = backup_checksums(path, manifest)
        (path / "manifest.json").write_text(json.dumps(manifest))
        return path

    return create


def invoke(*args, success=True):
    with patch("arcaneum.migrations.run_migration_if_needed") as migrate:
        result = CliRunner().invoke(cli, ["--json", "container", *map(str, args)])
    assert (result.exit_code == 0) is success, (result.output, result.exception)
    migrate.assert_not_called()
    payload = json.loads(result.stdout)
    assert payload["status"] == ("success" if success else "error")
    return payload


def test_list_backups_sorted_by_manifest_time_and_reports_skips(tmp_path, make_backup):
    old = make_backup("z-old", days=60)
    new = make_backup("a-new", days=1)
    (tmp_path / "unfinished").mkdir()
    (tmp_path / "linked").symlink_to(old, target_is_directory=True)
    payload = invoke("backup-list", "--root", tmp_path)["data"]
    assert [item["path"] for item in payload["backups"]] == [str(new), str(old)]
    assert payload["backups"][0]["corpora"] == ["Docs"]
    assert payload["backups"][0]["size_bytes"] == sum(
        path.stat().st_size
        for subdir in ("qdrant", "meilisearch")
        for path in (new / subdir).iterdir()
    )
    assert len(payload["skipped"]) == 2


def test_list_uses_configured_root_without_creating_it(tmp_path, monkeypatch):
    config_dir = tmp_path / "config" / "arcaneum"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text("backup:\n  path: archives\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    payload = invoke("backup-list")["data"]
    assert payload["root"] == str(config_dir / "archives")
    assert payload["backups"] == []
    assert not (config_dir / "archives").exists()


@pytest.mark.parametrize("checksums", [True, False])
def test_verify_new_and_legacy_backups(make_backup, checksums):
    path = make_backup(checksums=checksums)
    report = invoke("backup-verify", path)["data"]
    assert report["valid"]
    assert report["files_checked"] == 3
    assert report["checksums_verified"] == (3 if checksums else 0)
    assert bool(report["warnings"]) is not checksums


@pytest.mark.parametrize("damage", ["corruption", "missing", "json", "count", "metadata"])
def test_verify_reports_corrupt_or_incomplete_artifacts(make_backup, damage):
    path = make_backup(checksums=damage not in ("json", "count", "metadata"))
    if damage == "corruption":
        snapshot = path / "qdrant" / "Docs.snapshot"
        snapshot.write_bytes(b"X" * snapshot.stat().st_size)
    elif damage == "missing":
        (path / "qdrant" / "Docs.snapshot").unlink()
    elif damage == "json":
        (path / "meilisearch" / "Docs.documents.jsonl").write_text("not json\n")
    elif damage == "count":
        (path / "meilisearch" / "Docs.documents.jsonl").write_text("")
    else:
        (path / "meilisearch" / "Docs.metadata.json").write_text("[]")
    report = invoke("backup-verify", path, success=False)["data"]
    assert not report["valid"]
    assert report["errors"]


@pytest.mark.parametrize("damage", ["version", "timestamp", "checksums", "traversal", "symlink"])
def test_verify_rejects_invalid_manifests_and_unsafe_paths(make_backup, tmp_path, damage):
    path = make_backup()
    manifest_file = path / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    if damage == "version":
        manifest["version"] = 99
    elif damage == "timestamp":
        manifest["created_at"] = "bad"
    elif damage == "checksums":
        manifest["checksums"] = {}
    elif damage == "traversal":
        del manifest["checksums"]
        manifest["qdrant"][0]["file"] = "../outside.snapshot"
        (tmp_path / "outside.snapshot").write_bytes(b"outside")
    else:
        snapshot = path / "qdrant" / "Docs.snapshot"
        snapshot.unlink()
        snapshot.symlink_to(tmp_path / "outside.snapshot")
    manifest_file.write_text(json.dumps(manifest))
    report = invoke("backup-verify", path, success=False)["data"]
    assert not report["valid"]
    assert report["errors"]


def test_prune_combines_keep_and_age_and_preserves_invalid_backups(tmp_path, make_backup):
    newest = make_backup("newest", days=1)
    recent = make_backup("recent", days=10)
    old = make_backup("old", days=60)
    oldest = make_backup("oldest", days=90)
    broken = make_backup("broken", days=100)
    (broken / "qdrant" / "Docs.snapshot").unlink()
    (tmp_path / "unfinished").mkdir()
    (tmp_path / "linked").symlink_to(oldest, target_is_directory=True)
    args = ("backup-prune", "--root", tmp_path, "--keep", "1", "--older-than", "30")
    preview = invoke(*args, "--dry-run")["data"]
    assert [item["path"] for item in preview["selected"]] == [str(old), str(oldest)]
    assert preview["deleted"] == []
    assert all(path.exists() for path in (old, oldest))
    result = invoke(*args, "--confirm")["data"]
    assert result["deleted"] == [str(old), str(oldest)]
    assert not old.exists() and not oldest.exists()
    assert all(path.exists() for path in (newest, recent, broken, tmp_path / "unfinished"))
    assert (tmp_path / "linked").is_symlink()


def test_prune_keeps_healthy_backup_when_newest_is_corrupt(tmp_path, make_backup):
    corrupt = make_backup("corrupt", days=0)
    (corrupt / "qdrant" / "Docs.snapshot").write_bytes(b"corrupt")
    good = make_backup("good", days=10)
    old = make_backup("old", days=20)
    result = invoke("backup-prune", "--root", tmp_path, "--keep", "1", "--confirm")["data"]
    assert result["deleted"] == [str(old)]
    assert corrupt.exists() and good.exists()
    assert any(item["path"] == str(corrupt) for item in result["skipped"])


@pytest.mark.parametrize("filters", [("--keep", "1"), ("--older-than", "30")])
def test_prune_single_filter(tmp_path, make_backup, filters):
    new = make_backup("new", days=0)
    old = make_backup("old", days=60)
    result = invoke("backup-prune", "--root", tmp_path, *filters, "--dry-run")["data"]
    assert [item["path"] for item in result["selected"]] == [str(old)]
    assert new.exists() and old.exists()


@pytest.mark.parametrize("args", [(), ("--keep", "1")])
def test_prune_requires_retention_and_explicit_confirmation(tmp_path, make_backup, args):
    backup = make_backup()
    invoke("backup-prune", "--root", tmp_path, *args, success=False)
    assert backup.exists()


def test_prune_reports_deletion_failure_without_touching_retained_backup(tmp_path, make_backup):
    retained = make_backup("retained")
    old = make_backup("old", days=60)
    with patch("arcaneum.cli.backups.shutil.rmtree", side_effect=PermissionError("read-only disk")):
        result = invoke(
            "backup-prune", "--root", tmp_path, "--keep", "1", "--confirm", success=False
        )["data"]
    assert result["deleted"] == []
    assert "read-only disk" in result["errors"][0]
    assert retained.exists() and old.exists()


def test_prune_refuses_backup_changed_after_selection(tmp_path, make_backup):
    retained = make_backup("retained")
    old = make_backup("old", days=60)
    oldest = make_backup("oldest", days=90)
    remove = shutil.rmtree

    def delete_and_change_next(path):
        assert path == old
        remove(path)
        (oldest / "manifest.json").write_text("{}")

    with patch("arcaneum.cli.backups.shutil.rmtree", side_effect=delete_and_change_next):
        result = invoke(
            "backup-prune", "--root", tmp_path, "--keep", "1", "--confirm", success=False
        )["data"]
    assert result["deleted"] == [str(old)]
    assert "changed after selection" in result["errors"][0]
    assert retained.exists() and oldest.exists()


def test_local_maintenance_text_output(make_backup, tmp_path):
    path = make_backup()
    for args, expected in (
        (["backup-list", "--root", str(tmp_path)], "Docs"),
        (["backup-verify", str(path)], "Backup verified"),
        (
            ["backup-prune", "--root", str(tmp_path), "--older-than", "1", "--dry-run"],
            "Would delete 0 backups",
        ),
    ):
        result = CliRunner().invoke(cli, ["container", *args])
        assert result.exit_code == 0, result.output
        assert expected in result.output


def test_snapshot_list_includes_names_sizes_and_dates():
    with patch(
        "arcaneum.cli.backups._request_json",
        side_effect=[
            {"result": {"collections": [{"name": "Docs"}]}},
            {"result": [{"name": "old.snapshot", "size": 42, "creation_time": "2026-01-01"}]},
        ],
    ) as request:
        data = invoke("snapshot-list")["data"]
    assert data["snapshots"] == [
        {"collection": "Docs", "name": "old.snapshot", "size": 42, "creation_time": "2026-01-01"}
    ]
    assert all(call.args[0] == "GET" for call in request.call_args_list)


@pytest.mark.parametrize("dry_run", [True, False])
def test_snapshot_cleanup_deletes_only_explicit_names_and_encodes_urls(dry_run):
    def request(method, url, **kwargs):
        if method == "GET":
            return {"result": [{"name": "old #1.snapshot"}, {"name": "keep.snapshot"}]}
        assert method == "DELETE"
        assert not dry_run
        assert url.endswith("/collections/Docs%20Space/snapshots/old%20%231.snapshot")
        assert kwargs == {"params": {"wait": "true"}, "timeout": 300}
        return {"result": True}

    with patch("arcaneum.cli.backups._request_json", side_effect=request) as mock_request:
        data = invoke(
            "snapshot-cleanup",
            "--collection",
            "Docs Space",
            "--snapshot",
            "old #1.snapshot",
            "--snapshot",
            "old #1.snapshot",
            "--dry-run" if dry_run else "--confirm",
        )["data"]
    assert data["deleted"] == ([] if dry_run else ["old #1.snapshot"])
    assert len(data["selected"]) == 1
    assert mock_request.call_count == (1 if dry_run else 2)


def test_snapshot_cleanup_preflights_all_names_before_deleting():
    with patch(
        "arcaneum.cli.backups._request_json",
        return_value={"result": [{"name": "existing.snapshot"}]},
    ) as request:
        invoke(
            "snapshot-cleanup",
            "--collection",
            "Docs",
            "--snapshot",
            "existing.snapshot",
            "--snapshot",
            "missing.snapshot",
            "--confirm",
            success=False,
        )
    assert request.call_count == 1
    assert request.call_args.args[0] == "GET"


@pytest.mark.parametrize("dry_run", [True, False])
def test_snapshot_cleanup_defaults_to_all_collections(dry_run):
    responses = [
        {"result": {"collections": [{"name": "Docs"}, {"name": "Claude"}]}},
        {"result": [{"name": "shared.snapshot"}, {"name": "keep.snapshot"}]},
        {"result": [{"name": "shared.snapshot"}]},
    ]
    if not dry_run:
        responses.extend([{"result": True}, {"result": True}])
    with patch("arcaneum.cli.backups._request_json", side_effect=responses) as request:
        data = invoke(
            "snapshot-cleanup",
            "--snapshot",
            "shared.snapshot",
            "--dry-run" if dry_run else "--confirm",
        )["data"]
    pairs = [{"collection": name, "name": "shared.snapshot"} for name in ("Docs", "Claude")]
    assert data["collection"] is None
    assert data["selected"] == pairs
    assert data["deleted_snapshots"] == ([] if dry_run else pairs)
    deletes = [call.args[1] for call in request.call_args_list if call.args[0] == "DELETE"]
    assert deletes == (
        []
        if dry_run
        else [
            f"http://localhost:6333/collections/{name}/snapshots/shared.snapshot"
            for name in ("Docs", "Claude")
        ]
    )


@pytest.mark.parametrize("failure", ["missing", "unavailable"])
def test_snapshot_cleanup_all_collections_preflight_failure_deletes_nothing(failure):
    last_response = (
        requests.HTTPError("unavailable") if failure == "unavailable" else {"result": []}
    )
    with patch(
        "arcaneum.cli.backups._request_json",
        side_effect=[
            {"result": {"collections": [{"name": "Docs"}, {"name": "Claude"}]}},
            {"result": [{"name": "old.snapshot"}]},
            last_response,
        ],
    ) as request:
        invoke(
            "snapshot-cleanup",
            "--snapshot",
            "old.snapshot",
            "--snapshot",
            "missing.snapshot",
            "--confirm",
            success=False,
        )
    assert all(call.args[0] == "GET" for call in request.call_args_list)


def test_snapshot_cleanup_requires_confirmation():
    with patch("arcaneum.cli.backups._request_json") as request:
        invoke(
            "snapshot-cleanup", "--collection", "Docs", "--snapshot", "old.snapshot", success=False
        )
    request.assert_not_called()


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("collection", [None, "Docs"])
def test_snapshot_cleanup_defaults_to_all_snapshots(dry_run, collection):
    responses = []
    if collection is None:
        responses.append({"result": {"collections": [{"name": "Docs"}, {"name": "Claude"}]}})
    responses.append({"result": [{"name": "first.snapshot"}, {"name": "second.snapshot"}]})
    if collection is None:
        responses.append({"result": [{"name": "third.snapshot"}]})
    pairs = [
        {"collection": "Docs", "name": "first.snapshot"},
        {"collection": "Docs", "name": "second.snapshot"},
    ]
    if collection is None:
        pairs.append({"collection": "Claude", "name": "third.snapshot"})
    if not dry_run:
        responses.extend([{"result": True}] * len(pairs))
    args = ["snapshot-cleanup", "--dry-run" if dry_run else "--confirm"]
    if collection:
        args.extend(["--collection", collection])
    with patch("arcaneum.cli.backups._request_json", side_effect=responses) as request:
        data = invoke(*args)["data"]
    assert data["selected"] == pairs
    assert data["deleted_snapshots"] == ([] if dry_run else pairs)
    deletes = [call for call in request.call_args_list if call.args[0] == "DELETE"]
    assert len(deletes) == (0 if dry_run else len(pairs))


def test_snapshot_cleanup_all_requires_confirmation_and_handles_empty_server():
    with patch("arcaneum.cli.backups._request_json") as request:
        invoke("snapshot-cleanup", success=False)
    request.assert_not_called()
    with patch(
        "arcaneum.cli.backups._request_json", return_value={"result": {"collections": []}}
    ) as request:
        data = invoke("snapshot-cleanup", "--confirm")["data"]
    assert data["selected"] == []
    assert data["deleted_snapshots"] == []
    assert request.call_count == 1


def test_snapshot_cleanup_reports_partial_failure():
    with patch(
        "arcaneum.cli.backups._request_json",
        side_effect=[
            {"result": [{"name": "one"}, {"name": "two"}]},
            {"result": True},
            requests.HTTPError("server unavailable"),
        ],
    ):
        report = invoke(
            "snapshot-cleanup",
            "--collection",
            "Docs",
            "--snapshot",
            "one",
            "--snapshot",
            "two",
            "--confirm",
            success=False,
        )["data"]
    assert report["deleted"] == ["one"]
    assert "two" in report["errors"][0]


def test_verify_empty_snapshot_rejected_even_without_checksums(make_backup):
    path = make_backup(checksums=False)
    (path / "qdrant" / "Docs.snapshot").write_bytes(b"")
    assert not verify_backup(path)["valid"]

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock

import pytest

from arcaneum.cli import sync


class _Discovery:
    metadata = {}

    def extract_metadata(self, root):
        return self.metadata.get(root)


def _meta(identifier, commit_hash):
    return SimpleNamespace(identifier=identifier, commit_hash=commit_hash)


def test_automatic_git_skip_requires_clean_matching_success_stamp(monkeypatch):
    _Discovery.metadata = {
        "/clean": _meta("clean-project", "abc123"),
        "/dirty": _meta("dirty-project", "def456"),
        "/changed": _meta("changed-project", "new789"),
    }
    monkeypatch.setattr(sync, "GitProjectDiscovery", _Discovery)
    monkeypatch.setattr(sync, "_repo_has_tracked_changes", lambda root: root == "/dirty")

    skipped, clean_heads = sync._automatic_git_skip_roots(
        ["/clean", "/dirty", "/changed"],
        {
            sync.GIT_SYNC_HEADS_FIELD: {
                "clean-project": "abc123",
                "dirty-project": "def456",
                "changed-project": "old789",
            }
        },
    )

    assert skipped == {"/clean"}
    assert clean_heads == {"clean-project": "abc123", "changed-project": "new789"}


@pytest.mark.parametrize(
    "override",
    [
        {"corpus_type": "pdf"},
        {"dir_paths": []},
        {"single_files": [Path("one.py")]},
        {"from_file": "paths.txt"},
        {"file_types": ".py"},
        {"force": True},
        {"parity": True},
        {"repair": True},
        {"git_update": True},
        {"git_version": True},
        {"dry_run": True},
    ],
)
def test_automatic_git_fast_path_rejects_partial_or_special_modes(override):
    arguments = {
        "corpus_type": "code",
        "dir_paths": [Path("repo")],
        "single_files": [],
        "from_file": None,
        "file_types": None,
        "force": False,
        "parity": False,
        "repair": False,
        "git_update": False,
        "git_version": False,
        "dry_run": False,
    }
    arguments.update(override)

    assert not sync._automatic_git_fast_path_enabled(**arguments)


def test_automatic_git_fast_path_accepts_default_full_code_directory_sync():
    assert sync._automatic_git_fast_path_enabled(
        corpus_type="code",
        dir_paths=[Path("repo")],
        single_files=[],
        from_file=None,
        file_types=None,
        force=False,
        parity=False,
        repair=False,
        git_update=False,
        git_version=False,
        dry_run=False,
    )


def test_stamp_merges_out_of_scope_heads_and_removes_dirty_scoped_head(monkeypatch):
    _Discovery.metadata = {
        "/clean": _meta("clean-project", "new"),
        "/dirty": _meta("dirty-project", "dirty"),
    }
    monkeypatch.setattr(sync, "GitProjectDiscovery", _Discovery)
    update = Mock()
    monkeypatch.setattr(sync, "update_collection_metadata", update)
    metadata = {
        sync.GIT_SYNC_HEADS_FIELD: {
            "clean-project": "old",
            "dirty-project": "old-dirty",
            "other-project": "preserved",
        }
    }

    sync._stamp_git_sync_heads(
        Mock(),
        "corpus",
        metadata,
        ["/clean", "/dirty"],
        {"clean-project": "new"},
    )

    update.assert_called_once_with(
        ANY,
        "corpus",
        git_sync_heads={"clean-project": "new", "other-project": "preserved"},
    )

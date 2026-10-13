"""`arc corpus sync --from-index <selector>`: re-index files chosen by stored metadata."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from arcaneum.cli import sync as sync_mod
from arcaneum.cli.errors import InvalidArgumentError
from arcaneum.cli.main import cli


@pytest.fixture
def isolated_locks(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))


def _run(args, impl):
    with patch("arcaneum.cli.sync._sync_directory_locked", impl):
        return CliRunner().invoke(cli, args, catch_exceptions=False)


def test_from_index_is_forwarded_without_paths(isolated_locks):
    seen = {}

    def impl(corpus, paths, *args, **kwargs):
        seen["corpus"] = corpus
        seen["paths"] = paths
        seen["from_index"] = kwargs.get("from_index")

    result = _run(["corpus", "sync", "Papers", "--from-index", "method=pymupdf_normalized"], impl)

    assert result.exit_code == 0, result.output
    assert seen == {"corpus": "Papers", "paths": (), "from_index": "method=pymupdf_normalized"}


@pytest.mark.parametrize(
    "extra",
    [
        ["{path}"],
        ["--from-file", "{path}"],
        ["--changed-since", "HEAD"],
        ["--drain-spool"],
        ["--parity"],
    ],
)
def test_from_index_rejects_other_path_sources(tmp_path, isolated_locks, extra):
    listed = tmp_path / "paths.txt"
    listed.write_text("")
    args = [a.format(path=listed) for a in extra]

    def impl(*args, **kwargs):
        raise AssertionError("sync must not run")

    with patch("arcaneum.cli.sync._sync_directory_locked", impl):
        result = CliRunner().invoke(
            cli, ["corpus", "sync", "Papers", "--from-index", "method=x", *args]
        )

    assert result.exit_code == 2
    assert "--from-index" in result.output


def test_missing_path_error_names_from_index(isolated_locks):
    result = CliRunner().invoke(cli, ["corpus", "sync", "Papers"])

    assert result.exit_code == 2
    assert "--from-index" in result.output


class _FakeQdrant:
    def __init__(self, payloads):
        self.payloads = payloads

    def scroll(self, collection_name, limit, offset=None, with_payload=True, with_vectors=False):
        return [SimpleNamespace(payload=p) for p in self.payloads], None


def _normalized(path):
    return {"file_path": str(path), "quality_manifest": {"extraction_method": "pymupdf_normalized"}}


def test_selected_files_on_disk_are_returned_and_missing_ones_skipped(tmp_path, capsys):
    present = tmp_path / "present.pdf"
    present.write_bytes(b"%PDF-1.4\n")
    gone = tmp_path / "gone.pdf"
    qdrant = _FakeQdrant([_normalized(present), _normalized(gone)])

    files = sync_mod._resolve_from_index(
        qdrant, "Papers", "method=pymupdf_normalized", dry_run=False, output_json=False
    )

    assert files == [present.resolve()]
    out = capsys.readouterr().out
    assert "1 file no longer exists on disk" in out
    assert "gone.pdf" in out


def test_dry_run_lists_selection_and_stops(tmp_path, capsys):
    present = tmp_path / "present.pdf"
    present.write_bytes(b"%PDF-1.4\n")
    qdrant = _FakeQdrant([_normalized(present)])

    files = sync_mod._resolve_from_index(
        qdrant, "Papers", "method=pymupdf_normalized", dry_run=True, output_json=True
    )

    assert files is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["would_reindex"] == 1
    assert payload["data"]["files"] == [str(present.resolve())]
    assert payload["data"]["selector"] == "method=pymupdf_normalized"


def test_empty_selection_stops_without_indexing(tmp_path, capsys):
    qdrant = _FakeQdrant(
        [{"file_path": "/p/a.pdf", "quality_manifest": {"extraction_method": "pymupdf4llm"}}]
    )

    files = sync_mod._resolve_from_index(
        qdrant, "Papers", "method=pymupdf_normalized", dry_run=False, output_json=False
    )

    assert files is None
    assert "No indexed files match" in capsys.readouterr().out


def test_invalid_selector_is_an_invalid_argument():
    qdrant = _FakeQdrant([{"file_path": "/p/notes.md"}])

    with pytest.raises(InvalidArgumentError, match="extraction_method"):
        sync_mod._resolve_from_index(
            qdrant, "Notes", "method=pymupdf_normalized", dry_run=False, output_json=False
        )

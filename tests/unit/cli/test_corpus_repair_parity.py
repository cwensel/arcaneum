"""`arc corpus repair` also restores Qdrant/MeiliSearch parity.

Repair re-indexes files the Qdrant verifier selects, then runs the
`arc corpus parity --verify` pass so the drift `arc corpus verify` reports
(qdrant_only, meili_only, chunk-count mismatch) is fixed by the same command.
"""

from __future__ import annotations

import json
import platform
import sys
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from arcaneum.cli import corpus_lock
from arcaneum.cli.errors import ResourceNotFoundError
from arcaneum.cli.main import cli
from arcaneum.cli.output import print_json

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="flock-based corpus lock is a no-op on Windows",
)


@pytest.fixture
def isolated_locks(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    yield tmp_path / "arcaneum" / "locks"


def _content_ok(corpus, *_args, **_kwargs):
    print_json("success", "Indexed 1 files (4 chunks)", data={"repair": {"improved": 1}})


def _parity_ok(**_kwargs):
    print_json(
        "success",
        f"Parity restored for corpus '{_kwargs['corpus']}'",
        data={"meili_backfilled": 2, "qdrant_backfilled": 0},
    )


def _run(args, content=_content_ok, parity=_parity_ok):
    runner = CliRunner()
    with (
        patch("arcaneum.cli.sync._sync_directory_locked", content),
        patch("arcaneum.cli.sync._parity_single_corpus", parity),
        patch("arcaneum.cli.utils.create_qdrant_client", side_effect=RuntimeError("no server")),
    ):
        return runner.invoke(cli, args, catch_exceptions=False)


def test_repair_runs_parity_verify_after_content_repair(isolated_locks):
    calls = []

    def content(corpus, *args, **kwargs):
        calls.append(("content", kwargs["repair"]))

    def parity(**kwargs):
        calls.append(("parity", kwargs["corpus"], kwargs["verify"], kwargs["dry_run"]))

    result = _run(["corpus", "repair", "Docs"], content, parity)

    assert result.exit_code == 0, result.output
    assert calls == [("content", True), ("parity", "Docs", True, False)]


def test_repair_dry_run_previews_parity(isolated_locks):
    seen = {}

    def parity(**kwargs):
        seen["dry_run"] = kwargs["dry_run"]

    result = _run(["corpus", "repair", "Docs", "--dry-run"], parity=parity)

    assert result.exit_code == 0, result.output
    assert seen["dry_run"] is True


def test_repair_holds_the_lock_through_parity(isolated_locks):
    observed = {}

    def parity(**kwargs):
        observed["holder"] = corpus_lock.read_lock_holder(
            corpus_lock.corpus_lock_path(kwargs["corpus"])
        )

    result = _run(["corpus", "repair", "Docs"], parity=parity)

    assert result.exit_code == 0, result.output
    assert observed["holder"] is not None
    assert observed["holder"]["corpus"] == "Docs"


def test_repair_json_is_one_document_with_parity_nested(isolated_locks):
    result = _run(["corpus", "repair", "Docs", "--json"])

    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["status"] == "success"
    assert doc["message"] == "Indexed 1 files (4 chunks)"
    # Existing repair fields stay where consumers expect them.
    assert doc["data"]["repair"] == {"improved": 1}
    assert doc["data"]["parity"] == {
        "status": "success",
        "message": "Parity restored for corpus 'Docs'",
        "meili_backfilled": 2,
        "qdrant_backfilled": 0,
    }


def test_repair_json_takes_the_worse_status(isolated_locks):
    def content(corpus, *_args, **_kwargs):
        print_json("warning", "No repairable files found", data={"repaired": 0})

    result = _run(["corpus", "repair", "Docs", "--json"], content=content)

    doc = json.loads(result.stdout)
    assert doc["status"] == "warning"
    assert doc["data"]["repaired"] == 0
    assert doc["data"]["parity"]["status"] == "success"


def test_repair_skips_parity_when_content_repair_fails(isolated_locks):
    ran = []

    def content(corpus, *_args, **_kwargs):
        print_json("error", "[ERROR] Failed to sync directory: boom", errors=["boom"])
        sys.exit(1)

    result = _run(
        ["corpus", "repair", "Docs", "--json"],
        content=content,
        parity=lambda **_kwargs: ran.append(True),
    )

    assert result.exit_code == 1
    assert ran == []
    # The content failure is reported exactly as before.
    doc = json.loads(result.stdout)
    assert doc["status"] == "error"
    assert doc["errors"] == [{"code": "error", "message": "boom"}]


def test_repair_reports_parity_skipped_when_store_missing(isolated_locks):
    def parity(**_kwargs):
        raise ResourceNotFoundError("MeiliSearch index 'Docs' not found.")

    result = _run(["corpus", "repair", "Docs", "--json"], parity=parity)

    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["status"] == "warning"
    assert doc["data"]["parity"] == {
        "status": "skipped",
        "message": "MeiliSearch index 'Docs' not found.",
    }


def test_repair_exits_nonzero_when_parity_fails(isolated_locks):
    def parity(**_kwargs):
        raise RuntimeError("backfill exploded")

    result = _run(["corpus", "repair", "Docs", "--json"], parity=parity)

    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert doc["status"] == "error"
    assert doc["data"]["repair"] == {"improved": 1}
    assert doc["data"]["parity"] == {"status": "error", "message": "backfill exploded"}
    assert doc["errors"] == [
        {"code": "error", "message": "Parity repair failed: backfill exploded"}
    ]


def test_repair_text_output_reports_parity_failure(isolated_locks):
    def parity(**_kwargs):
        raise RuntimeError("backfill exploded")

    result = _run(["corpus", "repair", "Docs"], parity=parity)

    assert result.exit_code == 1
    assert "Parity repair failed: backfill exploded" in result.output


class _CountingQdrant:
    """Qdrant double whose facets count two documents in the corpus."""

    def facet(self, collection_name, key, **_kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(
            hits=[SimpleNamespace(value=f"/docs/{i}.pdf", count=1) for i in range(2)]
        )


def _run_with_counts(args, updates, manifests=None, corpus_type="pdf"):
    runner = CliRunner()
    with (
        patch("arcaneum.cli.sync._sync_directory_locked", _content_ok),
        patch("arcaneum.cli.sync._parity_single_corpus", _parity_ok),
        patch("arcaneum.cli.utils.create_qdrant_client", lambda **_kwargs: _CountingQdrant()),
        patch("arcaneum.cli.sync.get_collection_type", lambda _client, _corpus: corpus_type),
        patch(
            "arcaneum.cli.sync.update_collection_metadata",
            lambda _client, name, **metadata: updates.append((name, metadata)),
        ),
        patch(
            "arcaneum.cli.sync.file_manifests_ready",
            lambda _client, _corpus: manifests is not None,
        ),
        patch(
            "arcaneum.cli.sync.MetadataBasedSync.get_file_manifest_snapshot",
            lambda _self, _corpus: manifests,
        ),
    ):
        return runner.invoke(cli, args, catch_exceptions=False)


def test_repair_stamps_list_counts_from_the_index(isolated_locks):
    """Listing no longer persists counts, so repair stamps them (RDR-026 Q10)."""
    updates = []

    result = _run_with_counts(["corpus", "repair", "Docs"], updates)

    assert result.exit_code == 0, result.output
    assert len(updates) == 1
    name, metadata = updates[0]
    assert name == "Docs"
    assert metadata["item_count"] == 2
    assert metadata["item_unit"] == "documents"
    assert metadata["count_source"] == "repair"
    assert metadata["counts_updated_at"]


def test_repair_counts_duplicate_pdf_aliases_from_manifests(isolated_locks):
    """An alias path has a manifest but no chunks, so chunk facets miss it."""
    updates = []
    manifests = {
        "/docs/0.pdf": {"file_path": "/docs/0.pdf"},
        "/docs/1.pdf": {"file_path": "/docs/1.pdf"},
        "/docs/copy-of-1.pdf": {"file_path": "/docs/copy-of-1.pdf", "canonical_path": "/docs/1.pdf"},
    }

    result = _run_with_counts(["corpus", "repair", "Docs"], updates, manifests=manifests)

    assert result.exit_code == 0, result.output
    assert len(updates) == 1
    _name, metadata = updates[0]
    assert metadata["item_count"] == 3
    assert metadata["item_unit"] == "documents"
    assert metadata["count_source"] == "repair"


def test_repair_counts_chunkless_markdown_files_from_manifests(isolated_locks):
    """An empty Markdown file gets a manifest but no chunks; full sync counts it."""
    updates = []
    manifests = {
        "/notes/a.md": {"file_path": "/notes/a.md", "chunk_count": 3},
        "/notes/empty.md": {"file_path": "/notes/empty.md", "chunk_count": 0},
        "/notes/b.md": {"file_path": "/notes/b.md", "chunk_count": 1},
    }

    result = _run_with_counts(
        ["corpus", "repair", "Notes"], updates, manifests=manifests, corpus_type="markdown"
    )

    assert result.exit_code == 0, result.output
    _name, metadata = updates[0]
    assert metadata["item_count"] == 3
    assert metadata["item_unit"] == "files"


def test_repair_counts_chunkless_code_files_from_manifests(isolated_locks):
    """Code keeps repositories from facets; its file count includes empty files."""
    updates = []
    manifests = {f"/repo/{i}.py": {"file_path": f"/repo/{i}.py"} for i in range(3)}

    result = _run_with_counts(
        ["corpus", "repair", "Code"], updates, manifests=manifests, corpus_type="code"
    )

    assert result.exit_code == 0, result.output
    _name, metadata = updates[0]
    assert metadata["item_count"] == 2
    assert metadata["item_unit"] == "repositories"
    assert metadata["file_count"] == 3
    assert metadata["file_unit"] == "source files"


def test_repair_dry_run_stamps_no_counts(isolated_locks):
    updates = []

    result = _run_with_counts(["corpus", "repair", "Docs", "--dry-run"], updates)

    assert result.exit_code == 0, result.output
    assert updates == []

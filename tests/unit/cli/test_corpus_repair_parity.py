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
    assert doc["errors"] == ["boom"]


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
    assert doc["errors"] == ["Parity repair failed: backfill exploded"]


def test_repair_text_output_reports_parity_failure(isolated_locks):
    def parity(**_kwargs):
        raise RuntimeError("backfill exploded")

    result = _run(["corpus", "repair", "Docs"], parity=parity)

    assert result.exit_code == 1
    assert "Parity repair failed: backfill exploded" in result.output

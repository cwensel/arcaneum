"""`arc corpus parity` across all corpora (no NAME)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from arcaneum.cli.main import cli
from arcaneum.cli.output import print_json


@pytest.fixture(autouse=True)
def isolated_locks(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))


def _corpus(name):
    return {"name": name, "status": "chunk_mismatch", "to_meili": 3, "to_qdrant": 0}


def _run(args, single):
    runner = CliRunner()
    with (
        patch(
            "arcaneum.cli.sync._discover_corpora_for_parity",
            lambda: [_corpus("Alpha"), _corpus("Beta")],
        ),
        patch("arcaneum.cli.sync._parity_single_corpus", single),
    ):
        return runner.invoke(cli, args, catch_exceptions=False)


def test_parity_all_json_is_one_document_with_per_corpus_results():
    def single(**kwargs):
        print_json(
            "success",
            f"Parity restored for corpus '{kwargs['corpus']}'",
            data={"corpus": kwargs["corpus"], "meili_backfilled": 3, "qdrant_backfilled": 0},
        )

    result = _run(["corpus", "parity", "--json", "--confirm"], single)

    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["message"] == "Parity completed for 2 corpora"
    assert doc["data"]["results"] == [
        {
            "corpus": "Alpha",
            "status": "success",
            "message": "Parity restored for corpus 'Alpha'",
            "meili_backfilled": 3,
            "qdrant_backfilled": 0,
        },
        {
            "corpus": "Beta",
            "status": "success",
            "message": "Parity restored for corpus 'Beta'",
            "meili_backfilled": 3,
            "qdrant_backfilled": 0,
        },
    ]


def test_parity_all_json_records_a_failed_corpus_without_extra_documents():
    def single(**kwargs):
        if kwargs["corpus"] == "Beta":
            raise RuntimeError("backfill exploded")
        print_json("success", "Indexes are already in parity", data={"corpus": "Alpha"})

    result = _run(["corpus", "parity", "--json", "--confirm"], single)

    doc = json.loads(result.stdout)
    assert doc["data"]["successes"] == 1
    assert doc["data"]["failures"] == 1
    assert doc["data"]["results"] == [
        {"corpus": "Alpha", "status": "success", "message": "Indexes are already in parity"},
        {"corpus": "Beta", "status": "failed", "error": "backfill exploded"},
    ]

"""`arc corpus doctor`: read-only health sweep that names the fix for each corpus."""

from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from arcaneum.cli import corpus_doctor
from arcaneum.cli.corpus_doctor import diagnose_corpus
from arcaneum.cli.main import cli


def _verification(repairable=(), policy_only=(), recovery_exhausted=(), errors=(), **counts):
    files = [
        SimpleNamespace(
            file_path=path,
            policy_only_repair=path in policy_only,
            fidelity_degraded=False,
            recovery_exhausted=path in recovery_exhausted,
        )
        for path in {*repairable, *policy_only, *recovery_exhausted}
    ]
    return SimpleNamespace(
        collection_type=counts.pop("collection_type", "pdf"),
        files=files,
        errors=list(errors),
        get_items_needing_repair=lambda include_stale_policy=True: sorted(repairable),
        **counts,
    )


def _synced():
    return {
        "status": "synced",
        "summary": {"synced": 3, "mismatch": 0, "qdrant_only": 0, "meili_only": 0},
        "items": [],
    }


def test_healthy_corpus_is_ok():
    result = diagnose_corpus("Docs", "both", _verification(), _synced(), None)

    assert result == {
        "corpus": "Docs",
        "status": "ok",
        "findings": [],
        "commands": [],
        "optional_commands": [],
    }


def test_repair_command_combines_content_parity_and_orphans():
    parity = {
        "status": "mismatch",
        "summary": {"synced": 1, "mismatch": 1, "qdrant_only": 1, "meili_only": 2},
        "items": [
            {"id": "/d/a.pdf", "status": "mismatch"},
            {"id": "/d/b.pdf", "status": "qdrant_only"},
            {"id": "/d/on-disk.pdf", "status": "meili_only"},
            {"id": "/d/gone.pdf", "status": "meili_only"},
        ],
    }
    orphans = {"prunable": ["/d/gone.pdf"], "dir_missing": [], "referenced": []}

    result = diagnose_corpus(
        "Docs",
        "both",
        _verification(repairable=["/d/x.pdf"], incomplete_items=1),
        parity,
        orphans,
    )

    assert result["status"] == "needs_action"
    assert result["commands"] == ["arc corpus repair Docs --prune-orphans"]
    assert [(f["code"], f["count"]) for f in result["findings"]] == [
        ("repairable_files", 1),
        # The prunable orphan is reported separately, not as fixable drift.
        ("parity_mismatch", 3),
        ("orphans_prunable", 1),
    ]
    assert result["findings"][0]["details"] == {"incomplete_items": 1}


def test_policy_only_files_are_optional():
    result = diagnose_corpus(
        "Docs", "both", _verification(policy_only=["/d/old.pdf"]), _synced(), None
    )

    assert result["status"] == "ok"
    assert result["commands"] == []
    assert result["optional_commands"] == ["arc corpus repair Docs --include-stale-policy"]
    assert result["findings"][0]["severity"] == "optional"


def test_orphans_in_missing_directory_never_get_a_required_command():
    parity = {
        "status": "mismatch",
        "summary": {"synced": 2, "mismatch": 0, "qdrant_only": 0, "meili_only": 1},
        "items": [{"id": "/Volumes/Ext/a.pdf", "status": "meili_only"}],
    }
    orphans = {"prunable": [], "dir_missing": ["/Volumes/Ext/a.pdf"], "referenced": []}

    result = diagnose_corpus("Docs", "both", _verification(), parity, orphans)

    assert result["status"] == "needs_action"
    assert result["commands"] == []
    assert result["optional_commands"] == ["arc corpus repair Docs --force-prune"]
    assert [f["code"] for f in result["findings"]] == ["orphans_dir_missing"]


def test_qdrant_only_corpus_suggests_create_missing():
    result = diagnose_corpus("Docs", "qdrant_only", _verification(), None, None)

    assert result["status"] == "needs_action"
    assert result["commands"] == ["arc corpus parity Docs --create-missing"]


def test_meili_only_corpus_needs_manual_creation():
    result = diagnose_corpus("Docs", "meili_only", None, None, None)

    assert result["status"] == "needs_action"
    assert result["commands"] == []
    assert "arc corpus create Docs" in result["findings"][0]["message"]


def test_failed_checks_are_errors():
    unknown = {"status": "unknown", "reason": "Parity scan failed: timeout"}

    result = diagnose_corpus(
        "Docs", "both", _verification(errors=["schema too new"]), unknown, None
    )

    assert result["status"] == "error"
    assert [f["code"] for f in result["findings"]] == ["verification_errors", "parity_unknown"]


def test_recovery_exhausted_files_are_informational():
    result = diagnose_corpus(
        "Docs", "both", _verification(recovery_exhausted=["/d/scan.pdf"]), _synced(), None
    )

    assert result["status"] == "ok"
    assert result["findings"][0]["severity"] == "info"


def _diagnoses():
    return {
        "Alpha": diagnose_corpus("Alpha", "both", _verification(), _synced(), None),
        "Beta": diagnose_corpus(
            "Beta", "both", _verification(repairable=["/b/x.pdf"]), _synced(), None
        ),
    }


def _run(args):
    diagnoses = _diagnoses()
    with (
        patch.object(
            corpus_doctor,
            "_discover_corpora",
            lambda _q, _m: {"Alpha": "both", "Beta": "both"},
        ),
        patch.object(corpus_doctor, "_open_clients", lambda: (object(), object())),
        patch.object(
            corpus_doctor,
            "_check_corpus",
            lambda _q, _m, name, _presence: diagnoses[name],
        ),
    ):
        return CliRunner().invoke(cli, args, catch_exceptions=False)


def test_doctor_json_is_one_document_listing_commands():
    result = _run(["corpus", "doctor", "--json"])

    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["status"] == "warning"
    assert doc["data"]["summary"] == {"ok": 1, "needs_action": 1, "error": 0}
    assert doc["data"]["commands"] == ["arc corpus repair Beta"]
    assert [c["corpus"] for c in doc["data"]["corpora"]] == ["Alpha", "Beta"]


def test_doctor_strict_exits_nonzero_when_action_needed():
    result = _run(["corpus", "doctor", "--strict", "--json"])

    assert result.exit_code == 1


def test_doctor_checks_only_named_corpora():
    result = _run(["corpus", "doctor", "Alpha", "--json"])

    doc = json.loads(result.stdout)
    assert [c["corpus"] for c in doc["data"]["corpora"]] == ["Alpha"]
    assert doc["status"] == "success"


def test_doctor_rejects_unknown_corpus():
    result = _run(["corpus", "doctor", "Nope"])

    assert result.exit_code != 0
    assert "Nope" in result.output


def test_doctor_text_lists_commands_to_run():
    result = _run(["corpus", "doctor"])

    assert result.exit_code == 0, result.output
    assert "arc corpus repair Beta" in result.output
    assert "1 ok, 1 need action, 0 errors" in result.output


def _noisy_check(_q, _m, name, _presence):
    logging.getLogger("arcaneum.indexing.verify").warning("Inconsistent chunk_count metadata")
    return diagnose_corpus(name, "both", _verification(), _synced(), None)


def test_doctor_quiets_verifier_warnings_unless_verbose(caplog):
    with (
        patch.object(corpus_doctor, "_discover_corpora", lambda _q, _m: {"Alpha": "both"}),
        patch.object(corpus_doctor, "_open_clients", lambda: (object(), object())),
        patch.object(corpus_doctor, "_check_corpus", _noisy_check),
    ):
        CliRunner().invoke(cli, ["corpus", "doctor"], catch_exceptions=False)
        assert "Inconsistent chunk_count" not in caplog.text

        CliRunner().invoke(cli, ["corpus", "doctor", "--verbose"], catch_exceptions=False)
        assert "Inconsistent chunk_count" in caplog.text


def test_doctor_checks_corpora_concurrently_but_reports_in_order():
    started = []

    def check(_q, _m, name, _presence):
        started.append(threading.current_thread().name)
        return diagnose_corpus(name, "both", _verification(), _synced(), None)

    names = {f"C{i}": "both" for i in range(6)}
    with (
        patch.object(corpus_doctor, "_discover_corpora", lambda _q, _m: names),
        patch.object(corpus_doctor, "_open_clients", lambda: (object(), object())),
        patch.object(corpus_doctor, "_check_corpus", check),
    ):
        result = CliRunner().invoke(cli, ["corpus", "doctor", "--json"], catch_exceptions=False)

    doc = json.loads(result.stdout)
    assert [c["corpus"] for c in doc["data"]["corpora"]] == list(names)
    assert threading.main_thread().name not in started


class _Qdrant:
    pass


def test_check_corpus_classifies_only_meili_only_files_missing_from_disk(tmp_path):
    on_disk = tmp_path / "here.pdf"
    on_disk.write_text("x")
    gone = str(tmp_path / "gone.pdf")
    classified = {}

    def classify(_qdrant, _sync_manager, _corpus, paths, force_prune=False):
        classified["paths"] = paths
        return {"prunable": paths, "dir_missing": [], "referenced": []}

    def no_rescan(*_args):
        raise AssertionError("the verifier already counted Qdrant points")

    def scan_meili(_meili, _name, _type, ids_only=False):
        assert ids_only
        return {str(on_disk): {"meili_chunks": 1}, gone: {"meili_chunks": 2}}

    with (
        patch(
            "arcaneum.indexing.verify.CollectionVerifier.verify_collection",
            lambda *_a, **_k: _verification(),
        ),
        patch("arcaneum.cli.corpus._scan_qdrant_parity_items", no_rescan),
        patch("arcaneum.cli.corpus._scan_meili_parity_items", scan_meili),
        patch("arcaneum.cli.sync._classify_meili_orphans", classify),
        patch("arcaneum.cli.sync.MetadataBasedSync", lambda _q: object()),
    ):
        result = corpus_doctor._check_corpus(_Qdrant(), object(), "Docs", "both")

    assert classified["paths"] == [gone]
    assert result["commands"] == ["arc corpus repair Docs --prune-orphans"]

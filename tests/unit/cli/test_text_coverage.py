"""Status surfaces report text coverage for a corpus whose text backend is qdrant (RDR-027 Q8).

A Qdrant corpus has no MeiliSearch side, so "parity" is coverage: the text
index's points against the chunk points not flagged ``text_tokens: false``.
``LocalQdrant`` reports a text index's points as the server does, and every
command runs end to end through the CLI over it.
"""

from __future__ import annotations

import json
import platform
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from fulltext.conftest import FakeFullTextClient, LocalQdrant

from arcaneum.indexing.collection_metadata import set_collection_metadata
from arcaneum.retrieval.qdrant_text import QdrantTextBackend, zero_token

CORPUS = "Notes"
MARKUP = "### ** `` __"
CHUNKS = [
    ("00000000-0000-0000-0000-0000000000a0", "/notes/a.md", 0, "The write-ahead log is flushed"),
    ("00000000-0000-0000-0000-0000000000a1", "/notes/a.md", 1, MARKUP),
    ("00000000-0000-0000-0000-0000000000b0", "/notes/b.md", 0, "Pages are checksummed on read"),
]


class RefusingMeili:
    """A MeiliSearch client no call may reach."""

    def __getattr__(self, name):
        raise AssertionError(f"MeiliSearch was called: {name}")


def build_qdrant_corpus(qdrant, name=CORPUS, *, flag=True):
    """A markdown corpus on qdrant: two tokenful chunks and one markup chunk."""
    from qdrant_client.http import models

    qdrant.create_collection_with_vector(name)
    set_collection_metadata(qdrant, name, "markdown", "arctic-m", text_backend="qdrant")
    QdrantTextBackend(qdrant).ensure(name, "markdown")
    points = []
    for chunk_id, path, index, text in CHUNKS:
        payload = {
            "text": text,
            "file_path": path,
            "filename": path.rsplit("/", 1)[-1],
            "chunk_index": index,
            "chunk_count": sum(1 for c in CHUNKS if c[1] == path),
        }
        if flag and zero_token(text, "markdown"):
            payload["text_tokens"] = False
        points.append(models.PointStruct(id=chunk_id, vector={"stub": [1.0, 0.0]}, payload=payload))
    qdrant.upsert(name, points=points)


def retain_stale_meili_index(meili, name=CORPUS):
    """The MeiliSearch index a migrated corpus keeps until --finalize: one stale document."""
    meili.create_index(name, settings={"filterableAttributes": ["file_path"]})
    meili.indexes[name]["docs"]["stale"] = {
        "id": "stale",
        "file_path": "/notes/gone.md",
        "filename": "gone.md",
        "content": "stale",
    }


@pytest.fixture
def services(monkeypatch, tmp_path):
    from arcaneum.cli import corpus, doctor, sync, utils

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    qdrant = LocalQdrant()
    state = {"meili": FakeFullTextClient()}
    monkeypatch.setattr(corpus, "create_qdrant_client", lambda: qdrant)
    monkeypatch.setattr(corpus, "create_meili_client", lambda: state["meili"])
    monkeypatch.setattr(corpus, "interaction_logger", MagicMock())
    monkeypatch.setattr(sync, "create_qdrant_client", lambda **_kw: qdrant)
    monkeypatch.setattr(sync, "get_meili_client", lambda: state["meili"])
    monkeypatch.setattr(sync, "interaction_logger", MagicMock())
    monkeypatch.setattr(utils, "create_qdrant_client", lambda **_kw: qdrant)
    monkeypatch.setattr(doctor, "create_qdrant_client", lambda **_kw: qdrant)
    build_qdrant_corpus(qdrant)
    return qdrant, state


def _json(*args):
    from arcaneum.cli.main import cli

    result = CliRunner().invoke(cli, [*args, "--json"], catch_exceptions=False)
    return result, json.loads(result.stdout)


def _listed(name=CORPUS):
    _result, doc = _json("corpus", "list")
    return {c["name"]: c for c in doc["data"]["corpora"]}[name]


def _doctor(name=CORPUS):
    _result, doc = _json("corpus", "doctor", name)
    return doc["data"]["corpora"][0]


def _drop_text_index(qdrant):
    qdrant.delete_payload_index(CORPUS, "text")
    qdrant.index_calls.clear()


# -- complete coverage -------------------------------------------------------


def test_list_reports_complete_coverage(services):
    entry = _listed()

    assert entry["text_backend"] == "qdrant"
    assert entry["status"] == "synced"
    assert (entry["qdrant_chunks"], entry["text_chunks"], entry["zero_token_chunks"]) == (3, 2, 1)
    assert not [key for key in entry if key.startswith("meili_")]


def test_a_meili_corpus_gains_text_chunks_in_the_listing(services):
    _qdrant, state = services
    state["meili"].create_index("IndexOnly")

    entry = _listed("IndexOnly")

    assert entry["text_backend"] == "meili"
    assert entry["text_chunks"] == entry["meili_chunks"] == 0


def test_info_reports_complete_coverage_without_meilisearch(services):
    _qdrant, state = services
    state["meili"] = RefusingMeili()

    result, doc = _json("corpus", "info", CORPUS)

    assert result.exit_code == 0, result.output
    data = doc["data"]
    assert data["parity"]["status"] == "synced"
    assert data["text_chunks"] == 2
    assert data["coverage"]["zero_token_chunks"] == 1
    assert data["meilisearch"] is None


def test_items_report_searchable_chunks_per_file(services):
    _qdrant, state = services
    state["meili"] = RefusingMeili()

    result, doc = _json("corpus", "items", CORPUS)

    assert result.exit_code == 0, result.output
    items = {item["id"]: item for item in doc["data"]["items"]}
    assert items["/notes/a.md"]["qdrant_chunks"] == 2
    assert items["/notes/a.md"]["searchable_chunks"] == 1
    assert items["/notes/a.md"]["zero_token_chunks"] == 1
    assert {item["status"] for item in items.values()} == {"synced"}
    assert not [key for item in items.values() for key in item if key.startswith("meili_")]
    assert doc["data"]["coverage"]["status"] == "synced"


def test_verify_reports_complete_coverage(services):
    _qdrant, state = services
    state["meili"] = RefusingMeili()

    result, doc = _json("corpus", "verify", CORPUS)

    assert result.exit_code == 0, result.output
    data = doc["data"]
    assert data["parity_status"] == "synced"
    assert data["parity"]["zero_token_chunks"] == 1
    assert data["meilisearch"] is None
    assert data["overall_healthy"] is True, data


def test_doctor_lists_the_zero_token_chunk_and_no_parity_finding(services):
    result = _doctor()

    assert result["status"] == "ok", result
    assert [f["code"] for f in result["findings"]] == ["zero_token_chunks"]
    finding = result["findings"][0]
    assert finding["severity"] == "info"
    assert finding["command"] is None
    assert finding["chunks"] == [{"file_path": "/notes/a.md", "chunk_index": 1}]


# -- a dropped text index ----------------------------------------------------


def test_a_dropped_index_is_listed_missing_and_doctor_names_repair(services):
    qdrant, _state = services
    _drop_text_index(qdrant)

    assert _listed()["status"] == "text_index_missing"
    result = _doctor()
    assert result["status"] == "needs_action"
    assert [f["code"] for f in result["findings"] if f["severity"] == "action"] == [
        "text_index_missing"
    ]
    assert result["commands"] == [f"arc corpus repair {CORPUS}"]


def test_a_dropped_index_makes_verify_unhealthy(services):
    qdrant, _state = services
    _drop_text_index(qdrant)

    _result, doc = _json("corpus", "verify", CORPUS)

    assert doc["data"]["parity_status"] == "text_index_missing"
    assert doc["data"]["overall_healthy"] is False


def test_an_unflagged_zero_token_chunk_is_a_coverage_gap(services):
    qdrant, _state = services
    build_qdrant_corpus(qdrant, "Unflagged", flag=False)

    assert _listed("Unflagged")["status"] == "coverage_gap"
    result = _doctor("Unflagged")
    assert "text_coverage_gap" in [f["code"] for f in result["findings"]]
    assert result["status"] == "needs_action"


@pytest.mark.skipif(platform.system() == "Windows", reason="flock-based corpus lock")
def test_repair_recreates_the_index_without_meilisearch(services, monkeypatch):
    from arcaneum.cli import sync

    qdrant, state = services
    _drop_text_index(qdrant)
    state["meili"] = RefusingMeili()
    monkeypatch.setattr(sync, "_sync_directory_locked", lambda *a, **k: None)

    result, doc = _json("corpus", "repair", CORPUS)

    assert result.exit_code == 0, result.output
    assert doc["data"]["parity"]["coverage"]["status"] == "synced"
    assert doc["data"]["parity"]["index_created"] is True
    assert QdrantTextBackend(qdrant).exists(CORPUS)
    assert _listed()["status"] == "synced"


def test_repair_dry_run_creates_nothing(services, monkeypatch):
    from arcaneum.cli import sync

    qdrant, _state = services
    _drop_text_index(qdrant)
    monkeypatch.setattr(sync, "_sync_directory_locked", lambda *a, **k: None)

    result, doc = _json("corpus", "repair", CORPUS, "--dry-run")

    assert result.exit_code == 0, result.output
    assert doc["data"]["parity"]["would_create_index"] is True
    assert qdrant.index_calls == []
    assert not QdrantTextBackend(qdrant).exists(CORPUS)


# -- negatives ---------------------------------------------------------------


@pytest.mark.skipif(platform.system() == "Windows", reason="flock-based corpus lock")
def test_a_retained_meili_index_is_left_untouched_by_parity_and_repair(services, monkeypatch):
    from arcaneum.cli import sync

    _qdrant, state = services
    retain_stale_meili_index(state["meili"])
    before = json.dumps(state["meili"].indexes, sort_keys=True)
    monkeypatch.setattr(sync, "_sync_directory_locked", lambda *a, **k: None)

    for args in (("corpus", "parity", CORPUS, "--verify"), ("corpus", "repair", CORPUS)):
        result, doc = _json(*args)
        assert result.exit_code == 0, result.output
        assert "chunk_mismatch" not in result.stdout
        assert "meili_backfilled" not in result.stdout

    assert json.dumps(state["meili"].indexes, sort_keys=True) == before
    assert _listed()["status"] == "synced"
    assert not [f for f in _doctor()["findings"] if f["code"].startswith("parity")]


def test_a_finalized_corpus_is_never_qdrant_only(services):
    from arcaneum.cli import sync

    codes = [f["code"] for f in _doctor()["findings"]]
    assert "qdrant_only_corpus" not in codes
    parity = {c["name"]: c for c in sync._discover_corpora_for_parity()}
    assert parity[CORPUS]["status"] == "synced"
    assert parity[CORPUS]["text_backend"] == "qdrant"
    assert parity[CORPUS]["to_meili"] == 0


def test_parity_all_never_creates_a_meili_index_for_a_qdrant_corpus(services):
    _qdrant, state = services

    result, doc = _json("corpus", "parity", "--create-missing", "--confirm")

    assert result.exit_code == 0, result.output
    assert CORPUS not in state["meili"].indexes


def test_list_reports_coverage_with_meilisearch_down(services):
    _qdrant, state = services
    state["meili"] = FakeFullTextClient(healthy=False)

    _result, doc = _json("corpus", "list")

    entry = {c["name"]: c for c in doc["data"]["corpora"]}[CORPUS]
    assert entry["status"] == "synced"
    assert entry["text_chunks"] == 2
    assert doc["data"]["service_errors"]


def test_no_read_creates_a_payload_index(services):
    qdrant, _state = services
    _drop_text_index(qdrant)

    for args in (
        ("corpus", "list"),
        ("corpus", "info", CORPUS),
        ("corpus", "items", CORPUS),
        ("corpus", "verify", CORPUS),
        ("corpus", "doctor"),
        ("corpus", "parity", CORPUS, "--dry-run"),
    ):
        _json(*args)

    assert qdrant.index_calls == []


def test_list_keeps_the_corpus_name_contract(services):
    _result, doc = _json("corpus", "list")
    assert CORPUS in [c["name"] for c in doc["data"]["corpora"]]



# -- the listing's chunk count (arcaneum#184) ----------------------------------


def _spy_bare_chunk_counts(qdrant, monkeypatch):
    """Record every count whose filter is the bare chunk filter: a full scan on a server."""
    from arcaneum.retrieval.qdrant_text import _chunk_filter

    calls = []
    real = qdrant._client.count

    def count(collection_name, count_filter=None, **kwargs):
        if count_filter == _chunk_filter():
            calls.append(collection_name)
        return real(collection_name, count_filter=count_filter, **kwargs)

    monkeypatch.setattr(qdrant, "count", count, raising=False)
    return calls


@pytest.mark.parametrize("args", [("corpus", "list"), ("corpus", "info", CORPUS)])
def test_display_paths_reuse_the_chunk_count(services, monkeypatch, args):
    qdrant, state = services
    state["meili"] = RefusingMeili()
    bare = _spy_bare_chunk_counts(qdrant, monkeypatch)

    result, doc = _json(*args)

    assert result.exit_code == 0, result.output
    assert bare == []
    if args[1] == "list":
        entry = {c["name"]: c for c in doc["data"]["corpora"]}[CORPUS]
        counts = (entry["qdrant_chunks"], entry["text_chunks"], entry["zero_token_chunks"])
    else:
        entry = doc["data"]["coverage"]
        counts = (entry["chunks"], entry["text_chunks"], entry["zero_token_chunks"])
    assert entry["status"] == "synced"
    assert counts == (3, 2, 1)


@pytest.mark.parametrize(
    "args",
    [("corpus", "verify", CORPUS), ("corpus", "doctor", CORPUS)],
)
def test_correctness_paths_keep_the_exact_count(services, monkeypatch, args):
    qdrant, _state = services
    bare = _spy_bare_chunk_counts(qdrant, monkeypatch)

    result, _doc = _json(*args)

    assert result.exit_code == 0, result.output

    assert CORPUS in bare


def test_parity_discovery_keeps_the_exact_count(services, monkeypatch):
    from arcaneum.cli import sync

    qdrant, _state = services
    bare = _spy_bare_chunk_counts(qdrant, monkeypatch)

    sync._discover_corpora_for_parity()

    assert CORPUS in bare


# -- arc doctor ----------------------------------------------------------------


def _text_backend_check(qdrant, version):
    from arcaneum.cli import doctor

    qdrant.info = lambda: SimpleNamespace(version=version)
    return doctor.check_text_backends()


def test_arc_doctor_counts_corpora_per_backend(services):
    qdrant, _state = services
    qdrant.create_collection_with_vector("Dual")
    set_collection_metadata(qdrant, "Dual", "pdf", "arctic-m")

    ok, message = _text_backend_check(qdrant, "1.20.2")

    assert ok is True
    assert "1 meili" in message and "1 qdrant" in message


def test_arc_doctor_fails_below_qdrant_1_20_when_a_corpus_is_on_qdrant(services):
    qdrant, _state = services

    ok, message = _text_backend_check(qdrant, "1.19.2")

    assert ok is False
    assert "1.20" in message


def test_arc_doctor_accepts_an_old_qdrant_with_only_meili_corpora(monkeypatch):
    from arcaneum.cli import doctor

    qdrant = LocalQdrant()
    qdrant.create_collection_with_vector("Dual")
    set_collection_metadata(qdrant, "Dual", "pdf", "arctic-m")
    qdrant.info = lambda: SimpleNamespace(version="1.18.2")
    monkeypatch.setattr(doctor, "create_qdrant_client", lambda **_kw: qdrant)

    ok, _message = doctor.check_text_backends()

    assert ok is True


@pytest.mark.parametrize("dropped", [False, True])
def test_human_output_reports_coverage(services, dropped, monkeypatch):
    from rich.console import Console

    from arcaneum.cli import corpus
    from arcaneum.cli.main import cli

    monkeypatch.setattr(corpus, "console", Console(width=200))  # the list table unwrapped
    qdrant, state = services
    state["meili"] = RefusingMeili()
    if dropped:
        _drop_text_index(qdrant)
    expected = "text_index_missing" if dropped else "synced"

    for args, text in (
        (("corpus", "list", "--details"), expected),
        (("corpus", "info", CORPUS), "Qdrant Text Index"),
        (("corpus", "items", CORPUS), "Coverage:"),
        (("corpus", "verify", CORPUS, "--verbose"), "Text Coverage"),
        (("corpus", "parity", CORPUS, "--dry-run"), "Zero-token chunks: 1"),
    ):
        result = CliRunner().invoke(cli, list(args), catch_exceptions=False)
        assert result.exit_code == 0, (args, result.output)
        assert text in result.output, (args, result.output)
        assert ("arc corpus repair" in result.output) == (dropped and args[1] in ("info", "verify"))

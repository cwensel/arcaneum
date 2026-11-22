"""``arc corpus migrate-text`` wiring and its gate helpers (RDR-027 Q9).

The migration itself runs against real stores in
``tests/integration/test_migrate_text.py``.
"""

from unittest.mock import patch

from click.testing import CliRunner

from arcaneum.cli import migrate_text
from arcaneum.cli.main import cli


def test_migrate_text_is_a_corpus_command():
    with patch.object(migrate_text, "migrate_text_command") as command:
        result = CliRunner().invoke(
            cli, ["corpus", "migrate-text", "Docs", "--to", "qdrant", "--dry-run", "--json"]
        )

    assert result.exit_code == 0, result.output
    command.assert_called_once_with("Docs", "qdrant", False, True, False, True)


def test_migrate_text_finalize_passes_through():
    with patch.object(migrate_text, "migrate_text_command") as command:
        result = CliRunner().invoke(cli, ["corpus", "migrate-text", "Docs", "--finalize"])

    assert result.exit_code == 0, result.output
    command.assert_called_once_with("Docs", None, True, False, False, False)


def test_migrate_text_needs_exactly_one_of_to_and_finalize():
    for args in ([], ["--to", "meili", "--finalize"]):
        result = CliRunner().invoke(cli, ["corpus", "migrate-text", "Docs", *args])
        assert result.exit_code == 2, result.output
        assert "exactly one of --to" in result.output


def test_migrate_text_refuses_an_unknown_backend():
    result = CliRunner().invoke(cli, ["corpus", "migrate-text", "Docs", "--to", "sqlite"])

    assert result.exit_code == 2


def test_gate_sample_is_one_percent_at_least_two_hundred():
    assert migrate_text.gate_sample_size(7) == 7
    assert migrate_text.gate_sample_size(5_000) == 200
    assert migrate_text.gate_sample_size(421_078) == 4_211


def test_distinctive_span_takes_the_highest_summed_idf_window():
    tokens = ["the", "log", "flushes", "dirty", "pages", "before", "checkpoint", "zebra"]
    idf = {t: 0.1 for t in tokens} | {"zebra": 9.0}

    assert migrate_text.distinctive_span(tokens, idf) == tokens[2:]
    assert migrate_text.distinctive_span(tokens[:3], idf) == tokens[:3]


# ---------------------------------------------------------------------------
# --all: every corpus not already on the target, one at a time
# ---------------------------------------------------------------------------

import json  # noqa: E402

import pytest  # noqa: E402

from arcaneum.cli.errors import CorpusLockUnavailable  # noqa: E402


@pytest.fixture
def meili_on(tmp_path, monkeypatch):
    """An existing user: MeiliSearch enabled."""
    home = tmp_path / "arcaneum"
    home.mkdir()
    (home / "config.yaml").write_text("meilisearch:\n  enabled: true\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    return home


@pytest.fixture
def stores():
    with (
        patch("arcaneum.cli.utils.create_qdrant_client") as qdrant,
        patch("arcaneum.cli.utils.create_meili_client") as meili,
    ):
        yield qdrant, meili


def _all(*args):
    return CliRunner().invoke(cli, ["corpus", "migrate-text", "--all", *args])


def _report(corpus, to="qdrant", dry_run=False):
    return {"corpus": corpus, "to": to, "dry_run": dry_run, "switched": not dry_run}


def test_corpora_on_reads_each_collections_text_backend():
    from fulltext.conftest import LocalQdrant

    from arcaneum.indexing.collection_metadata import set_collection_metadata
    from arcaneum.retrieval.corpora import corpora_on, meili_corpora

    qdrant = LocalQdrant()
    for name, backend in (("Papers", "meili"), ("Code", "qdrant"), ("Notes", None)):
        qdrant.create_collection_with_vector(name)
        if backend:
            set_collection_metadata(qdrant, name, "pdf", "stella", text_backend=backend)

    assert corpora_on(qdrant, "qdrant") == ["Code"]
    # No text_backend reads as meili, the legacy default.
    assert corpora_on(qdrant, "meili") == ["Notes", "Papers"]
    assert meili_corpora(qdrant) == ["Notes", "Papers"]


def test_all_to_qdrant_skips_a_failure_and_reports_it(meili_on, stores):
    def migrate(_q, _m, corpus, *, dry_run, say):
        if corpus == "B":
            raise migrate_text.MigrationError("Qdrant and MeiliSearch disagree for 'B'")
        return _report(corpus)

    with (
        patch("arcaneum.retrieval.corpora.corpora_on", return_value=["A", "B", "C"]) as on,
        patch.object(migrate_text, "migrate_to_qdrant", side_effect=migrate) as moved,
    ):
        result = _all("--to", "qdrant", "--json")

    assert result.exit_code == 1, result.output
    on.assert_called_once_with(stores[0].return_value, "meili")
    assert [c.args[2] for c in moved.call_args_list] == ["A", "B", "C"]
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    assert [(r["corpus"], r["ok"]) for r in payload["data"]["results"]] == [
        ("A", True),
        ("B", False),
        ("C", True),
    ]
    assert "disagree for 'B'" in payload["data"]["results"][1]["error"]
    assert "B" in payload["message"]


def test_all_to_qdrant_text_summary_names_the_failure(meili_on, stores):
    def migrate(_q, _m, corpus, *, dry_run, say):
        if corpus == "B":
            raise CorpusLockUnavailable("Another sync holds 'B'")
        return _report(corpus)

    with (
        patch("arcaneum.retrieval.corpora.corpora_on", return_value=["A", "B"]),
        patch.object(migrate_text, "migrate_to_qdrant", side_effect=migrate),
    ):
        result = _all("--to", "qdrant")

    assert result.exit_code == 1
    assert "1 of 2" in result.output
    assert "Another sync holds 'B'" in result.output


def test_all_finalize_iterates_only_qdrant_corpora(meili_on, stores):
    with (
        patch("arcaneum.retrieval.corpora.corpora_on", return_value=["Code"]) as on,
        patch.object(
            migrate_text,
            "finalize",
            return_value={"corpus": "Code", "finalize": True, "dry_run": False, "deleted": True},
        ) as finalized,
    ):
        result = _all("--finalize", "--json")

    assert result.exit_code == 0, result.output
    on.assert_called_once_with(stores[0].return_value, "qdrant")
    assert finalized.call_args.args[2] == "Code"
    assert json.loads(result.output)["status"] == "success"


def test_all_with_no_candidates_exits_zero(meili_on, stores):
    with (
        patch("arcaneum.retrieval.corpora.corpora_on", return_value=[]),
        patch.object(migrate_text, "migrate_to_qdrant") as moved,
    ):
        result = _all("--to", "qdrant")

    assert result.exit_code == 0, result.output
    assert "No corpus is on meili" in result.output
    assert not moved.called


def test_all_dry_run_takes_no_lock(meili_on, stores):
    with (
        patch("arcaneum.retrieval.corpora.corpora_on", return_value=["A", "B"]),
        patch.object(
            migrate_text,
            "migrate_to_qdrant",
            side_effect=lambda _q, _m, c, **_k: _report(c, dry_run=True),
        ) as moved,
        patch("arcaneum.cli.corpus_lock.acquire_corpus_lock") as lock,
    ):
        result = _all("--to", "qdrant", "--dry-run")

    assert result.exit_code == 0, result.output
    assert not lock.called
    assert all(c.kwargs["dry_run"] for c in moved.call_args_list)


def test_all_takes_each_corpus_lock(meili_on, stores):
    with (
        patch("arcaneum.retrieval.corpora.corpora_on", return_value=["A", "B"]),
        patch.object(
            migrate_text, "migrate_to_qdrant", side_effect=lambda _q, _m, c, **_k: _report(c)
        ),
        patch("arcaneum.cli.corpus_lock.acquire_corpus_lock") as lock,
    ):
        result = _all("--to", "qdrant")

    assert result.exit_code == 0, result.output
    assert [c.args[0] for c in lock.call_args_list] == ["A", "B"]


@pytest.mark.parametrize(
    "args",
    [
        ["corpus", "migrate-text", "Docs", "--all", "--to", "qdrant"],
        ["corpus", "migrate-text", "--to", "qdrant"],
        ["corpus", "migrate-text", "--all", "--to", "meili"],
    ],
)
def test_all_usage_errors(args):
    with patch.object(migrate_text, "migrate_text_all_command", create=True) as command:
        result = CliRunner().invoke(cli, args)

    assert result.exit_code == 2, result.output
    assert not command.called


def test_all_to_meili_names_the_per_corpus_command():
    result = CliRunner().invoke(cli, ["corpus", "migrate-text", "--all", "--to", "meili"])

    assert "arc corpus migrate-text <corpus> --to meili" in result.output


def test_all_finalize_refused_while_meilisearch_is_disabled(tmp_path, monkeypatch, stores):
    home = tmp_path / "arcaneum"
    home.mkdir()
    (home / "meilisearch.key").write_text("an-existing-key\n")
    (home / "config.yaml").write_text("meilisearch:\n  enabled: false\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    result = _all("--finalize")

    assert result.exit_code != 0
    assert "meilisearch.enabled" in result.output
    assert not stores[0].called
    assert not stores[1].called

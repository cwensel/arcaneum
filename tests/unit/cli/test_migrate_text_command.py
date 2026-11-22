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

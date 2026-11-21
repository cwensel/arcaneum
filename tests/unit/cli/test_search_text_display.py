"""`arc search text`'s human and JSON display across text backends (RDR-027 Q5)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from fulltext.conftest import FakeFullTextClient, LocalQdrant

from arcaneum.cli.fulltext import highlight_terms
from arcaneum.cli.main import cli
from arcaneum.fulltext.backend import MeiliTextBackend
from arcaneum.indexing.collection_metadata import set_collection_metadata
from arcaneum.retrieval.qdrant_text import QdrantTextBackend
from arcaneum.schema.document import DualIndexDocument

CHUNK = "00000000-0000-0000-0000-00000000c001"


@pytest.fixture
def stores(monkeypatch):
    from arcaneum.cli import fulltext

    qdrant = LocalQdrant()
    meili = FakeFullTextClient()
    for name, backend in (("Q", "qdrant"), ("M", "meili")):
        qdrant.create_collection_with_vector(name)
        set_collection_metadata(qdrant, name, "pdf", "stella", text_backend=backend)
        doc = DualIndexDocument(
            id=CHUNK,
            content="The write-ahead log flushes before the checkpoint.",
            file_path=f"/docs/{name}.pdf",
            filename=f"{name}.pdf",
            page_number=2,
            vectors={"stub": [1.0, 0.0]},
        )
        QdrantTextBackend(qdrant).add(name, [doc])
        if backend == "qdrant":
            QdrantTextBackend(qdrant).ensure(name, "pdf")
        else:
            MeiliTextBackend(meili).ensure(name, "pdf")
            MeiliTextBackend(meili).add(name, [doc])
    monkeypatch.setattr(fulltext, "create_meili_client", lambda: meili)
    monkeypatch.setattr(fulltext, "create_qdrant_client", lambda **kwargs: qdrant)
    monkeypatch.setattr(fulltext, "interaction_logger", MagicMock())
    return qdrant, meili


def _run(query, *corpora, json_output=False):
    args = ["search", "text", query, *[a for c in corpora for a in ("--corpus", c)]]
    return CliRunner().invoke(cli, args + (["--json"] if json_output else []))


class TestHighlightTerms:
    def test_query_tokens_are_marked_case_and_accent_blind(self):
        assert highlight_terms("Café LOG and log", "cafe log", "pdf") == (
            "[yellow]Café[/yellow] [yellow]LOG[/yellow] and [yellow]log[/yellow]"
        )

    def test_stop_words_follow_the_corpus_type(self):
        assert highlight_terms("the log", "the log", "pdf") == "the [yellow]log[/yellow]"
        assert highlight_terms("the log", "the log", "code") == (
            "[yellow]the[/yellow] [yellow]log[/yellow]"
        )

    def test_the_text_between_marks_is_escaped(self):
        assert highlight_terms("[bold]log[/bold]", "log", "pdf") == (
            "\\[bold][yellow]log[/yellow]\\[/bold]"
        )

    def test_a_phrase_marks_its_words(self):
        assert highlight_terms("write-ahead log", '"write-ahead log"', "pdf") == (
            "[yellow]write[/yellow]-[yellow]ahead[/yellow] [yellow]log[/yellow]"
        )


class TestTextDisplay:
    def test_human_output_reports_hits_returned_and_no_timing(self, stores):
        result = _run('"write-ahead log"', "Q", "M")

        assert result.exit_code == 0, result.output
        assert "Found 2 matches in '[Q, M]'" in result.output
        assert "ms)" not in result.output
        assert "Q:/docs/Q.pdf#page=2" in result.output
        assert "write-ahead log flushes" in result.output

    def test_zero_hits_suggest_semantic_search(self, stores):
        result = _run("nonexistentterm", "Q")

        assert result.exit_code == 0, result.output
        assert "No results found" in result.output
        assert 'arc search semantic nonexistentterm --corpus Q' in result.output

    def test_json_reports_hits_returned_and_the_hint_on_zero(self, stores):
        found = json.loads(_run("checkpoint", "Q", json_output=True).output)
        empty = json.loads(_run("nonexistentterm", "Q", "M", json_output=True).output)

        assert found["message"] == "Found 1 results"
        assert empty["message"] == (
            "Found 0 results. For a conceptual match try: "
            "arc search semantic nonexistentterm --corpus Q --corpus M"
        )
        assert empty["data"]["hits"] == []
        assert set(found["data"]) == set(empty["data"])

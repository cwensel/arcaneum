"""`arc corpus create --text-backend`, and list and info showing it (RDR-027 Q2)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from fulltext.conftest import FakeFullTextClient, LocalQdrant

from arcaneum.indexing.collection_metadata import get_collection_metadata
from arcaneum.retrieval.qdrant_text import QdrantTextBackend


class RefusingMeili:
    """A MeiliSearch client no call may reach."""

    def __getattr__(self, name):
        raise AssertionError(f"MeiliSearch was called: {name}")


@pytest.fixture
def services(monkeypatch):
    from arcaneum.cli import corpus

    qdrant = LocalQdrant()
    state = {"meili": FakeFullTextClient()}
    monkeypatch.setattr(corpus, "create_qdrant_client", lambda: qdrant)
    monkeypatch.setattr(corpus, "create_meili_client", lambda: state["meili"])
    monkeypatch.setattr(corpus, "interaction_logger", MagicMock())
    return qdrant, state


def _create(name, *extra):
    from arcaneum.cli.main import cli

    return CliRunner().invoke(
        cli, ["corpus", "create", name, "--type", "pdf", "--models", "stella", "--json", *extra]
    )


def test_a_qdrant_corpus_is_created_without_meilisearch(services):
    qdrant, state = services
    state["meili"] = RefusingMeili()

    result = _create("Docs", "--text-backend", "qdrant")

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)["data"]
    assert data["text_backend"] == "qdrant"
    assert data["meilisearch_index"] is None
    assert get_collection_metadata(qdrant, "Docs")["text_backend"] == "qdrant"
    assert QdrantTextBackend(qdrant).exists("Docs")
    assert ("Docs", "file_path") in qdrant.requested  # every corpus's indexes still made


def test_the_default_backend_is_meilisearch(services):
    qdrant, state = services

    result = _create("Docs")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["data"]["text_backend"] == "meili"
    assert get_collection_metadata(qdrant, "Docs")["text_backend"] == "meili"
    assert state["meili"].index_exists("Docs")
    assert not QdrantTextBackend(qdrant).exists("Docs")


def test_an_unknown_backend_is_refused_by_the_cli(services):
    qdrant, _state = services

    result = _create("Docs", "--text-backend", "sqlite")

    assert result.exit_code == 2
    assert "sqlite" in result.output
    assert not qdrant.collection_exists("Docs")


def test_list_and_info_show_each_corpus_backend(services):
    from arcaneum.cli.main import cli

    qdrant, state = services
    assert _create("Text", "--text-backend", "qdrant").exit_code == 0
    assert _create("Dual").exit_code == 0
    state["meili"].create_index("IndexOnly")

    listing = json.loads(CliRunner().invoke(cli, ["corpus", "list", "--json"]).output)
    backends = {c["name"]: c["text_backend"] for c in listing["data"]["corpora"]}
    assert backends == {"Dual": "meili", "IndexOnly": "meili", "Text": "qdrant"}

    for name, backend in (("Text", "qdrant"), ("Dual", "meili")):
        info = CliRunner().invoke(cli, ["corpus", "info", name, "--json"])
        assert json.loads(info.output)["data"]["text_backend"] == backend, info.output

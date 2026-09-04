"""Ordinary semantic search suppresses bibliography chunks by default."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from qdrant_client.http import models

from arcaneum.cli.main import cli
from arcaneum.schema.document import DualIndexDocument, to_meilisearch_doc, to_qdrant_point
from arcaneum.search.searcher import search_collection


def _run_search(*, include_references=False, query_filter=None):
    client = MagicMock()
    client.query_points.return_value.points = []
    embedder = MagicMock()
    embedder.generate_query_embedding.return_value = ("vector", [0.1, 0.2])

    search_collection(
        client,
        embedder,
        "evidence",
        "papers",
        query_filter=query_filter,
        include_references=include_references,
    )
    return client.query_points.call_args.kwargs["query_filter"]


def test_semantic_search_excludes_references_by_default_and_preserves_user_filter():
    user_condition = models.FieldCondition(key="author", match=models.MatchValue(value="Example"))
    query_filter = _run_search(query_filter=models.Filter(must=[user_condition]))

    assert user_condition in query_filter.must
    assert any(
        condition.key == "section_type" and condition.match.value == "references"
        for condition in query_filter.must_not
    )


def test_semantic_search_can_include_references_explicitly():
    query_filter = _run_search(include_references=True)

    assert not any(
        getattr(condition, "key", None) == "section_type" for condition in query_filter.must_not
    )


def test_semantic_search_cli_exposes_include_references_option():
    runner = CliRunner()
    with patch("arcaneum.cli.search.search_command") as search_command:
        result = runner.invoke(
            cli,
            [
                "search",
                "semantic",
                "evidence",
                "--corpus",
                "papers",
                "--include-references",
            ],
        )

    assert result.exit_code == 0
    assert search_command.call_args.kwargs["include_references"] is True


def test_section_metadata_is_persisted_to_both_search_indexes():
    doc = DualIndexDocument(
        content="[1] A cited work",
        file_path="/papers/example.pdf",
        filename="example.pdf",
        file_extension=".pdf",
        section_type="references",
        section_title="References",
        vectors={"vector": [0.1, 0.2]},
    )

    qdrant_payload = to_qdrant_point(doc).payload
    meili_doc = to_meilisearch_doc(doc)
    assert qdrant_payload["section_type"] == "references"
    assert qdrant_payload["section_title"] == "References"
    assert meili_doc["section_type"] == "references"
    assert meili_doc["section_title"] == "References"

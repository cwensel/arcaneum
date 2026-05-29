"""Regression tests for hiding Qdrant metadata sentinel points."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from qdrant_client.models import FieldCondition, Filter, MatchValue

from arcaneum.cli.export_import import BinaryExporter
from arcaneum.indexing.collection_metadata import metadata_exclusion_filter
from arcaneum.indexing.verify import CollectionVerifier
from arcaneum.search.searcher import search_collection


def _metadata_must_not(filter_: Filter) -> bool:
    return any(
        condition
        == FieldCondition(
            key="is_metadata",
            match=MatchValue(value=True),
        )
        for condition in filter_.must_not or []
    )


def test_metadata_exclusion_filter_preserves_existing_filter():
    repo_condition = FieldCondition(
        key="git_project_name",
        match=MatchValue(value="arcaneum"),
    )

    result = metadata_exclusion_filter(Filter(must=[repo_condition]))

    assert result.must == [repo_condition]
    assert _metadata_must_not(result)


def test_metadata_exclusion_filter_preserves_single_must_not_condition():
    archived_condition = FieldCondition(
        key="archived",
        match=MatchValue(value=True),
    )

    result = metadata_exclusion_filter(Filter(must_not=archived_condition))

    assert archived_condition in result.must_not
    assert _metadata_must_not(result)


def test_search_collection_excludes_metadata_points_from_qdrant_query():
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(points=[])
    embedder = MagicMock()
    embedder.generate_query_embedding.return_value = ("model", [0.1, 0.2])

    search_collection(client, embedder, "query", "docs")

    query_filter = client.query_points.call_args.kwargs["query_filter"]
    assert _metadata_must_not(query_filter)


def test_export_scroll_excludes_metadata_points_server_side():
    client = MagicMock()
    client.scroll.return_value = ([], None)
    exporter = BinaryExporter(client)

    list(exporter._scroll_points("docs", include_metadata_point=False))

    scroll_filter = client.scroll.call_args.kwargs["scroll_filter"]
    assert _metadata_must_not(scroll_filter)


def test_verify_code_collection_excludes_metadata_points_from_scroll():
    qdrant = MagicMock()
    qdrant.scroll.return_value = ([], None)
    verifier = CollectionVerifier(qdrant)

    verifier._verify_code_collection("code", total_points=1)

    scroll_filter = qdrant.scroll.call_args.kwargs["scroll_filter"]
    assert _metadata_must_not(scroll_filter)

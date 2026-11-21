"""``MeiliTextBackend`` behaviour beyond the shared contract (RDR-027 Q1)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from arcaneum.cli.errors import BadFilterError, ServiceDownError
from arcaneum.search.filters import parse_filter

from .conftest import FakeFullTextClient

SRC = Path(__file__).resolve().parents[3] / "src" / "arcaneum"


def _backend(**kwargs):
    from arcaneum.fulltext.backend import MeiliTextBackend

    client = FakeFullTextClient(**kwargs)
    client.create_index("Docs", settings={"filterableAttributes": ["file_path"]})
    client.add_documents_batch_parallel(
        "Docs",
        [[{"id": "a0", "file_path": "/a", "content": "x", "git_project_identifier": "p#main"}]],
    )
    return MeiliTextBackend(client), client


def _sdk_importers() -> list[str]:
    importers = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            if any(name == "meilisearch" or name.startswith("meilisearch.") for name in names):
                importers.append(str(path.relative_to(SRC)))
    return sorted(set(importers))


def test_only_the_fulltext_package_imports_the_meilisearch_sdk():
    outside = [path for path in _sdk_importers() if not path.startswith("fulltext/")]
    assert outside == []


def test_the_health_failure_names_how_to_start_meilisearch():
    backend, _client = _backend(healthy=False)
    with pytest.raises(ServiceDownError) as exc_info:
        backend.health()
    assert "MeiliSearch server not available" in str(exc_info.value)


def test_a_qdrant_json_filter_is_bad_filter_before_any_request():
    backend, _client = _backend()
    with pytest.raises(BadFilterError) as exc_info:
        backend.check_filter(parse_filter('{"must": [{"key": "a", "match": {"value": 1}}]}'))
    assert str(exc_info.value).startswith("Invalid filter:")


def test_a_native_expression_passes_through_to_meilisearch():
    backend, client = _backend()
    seen = []
    search = client.search

    def record(index_name, query, filter=None, **kwargs):
        seen.append(filter)
        return search(index_name, query, filter=filter, **kwargs)

    client.search = record
    backend.search("Docs", "x", filter=parse_filter('file_path = "/a"'))
    assert seen == ['file_path = "/a"']


def test_iter_documents_yields_dicts_across_pages():
    backend, client = _backend()
    client.add_documents_batch_parallel(
        "Docs", [[{"id": f"d{i}", "file_path": f"/d{i}", "content": "y"} for i in range(5)]]
    )
    documents = list(backend.iter_documents("Docs", ["id", "file_path"], batch_size=2))
    assert [d["id"] for d in documents] == ["a0", "d0", "d1", "d2", "d3", "d4"]
    assert all(isinstance(d, dict) and set(d) == {"id", "file_path"} for d in documents)


def test_update_documents_merges_fields_by_id():
    backend, client = _backend()
    backend.update_documents("Docs", [{"id": "a0", "git_version_identifier": "v"}])
    assert client.get_document("Docs", "a0")["git_version_identifier"] == "v"
    assert client.get_document("Docs", "a0")["content"] == "x"


def test_queued_documents_report_their_task_status():
    backend, client = _backend()
    task_uid = backend.enqueue_documents("Docs", [{"id": "n0", "file_path": "/n", "content": "z"}])
    status, error = backend.wait_for_task(task_uid, timeout_ms=1000)
    assert (status, error) == ("succeeded", None)


def test_the_connection_error_types_include_the_sdks():
    from meilisearch.errors import MeilisearchCommunicationError

    from arcaneum.fulltext.backend import connection_error_types

    assert MeilisearchCommunicationError in connection_error_types()

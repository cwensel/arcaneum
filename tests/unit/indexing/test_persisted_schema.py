"""Regression tests for persisted schema metadata compatibility."""

from types import SimpleNamespace

from arcaneum.embeddings.client import get_embedding_prompt_policy
from arcaneum.indexing.collection_metadata import (
    persisted_schema_issues,
    prompt_policy_issues,
    set_collection_metadata,
)
from arcaneum.indexing.verify import CollectionVerifier
from arcaneum.schema.document import PERSISTED_SCHEMA_VERSION


def _collection_info():
    return SimpleNamespace(
        points_count=0,
        config=SimpleNamespace(
            params=SimpleNamespace(vectors=SimpleNamespace(size=2)),
        ),
    )


class Qdrant:
    def __init__(self, metadata=None):
        self.metadata = metadata
        self.points = []
        self.scroll_calls = 0

    def get_collection(self, _name):
        return _collection_info()

    def upsert(self, collection_name, points):
        self.points = points

    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        if self.metadata is not None:
            return [SimpleNamespace(payload=self.metadata)]
        return []

    def scroll(self, **kwargs):
        self.scroll_calls += 1
        return [], None


def test_collection_metadata_defaults_include_persisted_schema_fields():
    qdrant = Qdrant()

    set_collection_metadata(qdrant, "docs", "markdown", "stella")

    payload = qdrant.points[0].payload
    assert payload["schema_version"] == PERSISTED_SCHEMA_VERSION
    assert payload["app_version"] != ""
    assert payload["embedding_prompt_policy"]["stella"] == get_embedding_prompt_policy("stella")


def test_persisted_schema_issues_flags_legacy_metadata():
    issues = persisted_schema_issues({})

    assert issues == [
        "collection metadata is legacy schema v0; reindex or backfill "
        "schema_version/app_version before relying on persisted compatibility"
    ]


def test_prompt_policy_issues_flag_missing_and_changed_policy():
    assert "missing embedding_prompt_policy" in prompt_policy_issues({}, "stella")[0]

    stale_metadata = {
        "embedding_prompt_policy": {
            "stella": {
                **get_embedding_prompt_policy("stella"),
                "query": {"method": "encode"},
            }
        }
    }

    assert "differs" in prompt_policy_issues(stale_metadata, "stella")[0]
    assert prompt_policy_issues(
        {"embedding_prompt_policy": {"stella": get_embedding_prompt_policy("stella")}},
        "stella",
    ) == []


def test_prompt_policy_issues_reject_legacy_jina_code_backend():
    legacy_jina_code_policy = {
        **get_embedding_prompt_policy("jina-code-st"),
        "model": "jina-code",
    }

    issues = prompt_policy_issues(
        {"embedding_prompt_policy": {"jina-code": legacy_jina_code_policy}},
        "jina-code",
    )

    assert "differs" in issues[0]


def test_prompt_policy_issues_accept_legacy_jina_provider_id():
    metadata = {
        "embedding_prompt_policy": {"jina-code": get_embedding_prompt_policy("jina-code")}
    }

    assert prompt_policy_issues(
        metadata,
        "jinaai/jina-embeddings-v2-base-code",
    ) == []


def test_persisted_schema_issues_rejects_non_integer_version():
    assert persisted_schema_issues({"schema_version": True}) == [
        "collection metadata has invalid schema_version True"
    ]
    assert persisted_schema_issues({"schema_version": 1.9}) == [
        "collection metadata has invalid schema_version 1.9"
    ]


def test_verify_marks_legacy_collection_unhealthy():
    qdrant = Qdrant()
    verifier = CollectionVerifier(qdrant)

    result = verifier.verify_collection("docs")

    assert result.is_healthy is False
    assert result.schema_version is None
    assert "legacy schema v0" in result.errors[0]

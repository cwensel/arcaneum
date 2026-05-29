"""CLI tests for corpus management commands.

Tests for 'arc corpus' subcommands: create, list, delete, info, items, verify, parity, sync.
"""

import json
from types import SimpleNamespace


def _collection_info(points_count=6, vectors=None):
    if vectors is None:
        vectors = SimpleNamespace(size=768, distance="Cosine")
    return SimpleNamespace(
        points_count=points_count,
        config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)),
    )


def _mock_corpus_list_clients(
    monkeypatch,
    metadata_by_name,
    collection_info_by_name,
    meili_chunks=None,
    qdrant_payloads_by_name=None,
    qdrant_scroll_calls=None,
):
    from arcaneum.cli import corpus

    meili_chunks = meili_chunks or {}
    qdrant_payloads_by_name = qdrant_payloads_by_name or {}

    class Qdrant:
        def get_collections(self):
            return SimpleNamespace(
                collections=[SimpleNamespace(name=name) for name in metadata_by_name]
            )

        def get_collection(self, name):
            return collection_info_by_name[name]

        def scroll(self, collection_name, **_kwargs):
            if qdrant_scroll_calls is not None:
                qdrant_scroll_calls[collection_name] = (
                    qdrant_scroll_calls.get(collection_name, 0) + 1
                )
            payloads = qdrant_payloads_by_name.get(collection_name, [])
            return [SimpleNamespace(payload=payload) for payload in payloads], None

    class Meili:
        def health_check(self):
            return True

        def list_indexes(self):
            return [{"uid": name} for name in meili_chunks]

        def get_index_stats(self, name):
            return {"numberOfDocuments": meili_chunks[name]}

    monkeypatch.setattr(corpus, "create_qdrant_client", lambda: Qdrant())
    monkeypatch.setattr(corpus, "create_meili_client", lambda: Meili())
    monkeypatch.setattr(
        corpus,
        "get_collection_metadata",
        lambda _client, name: metadata_by_name.get(name, {}),
    )


def _mock_corpus_update_clients(monkeypatch, metadata):
    from arcaneum.cli import corpus

    class Qdrant:
        def get_collection(self, name):
            return _collection_info()

    monkeypatch.setattr(corpus, "create_qdrant_client", lambda: Qdrant())
    monkeypatch.setattr(
        corpus,
        "update_collection_metadata",
        lambda _client, _name, **_updates: metadata,
    )


def test_corpus_module_exports_expected_commands():
    """Every documented 'arc corpus' subcommand function is importable."""
    from arcaneum.cli import corpus

    expected = {
        "create_corpus_command",
        "update_corpus_command",
        "list_corpora_command",
        "delete_corpus_command",
        "corpus_info_command",
        "corpus_verify_command",
    }
    missing = [name for name in expected if not callable(getattr(corpus, name, None))]
    assert missing == [], f"corpus module missing exported commands: {missing}"


def test_corpus_module_uses_interaction_logger():
    """corpus commands rely on the shared interaction_logger singleton."""
    from arcaneum.cli.corpus import interaction_logger
    from arcaneum.cli.interaction_logger import interaction_logger as canonical

    # Must be the same singleton, not some unrelated object with matching attrs
    assert interaction_logger is canonical


class TestCorpusCollectionMetadata:
    """CollectionType enum covers all corpus types the CLI advertises."""

    def test_collection_type_values_include_pdf_code_markdown(self):
        from arcaneum.indexing.collection_metadata import CollectionType

        values = set(CollectionType.values())
        assert {"pdf", "code", "markdown"}.issubset(values), (
            f"CollectionType missing required values: {values}"
        )


class TestCorpusDefaultModels:
    """Stable model defaults are inferred from corpus type."""

    def test_document_corpora_default_to_arctic_m(self):
        from arcaneum.cli.corpus import default_models_for_type

        assert default_models_for_type("pdf") == "arctic-m"
        assert default_models_for_type("markdown") == "arctic-m"

    def test_code_corpora_default_to_lightweight_code_model(self):
        from arcaneum.cli.corpus import default_models_for_type

        assert default_models_for_type("code") == "jina-code"


class TestCorpusMarkdownChunking:
    """Corpus sync markdown chunking should avoid embedding-time truncation."""

    def test_chunk_markdown_file_applies_hard_max_chars(self, tmp_path):
        from arcaneum.cli.sync import chunk_markdown_file

        marker = "TAIL_SENTINEL"
        file_path = tmp_path / "large.md"
        file_path.write_text("# Big\n\n" + ("A" * 900) + marker, encoding="utf-8")

        chunks = chunk_markdown_file(
            file_path,
            chunk_size=1000,
            chunk_overlap=10,
            hard_max_chars=300,
        )

        assert len(chunks) > 1
        assert all(len(chunk["text"]) <= 300 for chunk in chunks)
        assert any(marker in chunk["text"] for chunk in chunks)
        assert any(chunk["metadata"].get("hard_split") is True for chunk in chunks)


class TestCorpusListModelInfo:
    """Corpus list exposes embedding model metadata for selection and audits."""

    def test_json_includes_single_model_backend_vector_and_policy(self, monkeypatch, capsys):
        from arcaneum.cli.corpus import list_corpora_command

        _mock_corpus_list_clients(
            monkeypatch,
            metadata_by_name={
                "docs": {
                    "collection_type": "pdf",
                    "model": "arctic-m",
                    "prompt_policy": "none",
                    "schema_policy": "rdr-009",
                }
            },
            collection_info_by_name={
                "docs": _collection_info(vectors=SimpleNamespace(size=768, distance="Cosine"))
            },
            meili_chunks={"docs": 5},
        )

        list_corpora_command(details=False, output_json=True)

        payload = json.loads(capsys.readouterr().out)
        corpus = payload["data"]["corpora"][0]
        assert corpus["description"] is None
        assert corpus["model_summary"] == "arctic-m (fastembed)"
        assert corpus["models"] == [
            {
                "alias": "arctic-m",
                "name": "snowflake/snowflake-arctic-embed-m",
                "backend": "fastembed",
                "vector_name": "unknown/legacy",
                "dimensions": 768,
                "distance": "Cosine",
                "policy": {
                    "prompt_policy": "none",
                    "schema_policy": "rdr-009",
                },
            }
        ]

    def test_json_includes_named_vector_multi_model_details(self, monkeypatch, capsys):
        from arcaneum.cli.corpus import list_corpora_command

        _mock_corpus_list_clients(
            monkeypatch,
            metadata_by_name={
                "code": {
                    "collection_type": "code",
                    "model": "jina-code,codesage-large",
                }
            },
            collection_info_by_name={
                "code": _collection_info(
                    vectors={
                        "codesage-large": SimpleNamespace(size=1024, distance="Cosine"),
                        "jina-code": SimpleNamespace(size=768, distance="Cosine"),
                    }
                )
            },
            meili_chunks={"code": 5},
        )

        list_corpora_command(details=False, output_json=True)

        corpus = json.loads(capsys.readouterr().out)["data"]["corpora"][0]
        assert corpus["model_summary"] == (
            "jina-code (fastembed), codesage-large (sentence-transformers)"
        )
        assert [model["vector_name"] for model in corpus["models"]] == [
            "jina-code",
            "codesage-large",
        ]
        assert [model["backend"] for model in corpus["models"]] == [
            "fastembed",
            "sentence-transformers",
        ]

    def test_json_marks_legacy_corpus_without_metadata_explicitly(self, monkeypatch, capsys):
        from arcaneum.cli.corpus import UNKNOWN_LEGACY, list_corpora_command

        _mock_corpus_list_clients(
            monkeypatch,
            metadata_by_name={"legacy": {}},
            collection_info_by_name={"legacy": _collection_info(points_count=1, vectors=None)},
        )

        list_corpora_command(details=False, output_json=True)

        corpus = json.loads(capsys.readouterr().out)["data"]["corpora"][0]
        assert corpus["model"] is None
        assert corpus["model_summary"] == UNKNOWN_LEGACY
        assert corpus["models"][0]["alias"] == UNKNOWN_LEGACY
        assert corpus["models"][0]["backend"] == UNKNOWN_LEGACY

    def test_json_uses_named_vector_when_model_metadata_is_missing(self, monkeypatch, capsys):
        from arcaneum.cli.corpus import list_corpora_command

        _mock_corpus_list_clients(
            monkeypatch,
            metadata_by_name={"legacy-code": {}},
            collection_info_by_name={
                "legacy-code": _collection_info(
                    vectors={
                        "jina-code": SimpleNamespace(size=768, distance="Cosine"),
                    }
                )
            },
        )

        list_corpora_command(details=False, output_json=True)

        corpus = json.loads(capsys.readouterr().out)["data"]["corpora"][0]
        assert corpus["model_summary"] == "jina-code (fastembed)"
        assert corpus["models"][0]["alias"] == "jina-code"
        assert corpus["models"][0]["vector_name"] == "jina-code"

    def test_table_view_shows_concise_model_info_without_verbose(self, monkeypatch, capsys):
        from arcaneum.cli.corpus import list_corpora_command

        _mock_corpus_list_clients(
            monkeypatch,
            metadata_by_name={"docs": {"collection_type": "pdf", "model": "arctic-m"}},
            collection_info_by_name={"docs": _collection_info()},
        )

        list_corpora_command(details=False, output_json=False)

        output = capsys.readouterr().out
        assert "arctic-m" in output
        assert "fastembed" in output

    def test_json_default_skips_exact_item_counts(self, monkeypatch, capsys):
        from arcaneum.cli.corpus import UNKNOWN_LEGACY, list_corpora_command

        scroll_calls = {}
        _mock_corpus_list_clients(
            monkeypatch,
            metadata_by_name={
                "code": {
                    "collection_type": "code",
                    "model": "jina-code",
                    "last_sync": "2026-05-28T22:01:02+00:00",
                },
                "docs": {
                    "collection_type": "pdf",
                    "model": "arctic-m",
                    "last_sync": "2026-05-28T22:03:04+00:00",
                },
                "legacy": {},
                "markdown": {
                    "collection_type": "markdown",
                    "model": "arctic-m",
                },
            },
            collection_info_by_name={
                "code": _collection_info(points_count=4),
                "docs": _collection_info(points_count=4),
                "legacy": _collection_info(points_count=1, vectors=None),
                "markdown": _collection_info(points_count=3),
            },
            qdrant_payloads_by_name={
                "code": [
                    {"file_path": "/repo/a.py", "file_paths": ["/repo/a.py", "/repo/c.py"]},
                    {"file_path": "/repo/a.py"},
                    {"file_path": "/repo/b.py"},
                ],
                "docs": [
                    {"file_path": "/docs/guide.pdf"},
                    {"file_path": "/docs/guide.pdf"},
                    {"file_path": "/docs/spec.pdf"},
                ],
                "markdown": [
                    {"file_path": "/notes/a.md"},
                    {"file_path": "/notes/b.md"},
                ],
            },
            qdrant_scroll_calls=scroll_calls,
        )

        list_corpora_command(details=False, output_json=True)

        corpora = {
            item["name"]: item for item in json.loads(capsys.readouterr().out)["data"]["corpora"]
        }
        assert corpora["code"]["last_sync"] == "2026-05-28T22:01:02+00:00"
        assert corpora["code"]["item_count"] is None
        assert corpora["code"]["item_unit"] == "repositories"
        assert corpora["code"]["file_count"] is None
        assert corpora["code"]["file_unit"] is None
        assert corpora["code"]["item_count_status"] == "not_requested"
        assert corpora["docs"]["item_count"] is None
        assert corpora["docs"]["item_unit"] == "documents"
        assert corpora["markdown"]["last_sync"] is None
        assert corpora["markdown"]["last_sync_status"] == "unknown"
        assert corpora["markdown"]["item_count"] is None
        assert corpora["markdown"]["item_unit"] == "files"
        assert corpora["legacy"]["last_sync_status"] == "unknown"
        assert corpora["legacy"]["item_count"] is None
        assert corpora["legacy"]["item_unit"] == UNKNOWN_LEGACY
        assert scroll_calls == {}

    def test_json_marks_chunk_mismatches(self, monkeypatch, capsys):
        from arcaneum.cli.corpus import list_corpora_command

        _mock_corpus_list_clients(
            monkeypatch,
            metadata_by_name={
                "docs": {
                    "collection_type": "pdf",
                    "model": "arctic-m",
                }
            },
            collection_info_by_name={
                "docs": _collection_info(points_count=6),
            },
            meili_chunks={"docs": 4},
        )

        list_corpora_command(details=False, output_json=True)

        corpus = json.loads(capsys.readouterr().out)["data"]["corpora"][0]
        assert corpus["status"] == "chunk_mismatch"
        assert corpus["qdrant_chunks"] == 5
        assert corpus["meili_chunks"] == 4

    def test_chunk_summary_collapses_equal_counts(self):
        from arcaneum.cli.corpus import _format_chunk_summary

        assert (
            _format_chunk_summary(
                {
                    "qdrant_chunks": 5,
                    "meili_chunks": 5,
                }
            )
            == "5"
        )
        assert (
            _format_chunk_summary(
                {
                    "qdrant_chunks": 5,
                    "meili_chunks": 4,
                }
            )
            == "Q: 5 / M: 4"
        )

    def test_json_details_includes_last_sync_and_type_specific_item_counts(
        self, monkeypatch, capsys
    ):
        from arcaneum.cli.corpus import UNKNOWN_LEGACY, list_corpora_command

        _mock_corpus_list_clients(
            monkeypatch,
            metadata_by_name={
                "code": {
                    "collection_type": "code",
                    "model": "jina-code",
                    "last_sync": "2026-05-28T22:01:02+00:00",
                },
                "docs": {
                    "collection_type": "pdf",
                    "model": "arctic-m",
                    "last_sync": "2026-05-28T22:03:04+00:00",
                },
                "legacy": {},
                "markdown": {
                    "collection_type": "markdown",
                    "model": "arctic-m",
                },
            },
            collection_info_by_name={
                "code": _collection_info(points_count=4),
                "docs": _collection_info(points_count=4),
                "legacy": _collection_info(points_count=1, vectors=None),
                "markdown": _collection_info(points_count=3),
            },
            qdrant_payloads_by_name={
                "code": [
                    {
                        "git_project_identifier": "repo-a#main",
                        "file_path": "/repo-a/a.py",
                        "file_paths": ["/repo-a/a.py", "/repo-a/c.py"],
                    },
                    {
                        "git_project_identifier": "repo-a#main",
                        "file_path": "/repo-a/a.py",
                    },
                    {
                        "git_project_identifier": "repo-b#main",
                        "file_path": "/repo-b/b.py",
                    },
                ],
                "docs": [
                    {"file_path": "/docs/guide.pdf"},
                    {"file_path": "/docs/guide.pdf"},
                    {"file_path": "/docs/spec.pdf"},
                ],
                "markdown": [
                    {"file_path": "/notes/a.md"},
                    {"file_path": "/notes/b.md"},
                ],
            },
        )

        list_corpora_command(details=True, output_json=True)

        corpora = {
            item["name"]: item for item in json.loads(capsys.readouterr().out)["data"]["corpora"]
        }
        assert corpora["code"]["last_sync"] == "2026-05-28T22:01:02+00:00"
        assert corpora["code"]["item_count"] == 2
        assert corpora["code"]["item_unit"] == "repositories"
        assert corpora["code"]["file_count"] == 3
        assert corpora["code"]["file_unit"] == "source files"
        assert corpora["code"]["item_count_status"] == "exact"
        assert corpora["docs"]["item_count"] == 2
        assert corpora["docs"]["item_unit"] == "documents"
        assert corpora["markdown"]["last_sync"] is None
        assert corpora["markdown"]["last_sync_status"] == "unknown"
        assert corpora["markdown"]["item_count"] == 2
        assert corpora["markdown"]["item_unit"] == "files"
        assert corpora["legacy"]["last_sync_status"] == "unknown"
        assert corpora["legacy"]["item_count"] is None
        assert corpora["legacy"]["item_unit"] == UNKNOWN_LEGACY

    def test_qdrant_item_count_uses_repositories_for_code(self):
        from arcaneum.cli.corpus import _get_qdrant_item_count

        class Qdrant:
            def scroll(self, **_kwargs):
                return [
                    SimpleNamespace(
                        payload={
                            "git_project_identifier": "repo-a#main",
                        }
                    ),
                    SimpleNamespace(
                        payload={
                            "git_project_identifier": "repo-a#main",
                        }
                    ),
                    SimpleNamespace(
                        payload={
                            "git_project_identifier": "repo-b#main",
                        }
                    ),
                ], None

        assert _get_qdrant_item_count(Qdrant(), "code", "code") == 2

    def test_qdrant_item_count_includes_deduplicated_file_paths_for_docs(self):
        from arcaneum.cli.corpus import _get_qdrant_item_count

        class Qdrant:
            def scroll(self, **_kwargs):
                return [
                    SimpleNamespace(
                        payload={
                            "file_path": "/docs/renamed-a.md",
                            "file_paths": ["/docs/a.md", "/docs/b.md"],
                        }
                    ),
                    SimpleNamespace(payload={"file_path": "/docs/c.md"}),
                ], None

        assert _get_qdrant_item_count(Qdrant(), "docs", "markdown") == 3

    def test_qdrant_list_item_count_uses_repositories_for_code(self):
        from arcaneum.cli.corpus import _get_qdrant_list_item_count

        class Qdrant:
            def scroll(self, **_kwargs):
                return [
                    SimpleNamespace(
                        payload={
                            "git_project_identifier": "repo-a#main",
                            "file_path": "/repo-a/renamed-a.py",
                            "file_paths": ["/repo-a/a.py", "/repo-a/b.py"],
                        }
                    ),
                    SimpleNamespace(
                        payload={
                            "git_project_identifier": "repo-a#main",
                            "file_path": "/repo-a/c.py",
                        }
                    ),
                    SimpleNamespace(
                        payload={
                            "git_project_identifier": "repo-b#main",
                            "file_path": "/repo-b/d.py",
                        }
                    ),
                ], None

        assert _get_qdrant_list_item_count(Qdrant(), "code", "code") == {
            "item_count": 2,
            "item_unit": "repositories",
            "file_count": 4,
            "file_unit": "source files",
        }


class TestCorpusDescriptions:
    """Corpus description metadata is create/update/list/info visible."""

    def test_list_json_includes_description(self, monkeypatch, capsys):
        from arcaneum.cli.corpus import list_corpora_command

        _mock_corpus_list_clients(
            monkeypatch,
            metadata_by_name={
                "docs": {
                    "collection_type": "markdown",
                    "model": "arctic-m",
                    "description": "Design notes and decision records",
                }
            },
            collection_info_by_name={"docs": _collection_info()},
        )

        list_corpora_command(details=False, output_json=True)

        corpus = json.loads(capsys.readouterr().out)["data"]["corpora"][0]
        assert corpus["description"] == "Design notes and decision records"

    def test_update_description_outputs_new_metadata(self, monkeypatch, capsys):
        from arcaneum.cli.corpus import update_corpus_command

        _mock_corpus_update_clients(
            monkeypatch,
            {"description": "Updated project notes"},
        )

        update_corpus_command(
            "docs",
            description="Updated project notes",
            clear_description=False,
            output_json=True,
        )

        payload = json.loads(capsys.readouterr().out)
        assert payload["data"] == {
            "corpus": "docs",
            "description": "Updated project notes",
        }

    def test_update_allows_empty_description(self, monkeypatch, capsys):
        from arcaneum.cli.corpus import update_corpus_command

        _mock_corpus_update_clients(monkeypatch, {"description": ""})

        update_corpus_command(
            "docs",
            description="",
            clear_description=False,
            output_json=True,
        )

        payload = json.loads(capsys.readouterr().out)
        assert payload["data"]["description"] == ""

    def test_clear_description_removes_metadata_value(self, monkeypatch, capsys):
        from arcaneum.cli.corpus import update_corpus_command

        _mock_corpus_update_clients(monkeypatch, {})

        update_corpus_command(
            "docs",
            description=None,
            clear_description=True,
            output_json=True,
        )

        payload = json.loads(capsys.readouterr().out)
        assert payload["data"]["description"] is None

    def test_metadata_update_preserves_existing_fields(self, monkeypatch):
        from arcaneum.indexing import collection_metadata
        from arcaneum.indexing.collection_metadata import update_collection_metadata

        class Qdrant:
            def __init__(self):
                self.points = None

            def get_collection(self, name):
                return _collection_info()

            def upsert(self, collection_name, points):
                self.points = points

        qdrant = Qdrant()
        monkeypatch.setattr(
            collection_metadata,
            "get_collection_metadata",
            lambda _client, _name: {
                "collection_type": "markdown",
                "model": "arctic-m",
                "created_at": "2026-05-26T00:00:00",
            },
        )

        updated = update_collection_metadata(
            qdrant,
            "docs",
            description="Project notes",
        )

        assert updated["collection_type"] == "markdown"
        assert updated["model"] == "arctic-m"
        assert updated["created_at"] == "2026-05-26T00:00:00"
        assert updated["description"] == "Project notes"
        assert "schema_version" not in updated
        assert "app_version" not in updated
        assert qdrant.points[0].payload["is_metadata"] is True
        assert qdrant.points[0].payload["description"] == "Project notes"

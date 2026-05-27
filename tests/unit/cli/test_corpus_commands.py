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
):
    from arcaneum.cli import corpus

    meili_chunks = meili_chunks or {}

    class Qdrant:
        def get_collections(self):
            return SimpleNamespace(
                collections=[SimpleNamespace(name=name) for name in metadata_by_name]
            )

        def get_collection(self, name):
            return collection_info_by_name[name]

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


def test_corpus_module_exports_expected_commands():
    """Every documented 'arc corpus' subcommand function is importable."""
    from arcaneum.cli import corpus

    expected = {
        'create_corpus_command',
        'list_corpora_command',
        'delete_corpus_command',
        'corpus_info_command',
        'corpus_verify_command',
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
        assert {'pdf', 'code', 'markdown'}.issubset(values), \
            f"CollectionType missing required values: {values}"


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
                "docs": _collection_info(
                    vectors=SimpleNamespace(size=768, distance="Cosine")
                )
            },
            meili_chunks={"docs": 5},
        )

        list_corpora_command(verbose=False, output_json=True)

        payload = json.loads(capsys.readouterr().out)
        corpus = payload["data"]["corpora"][0]
        assert corpus["model_summary"] == "arctic-m (fastembed)"
        assert corpus["models"] == [{
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
        }]

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

        list_corpora_command(verbose=False, output_json=True)

        corpus = json.loads(capsys.readouterr().out)["data"]["corpora"][0]
        assert corpus["model_summary"] == (
            "jina-code (sentence-transformers), "
            "codesage-large (sentence-transformers)"
        )
        assert [model["vector_name"] for model in corpus["models"]] == [
            "jina-code",
            "codesage-large",
        ]
        assert [model["backend"] for model in corpus["models"]] == [
            "sentence-transformers",
            "sentence-transformers",
        ]

    def test_json_marks_legacy_corpus_without_metadata_explicitly(self, monkeypatch, capsys):
        from arcaneum.cli.corpus import UNKNOWN_LEGACY, list_corpora_command

        _mock_corpus_list_clients(
            monkeypatch,
            metadata_by_name={"legacy": {}},
            collection_info_by_name={
                "legacy": _collection_info(points_count=1, vectors=None)
            },
        )

        list_corpora_command(verbose=False, output_json=True)

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

        list_corpora_command(verbose=False, output_json=True)

        corpus = json.loads(capsys.readouterr().out)["data"]["corpora"][0]
        assert corpus["model_summary"] == "jina-code (sentence-transformers)"
        assert corpus["models"][0]["alias"] == "jina-code"
        assert corpus["models"][0]["vector_name"] == "jina-code"

    def test_table_view_shows_concise_model_info_without_verbose(self, monkeypatch, capsys):
        from arcaneum.cli.corpus import list_corpora_command

        _mock_corpus_list_clients(
            monkeypatch,
            metadata_by_name={
                "docs": {"collection_type": "pdf", "model": "arctic-m"}
            },
            collection_info_by_name={"docs": _collection_info()},
        )

        list_corpora_command(verbose=False, output_json=False)

        output = capsys.readouterr().out
        assert "arctic-m" in output
        assert "fastembed" in output

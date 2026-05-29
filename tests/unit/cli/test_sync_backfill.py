"""Regression tests for sync backfill helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from arcaneum.cli import sync as sync_module
from arcaneum.cli.sync import (
    _backfill_meili_to_qdrant,
    _fetch_chunks_for_files_bulk,
    _maybe_backfill_legacy_prompt_policy,
    _repair_meili_metadata,
)
from arcaneum.embeddings.client import get_embedding_prompt_policy


class MetadataQdrant:
    def __init__(self, metadata):
        self.metadata = dict(metadata)
        self.upserted = None

    def get_collection(self, _name):
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=2)))
        )

    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        return [SimpleNamespace(payload={**self.metadata, "is_metadata": True})]

    def upsert(self, collection_name, points):
        self.upserted = points[0].payload
        self.metadata = dict(self.upserted)


@pytest.mark.parametrize("corpus_type", ["pdf", "markdown", "code"])
def test_corpus_sync_backfills_missing_prompt_policy_for_all_types(corpus_type):
    qdrant = MetadataQdrant({"collection_type": corpus_type, "model": "stella"})

    metadata, issues = _maybe_backfill_legacy_prompt_policy(
        qdrant=qdrant,
        corpus="Docs",
        corpus_type=corpus_type,
        model_list=["stella"],
        metadata={"collection_type": corpus_type, "model": "stella"},
        output_json=True,
    )

    assert issues == []
    assert metadata["embedding_prompt_policy"]["stella"] == get_embedding_prompt_policy("stella")
    assert qdrant.upserted["collection_type"] == corpus_type


def test_corpus_sync_does_not_backfill_changed_prompt_policy():
    stale_policy = {
        **get_embedding_prompt_policy("stella"),
        "query": {"method": "encode"},
    }
    metadata = {
        "collection_type": "markdown",
        "model": "stella",
        "embedding_prompt_policy": {"stella": stale_policy},
    }
    qdrant = MetadataQdrant(metadata)

    _, issues = _maybe_backfill_legacy_prompt_policy(
        qdrant=qdrant,
        corpus="Docs",
        corpus_type="markdown",
        model_list=["stella"],
        metadata=metadata,
        output_json=True,
    )

    assert "differs" in issues[0]
    assert qdrant.upserted is None


def test_fetch_chunks_for_files_bulk_preserves_pdf_ocr_metadata():
    qdrant = MagicMock()
    qdrant.scroll.return_value = (
        [
            SimpleNamespace(
                id="point-1",
                payload={
                    "text": "Recovered OCR text",
                    "file_path": "/docs/report.pdf",
                    "filename": "report.pdf",
                    "file_extension": ".pdf",
                    "chunk_index": 3,
                    "document_type": "pdf",
                    "page_number": 2,
                    "ocr_confidence": 64.5,
                    "ocr_language": "eng",
                    "ocr_pages_processed": 4,
                    "ocr_pages_failed": 1,
                    "ocr_low_confidence_word_count": 7,
                    "ocr_merge_strategy": "append_missing_pages",
                    "ocr_triggered_by": "quality_gate",
                    "quality_manifest": {
                        "schema_version": 1,
                        "source_hash": "abc123",
                        "quality_warnings": ["low_text_pages"],
                    },
                },
            )
        ],
        None,
    )

    chunks_by_file, error = _fetch_chunks_for_files_bulk(
        qdrant,
        "test-corpus",
        {"/docs/report.pdf"},
        verbose=False,
        output_json=True,
        console=MagicMock(),
    )

    assert error is None
    doc = chunks_by_file["/docs/report.pdf"][0]
    assert doc["ocr_confidence"] == 64.5
    assert doc["ocr_language"] == "eng"
    assert doc["ocr_pages_processed"] == 4
    assert doc["ocr_pages_failed"] == 1
    assert doc["ocr_low_confidence_word_count"] == 7
    assert doc["ocr_merge_strategy"] == "append_missing_pages"
    assert doc["ocr_triggered_by"] == "quality_gate"
    assert doc["quality_manifest"]["source_hash"] == "abc123"


def test_backfill_meili_to_qdrant_builds_code_quality_manifest(tmp_path):
    source = tmp_path / "example.py"
    source.write_text("def hello():\n    return 'world'\n")
    qdrant = MagicMock()
    embedding_client = MagicMock()
    embedding_client.embed.return_value = [[0.1, 0.2]]
    progress = MagicMock()

    files_success, chunks_success, files_failed, skipped = _backfill_meili_to_qdrant(
        qdrant=qdrant,
        embedding_client=embedding_client,
        corpus="code-corpus",
        corpus_type="code",
        model_list=["test-model"],
        model_config={"chunk_size": 8000, "chunk_overlap": 20},
        file_paths=[str(source)],
        verbose=False,
        output_json=True,
        progress=progress,
        backfill_task=1,
        text_workers=1,
    )

    assert files_success == 1
    assert chunks_success >= 1
    assert files_failed == 0
    assert skipped == []
    points = qdrant.upsert.call_args.kwargs["points"]
    manifest = points[0].payload["quality_manifest"]
    assert manifest["extractor"] == "code"
    assert manifest["source_hash"] == points[0].payload["source_hash"]


def test_backfill_meili_to_qdrant_batches_embeddings_per_file(monkeypatch, tmp_path):
    source = tmp_path / "example.py"
    source.write_text("print('hello')\n")
    chunks = [
        {"text": "first", "metadata": {"method": "line"}},
        {"text": "second", "metadata": {"method": "line"}},
        {"text": "third", "metadata": {"method": "line"}},
    ]
    monkeypatch.setattr(sync_module, "chunk_code_file", lambda *args, **kwargs: chunks)

    qdrant = MagicMock()
    embedding_client = MagicMock()

    def embed(texts, model, max_internal_batch=None):
        base = 10 if model == "model-a" else 20
        return [[base + i] for i, _ in enumerate(texts)]

    embedding_client.embed.side_effect = embed

    files_success, chunks_success, files_failed, skipped = sync_module._backfill_meili_to_qdrant(
        qdrant=qdrant,
        embedding_client=embedding_client,
        corpus="code-corpus",
        corpus_type="code",
        model_list=["model-a", "model-b"],
        model_config={"chunk_size": 8000, "chunk_overlap": 20},
        file_paths=[str(source)],
        verbose=False,
        output_json=True,
        progress=MagicMock(),
        backfill_task=1,
        text_workers=1,
        max_embedding_batch=2,
    )

    assert files_success == 1
    assert chunks_success == 3
    assert files_failed == 0
    assert skipped == []
    assert embedding_client.embed.call_count == 2
    for call in embedding_client.embed.call_args_list:
        assert call.args[0] == ["first", "second", "third"]
        assert call.kwargs["max_internal_batch"] == 2

    points = qdrant.upsert.call_args.kwargs["points"]
    assert [point.vector["model-a"] for point in points] == [[10], [11], [12]]
    assert [point.vector["model-b"] for point in points] == [[20], [21], [22]]


def test_repair_meili_version_identifier_stamps_persisted_schema():
    index = MagicMock()
    index.get_documents.side_effect = [
        {
            "results": [
                {
                    "id": "doc-1",
                    "file_path": "/repo/example.py",
                    "chunk_index": 0,
                    "git_project_identifier": "proj#main",
                    "git_project_name": "proj",
                    "git_branch": "main",
                    "git_commit_hash": "abcdef123456",
                }
            ]
        },
        {"results": []},
    ]
    index.update_documents.return_value = SimpleNamespace(task_uid=123)
    meili = MagicMock()
    meili.get_index.return_value = index

    updated, failed = _repair_meili_metadata(
        qdrant=MagicMock(),
        meili=meili,
        corpus="code-corpus",
        output_json=True,
        console=MagicMock(),
    )

    assert updated == 1
    assert failed == 0
    update_doc = index.update_documents.call_args.args[0][0]
    assert update_doc["schema_version"] == 1
    assert update_doc["app_version"]
    assert update_doc["git_version_identifier"] == "proj#main@abcdef1"

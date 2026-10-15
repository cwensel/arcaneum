"""The `fail` policy ends an index run instead of failing one file at a time (RDR-025)."""

from unittest.mock import MagicMock, Mock

import pytest

from arcaneum.embeddings.client import EmbeddingClient
from arcaneum.embeddings.remote.errors import RemoteEmbeddingUnavailable
from arcaneum.embeddings.remote.routing import install_router
from arcaneum.indexing.markdown.chunker import SemanticMarkdownChunker
from arcaneum.indexing.markdown.pipeline import MarkdownIndexingPipeline
from arcaneum.indexing.uploader import PDFBatchUploader

UNAVAILABLE = RemoteEmbeddingUnavailable("No remote embedding endpoint can serve arctic-m")


class FailingRouter:
    def routes(self, model, prompt_type):
        return prompt_type == "document"

    def embed(self, texts, model, validate=None):
        raise UNAVAILABLE

    def close(self):
        return None


def test_embed_parallel_cpu_path_does_not_swallow_the_fail_policy(tmp_path):
    install_router(FailingRouter())
    try:
        client = EmbeddingClient(cache_dir=str(tmp_path))
        with pytest.raises(RemoteEmbeddingUnavailable):
            client.embed_parallel(["a", "b", "c"], "arctic-m", batch_size=1)
    finally:
        install_router(None)


def test_markdown_file_processing_does_not_swallow_the_fail_policy(tmp_path):
    source = tmp_path / "note.md"
    source.write_text("# Title\n\nSome body text that becomes a chunk.\n", encoding="utf-8")
    embeddings = Mock()
    embeddings.embed_parallel.side_effect = UNAVAILABLE
    pipeline = MarkdownIndexingPipeline(qdrant_client=Mock(), embedding_client=embeddings)
    pipeline.sync = Mock()
    pipeline.sync.find_file_by_content_hash.return_value = []

    with pytest.raises(RemoteEmbeddingUnavailable):
        pipeline._process_single_markdown(
            source, "docs", "arctic-m", {}, SemanticMarkdownChunker(), 1, False, 1, 1
        )


def test_pdf_file_processing_does_not_swallow_the_fail_policy(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    qdrant = MagicMock()
    qdrant.scroll.return_value = ([], None)
    qdrant.get_collection.return_value = MagicMock(points_count=0, payload_schema={})
    uploader = PDFBatchUploader(
        qdrant_client=qdrant, embedding_client=MagicMock(), file_workers=1, ocr_enabled=False
    )
    uploader.sync = MagicMock()
    uploader.sync.find_file_by_content_hash.return_value = []
    uploader.extractor.extract = MagicMock(return_value=("Body text of the paper.", {}))
    chunk = MagicMock(text="Body text of the paper.", metadata={})
    chunker = MagicMock()
    chunker.chunk.return_value = [chunk]
    uploader._shared_embedding_client.embed_parallel.side_effect = UNAVAILABLE

    with pytest.raises(RemoteEmbeddingUnavailable):
        uploader._process_single_pdf(
            pdf_path=pdf_path,
            collection_name="docs",
            model_name="arctic-m",
            chunker=chunker,
            point_id_start=1,
            verbose=False,
            pdf_idx=1,
            total_pdfs=1,
        )

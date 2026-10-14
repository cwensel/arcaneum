"""Pinned fastembed model revisions (RDR-025 Step 2)."""

import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from arcaneum.embeddings.client import EMBEDDING_MODELS, EmbeddingClient, model_revision

ARCTIC_REVISION = "fc74610d18462d218e312aa986ec5c8a75a98152"


def make_snapshot(cache_dir: Path, revision: str = ARCTIC_REVISION) -> Path:
    snapshot = (
        cache_dir / "models--Snowflake--snowflake-arctic-embed-m" / "snapshots" / revision / "onnx"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "model.onnx").write_bytes(b"onnx")
    return snapshot.parent


def test_arctic_m_pins_the_verified_revision():
    assert EMBEDDING_MODELS["arctic-m"]["revision"] == ARCTIC_REVISION


def test_model_revision_reports_pinned_models_only():
    assert model_revision("arctic-m") == ARCTIC_REVISION
    assert re.fullmatch(r"[0-9a-f]{40}", model_revision("qwen3-embed"))
    assert model_revision("bge-small") is None


def test_pinned_fastembed_model_loads_the_snapshot_as_specific_model_path(tmp_path):
    snapshot = make_snapshot(tmp_path)
    client = EmbeddingClient(cache_dir=str(tmp_path))

    with (
        patch("arcaneum.embeddings.client._snapshot_download", return_value=str(snapshot)) as dl,
        patch("arcaneum.embeddings.client.TextEmbedding") as text_embedding,
    ):
        client.get_model("arctic-m")

    dl.assert_called_once()
    kwargs = dl.call_args.kwargs
    assert kwargs["repo_id"] == "Snowflake/snowflake-arctic-embed-m"
    assert kwargs["revision"] == ARCTIC_REVISION
    assert kwargs["cache_dir"] == str(tmp_path)
    assert kwargs["local_files_only"] is True
    assert "onnx/model.onnx" in kwargs["allow_patterns"]
    assert "tokenizer.json" in kwargs["allow_patterns"]
    text_embedding.assert_called_once()
    assert text_embedding.call_args.kwargs["model_name"] == "snowflake/snowflake-arctic-embed-m"
    assert text_embedding.call_args.kwargs["specific_model_path"] == str(snapshot)


def test_pinned_fastembed_model_downloads_the_revision_when_not_cached(tmp_path):
    snapshot = tmp_path / "downloaded"
    (snapshot / "onnx").mkdir(parents=True)
    (snapshot / "onnx" / "model.onnx").write_bytes(b"onnx")
    client = EmbeddingClient(cache_dir=str(tmp_path))
    calls = []

    def download(**kwargs):
        calls.append(kwargs["local_files_only"])
        if kwargs["local_files_only"]:
            raise FileNotFoundError("not cached")
        return str(snapshot)

    with (
        patch("arcaneum.embeddings.client._snapshot_download", side_effect=download),
        patch("arcaneum.embeddings.client.TextEmbedding") as text_embedding,
    ):
        client.get_model("arctic-m")

    assert calls == [True, False]
    assert text_embedding.call_args.kwargs["specific_model_path"] == str(snapshot)


def test_pinned_snapshot_missing_its_model_file_is_redownloaded(tmp_path):
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    complete = make_snapshot(tmp_path)
    client = EmbeddingClient(cache_dir=str(tmp_path))
    results = iter([str(incomplete), str(complete)])

    with (
        patch(
            "arcaneum.embeddings.client._snapshot_download",
            side_effect=lambda **kwargs: next(results),
        ) as dl,
        patch("arcaneum.embeddings.client.TextEmbedding") as text_embedding,
    ):
        client.get_model("arctic-m")

    assert [call.kwargs["local_files_only"] for call in dl.call_args_list] == [True, False]
    assert text_embedding.call_args.kwargs["specific_model_path"] == str(complete)


def test_unpinned_fastembed_models_keep_the_existing_loader(tmp_path):
    client = EmbeddingClient(cache_dir=str(tmp_path))

    with (
        patch("arcaneum.embeddings.client._snapshot_download") as dl,
        patch("arcaneum.embeddings.client.TextEmbedding") as text_embedding,
    ):
        client.get_model("bge-small")

    dl.assert_not_called()
    assert "specific_model_path" not in text_embedding.call_args.kwargs


def test_pinned_model_is_cached_only_when_its_revision_is_present(tmp_path):
    client = EmbeddingClient(cache_dir=str(tmp_path))
    assert not client.is_model_cached("arctic-m")

    make_snapshot(tmp_path, revision="0" * 40)
    assert not client.is_model_cached("arctic-m")

    make_snapshot(tmp_path)
    assert client.is_model_cached("arctic-m")


@pytest.mark.slow
def test_pinned_arctic_m_matches_the_unpinned_loader_vectors(tmp_path):
    """Real model: the pinned snapshot embeds exactly as fastembed's own download."""
    pytest.importorskip("fastembed")
    from arcaneum.paths import get_models_dir

    cache_dir = str(get_models_dir())
    client = EmbeddingClient(cache_dir=cache_dir)
    if not client.is_model_cached("arctic-m"):
        pytest.skip("arctic-m pinned snapshot is not cached locally")

    texts = ["Pinned revisions keep vectors stable.", "A **bold** markdown chunk."]
    pinned = client.embed(texts, "arctic-m")

    from fastembed import TextEmbedding

    unpinned = TextEmbedding(
        model_name=EMBEDDING_MODELS["arctic-m"]["name"],
        cache_dir=cache_dir,
        local_files_only=True,
    )
    reference = np.asarray(list(unpinned.embed(texts)), dtype=np.float32)

    np.testing.assert_allclose(pinned, reference, atol=1e-6)

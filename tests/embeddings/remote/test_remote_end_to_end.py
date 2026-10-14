"""Routed EmbeddingClient -> pool -> server with a real model (RDR-025)."""

import numpy as np
import pytest

from arcaneum.embeddings.client import EmbeddingClient
from arcaneum.embeddings.remote.config import EndpointConfig, RemoteEmbeddingConfig
from arcaneum.embeddings.remote.routing import RemoteRouter, install_router
from arcaneum.embeddings.remote.server import EmbeddingServer, ServerSettings
from arcaneum.paths import get_models_dir


@pytest.mark.slow
def test_remote_routed_documents_match_local_cpu_embedding():
    cache_dir = str(get_models_dir())
    local = EmbeddingClient(cache_dir=cache_dir)
    if not local.is_model_cached("arctic-m"):
        pytest.skip("arctic-m pinned snapshot is not cached locally")
    texts = [f"Document {i} about remote embedding workers and vector parity." for i in range(9)]
    texts.append("A chunk ending in **markdown emphasis**")
    expected = np.asarray(local.embed(texts, "arctic-m"), dtype=np.float32)

    server = EmbeddingServer(
        ServerSettings(port=0, bind="127.0.0.1", cache_dir=cache_dir, batch_sizes={"arctic-m": 4})
    )
    server.start()
    config = RemoteEmbeddingConfig(
        endpoints=[EndpointConfig(host="127.0.0.1", port=server.address[1])],
        unavailable="fail",
    )
    install_router(RemoteRouter(config))
    try:
        client = EmbeddingClient(cache_dir=cache_dir)
        single = np.asarray(client.embed(texts, "arctic-m"), dtype=np.float32)
        parallel = np.asarray(client.embed_parallel(texts, "arctic-m"), dtype=np.float32)
        query = client.embed(["what is a worker?"], "arctic-m", prompt_type="query")
    finally:
        install_router(None)
        server.stop()

    np.testing.assert_allclose(single, expected, atol=1e-5)
    np.testing.assert_allclose(parallel, expected, atol=1e-5)
    # Documents never loaded a local model; the query did, locally.
    assert np.asarray(query).shape == (1, 768)
    assert client.local_models == ["arctic-m"]

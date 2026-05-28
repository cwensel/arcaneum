"""Unit tests for the process-wide embedding model cache."""

from unittest.mock import MagicMock

from arcaneum.embeddings import model_cache


def test_cached_model_reuses_same_cache_directory(monkeypatch, tmp_path):
    clients = []

    def fake_client(**kwargs):
        client = MagicMock()
        client.cache_kwargs = kwargs
        clients.append(client)
        return client

    monkeypatch.setattr(model_cache, "_model_cache", {})
    monkeypatch.setattr(model_cache, "EmbeddingClient", fake_client)

    cache_dir = tmp_path / "models"
    first = model_cache.get_cached_model("stella", cache_dir=str(cache_dir), use_gpu=False)
    second = model_cache.get_cached_model("stella", cache_dir=str(cache_dir), use_gpu=False)

    assert second is first
    assert len(clients) == 1
    first.get_model.assert_called_once_with("stella")


def test_cached_model_isolates_different_cache_directories(monkeypatch, tmp_path):
    clients = []

    def fake_client(**kwargs):
        client = MagicMock()
        client.cache_kwargs = kwargs
        clients.append(client)
        return client

    monkeypatch.setattr(model_cache, "_model_cache", {})
    monkeypatch.setattr(model_cache, "EmbeddingClient", fake_client)

    first = model_cache.get_cached_model(
        "stella",
        cache_dir=str(tmp_path / "models-a"),
        use_gpu=False,
    )
    second = model_cache.get_cached_model(
        "stella",
        cache_dir=str(tmp_path / "models-b"),
        use_gpu=False,
    )

    assert second is not first
    assert [client.cache_kwargs["cache_dir"] for client in clients] == [
        str(tmp_path / "models-a"),
        str(tmp_path / "models-b"),
    ]
    assert [client.get_model.call_args.args for client in clients] == [("stella",), ("stella",)]


def test_cached_model_isolates_gpu_mode(monkeypatch, tmp_path):
    clients = []

    def fake_client(**kwargs):
        client = MagicMock()
        client.cache_kwargs = kwargs
        clients.append(client)
        return client

    monkeypatch.setattr(model_cache, "_model_cache", {})
    monkeypatch.setattr(model_cache, "EmbeddingClient", fake_client)

    cpu_client = model_cache.get_cached_model("stella", cache_dir=str(tmp_path), use_gpu=False)
    gpu_client = model_cache.get_cached_model("stella", cache_dir=str(tmp_path), use_gpu=True)

    assert gpu_client is not cpu_client
    assert [client.cache_kwargs["use_gpu"] for client in clients] == [False, True]

import threading
import time


class CountingExecutor:
    max_outstanding = 0
    outstanding = 0

    def __init__(self, *args, **kwargs):
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(*args, **kwargs)

    def __enter__(self):
        self._executor.__enter__()
        return self

    def __exit__(self, *args):
        return self._executor.__exit__(*args)

    def submit(self, *args, **kwargs):
        type(self).outstanding += 1
        type(self).max_outstanding = max(type(self).max_outstanding, type(self).outstanding)
        future = self._executor.submit(*args, **kwargs)
        future.add_done_callback(lambda _: self._mark_done())
        return future

    @classmethod
    def _mark_done(cls):
        cls.outstanding -= 1

    def shutdown(self, *args, **kwargs):
        return self._executor.shutdown(*args, **kwargs)


def test_cpu_streaming_keeps_submitted_batches_bounded(monkeypatch):
    from arcaneum.embeddings.client import EmbeddingClient

    CountingExecutor.max_outstanding = 0
    CountingExecutor.outstanding = 0
    monkeypatch.setattr("arcaneum.embeddings.client.ThreadPoolExecutor", CountingExecutor)

    client = EmbeddingClient(cache_dir="/tmp/models", use_gpu=False, cpu_workers=2)
    monkeypatch.setattr(client, "get_model", lambda model_name: object())
    monkeypatch.setattr(client, "_get_optimal_batch_size", lambda model_name: 1)
    release = threading.Event()
    result = []

    def embed(batch, model_name):
        release.wait(timeout=1)
        return [[batch[0]]]

    monkeypatch.setattr(client, "embed", embed)

    worker = threading.Thread(
        target=lambda: result.append(
            client.embed_parallel(
                [f"text-{i}" for i in range(8)],
                "jina-code",
                batch_size=1,
                accumulate=False,
                on_batch_complete=lambda *args: None,
            )
        )
    )
    worker.start()

    deadline = time.time() + 1
    while CountingExecutor.max_outstanding < 2 and time.time() < deadline:
        time.sleep(0.001)

    time.sleep(0.05)
    assert CountingExecutor.max_outstanding <= 2

    release.set()
    worker.join(timeout=2)
    assert result == [None]
    assert CountingExecutor.max_outstanding <= 2


def test_cpu_streaming_callbacks_preserve_batch_order(monkeypatch):
    from arcaneum.embeddings.client import EmbeddingClient

    client = EmbeddingClient(cache_dir="/tmp/models", use_gpu=False, cpu_workers=2)
    monkeypatch.setattr(client, "get_model", lambda model_name: object())

    def embed(batch, model_name):
        if batch == ["text-0"]:
            time.sleep(0.05)
        return [[batch[0]]]

    seen_batches = []
    seen_progress = []
    monkeypatch.setattr(client, "embed", embed)

    result = client.embed_parallel(
        [f"text-{i}" for i in range(4)],
        "jina-code",
        batch_size=1,
        accumulate=False,
        on_batch_complete=lambda batch_idx, start_idx, embeddings: seen_batches.append(
            (batch_idx, start_idx, embeddings[0][0])
        ),
        progress_callback=lambda completed, total: seen_progress.append((completed, total)),
    )

    assert result is None
    assert seen_batches == [
        (0, 0, "text-0"),
        (1, 1, "text-1"),
        (2, 2, "text-2"),
        (3, 3, "text-3"),
    ]
    assert seen_progress == [(1, 4), (2, 4), (3, 4), (4, 4)]

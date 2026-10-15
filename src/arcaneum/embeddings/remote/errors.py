"""Remote embedding exceptions, importable without the pool or server (RDR-025)."""


class RemoteEmbeddingUnavailable(RuntimeError):
    """No endpoint can serve a remote-routed model and the policy is ``fail``.

    Sync re-raises this past its per-file error handling so the run ends.
    """


class RemoteEmbeddingBatchError(RuntimeError):
    """A batch failed on retry for reasons of its own (an encode error or invalid
    vectors), not because an endpoint is down; the caller's file fails, the run goes on."""


class RemoteEmbeddingFallback(Exception):
    """No endpoint can serve the model and the policy is ``local``: embed locally."""

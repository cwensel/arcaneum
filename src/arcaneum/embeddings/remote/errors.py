"""Remote embedding exceptions, importable without the pool or server (RDR-025)."""


class RemoteEmbeddingUnavailable(RuntimeError):
    """No endpoint can serve a remote-routed model and the policy is ``fail``.

    Sync re-raises this past its per-file error handling so the run ends.
    """


class RemoteEmbeddingFallback(Exception):
    """No endpoint can serve the model and the policy is ``local``: embed locally."""

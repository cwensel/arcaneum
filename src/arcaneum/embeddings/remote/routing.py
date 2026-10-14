"""Per-process routing of document embedding to the remote pool (RDR-025).

Commands that embed documents (``arc corpus sync``/``repair``/``parity`` and the
``arc index`` commands) install a router before embedding; ``EmbeddingClient``
consults it before constructing any local model. Nothing else installs one, so
query embedding, the server's own client, and library use stay local.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable, Optional

from arcaneum.embeddings.remote.config import RemoteEmbeddingConfig, load_remote_embedding_config

if TYPE_CHECKING:
    import numpy as np

    from arcaneum.embeddings.remote.pool import RemoteEmbeddingPool

# default: route `remote` models; remote: also `on-request` models (--remote-embed);
# local: route nothing (--local-embed).
MODES = ("default", "remote", "local")


class RemoteRouter:
    def __init__(self, config: RemoteEmbeddingConfig, *, mode: str = "default") -> None:
        if mode not in MODES:
            raise ValueError(f"Unknown remote embedding mode {mode!r}; expected one of {MODES}")
        self.config = config
        self.mode = mode
        self._pool: Optional["RemoteEmbeddingPool"] = None
        self._pool_lock = threading.Lock()

    def routes(self, model: str, prompt_type: str) -> bool:
        """Whether this embed call goes to the pool. Queries never do."""
        if prompt_type != "document" or self.mode == "local" or not self.config.endpoints:
            return False
        routing = self.config.model_routing(model)
        return routing == "remote" or (routing == "on-request" and self.mode == "remote")

    @property
    def pool(self) -> "RemoteEmbeddingPool":
        with self._pool_lock:
            if self._pool is None:
                from arcaneum.embeddings.remote.pool import RemoteEmbeddingPool

                self._pool = RemoteEmbeddingPool(self.config)
            return self._pool

    def embed(
        self,
        texts: list[str],
        model: str,
        validate: Optional[Callable[[Any, int], bool]] = None,
    ) -> "np.ndarray":
        return self.pool.embed(texts, model, validate=validate)

    def close(self) -> None:
        with self._pool_lock:
            if self._pool is not None:
                self._pool.close()
                self._pool = None


_active: Optional[RemoteRouter] = None
_active_lock = threading.Lock()


def active_router() -> Optional[RemoteRouter]:
    return _active


def install_router(router: Optional[RemoteRouter]) -> Optional[RemoteRouter]:
    """Install (or with None, remove) the process router; returns the previous one."""
    global _active
    with _active_lock:
        previous, _active = _active, router
    if previous is not None and previous is not router:
        previous.close()
    return previous


def configure_remote_embedding(mode: str = "default") -> Optional[RemoteRouter]:
    """Load configuration and install a router for this command's document embedding.

    Raises RemoteEmbeddingConfigError for invalid configuration. Without configured
    endpoints nothing is installed and embedding stays local.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown remote embedding mode {mode!r}; expected one of {MODES}")
    config = load_remote_embedding_config()
    if not config.endpoints:
        install_router(None)
        return None
    router = RemoteRouter(config, mode=mode)
    install_router(router)
    return router

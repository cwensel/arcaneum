"""Per-process routing of document embedding to the remote pool (RDR-025).

Commands that embed documents (``arc corpus sync``/``repair``/``parity`` and the
``arc index`` commands) install a router before embedding; ``EmbeddingClient``
consults it before constructing any local model. Nothing else installs one, so
query embedding, the server's own client, and library use stay local.

Installing a router that routes models also starts the run-start check of its
workers (``preflight``): before extraction, in the background, and with
auto-deploy of workers running other code unless ``auto_deploy`` is off.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import TYPE_CHECKING, Any, Callable, Optional

from arcaneum.embeddings.remote.config import RemoteEmbeddingConfig, load_remote_embedding_config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np

    from arcaneum.embeddings.remote.pool import RemoteEmbeddingPool

# default: route `remote` models; remote: also `on-request` models (--remote-embed);
# local: route nothing (--local-embed).
MODES = ("default", "remote", "local")


class RemoteRouter:
    def __init__(
        self,
        config: RemoteEmbeddingConfig,
        *,
        mode: str = "default",
        auto_deploy: Optional[bool] = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"Unknown remote embedding mode {mode!r}; expected one of {MODES}")
        self.config = config
        self.mode = mode
        # None follows remote_embedding.auto_deploy; False is --no-auto-deploy.
        self.auto_deploy = config.auto_deploy if auto_deploy is None else auto_deploy
        # Whether a run-start check of this router's workers has started; a
        # preview (dry run) installs a router without one.
        self.checked = False
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
                from arcaneum.embeddings.remote.deploy import deploy_lock_held
                from arcaneum.embeddings.remote.preflight import DEPLOYS

                self._pool = RemoteEmbeddingPool(
                    self.config, deploys=DEPLOYS, deploy_lock_held=deploy_lock_held
                )
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


def start_preflight(router: RemoteRouter, models: list[str]) -> None:
    """Check the router's workers in the background (tests replace this)."""
    from arcaneum.embeddings.remote import preflight

    preflight.start_preflight(router, models)


def configure_remote_embedding(
    mode: str = "default", *, auto_deploy: Optional[bool] = None, check_workers: bool = True
) -> Optional[RemoteRouter]:
    """Load configuration and install a router for this command's document embedding.

    Raises RemoteEmbeddingConfigError for invalid configuration. Without configured
    endpoints nothing is installed and embedding stays local. ``auto_deploy`` None
    follows ``remote_embedding.auto_deploy``; False is the run's --no-auto-deploy.
    ``check_workers`` False (a dry run) skips the run-start check, which wakes
    workers and may deploy to them.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown remote embedding mode {mode!r}; expected one of {MODES}")
    config = load_remote_embedding_config()
    if not config.endpoints:
        install_router(None)
        return None
    current = active_router()
    router = RemoteRouter(config, mode=mode, auto_deploy=auto_deploy)
    if (
        current is not None
        and current.mode == mode
        and current.config == config
        and current.auto_deploy == router.auto_deploy
    ):
        # Repair runs sync then parity, and a spool drain syncs once per batch;
        # keep the pool's worker connections across them, and check them once.
        router = current
    else:
        install_router(router)
    models = [model for model in config.routed_models() if router.routes(model, "document")]
    if models and router is not current:
        message = (
            f"Remote embedding: {', '.join(models)} on "
            f"{', '.join(endpoint.name for endpoint in config.endpoints)}"
        )
        logger.info(message)
        print(f"   {message}", file=sys.stderr, flush=True)
    if models and check_workers and not router.checked:
        router.checked = True
        start_preflight(router, models)
    return router

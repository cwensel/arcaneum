"""Remote embedding worker commands: `arc models serve` (RDR-025)."""

from __future__ import annotations

import logging
import os
import signal
import sys
from typing import Optional

from arcaneum.embeddings.client import EMBEDDING_MODELS, model_revision
from arcaneum.embeddings.remote.server import EmbeddingServer, ServerSettings


def parse_served_models(value: Optional[str]) -> Optional[tuple[str, ...]]:
    """Parse a comma-separated model list; None or empty serves every pinned model."""
    if not value or not value.strip():
        return None
    models = tuple(part.strip() for part in value.split(",") if part.strip())
    for model in models:
        if model not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown model '{model}'")
        if model_revision(model) is None:
            raise ValueError(
                f"Model '{model}' cannot be served remotely: it has no pinned revision"
            )
    return models


def parse_batch_sizes(value: Optional[str]) -> dict[str, int]:
    """Parse ``MODEL=N[,MODEL=N]`` encode batch sizes."""
    sizes: dict[str, int] = {}
    if not value:
        return sizes
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        model, separator, raw = part.partition("=")
        if not separator or not model.strip() or not raw.strip():
            raise ValueError(f"Batch sizes must be MODEL=N, not {part!r}")
        size = int(raw)
        if size < 1:
            raise ValueError(f"Batch size for {model.strip()} must be positive")
        sizes[model.strip()] = size
    return sizes


def serve_command(
    *,
    port: int,
    bind: str,
    served_models: Optional[str],
    batch_sizes: Optional[str],
    idle_grace: float,
    turn_length: float,
    cache_dir: Optional[str],
) -> None:
    """Run the embedding server until SIGTERM or SIGINT."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    authkey = os.environ.get("ARC_EMBED_AUTHKEY") or None
    settings = ServerSettings(
        port=port,
        bind=bind,
        authkey=authkey.encode() if authkey else None,
        models=parse_served_models(served_models),
        batch_sizes=parse_batch_sizes(batch_sizes),
        idle_grace=idle_grace,
        turn_length=turn_length,
        cache_dir=cache_dir,
    )
    server = EmbeddingServer(settings)

    def stop(signum, frame):
        logging.getLogger(__name__).info("Received signal %d; stopping", signum)
        server.stop()

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        server.serve_forever()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

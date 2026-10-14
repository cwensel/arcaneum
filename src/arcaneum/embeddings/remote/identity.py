"""Handshake identity for remote embedding connections (RDR-025).

Both ends compute these fields from their own code; any difference excludes the
endpoint for that model. Every field except the model key comes from code, so in
practice a mismatch is a fingerprint mismatch, and the other fields say what differs.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import metadata
from typing import Any, Optional

import arcaneum
from arcaneum.embeddings.client import (
    EMBEDDING_MODELS,
    get_embedding_prompt_policy,
    model_revision,
)
from arcaneum.embeddings.remote.fingerprint import code_fingerprint
from arcaneum.embeddings.worker_protocol import PROTOCOL_VERSION

IDENTITY_FIELDS = (
    "protocol_version",
    "arcaneum_version",
    "code_fingerprint",
    "model",
    "revision",
    "prompt_policy_version",
    "dimensions",
)

DEPENDENCIES = ("torch", "onnxruntime", "fastembed", "sentence-transformers", "transformers")


def local_identity(model: Optional[str]) -> dict[str, Any]:
    """Handshake fields as this process's code defines them.

    Without a model the identity describes a status-only connection.
    """
    identity: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "arcaneum_version": arcaneum.__version__,
        "code_fingerprint": code_fingerprint(),
    }
    if model is not None:
        identity.update(
            model=model,
            revision=model_revision(model),
            prompt_policy_version=get_embedding_prompt_policy(model)["version"],
            dimensions=EMBEDDING_MODELS[model]["dimensions"],
        )
    return identity


def identity_mismatches(local: dict[str, Any], remote: dict[str, Any]) -> list[str]:
    """Names of the handshake fields that differ, in handshake order."""
    return [
        field
        for field in IDENTITY_FIELDS
        if field in local and (field not in remote or remote[field] != local[field])
    ]


@lru_cache(maxsize=1)
def _dependency_versions() -> tuple[tuple[str, Optional[str]], ...]:
    versions = []
    for name in DEPENDENCIES:
        try:
            versions.append((name, metadata.version(name)))
        except metadata.PackageNotFoundError:
            versions.append((name, None))
    return tuple(versions)


def dependency_versions() -> dict[str, Optional[str]]:
    """Installed versions of the inference packages that shape vectors."""
    return dict(_dependency_versions())

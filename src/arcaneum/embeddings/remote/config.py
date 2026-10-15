"""Remote embedding configuration: endpoints, routing, unavailable policy (RDR-025).

Lives in the ``remote_embedding`` section of ``~/.config/arcaneum/config.yaml``::

    remote_embedding:
      authkey: <shared secret>        # optional; ARC_EMBED_AUTHKEY overrides
      unavailable: wait               # wait | local | fail
      routing: {arctic-m: remote, qwen3-embed: on-request}
      endpoints:
        - host: dev-0
          port: 7300
          bind: tailscale             # all | tailscale | <IP address>
          ssh: dev-0                  # deploy target; default: host
          models: [arctic-m]          # default: every remote/on-request model
          batch_size: {qwen3-embed: 8}

``ARC_EMBED_ENDPOINTS=host[:port],...`` replaces the configured endpoint list for
one process (endpoints given that way use default settings; empty disables the pool).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from arcaneum.embeddings.client import EMBEDDING_MODELS, model_revision
from arcaneum.embeddings.remote import DEFAULT_PORT

Routing = Literal["remote", "on-request", "local"]
Unavailable = Literal["wait", "local", "fail"]

DEFAULT_ROUTING: dict[str, Routing] = {"arctic-m": "remote", "qwen3-embed": "on-request"}
ENDPOINTS_ENV = "ARC_EMBED_ENDPOINTS"
AUTHKEY_ENV = "ARC_EMBED_AUTHKEY"


class RemoteEmbeddingConfigError(ValueError):
    """The remote embedding configuration is invalid."""


class EndpointConfig(BaseModel):
    """How to reach one worker and how to run it."""

    host: str
    port: int = DEFAULT_PORT
    ssh: Optional[str] = None
    bind: str = "all"
    models: Optional[list[str]] = None
    batch_size: dict[str, int] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def ssh_target(self) -> str:
        return self.ssh or self.host

    @field_validator("bind")
    @classmethod
    def _check_bind(cls, value: str) -> str:
        import ipaddress

        if value not in ("all", "tailscale"):
            try:
                ipaddress.ip_address(value)
            except ValueError as exc:
                raise ValueError("bind must be 'all', 'tailscale' or an IP address") from exc
        return value


class DeadlineConfig(BaseModel):
    """Per-state deadlines (seconds); none is short enough to trip on legitimate work."""

    loading: float = 1800.0
    # Ten times the spike's dev-0 time per chunk.
    encode_per_chunk: dict[str, float] = Field(
        default_factory=lambda: {"arctic-m": 10.0, "qwen3-embed": 160.0}
    )
    encode_per_chunk_default: float = 160.0
    heartbeat_interval: float = 5.0
    heartbeat_timeout: float = 30.0
    connect_timeout: float = 10.0

    def encode_allowance(self, model: str, chunks: int) -> float:
        per_chunk = self.encode_per_chunk.get(model, self.encode_per_chunk_default)
        return per_chunk * max(1, chunks)


class RemoteEmbeddingConfig(BaseModel):
    endpoints: list[EndpointConfig] = Field(default_factory=list)
    routing: dict[str, Routing] = Field(default_factory=dict)
    unavailable: Unavailable = "wait"
    authkey: Optional[str] = None
    idle_grace: float = 60.0
    turn_length: float = 600.0
    probe_interval: float = 10.0
    deadlines: DeadlineConfig = Field(default_factory=DeadlineConfig)

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self._check_routing()

    def _check_routing(self) -> None:
        for model, mode in self.routing.items():
            if model not in EMBEDDING_MODELS:
                raise RemoteEmbeddingConfigError(f"Unknown model '{model}' in remote routing")
            if mode != "local" and model_revision(model) is None:
                raise RemoteEmbeddingConfigError(
                    f"Model '{model}' cannot be routed '{mode}': it has no pinned revision "
                    "in the model registry, so a worker could load different weights"
                )

    def model_routing(self, model: str) -> Routing:
        return self.routing.get(model) or DEFAULT_ROUTING.get(model, "local")

    def routed_models(self) -> list[str]:
        return [model for model in EMBEDDING_MODELS if self.model_routing(model) != "local"]

    def endpoint_models(self, endpoint: EndpointConfig) -> list[str]:
        return list(endpoint.models) if endpoint.models else self.routed_models()

    @property
    def authkey_bytes(self) -> Optional[bytes]:
        return self.authkey.encode() if self.authkey else None


def parse_endpoint_list(value: str) -> list[EndpointConfig]:
    """Parse ``host[:port],...`` from ARC_EMBED_ENDPOINTS."""
    endpoints = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        host, separator, raw_port = part.rpartition(":")
        if not separator:
            host, raw_port = part, str(DEFAULT_PORT)
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise RemoteEmbeddingConfigError(
                f"{ENDPOINTS_ENV} entry {part!r} has an invalid port"
            ) from exc
        endpoints.append(EndpointConfig(host=host, port=port))
    return endpoints


def _default_config_path() -> Path:
    from arcaneum.cli.utils import resolve_config_path

    return resolve_config_path()


def load_remote_embedding_config(config_path: Optional[Path] = None) -> RemoteEmbeddingConfig:
    """Load the ``remote_embedding`` section, applying environment overrides."""
    path = Path(config_path) if config_path is not None else _default_config_path()
    data: dict = {}
    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise RemoteEmbeddingConfigError(f"Invalid YAML in {path}") from exc
        if not isinstance(loaded, dict):
            raise RemoteEmbeddingConfigError(f"{path} must be a YAML mapping")
        data = dict(loaded.get("remote_embedding") or {})
    try:
        config = RemoteEmbeddingConfig(**data)
    except ValidationError as exc:
        raise RemoteEmbeddingConfigError(f"Invalid remote_embedding configuration: {exc}") from exc
    if ENDPOINTS_ENV in os.environ:
        config.endpoints = parse_endpoint_list(os.environ[ENDPOINTS_ENV])
    if os.environ.get(AUTHKEY_ENV):
        config.authkey = os.environ[AUTHKEY_ENV]
    return config

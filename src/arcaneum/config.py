"""Configuration management for Arcaneum (RDR-003)."""

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from arcaneum.embeddings.client import EMBEDDING_MODELS


class ModelConfig(BaseModel):
    """Configuration for a single embedding model."""

    name: str
    dimensions: int
    chunk_size: int
    chunk_overlap: int
    distance: Literal["cosine", "euclid", "dot"] = "cosine"
    late_chunking: bool = False
    char_to_token_ratio: float = 3.3
    min_chunk_chars: int = Field(default=200, ge=0)


class QdrantConfig(BaseModel):
    """Qdrant server configuration."""

    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    timeout: int = 120  # General timeout for indexing operations
    search_timeout: int = 60  # Timeout for search operations (can be longer)


class CacheConfig(BaseModel):
    """Model cache configuration."""

    models_dir: Path = Path("./models_cache")
    max_size_gb: int = 10


class BackupConfig(BaseModel):
    """Full backup configuration."""

    path: Optional[Path] = None


class CollectionTemplate(BaseModel):
    """Template for collection creation."""

    models: List[str]
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    on_disk_payload: bool = True
    indexes: List[str] = Field(default_factory=list)


class ArcaneumConfig(BaseModel):
    """Root configuration."""

    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    backup: BackupConfig = Field(default_factory=BackupConfig)
    models: Dict[str, ModelConfig]
    collections: Dict[str, CollectionTemplate] = Field(default_factory=dict)


def load_config(config_path: Path) -> ArcaneumConfig:
    """Load and validate configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated ArcaneumConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return ArcaneumConfig(**data)


def load_backup_config(config_path: Path) -> BackupConfig:
    """Load only the backup section from a configuration file.

    Container backups do not depend on embedding model configuration, so this
    loader validates the relevant section without requiring a complete
    :class:`ArcaneumConfig`.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in configuration file: {config_path}") from exc

    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")

    return BackupConfig.model_validate(data.get("backup", {}))


# The text backends a corpus can be created on (indexing.collection_metadata.TEXT_BACKENDS).
_CREATABLE_TEXT_BACKENDS = ("meili", "qdrant")


@dataclass(frozen=True)
class MeiliSettings:
    """Whether MeiliSearch is in use, and the text backend new corpora get (arcaneum#183).

    ``meilisearch.enabled`` and ``corpus.default_text_backend`` in config.yaml
    win. Absent, they are inferred from whether the user has ever had a
    MeiliSearch key: one stored (``arc container start`` writes it) or set in
    ``MEILISEARCH_API_KEY`` reads as an existing user, MeiliSearch on and new
    corpora on ``meili``; none reads as a new user, MeiliSearch off and new
    corpora on ``qdrant``. An explicit ``enabled`` with no backend gives the
    backend that goes with it.
    """

    enabled: bool
    default_text_backend: str
    config_path: Path
    enabled_explicit: bool
    backend_explicit: bool
    # The top-level sections config.yaml already has (``meilisearch``, ``corpus``).
    sections: frozenset = frozenset()

    @property
    def inferred(self) -> bool:
        """True when either value came from inference, not config.yaml."""
        return not (self.enabled_explicit and self.backend_explicit)

    def disabled_reason(self) -> str:
        """Why MeiliSearch is off, naming the setting that turns it on."""
        return (
            f"MeiliSearch is disabled (meilisearch.enabled: false in {self.config_path}); "
            "set it to true to use MeiliSearch"
        )


def _meilisearch_key_stored() -> bool:
    """Whether this user has a MeiliSearch key, in the environment or stored (writes nothing)."""
    if os.environ.get("MEILISEARCH_API_KEY"):
        return True
    config_home = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return (Path(config_home) / "arcaneum" / "meilisearch.key").exists()


def _read_yaml_mapping(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in configuration file: {config_path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return data


def _section_leaf(data: dict, section: str, leaf: str):
    value = data.get(section)
    return value.get(leaf) if isinstance(value, dict) else None


def resolve_meili_settings(config_path: Optional[Path] = None) -> MeiliSettings:
    """Read ``meilisearch.enabled`` and ``corpus.default_text_backend``, inferring what is absent.

    Reads the config in place (``config_path``, else the default path, a
    legacy one unmigrated) and writes nothing: no key, no config.

    Raises:
        ValueError: config.yaml is not a mapping, or holds an invalid value
    """
    if config_path is None:
        from .cli.utils import resolve_config_path

        config_path = resolve_config_path(migrate=False)
    data = _read_yaml_mapping(config_path)

    enabled = _section_leaf(data, "meilisearch", "enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError(
            f"meilisearch.enabled must be true or false, not {enabled!r} ({config_path})"
        )
    backend = _section_leaf(data, "corpus", "default_text_backend")
    if backend is not None and backend not in _CREATABLE_TEXT_BACKENDS:
        raise ValueError(
            f"corpus.default_text_backend must be one of {', '.join(_CREATABLE_TEXT_BACKENDS)}, "
            f"not {backend!r} ({config_path})"
        )

    resolved_enabled = enabled if enabled is not None else _meilisearch_key_stored()
    resolved_backend = backend or ("meili" if resolved_enabled else "qdrant")
    return MeiliSettings(
        enabled=resolved_enabled,
        default_text_backend=resolved_backend,
        config_path=config_path,
        enabled_explicit=enabled is not None,
        backend_explicit=backend is not None,
        sections=frozenset(key for key in ("meilisearch", "corpus") if key in data),
    )


def persist_meili_settings(settings: MeiliSettings) -> List[str]:
    """Record inferred values in config.yaml, so the choice never flips later.

    Appends a ``meilisearch:`` or ``corpus:`` block for each value that was
    inferred and whose top-level section is absent, keeping the file's other
    content and comments; the write is atomic (a temporary file, then
    ``os.replace``) and keeps the file's mode. A section that exists without
    the value is not rewritten: the returned notes name the line to add.

    Returns:
        One note per value recorded or left to the user.
    """
    blocks: List[str] = []
    notes: List[str] = []
    enabled_line = f"enabled: {'true' if settings.enabled else 'false'}"
    backend_line = f"default_text_backend: {settings.default_text_backend}"
    for explicit, section, line in (
        (settings.enabled_explicit, "meilisearch", enabled_line),
        (settings.backend_explicit, "corpus", backend_line),
    ):
        if explicit:
            continue
        if section in settings.sections:
            notes.append(
                f"Add '{line}' under '{section}:' in {settings.config_path} "
                f"to keep {section}.{line}"
            )
            continue
        blocks.append(f"{section}:\n  {line}\n")
        notes.append(f"Recorded {section}.{line} in {settings.config_path}")
    if not blocks:
        return notes

    path = settings.config_path
    existing = path.read_text() if path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    header = (
        "# Recorded by 'arc container start' (MeiliSearch is optional; see "
        "docs/guides/cli-reference.md)\n"
    )
    content = existing + ("\n" if existing else "") + header + "".join(blocks)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        if path.exists():
            shutil.copymode(path, temp_name)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return notes


def meili_corpora_conflict(settings: MeiliSettings, corpora: List[str]) -> Optional[str]:
    """The refusal for MeiliSearch disabled while ``corpora`` still search text on it."""
    if settings.enabled or not corpora:
        return None
    commands = "; ".join(f"arc corpus migrate-text {name} --to qdrant" for name in corpora)
    return (
        f"meilisearch.enabled is false in {settings.config_path}, but "
        f"{'these corpora still search' if len(corpora) > 1 else 'this corpus still searches'} "
        f"text on MeiliSearch: {', '.join(corpora)}. Set meilisearch.enabled: true, run "
        f"'arc container start', then migrate: {commands}"
    )


def save_config(config: ArcaneumConfig, config_path: Path):
    """Save configuration to file.

    Args:
        config: Configuration to save
        config_path: Path to save YAML file
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, default_flow_style=False)


def _build_default_models() -> Dict[str, ModelConfig]:
    """Build DEFAULT_MODELS from EMBEDDING_MODELS with PDF-specific chunking parameters.

    Consolidates model definitions into a single source of truth (EMBEDDING_MODELS),
    then adds PDF-specific chunking based on model backend and dimensions.

    Chunking strategy:
    - SentenceTransformers models: AST-aligned chunks at 512 (768d) or 768 (1024+d).
      Smaller targets emit naturally clean function/struct-aligned chunks; larger
      targets force the AST chunker to greedy-merge then resplit at character
      boundaries (destroys AST alignment).
    - FastEmbed models: 460 tokens, safe margin from the 512-token model limit.
    - Overlap: 15% of chunk_size for context continuity.
    """
    defaults = {}

    for model_id, config in EMBEDDING_MODELS.items():
        backend = config.get("backend", "fastembed")
        dimensions = config.get("dimensions", 768)

        # late_chunking is consumed by the PDF chunker. It is not implemented
        # in the code/markdown embedding path, so code chunk sizing below is
        # based on AST cleanliness and attention memory instead of late pooling.
        supports_late_chunking = backend == "sentence-transformers"

        # Determine chunk size based on backend and model characteristics.
        #
        # Sweet spot for AST chunking is just above typical function size so
        # the chunker emits AST-aligned chunks without resplitting. Sampling
        # ~300 langref files at chunk_size=512 produced 0% mid-AST resplits
        # and max chunk ~1800 chars — comfortably within jina-code's 2048-token
        # max_seq_length and ~half the attention memory of larger targets.
        # Larger chunk_size values force the AST chunker to greedily merge
        # multiple functions/structs into one chunk, then the hard_max_chars
        # re-splitter cuts them at character boundaries — destroying both AST
        # alignment and embedding quality.
        if supports_late_chunking:
            if dimensions >= 1024:
                chunk_size = 768  # qwen3-embed, stella, codesage-large
            else:
                chunk_size = (
                    512  # jina-code: AST-clean, no resplits, ~half attention memory vs 1024
                )
        else:
            # FastEmbed models: limited by token budget, use conservative sizing
            chunk_size = 460  # Safe margin from 512 token limit

        # Calculate overlap at ~15% of chunk_size. Round down to a multiple of
        # 23 to preserve the historical default for 1536-token chunks while
        # keeping smaller AST-aligned chunks proportionally overlapped.
        chunk_overlap = max(int(chunk_size * 0.15 / 23) * 23, 1)

        char_to_token_ratio = 3.3

        defaults[model_id] = ModelConfig(
            name=config["name"],
            dimensions=dimensions,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            late_chunking=supports_late_chunking,
            char_to_token_ratio=char_to_token_ratio,
        )

    return defaults


# Default model configurations derived from EMBEDDING_MODELS with PDF-specific parameters
DEFAULT_MODELS = _build_default_models()

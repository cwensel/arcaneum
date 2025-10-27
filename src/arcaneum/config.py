"""Configuration management for Arcaneum (RDR-003)."""

from pydantic import BaseModel, Field, HttpUrl
from pathlib import Path
from typing import Dict, List, Literal
import yaml


class ModelConfig(BaseModel):
    """Configuration for a single embedding model."""
    name: str
    dimensions: int
    chunk_size: int
    chunk_overlap: int
    distance: Literal["cosine", "euclid", "dot"] = "cosine"


class QdrantConfig(BaseModel):
    """Qdrant server configuration."""
    url: str = "http://localhost:6333"
    timeout: int = 30
    grpc: bool = False


class CacheConfig(BaseModel):
    """Model cache configuration."""
    models_dir: Path = Path("./models_cache")
    max_size_gb: int = 10


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


def save_config(config: ArcaneumConfig, config_path: Path):
    """Save configuration to file.

    Args:
        config: Configuration to save
        config_path: Path to save YAML file
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config.model_dump(mode='json'), f, default_flow_style=False)


# Default model configurations (from RDR-003)
DEFAULT_MODELS = {
    "stella": ModelConfig(
        name="BAAI/bge-large-en-v1.5",
        dimensions=1024,
        chunk_size=512,
        chunk_overlap=51,
    ),
    "modernbert": ModelConfig(
        name="answerdotai/ModernBERT-base",
        dimensions=768,
        chunk_size=2048,
        chunk_overlap=205,
    ),
    "bge": ModelConfig(
        name="BAAI/bge-large-en-v1.5",
        dimensions=1024,
        chunk_size=460,
        chunk_overlap=46,
    ),
    "jina": ModelConfig(
        name="jinaai/jina-embeddings-v2-base-code",
        dimensions=768,
        chunk_size=1024,
        chunk_overlap=102,
    ),
}

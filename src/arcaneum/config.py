"""Configuration management for Arcaneum (RDR-003)."""

from pydantic import BaseModel, Field, HttpUrl
from pathlib import Path
from typing import Dict, List, Literal
import yaml
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


class QdrantConfig(BaseModel):
    """Qdrant server configuration."""
    url: str = "http://localhost:6333"
    timeout: int = 30  # General timeout for indexing operations
    search_timeout: int = 60  # Timeout for search operations (can be longer)


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


class PDFProcessingConfig(BaseModel):
    """PDF processing configuration (RDR-004)."""
    ocr_enabled: bool = False
    ocr_engine: str = "tesseract"
    ocr_language: str = "eng"
    ocr_threshold: int = 100
    batch_size: int = 512  # GPU-optimal batch size (arcaneum-2m1i, arcaneum-i7oa)
    parallel_workers: int = 4
    # Timeout settings (seconds)
    pdf_timeout: int = 600  # Total timeout per PDF file
    ocr_page_timeout: int = 60  # Timeout per OCR page
    embedding_timeout: int = 300  # Timeout for embedding generation


class ArcaneumConfig(BaseModel):
    """Root configuration."""
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    models: Dict[str, ModelConfig]
    collections: Dict[str, CollectionTemplate] = Field(default_factory=dict)
    pdf_processing: PDFProcessingConfig = Field(default_factory=PDFProcessingConfig)


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


def _build_default_models() -> Dict[str, ModelConfig]:
    """Build DEFAULT_MODELS from EMBEDDING_MODELS with PDF-specific chunking parameters.

    Consolidates model definitions into a single source of truth (EMBEDDING_MODELS),
    then adds PDF-specific chunking based on model backend and dimensions.

    Chunking strategy:
    - SentenceTransformers models: Support late_chunking, use larger chunks (1024-1536)
    - FastEmbed models: No late_chunking support, use conservative chunks (460 safe from 512 limit)
    - Overlap: 15% of chunk_size for context continuity
    """
    defaults = {}

    for model_id, config in EMBEDDING_MODELS.items():
        backend = config.get("backend", "fastembed")
        dimensions = config.get("dimensions", 768)

        # Determine late_chunking support by backend
        supports_late_chunking = backend == "sentence-transformers"

        # Determine chunk size based on backend and model characteristics
        if supports_late_chunking:
            # SentenceTransformers models: can use larger chunks with late pooling
            if dimensions >= 1024:
                chunk_size = 768  # stella, jina-v3
            else:
                chunk_size = 1536  # modernbert, jina-code, jina-v3
        else:
            # FastEmbed models: limited by token budget, use conservative sizing
            chunk_size = 460  # Safe margin from 512 token limit

        # Calculate overlap (15% of chunk_size, rounded to nearest multiple of 23 for consistency)
        chunk_overlap = max(int(chunk_size * 0.15 / 23) * 23, 1)

        # Determine char_to_token_ratio by backend
        if backend == "sentence-transformers":
            char_to_token_ratio = 3.3 if dimensions <= 768 else 3.3
        else:
            char_to_token_ratio = 3.3  # FastEmbed models

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

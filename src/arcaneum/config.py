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
    batch_size: int = 100
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


# Default model configurations (from RDR-003, updated for RDR-004)
DEFAULT_MODELS = {
    "stella": ModelConfig(
        name="BAAI/bge-large-en-v1.5",
        dimensions=1024,
        chunk_size=768,  # Conservative for PDF
        chunk_overlap=115,  # 15% overlap
        late_chunking=True,
        char_to_token_ratio=3.3,
    ),
    "modernbert": ModelConfig(
        name="answerdotai/ModernBERT-base",
        dimensions=768,
        chunk_size=1536,  # Conservative for long context
        chunk_overlap=230,  # 15% overlap
        late_chunking=True,
        char_to_token_ratio=3.4,
    ),
    "bge": ModelConfig(
        name="BAAI/bge-large-en-v1.5",
        dimensions=1024,
        chunk_size=460,  # Safe margin from 512 limit
        chunk_overlap=69,  # 15% overlap
        late_chunking=False,  # Not supported (512 token limit)
        char_to_token_ratio=3.3,
    ),
    "jina": ModelConfig(
        name="jinaai/jina-embeddings-v2-base-code",
        dimensions=768,
        chunk_size=1536,
        chunk_overlap=230,  # 15% overlap
        late_chunking=True,
        char_to_token_ratio=3.2,
    ),
}

# Recommendation 003: CLI Tool for Qdrant Collection Creation with Named Vectors

## Metadata
- **Date**: 2025-10-19
- **Status**: Recommendation
- **Type**: Feature
- **Priority**: High
- **Related Issues**: arcaneum-3
- **Related Tests**: Collection creation, model management, CLI integration tests

## Problem Statement

Create a command-line tool for managing Qdrant collections with multiple embedding models. The tool must:

1. **Support multiple embedding models** with dynamic downloading and caching
2. **Create collections with named vectors** following RDR-002 architecture
3. **Optimize chunking** based on model-specific token constraints
4. **Provide CLI-driven configuration** without relying on environment variables
5. **Be self-contained and simple** for Claude Code plugin integration

This addresses the need for:
- Collection lifecycle management (create, list, delete, inspect)
- Embedding model management (download, cache, validate)
- Configuration management via CLI flags and optional config files
- Integration with bulk upload (RDR-006) and search (RDR-007) workflows

## Context

### Background

Arcaneum requires a vector database CLI tool to support semantic search across multiple document types. Following RDR-002's architecture, collections use **named vectors** to support multiple embedding models per collection, sharing payload storage for efficiency.

The CLI tool serves as the foundation for:
- **PDF indexing** (RDR-004) - creating collections for document search
- **Source code indexing** (RDR-005) - managing code collections with git metadata
- **Bulk upload workflows** (RDR-006) - orchestrating large-scale indexing
- **Search operations** (RDR-007) - querying with model-specific vectors

Key requirements:
- All configuration via CLI flags or config file (no hidden environment variables)
- Dynamic embedding model support (not hard-coded)
- Model-aware chunking optimization
- Simple enough for local development, robust enough for production

### Technical Environment

- **Qdrant**: v1.15.4+ (Docker deployment from RDR-002)
- **Python**: >= 3.12
- **qdrant-client**: >= 1.15.0 (with FastEmbed integration)
- **FastEmbed**: >= 0.3.0 (ONNX-based embeddings)
- **CLI Framework**: Click or Typer
- **Config Format**: YAML with Pydantic validation
- **Output Formatting**: Rich library for terminal UI

**Target Embedding Models:**
- stella_en_1.5B_v5 (BAAI/bge-large-en-v1.5): 1024 dimensions
- modernbert (answerdotai/ModernBERT-base): 768 dimensions
- bge-large-en-v1.5 (BAAI/bge-large-en-v1.5): 1024 dimensions
- jina-embeddings-v3 (jinaai/jina-embeddings-v2-base-code): 768 dimensions

## Research Findings

### Investigation Process

Six parallel research tracks were conducted (tracked in Beads issues arcaneum-8 through arcaneum-13):

1. **Embedding Model Libraries** - Compared FastEmbed vs sentence-transformers
2. **Model Token Constraints** - Analyzed training parameters for 4 target models
3. **ChromaDB Patterns** - Reviewed existing implementation for migration insights
4. **Qdrant Client API** - Analyzed official Python client source code
5. **Opensource Tools** - Evaluated 9+ existing Qdrant management tools
6. **CLI Architecture** - Designed init vs collection command structure

### Key Discoveries

#### 1. Embedding Library Selection: FastEmbed vs Sentence-Transformers

**FastEmbed (RECOMMENDED):**
- **Runtime**: ONNX Runtime (no PyTorch dependency)
- **Size**: ~100MB dependencies vs ~7GB for PyTorch
- **Speed**: 2-3x faster on CPU than sentence-transformers
- **Caching**: Automatic with explicit `cache_dir` parameter
- **Models**: 40+ curated models (covers our requirements)
- **Integration**: Native support in qdrant-client library
- **License**: Apache 2.0

**Performance Comparison:**
| Metric | FastEmbed | Sentence-Transformers |
|--------|-----------|----------------------|
| CPU Speed | 2,900 chars/sec | 2,800 chars/sec |
| Dependencies | 100MB | 7GB+ |
| Initialization (cached) | ~2 seconds | ~5-10 seconds |
| Embedding Accuracy | 0.9999 cosine similarity | Reference |
| GPU Support | Optional package | Built-in |

**Decision**: Use FastEmbed for lightweight CLI distribution, sentence-transformers fallback for advanced use cases.

#### 2. Model Token Constraints and Optimal Chunking

| Model | Max Tokens | Optimal Chunk | Dimensions | Overlap | Use Case |
|-------|------------|---------------|------------|---------|----------|
| **stella_en_1.5B_v5** | 512 (training)<br>8192 (capable) | 512-1024 | 1024 | 10-20% (50-200) | General embeddings, benefits from larger chunks |
| **ModernBERT-base** | 8192 (native) | 1024-2048<br>(or full 8K) | 768 | 10-20% (100-400) | Long-context code, 2-4x faster, supports late chunking |
| **bge-large-en-v1.5** | 512 (hard limit) | 256-512 | 1024 | 10-20% (50-100) | Production-proven, MTEB #1 at release |
| **jina-embeddings-v3** | 8192 | 1024-2048 | 1024 (default)<br>Supports: 32-1024 via MRL | 10-20% (100-400) | Multilingual (89 langs), task adapters, flexible dimensions |

**Chunking Strategy by Store Type:**
- **PDF**: Default model settings, standard token-aware chunking
- **Source Code**: Reduce chunk size by 60 tokens, reduce overlap by 50%, use AST-aware chunking
- **Markdown**: Reduce chunk size by 30 tokens, use heading-aware chunking

**Character-to-Token Ratios:**
- stella: 3.2 chars/token
- modernbert: 3.4 chars/token
- bge-large: 3.3 chars/token
- jina-code: 3.2 chars/token

#### 3. ChromaDB to Qdrant Migration Insights

**What Transfers Directly:**
- ✅ Same embedding models (stella, modernbert, bge-large)
- ✅ Same chunk sizes with 10% safety margins
- ✅ Same store-specific adjustments (source-code, markdown, PDF)
- ✅ Same AST-aware chunking (ASTChunk library)
- ✅ Same metadata schema patterns

**What Changes for Qdrant:**
- 🔄 Batch size: Increase from 50 to 100-200 (Qdrant handles larger)
- 🔄 Embedding generation: Move to client-side (was server-side in ChromaDB)
- 🔄 Point IDs: Use integers (not string hashes)
- 🔄 Filtering: Use Qdrant Filter API (different syntax than ChromaDB)

**Critical Validation Needed:**
- ⚠️ Verify FastEmbed produces embeddings equivalent to ChromaDB's models
- ⚠️ Test chunk sizes work within Qdrant's upload limits
- ⚠️ Confirm batch size 100-200 performs well

#### 4. Qdrant Client Embedding Integration

**FastEmbed Integration:**
```python
from qdrant_client import QdrantClient

# Install with FastEmbed support
# pip install "qdrant-client[fastembed]>=1.14.2"

# Automatic dimension detection
client.set_model(embedding_model_name="BAAI/bge-small-en")
dim = client.get_embedding_size("BAAI/bge-small-en")

# Document-level model specification
from qdrant_client.models import Document
docs = [
    Document(text="Hello", model="sentence-transformers/all-MiniLM-L6-v2"),
]
client.add(collection_name="demo", documents=docs)
```

**Collection Creation API:**
```python
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff

client.create_collection(
    collection_name="source-code",
    vectors_config={
        "stella": VectorParams(size=1024, distance=Distance.COSINE),
        "jina": VectorParams(size=768, distance=Distance.COSINE),
    },
    hnsw_config=HnswConfigDiff(
        m=16,  # Connections per node
        ef_construct=100,  # Construction quality
    ),
    on_disk_payload=True,  # Save RAM
)
```

**Named Vectors Best Practice** (from Qdrant docs):
- ✅ **Use named vectors** for multiple models per collection
- ✅ Shared payload storage (lower memory overhead)
- ✅ Independent HNSW indexes per vector
- ❌ Avoid multiple collections (only for strong tenant isolation)

#### 5. Opensource Project Evaluation

**9 Projects Analyzed:**

| Project | Type | License | Embedding Support | Verdict |
|---------|------|---------|------------------|---------|
| **mcp-server-qdrant** | MCP server | Apache 2.0 | FastEmbed only | Reference for MCP patterns |
| **qdrant-haystack** | Framework integration | Apache 2.0 | Via Haystack | Too framework-specific |
| **analogrithems/qdrant-cli** | CLI tool | GPL-3.0 | None | Good CLI structure inspiration |
| **QdrantUI** | Web UI | MIT | Minimal | Bundle as companion tool |
| **qdrant-loader** | Enterprise toolkit | GPL-3.0 | Multi-provider | Reference for multi-provider config |
| **LangChain integration** | Framework | MIT | Via LangChain | Too framework-coupled |
| **LlamaIndex integration** | Framework | MIT | Via LlamaIndex | Too framework-coupled |
| **qdrant-client + FastEmbed** | Library | Apache 2.0 | Built-in | **Foundation library** |

**Decision**: Build from scratch using qdrant-client + FastEmbed, inspired by analogrithems/qdrant-cli structure and qdrant-loader's multi-provider config patterns. GPL-3.0 acceptable per user.

#### 6. Configuration Management Approach

**User Requirement**: All configuration via CLI flags or config file, no reliance on environment variables (unless CLI sets them for child processes).

**Configuration Hierarchy:**
1. **CLI flags** (highest priority) - Explicit per-command
2. **Config file** (middle priority) - Project or global
3. **Defaults** (lowest priority) - Sensible fallbacks

**Benefits:**
- Explicit and debuggable (no hidden env vars)
- Reproducible (config file in git)
- Flexible (can override any setting per command)
- Self-documenting (--help shows all options)

## Proposed Solution

### Approach

**CLI-First Configuration with Optional Config File**

The tool provides three ways to configure:
1. **CLI flags only** - Quick operations without files
2. **Config file** - Persistent settings for projects
3. **Mixed** - Config file with CLI flag overrides

All configuration is explicit - the CLI parses settings and passes them directly to libraries (no hidden environment variables).

### Technical Design

#### CLI Command Structure

```bash
# Initialize workspace with config file
arc init \
  --qdrant-url http://localhost:6333 \
  --cache-dir ./models_cache \
  --config-file ./arcaneum.yaml \
  --validate

# Collection management
arc collection create SOURCE_CODE \
  --config ./arcaneum.yaml \
  --models stella,jina,bge \
  --hnsw-m 16 \
  --hnsw-ef 100 \
  --on-disk-payload

arc collection list --config ./arcaneum.yaml --format table

arc collection info SOURCE_CODE --config ./arcaneum.yaml

arc collection delete SOURCE_CODE --config ./arcaneum.yaml --confirm

# Model management
arc models list --config ./arcaneum.yaml

arc models download stella --cache-dir ./models_cache --validate

arc models info stella --show-cached
```

#### Configuration File Schema (arcaneum.yaml)

```yaml
# Qdrant server configuration
qdrant:
  url: http://localhost:6333
  timeout: 30
  grpc: false  # Use gRPC instead of REST

# Model cache configuration
cache:
  models_dir: ./models_cache
  max_size_gb: 10

# Embedding models with token constraints
models:
  stella:
    name: BAAI/bge-large-en-v1.5
    dimensions: 1024
    chunk_size: 512
    chunk_overlap: 51  # 10%
    distance: cosine

  modernbert:
    name: answerdotai/ModernBERT-base
    dimensions: 768
    chunk_size: 2048
    chunk_overlap: 205  # 10%
    distance: cosine

  bge:
    name: BAAI/bge-large-en-v1.5
    dimensions: 1024
    chunk_size: 460
    chunk_overlap: 46  # 10%
    distance: cosine

  jina:
    name: jinaai/jina-embeddings-v2-base-code
    dimensions: 768
    chunk_size: 1024
    chunk_overlap: 102  # 10%
    distance: cosine

# Collection templates
collections:
  source-code:
    models: [stella, jina]
    hnsw_m: 16
    hnsw_ef_construct: 100
    on_disk_payload: true
    indexes: [programming_language, git_project_root, file_extension]

  pdf-docs:
    models: [stella, bge]
    hnsw_m: 16
    hnsw_ef_construct: 100
    on_disk_payload: true
    indexes: [filename, file_path]
```

#### Python Architecture

**Configuration Module** (`src/arcaneum/config.py`):

```python
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
    url: HttpUrl = "http://localhost:6333"
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
    """Load and validate configuration file."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return ArcaneumConfig(**data)

def save_config(config: ArcaneumConfig, config_path: Path):
    """Save configuration to file."""
    with open(config_path, 'w') as f:
        yaml.dump(config.model_dump(mode='json'), f, default_flow_style=False)

# Default model configurations
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
```

**Embedding Client** (`src/arcaneum/embeddings/client.py`):

```python
from fastembed import TextEmbedding
from pathlib import Path
from typing import Dict, List
from ..config import ModelConfig

class EmbeddingClient:
    """Manages embedding models with explicit configuration."""

    def __init__(self, cache_dir: Path, models_config: Dict[str, ModelConfig]):
        """
        Initialize with explicit paths - NO environment variables.

        Args:
            cache_dir: Explicit path to model cache directory
            models_config: Model configurations from CLI/config file
        """
        self.cache_dir = Path(cache_dir).resolve()
        self.models_config = models_config
        self._models: Dict[str, TextEmbedding] = {}

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model(self, model_key: str) -> TextEmbedding:
        """
        Get or initialize embedding model with lazy loading.

        Args:
            model_key: Model identifier (stella, jina, bge, modernbert)

        Returns:
            Initialized TextEmbedding model
        """
        if model_key not in self._models:
            if model_key not in self.models_config:
                raise ValueError(
                    f"Unknown model: {model_key}. "
                    f"Available: {list(self.models_config.keys())}"
                )

            config = self.models_config[model_key]

            # Load model with explicit cache_dir (no env vars)
            self._models[model_key] = TextEmbedding(
                model_name=config.name,
                cache_dir=str(self.cache_dir),
            )

        return self._models[model_key]

    def embed(self, texts: List[str], model_key: str) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings to embed
            model_key: Model to use (stella, jina, bge, modernbert)

        Returns:
            List of embedding vectors
        """
        model = self.get_model(model_key)
        embeddings = list(model.embed(texts))
        return embeddings

    def preload_models(self, model_keys: List[str]):
        """
        Preload models to cache. Useful for init/setup.

        Args:
            model_keys: Models to preload
        """
        for key in model_keys:
            try:
                self.get_model(key)
                config = self.models_config[key]
                print(f"✅ Loaded {key} ({config.dimensions}D)")
            except Exception as e:
                print(f"❌ Failed to load {key}: {e}")
```

**CLI Implementation** (`src/arcaneum/cli/main.py`):

```python
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff
from ..config import load_config, save_config, ArcaneumConfig, DEFAULT_MODELS
from ..embeddings.client import EmbeddingClient

console = Console()

@click.group()
def cli():
    """Arcaneum CLI - Qdrant collection management with named vectors."""
    pass

@cli.command()
@click.option('--qdrant-url', default='http://localhost:6333', help='Qdrant server URL')
@click.option('--cache-dir', type=click.Path(), default='./models_cache', help='Model cache directory')
@click.option('--config-file', type=click.Path(), default='./arcaneum.yaml', help='Config file path')
@click.option('--validate/--no-validate', default=True, help='Validate setup after init')
def init(qdrant_url, cache_dir, config_file, validate):
    """Initialize Arcaneum configuration and validate setup."""

    # Create config
    config = ArcaneumConfig(
        qdrant={'url': qdrant_url},
        cache={'models_dir': cache_dir},
        models=DEFAULT_MODELS,
    )

    # Save config file
    config_path = Path(config_file)
    save_config(config, config_path)
    console.print(f"✅ Created config file: {config_path}")

    if validate:
        # Test Qdrant connection
        try:
            client = QdrantClient(url=qdrant_url)
            client.get_collections()
            console.print(f"✅ Qdrant server accessible at {qdrant_url}")
        except Exception as e:
            console.print(f"❌ Qdrant connection failed: {e}", style="red")
            raise click.Abort()

        # Create cache directory
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        console.print(f"✅ Model cache directory: {cache_path}")

@cli.group()
def collection():
    """Collection management commands."""
    pass

@collection.command('create')
@click.argument('name')
@click.option('--config', type=click.Path(exists=True), help='Config file path')
@click.option('--qdrant-url', help='Override Qdrant URL')
@click.option('--models', help='Comma-separated model names (e.g., stella,jina)')
@click.option('--hnsw-m', type=int, default=16, help='HNSW m parameter')
@click.option('--hnsw-ef', type=int, default=100, help='HNSW ef_construct parameter')
@click.option('--on-disk-payload/--no-on-disk-payload', default=True, help='Store payload on disk')
@click.option('--indexes', help='Comma-separated field names to index')
def collection_create(name, config, qdrant_url, models, hnsw_m, hnsw_ef, on_disk_payload, indexes):
    """Create a new collection with named vectors."""

    # Load configuration
    if config:
        cfg = load_config(Path(config))
    else:
        cfg = ArcaneumConfig(models=DEFAULT_MODELS)

    # CLI flags override config
    url = qdrant_url or str(cfg.qdrant.url)
    model_list = models.split(',') if models else ['stella', 'jina']

    # Build vectors config
    vectors_config = {}
    for model_key in model_list:
        if model_key not in cfg.models:
            console.print(f"❌ Unknown model: {model_key}", style="red")
            raise click.Abort()

        model_cfg = cfg.models[model_key]
        distance = Distance.COSINE  # Default, could be configurable

        vectors_config[model_key] = VectorParams(
            size=model_cfg.dimensions,
            distance=distance,
        )

    # Create collection
    try:
        client = QdrantClient(url=url)
        client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            hnsw_config=HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef),
            on_disk_payload=on_disk_payload,
        )

        console.print(f"✅ Created collection '{name}' with {len(model_list)} models")

        # Create payload indexes if specified
        if indexes:
            index_list = indexes.split(',')
            for field_name in index_list:
                client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema="keyword",
                )
                console.print(f"  ✅ Indexed field: {field_name}")

    except Exception as e:
        console.print(f"❌ Failed to create collection: {e}", style="red")
        raise click.Abort()

@collection.command('list')
@click.option('--config', type=click.Path(exists=True), help='Config file path')
@click.option('--qdrant-url', help='Override Qdrant URL')
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), default='table')
def collection_list(config, qdrant_url, format):
    """List all collections."""

    # Load configuration
    if config:
        cfg = load_config(Path(config))
        url = qdrant_url or str(cfg.qdrant.url)
    else:
        url = qdrant_url or 'http://localhost:6333'

    try:
        client = QdrantClient(url=url)
        collections = client.get_collections()

        if format == 'table':
            table = Table(title="Qdrant Collections")
            table.add_column("Name", style="cyan")
            table.add_column("Vectors", style="green")
            table.add_column("Points", style="yellow")

            for col in collections.collections:
                col_info = client.get_collection(col.name)
                vector_count = len(col_info.config.params.vectors) if hasattr(col_info.config.params, 'vectors') else 1
                table.add_row(col.name, str(vector_count), str(col_info.points_count))

            console.print(table)
        elif format == 'json':
            import json
            print(json.dumps([col.model_dump() for col in collections.collections], indent=2))
        elif format == 'yaml':
            import yaml
            print(yaml.dump([col.model_dump() for col in collections.collections]))

    except Exception as e:
        console.print(f"❌ Failed to list collections: {e}", style="red")
        raise click.Abort()

@collection.command('info')
@click.argument('name')
@click.option('--config', type=click.Path(exists=True), help='Config file path')
@click.option('--qdrant-url', help='Override Qdrant URL')
def collection_info(name, config, qdrant_url):
    """Show detailed information about a collection."""

    # Load configuration
    if config:
        cfg = load_config(Path(config))
        url = qdrant_url or str(cfg.qdrant.url)
    else:
        url = qdrant_url or 'http://localhost:6333'

    try:
        client = QdrantClient(url=url)
        info = client.get_collection(name)

        console.print(f"\n[bold cyan]Collection: {name}[/bold cyan]")
        console.print(f"Points: {info.points_count}")
        console.print(f"Status: {info.status}")
        console.print(f"\n[bold]Vectors:[/bold]")

        if hasattr(info.config.params, 'vectors') and isinstance(info.config.params.vectors, dict):
            for vector_name, vector_params in info.config.params.vectors.items():
                console.print(f"  • {vector_name}: {vector_params.size}D ({vector_params.distance})")

        console.print(f"\n[bold]HNSW Config:[/bold]")
        hnsw = info.config.hnsw_config
        console.print(f"  m: {hnsw.m}")
        console.print(f"  ef_construct: {hnsw.ef_construct}")

    except Exception as e:
        console.print(f"❌ Failed to get collection info: {e}", style="red")
        raise click.Abort()

@collection.command('delete')
@click.argument('name')
@click.option('--config', type=click.Path(exists=True), help='Config file path')
@click.option('--qdrant-url', help='Override Qdrant URL')
@click.option('--confirm/--no-confirm', default=False, help='Skip confirmation prompt')
def collection_delete(name, config, qdrant_url, confirm):
    """Delete a collection."""

    if not confirm:
        if not click.confirm(f"Delete collection '{name}'? This cannot be undone."):
            console.print("Cancelled.")
            return

    # Load configuration
    if config:
        cfg = load_config(Path(config))
        url = qdrant_url or str(cfg.qdrant.url)
    else:
        url = qdrant_url or 'http://localhost:6333'

    try:
        client = QdrantClient(url=url)
        client.delete_collection(name)
        console.print(f"✅ Deleted collection '{name}'")

    except Exception as e:
        console.print(f"❌ Failed to delete collection: {e}", style="red")
        raise click.Abort()

@cli.group()
def models():
    """Model management commands."""
    pass

@models.command('list')
@click.option('--config', type=click.Path(exists=True), help='Config file path')
@click.option('--cache-dir', type=click.Path(), help='Override cache directory')
@click.option('--show-cached/--no-show-cached', default=True, help='Show cached status')
def models_list(config, cache_dir, show_cached):
    """List available embedding models."""

    # Load configuration
    if config:
        cfg = load_config(Path(config))
        cache = Path(cache_dir) if cache_dir else cfg.cache.models_dir
        models_cfg = cfg.models
    else:
        cache = Path(cache_dir) if cache_dir else Path('./models_cache')
        models_cfg = DEFAULT_MODELS

    table = Table(title="Embedding Models")
    table.add_column("Key", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Dimensions", style="yellow")
    table.add_column("Chunk Size", style="magenta")
    if show_cached:
        table.add_column("Cached", style="blue")

    for key, model_cfg in models_cfg.items():
        # Check if cached
        cached = "✅" if show_cached and (cache / key).exists() else "❌"

        row = [
            key,
            model_cfg.name,
            str(model_cfg.dimensions),
            str(model_cfg.chunk_size),
        ]
        if show_cached:
            row.append(cached)

        table.add_row(*row)

    console.print(table)

@models.command('download')
@click.argument('model')
@click.option('--config', type=click.Path(exists=True), help='Config file path')
@click.option('--cache-dir', type=click.Path(), help='Override cache directory')
@click.option('--validate/--no-validate', default=True, help='Test model after download')
def models_download(model, config, cache_dir, validate):
    """Download and cache an embedding model."""

    # Load configuration
    if config:
        cfg = load_config(Path(config))
        cache = Path(cache_dir) if cache_dir else cfg.cache.models_dir
        models_cfg = cfg.models
    else:
        cache = Path(cache_dir) if cache_dir else Path('./models_cache')
        models_cfg = DEFAULT_MODELS

    if model not in models_cfg:
        console.print(f"❌ Unknown model: {model}", style="red")
        console.print(f"Available: {', '.join(models_cfg.keys())}")
        raise click.Abort()

    model_cfg = models_cfg[model]

    console.print(f"📥 Downloading {model} ({model_cfg.name})...")

    try:
        # Initialize embedding client
        client = EmbeddingClient(cache, models_cfg)
        embedding_model = client.get_model(model)

        console.print(f"✅ Model cached in {cache}")

        if validate:
            # Test the model
            test_text = "Hello world"
            embedding = client.embed([test_text], model)[0]
            console.print(f"✅ Model validated ({len(embedding)} dimensions)")

    except Exception as e:
        console.print(f"❌ Failed to download model: {e}", style="red")
        raise click.Abort()

@models.command('info')
@click.argument('model')
@click.option('--config', type=click.Path(exists=True), help='Config file path')
@click.option('--show-cached/--no-show-cached', default=True, help='Show cache status')
def models_info(model, config, show_cached):
    """Show detailed information about a model."""

    # Load configuration
    if config:
        cfg = load_config(Path(config))
        cache = cfg.cache.models_dir
        models_cfg = cfg.models
    else:
        cache = Path('./models_cache')
        models_cfg = DEFAULT_MODELS

    if model not in models_cfg:
        console.print(f"❌ Unknown model: {model}", style="red")
        raise click.Abort()

    model_cfg = models_cfg[model]

    console.print(f"\n[bold cyan]Model: {model}[/bold cyan]")
    console.print(f"Name: {model_cfg.name}")
    console.print(f"Dimensions: {model_cfg.dimensions}")
    console.print(f"Chunk Size: {model_cfg.chunk_size} tokens")
    console.print(f"Chunk Overlap: {model_cfg.chunk_overlap} tokens ({model_cfg.chunk_overlap/model_cfg.chunk_size*100:.0f}%)")
    console.print(f"Distance: {model_cfg.distance}")

    if show_cached:
        cached = (cache / model).exists()
        status = "✅ Cached" if cached else "❌ Not cached"
        console.print(f"\nCache Status: {status}")
        if cached:
            console.print(f"Cache Location: {cache / model}")

if __name__ == '__main__':
    cli()
```

### Implementation Example

**Complete Workflow:**

```bash
# 1. Initialize configuration
arc init \
  --qdrant-url http://localhost:6333 \
  --cache-dir ./models_cache \
  --config-file ./arcaneum.yaml \
  --validate

# Config file created: ./arcaneum.yaml

# 2. Download embedding models
arc models download stella --validate
arc models download jina --validate

# 3. Create source code collection with multiple models
arc collection create source-code \
  --config ./arcaneum.yaml \
  --models stella,jina \
  --indexes programming_language,git_project_root

# 4. Create PDF documents collection
arc collection create pdf-docs \
  --config ./arcaneum.yaml \
  --models stella,bge \
  --indexes filename,file_path

# 5. List all collections
arc collection list --config ./arcaneum.yaml

# 6. View collection details
arc collection info source-code --config ./arcaneum.yaml

# 7. Delete collection (with confirmation)
arc collection delete old-collection --config ./arcaneum.yaml
```

## Alternatives Considered

### Alternative 1: Environment Variable-Based Configuration

**Description**: Use environment variables for all configuration (QDRANT_URL, FASTEMBED_CACHE_PATH, etc.)

**Pros**:
- Common pattern in containerized environments
- Works well with Docker Compose
- Minimal CLI flag requirements

**Cons**:
- Hidden configuration (not explicit)
- Harder to debug
- Less reproducible
- Conflicts when running multiple instances

**Reason for rejection**: User explicitly requested CLI-driven configuration without reliance on environment variables.

### Alternative 2: Global Config File Only (~/.arcaneum.yaml)

**Description**: Single user-wide configuration file, no project-local configs.

**Pros**:
- Single source of truth
- User-wide settings apply everywhere
- Simpler mental model

**Cons**:
- Less flexible for multi-project setups
- Cannot have project-specific model selections
- Hard to version control settings with project

**Reason for rejection**: Projects should be self-contained. Support both project-local (./arcaneum.yaml) and global (~/.arcaneum.yaml) with project-local taking precedence.

### Alternative 3: Config File Required (No Standalone CLI Flags)

**Description**: All operations require a config file, no standalone CLI flag operations.

**Pros**:
- Simpler CLI implementation
- Enforces configuration management
- All settings in one place

**Cons**:
- Less flexible for quick operations
- Cannot do one-off commands without config
- Harder to script

**Reason for rejection**: Config file should be optional. CLI flags work standalone for maximum flexibility.

### Alternative 4: Separate Init Subcommands

**Description**: Break init into multiple subcommands:
```bash
arc init config --qdrant-url ...
arc init cache --cache-dir ...
arc init validate
```

**Pros**:
- More granular control
- Can run steps independently
- Clear separation of concerns

**Cons**:
- More commands to remember
- More verbose for common case
- Init is usually one-time, granularity unnecessary

**Reason for rejection**: Single `init` command with flags covers all cases more simply. Can add subcommands later if needed.

### Alternative 5: Use Existing Tool (analogrithems/qdrant-cli or qdrant-loader)

**Description**: Fork and extend an existing GPL-3.0 tool rather than building from scratch.

**Pros**:
- Faster initial development
- Existing CLI patterns
- Community-tested code

**Cons**:
- **qdrant-cli**: No embedding support (major gap), cluster-focused not collection-focused
- **qdrant-loader**: Over-engineered with unnecessary features (Confluence, JIRA), enterprise-focused

**Reason for rejection**: Both tools have fundamental architecture mismatches with our requirements. Qdrant-cli lacks embeddings entirely, qdrant-loader is too heavy. Building from scratch with inspiration is cleaner.

## Trade-offs and Consequences

### Positive Consequences

- **Explicit Configuration**: All settings visible via --help, no hidden environment variables
- **Flexible Configuration**: CLI flags, config file, or both - user's choice
- **Reproducible**: Config files in git ensure consistent team/CI settings
- **Self-Contained**: No external dependencies on environment setup
- **Debuggable**: Easy to see exactly what configuration is being used
- **Lightweight**: FastEmbed keeps dependencies minimal (~100MB vs ~7GB)
- **Fast**: Client-side embeddings with ONNX Runtime (2-3x faster on CPU)
- **Named Vectors**: Resource-efficient multi-model support per RDR-002
- **Model Flexibility**: Easy to add new models via config file
- **Cache Management**: Explicit cache directory, predictable locations

### Negative Consequences

- **More CLI Flags**: More options to remember (mitigated by --help and config file)
- **Config File Management**: Users must manage config files (but optional)
- **Client-side Overhead**: Application handles embedding generation (but faster than server-side)
- **Model Consistency**: Must ensure same model for indexing and querying (documented in best practices)

### Risks and Mitigations

- **Risk**: Config file and CLI flags get out of sync
  **Mitigation**: Clear precedence order (flags > file > defaults), validation on load

- **Risk**: Users forget to download models before indexing
  **Mitigation**: Lazy loading downloads models on first use, --validate flag in init command

- **Risk**: Cache directory grows too large
  **Mitigation**: max_size_gb config option, cache-info command to monitor, document cleanup procedures

- **Risk**: Model dimensions mismatch with collection
  **Mitigation**: Validate dimensions during collection creation, auto-detect from model config

- **Risk**: ChromaDB embedding equivalence not validated
  **Mitigation**: Document validation steps in testing approach, add validation command

- **Risk**: Batch size 100-200 may not be optimal
  **Mitigation**: Make batch size configurable, document tuning guidelines, start with 100

## Implementation Plan

### Prerequisites

- [x] Docker and Docker Compose installed (from RDR-002)
- [x] Python 3.12+ environment
- [ ] qdrant-client Python package (>= 1.15.0)
- [ ] FastEmbed Python package (>= 0.3.0)
- [ ] Click or Typer CLI framework
- [ ] Rich library for terminal output
- [ ] Pydantic for configuration validation
- [ ] PyYAML for config file parsing

### Step-by-Step Implementation

#### Step 1: Project Structure Setup

Create directory structure:

```
src/arcaneum/
├── __init__.py
├── config.py              # Configuration models (Pydantic)
├── cli/
│   ├── __init__.py
│   └── main.py            # CLI commands (Click)
├── embeddings/
│   ├── __init__.py
│   └── client.py          # EmbeddingClient class
└── collections/
    ├── __init__.py
    └── manager.py         # Collection management utilities
```

#### Step 2: Configuration System

Implement `src/arcaneum/config.py`:
- Pydantic models for all config sections
- `load_config()` function with validation
- `save_config()` function
- DEFAULT_MODELS constant with model configs
- Configuration hierarchy resolution (flags > file > defaults)

#### Step 3: Embedding Client

Implement `src/arcaneum/embeddings/client.py`:
- `EmbeddingClient` class with explicit cache_dir
- Model registry from configuration
- Lazy loading with `get_model()`
- Batch embedding with `embed()`
- Preload utility for init command

#### Step 4: Init Command

Implement `arcaneum init`:
- Create config file from CLI flags
- Validate Qdrant server connection
- Create cache directory
- Optional model preloading
- Rich formatted output

#### Step 5: Collection Commands

Implement collection management:
- `arcaneum collection create` - Named vectors, HNSW config, indexes
- `arcaneum collection list` - Table/JSON/YAML output
- `arcaneum collection info` - Detailed collection inspection
- `arcaneum collection delete` - With confirmation prompt

#### Step 6: Model Commands

Implement model management:
- `arcaneum models list` - Show available models with cache status
- `arcaneum models download` - Download and cache model
- `arcaneum models info` - Show model details

#### Step 7: Testing

Create integration tests:
- Config file loading and validation
- CLI flag overrides
- Collection creation with named vectors
- Model downloading and caching
- End-to-end workflow tests

#### Step 8: Documentation

Create usage documentation:
- README with quick start guide
- Command reference (all flags documented)
- Configuration file examples
- Best practices guide
- Troubleshooting section

### Files to Create

- `src/arcaneum/config.py` - Configuration models and loaders
- `src/arcaneum/cli/main.py` - CLI commands implementation
- `src/arcaneum/embeddings/client.py` - Embedding generation client
- `src/arcaneum/collections/manager.py` - Collection utilities
- `tests/test_config.py` - Configuration tests
- `tests/test_cli.py` - CLI integration tests
- `tests/test_embeddings.py` - Embedding client tests
- `docs/cli-reference.md` - Complete CLI documentation
- `docs/configuration.md` - Config file documentation
- `examples/arcaneum.yaml` - Example configuration file

### Files to Modify

- `pyproject.toml` - Add dependencies and CLI entry point
- `.gitignore` - Add models_cache/, *.yaml (except examples/)
- `README.md` - Add CLI tool overview

### Dependencies

Add to `pyproject.toml`:

```toml
[project]
name = "arcaneum"
version = "0.1.0"
dependencies = [
    "qdrant-client[fastembed]>=1.15.0",
    "fastembed>=0.3.0",
    "click>=8.1.0",
    "rich>=13.0.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
]

[project.scripts]
arc = "arcaneum.cli.main:cli"

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
```

## Validation

### Testing Approach

1. **Unit Tests**: Configuration loading, model registry, CLI flag parsing
2. **Integration Tests**: End-to-end workflows with test Qdrant instance
3. **Manual Testing**: Real-world usage scenarios
4. **Performance Tests**: Model loading times, embedding generation speed
5. **Validation Tests**: FastEmbed vs ChromaDB embedding equivalence

### Test Scenarios

1. **Scenario**: Run `arcaneum init` with defaults
   **Expected Result**: Config file created at ./arcaneum.yaml, Qdrant connection validated, cache directory created

2. **Scenario**: Create collection with `--models stella,jina` flag (no config file)
   **Expected Result**: Collection created with two named vectors (stella: 1024D, jina: 768D), COSINE distance

3. **Scenario**: Create collection with config file, override with CLI flag `--hnsw-m 32`
   **Expected Result**: Collection uses m=32 (CLI override), other settings from config file

4. **Scenario**: Download stella model with `--validate` flag
   **Expected Result**: Model downloaded to cache, test embedding generated, dimensions validated (1024)

5. **Scenario**: List collections with `--format json`
   **Expected Result**: Valid JSON output with collection names, vector counts, point counts

6. **Scenario**: Create config file, modify models section, use in collection create
   **Expected Result**: Custom model config used, dimensions match config, chunk_size accessible for downstream use

7. **Scenario**: Delete collection without `--confirm` flag
   **Expected Result**: Interactive confirmation prompt, only delete if confirmed

8. **Scenario**: Run all commands without config file (CLI flags only)
   **Expected Result**: All commands work with defaults, no config file created unless init runs

### Performance Validation

**Model Download & Caching:**
- First download: < 30 seconds per model
- Cached load: < 2 seconds
- Cache size: ~3-4GB for 4 models (stella, modernbert, bge, jina)

**Collection Creation:**
- Simple collection (2 vectors): < 1 second
- Complex collection (4 vectors + indexes): < 3 seconds

**Embedding Generation:**
- 100 texts (stella, 1024D): < 5 seconds on CPU
- 1000 texts (jina, 768D): < 30 seconds on CPU

**Configuration Loading:**
- YAML parsing: < 100ms
- Pydantic validation: < 50ms

### Security Validation

- No hardcoded credentials or API keys
- Config file in .gitignore (except examples/)
- Cache directory permissions: 0755 (user-only write)
- No shell injection in CLI arguments (Click handles escaping)
- No eval() or exec() in config parsing (YAML safe_load)

## References

### Official Documentation
- **Qdrant Documentation**: https://qdrant.tech/documentation/
- **Qdrant Collections**: https://qdrant.tech/documentation/concepts/collections/
- **Qdrant Named Vectors**: https://qdrant.tech/documentation/concepts/vectors/
- **Qdrant Python Client**: https://python-client.qdrant.tech/
- **FastEmbed GitHub**: https://github.com/qdrant/fastembed
- **FastEmbed Documentation**: https://qdrant.github.io/fastembed/

### Research Sources
- **ChromaDB Implementation**: `/Users/cwensel/sandbox/outstar/research/chroma-embedded/upload.sh`
- **Qdrant Local Setup**: `/Users/cwensel/sandbox/outstar/research/qdrant-local/`
- **RAG Requirements**: `/Users/cwensel/sandbox/outstar/research/outstar-rag-requirements.md`

### Opensource Projects Evaluated
- **mcp-server-qdrant**: https://github.com/qdrant/mcp-server-qdrant
- **analogrithems/qdrant-cli**: https://github.com/analogrithems/qdrant-cli
- **qdrant-loader**: https://github.com/martin-papy/qdrant-loader
- **QdrantUI**: https://github.com/imadfaouzi/QdrantUI

### Model Documentation
- **stella_en_1.5B_v5**: https://huggingface.co/dunzhang/stella_en_1.5B_v5
- **ModernBERT**: https://huggingface.co/answerdotai/ModernBERT-base
- **bge-large-en-v1.5**: https://huggingface.co/BAAI/bge-large-en-v1.5
- **jina-embeddings-v3**: https://huggingface.co/jinaai/jina-embeddings-v3

### Python Libraries
- **Click**: https://click.palletsprojects.com/
- **Typer**: https://typer.tiangolo.com/
- **Rich**: https://rich.readthedocs.io/
- **Pydantic**: https://docs.pydantic.dev/

### Related RDRs
- **RDR-002**: Qdrant Server Setup (dependency)
- **RDR-004**: PDF Indexing (will use this CLI tool)
- **RDR-005**: Source Code Indexing (will use this CLI tool)
- **RDR-006**: Bulk Upload Plugin (will integrate this CLI)
- **RDR-007**: Search Plugin (will query collections created by this tool)

## Notes

### Future Enhancements

1. **Global Config File Support** (~/.arcaneum.yaml)
   - User-wide defaults
   - Project-local configs override global
   - Environment-specific profiles (dev, staging, prod)

2. **Batch Operations**
   - Create multiple collections from template
   - Bulk model downloads
   - Collection cloning/forking

3. **Collection Templates**
   - Pre-defined templates for common use cases
   - Template registry with versioning
   - Template import/export

4. **Model Registry Extensions**
   - Custom model registration
   - HuggingFace model import
   - Model version pinning

5. **Validation Commands**
   - Embedding equivalence testing (FastEmbed vs ChromaDB)
   - Performance benchmarking
   - Collection health checks

6. **Shell Completion**
   - Bash completion scripts
   - Zsh completion scripts
   - Fish completion scripts

7. **Interactive Mode**
   - Guided collection creation wizard
   - Interactive config editor
   - TUI for collection management

8. **Advanced Indexing**
   - Geo-spatial index creation
   - Full-text index integration (if Qdrant adds support)
   - Composite index configuration

### Known Limitations

- **Single-threaded embedding generation**: FastEmbed doesn't parallelize automatically (can be added via multiprocessing)
- **No model auto-selection**: User must specify models (could add heuristics)
- **No collection migration tools**: Cannot easily change vector dimensions (requires recreation)
- **Limited output formats**: Only table, JSON, YAML (could add CSV, Parquet)
- **No server management**: Assumes Qdrant already running (RDR-002 covers deployment)

### Migration Path

**From ChromaDB to Qdrant:**

1. **Model Equivalence Validation**:
   ```bash
   # Test stella embeddings match between ChromaDB and Qdrant
   arcaneum models download stella --validate
   # Compare embeddings for sample texts
   # Document: models produce 0.9999+ cosine similarity
   ```

2. **Collection Creation**:
   ```bash
   # Create collections with same models as ChromaDB
   arcaneum collection create source-code --models stella,jina
   arcaneum collection create pdf-docs --models stella,bge
   ```

3. **Chunking Validation**:
   - Use same chunk sizes from ChromaDB (460-920 tokens)
   - Apply same store-specific adjustments
   - Test batch size increase (50 → 100-200)

4. **Data Migration** (covered in RDR-006):
   - Export from ChromaDB
   - Re-embed with FastEmbed
   - Validate embedding similarity
   - Upload to Qdrant with increased batch size

### Configuration Best Practices

**Project-Local Config** (recommended):
```yaml
# ./arcaneum.yaml - checked into git
qdrant:
  url: http://localhost:6333

cache:
  models_dir: ./models_cache

models:
  # Only list models you actually use
  stella: ...
  jina: ...

collections:
  # Define your project's collections
  my-docs: ...
```

**Global Config** (future):
```yaml
# ~/.arcaneum.yaml - user-wide settings
qdrant:
  url: http://localhost:6333
  timeout: 60

cache:
  models_dir: ~/.cache/arcaneum/models
  max_size_gb: 20

# Default models available globally
models:
  stella: ...
  modernbert: ...
  bge: ...
  jina: ...
```

**Precedence**: Project config > Global config > CLI flags > Defaults

This RDR provides the complete specification for implementing the Arcaneum CLI tool for Qdrant collection management with named vectors and CLI-driven configuration.

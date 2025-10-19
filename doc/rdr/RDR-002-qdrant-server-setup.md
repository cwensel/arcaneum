# Recommendation 002: Qdrant Server Setup with Client-side Embeddings

## Metadata
- **Date**: 2025-10-19
- **Status**: Recommendation
- **Type**: Architecture
- **Priority**: High
- **Related Issues**: arcaneum-2
- **Related Tests**: Qdrant deployment and collection creation tests

## Problem Statement

Establish a standardized Qdrant server deployment that supports:
1. **Multiple embedding models** efficiently in a single instance
2. **Data persistence** across container restarts
3. **Client-side embedding generation** using FastEmbed
4. **Simple deployment** suitable for local development and single-server production

This RDR addresses:
- Docker vs local binary decision
- Embedding model configuration strategy
- Collection architecture for multiple models
- Volume persistence configuration
- Server management operations

## Context

### Background

Arcaneum requires a vector database to support semantic search across multiple document types (source code, PDFs, markdown). The system must:
- Support multiple embedding models (stella, modernbert, bge-large, jina-code)
- Allow different document types to use different or multiple models
- Persist data reliably across container restarts
- Be simple enough for local development
- Scale to production single-server deployments

Initial reference implementation exists at `/Users/cwensel/sandbox/outstar/research/qdrant-local/server.sh`, providing a basic Docker run script.

### Technical Environment

- **Qdrant**: v1.15.4 (current stable)
- **Docker**: Official `qdrant/qdrant` image
- **Python**: >= 3.12
- **FastEmbed**: Client-side embedding library (via qdrant-client)
- **Supported Models**:
  - stella_en_1.5B_v5 (1024 dimensions)
  - modernbert (1024 dimensions)
  - bge-large-en-v1.5 (1024 dimensions)
  - jina-code (768 dimensions)

## Research Findings

### Investigation Process

1. **Qdrant Official Documentation Review**
   - Architecture and embedding strategies
   - Docker deployment patterns
   - Collection configuration options
   - Multitenancy best practices

2. **Qdrant Source Code Analysis**
   - Repository: https://github.com/qdrant/qdrant
   - Key findings:
     - No built-in embedding generation in core
     - Inference service framework exists but forwards to external services
     - Collections fully support multiple vector dimensions
     - Named vectors architecture for multi-model scenarios

3. **Docker Deployment Patterns**
   - Official Docker Hub images and documentation
   - Multi-node cluster configurations
   - Volume persistence strategies
   - Environment variable configuration

4. **FastEmbed Integration**
   - Repository: https://github.com/qdrant/fastembed
   - Client-side ONNX-based embedding generation
   - Lightweight, serverless-compatible
   - Integrated into qdrant-client library

### Key Discoveries

**Critical Finding: Qdrant Does NOT Generate Embeddings**
- Qdrant is a pure vector database - it stores and searches pre-computed vectors
- All embedding generation happens client-side (application code)
- FastEmbed is a separate library, not part of Qdrant server
- This is by design to give users control over compute resources

**Named Vectors vs Multiple Collections**
- Qdrant **strongly recommends** named vectors over multiple collections
- Named vectors: Single collection with multiple embedding types per point
- Benefits: Shared payload storage, lower resource overhead, more efficient
- Multiple collections: Only for scenarios requiring strong tenant isolation
- Resource overhead: Each collection requires separate indexes and memory

**Multi-Model Support Architecture**
```python
# Named vectors example (RECOMMENDED)
client.create_collection(
    collection_name="source-code",
    vectors_config={
        "stella": VectorParams(size=1024, distance=Distance.COSINE),
        "modernbert": VectorParams(size=1024, distance=Distance.COSINE),
        "jina": VectorParams(size=768, distance=Distance.COSINE),
    }
)

# Search using specific model
client.search(
    collection_name="source-code",
    query_vector=NamedVector(name="stella", vector=embedding),
    limit=10
)
```

**Docker Deployment Best Practices**
- Official image: `qdrant/qdrant:v1.15.4` (use version tags, not `latest`)
- Primary storage path: `/qdrant/storage`
- Snapshots path: `/qdrant/snapshots`
- Ports: 6333 (REST), 6334 (gRPC), 6335 (cluster P2P - internal only)
- Health endpoints: `/healthz`, `/readyz`, `/livez`
- Configuration priority: Environment variables > config files

**Resource Requirements**
- Formula: `memory_size = vectors × dimensions × 4 bytes × 1.5`
- Example: 1M vectors × 1024 dims = ~5.7GB RAM (in-memory)
- On-disk storage option: ~135MB for 1M vectors (requires fast SSD)
- Recommended: 4GB RAM for moderate workloads, 2 CPUs

## Proposed Solution

### Approach

**Single-Node Docker Deployment with Named Vectors Architecture**

1. **Docker Compose for Deployment**
   - Static docker-compose.yml template
   - Version-pinned image for stability
   - Volume mounts for persistence
   - Resource limits for production safety

2. **Client-side Embedding Generation**
   - FastEmbed library in Python application code
   - Models cached in mounted volume to avoid re-downloads
   - Environment variable: `SENTENCE_TRANSFORMERS_HOME=/models`

3. **Named Vectors Architecture**
   - One collection per document type (e.g., `source-code`, `pdf-docs`)
   - Multiple named vectors per collection (one per embedding model)
   - Shared payload across all vectors (single copy of metadata)

4. **Management Scripts**
   - Simple bash script for start/stop/logs operations
   - Health check verification
   - Status reporting

### Technical Design

#### Docker Compose Configuration

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:v1.15.4
    container_name: qdrant-arcaneum
    restart: unless-stopped

    ports:
      - "6333:6333"  # REST API
      - "6334:6334"  # gRPC API

    volumes:
      - ./qdrant_storage:/qdrant/storage
      - ./qdrant_snapshots:/qdrant/snapshots
      - ./models_cache:/models

    environment:
      - QDRANT__LOG_LEVEL=INFO
      - SENTENCE_TRANSFORMERS_HOME=/models

    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

volumes:
  qdrant_storage:
  qdrant_snapshots:
  models_cache:
```

#### Collection Architecture

**Document Type Collections**:
- `source-code` - All source code chunks
- `pdf-docs` - All PDF document chunks
- `markdown-docs` - All markdown document chunks (future)

**Named Vectors per Collection**:
```python
# Example: source-code collection
{
    "stella": VectorParams(size=1024, distance=Distance.COSINE),
    "modernbert": VectorParams(size=1024, distance=Distance.COSINE),
    "bge": VectorParams(size=1024, distance=Distance.COSINE),
    "jina": VectorParams(size=768, distance=Distance.COSINE),
}
```

**Payload Schema** (shared across all vectors):
```python
{
    "file_path": "src/main.py",
    "filename": "main.py",
    "chunk_index": 0,
    "chunk_count": 5,
    "doc_type": "source-code",
    "programming_language": "python",
    "created_at": "2025-10-19T12:00:00Z",
    # Git metadata (for source code)
    "git_project_root": "/path/to/repo",
    "git_commit_hash": "abc123",
    "git_remote_url": "https://github.com/org/repo",
}
```

#### Directory Structure

```
arcaneum/
├── docker-compose.yml           # Qdrant deployment
├── qdrant_storage/              # Data persistence (mounted volume)
├── qdrant_snapshots/            # Backup snapshots (mounted volume)
├── models_cache/                # FastEmbed model cache (mounted volume)
├── scripts/
│   └── qdrant-manage.sh        # Server management script
└── src/
    └── arcaneum/
        ├── embeddings/          # FastEmbed integration
        ├── collections/         # Collection creation utilities
        └── indexing/            # Bulk upload logic
```

### Implementation Example

#### Server Management Script

```bash
#!/bin/bash
# scripts/qdrant-manage.sh

case "$1" in
    start)
        docker compose up -d
        sleep 2
        curl -s http://localhost:6333/healthz && echo "✅ Qdrant started"
        ;;
    stop)
        docker compose down
        ;;
    restart)
        docker compose restart
        ;;
    logs)
        docker compose logs -f qdrant
        ;;
    status)
        docker compose ps
        curl -s http://localhost:6333/healthz
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status}"
        exit 1
        ;;
esac
```

#### Collection Creation with Named Vectors

```python
from qdrant_client import QdrantClient, models

def create_source_code_collection(client: QdrantClient):
    """Create source-code collection with multiple embedding models."""

    client.create_collection(
        collection_name="source-code",
        vectors_config={
            # High-quality general embeddings
            "stella": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
                on_disk=False,  # Keep in RAM for speed
            ),
            # Transformer-based embeddings
            "modernbert": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
                on_disk=False,
            ),
            # BGE embeddings
            "bge": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
                on_disk=False,
            ),
            # Code-optimized embeddings
            "jina": models.VectorParams(
                size=768,
                distance=models.Distance.COSINE,
                on_disk=False,
            ),
        },
        # Optimize for code search
        hnsw_config=models.HnswConfigDiff(
            m=16,  # Good balance
            ef_construct=100,
            full_scan_threshold=10000,
        ),
        # Shared payload on disk to save RAM
        on_disk_payload=True,
    )

    # Create indexes on frequently filtered fields
    client.create_payload_index(
        collection_name="source-code",
        field_name="programming_language",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    client.create_payload_index(
        collection_name="source-code",
        field_name="git_project_root",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
```

#### Embedding Generation with FastEmbed

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, NamedVector
from fastembed import TextEmbedding

def index_code_chunks(client: QdrantClient, chunks: list[dict]):
    """Index code chunks with multiple embedding models."""

    # Initialize embedding models (cached in /models volume)
    stella_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
    jina_model = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-code")

    points = []
    for idx, chunk in enumerate(chunks):
        # Generate embeddings client-side
        stella_embedding = list(stella_model.embed([chunk["text"]]))[0]
        jina_embedding = list(jina_model.embed([chunk["text"]]))[0]

        # Create point with named vectors
        point = PointStruct(
            id=idx,
            vector={
                "stella": stella_embedding,
                "jina": jina_embedding,
            },
            payload={
                "file_path": chunk["file_path"],
                "text": chunk["text"],
                "doc_type": "source-code",
                "programming_language": chunk["language"],
            }
        )
        points.append(point)

    # Batch upload
    client.upsert(
        collection_name="source-code",
        points=points,
    )
```

#### Querying with Specific Model

```python
def search_code(client: QdrantClient, query: str, model: str = "stella"):
    """Search source code using specific embedding model."""

    # Generate query embedding with same model
    if model == "stella":
        embedding_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
    elif model == "jina":
        embedding_model = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-code")

    query_embedding = list(embedding_model.embed([query]))[0]

    # Search using named vector
    results = client.search(
        collection_name="source-code",
        query_vector=NamedVector(
            name=model,
            vector=query_embedding,
        ),
        limit=10,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="programming_language",
                    match=models.MatchValue(value="python"),
                )
            ]
        ),
    )

    return results
```

## Alternatives Considered

### Alternative 1: Multiple Collections (One per Model + DocType)

**Description**: Create separate collections like `source-code-stella`, `source-code-jina`, `pdf-docs-stella`, etc.

**Pros**:
- Stronger isolation between models
- Simpler per-collection logic (no named vectors)
- Independent HNSW tuning per model
- Easier to delete/recreate single model

**Cons**:
- **High resource overhead** - each collection requires separate indexes
- **Duplicated payload storage** - same metadata stored multiple times
- **Not Qdrant-recommended** - violates best practices
- More complex management (6+ collections vs 2-3)

**Reason for rejection**: Resource inefficiency and against Qdrant's recommendations. Named vectors solve the same problems with lower overhead.

### Alternative 2: Server-side Embeddings via Inference Service

**Description**: Deploy separate embedding service container that Qdrant calls for on-the-fly embedding generation.

**Pros**:
- Centralized embedding logic
- Could reduce client-side dependencies
- Single service for all embedding needs

**Cons**:
- **Not Qdrant's recommended pattern** - they advocate client-side
- Requires additional Docker service and management
- Adds network latency for every embedding
- More complex deployment and troubleshooting
- FastEmbed designed for client-side use

**Reason for rejection**: Adds complexity without clear benefits. Client-side embedding is simpler, faster, and follows Qdrant's design philosophy.

### Alternative 3: Dynamic Docker Compose Generation

**Description**: CLI tool that generates docker-compose.yml based on user input (ports, volumes, resources).

**Pros**:
- More user-friendly for beginners
- Can customize for different environments
- Validates configuration before writing

**Cons**:
- Added complexity in tooling
- Users may prefer transparent static files
- Harder to version control generated files
- Unnecessary for single-service deployment

**Reason for rejection**: Static template is sufficient for initial implementation. Can add generation later if needed.

## Trade-offs and Consequences

### Positive Consequences

- **Resource Efficiency**: Named vectors reduce memory usage by 3-4x compared to multiple collections
- **Simple Deployment**: Single Docker Compose file, no complex orchestration
- **Flexible Search**: Can query with any embedding model independently
- **Best Practices**: Follows Qdrant's official recommendations
- **Easy Development**: Local development matches production configuration
- **Data Persistence**: Automatic persistence with volume mounts
- **Model Caching**: FastEmbed models downloaded once and reused

### Negative Consequences

- **Client-side Overhead**: Application must handle embedding generation
- **Model Consistency**: Developers must ensure same model used for indexing and querying
- **Single-node Limit**: No high-availability or horizontal scaling (deferred to future RDR)
- **Resource Sharing**: All collections share same server resources

### Risks and Mitigations

- **Risk**: Running out of memory with large collections
  **Mitigation**: Configure `on_disk=True` for vectors, use `on_disk_payload=True`, monitor memory usage

- **Risk**: Docker container data loss
  **Mitigation**: Volume mounts ensure persistence, document snapshot/backup procedures

- **Risk**: Embedding model inconsistency (indexing with model A, querying with model B)
  **Mitigation**: Store model name in payload metadata, validate at query time, document best practices

- **Risk**: Performance degradation with multiple models
  **Mitigation**: HNSW indexes are independent per named vector, no cross-model performance impact

- **Risk**: Docker Compose health check limitations (no curl in image)
  **Mitigation**: External health check script, document workaround, or use Kubernetes for production

## Implementation Plan

### Prerequisites

- [x] Docker and Docker Compose installed
- [x] Python 3.12+ environment
- [ ] qdrant-client Python package
- [ ] FastEmbed Python package (or alternative embedding library)

### Step-by-Step Implementation

#### Step 1: Create Docker Compose Configuration

Create `docker-compose.yml` at repository root:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:v1.15.4
    container_name: qdrant-arcaneum
    restart: unless-stopped

    ports:
      - "6333:6333"
      - "6334:6334"

    volumes:
      - ./qdrant_storage:/qdrant/storage
      - ./qdrant_snapshots:/qdrant/snapshots
      - ./models_cache:/models

    environment:
      - QDRANT__LOG_LEVEL=INFO
      - SENTENCE_TRANSFORMERS_HOME=/models

    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

volumes:
  qdrant_storage:
  qdrant_snapshots:
  models_cache:
```

#### Step 2: Create Management Script

Create `scripts/qdrant-manage.sh`:

```bash
#!/bin/bash
set -e

case "$1" in
    start)
        echo "🚀 Starting Qdrant..."
        docker compose up -d
        sleep 3
        if curl -sf http://localhost:6333/healthz > /dev/null; then
            echo "✅ Qdrant started successfully"
            echo "📊 REST API: http://localhost:6333"
            echo "🔗 Dashboard: http://localhost:6333/dashboard"
        else
            echo "❌ Qdrant failed to start"
            exit 1
        fi
        ;;
    stop)
        echo "🛑 Stopping Qdrant..."
        docker compose down
        echo "✅ Qdrant stopped"
        ;;
    restart)
        echo "🔄 Restarting Qdrant..."
        docker compose restart
        sleep 2
        curl -sf http://localhost:6333/healthz && echo "✅ Restarted"
        ;;
    logs)
        docker compose logs -f qdrant
        ;;
    status)
        echo "📊 Qdrant Status:"
        docker compose ps
        echo ""
        curl -s http://localhost:6333/healthz && echo "✅ Healthy" || echo "❌ Unhealthy"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status}"
        exit 1
        ;;
esac
```

Make executable: `chmod +x scripts/qdrant-manage.sh`

#### Step 3: Create Collection Initialization Module

Create `src/arcaneum/collections/init.py`:

```python
from qdrant_client import QdrantClient, models
from typing import Dict, List

COLLECTION_CONFIGS = {
    "source-code": {
        "vectors": {
            "stella": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "modernbert": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "bge": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "jina": models.VectorParams(size=768, distance=models.Distance.COSINE),
        },
        "hnsw_config": models.HnswConfigDiff(m=16, ef_construct=100),
        "on_disk_payload": True,
        "indexes": ["programming_language", "git_project_root", "file_extension"],
    },
    "pdf-docs": {
        "vectors": {
            "stella": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "modernbert": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "bge": models.VectorParams(size=1024, distance=models.Distance.COSINE),
        },
        "hnsw_config": models.HnswConfigDiff(m=16, ef_construct=100),
        "on_disk_payload": True,
        "indexes": ["filename", "file_path"],
    },
}

def init_collections(url: str = "http://localhost:6333"):
    """Initialize all Arcaneum collections."""
    client = QdrantClient(url=url)

    for name, config in COLLECTION_CONFIGS.items():
        print(f"Creating collection: {name}")

        client.create_collection(
            collection_name=name,
            vectors_config=config["vectors"],
            hnsw_config=config["hnsw_config"],
            on_disk_payload=config["on_disk_payload"],
        )

        # Create payload indexes
        for field_name in config["indexes"]:
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        print(f"✅ Created {name} with {len(config['vectors'])} vector types")

if __name__ == "__main__":
    init_collections()
```

#### Step 4: Create Embedding Utilities Module

Create `src/arcaneum/embeddings/client.py`:

```python
from fastembed import TextEmbedding
from typing import Dict, List
import os

# Model configurations
EMBEDDING_MODELS = {
    "stella": {
        "name": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
    },
    "jina": {
        "name": "jinaai/jina-embeddings-v2-base-code",
        "dimensions": 768,
    },
    # Add more models as needed
}

class EmbeddingClient:
    """Manages embedding model instances with caching."""

    def __init__(self, cache_dir: str = "./models_cache"):
        self.cache_dir = cache_dir
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
        self._models: Dict[str, TextEmbedding] = {}

    def get_model(self, model_name: str) -> TextEmbedding:
        """Get or initialize embedding model."""
        if model_name not in self._models:
            config = EMBEDDING_MODELS[model_name]
            self._models[model_name] = TextEmbedding(
                model_name=config["name"],
                cache_dir=self.cache_dir,
            )
        return self._models[model_name]

    def embed(self, texts: List[str], model_name: str) -> List[List[float]]:
        """Generate embeddings for texts using specified model."""
        model = self.get_model(model_name)
        embeddings = list(model.embed(texts))
        return embeddings
```

#### Step 5: Document Usage and Best Practices

Create `docs/qdrant-setup.md` with:
- Server startup procedures
- Collection creation examples
- Embedding generation patterns
- Query examples with different models
- Backup and restore procedures
- Troubleshooting guide

### Files to Create

- `docker-compose.yml` - Qdrant deployment configuration
- `scripts/qdrant-manage.sh` - Server management utilities
- `src/arcaneum/collections/init.py` - Collection initialization
- `src/arcaneum/embeddings/client.py` - Embedding generation utilities
- `docs/qdrant-setup.md` - Usage documentation
- `.gitignore` entries for `qdrant_storage/`, `qdrant_snapshots/`, `models_cache/`

### Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    "qdrant-client>=1.15.0",
    "fastembed>=0.3.0",
]
```

## Validation

### Testing Approach

1. **Deployment Validation**: Verify Docker Compose starts successfully
2. **Health Check Validation**: Confirm health endpoints respond
3. **Collection Creation**: Test named vector collection creation
4. **Embedding Generation**: Verify FastEmbed model loading and embedding
5. **Data Persistence**: Stop/restart container, verify data survives
6. **Multi-model Search**: Query same collection with different models

### Test Scenarios

1. **Scenario**: Fresh deployment with `docker compose up`
   **Expected Result**: Container starts, health check passes, dashboard accessible at :6333/dashboard

2. **Scenario**: Create collection with 4 named vectors
   **Expected Result**: Collection created with stella (1024d), modernbert (1024d), bge (1024d), jina (768d)

3. **Scenario**: Index 100 code chunks with stella and jina embeddings
   **Expected Result**: 100 points inserted with dual vectors, payload shared

4. **Scenario**: Query with stella model, then query same data with jina model
   **Expected Result**: Both return relevant results, different ranking orders

5. **Scenario**: Stop container, restart, query data
   **Expected Result**: All data persisted, queries work immediately

6. **Scenario**: Generate embeddings for 1000 texts with FastEmbed
   **Expected Result**: Models downloaded to cache once, subsequent runs use cache

### Performance Validation

- Initial load of 10K vectors across 2 models: < 5 minutes
- Query latency (1024d vector): < 50ms for 10K vectors
- Model loading time (first run): < 30 seconds
- Model loading time (cached): < 2 seconds
- Memory usage: < 4GB for 100K vectors in-memory

### Security Validation

- No exposed API keys or credentials in docker-compose.yml
- Volume mounts use relative paths (no absolute host paths)
- Resource limits prevent container from consuming all host resources
- No privileged mode or host network access required

## References

- **Qdrant Documentation**: https://qdrant.tech/documentation/
- **Qdrant Collections**: https://qdrant.tech/documentation/concepts/collections/
- **Qdrant Named Vectors**: https://qdrant.tech/documentation/concepts/vectors/
- **Qdrant Multitenancy**: https://qdrant.tech/documentation/guides/multiple-partitions/
- **Qdrant Docker Installation**: https://qdrant.tech/documentation/guides/installation/
- **Qdrant Configuration**: https://qdrant.tech/documentation/guides/configuration/
- **FastEmbed GitHub**: https://github.com/qdrant/fastembed
- **Qdrant Python Client**: https://python-client.qdrant.tech/
- **Docker Compose Specification**: https://docs.docker.com/compose/compose-file/
- **Reference Implementation**: `/Users/cwensel/sandbox/outstar/research/qdrant-local/server.sh`

## Notes

### Future Enhancements

1. **Multi-node Clustering** (RDR-006 or later)
   - 3-node cluster for high availability
   - Replication factor configuration
   - Load balancing strategies

2. **Security Hardening** (future RDR)
   - TLS certificate setup
   - API key authentication
   - Network isolation
   - Read-only root filesystem

3. **Automated Backups** (future RDR)
   - Scheduled snapshot creation
   - S3 backup integration
   - Restore testing automation

4. **Monitoring and Observability** (future RDR)
   - Prometheus metrics collection
   - Grafana dashboards
   - Alerting rules

5. **Additional Embedding Models**
   - Cohere embeddings
   - OpenAI embeddings (ada-002, text-embedding-3)
   - Custom fine-tuned models

### Known Limitations

- Single-node only (no horizontal scaling)
- No built-in backup automation
- Manual model consistency management
- Docker Compose health checks require external tools
- Resource limits apply to entire server (not per-collection)

### Migration Path

If users later need to migrate from named vectors to separate collections (or vice versa):

1. Create snapshot of existing collection
2. Create new collection(s) with desired structure
3. Write migration script to transform and re-upload data
4. Validate data integrity
5. Update application code to use new structure
6. Delete old collection after validation period

This RDR provides the foundation for subsequent RDRs covering collection management (RDR-003), PDF indexing (RDR-004), and source code indexing (RDR-005).

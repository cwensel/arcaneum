# Recommendation 008: Full-Text Search Server Setup (MeiliSearch)

## Metadata
- **Date**: 2025-10-21
- **Status**: Recommendation
- **Type**: Architecture
- **Priority**: High
- **Related Issues**: arcaneum-67, arcaneum-72, arcaneum-73, arcaneum-74, arcaneum-75, arcaneum-76, arcaneum-77
- **Related Tests**: MeiliSearch deployment tests, index creation tests, search functionality tests

## Problem Statement

Establish a full-text search server deployment that complements the Qdrant vector search setup from RDR-002, providing:
1. **Exact phrase matching** with quote syntax (`"def authenticate"`)
2. **Regex and wildcard search** for precise code/document queries
3. **Line-number precision** for exact code location results
4. **Simple Docker deployment** matching RDR-002's simplicity
5. **Complementary workflow** to semantic search (not merged, separate use cases)

This RDR addresses:
- Full-text engine selection (MeiliSearch vs Elasticsearch)
- Docker deployment configuration
- Index schema design for code and documents
- Integration with existing Qdrant workflow
- Python client and CLI patterns

## Context

### Background

Arcaneum provides semantic search via Qdrant (RDR-002, RDR-007) but requires complementary full-text search for:
- **Exact string matching**: Find literal code patterns like `"class UserAuth"`
- **Regex searches**: Complex patterns like `function\s+\w+\(.*\)`
- **Line-level precision**: Return `file.py:123` for exact locations
- **API/function name lookup**: Find exact identifiers without semantic ambiguity

**Expected Workflow**:
1. User: "Find authentication patterns" → Claude uses **semantic search** (RDR-007)
2. User: "Find exact string 'def authenticate'" → Claude uses **full-text search** (this RDR)

Reference: arcaneum-7 mentioned MeiliSearch as potential solution

### Technical Environment

- **MeiliSearch**: v1.32.x (latest stable as of Jan 2026)
- **Docker**: Official `getmeili/meilisearch` image
- **Python**: >= 3.12
- **meilisearch-python**: >= 0.39.0 (Python client library)
- **Existing Stack**: Qdrant v1.16.2 (RDR-002), CLI framework (RDR-003), Claude Code integration (RDR-006)

**Note on MeiliSearch versions**: This RDR was originally written for v1.24.0. Breaking changes since then:

- v1.30.0: Network object structure changed (affects clustering configurations)
- v1.31.0: S3-streaming snapshots now Enterprise Edition only

**Key Design Principles**:
- Mirror RDR-002's Docker simplicity
- Complementary to Qdrant, not replacement
- CLI-first approach matching RDR-003
- Independent datastores (no cross-communication)

## Research Findings

### Investigation Process

Six parallel research tracks completed via Beads issues (arcaneum-72 through arcaneum-77):

1. **MeiliSearch Docker Deployment** (arcaneum-72): Official image analysis, configuration options
2. **Elasticsearch Comparison** (arcaneum-73): Alternative evaluation, resource comparison
3. **Qdrant Integration Patterns** (arcaneum-74): Docker Compose structure, port conflicts, volume organization
4. **Index Schema Design** (arcaneum-75): Code vs document settings, filterable attributes
5. **Python Client Patterns** (arcaneum-76): meilisearch-python API, CLI command structure
6. **Engine Selection Decision** (arcaneum-77): MeiliSearch vs Elasticsearch trade-offs

### Key Discoveries

#### 1. MeiliSearch vs Elasticsearch: Decision Analysis

**MeiliSearch Advantages** (PRIMARY RECOMMENDATION):
- **54.8x less memory** than Elasticsearch (96-200MB vs 2-4GB)
- **Sub-50ms search latency** for typical queries
- **Simple setup**: No JVM tuning, no SSL certificate complexity
- **Built-in typo tolerance**: Works out of the box
- **Developer-friendly**: Single binary, minimal configuration
- **Instant search optimized**: Perfect for code/document lookup

**Elasticsearch Advantages** (ALTERNATIVE):
- Distributed architecture for horizontal scaling
- Advanced analytics and aggregations
- Complex query DSL for sophisticated searches
- Better for >100GB data and multi-node clusters
- ML features, graph exploration, geo-spatial queries

**Decision Rationale**:
- Arcaneum prioritizes **simplicity** (matches Qdrant choice from RDR-002)
- **Resource efficiency** critical for local development
- **Code/document search** doesn't require analytics features
- **CLI-first approach** benefits from simple APIs

**Recommendation**: Use MeiliSearch as primary, document Elasticsearch as alternative for advanced use cases

#### 2. MeiliSearch Docker Architecture

**Official Docker Image Analysis**:
- Base: Alpine 3.22 (lightweight Linux)
- Binary: `/bin/meilisearch`
- Default port: 7700 (HTTP API)
- Default data path: `/meili_data`
- Process manager: `tini` for proper signal handling

**Configuration via Environment Variables**:
- `MEILI_ENV`: `production` or `development`
- `MEILI_MASTER_KEY`: API authentication (required in production)
- `MEILI_HTTP_ADDR`: Bind address (default: `0.0.0.0:7700`)
- `MEILI_DB_PATH`: Database storage location
- `MEILI_MAX_INDEXING_MEMORY`: Memory limit for indexing operations
- `MEILI_MAX_INDEXING_THREADS`: Thread limit for indexing
- `MEILI_SNAPSHOT_DIR`: Snapshot storage location
- `MEILI_DUMP_DIR`: Backup dump location

**Key Findings**:

- Memory limit must be **manually set** (issue #4686): MeiliSearch doesn't auto-detect Docker container limits
- Recommendation: Set `MEILI_MAX_INDEXING_MEMORY` to **2/3 of container memory**
- Health endpoint: `/health` for monitoring
- v1.32.x improvements: Continued performance enhancements since v1.24.0

#### 3. Integration with Qdrant Workflow (RDR-002)

**Docker Compose Structure**:

| Aspect | Qdrant (RDR-002) | MeiliSearch (this RDR) |
|--------|------------------|----------------------|
| **Port** | 6333 (REST), 6334 (gRPC) | 7700 (HTTP) |
| **Volume** | qdrant-arcaneum-storage (named) | meilisearch-arcaneum-data (named) |
| **Image** | qdrant/qdrant:v1.16.2 | getmeili/meilisearch:v1.32 |
| **Health Check** | `/healthz` | `/health` |
| **Memory** | 4GB | 4GB (but uses 96-200MB) |

**Integration Pattern**:
- **Single docker-compose.yml** with both services
- **Independent datastores**: No service-to-service communication
- **Parallel indexing**: Application indexes to both (dual indexing)
- **Query routing**: Application chooses semantic vs full-text

#### 4. Index Schema Design for Code and Documents

**MeiliSearch Index Settings for Source Code**:
```json
{
  "searchableAttributes": [
    "content",
    "filename",
    "function_names",
    "class_names"
  ],
  "filterableAttributes": [
    "language",
    "project",
    "branch",
    "file_path"
  ],
  "sortableAttributes": [],
  "typoTolerance": {
    "enabled": true,
    "minWordSizeForTypos": {
      "oneTypo": 7,
      "twoTypos": 12
    }
  },
  "stopWords": []
}
```

**MeiliSearch Index Settings for PDF Documents**:
```json
{
  "searchableAttributes": [
    "content",
    "title",
    "author"
  ],
  "filterableAttributes": [
    "filename",
    "file_path",
    "page_number"
  ],
  "typoTolerance": {
    "enabled": true,
    "minWordSizeForTypos": {
      "oneTypo": 5,
      "twoTypos": 9
    }
  },
  "stopWords": ["the", "a", "an", "and", "or", "but"]
}
```

**Phrase Matching Syntax**:
- Exact phrase: `"def calculate_total"`
- Case-insensitive by default
- Handles soft separators: `-`, `_`, `|`
- Tokenization: whitespace, quotes, separators

**Metadata Alignment with Qdrant**:
- Use same field names as Qdrant payload (language, project, branch, file_path)
- Full-text engine stores actual text content
- Qdrant stores embeddings + metadata only

#### 5. Python Client Integration

**meilisearch-python Library**:
```python
import meilisearch

# Initialize client
client = meilisearch.Client('http://localhost:7700', 'master_key')

# Create index
index = client.index('source-code')

# Configure settings
index.update_settings({
    'filterableAttributes': ['language', 'project', 'branch'],
    'searchableAttributes': ['content', 'filename']
})

# Add documents
documents = [
    {
        'id': 1,
        'content': 'def calculate_total(items):',
        'language': 'python',
        'file_path': 'src/utils.py',
        'line_number': 42
    }
]
index.add_documents(documents)

# Search with exact phrase
results = index.search('"calculate_total"', {
    'filter': 'language = python'
})

# Search with filter
results = index.search('authentication', {
    'filter': 'project = my-app AND branch = main',
    'attributesToHighlight': ['content']
})
```

**API Characteristics**:
- Simple, synchronous API (async also available)
- Standard exceptions for error handling
- Batch operations: add_documents, update_documents, delete_documents
- Search options: filters, highlights, limits, facets

## Proposed Solution

### Approach

**Single Docker Compose with Dual Services**

Deploy MeiliSearch alongside Qdrant in unified docker-compose.yml, maintaining:
- **Independent operation**: Services don't communicate
- **Consistent patterns**: Mirror RDR-002 Docker deployment
- **Complementary indexing**: Application indexes to both
- **Query routing**: Application/Claude chooses search type

### Technical Design

#### Docker Compose Configuration

**Updated `docker-compose.yml`** (extends RDR-002):

```yaml
services:
  # Existing Qdrant service (from RDR-002)
  qdrant:
    image: qdrant/qdrant:v1.16.2
    container_name: qdrant-arcaneum
    restart: unless-stopped
    ports:
      - "6333:6333"  # REST API
      - "6334:6334"  # gRPC API
    volumes:
      - qdrant-arcaneum-storage:/qdrant/storage
      - qdrant-arcaneum-snapshots:/qdrant/snapshots
    environment:
      - QDRANT__LOG_LEVEL=INFO
      # Disk-biased configuration for low-memory operation
      - QDRANT__STORAGE__ON_DISK_PAYLOAD=true
      - QDRANT__STORAGE__WAL__WAL_CAPACITY_MB=16
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

  # New MeiliSearch service (this RDR)
  meilisearch:
    image: getmeili/meilisearch:v1.32
    container_name: meilisearch-arcaneum
    restart: unless-stopped
    ports:
      - "7700:7700"  # HTTP API
    volumes:
      - meilisearch-arcaneum-data:/meili_data        # Data persistence
      - meilisearch-arcaneum-dumps:/dumps            # Backup dumps
      - meilisearch-arcaneum-snapshots:/snapshots    # Snapshots
    environment:
      # Core settings (MEILI_* are MeiliSearch's internal env vars)
      - MEILI_ENV=production
      - MEILI_MASTER_KEY=${MEILISEARCH_API_KEY}  # Maps from Arcaneum's .env
      - MEILI_HTTP_ADDR=0.0.0.0:7700

      # Resource management (CRITICAL: Must set manually)
      - MEILI_MAX_INDEXING_MEMORY=2.5GiB  # 2/3 of 4GB container limit
      - MEILI_MAX_INDEXING_THREADS=4

      # Payload limits
      - MEILI_HTTP_PAYLOAD_SIZE_LIMIT=100MB

      # Persistence paths
      - MEILI_DB_PATH=/meili_data/data.ms
      - MEILI_DUMP_DIR=/dumps
      - MEILI_SNAPSHOT_DIR=/snapshots
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '4.0'
        reservations:
          memory: 1G
          cpus: '2.0'
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:7700/health"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s

volumes:
  qdrant-arcaneum-storage:
    driver: local
  qdrant-arcaneum-snapshots:
    driver: local
  meilisearch-arcaneum-data:
    driver: local
  meilisearch-arcaneum-dumps:
    driver: local
  meilisearch-arcaneum-snapshots:
    driver: local
```

**Environment Variables** (`.env` file):

```bash
# Qdrant settings (from RDR-002)
QDRANT_URL=http://localhost:6333

# MeiliSearch settings (new)
# These are Arcaneum's application-level variables (used by Python client)
# Docker Compose maps MEILISEARCH_API_KEY to MeiliSearch's internal MEILI_MASTER_KEY
MEILISEARCH_URL=http://localhost:7700
MEILISEARCH_API_KEY=your_secure_master_key_here_min_16_chars
```

#### Directory Structure

```
arcaneum/
├── deploy/
│   └── docker-compose.yml       # Combined Qdrant + MeiliSearch
├── .env                         # Environment variables
├── scripts/
│   ├── qdrant-manage.sh        # Qdrant management (from RDR-002)
│   └── meilisearch-manage.sh   # MeiliSearch management (NEW)
└── src/
    └── arcaneum/
        ├── fulltext/           # MeiliSearch integration (NEW)
        │   ├── __init__.py
        │   ├── client.py       # MeiliSearch client wrapper
        │   └── indexes.py      # Index configuration
        └── cli/
            └── fulltext.py     # CLI commands (NEW)

# Note: Data is stored in Docker named volumes (not local directories):
# - qdrant-arcaneum-storage
# - qdrant-arcaneum-snapshots
# - meilisearch-arcaneum-data
# - meilisearch-arcaneum-dumps
# - meilisearch-arcaneum-snapshots
```

#### Management Script

**`scripts/meilisearch-manage.sh`**:

```bash
#!/bin/bash
# scripts/meilisearch-manage.sh
# Management script for MeiliSearch service

set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

MEILISEARCH_URL="${MEILISEARCH_URL:-http://localhost:7700}"

case "$1" in
    start)
        echo "Starting MeiliSearch..."
        docker compose -f deploy/docker-compose.yml up -d meilisearch
        sleep 3
        if curl -sf "${MEILISEARCH_URL}/health" > /dev/null; then
            echo "MeiliSearch started successfully"
            echo "HTTP API: ${MEILISEARCH_URL}"
        else
            echo "MeiliSearch failed to start"
            exit 1
        fi
        ;;

    stop)
        echo "Stopping MeiliSearch..."
        docker compose -f deploy/docker-compose.yml stop meilisearch
        echo "MeiliSearch stopped"
        ;;

    restart)
        echo "Restarting MeiliSearch..."
        docker compose -f deploy/docker-compose.yml restart meilisearch
        sleep 2
        curl -sf "${MEILISEARCH_URL}/health" && echo "Restarted"
        ;;

    logs)
        docker compose -f deploy/docker-compose.yml logs -f meilisearch
        ;;

    status)
        echo "MeiliSearch Status:"
        docker compose -f deploy/docker-compose.yml ps meilisearch
        echo ""
        if curl -sf "${MEILISEARCH_URL}/health" > /dev/null; then
            echo "Healthy"
            curl -sf -H "Authorization: Bearer ${MEILISEARCH_API_KEY}" \
                "${MEILISEARCH_URL}/stats" | jq '.' || echo "Stats unavailable"
        else
            echo "Unhealthy"
        fi
        ;;

    create-dump)
        echo "Creating dump..."
        curl -X POST "${MEILISEARCH_URL}/dumps" \
          -H "Authorization: Bearer ${MEILISEARCH_API_KEY}"
        echo "Dump creation initiated"
        ;;

    list-indexes)
        echo "MeiliSearch Indexes:"
        curl -sf -H "Authorization: Bearer ${MEILISEARCH_API_KEY}" \
            "${MEILISEARCH_URL}/indexes" | jq '.results[] | {uid: .uid, primaryKey: .primaryKey}'
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|logs|status|create-dump|list-indexes}"
        exit 1
        ;;
esac
```

Make executable: `chmod +x scripts/meilisearch-manage.sh`

#### Python Client Integration

**`src/arcaneum/fulltext/client.py`**:

```python
"""MeiliSearch client wrapper for Arcaneum."""

import meilisearch
from typing import Dict, List, Optional, Any
from pathlib import Path


class FullTextClient:
    """Manages MeiliSearch client with explicit configuration."""

    def __init__(self, url: str, api_key: str):
        """
        Initialize MeiliSearch client.

        Args:
            url: MeiliSearch server URL (e.g., http://localhost:7700)
            api_key: Master key for authentication
        """
        self.url = url
        self.client = meilisearch.Client(url, api_key)

    def create_index(
        self,
        name: str,
        primary_key: str = "id",
        settings: Optional[Dict[str, Any]] = None
    ) -> meilisearch.index.Index:
        """
        Create a new index with optional settings.

        Args:
            name: Index name (e.g., 'source-code', 'pdf-docs')
            primary_key: Primary key field name
            settings: Index settings (searchable/filterable attributes, etc.)

        Returns:
            Created index object
        """
        # Create index - returns TaskInfo object (not dict)
        task = self.client.create_index(name, {'primaryKey': primary_key})
        self.client.wait_for_task(task.task_uid)

        index = self.client.index(name)

        # Apply settings if provided
        if settings:
            task = index.update_settings(settings)
            self.client.wait_for_task(task.task_uid)

        return index

    def get_index(self, name: str) -> meilisearch.index.Index:
        """Get existing index by name."""
        return self.client.index(name)

    def list_indexes(self) -> List[Dict[str, Any]]:
        """List all indexes with their configurations."""
        result = self.client.get_indexes()
        return result.get('results', [])

    def delete_index(self, name: str) -> None:
        """Delete an index."""
        task = self.client.delete_index(name)
        self.client.wait_for_task(task.task_uid)

    def add_documents(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        primary_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add documents to an index.

        Args:
            index_name: Target index name
            documents: List of document dictionaries
            primary_key: Optional primary key field

        Returns:
            Task information
        """
        index = self.get_index(index_name)
        task = index.add_documents(documents, primary_key)
        return task

    def search(
        self,
        index_name: str,
        query: str,
        filter: Optional[str] = None,
        limit: int = 20,
        attributes_to_highlight: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Search an index.

        Args:
            index_name: Index to search
            query: Search query (use quotes for exact phrases)
            filter: Filter expression (e.g., 'language = python')
            limit: Maximum results
            attributes_to_highlight: Fields to highlight in results

        Returns:
            Search results with hits, processing time, etc.
        """
        index = self.get_index(index_name)

        search_params = {'limit': limit}
        if filter:
            search_params['filter'] = filter
        if attributes_to_highlight:
            search_params['attributesToHighlight'] = attributes_to_highlight

        return index.search(query, search_params)

    def health_check(self) -> bool:
        """Check if MeiliSearch server is healthy."""
        try:
            health = self.client.health()
            return health.get('status') == 'available'
        except Exception:
            return False
```

**`src/arcaneum/fulltext/indexes.py`**:

```python
"""Index configuration templates for MeiliSearch."""

from typing import Dict, Any


# Index settings for source code
SOURCE_CODE_SETTINGS: Dict[str, Any] = {
    "searchableAttributes": [
        "content",
        "filename",
        "function_names",
        "class_names",
    ],
    "filterableAttributes": [
        "language",
        "project",
        "branch",
        "file_path",
        "file_extension",
    ],
    "sortableAttributes": [],
    "typoTolerance": {
        "enabled": True,
        "minWordSizeForTypos": {
            "oneTypo": 7,   # Higher threshold for code
            "twoTypos": 12
        }
    },
    "stopWords": [],  # Preserve all code keywords
    "pagination": {
        "maxTotalHits": 1000
    }
}


# Index settings for PDF documents
PDF_DOCS_SETTINGS: Dict[str, Any] = {
    "searchableAttributes": [
        "content",
        "title",
        "author",
        "filename",
    ],
    "filterableAttributes": [
        "filename",
        "file_path",
        "page_number",
        "document_type",
    ],
    "sortableAttributes": ["page_number"],
    "typoTolerance": {
        "enabled": True,
        "minWordSizeForTypos": {
            "oneTypo": 5,
            "twoTypos": 9
        }
    },
    "stopWords": [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"
    ],
    "pagination": {
        "maxTotalHits": 1000
    }
}


def get_index_settings(index_type: str) -> Dict[str, Any]:
    """
    Get index settings by type.

    Args:
        index_type: 'source-code' or 'pdf-docs'

    Returns:
        Index settings dictionary

    Raises:
        ValueError: If index_type is unknown
    """
    settings_map = {
        "source-code": SOURCE_CODE_SETTINGS,
        "pdf-docs": PDF_DOCS_SETTINGS,
    }

    if index_type not in settings_map:
        raise ValueError(
            f"Unknown index type: {index_type}. "
            f"Available: {list(settings_map.keys())}"
        )

    return settings_map[index_type]
```

#### CLI Integration

**Note on CLI Structure**: The main.py already defines `arc search text` for full-text search
(integrated into the `search` command group alongside `arc search semantic`). The implementation
below provides the `search_text_command` function that main.py imports, plus index management
commands under a separate `fulltext` group for administration tasks.

**`src/arcaneum/cli/fulltext.py`**:

```python
"""CLI commands for full-text search operations."""

import click
import os
from rich.console import Console
from rich.table import Table
from pathlib import Path
from ..fulltext.client import FullTextClient
from ..fulltext.indexes import get_index_settings


console = Console()


def get_client() -> FullTextClient:
    """Get MeiliSearch client from environment variables."""
    url = os.environ.get('MEILISEARCH_URL', 'http://localhost:7700')
    api_key = os.environ.get('MEILISEARCH_API_KEY')
    if not api_key:
        raise click.ClickException(
            "MEILISEARCH_API_KEY environment variable required"
        )
    return FullTextClient(url, api_key)


def search_text_command(query, index_name, filter_arg, limit, offset, output_json, verbose):
    """
    Implementation for 'arc search text' command.
    Called from main.py search group.
    """
    try:
        client = get_client()
        results = client.search(
            index_name,
            query,
            filter=filter_arg,
            limit=limit,
            attributes_to_highlight=['content']
        )

        if output_json:
            import json
            print(json.dumps(results, indent=2))
            return

        console.print(f"\n[bold]Search Results[/bold] ({results['processingTimeMs']}ms)")
        console.print(f"Found {results['estimatedTotalHits']} matches\n")

        for i, hit in enumerate(results['hits'], 1):
            console.print(f"[cyan]{i}. {hit.get('filename', hit.get('file_path', 'Unknown'))}[/cyan]")
            if 'line_number' in hit:
                console.print(f"   Line: {hit['line_number']}")

            # Show highlighted content
            if '_formatted' in hit and 'content' in hit['_formatted']:
                content = hit['_formatted']['content'][:200]
                console.print(f"   {content}...\n")

    except Exception as e:
        console.print(f"[ERROR] Search failed: {e}", style="red")
        raise SystemExit(1)


# Index management commands (arc fulltext ...)
@click.group()
def fulltext():
    """Full-text index management commands (MeiliSearch)."""
    pass


@fulltext.command('create-index')
@click.argument('name')
@click.option('--type', 'index_type',
              type=click.Choice(['source-code', 'pdf-docs']),
              help='Index type (determines settings)')
def create_index(name, index_type):
    """Create a new full-text search index."""
    try:
        client = get_client()

        # Get settings if type specified
        settings = None
        if index_type:
            settings = get_index_settings(index_type)
            console.print(f"Using {index_type} settings")

        client.create_index(name, primary_key='id', settings=settings)
        console.print(f"Created index '{name}'")

        if settings:
            console.print(f"  Searchable attributes: {len(settings['searchableAttributes'])}")
            console.print(f"  Filterable attributes: {len(settings['filterableAttributes'])}")

    except Exception as e:
        console.print(f"[ERROR] Failed to create index: {e}", style="red")
        raise SystemExit(1)


@fulltext.command('list-indexes')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def list_indexes(output_json):
    """List all full-text search indexes."""
    try:
        client = get_client()
        indexes = client.list_indexes()

        if output_json:
            import json
            print(json.dumps(indexes, indent=2))
            return

        table = Table(title="MeiliSearch Indexes")
        table.add_column("Name", style="cyan")
        table.add_column("Primary Key", style="green")
        table.add_column("Created", style="yellow")

        for idx in indexes:
            table.add_row(
                idx['uid'],
                idx.get('primaryKey', 'N/A'),
                idx.get('createdAt', 'N/A')[:10]
            )

        console.print(table)

    except Exception as e:
        console.print(f"[ERROR] Failed to list indexes: {e}", style="red")
        raise SystemExit(1)


@fulltext.command('delete-index')
@click.argument('name')
@click.option('--confirm/--no-confirm', default=False,
              help='Skip confirmation prompt')
def delete_index(name, confirm):
    """Delete a full-text search index."""
    if not confirm:
        if not click.confirm(f"Delete index '{name}'? This cannot be undone."):
            console.print("Cancelled.")
            return

    try:
        client = get_client()
        client.delete_index(name)
        console.print(f"Deleted index '{name}'")

    except Exception as e:
        console.print(f"[ERROR] Failed to delete index: {e}", style="red")
        raise SystemExit(1)
```

**Integration with main CLI** (`src/arcaneum/cli/main.py`):

The search command is already defined in main.py as `arc search text`. The fulltext
group provides index administration commands:

```python
# main.py already has:
@search.command('text')
def search_text(query, index_name, filter_arg, limit, offset, output_json, verbose):
    """Keyword-based full-text search"""
    from arcaneum.cli.fulltext import search_text_command
    search_text_command(query, index_name, filter_arg, limit, offset, output_json, verbose)

# Add fulltext admin commands
from .fulltext import fulltext
cli.add_command(fulltext)
```

**CLI Command Summary**:

- `arc search text "query" --index IndexName` - Full-text search (parallel to `arc search semantic`)
- `arc fulltext create-index name --type source-code` - Create index with settings
- `arc fulltext list-indexes` - List all MeiliSearch indexes
- `arc fulltext delete-index name` - Delete an index

### Implementation Example

**Complete Workflow**:

```bash
# 1. Start both services
docker compose -f deploy/docker-compose.yml up -d

# Verify both running
scripts/qdrant-manage.sh status
scripts/meilisearch-manage.sh status

# 2. Set environment variables
export MEILISEARCH_URL=http://localhost:7700
export MEILISEARCH_API_KEY=your_secure_key

# 3. Create full-text indexes
arc fulltext create-index source-code --type source-code
arc fulltext create-index pdf-docs --type pdf-docs

# 4. List indexes
arc fulltext list-indexes

# 5. Index documents (future: RDR-009 will implement dual indexing)
# Documents indexed to both Qdrant (vectors) and MeiliSearch (text)

# 6. Search examples
# Exact phrase search (using arc search text)
arc search text '"def authenticate"' --index source-code

# Filtered search
arc search text 'authentication' \
  --index source-code \
  --filter 'language = python AND project = my-app'

# PDF document search
arc search text 'machine learning' \
  --index pdf-docs \
  --filter 'page_number > 10'

# JSON output for scripting
arc search text 'authentication' --index source-code --json
```

## Alternatives Considered

### Alternative 1: Elasticsearch

**Description**: Use Elasticsearch instead of MeiliSearch for full-text search

**Pros**:
- More mature ecosystem
- Advanced analytics capabilities
- Distributed architecture support
- Better for large-scale deployments (>100GB)
- Complex query DSL for sophisticated searches
- ML features, graph exploration, geo-spatial queries

**Cons**:
- **54.8x more memory usage** (2-4GB minimum vs 96-200MB)
- **Complex setup**: SSL certificates, JVM tuning, ulimits configuration
- **Slower for simple queries**: Higher latency for typical search operations
- **Overkill for Arcaneum's use case**: Don't need analytics features
- **Against simplicity principle**: Contradicts RDR-002's Docker simplicity

**Reason for rejection**: Resource inefficiency and setup complexity contradict Arcaneum's
simplicity goals. MeiliSearch sufficient for code/document search use case.

### Alternative 2: Separate Docker Compose Files

**Description**: Maintain separate docker-compose.yml for Qdrant and MeiliSearch

**Pros**:
- Cleaner separation of concerns
- Can start/stop services independently
- Easier to version control changes

**Cons**:
- More files to manage
- Network configuration complexity
- Harder for users to manage both services
- No unified `docker compose up` workflow

**Reason for rejection**: Unified docker-compose.yml simpler for users, services don't need separate management

### Alternative 3: TypeSense

**Description**: Use TypeSense as alternative to MeiliSearch

**Pros**:
- Similar performance characteristics to MeiliSearch
- Open source with good documentation
- Typo tolerance and instant search

**Cons**:
- Less mature ecosystem than MeiliSearch
- Smaller community
- Fewer integrations and tools
- No significant advantages over MeiliSearch

**Reason for rejection**: MeiliSearch more established with better ecosystem

### Alternative 4: Apache Solr

**Description**: Use Apache Solr for full-text search

**Pros**:
- Very mature, battle-tested
- Rich feature set
- Good for complex schemas

**Cons**:
- Legacy technology (older than Elasticsearch)
- Complex configuration (XML-based)
- Higher resource usage
- Steep learning curve
- Against simplicity principle

**Reason for rejection**: Legacy complexity contradicts modern, simple deployment goals

## Trade-offs and Consequences

### Positive Consequences

- **Resource Efficiency**: 54.8x less memory than Elasticsearch (96-200MB typical)
- **Simple Deployment**: Single binary, minimal configuration (matches RDR-002)
- **Fast Search**: Sub-50ms latency for typical queries
- **Built-in Features**: Typo tolerance, phrase matching, highlighting
- **Developer Experience**: Simple API, easy integration
- **Complementary to Qdrant**: Clear use case separation (semantic vs exact)
- **Docker Simplicity**: Single docker-compose.yml, consistent patterns
- **CLI-First**: Matches established Arcaneum architecture

### Negative Consequences

- **Single-Node Only**: No distributed architecture (like Qdrant limitation)
- **No Analytics**: Limited to search, no aggregations/analytics
- **Index Size Growth**: Full-text indexes can be large for massive codebases
- **Manual Memory Config**: Must explicitly set MEILI_MAX_INDEXING_MEMORY

### Risks and Mitigations

- **Risk**: Docker memory limit not detected automatically (MeiliSearch issue #4686)
  **Mitigation**: Explicitly set MEILI_MAX_INDEXING_MEMORY to 2/3 container limit, document in RDR

- **Risk**: Index size exceeds disk capacity
  **Mitigation**: Monitor disk usage, document cleanup procedures, use snapshots for backups

- **Risk**: Query performance degrades with large indexes
  **Mitigation**: Monitor query times, optimize filterable attributes, consider index splitting

- **Risk**: Port conflict with other services
  **Mitigation**: Default port 7700 unlikely to conflict, document in docker-compose.yml

- **Risk**: Users forget MEILI_MASTER_KEY in production
  **Mitigation**: Docker Compose fails if not set, validation in management script

## Implementation Plan

### Prerequisites

- [x] RDR-002: Qdrant server setup (dependency for Docker patterns)
- [x] RDR-003: Collection management (dependency for CLI patterns)
- [ ] Docker and Docker Compose installed
- [ ] Python >= 3.12
- [ ] meilisearch-python package

### Step-by-Step Implementation

#### Step 1: Update Docker Compose Configuration

Extend existing docker-compose.yml from RDR-002:

1. Add meilisearch service with:
   - Version-pinned image (v1.32)
   - Port 7700 mapping
   - Three named volumes (data, dumps, snapshots)
   - Environment variables with MEILISEARCH_API_KEY
   - Resource limits (4GB memory, 4 CPUs)
   - Health check configuration

2. Add named volume definitions for MeiliSearch (matching Qdrant pattern)

3. Update .env.example with MEILISEARCH_URL and MEILISEARCH_API_KEY

#### Step 2: Create Management Script

Create `scripts/meilisearch-manage.sh`:

1. Implement start/stop/restart/logs/status commands
2. Add health check integration
3. Add create-dump command for backups
4. Add list-indexes command for inspection
5. Make executable with `chmod +x`

**Estimated effort**: 1 day

#### Step 3: Implement Python Client

Create `src/arcaneum/fulltext/` module:

1. **client.py**: FullTextClient class
   - __init__ with URL and API key
   - create_index, get_index, list_indexes, delete_index
   - add_documents, search methods
   - health_check method

2. **indexes.py**: Index configuration templates
   - SOURCE_CODE_SETTINGS dictionary
   - PDF_DOCS_SETTINGS dictionary
   - get_index_settings helper function

3. **__init__.py**: Module exports

**Estimated effort**: 2 days

#### Step 4: Implement CLI Commands

Create `src/arcaneum/cli/fulltext.py`:

1. Implement fulltext command group
2. Create create-index command with type selection
3. Create list-indexes command with table output
4. Create delete-index command with confirmation
5. Create search command with filters and highlighting
6. Register with main CLI in main.py

**Estimated effort**: 2 days

#### Step 5: Configuration Integration

Update configuration system to support MeiliSearch:

1. Environment variable support:
   - MEILISEARCH_URL (default: http://localhost:7700)
   - MEILISEARCH_API_KEY (required for all operations)
2. Update .env.example with MeiliSearch variables
3. Add MeiliSearch connection check to `arc doctor` command

#### Step 6: Testing

Create comprehensive tests:

1. Docker deployment tests:
   - Service startup
   - Health check verification
   - Volume persistence

2. Client tests:
   - Index creation with settings
   - Document operations
   - Search functionality
   - Error handling

3. CLI tests:
   - Command execution
   - Output formatting
   - Error messages

4. Integration tests:
   - End-to-end workflow
   - Dual indexing with Qdrant
   - Query routing

**Estimated effort**: 3 days

#### Step 7: Documentation

Create comprehensive documentation:

1. Update README with MeiliSearch setup
2. Create docs/fulltext-search.md with:
   - Architecture overview
   - Setup instructions
   - Index configuration guide
   - Search syntax examples
   - Troubleshooting guide

3. Update RDR index with RDR-008 reference

**Estimated effort**: 2 days

### Files to Create

**Docker & Infrastructure**:
- `docker-compose.yml` - Update with MeiliSearch service (extends RDR-002)
- `.env` - Add MEILI_MASTER_KEY
- `scripts/meilisearch-manage.sh` - Management script

**Python Modules**:
- `src/arcaneum/fulltext/__init__.py` - Module init
- `src/arcaneum/fulltext/client.py` - FullTextClient class
- `src/arcaneum/fulltext/indexes.py` - Index configurations

**CLI**:
- `src/arcaneum/cli/fulltext.py` - CLI commands
- Update `src/arcaneum/cli/main.py` - Register fulltext group

**Tests**:
- `tests/fulltext/test_client.py` - Client tests
- `tests/fulltext/test_indexes.py` - Configuration tests
- `tests/cli/test_fulltext.py` - CLI tests
- `tests/integration/test_dual_search.py` - Integration tests

**Documentation**:
- `docs/fulltext-search.md` - User guide
- Update `README.md` - Add MeiliSearch section

### Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    "qdrant-client>=1.16.1",             # Existing from RDR-002
    "meilisearch>=0.39.0",               # NEW for full-text search
    "click>=8.3.0",                      # Existing from RDR-003
    "rich>=14.2.0",                      # Existing from RDR-003
]
```

**Note**: The meilisearch-python client v0.39.0+ returns `TaskInfo` objects from async
operations. Use `task.task_uid` (attribute access) not `task['taskUid']` (dict access).

## Validation

### Testing Approach

1. **Deployment Testing**: Verify Docker services start and remain healthy
2. **Index Testing**: Create indexes with different settings, verify configuration
3. **Search Testing**: Test exact phrases, filters, highlights
4. **Integration Testing**: Verify dual indexing workflow with Qdrant
5. **Performance Testing**: Measure search latency, index creation time

### Test Scenarios

**Scenario 1: Fresh Deployment**
- **Action**: Run `docker compose up -d`
- **Expected**: Both Qdrant and MeiliSearch start, health checks pass
- **Validation**: `scripts/meilisearch-manage.sh status` shows "Healthy"

**Scenario 2: Index Creation with Settings**

- **Action**: `arc fulltext create-index source-code --type source-code`
- **Expected**: Index created with SOURCE_CODE_SETTINGS applied
- **Validation**: Verify filterable/searchable attributes via MeiliSearch API

**Scenario 3: Exact Phrase Search**

- **Action**: `arc search text '"def authenticate"' --index source-code`
- **Expected**: Returns only documents with exact phrase "def authenticate"
- **Validation**: Check results contain exact match, case-insensitive

**Scenario 4: Filtered Search**

- **Action**: `arc search text 'authentication' --index source-code --filter 'language = python'`
- **Expected**: Returns Python files only containing "authentication"
- **Validation**: All results have language='python' in metadata

**Scenario 5: Container Restart Data Persistence**
- **Action**: Add documents, restart container, search again
- **Expected**: All documents persist across restart
- **Validation**: Same search results before and after restart

**Scenario 6: Memory Limit Handling**
- **Action**: Index large dataset with MEILI_MAX_INDEXING_MEMORY set
- **Expected**: Indexing completes without OOM errors
- **Validation**: Monitor container memory usage stays within limits

### Performance Validation

- **Search Latency**: < 50ms for typical queries (measured via processingTimeMs)
- **Index Creation**: < 5 seconds for empty index with settings
- **Document Indexing**: ~1000 documents/second for typical code/PDF content
- **Memory Usage**: 96-200MB for typical workload (much less than 4GB limit)
- **Startup Time**: < 30 seconds from container start to health check passing

### Security Validation

- MEILISEARCH_API_KEY required in production mode
- No hardcoded credentials in docker-compose.yml (uses .env)
- Health endpoint accessible without authentication
- Search endpoints require API key
- Named volumes managed by Docker (not local directories)

## References

- **MeiliSearch Documentation**: https://www.meilisearch.com/docs
- **MeiliSearch GitHub**: https://github.com/meilisearch/meilisearch
- **meilisearch-python SDK**: https://github.com/meilisearch/meilisearch-python
- **Docker Hub Image**: https://hub.docker.com/r/getmeili/meilisearch
- **MeiliSearch Releases**: https://github.com/meilisearch/meilisearch/releases
- **Issue #4686**: Docker memory limit detection problem
- **Issue #1060**: Phrase search implementation
- **Beads Issues**: arcaneum-72 through arcaneum-77 (research findings)
- **RDR-002**: Qdrant Server Setup (Docker pattern reference)
- **RDR-003**: Collection Creation (CLI pattern reference)
- **RDR-006**: Claude Code Integration (slash command reference)
- **RDR-007**: Semantic Search (complementary workflow reference)

## Notes

### Key Design Decisions

1. **MeiliSearch over Elasticsearch**: Resource efficiency and simplicity prioritized
2. **Unified Docker Compose**: Single file for both Qdrant and MeiliSearch
3. **Complementary, Not Merged**: Separate search types (semantic vs full-text)
4. **CLI-First**: Consistent with existing Arcaneum architecture
5. **Manual Memory Configuration**: MEILI_MAX_INDEXING_MEMORY must be set explicitly

### Future Enhancements

1. **Dual Indexing CLI** (arcaneum-68): Commands to index to both Qdrant and MeiliSearch simultaneously
2. **Smart Query Routing**: CLI/Claude Code logic to choose semantic vs full-text automatically
3. **Index Optimization**: Analyze and tune index settings based on usage patterns
4. **Snapshot Automation**: Scheduled backups via cron or Docker healthcheck
5. **Multi-Language Support**: Extend typo tolerance and stop words for non-English content
6. **Hybrid Search**: Combine semantic and full-text results with ranking fusion

### Known Limitations

- **Single-Node Architecture**: No distributed mode (like Qdrant and Elasticsearch)
- **No Built-in Analytics**: Limited to search, no aggregations
- **Memory Detection**: Must manually configure MEILI_MAX_INDEXING_MEMORY
- **Index Rebuilds**: Changing filterable attributes requires full reindex
- **Version Migrations**: Use dumps for upgrades, snapshots only work within same version

### Integration with Future RDRs

**Dual Collection Creation** (arcaneum-68):
- Will extend this RDR's index creation to create both Qdrant collection and MeiliSearch index
- Use index naming conventions from this RDR
- Align filterable attributes with Qdrant payload schema

**Dual Indexing for PDFs** (arcaneum-69):
- Will index PDFs to both Qdrant (vectors) and MeiliSearch (text)
- Use PDF_DOCS_SETTINGS from this RDR
- Parallel indexing workflow

**Dual Indexing for Source Code** (arcaneum-70):
- Will index source code to both datastores
- Use SOURCE_CODE_SETTINGS from this RDR
- Git metadata synchronized across both

**Claude Code Full-Text Search Integration** (arcaneum-71):
- Will expose MeiliSearch search via slash commands
- Use search CLI commands from this RDR
- Complementary to RDR-007 semantic search slash commands

### Migration Path

**If Elasticsearch Becomes Needed**:

1. Evaluate requirements:
   - Data size exceeds single-machine capacity (>100GB)
   - Analytics features required
   - Distributed architecture needed

2. Migration approach:
   - Keep MeiliSearch for instant search
   - Add Elasticsearch for analytics
   - Use both: MeiliSearch for UI, Elasticsearch for reporting

3. Dual deployment:
   - Extend docker-compose.yml with Elasticsearch service
   - Implement Elasticsearch client alongside MeiliSearch
   - Query routing based on use case

**From ChromaDB (if applicable)**:
- Export documents from ChromaDB
- Index text content to MeiliSearch
- Index embeddings to Qdrant (from RDR-002)
- Verify search quality before deprecating ChromaDB

### Comparison to Qdrant (RDR-002)

| Aspect | Qdrant (RDR-002) | MeiliSearch (RDR-008) |
|--------|------------------|----------------------|
| **Purpose** | Vector similarity search | Full-text exact/phrase search |
| **Input** | Embeddings (vectors) | Text documents |
| **Query** | Vector similarity | Text queries with filters |
| **Use Case** | "Find similar authentication patterns" | "Find exact string 'def auth'" |
| **Docker Image** | qdrant/qdrant:v1.16.2 | getmeili/meilisearch:v1.32 |
| **Memory** | Scales with vectors | 96-200MB typical |
| **Setup** | Simple | Simple (slightly simpler) |
| **Client** | qdrant-client + FastEmbed | meilisearch-python >= 0.39.0 |
| **CLI** | `arc search semantic` | `arc search text` |
| **Volumes** | Named (qdrant-arcaneum-*) | Named (meilisearch-arcaneum-*) |

Both services follow same architectural principles: simplicity, Docker-first, CLI integration, complementary use cases.

This RDR provides the complete specification for implementing MeiliSearch full-text search server complementary to Qdrant vector search, maintaining Arcaneum's simplicity and resource efficiency goals.

---

**Revision History**:

- 2025-10-21: Initial RDR created
- 2026-01-14: Updated for current versions and codebase alignment:
  - MeiliSearch v1.24.0 → v1.32.x, meilisearch-python v0.31.0 → v0.39.0
  - Fixed TaskInfo attribute access (task.task_uid not task['taskUid'])
  - Aligned CLI with existing `arc search text` command pattern
  - Standardized on MEILISEARCH_* environment variables
  - Switched to named Docker volumes (matching Qdrant pattern)

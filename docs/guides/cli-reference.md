# Arc CLI Reference

The `arc` command-line tool provides all operations for semantic and full-text search with Qdrant and MeiliSearch.

## Installation & Setup

### Development Mode

```bash
# From repository root
bin/arc <command> [options]
```

### Installed Mode

```bash
# After pip install -e .
arc <command> [options]
```

## Command Overview

### Collection Management

```bash
arc collection create <name> --model <model>  # Create Qdrant collection
arc collection list                          # List all collections
arc collection info <name>                    # Show collection details
arc collection items <name>                   # List indexed files/repos
arc collection delete <name>                  # Delete collection
```

### Indexing Commands

```bash
arc index pdf <path> --collection <name>     # Index PDF files
arc index code <path> --collection <name>   # Index source code
```

### Search Commands

```bash
arc search <query> --collection <name>        # Semantic search
arc search text <query> --index <name>        # Full-text search
```

### Dual Indexing (Qdrant + MeiliSearch)

```bash
arc corpus create <name> --type <type>        # Create dual corpus
arc corpus sync <path> --corpus <name>     # Dual indexing
```

## Collection Management Examples

### Create Collection

```bash
# Create collection with type (model inferred automatically)
arc collection create pdf-docs --type pdf

# Create with specific model (optional override)
arc collection create pdf-docs --type pdf --model stella

# With custom HNSW parameters
arc collection create pdf-docs --type pdf --hnsw-m 16 --hnsw-ef 100

# Store vectors on disk (for large collections)
arc collection create pdf-docs --type pdf --on-disk
```

**Model Inference:** If `--model` is not specified, the model is automatically inferred from `--type`:

- `--type pdf` → `stella` (optimized for documents)
- `--type code` → `jina-code` (optimized for source code)
- `--type markdown` → `stella` (optimized for documents)

### List Collections

```bash
# Simple list (shows name, model, and point count)
arc collection list

# Verbose output (adds collection type and vector details)
arc collection list --verbose

# JSON output for scripting
arc collection list --json
```

**Output includes:**

- **Name**: Collection name
- **Model**: Embedding model used (e.g., stella, jina-code)
- **Points**: Number of indexed documents
- **Type** (verbose): Collection type (pdf, code, markdown)
- **Vectors** (verbose): Vector dimensions and distance metrics

### Collection Info

```bash
# Show collection details (including model and type)
arc collection info pdf-docs

# JSON output
arc collection info pdf-docs --json
```

Shows detailed information including collection type, model, point count, vector configuration, and HNSW index parameters.

### Delete Collection

```bash
# With confirmation prompt
arc collection delete pdf-docs

# Skip confirmation
arc collection delete pdf-docs --confirm
```

### List Collection Items

List all indexed files or repositories in a collection:

```bash
# Human-readable table output
arc collection items MyCode

# JSON output for automation
arc collection items MyCode --json
```

**Features:**

- **Type-aware output**: Different displays for code vs PDF/markdown collections
- **Metadata included**: File sizes, chunk counts, git information
- **Deduplication**: Shows unique files/repos (not individual chunks)
- **Efficient**: Batched retrieval for large collections

**Code Collections Output:**

Shows repositories with git metadata:

| Project | Branch | Commit | Chunks |
|---------|--------|--------|--------|
| my-app | main | a1b2c3d4e5f6 | 1,532 |
| my-lib | develop | f6e5d4c3b2a1 | 847 |

**PDF/Markdown Collections Output:**

Shows files with size information:

| File | Size | Chunks |
|------|------|--------|
| research-paper.pdf | 2.3MB | 42 |
| documentation.md | 15.2KB | 8 |

**JSON Output Format:**

```json
{
  "status": "success",
  "message": "Found 2 items in collection 'MyCode'",
  "data": {
    "collection": "MyCode",
    "type": "code",
    "item_count": 2,
    "items": [
      {
        "git_project_name": "my-app",
        "git_project_identifier": "/path/to/my-app",
        "git_branch": "main",
        "git_commit_hash": "a1b2c3d4e5f6",
        "git_remote_url": "https://github.com/user/my-app",
        "chunk_count": 1532
      }
    ]
  }
}
```

**Use Cases:**

- Verify indexing completed successfully
- Audit collection contents
- Check which branches are indexed
- Count total chunks per file/repo
- Export collection metadata for reporting

## PDF Indexing Examples

### Basic Usage

```bash
# GPU acceleration enabled by default
# Model is automatically retrieved from collection metadata
arc index pdf /path/to/pdfs --collection pdf-docs
```

### With OCR

```bash
arc index pdf /path/to/scanned-pdfs \
  --collection pdf-docs \
  --ocr-language eng
```

### Force Reindex

```bash
arc index pdf /path/to/pdfs \
  --collection pdf-docs \
  --force
```

### Performance Tuning

**Maximum throughput:**

```bash
arc index pdf /path/to/pdfs \
  --collection pdf-docs \
  --embedding-batch-size 500 \
  --process-priority low
```

Note: Larger embedding batches (300-500) improve throughput 10-20%. Process priority is for background indexing.

### GPU Control

```bash
# Default: GPU acceleration enabled (MPS on Apple Silicon, CUDA on NVIDIA)
arc index pdf /path/to/pdfs --collection pdf-docs

# Disable GPU for CPU-only mode
arc index pdf /path/to/pdfs --collection pdf-docs --no-gpu
```

**Note:** The `--model` flag is deprecated. Models are now set at collection
creation time with `arc collection create --type pdf`.

### Debug Mode

```bash
# Show all library warnings (including HuggingFace transformers)
arc index pdf /path/to/pdfs --collection pdf-docs --debug
```

## Model Selection

Available models:

| Model | Dimensions | Best For | Late Chunking |
|-------|------------|----------|---------------|
| `stella` | 1024D | Long documents, general purpose | ✅ |
| `bge` | 1024D | Precision, short documents | ❌ |
| `modernbert` | 768D | Long context, recent content | ✅ |
| `jina` | 768D | Code + text, multilingual | ✅ |

## Common Workflows

### Setup New Project

```bash
# 1. Start services
arc container start

# 2. Create collection (model inferred from type)
arc collection create my-docs --type pdf

# 3. Index documents (model retrieved from collection)
arc index pdf ./documents --collection my-docs
```

### Incremental Updates

```bash
# First run: indexes all PDFs
arc index pdf ./docs --collection my-docs

# Add new files to ./docs/...

# Second run: only indexes new/modified files
arc index pdf ./docs --collection my-docs
```

### JSON Output for Automation

```bash
# List collections
arc collection list --json | jq '.collections[].name'

# Index with JSON output
arc index pdf ./docs --collection my-docs --json > results.json

# Check results
jq '.stats.chunks' results.json
```

## Global Options

Most commands support:

- `--json`: Output JSON format (for scripting)
- `--verbose` / `-v`: Verbose output (show progress and stats, suppress library warnings)
- `--debug`: Debug mode (show all library warnings including transformers)
- `--help`: Show command help

### Indexing Options

Additional options for `arc index` commands:

**Basic Options:**

- `--no-gpu`: Disable GPU acceleration (GPU enabled by default for MPS/CUDA)
- `--workers N`: Number of parallel upload workers (default: 4)
- `--force`: Force reindex all files (skip incremental sync)
- `--offline`: Use cached models only (no network calls)

**Performance Tuning:**

- `--embedding-batch-size N`: Batch size for embedding generation [default: 200]
  - Larger batches (300-500) improve throughput 10-20%
  - Limited benefit from thread parallelism due to embedding lock (see Architecture Notes below)
- `--process-priority low|normal|high`: Process scheduling priority [default: normal]
  - Use `low` for background indexing to avoid blocking foreground tasks

**Note on Parallelism:** File and embedding worker flags were removed because they provided minimal benefit
due to the embedding lock (required for GPU thread-safety). The single-threaded embedding approach with
larger batches is actually more efficient. Use `--embedding-batch-size` for throughput tuning.

### GPU Acceleration

GPU acceleration is **enabled by default** for embedding generation:

- **Apple Silicon**: Uses MPS (Metal Performance Shaders) backend
- **NVIDIA GPUs**: Uses CUDA backend
- **No GPU**: Automatically falls back to CPU

**Performance**: 1.5-3x speedup with GPU for embedding generation.

**Compatible Models** (verified with GPU):

- `stella` - Full MPS support (recommended for PDFs/markdown)
- `jina-code` - Full MPS support (recommended for source code)
- `bge-small` - CoreML support
- `bge-base` - CoreML support

**Disable GPU** if needed (thermal concerns, battery life, etc.):

```bash
arc index pdf /path/to/pdfs --collection docs --no-gpu
```

## Exit Codes

- `0`: Success
- `1`: Error (with error message to stderr)

## Environment Variables

Configure via environment or `.env` file:

```bash
QDRANT_URL=http://localhost:6333
MEILISEARCH_URL=http://localhost:7700
MEILISEARCH_API_KEY=your-api-key
```

## For Claude Code Agents

The `arc` CLI is the entrypoint for all Claude Code plugins and slash commands:

```bash
# These commands are available in Claude Code via slash commands
/create-collection pdf-docs --model stella
/index-pdfs ./documents --collection pdf-docs --model stella
/search "machine learning" --collection pdf-docs
```

See individual slash command files in `/commands/` directory for detailed usage.

## Service Management

### Container Commands

Manage Qdrant and MeiliSearch container services:

```bash
# Start services
arc container start

# Stop services
arc container stop

# Check status
arc container status

# View logs
arc container logs

# Follow logs in real-time
arc container logs --follow

# Restart services
arc container restart

# Reset all data (WARNING: deletes all collections)
arc container reset --confirm
```

**Examples:**

```bash
# Start Qdrant before indexing
arc container start

# Check if healthy
arc container status

# View recent logs
arc container logs --tail 50

# Follow logs for debugging
arc container logs --follow

# Restart if having issues
arc container restart

# Nuclear option: delete everything and start fresh
arc container reset --confirm
```

**Data Location:**

- All data stored in `~/.arcaneum/data/qdrant/`
- Survives container restarts
- Easy to backup

## Configuration & Cache Management

### Cache Commands

Manage embedding model cache:

```bash
# Show cache location and sizes
arc config show-cache-dir

# Clear model cache to free space
arc config clear-cache --confirm
```

**Examples:**

```bash
# Check where models are stored
arc config show-cache-dir
# Output:
#   Arcaneum directories:
#     Root:   /Users/you/.arcaneum
#     Models: /Users/you/.arcaneum/models
#     Data:   /Users/you/.arcaneum/data
#     Models size: 2.5 GB
#     Data size: 266.8 MB

# Clear cache if running low on disk space
arc config clear-cache --confirm
```

**Cache Location:**

- Models stored in `~/.arcaneum/models/`
- Auto-downloaded on first use
- Shared across all arc commands
- ~1-2GB per model

## Troubleshooting

### Command Not Found

```bash
# Development mode
bin/arc --help

# After install
pip install -e .
arc --help
```

### Qdrant Connection Error

```bash
# Check if running
arc container status

# Start if needed
arc container start
```

### Permission Denied

```bash
chmod +x bin/arc
```

## More Documentation

- [PDF Indexing Guide](pdf-indexing.md) - Detailed PDF indexing documentation
- [RDR Directory](../docs/rdr/) - Technical specifications for all features

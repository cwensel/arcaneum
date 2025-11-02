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
arc collection delete <name>                  # Delete collection
```

### Indexing Commands

```bash
arc index pdfs <path> --collection <name>     # Index PDF files
arc index source <path> --collection <name>   # Index source code
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
# Create collection with single model
arc collection create pdf-docs --model stella

# With custom HNSW parameters
arc collection create pdf-docs --model stella --hnsw-m 16 --hnsw-ef 100

# Store vectors on disk (for large collections)
arc collection create pdf-docs --model stella --on-disk
```

### List Collections

```bash
# Simple list
arc collection list

# Verbose output
arc collection list --verbose

# JSON output for scripting
arc collection list --json
```

### Collection Info

```bash
arc collection info pdf-docs
arc collection info pdf-docs --json
```

### Delete Collection

```bash
# With confirmation prompt
arc collection delete pdf-docs

# Skip confirmation
arc collection delete pdf-docs --confirm
```

## PDF Indexing Examples

### Basic Usage

```bash
# GPU acceleration enabled by default
arc index pdfs /path/to/pdfs --collection pdf-docs --model stella
```

### With OCR

```bash
arc index pdfs /path/to/scanned-pdfs \
  --collection pdf-docs \
  --model stella \
  --ocr-language eng
```

### Force Reindex

```bash
arc index pdfs /path/to/pdfs \
  --collection pdf-docs \
  --model stella \
  --force
```

### Parallel Processing

```bash
arc index pdfs /path/to/pdfs \
  --collection pdf-docs \
  --model stella \
  --workers 8
```

### Performance Tuning

**Maximum performance (use all CPU cores):**

```bash
arc index pdfs /path/to/pdfs \
  --collection pdf-docs \
  --model stella \
  --embedding-worker-mult 1.0 \
  --embedding-batch-size 500 \
  --process-priority low
```

**Conservative (25% CPU, good for background):**

```bash
arc index pdfs /path/to/pdfs \
  --collection pdf-docs \
  --model stella \
  --embedding-worker-mult 0.25 \
  --process-priority low
```

**Absolute control (set exact worker count):**

```bash
arc index pdfs /path/to/pdfs \
  --collection pdf-docs \
  --model stella \
  --embedding-workers 8 \
  --embedding-batch-size 300
```

### GPU Control

```bash
# Default: GPU acceleration enabled (MPS on Apple Silicon, CUDA on NVIDIA)
arc index pdfs /path/to/pdfs --collection pdf-docs --model stella

# Disable GPU for CPU-only mode
arc index pdfs /path/to/pdfs --collection pdf-docs --model stella --no-gpu
```

### Debug Mode

```bash
# Show all library warnings (including HuggingFace transformers)
arc index pdfs /path/to/pdfs --collection pdf-docs --model stella --debug
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

# 2. Create collection
arc collection create my-docs --model stella --type pdf

# 3. Index documents
arc index pdfs ./documents --collection my-docs --model stella
```

### Incremental Updates

```bash
# First run: indexes all PDFs
arc index pdfs ./docs --collection my-docs --model stella

# Add new files to ./docs/...

# Second run: only indexes new/modified files
arc index pdfs ./docs --collection my-docs --model stella
```

### JSON Output for Automation

```bash
# List collections
arc collection list --json | jq '.collections[].name'

# Index with JSON output
arc index pdfs ./docs --collection my-docs --model stella --json > results.json

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

- `--file-workers N`: Absolute number of files to process in parallel (overrides multiplier) [default: 1]
  - PDF: Number of PDF files to process in parallel
  - Source: Number of source files to process in parallel within each repo
  - Markdown: Number of markdown files to process in parallel
- `--file-worker-mult FLOAT`: Multiplier of cpu_count for file processing (overridden by --file-workers)
- `--embedding-workers N`: Absolute number of embedding workers (overrides multiplier)
- `--embedding-worker-mult FLOAT`: Multiplier of cpu_count for embedding workers [default: 0.5]
- `--embedding-batch-size N`: Batch size for embedding generation [default: 200]
- `--process-priority low|normal|high`: Process scheduling priority [default: normal]
- `--max-perf`: Preset that sets embedding-worker-mult=1.0, batch-size=500, priority=low (does NOT change file workers)

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
arc index pdfs /path/to/pdfs --collection docs --no-gpu
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

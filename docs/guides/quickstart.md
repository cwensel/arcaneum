# Arcaneum Quick Start Guide

Get started with Arcaneum in 5-10 minutes. This guide walks you through installation, setup, and your first search.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.12+** - Check with `python --version`
- **Git** - For cloning the repository
- **Docker** - For running Qdrant and MeiliSearch
  - [Install Docker Desktop](https://docs.docker.com/get-docker/) (Mac/Windows)
  - Or Docker Engine (Linux)

### Verify Prerequisites

```bash
python --version  # Should show 3.12 or higher
git --version
docker --version
docker ps  # Verify Docker is running
```

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/cwensel/arcaneum
cd arcaneum
```

### 2. Install Arcaneum

```bash
pip install -e .
```

This installs Arcaneum in development mode and creates the `arc` command.

### 3. Verify Installation

```bash
arc doctor
```

The `doctor` command checks your setup and guides you through any missing prerequisites.

## Your First Search

Let's index some code and perform your first search.

### 1. Start Services

Start the search services (Qdrant for semantic search, MeiliSearch for full-text):

```bash
arc container start
```

You should see:

```text
[INFO] Starting container services...
Qdrant started successfully
  REST API: http://localhost:6333
  Dashboard: http://localhost:6333/dashboard
MeiliSearch started successfully
  HTTP API: http://localhost:7700

[INFO] Data directory: /Users/you/.local/share/arcaneum
```

Note: MeiliSearch API key is auto-generated and stored in `~/.config/arcaneum/meilisearch.key`.

### 2. Create a Corpus

Create a corpus for code (indexes to both Qdrant and MeiliSearch):

```bash
arc corpus create MyCode --type code
```

### 3. Sync Your Code

Sync a directory of source code (indexes to both search systems):

```bash
arc corpus sync MyCode ~/my-project
```

Example output:

```text
Source Code Indexing Configuration
  Corpus: MyCode (type: code)
  Embedding: jinaai/jina-code-embeddings-0.5b
  Vector: jina-code-0.5b

Git discovery completed: 1 repos, 3 branches
Indexed 247 files → 1,532 chunks (Qdrant + MeiliSearch)
```

### 4. Search

Now search your indexed code with semantic or full-text queries:

```bash
# Semantic search (conceptual similarity)
arc search semantic "authentication logic" --corpus MyCode --limit 5

# Full-text search (exact matches)
arc search text "def authenticate" --corpus MyCode
```

You'll see code chunks ranked by relevance!

## Common Workflows

### Indexing PDFs (Recommended: Corpus)

```bash
# Create corpus for PDFs (indexes to both Qdrant and MeiliSearch)
arc corpus create MyDocs --type pdf

# Sync PDFs (with OCR for scanned documents)
arc corpus sync MyDocs ~/Documents/papers

# Sync multiple directories at once
arc corpus sync MyDocs ~/Documents/papers ~/Documents/specs

# Semantic search (conceptual matches)
arc search semantic "machine learning techniques" --corpus MyDocs

# Full-text search (exact phrases)
arc search text '"neural network"' --corpus MyDocs
```

**Tip:** Using `arc corpus` gives you both semantic and full-text search in one workflow.
See [PDF Indexing Guide](pdf-indexing.md) for advanced options.

### Multi-Branch Code Indexing

Arcaneum automatically indexes all branches of git repositories:

```bash
# Create code corpus and sync
arc corpus create MyCode --type code
arc corpus sync MyCode ~/projects/my-app

# Search finds code across all branches
arc search semantic "payment processing" --corpus MyCode
arc search text "async def process_payment" --corpus MyCode
```

### Checking Status

```bash
# List all corpora (shows both Qdrant and MeiliSearch status)
arc corpus list

# List what's indexed with parity status
arc corpus items MyCode

# Check corpus health
arc corpus verify MyCode

# Check container status
arc container status

# View container logs
arc container logs
```

### Single-System Indexing (Advanced)

If you only need one type of search, use collections or indexes directly:

```bash
# Semantic search only (Qdrant)
arc collection create MyCode --type code
arc index code ~/project --collection MyCode
arc search semantic "query" --corpus MyCode

# Full-text search only (MeiliSearch)
arc indexes create MyDocs --type pdf
arc index text pdf ~/docs --index MyDocs
arc search text "query" --corpus MyDocs
```

## Configuration

### Cache Management

Arcaneum stores embedding models in XDG-compliant locations:

```bash
# View cache location and size
arc config show-cache-dir

# Clear cache to free space
arc config clear-cache --confirm
```

### Data Location

Arcaneum stores data in XDG-compliant locations:

```text
~/.cache/arcaneum/models/        # Embedding models (~1-2GB per model)
~/.local/share/arcaneum/         # Local databases and indexed content
~/.config/arcaneum/              # Configuration files (e.g., MeiliSearch key)
```

**Vector Database (Docker):**

Qdrant and MeiliSearch use Docker named volumes for data persistence:

```text
qdrant-arcaneum-storage    # Qdrant vector database storage
qdrant-arcaneum-snapshots  # Qdrant backup snapshots
meilisearch-arcaneum-data  # MeiliSearch full-text index
```

**Benefits:**

- XDG-compliant paths (follows Linux/macOS standards)
- Reliable data persistence across container restarts
- Easy backup via Qdrant snapshots

## Troubleshooting

### Docker Not Running

If `arc container start` fails:

```bash
# Check if Docker is running
docker ps

# Start Docker Desktop (Mac/Windows)
# Or: sudo systemctl start docker (Linux)
```

### SSL Certificate Errors (Corporate Networks)

If behind a corporate proxy with SSL issues:

**Option 1: Offline Mode (Recommended)**

Pre-download models on a machine with internet, then use offline mode:

```bash
# Set in your ~/.bashrc or ~/.zshrc
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Then run arc normally
arc index code ~/code --collection MyCode
```

**Option 2: Disable SSL Verification**

```bash
# Set in your ~/.bashrc or ~/.zshrc (use on trusted networks only)
export PYTHONHTTPSVERIFY=0
export REQUESTS_CA_BUNDLE=""
export CURL_CA_BUNDLE=""
```

See [docs/testing/offline-mode.md](../testing/offline-mode.md) for complete corporate network setup.

### Models Downloading Slowly

First-time indexing downloads embedding models (~1-2GB). This is normal and only happens once.

```bash
# Check download progress
arc config show-cache-dir

# Models are cached in ~/.arcaneum/models/
```

### Collection Already Exists

If you see "Collection already exists":

```bash
# List collections
arc collection list

# Delete and recreate
arc collection delete MyCode --confirm
arc collection create MyCode --model jina-code-0.5b --type code
```

### No Search Results

If searches return no results:

```bash
# Verify collection has data
arc collection info MyCode

# Check if indexing completed successfully
# Re-index with --force flag
arc index code ~/code --collection MyCode --force
```

### GPU Memory Errors (Apple Silicon)

If you see MPS out-of-memory errors on Apple Silicon Macs:

```text
ERROR: MPS backend out of memory (MPS allocated: X GiB, ...)
```

**Quick fixes:**

1. **Allow unlimited GPU memory** (recommended for most cases):

   ```bash
   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
   arc index code ~/project --collection MyCode
   ```

2. **Use smaller batch size:**

   ```bash
   arc index code ~/project --collection MyCode --embedding-batch-size 100
   ```

3. **Force CPU mode** (slower but reliable):

   ```bash
   arc index code ~/project --collection MyCode --no-gpu
   ```

**Permanent fix:** Add to your `~/.zshrc` or `~/.bashrc`:

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

**Why this happens:** Apple Silicon uses unified memory shared between CPU and GPU.
When other apps use significant memory, PyTorch's default safety limits can trigger
OOM errors even when memory is technically available.

## Claude Code Plugin

After completing the installation above, you can use Arcaneum commands directly in Claude Code.

### Plugin Installation

In Claude Code, add the local marketplace and install the plugin:

```text
/plugin marketplace add /path/to/arcaneum
/plugin install arc
```

Then restart Claude Code to activate the plugin.

### Using Commands in Claude Code

All `arc` commands are available as slash commands:

```text
/arc:doctor
/arc:corpus create MyCode --type code
/arc:corpus sync MyCode ~/my-project
/arc:search semantic "authentication logic" --corpus MyCode
/arc:search text "def authenticate" --corpus MyCode
```

The plugin provides the same functionality as the CLI, but integrated into your Claude Code workflow.

## Performance Tuning

Arcaneum provides granular control over indexing performance for different workload scenarios.

### Quick Start: Use Presets

For better throughput on batch workloads:

```bash
# PDF indexing with larger batch size
arc index pdf ~/Documents --collection MyDocs --embedding-batch-size 500

# Source code indexing with larger batch size
arc index code ~/projects --collection MyCode --embedding-batch-size 500
```

Note: Worker parallelism flags were removed due to embedding lock serialization.
Use --embedding-batch-size for throughput tuning.

### Advanced: Granular Control

For fine-tuned control, use individual flags:

```bash
# Conservative: smaller batches, low priority
arc index pdf ~/docs --collection MyDocs \
  --embedding-batch-size 100 \
  --process-priority low

# Balanced: default settings
arc index pdf ~/docs --collection MyDocs

# Maximum throughput: large batches
arc index pdf ~/docs --collection MyDocs \
  --embedding-batch-size 500 \
  --process-priority low
```

**Available Options:**

- `--embedding-batch-size N`: Batch size for embeddings (default: 200)
- `--process-priority low|normal|high`: OS scheduling priority

See the [CLI Reference](cli-reference.md) for complete performance tuning documentation.

## Next Steps

### Learn More

- **[CLI Reference](cli-reference.md)** - Complete command documentation
- **[PDF Indexing Guide](pdf-indexing.md)** - Advanced PDF indexing with OCR
- **[README](../../README.md)** - Project overview and plugin documentation

### Advanced Features

- **Multiple Models**: Index with different embedding models for different use cases
- **Incremental Indexing**: Re-run `arc index code` to update only changed files
- **Branch Tracking**: Automatically track new branches in git repositories
- **Filter Searches**: Use metadata filters to narrow results

### Get Help

- Run `arc --help` for command overview
- Run `arc <command> --help` for detailed command help
- Check [docs/](../) for detailed documentation
- Report issues on GitHub

## Quick Reference

```bash
# Service Management
arc container start          # Start services
arc container stop           # Stop services
arc container status         # Check status

# Corpus (Recommended - Dual Indexing)
arc corpus create NAME --type TYPE       # pdf, code, or markdown
arc corpus list                          # List all corpora
arc corpus sync NAME PATH [PATH...]      # Sync directories to both systems
arc corpus items NAME                    # List items with parity status
arc corpus verify NAME                   # Check corpus health
arc corpus delete NAME --confirm

# Searching
arc search semantic "query" --corpus NAME    # Semantic search (Qdrant)
arc search text "query" --corpus NAME        # Full-text search (MeiliSearch)

# Collections (Qdrant Only - Advanced)
arc collection create NAME --type TYPE
arc collection list
arc collection items NAME
arc index code PATH --collection NAME
arc index pdf PATH --collection NAME

# Indexes (MeiliSearch Only - Advanced)
arc indexes create NAME --type TYPE
arc indexes list
arc index text pdf PATH --index NAME

# Configuration
arc config show-cache-dir    # Show cache location
arc doctor                   # Verify setup
```

## Tips for New Users

1. **Start with `arc doctor`** - It checks your setup and provides guidance
2. **Use `arc container status`** - Verify services are running before indexing
3. **Index small directories first** - Test with a small codebase before indexing large projects
4. **Watch the dashboard** - Visit <http://localhost:6333/dashboard> to see Qdrant's UI
5. **Check collection info** - Use `arc collection info` to verify indexing completed
6. **Use --verbose** - Add `-v` flag to commands for detailed progress

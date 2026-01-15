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

### 2. Create a Collection

Create a collection for code (model automatically inferred from type):

```bash
arc collection create MyCode --type code
```

### 3. Index Your Code

Index a directory of source code:

```bash
arc index code ~/my-project --collection MyCode
```

Example output:

```text
Source Code Indexing Configuration
  Collection: MyCode (type: code)
  Embedding: jinaai/jina-code-embeddings-0.5b
  Vector: jina-code-0.5b

Git discovery completed: 1 repos, 3 branches
Indexed 247 files → 1,532 chunks
```

### 4. Search

Now search your indexed code:

```bash
arc search semantic "authentication logic" --collection MyCode --limit 5
```

You'll see semantically similar code chunks ranked by relevance!

## Common Workflows

### Indexing PDFs

```bash
# Create PDF collection (model inferred from type)
arc collection create MyDocs --type pdf

# Index PDFs (with OCR for scanned documents)
arc index pdf ~/Documents/papers --collection MyDocs

# Index with larger batch size for better throughput
arc index pdf ~/Documents/papers --collection MyDocs --embedding-batch-size 500

# Search PDFs
arc search semantic "machine learning techniques" --collection MyDocs
```

### Multi-Branch Code Indexing

Arcaneum automatically indexes all branches of git repositories:

```bash
# Index with git-aware chunking (uses the model from collection)
arc index code ~/projects/my-app --collection MyCode

# Search finds code across all branches
arc search semantic "payment processing" --collection MyCode
```

### Checking Status

```bash
# List all collections
arc collection list

# Show collection details
arc collection info MyCode

# List what's indexed in a collection
arc collection items MyCode

# Check container status
arc container status

# View container logs
arc container logs
```

## Configuration

### Cache Management

Arcaneum stores embedding models in `~/.arcaneum/models/`:

```bash
# View cache location and size
arc config show-cache-dir

# Clear cache to free space
arc config clear-cache --confirm
```

### Data Location

All data is stored in `~/.arcaneum/`:

```text
~/.arcaneum/
├── models/              # Embedding models (auto-downloaded)
├── data/
│   ├── qdrant/         # Vector database storage
│   └── qdrant_snapshots/  # Backups
```

**Benefits:**

- Easy to backup (just backup `~/.arcaneum/`)
- Easy to find and manage
- Survives container restarts

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
/doctor
/collection create MyCode --model jina-code-0.5b --type code
/index code ~/my-project --collection MyCode
/search semantic "authentication logic" --collection MyCode
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

# Collections
arc collection list                    # List all collections
arc collection create NAME --model MODEL --type TYPE
arc collection items NAME              # List indexed files/repos
arc collection delete NAME --confirm

# Indexing
arc index code PATH --collection NAME
arc index pdf PATH --collection NAME

# Searching
arc search semantic "query" --collection NAME
arc search semantic "query" --collection NAME --limit 20

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

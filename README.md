# Arcaneum

CLI tools and Claude Code plugins for semantic search across Qdrant vector databases.

## Overview

Arcaneum provides semantic search for documents using Qdrant vector embeddings. The system supports PDF
documents and source code with git-aware, AST-based chunking.

**Currently Available:** Semantic search with Qdrant

**Planned:** MeiliSearch integration for full-text search (RDR-008 through RDR-012)

## Features

### Search Capabilities

- **Semantic Search (Qdrant)**: Find conceptually similar content using vector embeddings
- **Full-Text Search**: Planned - MeiliSearch integration for exact phrase matching (RDR-008 through RDR-012)

### Indexing

- **PDF Indexing**: OCR support for scanned documents, page-level metadata, parallel processing
- **Source Code Indexing**: Git-aware with AST chunking, multi-branch support, 165+ languages
- **Markdown Indexing**: YAML frontmatter extraction, semantic chunking, incremental sync
- **Dual Indexing**: Single command to index to both search engines
- **Performance Tuning**: Granular control over workers, batch sizes, and process priority with `--max-perf` preset

### Multiple Embedding Models

- **stella** (1024D) - High-quality general purpose embeddings, optimized for PDFs and documents
- **jina-code** (768D) - Code-specific embeddings, optimized for source code analysis
- **bge-large** (1024D) - BGE large embeddings, balanced performance
- **jina-v3** (1024D) - Multilingual embeddings with extended 8K context
- **bge-base** (768D) - BGE base embeddings, balanced performance and speed
- **bge-small** (384D) - BGE small embeddings, fastest for size-constrained scenarios

See `arc models list` for complete model information and recommendations.

### GPU Acceleration

- **Enabled by default** for 1.5-3x faster embedding generation
- Supports Apple Silicon (MPS) and NVIDIA GPUs (CUDA)
- Automatic CPU fallback when GPU unavailable
- Use `--no-gpu` flag to force CPU-only mode

### CLI-First Design

- All operations via command-line interface
- JSON output mode for automation
- Structured error messages with exit codes
- Python >= 3.12 required

### Claude Code Integration

- Slash commands for all operations (`/arc:search`, `/arc:index`, `/arc:collection`, etc.)
- Discoverable via `/help` or `/commands` in Claude Code
- No MCP overhead - direct CLI execution

## Quick Start

Get started with Arcaneum in just a few commands:

```bash
# 1. Install
git clone https://github.com/cwensel/arcaneum
cd arcaneum
pip install -e .

# 2. Install Claude Code plugin (optional)
# In Claude Code:
# /plugin marketplace add /path/to/arcaneum
# /plugin install arc
# (then restart Claude Code)

# 3. Verify and start services
arc doctor
arc container start

# 4. Index and search your code
arc collection create MyCode --model jina-code --type code
arc index code ~/my-project --collection MyCode
arc search semantic "authentication logic" --collection MyCode
```

**First time?** Run `arc doctor` to check prerequisites and get setup guidance.

👉 **[Full Quick Start Guide](docs/guides/quickstart.md)** - Detailed walkthrough with troubleshooting

## Common Workflows

### Search Your Code

```bash
# Create a code collection
arc collection create MyCode --model jina-code --type code

# Index your project (git-aware, multi-branch)
arc index code ~/projects/my-app --collection MyCode

# Search semantically
arc search semantic "authentication middleware" --collection MyCode --limit 10
```

### Search PDFs

```bash
# Create a PDF collection
arc collection create MyDocs --model stella --type pdf

# Index PDFs (with OCR for scanned documents)
arc index pdf ~/Documents/papers --collection MyDocs

# Index with maximum performance (uses all CPU cores)
arc index pdf ~/Documents/papers --collection MyDocs --max-perf

# Search for concepts
arc search semantic "neural network architectures" --collection MyDocs
```

### Index Markdown Files

```bash
# Index documentation or notes
arc collection create Notes --model stella --type markdown
arc index markdown ~/obsidian-vault --collection Notes

# With custom options
arc index markdown ~/docs --collection Docs \
  --exclude ".obsidian,templates" \
  --chunk-size 512 \
  --no-recursive

# Search your notes
arc search semantic "project planning" --collection Notes
```

**Features:**

- YAML frontmatter extraction (title, tags, category, etc.)
- Semantic chunking preserving document structure
- Incremental sync (SHA256 content hashing)
- Custom exclude patterns
- Supports .md, .markdown, .mdown extensions

### Store Agent Memory

```bash
# Store agent-generated content (for Claude skills/agents)
arc collection create Memory --model stella --type markdown

# Store from file with metadata
arc store analysis.md --collection Memory \
  --title "Security Analysis" \
  --category "security" \
  --tags "audit,findings"

# Store from stdin (agent workflow)
echo "# Research\n\nFindings..." | arc store - --collection Memory

# Content persisted to: ~/.arcaneum/agent-memory/{collection}/
# Enables re-indexing and full-text retrieval
```

**Use Case:** Designed for AI agents to store research, analysis, and synthesized
information with rich metadata. Content is automatically persisted for durability.

### Manage Services

```bash
arc container start    # Start Qdrant
arc container status   # Check health
arc container logs     # View logs
arc container stop     # Stop services
```

## Installation

### Prerequisites

- **Python 3.12+** - Check with `python --version`
- **Git** - For cloning the repository
- **Docker** - [Install Docker Desktop](https://docs.docker.com/get-docker/) (Mac/Windows) or Docker Engine (Linux)

### Install

```bash
git clone https://github.com/cwensel/arcaneum
cd arcaneum
pip install -e .
```

### Verify Setup

```bash
arc doctor
```

The `doctor` command checks your environment and guides you through any issues.

👉 **[Full Installation Guide](docs/guides/quickstart.md)** - Complete walkthrough with troubleshooting

### Data Storage

Arcaneum stores data across two locations:

**Embedding Models:**

```text
~/.arcaneum/models/     # Auto-downloaded, ~1-2GB per model
```

**Vector Database:**

Qdrant uses Docker named volumes for data persistence and safety:

```text
qdrant-arcaneum-storage    # Main vector database storage
qdrant-arcaneum-snapshots  # Backup snapshots
```

Named volumes store data on a Linux ext4 filesystem inside Docker, providing better
reliability and performance than bind mounts.

**Migration Note:** If you're upgrading from bind mounts to named volumes, see
**[Qdrant Migration Guide](docs/guides/qdrant-migration.md)** for detailed migration
steps.

**Benefits:**

- Reliable data persistence across container restarts
- Better performance compared to bind mounts
- Easy backup via Qdrant snapshots
- Native Linux filesystem (ext4) for data safety

### Corporate Networks

Behind a VPN with SSL issues? See **[Corporate Network Setup](docs/testing/offline-mode.md)** for:

- Offline mode setup
- SSL certificate workarounds
- Model pre-downloading

## Claude Code Plugin

### Installation

**1. Install the Python package** (required first):

```bash
git clone https://github.com/cwensel/arcaneum
cd arcaneum
pip install -e .
```

**2. Install Claude Code plugin**:

In Claude Code, add the local marketplace and install the plugin:

```text
/plugin marketplace add /path/to/arcaneum
/plugin install arc@arcaneum-marketplace
```

Then restart Claude Code to activate the plugin.

### Available Commands

All commands use the `arc:` namespace prefix:

- `/arc:collection` - Manage Qdrant collections (create, list, info, delete)
- `/arc:config` - Manage configuration and cache
- `/arc:container` - Manage Docker containers (start, stop, status, logs)
- `/arc:corpus` - Manage dual-index corpora (vector + full-text)
- `/arc:doctor` - Verify setup and prerequisites
- `/arc:index` - Index PDF, code, or markdown into collections
- `/arc:models` - List available embedding models
- `/arc:search` - Semantic or full-text search

**Usage Examples:**

```text
/arc:collection create my-docs --model stella --type pdf
/arc:index pdf ~/Documents --collection my-docs
/arc:search "example query" --collection my-docs
/arc:models list
```

Use `/help` in Claude Code to see all available commands or `/arc:doctor` to check your setup.

**For Developers:** See **[Claude Code Plugin Testing Guide](docs/guides/claude-code-plugin.md)** for local testing instructions.

## Development

### Architecture Principles

1. **CLI-First**: All functionality as CLI tools (RDR-001, RDR-006)
2. **Slash Commands**: Thin wrappers calling CLI via Bash (RDR-006)
3. **No MCP (v1)**: Avoid MCP overhead, use direct CLI execution (RDR-006)
4. **Local Docker**: Databases run locally with volume persistence (RDR-002, RDR-008)
5. **RDR-Based Planning**: Detailed design before implementation (docs/rdr/)

### Implementation Status

- ✅ **RDR-001**: Project structure (COMPLETED)
- ✅ **RDR-002**: Qdrant server setup (COMPLETED)
- ✅ **RDR-003**: Collection management (COMPLETED)
- ✅ **RDR-004**: PDF bulk indexing (COMPLETED)
- ✅ **RDR-005**: Source code indexing (COMPLETED)
- ✅ **RDR-006**: Claude Code integration (COMPLETED)
- ✅ **RDR-007**: Semantic search (COMPLETED)
- ⏱️ **RDR-008**: MeiliSearch setup (PENDING)
- ⏱️ **RDR-009**: Dual indexing strategy (PENDING)
- ⏱️ **RDR-010**: PDF full-text indexing (PENDING)
- ⏱️ **RDR-011**: Source code full-text indexing (PENDING)
- ⏱️ **RDR-012**: Full-text search integration (PENDING)
- ✅ **RDR-014**: Markdown indexing (COMPLETED)

### Testing

Tests will be added as features are implemented.

```bash
# Run tests (after implementation)
pytest tests/

# Run with coverage
pytest --cov=arcaneum tests/
```

## Documentation

### User Guides

- **[Quick Start Guide](docs/guides/quickstart.md)** - Installation, setup, and your first search
- **[CLI Reference](docs/guides/cli-reference.md)** - Complete command documentation and options
- **[PDF Indexing Guide](docs/guides/pdf-indexing.md)** - Advanced PDF indexing with OCR
  support, performance tuning, and troubleshooting
- **[Qdrant Migration Guide](docs/guides/qdrant-migration.md)** - Migrate from bind mounts to Docker named volumes
- **[Corporate Network Setup](docs/testing/offline-mode.md)** - Setup for VPN, SSL certificates, and offline mode
- **[Claude Code Plugin Testing Guide](docs/guides/claude-code-plugin.md)** - Local development and testing for plugin developers

### Development

- **[RDR Process](docs/rdr/README.md)** - Recommendation Data Records workflow for complex features
- **[Individual RDRs](docs/rdr/)** - Technical specifications and design decisions for each feature
- **[Slash Commands](commands/)** - Claude Code plugin command implementations

## Contributing

We welcome contributions! See **[CONTRIBUTING.md](CONTRIBUTING.md)** for detailed guidelines on:

- Development setup and workflow
- Using beads for task tracking
- When to create RDRs
- Code and documentation standards
- Pull request process

**Quick Start for Contributors:**

1. Read `docs/rdr/README.md` for RDR-based development workflow
2. Create an RDR for complex features before implementation
3. Follow CLI-first architecture pattern
4. Add tests for new functionality
5. Update this README with implementation status

## License

MIT - See LICENSE file for details

## Acknowledgments

- Inspired by [Beads](https://github.com/steveyegge/beads) for Claude Code plugin patterns
- Built on [Qdrant](https://qdrant.tech/) (MeiliSearch integration planned)

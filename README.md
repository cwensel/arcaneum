# Arcaneum

CLI tools and Claude Code plugins for semantic and full-text search.

## Overview

Arcaneum helps you discover and understand project dependencies, documentation, and reference implementations.
By indexing libraries, frameworks, and technical papers, you can semantically search for patterns, APIs,
and concepts when building new projects. Works especially well with the
[RDR (Recommendation Data Record)](https://github.com/cwensel/rdr) model for AI-assisted development planning.

The system supports PDF documents and source code with git-aware, AST-based chunking.

**Currently Available:**

- Semantic search with Qdrant (vector embeddings)
- Full-text search with MeiliSearch (exact phrase matching)
- Dual indexing workflow for comprehensive search

## Features

### Search Capabilities

- **Semantic Search (Qdrant)**: Find conceptually similar content using vector embeddings
- **Full-Text Search (MeiliSearch)**: Exact phrase matching, keyword search, and typo-tolerant queries

### Indexing

- **PDF Indexing**: OCR support for scanned documents, page-level metadata, parallel processing
- **Source Code Indexing**: Git-aware with AST chunking, multi-branch support, 165+ languages
- **Markdown Indexing**: YAML frontmatter extraction, semantic chunking, incremental sync
- **Dual Indexing**: Single command to index to both search engines
- **Performance Tuning**: Granular control over workers, batch sizes, and process priority via
  `--embedding-batch-size` and `--process-priority` flags

### Multiple Embedding Models

- **stella** (1024D) - High-quality general purpose embeddings, optimized for PDFs and documents
- **jina-code-0.5b** (896D) - **RECOMMENDED** for code - SOTA Sept 2025, 32K context, fast (default)
- **jina-code-1.5b** (1536D) - Highest quality code embeddings, SOTA Sept 2025
- **codesage-large** (1024D) - CodeSage V2, 9 programming languages
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

# 4. Create a corpus and sync content (indexes to both Qdrant and MeiliSearch)
arc corpus create Frameworks --type code
arc corpus sync Frameworks ~/libs/fastapi ~/libs/sqlalchemy

# 5. Search with semantic or full-text queries
arc search semantic "dependency injection pattern" --corpus Frameworks
arc search text "async def" --corpus Frameworks
```

**First time?** Run `arc doctor` to check prerequisites and get setup guidance.

👉 **[Full Quick Start Guide](docs/guides/quickstart.md)** - Detailed walkthrough with troubleshooting

## Quick Reference

```bash
# Service Management
arc container start          # Start Qdrant and MeiliSearch
arc container status         # Check service health
arc doctor                   # Verify setup

# Corpus (Recommended - Dual Indexing to Both Systems)
arc corpus create NAME --type TYPE       # pdf, code, or markdown
arc corpus list                          # List all corpora
arc corpus sync NAME PATH [PATH...]      # Sync one or more directories
arc corpus items NAME                    # List items with parity status
arc corpus parity NAME                   # Check/restore parity between systems
arc corpus delete NAME                   # Delete both collection and index

# Search (Works with corpus, collection, or index)
arc search semantic "query" --corpus NAME              # Conceptual similarity
arc search semantic "query" --corpus N1 --corpus N2   # Multi-corpus
arc search text "query" --corpus NAME                  # Exact phrase matching

# Collections (Qdrant Only - Semantic Search)
arc collection create NAME --type TYPE   # When you only need semantic search
arc collection list
arc collection items NAME
arc index pdf PATH --collection NAME
arc index code PATH --collection NAME

# Indexes (MeiliSearch Only - Full-Text Search)
arc indexes create NAME --type TYPE      # When you only need full-text search
arc indexes list
arc index text pdf PATH --index NAME
arc index text code PATH --index NAME
```

## Common Workflows

### Search Dependencies and Libraries (Recommended)

```bash
# Create a corpus for framework source code
arc corpus create Frameworks --type code

# Sync framework directories (indexes to both Qdrant and MeiliSearch)
arc corpus sync Frameworks ~/libs/fastapi ~/libs/sqlalchemy

# List what's indexed
arc corpus items Frameworks

# Semantic search for patterns and APIs
arc search semantic "dependency injection pattern" --corpus Frameworks --limit 10

# Full-text search for exact code
arc search text "async def create_app" --corpus Frameworks
```

### Search Technical Documentation

```bash
# Create a corpus for PDF documents
arc corpus create Papers --type pdf

# Sync documentation directories
arc corpus sync Papers ~/Documents/papers ~/Documents/specs

# Semantic search for concepts
arc search semantic "distributed consensus algorithms" --corpus Papers

# Full-text search for exact phrases
arc search text '"rate limiting"' --corpus Papers
```

### Index Markdown Files

```bash
# Create a corpus for notes and documentation
arc corpus create Notes --type markdown

# Sync your notes directory
arc corpus sync Notes ~/obsidian-vault

# Semantic search
arc search semantic "project planning" --corpus Notes

# Full-text search
arc search text "meeting notes" --corpus Notes
```

**Features:**

- YAML frontmatter extraction (title, tags, category, etc.)
- Semantic chunking preserving document structure
- Incremental sync (SHA256 content hashing)
- Custom exclude patterns
- Supports .md, .markdown, .mdown extensions

### Single-System Indexing (Advanced)

Use collections or indexes directly when you only need one type of search:

```bash
# Semantic search only (Qdrant collection)
arc collection create MyCollection --type code
arc index code ~/project --collection MyCollection
arc search semantic "query" --corpus MyCollection

# Full-text search only (MeiliSearch index)
arc indexes create MyIndex --type pdf
arc index text pdf ~/docs --index MyIndex
arc search text "query" --corpus MyIndex
```

### Store Agent Memory

```bash
# Create a corpus for agent-generated content
arc corpus create Memory --type markdown

# Store from file with metadata
arc store analysis.md --corpus Memory \
  --title "Security Analysis" \
  --category "security" \
  --tags "audit,findings"

# Store from stdin (agent workflow)
echo "# Research\n\nFindings..." | arc store - --corpus Memory

# Search agent memory
arc search semantic "security vulnerabilities" --corpus Memory
arc search text "SQL injection" --corpus Memory

# Content persisted to: ~/.local/share/arcaneum/agent-memory/{corpus}/
# Enables re-indexing and full-text retrieval
```

**Use Case:** Designed for AI agents to store research, analysis, and synthesized
information with rich metadata. Content is automatically persisted for durability.

### Manage Services

```bash
arc container start    # Start Qdrant and MeiliSearch
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

Arcaneum stores data in XDG-compliant locations:

**Cache (Re-downloadable):**

```text
~/.cache/arcaneum/models/     # Embedding models, ~1-2GB per model
```

**Data (User-created):**

```text
~/.local/share/arcaneum/      # Local databases and indexed content
```

**Vector Database (Docker):**

Qdrant uses Docker named volumes for data persistence and safety:

```text
qdrant-arcaneum-storage    # Main vector database storage
qdrant-arcaneum-snapshots  # Backup snapshots
```

Named volumes store data on a Linux ext4 filesystem inside Docker, providing better
reliability and performance than bind mounts.

**Legacy Migration:**

If upgrading from an older version with `~/.arcaneum/`, the directory will be
automatically migrated to XDG-compliant locations on first run.

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

| Command | Description |
| ------- | ----------- |
| `/arc:corpus` | **Recommended** - Manage dual-index corpora (Qdrant + MeiliSearch) |
| `/arc:search` | Semantic or full-text search |
| `/arc:index` | Index PDF, code, or markdown content |
| `/arc:store` | Store agent-generated content for memory |
| `/arc:container` | Manage Docker services (start, stop, status) |
| `/arc:doctor` | Verify setup and prerequisites |
| `/arc:models` | List available embedding models |
| `/arc:config` | Manage configuration and cache |
| `/arc:collection` | Manage Qdrant collections (semantic search only) |
| `/arc:indexes` | Manage MeiliSearch indexes (full-text search only) |

**Usage Examples:**

```text
/arc:corpus create my-docs --type pdf
/arc:corpus sync my-docs ~/Documents
/arc:search semantic "example query" --corpus my-docs
/arc:search text "exact phrase" --corpus my-docs
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
- ✅ **RDR-008**: MeiliSearch setup (COMPLETED)
- ✅ **RDR-009**: Dual indexing strategy (COMPLETED)
- ✅ **RDR-010**: PDF full-text indexing (COMPLETED)
- ✅ **RDR-011**: Source code full-text indexing (COMPLETED)
- ✅ **RDR-012**: Full-text search integration (COMPLETED)
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

- Built on [Qdrant](https://qdrant.tech/) and [MeiliSearch](https://www.meilisearch.com/)

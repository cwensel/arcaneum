# Arcaneum

CLI tools and Claude Code plugins for semantic search across Qdrant vector databases.

## Overview

Arcaneum provides semantic search for documents using Qdrant vector embeddings. The system supports PDF documents and source code with git-aware, AST-based chunking.

**Currently Available:** Semantic search with Qdrant
**Planned:** MeiliSearch integration for full-text search (RDR-008 through RDR-012)

## Features

### Search Capabilities

- **Semantic Search (Qdrant)**: Find conceptually similar content using vector embeddings
- **Full-Text Search**: Planned - MeiliSearch integration for exact phrase matching (RDR-008 through RDR-012)

### Indexing

- **PDF Indexing**: OCR support for scanned documents, page-level metadata
- **Source Code Indexing**: Git-aware with AST chunking, multi-branch support, 165+ languages
- **Dual Indexing**: Single command to index to both search engines

### Multiple Embedding Models

- stella_en_1.5B_v5 (1024D) - High-quality general embeddings
- modernbert (1024D) - Transformer-based embeddings
- bge-large-en-v1.5 (1024D) - BGE embeddings
- jina-code-embeddings (768D/1536D) - Code-optimized embeddings

### CLI-First Design

- All operations via command-line interface
- JSON output mode for automation
- Structured error messages with exit codes
- Python >= 3.12 required

### Claude Code Integration

- Slash commands for all operations (`/search`, `/search-text`, `/index-pdfs`, etc.)
- Discoverable via `/help` command
- No MCP overhead - direct CLI execution

## Quick Start (5 Minutes)

Get started with Arcaneum in just a few commands:

```bash
# 1. Install
git clone https://github.com/cwensel/arcaneum
cd arcaneum
pip install -e .

# 2. Verify and start services
arc doctor
arc container start

# 3. Index and search your code
arc collection create MyCode --model jina-code --type code
arc index source ~/my-project --collection MyCode
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
arc index source ~/projects/my-app --collection MyCode

# Search semantically
arc search semantic "authentication middleware" --collection MyCode --limit 10
```

### Search PDFs

```bash
# Create a PDF collection
arc collection create MyDocs --model stella --type pdf

# Index PDFs (with OCR for scanned documents)
arc index pdfs ~/Documents/papers --collection MyDocs

# Search for concepts
arc search semantic "neural network architectures" --collection MyDocs
```

### Manage Services

```bash
arc container start    # Start Qdrant
arc container status   # Check health
arc container logs     # View logs
arc container stop     # Stop services
```

## Project Structure

```text
arcaneum/
├── .claude-plugin/              # Claude Code plugin metadata
│   ├── plugin.json              # Plugin configuration
│   └── marketplace.json         # Marketplace catalog
│
├── commands/                    # Slash commands (*.md files)
│   ├── doctor.md                # Setup verification
│   ├── create-collection.md     # Create Qdrant collection
│   ├── list-collections.md      # List collections
│   ├── index-pdfs.md            # Index PDF files
│   ├── index-source.md          # Index source code
│   ├── search.md                # Semantic search
│   ├── search-text.md           # Full-text search
│   ├── create-corpus.md         # Create dual corpus
│   └── sync-directory.md        # Dual indexing
│
├── src/arcaneum/               # Python implementation
│   ├── __init__.py             # Version: 0.1.0
│   ├── cli/                    # CLI commands
│   │   └── main.py             # Centralized dispatcher
│   ├── embeddings/             # Embedding generation (RDR-003)
│   ├── collections/            # Collection management (RDR-003)
│   ├── indexing/               # Indexing pipelines (RDR-004, RDR-005, RDR-009)
│   ├── search/                 # Search logic (RDR-007, RDR-012)
│   ├── fulltext/               # MeiliSearch integration (RDR-008)
│   └── schema/                 # Shared schemas (RDR-009)
│
├── scripts/                         # Management and testing scripts
│   ├── qdrant-manage.sh             # Qdrant operations (RDR-002)
│   ├── meilisearch-manage.sh        # MeiliSearch operations (RDR-008)
│   ├── validate-plugin.sh           # Plugin validation (RDR-006)
│   ├── test-plugin-commands.sh      # Command testing (RDR-006)
│   └── test-claude-integration.sh   # Claude integration tests (RDR-006)
│
├── docs/rdr/                   # Recommendation Data Records
│   ├── README.md               # RDR process guide
│   ├── TEMPLATE.md             # RDR template
│   └── RDR-*.md                # Individual RDRs
│
├── docker-compose.yml          # Added in RDR-002, extended in RDR-008
├── .env                        # Environment variables
├── .gitignore                  # Python, Docker volumes, config files
├── LICENSE                     # MIT
└── README.md                   # This file
```

## Installation

### Prerequisites

- **Python 3.12+** - Check with `python --version`
- **Git** - For cloning the repository
- **Docker** - [Install Docker Desktop](https://docs.docker.com/get-docker/)

### Install

```bash
git clone https://github.com/yourorg/arcaneum
cd arcaneum
pip install -e .
```

### Verify Setup

```bash
arc doctor
```

The `doctor` command checks your environment and guides you through any issues.

### Data Storage

Arcaneum stores all data in `~/.arcaneum/`:

```text
~/.arcaneum/
├── models/              # Embedding models (auto-downloaded, ~1-2GB)
├── data/
│   ├── qdrant/         # Vector database
│   └── qdrant_snapshots/  # Backups
```

**Benefits:**

- Easy to backup (just backup `~/.arcaneum/`)
- Easy to find and manage
- No hidden Docker volumes

### Corporate Networks

Behind a VPN with SSL issues? See **[Corporate Network Setup](docs/testing/offline-mode.md)** for:

- Offline mode setup
- SSL certificate workarounds
- Model pre-downloading

## Claude Code Plugin

### Installation

In Claude Code:

```text
/plugin marketplace add cwensel/arcaneum
/plugin install arcaneum
```

### Available Commands

- `/doctor` - Verify setup and prerequisites
- `/create-collection` - Create Qdrant collection
- `/list-collections` - List all collections
- `/index-pdfs` - Index PDF files
- `/index-source` - Index source code
- `/search` - Semantic search
- `/search-text` - Full-text search (planned, RDR-012)
- `/create-corpus` - Create dual corpus (planned, RDR-009)
- `/sync-directory` - Dual indexing (planned, RDR-009)

Use `/help` to see all available commands or `/doctor` to check your setup.

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

- **[Quick Start Guide](docs/guides/quickstart.md)** - Complete walkthrough with troubleshooting
- **[CLI Reference](docs/guides/cli-reference.md)** - All commands and options
- **[PDF Indexing Guide](docs/guides/pdf-indexing.md)** - Advanced PDF indexing with OCR
- **[Corporate Network Setup](docs/testing/offline-mode.md)** - SSL and offline mode

### Development

- **[RDR Process](docs/rdr/README.md)** - Detailed planning workflow
- **[Individual RDRs](docs/rdr/)** - Technical specifications for each feature
- **[Slash Commands](commands/)** - Claude Code integration

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

# Arcaneum

CLI tools and Claude Code plugins for semantic and full-text search across Qdrant and MeiliSearch.

## Overview

Arcaneum provides a unified workflow for indexing and searching documents with both semantic similarity (via Qdrant vector embeddings) and exact phrase matching (via MeiliSearch full-text search). The system supports PDF documents and source code with git-aware, AST-based chunking.

## Features

### Dual Search Capabilities
- **Semantic Search (Qdrant)**: Find conceptually similar content using vector embeddings
- **Full-Text Search (MeiliSearch)**: Find exact phrases, function names, and specific strings
- **Cooperative Workflow**: Use semantic search for discovery, full-text for verification

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

## Project Structure

```
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
├── doc/rdr/                    # Recommendation Data Records
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

- Python >= 3.12
- Git
- Docker (for Qdrant and MeiliSearch)

### Clone Repository

```bash
git clone https://github.com/yourorg/arcaneum
cd arcaneum
```

### Install Dependencies

```bash
pip install -e .
```

**Corporate Networks:** If behind VPN with SSL certificate issues:
- **Recommended:** Pre-download models, then set `export HF_HUB_OFFLINE=1` in your shell
- **Alternative:** Set SSL bypass env vars in `~/.bashrc`: `export PYTHONHTTPSVERIFY=0`
- See `doc/testing/OFFLINE-MODE.md` for complete corporate network setup

## Quick Start

### 1. Start Docker Services

Services will be added incrementally:
- RDR-002: Qdrant vector database
- RDR-008: MeiliSearch full-text search

```bash
# After RDR-002 implementation:
./scripts/qdrant-manage.sh start

# After RDR-008 implementation:
./scripts/meilisearch-manage.sh start
```

### 2. Create a Corpus (Dual Indexing)

After RDR-009 implementation:

```bash
python -m arcaneum.cli.main create-corpus MyCode --type source-code
```

### 3. Index Documents

```bash
# Index source code
python -m arcaneum.cli.main sync-directory ~/code/projects --corpus MyCode

# Or index PDFs
python -m arcaneum.cli.main sync-directory ~/Documents/papers --corpus Research
```

### 4. Search

```bash
# Semantic search (conceptual similarity)
python -m arcaneum.cli.main search "authentication patterns" --collection MyCode

# Full-text search (exact phrases)
python -m arcaneum.cli.main search-text '"def authenticate"' --index MyCode-fulltext
```

## Claude Code Plugin

### Installation

In Claude Code:
```
/plugin marketplace add yourorg/arcaneum
/plugin install arcaneum
```

### Available Commands

- `/doctor` - Verify setup and prerequisites
- `/create-collection` - Create Qdrant collection
- `/list-collections` - List all collections
- `/index-pdfs` - Index PDF files
- `/index-source` - Index source code
- `/search` - Semantic search
- `/search-text` - Full-text search
- `/create-corpus` - Create dual corpus
- `/sync-directory` - Dual indexing

Use `/help` to see all available commands or `/doctor` to check your setup.

## Development

### Architecture Principles

1. **CLI-First**: All functionality as CLI tools (RDR-001, RDR-006)
2. **Slash Commands**: Thin wrappers calling CLI via Bash (RDR-006)
3. **No MCP (v1)**: Avoid MCP overhead, use direct CLI execution (RDR-006)
4. **Local Docker**: Databases run locally with volume persistence (RDR-002, RDR-008)
5. **RDR-Based Planning**: Detailed design before implementation (doc/rdr/)

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

- **RDR Process**: See `doc/rdr/README.md` for detailed planning workflow
- **Individual RDRs**: See `doc/rdr/RDR-*.md` for feature specifications
- **Slash Commands**: See `commands/*.md` for Claude Code usage

## Contributing

1. Read `doc/rdr/README.md` for RDR-based development workflow
2. Create an RDR for complex features before implementation
3. Follow CLI-first architecture pattern
4. Add tests for new functionality
5. Update this README with implementation status

## License

MIT - See LICENSE file for details

## Acknowledgments

- Inspired by [Beads](https://github.com/steveyegge/beads) for Claude Code plugin patterns
- Built on [Qdrant](https://qdrant.tech/) and [MeiliSearch](https://www.meilisearch.com/)

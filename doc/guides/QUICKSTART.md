# Arcaneum Quick Start

Get started with the `arc` CLI in 5 minutes.

## Prerequisites

- Python >= 3.12
- Docker (for Qdrant)
- Git

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourorg/arcaneum
cd arcaneum
```

### 2. Install Dependencies

```bash
pip install -e .
```

### 3. Start Qdrant Server

```bash
docker compose -f deploy/docker-compose.yml up -d
```

### 4. Verify Installation

```bash
bin/arc --help
```

## Your First PDF Index

### 1. Create a Collection

```bash
bin/arc create-collection my-docs --model stella
```

### 2. Create Test Directory

```bash
mkdir test_pdfs
# Add some PDF files to test_pdfs/
```

### 3. Index PDFs

```bash
bin/arc index-pdfs ./test_pdfs --collection my-docs --model stella
```

### 4. Verify Indexing

```bash
bin/arc collection-info my-docs
```

## Quick Commands

### Collection Management

```bash
# Create
bin/arc create-collection <name> --model <model>

# List
bin/arc list-collections

# Info
bin/arc collection-info <name>

# Delete
bin/arc delete-collection <name> --confirm
```

### Indexing

```bash
# PDFs (text)
bin/arc index-pdfs <directory> --collection <name> --model <model>

# PDFs with OCR (scanned)
bin/arc index-pdfs <directory> --collection <name> --model <model> --ocr-enabled

# Source code
bin/arc index-source <directory> --collection <name> --model jina
```

### Search

```bash
# Semantic search
bin/arc search "machine learning" --collection <name>

# Full-text search (requires MeiliSearch)
bin/arc search-text "exact phrase" --index <name>
```

## Models

Choose a model based on your content:

- `stella` - Best for general documents, long context
- `bge` - Best for precision, shorter documents
- `modernbert` - Best for recent content, long context
- `jina` - Best for code + text, multilingual

## Next Steps

- **PDF Indexing**: See [docs/pdf-indexing.md](docs/pdf-indexing.md) for OCR, incremental indexing, and advanced options
- **CLI Reference**: See [docs/arc-cli-reference.md](docs/arc-cli-reference.md) for all commands
- **RDR Documentation**: See [doc/rdr/](doc/rdr/) for technical specifications

## Troubleshooting

### Command not found

```bash
# Make executable
chmod +x bin/arc

# Or use directly
python -m arcaneum.cli.main --help
```

### Qdrant not running

```bash
# Check status
docker compose -f deploy/docker-compose.yml ps

# Start services
docker compose -f deploy/docker-compose.yml up -d

# View logs
docker compose -f deploy/docker-compose.yml logs qdrant
```

### OCR not working

```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils
```

### SSL certificate errors

If model downloads fail with SSL errors (corporate proxies):

```bash
# Pre-download models on a machine with working SSL
python scripts/download-models.py

# Copy models_cache/ directory to your machine
# Models load from cache - no network needed
```

## Environment Setup

Create `.env` file (optional):

```bash
QDRANT_URL=http://localhost:6333
MEILISEARCH_URL=http://localhost:7700
MEILISEARCH_API_KEY=your-key-here
```

## Development Mode

```bash
# Use bin/arc directly
bin/arc <command> [options]

# After pip install -e .
arc <command> [options]
```

## Claude Code Integration

The `arc` CLI is used by Claude Code agents and plugins:

- All commands are available as slash commands
- Plugins can invoke `bin/arc` directly from repo root
- No MCP server needed

See [.claude-plugin/](.claude-plugin/) for plugin configuration.

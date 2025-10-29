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
arc create-collection <name> --model <model>  # Create Qdrant collection
arc list-collections                          # List all collections
arc collection-info <name>                    # Show collection details
arc delete-collection <name>                  # Delete collection
```

### Indexing Commands

```bash
arc index-pdfs <path> --collection <name>     # Index PDF files
arc index-source <path> --collection <name>   # Index source code
```

### Search Commands

```bash
arc search <query> --collection <name>        # Semantic search
arc search-text <query> --index <name>        # Full-text search
```

### Dual Indexing (Qdrant + MeiliSearch)

```bash
arc create-corpus <name> --type <type>        # Create dual corpus
arc sync-directory <path> --corpus <name>     # Dual indexing
```

## Collection Management Examples

### Create Collection

```bash
# Create collection with single model
arc create-collection pdf-docs --model stella

# With custom HNSW parameters
arc create-collection pdf-docs --model stella --hnsw-m 16 --hnsw-ef 100

# Store vectors on disk (for large collections)
arc create-collection pdf-docs --model stella --on-disk
```

### List Collections

```bash
# Simple list
arc list-collections

# Verbose output
arc list-collections --verbose

# JSON output for scripting
arc list-collections --json
```

### Collection Info

```bash
arc collection-info pdf-docs
arc collection-info pdf-docs --json
```

### Delete Collection

```bash
# With confirmation prompt
arc delete-collection pdf-docs

# Skip confirmation
arc delete-collection pdf-docs --confirm
```

## PDF Indexing Examples

### Basic Usage

```bash
arc index-pdfs /path/to/pdfs --collection pdf-docs --model stella
```

### With OCR

```bash
arc index-pdfs /path/to/scanned-pdfs \
  --collection pdf-docs \
  --model stella \
  --ocr-language eng
```

### Force Reindex

```bash
arc index-pdfs /path/to/pdfs \
  --collection pdf-docs \
  --model stella \
  --force
```

### Parallel Processing

```bash
arc index-pdfs /path/to/pdfs \
  --collection pdf-docs \
  --model stella \
  --workers 8
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
docker compose -f deploy/docker-compose.yml up -d

# 2. Create collection
arc create-collection my-docs --model stella

# 3. Index documents
arc index-pdfs ./documents --collection my-docs --model stella
```

### Incremental Updates

```bash
# First run: indexes all PDFs
arc index-pdfs ./docs --collection my-docs --model stella

# Add new files to ./docs/...

# Second run: only indexes new/modified files
arc index-pdfs ./docs --collection my-docs --model stella
```

### JSON Output for Automation

```bash
# List collections
arc list-collections --json | jq '.collections[].name'

# Index with JSON output
arc index-pdfs ./docs --collection my-docs --model stella --json > results.json

# Check results
jq '.stats.chunks' results.json
```

## Global Options

Most commands support:

- `--json`: Output JSON format (for scripting)
- `--verbose` / `-v`: Verbose output
- `--help`: Show command help

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
curl http://localhost:6333/health

# Start if needed
docker compose -f deploy/docker-compose.yml up -d
```

### Permission Denied

```bash
chmod +x bin/arc
```

## More Documentation

- [PDF Indexing Guide](pdf-indexing.md) - Detailed PDF indexing documentation
- [RDR Directory](../doc/rdr/) - Technical specifications for all features

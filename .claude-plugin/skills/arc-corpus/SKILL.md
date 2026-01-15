---
name: arc-corpus
description: Dual-index corpus management for combined semantic and full-text search. Use when user mentions corpus, dual indexing, syncing content to both Qdrant and MeiliSearch, checking parity between systems, or managing content that needs both search types.
allowed-tools: Bash(arc:*), Read
---

# Corpus Management (Dual-Index)

A corpus maintains both a Qdrant collection (semantic search) and a MeiliSearch index (full-text search) in sync.

```bash
# Create corpus (creates both collection and index)
arc corpus create MyCorpus --type pdf
arc corpus create MyCorpus --type code
arc corpus create MyCorpus --type markdown
arc corpus create MyCorpus --type pdf --models stella,jina  # Multiple models

# Sync files to both systems
arc corpus sync /path/to/files --corpus MyCorpus
arc corpus sync /path/to/files --corpus MyCorpus --force    # Force reindex
arc corpus sync /path/to/files --corpus MyCorpus --verify   # Verify after sync
arc corpus sync /path/to/files --corpus MyCorpus --verbose  # Show progress

# View corpus info (both systems)
arc corpus info MyCorpus
arc corpus info MyCorpus --json

# Check and restore parity between systems
arc corpus parity MyCorpus              # Check and backfill
arc corpus parity MyCorpus --dry-run    # Preview only
arc corpus parity MyCorpus --verbose    # Detailed progress
```

## When to Use Corpus vs Collection/Index

- **Use Corpus**: When you need both semantic search (conceptual queries) AND full-text search (exact phrases)
- **Use Collection alone**: When you only need semantic/conceptual search
- **Use Index alone**: When you only need exact keyword/phrase search

## Parity Behavior

The `parity` command ensures both systems have the same content:
- **Qdrant -> MeiliSearch**: Copies metadata (fast, no file access needed)
- **MeiliSearch -> Qdrant**: Re-chunks and embeds files (requires file access)

---
name: arc-corpus
description: Dual-index corpus management for combined semantic and full-text search. Use when user mentions corpus, dual indexing, syncing content to both Qdrant and MeiliSearch, checking parity between systems, deleting corpora, or managing content that needs both search types.
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

# Delete corpus (deletes both collection and index)
arc corpus delete MyCorpus              # With confirmation prompt
arc corpus delete MyCorpus --confirm    # Skip confirmation
arc corpus delete MyCorpus --confirm --json  # JSON output

# Sync files to both systems
arc corpus sync /path/to/files --corpus MyCorpus
arc corpus sync /path/to/files --corpus MyCorpus --force    # Force reindex
arc corpus sync /path/to/files --corpus MyCorpus --verify   # Verify after sync
arc corpus sync /path/to/files --corpus MyCorpus --verbose  # Show progress

# View corpus info (both systems)
arc corpus info MyCorpus
arc corpus info MyCorpus --json

# List indexed items with parity status
arc corpus items MyCorpus               # Table output with Q/M chunk counts
arc corpus items MyCorpus --json        # JSON output for automation

# Check and restore parity between systems
arc corpus parity MyCorpus              # Check and backfill
arc corpus parity MyCorpus --dry-run    # Preview only
arc corpus parity MyCorpus --verify     # Verify chunk counts match
arc corpus parity MyCorpus --repair-metadata  # Fix missing git metadata (code corpora)
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

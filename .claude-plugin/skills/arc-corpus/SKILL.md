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
arc corpus sync MyCorpus /path/to/files
arc corpus sync MyCorpus /path/one /path/two /path/three    # Multiple directories
arc corpus sync MyCorpus /path/to/files --force             # Force reindex
arc corpus sync MyCorpus /path/to/files --verify            # Verify after sync
arc corpus sync MyCorpus /path/to/files --verbose           # Show progress
arc corpus sync MyCorpus /path/to/files --no-gpu            # CPU-only mode (stable on Apple Silicon)

# View corpus info (both systems)
arc corpus info MyCorpus
arc corpus info MyCorpus --json

# List indexed items with parity status
arc corpus items MyCorpus               # Table output with Q/M chunk counts
arc corpus items MyCorpus --json        # JSON output for automation

# Check and restore parity between systems
arc corpus parity MyCorpus              # Check and backfill single corpus
arc corpus parity MyCorpus --dry-run    # Preview only
arc corpus parity MyCorpus --verify     # Verify chunk counts match
arc corpus parity MyCorpus --repair-metadata  # Fix missing git metadata (code corpora)
arc corpus parity MyCorpus --verbose    # Detailed progress

# All-corpora mode (no corpus name)
arc corpus parity                       # Process all corpora
arc corpus parity --dry-run             # Preview all
arc corpus parity --confirm             # Skip confirmation prompt

# Create missing MeiliSearch indexes for qdrant_only corpora
arc corpus parity --create-missing --dry-run   # Preview what would be created
arc corpus parity --create-missing --confirm   # Create and sync all
```

## When to Use Corpus vs Collection/Index

- **Use Corpus**: When you need both semantic search (conceptual queries) AND full-text search (exact phrases)
- **Use Collection alone**: When you only need semantic/conceptual search
- **Use Index alone**: When you only need exact keyword/phrase search

## Parity Behavior

The `parity` command ensures both systems have the same content:

- **Qdrant -> MeiliSearch**: Copies metadata (fast, no file access needed)
- **MeiliSearch -> Qdrant**: Re-chunks and embeds files (requires file access)

**Creating Missing Indexes:**

Use `--create-missing` to promote single-sided Qdrant collections into full corpora:

- Creates MeiliSearch indexes for `qdrant_only` collections
- Reads corpus type from Qdrant metadata
- Applies appropriate index settings automatically
- Then proceeds with normal parity sync

Note: `meili_only` corpora cannot be auto-created (require `--type` and `--model`).

## GPU Acceleration and Apple Silicon

By default, corpus sync uses GPU acceleration (MPS on Apple Silicon, CUDA on NVIDIA).

**Large models on Apple Silicon:** Models like `stella` (1.5B params) may cause system
instability on Macs with limited memory. If you experience lockups:

```bash
# Use CPU-only mode (slower but stable)
arc corpus sync MyCorpus /path --no-gpu

# Or use the environment variable
ARC_NO_GPU=1 arc corpus sync MyCorpus /path

# Or use a smaller model (bge is 0.3B params)
arc corpus sync MyCorpus /path --models bge
```

**Model sizes:**

- `bge`, `bge-base`, `bge-small`: Safe for all systems
- `stella` (1.5B): May cause issues on Macs with <16GB RAM
- `nomic-code` (7B): Requires dedicated GPU with significant VRAM

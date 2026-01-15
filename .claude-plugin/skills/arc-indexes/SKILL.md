---
name: arc-indexes
description: MeiliSearch index management for full-text search. Use when user mentions managing indexes, creating index, listing indexes, verifying index health, exporting or importing indexes, or viewing indexed items in MeiliSearch.
allowed-tools: Bash(arc:*), Read
---

# MeiliSearch Index Management

```bash
# List all indexes
arc indexes list
arc indexes list --json

# Create index
arc indexes create MyIndex
arc indexes create MyIndex --type source-code   # Code-optimized settings
arc indexes create MyIndex --type pdf-docs      # PDF-optimized settings
arc indexes create MyIndex --type markdown-docs # Markdown-optimized settings

# View index info
arc indexes info MyIndex
arc indexes info MyIndex --json

# List indexed items
arc indexes items MyIndex
arc indexes items MyIndex --limit 50
arc indexes items MyIndex --json

# Verify index health
arc indexes verify MyIndex
arc indexes verify MyIndex --json

# Update index settings
arc indexes update-settings MyIndex --type source-code

# Export index
arc indexes export MyIndex -o backup.jsonl
arc indexes export MyIndex -o backup.jsonl --json

# Import index
arc indexes import backup.jsonl
arc indexes import backup.jsonl --into NewIndex

# Delete index
arc indexes delete MyIndex --confirm
```

## Index Types

- **source-code** (or `code`): Optimized for code search with language filtering
- **pdf-docs** (or `pdf`): Optimized for document search with page filtering
- **markdown-docs** (or `markdown`): Optimized for documentation with heading search

---
name: arc-collection
description: Qdrant collection management for semantic search. Use when user mentions managing collections, creating collection, listing collections, verifying collection integrity, exporting or importing collections, or viewing indexed items in Qdrant.
allowed-tools: Bash(arc:*), Read
---

# Qdrant Collection Management

```bash
# List all collections
arc collection list
arc collection list --verbose     # Show type and vector details
arc collection list --json        # JSON output

# Create collection
arc collection create MyCollection --type pdf        # PDF collection (stella model)
arc collection create MyCollection --type code       # Code collection (jina-code)
arc collection create MyCollection --type markdown   # Markdown collection (stella)
arc collection create MyCollection --model stella    # Explicit model

# View collection info
arc collection info MyCollection
arc collection info MyCollection --json

# List indexed items
arc collection items MyCollection
arc collection items MyCollection --json

# Verify integrity (fsck-like check)
arc collection verify MyCollection
arc collection verify MyCollection --project myrepo#main  # Code collections
arc collection verify MyCollection --verbose --json

# Export collection
arc collection export MyCollection -o backup.arcexp
arc collection export MyCollection -o backup.jsonl --format jsonl
arc collection export MyCollection -o shareable.arcexp --detach  # Portable

# Import collection
arc collection import backup.arcexp
arc collection import backup.arcexp --into RestoredCollection
arc collection import shareable.arcexp --attach /new/root/path

# Delete collection
arc collection delete MyCollection --confirm
```

## Collection Types

- **pdf**: Documents and PDFs (stella model, 1024D)
- **code**: Source code repositories (jina-code, 768D)
- **markdown**: Documentation and notes (stella model, 1024D)

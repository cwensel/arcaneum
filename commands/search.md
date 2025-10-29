---
description: Search Qdrant collection semantically
argument-hint: "<query>" --collection <name> [options]
---

Perform semantic search across a Qdrant collection.

**Arguments:**
- "<query>": Search query (required, use quotes for multi-word queries)
- --collection <name>: Collection to search (required)
- --vector-name <name>: Vector to use (optional, auto-detects from collection)
- --filter <filter>: Metadata filter (key=value or JSON)
- --limit <n>: Number of results (default: 10)
- --score-threshold <float>: Minimum score threshold
- --verbose: Show detailed match information
- --json: Output JSON format

**Examples:**
```
/search "authentication patterns" --collection MyCode --limit 5
/search "machine learning" --collection Research --filter language=python
/search "error handling" --collection Documentation --verbose
```

**Execution:**
```bash
cd ${CLAUDE_PLUGIN_ROOT}
python -m arcaneum.cli.main search $ARGUMENTS
```

**Note:** Uses semantic similarity via vector embeddings to find conceptually
related content, not just exact matches. I'll present the search results showing:
- Relevance scores for each match
- Source file paths and locations
- Matching content snippets
- Metadata filters applied (if any)

For exact phrase matching, use /search-text instead. Full implementation in RDR-007.

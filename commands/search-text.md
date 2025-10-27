---
description: Search MeiliSearch index with exact phrases
argument-hint: "<query>" --index <name> [options]
---

Perform full-text search for exact phrases and keywords.

**Arguments:**
- "<query>": Search query (use quotes for exact phrases)
- --index <name>: MeiliSearch index to search (required)
- --filter <filter>: Metadata filter (e.g., language=python)
- --limit <n>: Number of results (default: 10)
- --verbose: Show detailed match information
- --json: Output JSON format

**Examples:**
```
/search-text '"def authenticate"' --index MyCode-fulltext
/search-text 'calculate_total' --index MyCode-fulltext --filter language=python
/search-text '"neural network"' --index PDFs --filter 'page_number > 5'
```

**Execution:**
```bash
cd ${CLAUDE_PLUGIN_ROOT}
python -m arcaneum.cli.main search-text $ARGUMENTS
```

**Note:** Uses full-text search for exact matches. For semantic similarity,
use /search instead. Full implementation in RDR-012.

---
description: Index directory to both Qdrant and MeiliSearch
argument-hint: <directory> --corpus <name> [options]
---

Index documents from a directory to both Qdrant and MeiliSearch for dual search capabilities.

**Arguments:**
- <directory>: Directory to sync (required)
- --corpus <name>: Corpus name (required)
- --models <models>: Embedding models, comma-separated (default: stella,jina)
- --file-types <types>: File extensions to index (e.g., .py,.md)
- --json: Output JSON format

**Examples:**
```
/sync-directory /Documents/papers --corpus Research
/sync-directory ~/code/projects --corpus MyCode --models stella,jina-code
/sync-directory ./docs --corpus Documentation --file-types .md,.txt
```

**Execution:**
```bash
cd ${CLAUDE_PLUGIN_ROOT}
python -m arcaneum.cli.main sync-directory $ARGUMENTS
```

**Note:** Requires corpus to be created first with /create-corpus.
Full implementation in RDR-009.

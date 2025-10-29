---
description: List all Qdrant collections
argument-hint: [--verbose] [--json]
---

List all Qdrant collections with their configurations.

**Arguments:**

- --verbose: Show detailed collection information
- --json: Output JSON format

**Examples:**

```text
/list-collections
/list-collections --verbose
/list-collections --json
```

**Execution:**

```bash
cd ${CLAUDE_PLUGIN_ROOT}
arc list-collections $ARGUMENTS
```

**Note:** This command returns quickly. I'll present the collections in a clear
table format showing collection names, document counts, and vector configurations.
Use --verbose for detailed information about each collection's settings.

Full implementation in RDR-003.

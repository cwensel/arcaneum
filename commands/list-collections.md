---
description: List all Qdrant collections
argument-hint: [--verbose] [--json]
---

List all Qdrant collections with their configurations.

**Arguments:**
- --verbose: Show detailed collection information
- --json: Output JSON format

**Examples:**
```
/list-collections
/list-collections --verbose
/list-collections --json
```

**Execution:**
```bash
cd ${CLAUDE_PLUGIN_ROOT}
python -m arcaneum.cli.main list-collections $ARGUMENTS
```

**Note:** Full implementation in RDR-003.

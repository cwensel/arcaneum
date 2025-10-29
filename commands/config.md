---
description: Configuration and cache management
argument-hint: <show-cache-dir|clear-cache> [options]
---

Manage Arcaneum configuration and model cache.

**Subcommands:**

- show-cache-dir: Display cache locations and sizes
- clear-cache: Clear model cache to free disk space

**Arguments:**

- --confirm: Confirm cache deletion (required for clear-cache)

**Examples:**

```text
/config show-cache-dir
/config clear-cache --confirm
```

**Execution:**

```bash
cd ${CLAUDE_PLUGIN_ROOT}
arc config $ARGUMENTS
```

**Note:** The model cache (~/.arcaneum/models/) stores downloaded embedding models.
First-time indexing downloads ~1-2GB of models which are then reused. I'll show you:

- Current cache directory locations
- Size of each directory (models, data)
- Free disk space information

Use clear-cache when models are corrupted or to free disk space (models will
be re-downloaded on next use).

**Related:**

- Models auto-downloaded to ~/.arcaneum/models/
- Data stored in ~/.arcaneum/data/
- Implemented in arcaneum-157 and arcaneum-162

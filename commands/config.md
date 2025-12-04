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
arc config $ARGUMENTS
```

**Note:** The model cache stores downloaded embedding models in XDG-compliant locations.
First-time indexing downloads ~1-2GB of models which are then reused. I'll show you:

- Current cache directory locations
- Size of each directory
- Free disk space information

Use clear-cache when models are corrupted or to free disk space (models will
be re-downloaded on next use).

**Directory Locations (XDG-compliant):**

- Models (cache): `~/.cache/arcaneum/models`
- Qdrant data: Docker volume `qdrant-arcaneum-storage`

**Related:**

- Implemented in arcaneum-157 and arcaneum-162

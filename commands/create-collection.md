---
description: Create a new Qdrant collection
argument-hint: <name> --model <model> [options]
---

Create a new Qdrant collection with specified embedding model and configuration.

**Arguments:**

- <name>: Collection name (required)
- --model <model>: Embedding model (stella, modernbert, bge, jina-code) (required)
- --hnsw-m <n>: HNSW index parameter m (default: 16)
- --hnsw-ef <n>: HNSW index parameter ef_construct (default: 100)
- --on-disk: Store vectors on disk (reduces RAM usage)
- --json: Output JSON format

**Examples:**

```text
/create-collection Research --model stella
/create-collection LargeArchive --model modernbert --on-disk
/create-collection CodeLibrary --model jina-code --hnsw-m 32
```

**Execution:**

```bash
cd ${CLAUDE_PLUGIN_ROOT}
arc create-collection $ARGUMENTS
```

**Note:** This command typically completes in seconds. I'll verify the collection
is created successfully and show you the configuration details including the
embedding model, vector dimensions, and HNSW parameters.

Full implementation in RDR-003.

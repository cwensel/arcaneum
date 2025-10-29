---
description: Index source code to Qdrant collection
argument-hint: <path> --collection <name> [options]
---

Index source code from git repositories to a Qdrant collection with AST-aware chunking.

**Arguments:**

- <path>: Directory containing git repositories
- --collection <name>: Target Qdrant collection name (required)
- --model <model>: Embedding model (default: jina-code)
- --workers <n>: Parallel workers (default: 4)
- --depth <n>: Git discovery depth (default: unlimited)
- --force: Force reindex all projects
- --verbose: Detailed progress output
- --json: Output JSON format

**Examples:**

```text
/index-source /code/projects --collection MyCode
/index-source ~/repos --collection OpenSource --model jina-code
/index-source . --collection CurrentProject --depth 0
```

**Execution:**

```bash
cd ${CLAUDE_PLUGIN_ROOT}
arc index-source $ARGUMENTS
```

**Note:** Git repositories are indexed from their current branch.
Multi-branch support available. I'll monitor the progress and show you:

- Git repositories discovered
- Files being processed with percentage completion
- AST-aware chunking progress for each project
- Final summary with total projects, files, and chunks indexed
- Any parsing errors or skipped files

Full implementation in RDR-005.

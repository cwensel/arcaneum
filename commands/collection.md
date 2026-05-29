---
description: Manage Qdrant collections (semantic search only)
argument-hint: <create|list|info|delete|items|verify|export|import> [name] [options]
---

Manage Qdrant vector collections for storing embeddings.

**Note:** For most users, `/arc:corpus` is recommended as it provides both semantic and full-text
search. Use `/arc:collection` when you only need semantic search.

**IMPORTANT:** You must specify a subcommand (`create`, `list`, `info`, `delete`, `items`, `verify`, `export`, or `import`).

**Subcommands (required):**

- `create`: Create a new collection with specified embedding model
- `list`: List all collections in Qdrant
- `info`: Show detailed information about a collection
- `delete`: Delete a collection permanently
- `items`: List indexed files/repos in a collection
- `verify`: Verify collection integrity (fsck-like check)
- `export`: Export collection to a portable format
- `import`: Import collection from an export file

**Common Options:**

- --json: Output in JSON format for programmatic use

**Examples:**

```text
/collection create MyDocs --type pdf
/collection create MyDocsQuality --model stella --type pdf
/collection list
/collection info MyDocs
/collection delete MyDocs --confirm
```

**Execution:**

```bash
arc collection $ARGUMENTS
```

**Collection Types:**

Collections should specify a type (pdf or code) to ensure content matches:

- pdf: For document collections (PDFs, text files)
- code: For source code repositories
- markdown: For markdown/document collections

**Available Models:**

- arctic-m: 768D, stable document/PDF/markdown default
- stella: 1024D, high-quality opt-in document model (requires `arcaneum[sentence-transformers]`)
- mxbai-large: 1024D, high-quality FastEmbed document model
- jina-code: 768D, stable FastEmbed source-code default
- bge: 1024D, general purpose

**Related Commands:**

- /arc:corpus create - Create corpus for dual indexing (recommended)
- /arc:corpus sync - Index to both systems (recommended)
- /arc:index pdf - Index PDFs into a collection (semantic only)
- /arc:index code - Index source code into a collection (semantic only)

**Implementation:**

- Defined in RDR-003 (Collection Creation)
- Enhanced in RDR-006 (Claude Code Integration)
- Type enforcement added in arcaneum-122

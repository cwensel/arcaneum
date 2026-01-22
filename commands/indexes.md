---
description: Manage MeiliSearch full-text indexes
argument-hint: <create|list|info|delete|items|verify|export|import> [name] [options]
---

Manage MeiliSearch full-text indexes (mirrors `arc collection` for Qdrant).

**IMPORTANT:** You must specify a subcommand (`create`, `list`, `info`, `delete`, `items`, `verify`, `export`, or `import`).

**Subcommands (required):**

- `create`: Create a new index with type-specific settings
- `list`: List all indexes in MeiliSearch
- `info`: Show detailed information about an index
- `delete`: Delete an index permanently
- `items`: List indexed files/documents in an index
- `verify`: Verify index health and integrity
- `export`: Export index to JSONL file
- `import`: Import index from JSONL file
- `update-settings`: Update index settings from preset type
- `list-projects`: List git projects in code index
- `delete-project`: Delete a git project from code index

**Common Options:**

- --json: Output in JSON format for programmatic use

**Examples:**

```text
/indexes create MyDocs --type pdf
/indexes list
/indexes info MyDocs
/indexes items MyDocs
/indexes verify MyDocs
/indexes export MyDocs -o backup.jsonl
/indexes import backup.jsonl --into MyDocs-restored
/indexes delete MyDocs --confirm
```

**Execution:**

```bash
arc indexes $ARGUMENTS
```

**Index Types:**

- `pdf` (alias: `pdf-docs`): PDF documents with stop words
- `code` (alias: `source-code`): Source code with higher typo thresholds
- `markdown` (alias: `markdown-docs`): Markdown with headings search

**Git Project Commands (for code indexes):**

```text
/indexes list-projects MyCode
/indexes delete-project arcaneum#main --index MyCode
```

**Related Commands:**

- /collection - Manage Qdrant collections (semantic search)
- /corpus create - Create both vector and full-text indexes
- /search text - Full-text search in MeiliSearch indexes

**Implementation:**

- Defined in RDR-008 (Full-Text Search Server Setup)
- Enhanced in RDR-011 (Source Code Full-Text Indexing)
- Enhanced in RDR-012 (Full-Text Search Claude Integration)

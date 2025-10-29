---
description: Manage Qdrant collections
argument-hint: <create|list|info|delete> [options]
---

Manage Qdrant vector collections for storing embeddings.

**Subcommands:**

- create: Create a new collection with specified embedding model
- list: List all collections in Qdrant
- info: Show detailed information about a collection
- delete: Delete a collection permanently

**Common Options:**

- --json: Output in JSON format for programmatic use

**Examples:**

```text
/collection create MyDocs --model stella --type pdf
/collection list
/collection info MyDocs
/collection delete MyDocs --confirm
```

**Execution:**

```bash
cd ${CLAUDE_PLUGIN_ROOT}
arc collection $ARGUMENTS
```

**Collection Types:**

Collections should specify a type (pdf or code) to ensure content matches:
- pdf: For document collections (PDFs, text files)
- code: For source code repositories

**Available Models:**

- stella: 1024D, best for documents and PDFs (default for pdfs)
- jina-code: 768D, optimized for source code (default for code)
- bge: 1024D, general purpose
- modernbert: 1024D, newer general-purpose model

**Related Commands:**

- /index pdfs - Index PDFs into a collection
- /index source - Index source code into a collection
- /corpus create - Create both vector and full-text indexes

**Implementation:**

- Defined in RDR-003 (Collection Creation)
- Enhanced in RDR-006 (Claude Code Integration)
- Type enforcement added in arcaneum-122

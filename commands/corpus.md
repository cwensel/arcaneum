---
description: Manage dual-index corpora
argument-hint: <create|sync> [options]
---

Manage corpora that combine both vector search (Qdrant) and full-text search (MeiliSearch) for the same content.

**Subcommands:**

- create: Create both Qdrant collection and MeiliSearch index
- sync: Index directory to both systems simultaneously

**Common Options:**

- --json: Output in JSON format

**Create Options:**

- name: Corpus name (required)
- --type: Corpus type - code or pdf (required)
- --models: Embedding models, comma-separated (default: stella,jina)

**Sync Options:**

- directory: Directory path to index (required)
- --corpus: Corpus name (required)
- --models: Embedding models (default: stella,jina)
- --file-types: File extensions to index (e.g., .py,.md)

**Examples:**

```text
/corpus create MyDocs --type pdf --models stella
/corpus sync ~/Documents --corpus MyDocs
/corpus create CodeBase --type code
/corpus sync ~/projects --corpus CodeBase --file-types .py,.js,.md
```

**Execution:**

```bash
cd ${CLAUDE_PLUGIN_ROOT}
arc corpus $ARGUMENTS
```

**What Is a Corpus?**

A corpus combines two search systems:
1. **Vector search** (Qdrant): Semantic similarity, concept matching
2. **Full-text search** (MeiliSearch): Keyword, phrase, boolean operators

This enables hybrid search strategies:
- Broad semantic discovery (vector search)
- Precise keyword refinement (full-text search)
- Combined results for best of both worlds

**When to Use Corpus vs Collection:**

**Use Corpus When:**
- Need both semantic and keyword search
- Users search different ways (concepts vs exact terms)
- Want fast keyword filtering of semantic results
- Building search UIs with multiple search modes

**Use Collection When:**
- Only need semantic search
- Working with embeddings/vectors directly
- Integrating with existing vector workflows
- MeiliSearch not available/needed

**How Sync Works:**

1. Discovers files in directory (respects .gitignore for code)
2. Chunks content appropriately (PDFs vs code)
3. Generates embeddings with specified models
4. Uploads to Qdrant (vector search)
5. Indexes to MeiliSearch (full-text search)
6. Both indexes share same document IDs and metadata

**Performance:**

Corpus sync is approximately 2x slower than single-system indexing due to dual upload, but still efficient:
- PDFs: ~5-15/minute
- Source files: 50-100 files/second

**Related Commands:**

- /collection create - Create vector-only collection
- /index pdf - Index PDFs to vector only
- /index code - Index code to vector only
- /search semantic - Search vector index
- /search text - Search full-text index

**Implementation:**

- RDR-009: Dual indexing strategy
- RDR-006: Claude Code integration

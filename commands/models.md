---
description: Manage embedding models
argument-hint: <list> [options]
---

Manage and view available embedding models for vector search.

**Subcommands:**

- list: List all available embedding models with details

**Options:**

- --json: Output in JSON format

**Examples:**

```text
/models list
/models list --json
```

**Execution:**

```bash
cd ${CLAUDE_PLUGIN_ROOT}
arc models $ARGUMENTS
```

**Available Models:**

The list command shows:
- Model name (for --model flags)
- Dimensions (vector size)
- Backend (fastembed, sentence-transformers)
- Best use case (PDFs, code, general)
- Model ID (HuggingFace identifier)

**Current Models:**

**For Documents/PDFs:**
- **stella** (1024D): Best for documents, PDFs, general text
- **bge-large** (1024D): General purpose, high quality
- **modernbert** (1024D): Newer general-purpose model

**For Source Code:**
- **jina-code** (768D): Optimized for code, cross-language
- **jina-v2-code** (768D): Alternative code model

**For General Use:**
- **bge** (1024D): High-quality general embeddings
- **bge-small** (384D): Faster, smaller, lower quality

**Model Selection Tips:**

1. **Match content type:**
   - PDFs/docs → stella or modernbert
   - Source code → jina-code
   - Mixed → stella or bge

2. **Consider dimensions:**
   - Higher dimensions (1024D) = better quality, more storage
   - Lower dimensions (384D, 768D) = faster, less storage

3. **Backend matters:**
   - fastembed: Faster, optimized, limited models
   - sentence-transformers: More models, HuggingFace ecosystem

4. **Collection consistency:**
   - Use same model for all documents in a collection
   - Cannot mix dimensions in one vector space

**Downloading Models:**

Models auto-download on first use (~1-2GB):
- Cached in ~/.arcaneum/models/
- Reused across indexing operations
- Use --offline flag to require cached models

**Pre-download for offline use:**

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jinaai/jina-embeddings-v2-base-code')"
```

**Related Commands:**

- /collection create - Create collection with specific model
- /index pdfs - Index with model selection
- /index source - Index with model selection

**Implementation:**

- RDR-002: Embedding client architecture
- RDR-006: Model listing CLI
- arcaneum-142: Multi-backend support

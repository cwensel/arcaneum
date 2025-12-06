---
description: Manage embedding models
argument-hint: list [--json]
---

Manage and view available embedding models for vector search.

**IMPORTANT:** You must specify a subcommand (currently only `list` is available).

**Subcommands (required):**

- `list`: List all available embedding models with details

**Options:**

- --json: Output in JSON format

**Examples:**

```text
/models list
/models list --json
```

**Execution:**

```bash
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

- **jina-code-0.5b** (896D): **RECOMMENDED** - SOTA Sept 2025, 32K context, fast (default)
- **jina-code-1.5b** (1536D): SOTA Sept 2025, 32K context, highest quality
- **codesage-large** (1024D): CodeSage V2, Dec 2024, 9 languages
- **jina-code** (768D): Legacy v2 model, 8K context
- **nomic-code** (3584D): 7B params, 6 languages, slower but comprehensive

**For General Use:**

- **bge** (1024D): High-quality general embeddings
- **bge-small** (384D): Faster, smaller, lower quality

**Model Selection Tips:**

1. **Match content type:**
   - PDFs/docs → stella or modernbert
   - Source code → jina-code-0.5b (fast) or jina-code-1.5b (quality)
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

- Cached in `~/.cache/arcaneum/models` (XDG-compliant)
- Reused across indexing operations
- Use --offline flag to require cached models

**Pre-download for offline use:**

```bash
# Download the recommended code model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jinaai/jina-code-embeddings-0.5b')"

# Or the legacy v2 model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jinaai/jina-embeddings-v2-base-code')"
```

**Related Commands:**

- /collection create - Create collection with specific model
- /index pdf - Index with model selection
- /index code - Index with model selection

**Implementation:**

- RDR-002: Embedding client architecture
- RDR-006: Model listing CLI
- arcaneum-142: Multi-backend support

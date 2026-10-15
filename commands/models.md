---
description: Manage embedding models and remote embedding workers
argument-hint: <list|workers|deploy|serve> [options]
---

Manage and view available embedding models for vector search, and the optional
remote embedding workers that embed documents on spare machines.

**IMPORTANT:** You must specify a subcommand.

**Subcommands (required):**

- `list`: List all available embedding models with details
- `workers`: Show each configured remote embedding worker's state, version,
  code fingerprint, resident model and dependency versions
- `deploy [HOSTS...]`: Install, upgrade or reconfigure remote embedding workers
  over SSH, then check their vectors match this workstation's
- `serve`: Run a remote embedding worker (normally inside the container
  `deploy` creates; not for use on the workstation)

**Options:**

- --json: Output in JSON format (`list`, `workers`)
- --dry-run: Show each worker's current and desired state without changes (`deploy`)
- --build: Deploy a wheel built from this source tree (`deploy`; needed for
  source checkouts and unreleased builds)
- --wheel PATH: Deploy a given wheel (`deploy`)
- --refresh: Rebuild the worker image without cache (`deploy`)
- --no-verify: Skip the vector parity check (`deploy`)

**Examples:**

```text
/models list
/models list --json
/models workers
/models deploy --dry-run
/models deploy --build
```

Remote embedding workers are optional and configured under `remote_embedding`
in `~/.config/arcaneum/config.yaml`; see
`docs/guides/remote-embedding-workers.md`. Query embedding always stays local.

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

- **arctic-m** (768D): **DEFAULT** - stable FastEmbed retrieval model
- **stella** (1024D): Highest-quality opt-in document model, requires `arcaneum[sentence-transformers]`
- **mxbai-large** (1024D): High-quality FastEmbed document model
- **bge-large** (1024D): Legacy BGE document model

**For Source Code:**

- **jina-code** (768D): **DEFAULT** - stable FastEmbed code model
- **jina-code-st** (768D): Legacy SentenceTransformers code path, requires `arcaneum[sentence-transformers]`
- **jina-code-0.5b** (896D): Higher-quality opt-in code model, 32K context, requires `arcaneum[sentence-transformers]`
- **jina-code-1.5b** (1536D): SOTA Sept 2025, 32K context, highest quality, requires `arcaneum[sentence-transformers]`
- **codesage-large** (1024D): CodeSage V2, Dec 2024, 9 languages, requires `arcaneum[sentence-transformers]`
- **nomic-code** (3584D): 7B params, 6 languages, slower but comprehensive, requires `arcaneum[sentence-transformers]`

**For General Use:**

- **bge** (1024D): High-quality general embeddings
- **bge-small** (384D): Faster, smaller, lower quality

**Model Selection Tips:**

1. **Match content type:**
   - PDFs/docs → arctic-m (stable default), stella (quality), or mxbai-large (FastEmbed quality)
   - Source code → jina-code (stable FastEmbed default), jina-code-0.5b or jina-code-1.5b (quality extra)
   - Mixed → arctic-m or mxbai-large

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

Models auto-download on first use. FastEmbed defaults are part of the core install;
SentenceTransformers models require `arcaneum[sentence-transformers]`.

- Cached in `~/.cache/arcaneum/models` (XDG-compliant)
- Reused across indexing operations
- Use --offline flag to require cached models

**Pre-download for offline use:**

```bash
# Download an opt-in higher-quality code model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jinaai/jina-code-embeddings-0.5b')"

# Or the legacy v2 SentenceTransformers model
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

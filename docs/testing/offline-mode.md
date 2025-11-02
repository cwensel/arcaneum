# Corporate Network Setup

## For Self-Signed SSL Certificates

Set environment variables BEFORE running arc commands:

```bash
# Disable SSL verification (use on trusted networks only)
export PYTHONHTTPSVERIFY=0
export REQUESTS_CA_BUNDLE=""
export CURL_CA_BUNDLE=""

# Then run normally
arc index pdf ./pdfs --collection docs --model stella
arc index code ./code --collection code --model jina-code
```

## For Offline Mode (Recommended)

Set environment variables to use only cached models:

```bash
# Enable offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Then run (models must be pre-downloaded)
arc index pdf ./pdfs --collection docs --model stella
arc index code ./code --collection code --model jina-code
```

Or use the `--offline` flag:
```bash
arc index pdf ./pdfs --collection docs --model stella --offline
arc index code ./code --collection code --model jina-code --offline
```

## When to Use

✅ **Use --offline when:**
- Behind corporate proxy with SSL issues
- Models are already downloaded
- No internet connection
- Want to ensure no network calls

❌ **Don't use --offline when:**
- Models not downloaded yet
- Want to check for model updates

## Complete Corporate Network Workflow

### Step 1: Pre-download Models (one-time setup)

On a machine with working internet:

```bash
# Models will be downloaded to ~/.arcaneum/models automatically
# Just run arc commands normally and models will be cached

# Or download specific models manually
python -c "
from fastembed import TextEmbedding
from sentence_transformers import SentenceTransformer
from pathlib import Path

cache_dir = str(Path.home() / '.arcaneum' / 'models')

# FastEmbed models
TextEmbedding('BAAI/bge-large-en-v1.5', cache_dir=cache_dir)
TextEmbedding('jinaai/jina-embeddings-v3', cache_dir=cache_dir)

# SentenceTransformers models
SentenceTransformer('dunzhang/stella_en_1.5B_v5', cache_folder=cache_dir)
SentenceTransformer('jinaai/jina-embeddings-v2-base-code', cache_folder=cache_dir)
"
```

### Step 2: Copy Models to Corporate Machine

```bash
# Copy the ~/.arcaneum/models directory
scp -r ~/.arcaneum/models/ corporate-machine:~/.arcaneum/
```

### Step 3: Set Environment Variables

Add to your `~/.bashrc` or `~/.zshrc` (BEFORE running any arc commands):

```bash
# For offline mode (recommended - works reliably)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# For SSL bypass (set in shell, not via --no-verify-ssl flag)
export PYTHONHTTPSVERIFY=0
export REQUESTS_CA_BUNDLE=""
export CURL_CA_BUNDLE=""
export SSL_CERT_FILE=""
```

Then restart your shell or run:
```bash
source ~/.bashrc  # or ~/.zshrc
```

**IMPORTANT:** Environment variables MUST be set in your shell before running arc. The `--no-verify-ssl` flag has limitations due to fastembed's early initialization.

### Step 4: Use arc Normally

```bash
# Models are cached in ~/.arcaneum/models, offline mode active - no network/SSL issues!
arc index pdf ./pdfs --collection docs --model stella
arc index code ./code --collection code --model jina-code
```

**Why environment variables?**
- ✓ Set once, works for all commands
- ✓ More reliable than runtime flags (set before Python imports)
- ✓ Works with fastembed's early initialization
- ✓ No code changes needed

**Note:** Models are now stored in `~/.arcaneum/models/` by default. Use `arc config show-cache-dir` to verify the location.

## Verification

### With --offline:
```
Source Code Indexing Configuration
  Collection: code (type: code)
  Embedding: jinaai/jina-embeddings-v2-base-code
  Vector: jina-code
  Pipeline: Git Discover → AST Chunk → Embed (batched) → Upload
  Mode: Offline (cached models only)
```

### With --no-verify-ssl:
```
PDF Indexing Configuration
  Collection: docs (type: pdf)
  Model: stella → dunzhang/stella_en_1.5B_v5
  OCR: tesseract (eng)
  Pipeline: PDF → Extract → [OCR if needed] → Chunk → Embed → Upload
  Upload: Atomic per-document (safer)
  SSL Verification: Disabled (VPN mode)
```

Both prevent SSL certificate errors!

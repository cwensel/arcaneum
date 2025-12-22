# Corporate Network Setup

## SSL Certificate Handling

SSL certificate verification is **disabled by default** for compatibility with corporate proxies
that use self-signed certificates. This means arc commands work out of the box behind corporate VPNs.

```bash
# Just works - no configuration needed
arc search semantic "query" --collection MyCollection
arc index code ./code --collection code
```

To enable strict SSL verification (not recommended for corporate networks):

```bash
export ARC_SSL_VERIFY=true
arc search semantic "query" --collection MyCollection
```

## For Offline Mode (Air-gapped Networks)

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

## When to Use Offline Mode

Use `--offline` when:

- Models are already downloaded
- No internet connection (air-gapped)
- Want to ensure no network calls

Don't use `--offline` when:

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

### Step 3: Use arc Normally

```bash
# SSL bypass is automatic, models cached - just works!
arc index pdf ./pdfs --collection docs --model stella
arc index code ./code --collection code --model jina-code
arc search semantic "query" --collection MyCollection
```

For air-gapped networks (no internet at all), add offline mode:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

**Note:** Models are stored in `~/.arcaneum/models/` by default. Use `arc config show-cache-dir` to verify the location.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARC_SSL_VERIFY` | `false` | Set to `true` to enable strict SSL verification |
| `HF_HUB_OFFLINE` | `0` | Set to `1` for offline mode (no network calls) |
| `TRANSFORMERS_OFFLINE` | `0` | Set to `1` for offline mode |

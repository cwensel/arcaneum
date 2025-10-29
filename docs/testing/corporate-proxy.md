# Corporate Proxy SSL Workaround

If you're behind a corporate proxy with self-signed certificates, model downloads will fail with SSL errors.

## The Solution

Pre-download models on a machine with working SSL, then copy to your target machine.

### Step 1: Download Models

On a machine with working internet/SSL:

```bash
python scripts/download-models.py
```

This downloads all embedding models to `models_cache/` directory.

### Step 2: Copy to Target Machine

```bash
# Transfer the models_cache directory
scp -r models_cache/ user@target-machine:/path/to/arcaneum/

# Or use any file transfer method (shared drive, USB, etc.)
```

### Step 3: Use Normally

On your target machine:

```bash
bin/arc collection create Standards --model stella
bin/arc index pdfs ./Standards --collection Standards --model stella
```

Models load from cache - no network access needed!

## Why This Works

- FastEmbed checks `models_cache/` before attempting downloads
- If model is cached, no network request is made
- SSL errors are completely avoided

## Models Downloaded

The script downloads these models:
- `stella` (BAAI/bge-large-en-v1.5) - 1024D, general purpose
- `bge` (BAAI/bge-large-en-v1.5) - 1024D, precision
- `modernbert` (nomic-ai/modernbert-embed-base) - 768D, long context
- `jina` (jinaai/jina-embeddings-v2-base-code) - 768D, code + text

## Cache Location

Default: `./models_cache/`

To use a different location:
```bash
python scripts/download-models.py --cache-dir /custom/path
```

Then set environment variable:
```bash
export SENTENCE_TRANSFORMERS_HOME=/custom/path
```

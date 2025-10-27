# Offline Mode

Use `--offline` to prevent all network calls when models are cached.

## Usage

```bash
bin/arc index-pdfs ./pdfs --collection docs --model stella --offline
```

## What It Does

Sets these environment variables:
- `HF_HUB_OFFLINE=1` - Blocks all HuggingFace network calls
- `HF_HUB_DISABLE_TELEMETRY=1` - Disables telemetry
- `TRANSFORMERS_OFFLINE=1` - Alternative offline flag

## When to Use

✅ **Use --offline when:**
- Behind corporate proxy with SSL issues
- Models are already downloaded
- No internet connection
- Want to ensure no network calls

❌ **Don't use --offline when:**
- Models not downloaded yet
- Want to check for model updates

## Corporate Proxy Workflow

```bash
# 1. Pre-download models (on machine with working internet)
python scripts/download-models.py

# 2. Copy models_cache/ to target machine

# 3. Use offline mode (no SSL errors!)
bin/arc index-pdfs ./pdfs --collection docs --model stella --offline
```

## Verification

With `--offline`, you should see:
```
Indexing PDFs
  Directory: ./pdfs
  Collection: docs
  Model: stella
  Mode: Offline (cached models only)
```

And NO SSL errors!

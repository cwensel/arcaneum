#!/usr/bin/env python3
"""Pre-download embedding models for offline use.

Use this script on a machine with working SSL, then copy models_cache/ to target machine.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fastembed import TextEmbedding

MODELS = {
    "stella": "BAAI/bge-large-en-v1.5",
    "modernbert": "nomic-ai/modernbert-embed-base",
    "bge": "BAAI/bge-large-en-v1.5",
    "jina": "jinaai/jina-embeddings-v2-base-code",
}

def download_models(cache_dir: str = "./models_cache"):
    """Download all embedding models to cache directory."""
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir

    print(f"Downloading models to: {cache_dir}\n")

    for name, model_id in MODELS.items():
        print(f"📥 Downloading {name} ({model_id})...")
        try:
            model = TextEmbedding(model_name=model_id, cache_dir=cache_dir)
            print(f"✅ {name} downloaded successfully\n")
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}\n")

    print("=" * 60)
    print("✅ All models downloaded!")
    print(f"📁 Models cached in: {Path(cache_dir).absolute()}")
    print("\nTo use on another machine:")
    print(f"  1. Copy {cache_dir}/ directory to target machine")
    print(f"  2. Models will load from cache without downloading")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download embedding models for offline use")
    parser.add_argument(
        "--cache-dir",
        default="./models_cache",
        help="Directory to cache models (default: ./models_cache)"
    )
    args = parser.parse_args()

    download_models(args.cache_dir)

"""Global model cache for persistent model loading across CLI invocations.

This module implements a global in-memory cache for embedding models, allowing
models to be loaded once and reused across multiple CLI commands within the same
process. This saves 7-8 seconds per command for model initialization.

For persistent caching across separate CLI invocations (e.g., running multiple
arc commands in sequence), models would need to be stored in a daemon process
or pre-warmed on startup.

Architecture:
- Global _model_cache dict stores loaded models
- Thread-safe access with locks
- Automatic cleanup on process exit
- Optional pre-warming on demand

Usage:
    from arcaneum.embeddings.model_cache import get_cached_model, warm_models

    # Within a CLI command:
    model = get_cached_model("stella", cache_dir=..., use_gpu=...)
    embeddings = model.encode(texts)

    # Optional: pre-warm common models on startup
    warm_models(["stella", "bge-large"], cache_dir=..., use_gpu=...)
"""

import threading
from typing import Dict, Optional, List
from arcaneum.embeddings.client import EmbeddingClient, EMBEDDING_MODELS

# Global model cache with thread-safe access
_model_cache: Dict[str, EmbeddingClient] = {}
_cache_lock = threading.Lock()

# Track cache configuration to ensure consistency
_cache_config = {
    "cache_dir": None,
    "use_gpu": None,
    "verify_ssl": True,
}


def _get_cache_key(model_name: str, use_gpu: bool) -> str:
    """Generate a cache key for a model configuration."""
    return f"{model_name}:{use_gpu}"


def get_cached_model(
    model_name: str,
    cache_dir: str,
    use_gpu: bool = False,
    verify_ssl: bool = True,
) -> EmbeddingClient:
    """Get or create a cached embedding model.

    Models are cached in-memory for the lifetime of the process. This avoids
    repeated model initialization (7-8 seconds per model load).

    Args:
        model_name: Model identifier (stella, bge-large, jina-v3, etc.)
        cache_dir: Directory for cached model files
        use_gpu: Enable GPU acceleration
        verify_ssl: Whether to verify SSL certificates

    Returns:
        EmbeddingClient instance with model pre-loaded

    Note:
        Thread-safe. Multiple threads can safely call this concurrently.
        All models in cache use the same cache_dir and GPU settings.
    """
    with _cache_lock:
        cache_key = _get_cache_key(model_name, use_gpu)

        if cache_key in _model_cache:
            return _model_cache[cache_key]

        # Create new client and load model
        client = EmbeddingClient(
            cache_dir=cache_dir,
            verify_ssl=verify_ssl,
            use_gpu=use_gpu,
        )
        client.get_model(model_name)

        # Update configuration for consistency checking
        if _cache_config["cache_dir"] is None:
            _cache_config["cache_dir"] = cache_dir
            _cache_config["use_gpu"] = use_gpu
            _cache_config["verify_ssl"] = verify_ssl
        elif (
            _cache_config["cache_dir"] != cache_dir
            or _cache_config["use_gpu"] != use_gpu
            or _cache_config["verify_ssl"] != verify_ssl
        ):
            # Warn if cache is being used with different settings
            import warnings
            warnings.warn(
                f"Model cache was initialized with different settings. "
                f"Previous: cache_dir={_cache_config['cache_dir']}, "
                f"use_gpu={_cache_config['use_gpu']}. "
                f"Current: cache_dir={cache_dir}, use_gpu={use_gpu}. "
                f"Using cached instance with previous settings.",
                RuntimeWarning,
            )

        _model_cache[cache_key] = client
        return client


def warm_models(
    model_names: List[str],
    cache_dir: str,
    use_gpu: bool = False,
    verify_ssl: bool = True,
) -> None:
    """Pre-load embedding models for faster first use.

    Optional step to warm up the model cache at startup. Useful if you know
    which models will be used frequently.

    Args:
        model_names: List of model identifiers to pre-load
        cache_dir: Directory for cached model files
        use_gpu: Enable GPU acceleration
        verify_ssl: Whether to verify SSL certificates

    Example:
        warm_models(["stella", "bge-large"], cache_dir="~/.arcaneum/models", use_gpu=True)
    """
    for model_name in model_names:
        if model_name not in EMBEDDING_MODELS:
            print(f"⚠️  Unknown model: {model_name}. Skipping.")
            continue

        try:
            get_cached_model(
                model_name,
                cache_dir=cache_dir,
                use_gpu=use_gpu,
                verify_ssl=verify_ssl,
            )
            print(f"✓ Warmed: {model_name}")
        except Exception as e:
            print(f"✗ Failed to warm {model_name}: {e}")


def clear_cache() -> None:
    """Clear all cached models from memory.

    Useful for testing or freeing GPU memory. Models can be reloaded on demand
    after clearing.
    """
    with _cache_lock:
        _model_cache.clear()
        _cache_config["cache_dir"] = None
        _cache_config["use_gpu"] = None


def get_cache_info() -> Dict:
    """Get information about the current model cache.

    Returns:
        Dictionary with cache statistics (cached models, size, etc.)
    """
    with _cache_lock:
        return {
            "cached_models": list(_model_cache.keys()),
            "model_count": len(_model_cache),
            "config": _cache_config.copy(),
        }

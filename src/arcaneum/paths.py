"""Arcaneum directory and path management."""

import os
from pathlib import Path


def get_arcaneum_dir() -> Path:
    """Get the main Arcaneum directory (~/.arcaneum).

    Creates the directory if it doesn't exist.

    Returns:
        Path to ~/.arcaneum directory
    """
    arcaneum_dir = Path.home() / ".arcaneum"
    arcaneum_dir.mkdir(parents=True, exist_ok=True)
    return arcaneum_dir


def get_models_dir() -> Path:
    """Get the models cache directory (~/.arcaneum/models).

    Creates the directory if it doesn't exist.

    Returns:
        Path to ~/.arcaneum/models directory
    """
    models_dir = get_arcaneum_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_data_dir() -> Path:
    """Get the data directory (~/.arcaneum/data).

    Creates the directory if it doesn't exist.

    Returns:
        Path to ~/.arcaneum/data directory
    """
    data_dir = get_arcaneum_dir() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_qdrant_data_dir() -> Path:
    """Get the Qdrant data directory (~/.arcaneum/data/qdrant).

    Creates the directory if it doesn't exist.

    Returns:
        Path to ~/.arcaneum/data/qdrant directory
    """
    qdrant_dir = get_data_dir() / "qdrant"
    qdrant_dir.mkdir(parents=True, exist_ok=True)
    return qdrant_dir


def get_meilisearch_data_dir() -> Path:
    """Get the MeiliSearch data directory (~/.arcaneum/data/meilisearch).

    Creates the directory if it doesn't exist.

    Returns:
        Path to ~/.arcaneum/data/meilisearch directory
    """
    meilisearch_dir = get_data_dir() / "meilisearch"
    meilisearch_dir.mkdir(parents=True, exist_ok=True)
    return meilisearch_dir


def configure_model_cache_env():
    """Configure environment variables for model caching.

    Sets TRANSFORMERS_CACHE and HF_HOME to ~/.arcaneum/models if not already set by the user.
    This ensures models are downloaded to a predictable, user-accessible location.

    Note:
        Respects existing user-set environment variables. Only sets defaults if not already configured.
    """
    models_dir_str = str(get_models_dir())

    # Only set if user hasn't already configured these
    if "TRANSFORMERS_CACHE" not in os.environ:
        os.environ["TRANSFORMERS_CACHE"] = models_dir_str

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = models_dir_str

    # Also set SENTENCE_TRANSFORMERS_HOME for SentenceTransformers backend
    if "SENTENCE_TRANSFORMERS_HOME" not in os.environ:
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = models_dir_str


# Configure model cache on module import (so it's set before any model loading)
configure_model_cache_env()

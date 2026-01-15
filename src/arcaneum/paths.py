"""Arcaneum directory and path management.

XDG Base Directory Specification compliant:
- Cache (re-downloadable): ~/.cache/arcaneum/
- Data (user-created): ~/.local/share/arcaneum/

Legacy installations in ~/.arcaneum/ are auto-migrated on first access.
"""

import os
from pathlib import Path


def get_legacy_arcaneum_dir() -> Path:
    """Get the legacy Arcaneum directory (~/.arcaneum).

    This is the old location used before XDG compliance.
    Used for migration detection only.

    Returns:
        Path to ~/.arcaneum directory (may not exist)
    """
    return Path.home() / ".arcaneum"


def get_models_dir() -> Path:
    """Get the models cache directory (XDG-compliant: ~/.cache/arcaneum/models).

    Models are cached, re-downloadable data. Following XDG Base Directory spec,
    they belong in ~/.cache rather than user data directories.

    Creates the directory if it doesn't exist.

    Returns:
        Path to ~/.cache/arcaneum/models directory
    """
    cache_home = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    models_dir = Path(cache_home) / "arcaneum" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_data_dir() -> Path:
    """Get the data directory (XDG-compliant: ~/.local/share/arcaneum).

    User data (Qdrant/MeiliSearch databases) is essential, non-regenerable content.
    Following XDG Base Directory spec, this belongs in ~/.local/share.

    Creates the directory if it doesn't exist.

    Returns:
        Path to ~/.local/share/arcaneum directory
    """
    data_home = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    data_dir = Path(data_home) / "arcaneum"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_qdrant_data_dir() -> Path:
    """Get the Qdrant data directory (XDG-compliant: ~/.local/share/arcaneum/qdrant).

    Creates the directory if it doesn't exist.

    Returns:
        Path to ~/.local/share/arcaneum/qdrant directory
    """
    qdrant_dir = get_data_dir() / "qdrant"
    qdrant_dir.mkdir(parents=True, exist_ok=True)
    return qdrant_dir


def get_meilisearch_data_dir() -> Path:
    """Get the MeiliSearch data directory (XDG-compliant: ~/.local/share/arcaneum/meilisearch).

    Creates the directory if it doesn't exist.

    Returns:
        Path to ~/.local/share/arcaneum/meilisearch directory
    """
    meilisearch_dir = get_data_dir() / "meilisearch"
    meilisearch_dir.mkdir(parents=True, exist_ok=True)
    return meilisearch_dir


def get_config_dir() -> Path:
    """Get the config directory (XDG-compliant: ~/.config/arcaneum).

    Configuration files and secrets belong in ~/.config per XDG spec.
    Creates the directory if it doesn't exist.

    Returns:
        Path to ~/.config/arcaneum directory
    """
    config_home = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    config_dir = Path(config_home) / "arcaneum"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_meilisearch_api_key() -> str:
    """Get or generate the MeiliSearch API key.

    The key is stored in ~/.config/arcaneum/meilisearch.key and is
    auto-generated on first access. This ensures MeiliSearch runs in
    production mode (secure) without manual configuration.

    Returns:
        The MeiliSearch API key (32 characters, URL-safe base64)
    """
    import secrets
    import base64

    key_file = get_config_dir() / "meilisearch.key"

    # Check environment variable first (allows override)
    env_key = os.environ.get("MEILISEARCH_API_KEY")
    if env_key and len(env_key) >= 16:
        return env_key

    # Read existing key or generate new one
    if key_file.exists():
        key = key_file.read_text().strip()
        if len(key) >= 16:
            return key

    # Generate a new secure key (32 bytes = 43 chars base64, URL-safe)
    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('ascii').rstrip('=')

    # Save with restricted permissions (readable only by owner)
    key_file.write_text(key)
    key_file.chmod(0o600)

    return key


def configure_model_cache_env():
    """Configure environment variables for model caching (XDG-compliant).

    Sets HF_HOME to ~/.cache/arcaneum/models if not already set by the user.
    This ensures models are downloaded to an XDG-compliant cache location.

    Following XDG Base Directory Specification:
    - Models are cached (re-downloadable) → ~/.cache/arcaneum/models
    - User data (databases) → ~/.local/share/arcaneum

    Note:
        Respects existing user-set environment variables. Only sets defaults if not already configured.
        Uses HF_HOME (not deprecated TRANSFORMERS_CACHE) per transformers v5 guidance.
    """
    models_dir_str = str(get_models_dir())

    # Only set if user hasn't already configured
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = models_dir_str

    # Also set SENTENCE_TRANSFORMERS_HOME for SentenceTransformers backend
    if "SENTENCE_TRANSFORMERS_HOME" not in os.environ:
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = models_dir_str


# Configure model cache on module import (so it's set before any model loading)
configure_model_cache_env()

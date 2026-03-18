"""Unit tests for arcaneum.paths — XDG path functions and API key management."""

import os
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch, tmp_path):
    """Remove XDG env vars and redirect HOME to tmp_path for every test."""
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("MEILISEARCH_API_KEY", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    # Patch Path.home() so XDG defaults resolve under tmp_path
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    yield tmp_path


# --- get_models_dir ---

def test_get_models_dir_default(isolate_env):
    from arcaneum import paths as p
    import importlib
    importlib.reload(p)
    models_dir = p.get_models_dir()
    assert models_dir == isolate_env / ".cache" / "arcaneum" / "models"
    assert models_dir.is_dir()


def test_get_models_dir_custom_xdg(monkeypatch, tmp_path):
    custom_cache = tmp_path / "mycache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(custom_cache))
    from arcaneum import paths as p
    import importlib
    importlib.reload(p)
    models_dir = p.get_models_dir()
    assert models_dir == custom_cache / "arcaneum" / "models"
    assert models_dir.is_dir()


# --- get_data_dir ---

def test_get_data_dir_default(isolate_env):
    from arcaneum import paths as p
    import importlib
    importlib.reload(p)
    data_dir = p.get_data_dir()
    assert data_dir == isolate_env / ".local" / "share" / "arcaneum"
    assert data_dir.is_dir()


def test_get_data_dir_custom_xdg(monkeypatch, tmp_path):
    custom_data = tmp_path / "mydata"
    monkeypatch.setenv("XDG_DATA_HOME", str(custom_data))
    from arcaneum import paths as p
    import importlib
    importlib.reload(p)
    data_dir = p.get_data_dir()
    assert data_dir == custom_data / "arcaneum"
    assert data_dir.is_dir()


# --- get_config_dir ---

def test_get_config_dir_default(isolate_env):
    from arcaneum import paths as p
    import importlib
    importlib.reload(p)
    config_dir = p.get_config_dir()
    assert config_dir == isolate_env / ".config" / "arcaneum"
    assert config_dir.is_dir()


def test_get_config_dir_custom_xdg(monkeypatch, tmp_path):
    custom_config = tmp_path / "myconfig"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(custom_config))
    from arcaneum import paths as p
    import importlib
    importlib.reload(p)
    config_dir = p.get_config_dir()
    assert config_dir == custom_config / "arcaneum"
    assert config_dir.is_dir()


# --- get_qdrant_data_dir / get_meilisearch_data_dir ---

def test_get_qdrant_data_dir_is_subdirectory(isolate_env):
    from arcaneum import paths as p
    import importlib
    importlib.reload(p)
    qdrant_dir = p.get_qdrant_data_dir()
    assert qdrant_dir.name == "qdrant"
    assert qdrant_dir.parent == p.get_data_dir()
    assert qdrant_dir.is_dir()


def test_get_meilisearch_data_dir_is_subdirectory(isolate_env):
    from arcaneum import paths as p
    import importlib
    importlib.reload(p)
    ms_dir = p.get_meilisearch_data_dir()
    assert ms_dir.name == "meilisearch"
    assert ms_dir.parent == p.get_data_dir()
    assert ms_dir.is_dir()


# --- get_meilisearch_api_key ---

def test_get_meilisearch_api_key_generates_and_caches(isolate_env):
    from arcaneum import paths as p
    import importlib
    importlib.reload(p)
    key1 = p.get_meilisearch_api_key()
    key2 = p.get_meilisearch_api_key()
    assert key1 == key2
    assert len(key1) >= 16

    key_file = p.get_config_dir() / "meilisearch.key"
    assert key_file.exists()
    assert key_file.read_text().strip() == key1
    assert (key_file.stat().st_mode & 0o777) == 0o600


def test_get_meilisearch_api_key_env_override(monkeypatch, isolate_env):
    monkeypatch.setenv("MEILISEARCH_API_KEY", "my-secret-key-1234567890")
    from arcaneum import paths as p
    import importlib
    importlib.reload(p)
    key = p.get_meilisearch_api_key()
    assert key == "my-secret-key-1234567890"


def test_get_meilisearch_api_key_short_env_ignored(monkeypatch, isolate_env):
    monkeypatch.setenv("MEILISEARCH_API_KEY", "short")
    from arcaneum import paths as p
    import importlib
    importlib.reload(p)
    key = p.get_meilisearch_api_key()
    # Short env key is ignored; a real key is generated
    assert key != "short"
    assert len(key) >= 16


# --- configure_model_cache_env ---

def test_configure_model_cache_env_sets_defaults(monkeypatch, isolate_env):
    from arcaneum import paths as p
    import importlib
    importlib.reload(p)
    p.configure_model_cache_env()
    assert "HF_HOME" in os.environ
    assert "SENTENCE_TRANSFORMERS_HOME" in os.environ
    models_dir = str(p.get_models_dir())
    assert os.environ["HF_HOME"] == models_dir
    assert os.environ["SENTENCE_TRANSFORMERS_HOME"] == models_dir


def test_configure_model_cache_env_respects_existing(monkeypatch, isolate_env):
    monkeypatch.setenv("HF_HOME", "/custom/hf")
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", "/custom/st")
    from arcaneum import paths as p
    import importlib
    importlib.reload(p)
    p.configure_model_cache_env()
    assert os.environ["HF_HOME"] == "/custom/hf"
    assert os.environ["SENTENCE_TRANSFORMERS_HOME"] == "/custom/st"

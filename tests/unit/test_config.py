"""Unit tests for arcaneum.config — load/save config and Pydantic models."""

import pytest
import yaml
from pathlib import Path

from arcaneum.config import (
    ArcaneumConfig,
    ModelConfig,
    QdrantConfig,
    CacheConfig,
    CollectionTemplate,
    load_config,
    save_config,
)


MINIMAL_MODEL = {
    "name": "test/model",
    "dimensions": 768,
    "chunk_size": 512,
    "chunk_overlap": 64,
}

MINIMAL_CONFIG_DATA = {
    "models": {
        "test-model": MINIMAL_MODEL,
    }
}


@pytest.fixture
def config_file(tmp_path):
    """Write a minimal valid config YAML and return its path."""
    path = tmp_path / "arcaneum.yaml"
    path.write_text(yaml.dump(MINIMAL_CONFIG_DATA))
    return path


# --- ModelConfig ---

class TestModelConfig:
    def test_defaults(self):
        m = ModelConfig(**MINIMAL_MODEL)
        assert m.distance == "cosine"
        assert m.late_chunking is False
        assert m.char_to_token_ratio == pytest.approx(3.3)

    def test_custom_values(self):
        m = ModelConfig(
            name="test/model",
            dimensions=1536,
            chunk_size=1024,
            chunk_overlap=128,
            distance="dot",
            late_chunking=True,
        )
        assert m.distance == "dot"
        assert m.late_chunking is True

    def test_invalid_distance_rejected(self):
        with pytest.raises(Exception):
            ModelConfig(
                name="test/model",
                dimensions=768,
                chunk_size=512,
                chunk_overlap=64,
                distance="invalid",
            )


# --- QdrantConfig ---

class TestQdrantConfig:
    def test_defaults(self):
        q = QdrantConfig()
        assert q.url == "http://localhost:6333"
        assert q.timeout == 120
        assert q.search_timeout == 60

    def test_custom_url(self):
        q = QdrantConfig(url="http://myhost:7777")
        assert q.url == "http://myhost:7777"


# --- ArcaneumConfig ---

class TestArcaneumConfig:
    def test_minimal_config(self):
        cfg = ArcaneumConfig(**MINIMAL_CONFIG_DATA)
        assert "test-model" in cfg.models
        assert isinstance(cfg.qdrant, QdrantConfig)
        assert isinstance(cfg.cache, CacheConfig)
        assert cfg.collections == {}

    def test_missing_models_rejected(self):
        with pytest.raises(Exception):
            ArcaneumConfig()  # models is required


# --- load_config ---

class TestLoadConfig:
    def test_loads_valid_yaml(self, config_file):
        cfg = load_config(config_file)
        assert isinstance(cfg, ArcaneumConfig)
        assert "test-model" in cfg.models
        assert cfg.models["test-model"].name == "test/model"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_loads_qdrant_section(self, tmp_path):
        data = dict(MINIMAL_CONFIG_DATA)
        data["qdrant"] = {"url": "http://remotehost:6333", "timeout": 60}
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(data))
        cfg = load_config(path)
        assert cfg.qdrant.url == "http://remotehost:6333"
        assert cfg.qdrant.timeout == 60

    def test_invalid_yaml_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("models:\n  bad: {missing_required_fields: true}\n")
        with pytest.raises(Exception):
            load_config(path)


# --- save_config ---

class TestSaveConfig:
    def test_save_creates_file(self, tmp_path):
        cfg = ArcaneumConfig(**MINIMAL_CONFIG_DATA)
        path = tmp_path / "out" / "config.yaml"
        save_config(cfg, path)
        assert path.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        cfg = ArcaneumConfig(**MINIMAL_CONFIG_DATA)
        path = tmp_path / "deep" / "nested" / "config.yaml"
        save_config(cfg, path)
        assert path.exists()

    def test_save_writes_valid_yaml(self, tmp_path):
        cfg = ArcaneumConfig(**MINIMAL_CONFIG_DATA)
        path = tmp_path / "config.yaml"
        save_config(cfg, path)
        data = yaml.safe_load(path.read_text())
        assert "models" in data
        assert "test-model" in data["models"]

    def test_round_trip(self, tmp_path):
        cfg = ArcaneumConfig(**MINIMAL_CONFIG_DATA)
        path = tmp_path / "config.yaml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.models["test-model"].name == cfg.models["test-model"].name
        assert loaded.models["test-model"].dimensions == cfg.models["test-model"].dimensions
        assert loaded.qdrant.url == cfg.qdrant.url

    def test_round_trip_with_collection(self, tmp_path):
        data = dict(MINIMAL_CONFIG_DATA)
        data["collections"] = {
            "MyCorpus": {
                "models": ["test-model"],
                "hnsw_m": 32,
                "hnsw_ef_construct": 200,
                "on_disk_payload": False,
                "indexes": ["title"],
            }
        }
        cfg = ArcaneumConfig(**data)
        path = tmp_path / "config.yaml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert "MyCorpus" in loaded.collections
        assert loaded.collections["MyCorpus"].hnsw_m == 32

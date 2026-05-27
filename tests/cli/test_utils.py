"""Tests for CLI utility functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import Distance, VectorParams

from arcaneum.cli.sync import read_path_list
from arcaneum.cli.utils import (
    build_vectors_config,
    create_qdrant_client,
    get_model_dimensions,
    validate_models,
)


class TestGetModelDimensions:
    """Tests for get_model_dimensions function."""

    def test_unknown_model_raises_value_error(self):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_model_dimensions("unknown-model-xyz")

        error_msg = str(exc_info.value)
        assert "Unknown model" in error_msg
        assert "unknown-model-xyz" in error_msg
        assert "Available models" in error_msg


class TestValidateModels:
    """Tests for validate_models function."""

    def test_invalid_model_raises_value_error(self):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_models(["stella", "invalid-model"])

        error_msg = str(exc_info.value)
        assert "Unknown model" in error_msg
        assert "invalid-model" in error_msg


class TestBuildVectorsConfig:
    """Tests for build_vectors_config function."""

    def test_single_model_config(self):
        """Test building config for single model."""
        config = build_vectors_config(["stella"])

        assert isinstance(config, dict)
        assert "stella" in config
        assert isinstance(config["stella"], VectorParams)
        assert config["stella"].distance == Distance.COSINE

    def test_multiple_models_config(self):
        """Test building config for multiple models."""
        config = build_vectors_config(["stella", "jina-code-0.5b"])

        assert len(config) == 2
        assert "stella" in config
        assert "jina-code-0.5b" in config

        # Both should have correct structure
        for model_name, params in config.items():
            assert isinstance(params, VectorParams)
            assert params.distance == Distance.COSINE
            assert params.size > 0

    def test_empty_list_returns_empty_dict(self):
        """Test that empty model list returns empty dict."""
        config = build_vectors_config([])
        assert config == {}

    def test_dimensions_match_model(self):
        """Test that dimensions in config match model dimensions."""
        config = build_vectors_config(["stella"])
        expected_dims = get_model_dimensions("stella")
        assert config["stella"].size == expected_dims


class TestCreateQdrantClient:
    """Tests for create_qdrant_client function."""

    @patch.dict("os.environ", {}, clear=True)
    @patch("arcaneum.cli.utils.QdrantClient")
    def test_defaults(self, mock_client_class):
        """Test client creation with defaults."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        result = create_qdrant_client(config_path=Path("/does/not/exist.yaml"))

        mock_client_class.assert_called_once_with(
            url="http://localhost:6333",
            api_key=None,
            timeout=120,
        )
        assert result == mock_client

    @patch.dict(
        "os.environ",
        {
            "QDRANT_URL": "https://qdrant.example",
            "QDRANT_API_KEY": "env-secret",
        },
        clear=True,
    )
    @patch("arcaneum.cli.utils.QdrantClient")
    def test_unprefixed_env_configures_hosted_qdrant(self, mock_client_class):
        """Test QDRANT_* environment variable support."""
        create_qdrant_client(config_path=Path("/does/not/exist.yaml"))

        mock_client_class.assert_called_once_with(
            url="https://qdrant.example",
            api_key="env-secret",
            timeout=120,
        )

    @patch.dict(
        "os.environ",
        {
            "QDRANT_URL": "https://qdrant.example",
            "ARC_QDRANT_URL": "https://arc-qdrant.example",
            "QDRANT_API_KEY": "env-secret",
            "ARC_QDRANT_API_KEY": "arc-secret",
        },
        clear=True,
    )
    @patch("arcaneum.cli.utils.QdrantClient")
    def test_prefixed_env_takes_precedence(self, mock_client_class):
        """Test ARC_QDRANT_* wins over QDRANT_*."""
        create_qdrant_client(config_path=Path("/does/not/exist.yaml"))

        mock_client_class.assert_called_once_with(
            url="https://arc-qdrant.example",
            api_key="arc-secret",
            timeout=120,
        )

    @patch.dict("os.environ", {}, clear=True)
    @patch("arcaneum.cli.utils.QdrantClient")
    def test_config_file_credentials(self, mock_client_class, tmp_path):
        """Test Qdrant API key can be loaded from config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
models:
  test-model:
    name: test/model
    dimensions: 768
    chunk_size: 512
    chunk_overlap: 64
qdrant:
  url: https://config-qdrant.example
  api_key: config-secret
  timeout: 90
  search_timeout: 30
""")

        create_qdrant_client(config_path=config_path)

        mock_client_class.assert_called_once_with(
            url="https://config-qdrant.example",
            api_key="config-secret",
            timeout=90,
        )

    @patch.dict("os.environ", {}, clear=True)
    @patch("arcaneum.cli.utils.QdrantClient")
    def test_default_config_path_uses_xdg_config(self, mock_client_class, tmp_path, monkeypatch):
        """Test default config path uses XDG config directory."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        config_dir = tmp_path / ".config" / "arcaneum"
        config_dir.mkdir(parents=True)
        (config_dir / "config.yaml").write_text("""
models:
  test-model:
    name: test/model
    dimensions: 768
    chunk_size: 512
    chunk_overlap: 64
qdrant:
  url: https://xdg-qdrant.example
  api_key: xdg-secret
  timeout: 45
""")

        create_qdrant_client()

        mock_client_class.assert_called_once_with(
            url="https://xdg-qdrant.example",
            api_key="xdg-secret",
            timeout=45,
        )

    @patch.dict("os.environ", {}, clear=True)
    @patch("arcaneum.cli.utils.QdrantClient")
    def test_legacy_config_is_migrated_to_xdg_config(
        self, mock_client_class, tmp_path, monkeypatch
    ):
        """Test legacy config is copied to XDG config on first default load."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        xdg_dir = tmp_path / ".config" / "arcaneum"
        xdg_dir.mkdir(parents=True)
        (xdg_dir / ".config.yaml.tmp").write_text("stale")
        legacy_dir = tmp_path / ".arcaneum"
        legacy_dir.mkdir()
        legacy_config = legacy_dir / "config.yaml"
        legacy_config.write_text("""
models:
  test-model:
    name: test/model
    dimensions: 768
    chunk_size: 512
    chunk_overlap: 64
qdrant:
  url: https://legacy-qdrant.example
  api_key: legacy-secret
  timeout: 75
""")

        create_qdrant_client()

        xdg_config = tmp_path / ".config" / "arcaneum" / "config.yaml"
        assert xdg_config.exists()
        assert xdg_config.read_text() == legacy_config.read_text()
        assert xdg_config.stat().st_mode & 0o777 == 0o600
        mock_client_class.assert_called_once_with(
            url="https://legacy-qdrant.example",
            api_key="legacy-secret",
            timeout=75,
        )

    @patch.dict("os.environ", {}, clear=True)
    @patch("arcaneum.cli.utils.QdrantClient")
    def test_legacy_config_read_failure_does_not_create_xdg_config(
        self, mock_client_class, tmp_path, monkeypatch
    ):
        """Test a failed legacy read does not leave an empty XDG config."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        legacy_dir = tmp_path / ".arcaneum"
        legacy_dir.mkdir()
        (legacy_dir / "config.yaml").write_bytes(b"\xff")

        create_qdrant_client()

        assert not (tmp_path / ".config" / "arcaneum" / "config.yaml").exists()
        mock_client_class.assert_called_once_with(
            url="http://localhost:6333",
            api_key=None,
            timeout=120,
        )

    @patch.dict("os.environ", {}, clear=True)
    @patch("arcaneum.cli.utils.QdrantClient")
    def test_legacy_config_used_when_xdg_parent_is_file(
        self, mock_client_class, tmp_path, monkeypatch
    ):
        """Test legacy config is still read when XDG parent cannot be a directory."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        xdg_parent = tmp_path / ".config" / "arcaneum"
        xdg_parent.parent.mkdir()
        xdg_parent.write_text("not a directory")
        legacy_dir = tmp_path / ".arcaneum"
        legacy_dir.mkdir()
        (legacy_dir / "config.yaml").write_text("""
models:
  test-model:
    name: test/model
    dimensions: 768
    chunk_size: 512
    chunk_overlap: 64
qdrant:
  url: https://legacy-qdrant.example
  api_key: legacy-secret
  timeout: 75
""")

        create_qdrant_client()

        mock_client_class.assert_called_once_with(
            url="https://legacy-qdrant.example",
            api_key="legacy-secret",
            timeout=75,
        )

    @patch.dict(
        "os.environ",
        {
            "ARC_QDRANT_URL": "https://env-qdrant.example",
            "ARC_QDRANT_API_KEY": "env-secret",
        },
        clear=True,
    )
    @patch("arcaneum.cli.utils.QdrantClient")
    def test_env_overrides_default_xdg_config(self, mock_client_class, tmp_path, monkeypatch):
        """Test environment variables still override XDG config credentials."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        config_dir = tmp_path / ".config" / "arcaneum"
        config_dir.mkdir(parents=True)
        (config_dir / "config.yaml").write_text("""
models:
  test-model:
    name: test/model
    dimensions: 768
    chunk_size: 512
    chunk_overlap: 64
qdrant:
  url: https://xdg-qdrant.example
  api_key: xdg-secret
  timeout: 90
""")

        create_qdrant_client()

        mock_client_class.assert_called_once_with(
            url="https://env-qdrant.example",
            api_key="env-secret",
            timeout=90,
        )

    @patch.dict(
        "os.environ",
        {
            "ARC_QDRANT_URL": "https://env-qdrant.example",
            "ARC_QDRANT_API_KEY": "env-secret",
        },
        clear=True,
    )
    @patch("arcaneum.cli.utils.QdrantClient")
    def test_env_only_default_does_not_create_xdg_config_dir(
        self, mock_client_class, tmp_path, monkeypatch
    ):
        """Test env-only config does not create a default XDG directory."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        create_qdrant_client()

        assert not (tmp_path / ".config" / "arcaneum").exists()
        mock_client_class.assert_called_once_with(
            url="https://env-qdrant.example",
            api_key="env-secret",
            timeout=120,
        )

    @patch.dict(
        "os.environ",
        {
            "ARC_QDRANT_URL": "https://env-qdrant.example",
            "ARC_QDRANT_API_KEY": "env-secret",
        },
        clear=True,
    )
    @patch("arcaneum.cli.utils.QdrantClient")
    def test_params_override_env(self, mock_client_class):
        """Test explicit params override environment credentials."""
        create_qdrant_client(
            url="https://param-qdrant.example",
            api_key="param-secret",
            timeout=10,
            config_path=Path("/does/not/exist.yaml"),
        )

        mock_client_class.assert_called_once_with(
            url="https://param-qdrant.example",
            api_key="param-secret",
            timeout=10,
        )

    @patch.dict("os.environ", {"ARC_QDRANT_API_KEY": "super-secret"}, clear=True)
    @patch("arcaneum.cli.utils.QdrantClient")
    def test_api_key_not_logged(self, mock_client_class, caplog):
        """Test secret values are not written to logs."""
        create_qdrant_client(config_path=Path("/does/not/exist.yaml"))

        assert "super-secret" not in caplog.text


class TestCreateMeiliClient:
    """Tests for create_meili_client function."""

    @patch("arcaneum.fulltext.client.FullTextClient")
    @patch("arcaneum.paths.get_meilisearch_api_key")
    def test_default_url_and_key(self, mock_get_key, mock_client_class):
        """Test client creation with defaults."""
        from arcaneum.cli.utils import create_meili_client

        mock_get_key.return_value = "test-api-key"
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        result = create_meili_client()

        mock_client_class.assert_called_once_with("http://localhost:7700", "test-api-key")
        assert result == mock_client

    @patch("arcaneum.fulltext.client.FullTextClient")
    @patch("arcaneum.paths.get_meilisearch_api_key")
    def test_custom_url(self, mock_get_key, mock_client_class):
        """Test client creation with custom URL."""
        from arcaneum.cli.utils import create_meili_client

        mock_get_key.return_value = "test-api-key"
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        create_meili_client(url="http://custom:8080")

        mock_client_class.assert_called_once_with("http://custom:8080", "test-api-key")

    @patch("arcaneum.fulltext.client.FullTextClient")
    @patch("arcaneum.paths.get_meilisearch_api_key")
    def test_custom_api_key(self, mock_get_key, mock_client_class):
        """Test client creation with custom API key."""
        from arcaneum.cli.utils import create_meili_client

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        create_meili_client(api_key="custom-key")

        # Should NOT call get_meilisearch_api_key when custom key provided
        mock_client_class.assert_called_once()
        call_args = mock_client_class.call_args[0]
        assert call_args[1] == "custom-key"

    @patch.dict("os.environ", {"MEILISEARCH_URL": "http://env-url:9000"})
    @patch("arcaneum.fulltext.client.FullTextClient")
    @patch("arcaneum.paths.get_meilisearch_api_key")
    def test_url_from_environment(self, mock_get_key, mock_client_class):
        """Test client creation with URL from environment."""
        from arcaneum.cli.utils import create_meili_client

        mock_get_key.return_value = "test-api-key"
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        create_meili_client()

        mock_client_class.assert_called_once_with("http://env-url:9000", "test-api-key")


class TestReadPathList:
    """Tests for read_path_list function (from sync.py)."""

    def test_read_paths_from_file(self, tmp_path):
        """Test reading paths from a file."""
        # Create test directories and files
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        file1 = tmp_path / "file1.txt"
        file1.write_text("content")

        # Create path list file
        path_list = tmp_path / "paths.txt"
        path_list.write_text(f"{dir1}\n{file1}\n")

        paths = read_path_list(str(path_list))

        assert len(paths) == 2
        assert dir1 in paths
        assert file1 in paths

    def test_skips_comments_and_empty_lines(self, tmp_path):
        """Test that comments and empty lines are skipped."""
        dir1 = tmp_path / "dir1"
        dir1.mkdir()

        path_list = tmp_path / "paths.txt"
        path_list.write_text(f"# This is a comment\n\n{dir1}\n\n# Another comment\n")

        paths = read_path_list(str(path_list))

        assert len(paths) == 1
        assert dir1 in paths

    def test_skips_nonexistent_paths(self, tmp_path):
        """Test that nonexistent paths are skipped with warning."""
        dir1 = tmp_path / "dir1"
        dir1.mkdir()

        path_list = tmp_path / "paths.txt"
        path_list.write_text(f"{dir1}\n/nonexistent/path\n")

        paths = read_path_list(str(path_list))

        assert len(paths) == 1
        assert dir1 in paths

    def test_resolves_relative_paths(self, tmp_path, monkeypatch):
        """Test that relative paths are resolved."""
        monkeypatch.chdir(tmp_path)

        dir1 = tmp_path / "relative_dir"
        dir1.mkdir()

        path_list = tmp_path / "paths.txt"
        path_list.write_text("relative_dir\n")

        paths = read_path_list(str(path_list))

        assert len(paths) == 1
        assert paths[0].is_absolute()
        assert paths[0] == dir1

    def test_returns_empty_list_for_nonexistent_file(self, tmp_path):
        """Test that nonexistent path list file returns empty list."""
        paths = read_path_list(str(tmp_path / "nonexistent.txt"))
        assert paths == []

    def test_accepts_both_files_and_directories(self, tmp_path):
        """Test that both files and directories are accepted."""
        dir1 = tmp_path / "mydir"
        dir1.mkdir()
        file1 = tmp_path / "myfile.pdf"
        file1.write_text("content")
        file2 = tmp_path / "another.md"
        file2.write_text("content")

        path_list = tmp_path / "paths.txt"
        path_list.write_text(f"{dir1}\n{file1}\n{file2}\n")

        paths = read_path_list(str(path_list))

        assert len(paths) == 3
        assert dir1 in paths
        assert file1 in paths
        assert file2 in paths

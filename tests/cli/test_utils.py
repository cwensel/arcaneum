"""Tests for CLI utility functions."""

import pytest
from unittest.mock import patch, MagicMock
from qdrant_client.models import VectorParams, Distance

from arcaneum.cli.utils import (
    get_model_dimensions,
    validate_models,
    build_vectors_config,
)


class TestGetModelDimensions:
    """Tests for get_model_dimensions function."""

    def test_known_model_stella(self):
        """Test getting dimensions for known model 'stella'."""
        dims = get_model_dimensions('stella')
        assert isinstance(dims, int)
        assert dims > 0

    def test_known_model_jina_code(self):
        """Test getting dimensions for known model 'jina-code-0.5b'."""
        dims = get_model_dimensions('jina-code-0.5b')
        assert isinstance(dims, int)
        assert dims > 0

    def test_unknown_model_raises_value_error(self):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_model_dimensions('unknown-model-xyz')

        error_msg = str(exc_info.value)
        assert "Unknown model" in error_msg
        assert "unknown-model-xyz" in error_msg
        assert "Available models" in error_msg


class TestValidateModels:
    """Tests for validate_models function."""

    def test_valid_single_model(self):
        """Test validation of single valid model."""
        # Should not raise
        validate_models(['stella'])

    def test_valid_multiple_models(self):
        """Test validation of multiple valid models."""
        # Should not raise
        validate_models(['stella', 'jina-code-0.5b'])

    def test_invalid_model_raises_value_error(self):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_models(['stella', 'invalid-model'])

        error_msg = str(exc_info.value)
        assert "Unknown model" in error_msg
        assert "invalid-model" in error_msg

    def test_empty_list_does_not_raise(self):
        """Test that empty list does not raise."""
        # Should not raise
        validate_models([])


class TestBuildVectorsConfig:
    """Tests for build_vectors_config function."""

    def test_single_model_config(self):
        """Test building config for single model."""
        config = build_vectors_config(['stella'])

        assert isinstance(config, dict)
        assert 'stella' in config
        assert isinstance(config['stella'], VectorParams)
        assert config['stella'].distance == Distance.COSINE

    def test_multiple_models_config(self):
        """Test building config for multiple models."""
        config = build_vectors_config(['stella', 'jina-code-0.5b'])

        assert len(config) == 2
        assert 'stella' in config
        assert 'jina-code-0.5b' in config

        # Both should have correct structure
        for model_name, params in config.items():
            assert isinstance(params, VectorParams)
            assert params.distance == Distance.COSINE
            assert params.size > 0

    def test_invalid_model_raises_value_error(self):
        """Test that invalid model in list raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            build_vectors_config(['stella', 'nonexistent-model'])

        assert "Unknown model" in str(exc_info.value)

    def test_empty_list_returns_empty_dict(self):
        """Test that empty model list returns empty dict."""
        config = build_vectors_config([])
        assert config == {}

    def test_dimensions_match_model(self):
        """Test that dimensions in config match model dimensions."""
        config = build_vectors_config(['stella'])
        expected_dims = get_model_dimensions('stella')
        assert config['stella'].size == expected_dims


class TestCreateMeiliClient:
    """Tests for create_meili_client function."""

    @patch('arcaneum.fulltext.client.FullTextClient')
    @patch('arcaneum.paths.get_meilisearch_api_key')
    def test_default_url_and_key(self, mock_get_key, mock_client_class):
        """Test client creation with defaults."""
        from arcaneum.cli.utils import create_meili_client

        mock_get_key.return_value = 'test-api-key'
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        result = create_meili_client()

        mock_client_class.assert_called_once_with('http://localhost:7700', 'test-api-key')
        assert result == mock_client

    @patch('arcaneum.fulltext.client.FullTextClient')
    @patch('arcaneum.paths.get_meilisearch_api_key')
    def test_custom_url(self, mock_get_key, mock_client_class):
        """Test client creation with custom URL."""
        from arcaneum.cli.utils import create_meili_client

        mock_get_key.return_value = 'test-api-key'
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        result = create_meili_client(url='http://custom:8080')

        mock_client_class.assert_called_once_with('http://custom:8080', 'test-api-key')

    @patch('arcaneum.fulltext.client.FullTextClient')
    @patch('arcaneum.paths.get_meilisearch_api_key')
    def test_custom_api_key(self, mock_get_key, mock_client_class):
        """Test client creation with custom API key."""
        from arcaneum.cli.utils import create_meili_client

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        result = create_meili_client(api_key='custom-key')

        # Should NOT call get_meilisearch_api_key when custom key provided
        mock_client_class.assert_called_once()
        call_args = mock_client_class.call_args[0]
        assert call_args[1] == 'custom-key'

    @patch.dict('os.environ', {'MEILISEARCH_URL': 'http://env-url:9000'})
    @patch('arcaneum.fulltext.client.FullTextClient')
    @patch('arcaneum.paths.get_meilisearch_api_key')
    def test_url_from_environment(self, mock_get_key, mock_client_class):
        """Test client creation with URL from environment."""
        from arcaneum.cli.utils import create_meili_client

        mock_get_key.return_value = 'test-api-key'
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        result = create_meili_client()

        mock_client_class.assert_called_once_with('http://env-url:9000', 'test-api-key')

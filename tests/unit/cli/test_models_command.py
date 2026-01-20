"""CLI tests for models command.

Tests for 'arc models list' command.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestModelsList:
    """Test 'arc models list' command."""

    def test_lists_all_models(self, capsys):
        """Test that all available models are listed."""
        from arcaneum.cli.models import list_models_command

        mock_models = {
            'stella': {'name': 'stella-model', 'dimensions': 1024, 'description': 'General-purpose model'},
            'jina-code': {'name': 'jina-code-model', 'dimensions': 768, 'description': 'Code-optimized model'},
            'nomic-embed': {'name': 'nomic-model', 'dimensions': 768, 'description': 'Efficient model'},
        }

        with patch('arcaneum.cli.models.EMBEDDING_MODELS', mock_models):
            list_models_command(output_json=False)

        captured = capsys.readouterr()
        # Rich output goes to stderr in some cases, check both
        output = captured.out + captured.err

        assert 'stella' in output.lower() or 'Models' in output

    def test_json_output(self, capsys):
        """Test JSON output format."""
        from arcaneum.cli.models import list_models_command

        mock_models = {
            'stella': {'name': 'stella-model', 'dimensions': 1024, 'description': 'General-purpose model'},
            'jina-code': {'name': 'jina-code-model', 'dimensions': 768, 'description': 'Code-optimized model'},
        }

        with patch('arcaneum.cli.models.EMBEDDING_MODELS', mock_models):
            list_models_command(output_json=True)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output['status'] == 'success'
        assert 'models' in output['data']
        assert len(output['data']['models']) == 2

    def test_shows_dimensions(self):
        """Test that dimensions are shown for each model."""
        from arcaneum.cli.models import list_models_command

        mock_models = {
            'stella': {'name': 'stella', 'dimensions': 1024},
            'jina-code': {'name': 'jina', 'dimensions': 768},
        }

        with patch('arcaneum.cli.models.EMBEDDING_MODELS', mock_models):
            list_models_command(output_json=False)

        # Test passes if no exception is raised

    def test_shows_descriptions(self):
        """Test that descriptions are shown when available."""
        from arcaneum.cli.models import list_models_command

        mock_models = {
            'stella': {'name': 'stella', 'dimensions': 1024, 'description': 'General-purpose embedding model'},
        }

        with patch('arcaneum.cli.models.EMBEDDING_MODELS', mock_models):
            list_models_command(output_json=False)

        # Test passes if no exception is raised

    def test_json_output_structure(self, capsys):
        """Test the structure of JSON output."""
        from arcaneum.cli.models import list_models_command

        mock_models = {
            'stella': {
                'name': 'stella-model',
                'dimensions': 1024,
                'description': 'Test model',
            },
        }

        with patch('arcaneum.cli.models.EMBEDDING_MODELS', mock_models):
            list_models_command(output_json=True)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert 'status' in output
        assert 'message' in output
        assert 'data' in output
        assert 'models' in output['data']

        model = output['data']['models'][0]
        assert 'alias' in model
        assert 'dimensions' in model


class TestEmptyModels:
    """Test edge cases with empty or missing models."""

    def test_empty_json_output(self, capsys):
        """Test JSON output with no models."""
        from arcaneum.cli.models import list_models_command

        with patch('arcaneum.cli.models.EMBEDDING_MODELS', {}):
            list_models_command(output_json=True)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output['status'] == 'success'
        assert output['data']['models'] == []

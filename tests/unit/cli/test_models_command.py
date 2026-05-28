"""CLI tests for models command.

Tests for 'arc models list' command.
"""

import json
from unittest.mock import patch


class TestModelsList:
    """Test 'arc models list' command."""

    def test_json_output(self, capsys):
        """Test JSON output format."""
        from arcaneum.cli.models import list_models_command

        mock_models = {
            'stella': {
                'name': 'stella-model',
                'dimensions': 1024,
                'description': 'General-purpose model',
            },
            'jina-code': {
                'name': 'jina-code-model',
                'dimensions': 768,
                'description': 'Code-optimized model',
            },
        }

        with patch('arcaneum.cli.models.EMBEDDING_MODELS', mock_models):
            list_models_command(output_json=True)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output['status'] == 'success'
        assert 'models' in output['data']
        assert len(output['data']['models']) == 2

    def test_shows_dimensions(self):
        """Test that dimension values are rendered in the table output."""
        from arcaneum.cli import models as models_module
        from arcaneum.cli.models import list_models_command

        mock_models = {
            'stella': {'name': 'stella', 'dimensions': 1024},
            'jina-code': {'name': 'jina', 'dimensions': 768},
        }

        # Capture the Table that the command hands to console.print by mocking
        # the console itself; table output goes through rich, not stdout.
        with patch('arcaneum.cli.models.EMBEDDING_MODELS', mock_models):
            with patch.object(models_module, 'console') as mock_console:
                list_models_command(output_json=False)

        # The first printed object should be the table containing both models'
        # dimensions as row data.
        first_call_arg = mock_console.print.call_args_list[0].args[0]
        # Rich Table exposes columns via `.columns`
        dim_column = next((c for c in first_call_arg.columns if c.header == 'Dims'), None)
        assert dim_column is not None, "No 'Dims' column found"
        cells = list(dim_column.cells)
        assert '1024' in cells
        assert '768' in cells

    def test_shows_descriptions(self):
        """Test that descriptions are rendered in the table output."""
        from arcaneum.cli import models as models_module
        from arcaneum.cli.models import list_models_command

        mock_models = {
            'stella': {
                'name': 'stella',
                'dimensions': 1024,
                'description': 'General-purpose embedding model',
            },
        }

        with patch('arcaneum.cli.models.EMBEDDING_MODELS', mock_models):
            with patch.object(models_module, 'console') as mock_console:
                list_models_command(output_json=False)

        first_call_arg = mock_console.print.call_args_list[0].args[0]
        desc_column = next((c for c in first_call_arg.columns if c.header == 'Description'), None)
        assert desc_column is not None, "No 'Description' column found"
        cells = list(desc_column.cells)
        assert 'General-purpose embedding model' in cells

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
        assert 'backend' in model
        assert 'recommended_for' in model
        assert 'default_for' in model
        assert 'support_tier' in model
        assert 'install_extra' in model
        assert 'prompt_policy' in model
        assert 'context_limit' in model
        assert 'params_billions' in model
        assert 'risk_tier' in model
        assert 'hardware' in model
        assert 'suggested_batches' in model
        assert 'reindex_warning' in model

    def test_json_output_includes_llm_selection_policy(self, capsys):
        """Test JSON exposes defaults, risk, hardware, and batch guidance."""
        from arcaneum.cli.models import list_models_command

        mock_models = {
            'arctic-m': {
                'name': 'snowflake/snowflake-arctic-embed-m',
                'dimensions': 768,
                'backend': 'fastembed',
                'description': 'stable docs model',
                'recommended_for': 'docs',
            },
            'nomic-code': {
                'name': 'nomic-ai/nomic-embed-code',
                'dimensions': 3584,
                'backend': 'sentence-transformers',
                'params_billions': 7.0,
                'mps_max_batch': 1,
                'recommended_for': 'code',
            },
        }

        with patch('arcaneum.cli.models.EMBEDDING_MODELS', mock_models):
            list_models_command(output_json=True)

        output = json.loads(capsys.readouterr().out)
        models = {model['alias']: model for model in output['data']['models']}

        assert models['arctic-m']['default_for'] == ['pdf', 'markdown']
        assert models['arctic-m']['support_tier'] == 'stable-default'
        assert models['arctic-m']['hardware']['mps'] == 'experimental-coreml'
        assert models['arctic-m']['suggested_batches']['outer'] == {
            'cpu': 512,
            'cuda': 768,
            'mps': 512,
        }
        assert models['arctic-m']['suggested_batches']['sentence_transformers_encode'] is None

        assert models['nomic-code']['risk_tier'] == 'very-high'
        assert models['nomic-code']['support_tier'] == 'gpu-opt-in'
        assert models['nomic-code']['hardware']['cuda'] is True
        assert models['nomic-code']['suggested_batches']['outer'] == {
            'cpu': 512,
            'cuda': 512,
            'mps': 128,
        }
        assert models['nomic-code']['suggested_batches']['sentence_transformers_encode'] == {
            'cpu_max': 256,
            'gpu_dynamic': 'memory-probed',
            'mps_max': 1,
        }

    def test_table_output_exposes_selection_columns(self):
        """Test table output shows the compact LLM selection columns."""
        from arcaneum.cli import models as models_module
        from arcaneum.cli.models import list_models_command

        mock_models = {
            'arctic-m': {
                'name': 'snowflake/snowflake-arctic-embed-m',
                'dimensions': 768,
                'backend': 'fastembed',
                'recommended_for': 'docs',
                'description': 'stable docs model',
            },
        }

        with patch('arcaneum.cli.models.EMBEDDING_MODELS', mock_models):
            with patch.object(models_module, 'console') as mock_console:
                list_models_command(output_json=False)

        table = mock_console.print.call_args_list[0].args[0]
        headers = [column.header for column in table.columns]

        assert 'Backend' in headers
        assert 'Use' in headers
        assert 'Default' in headers
        assert 'Tier' in headers
        assert 'Risk' in headers
        assert 'Batch' in headers


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

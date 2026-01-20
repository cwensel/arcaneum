"""CLI tests for 'arc doctor' command.

Tests for system diagnostics and setup verification.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestDoctorChecks:
    """Test individual diagnostic checks."""

    def test_python_version_check_success(self):
        """Test Python version check when version is sufficient."""
        from arcaneum.cli.doctor import check_python_version

        with patch.object(sys, 'version_info', (3, 12, 0)):
            success, message = check_python_version()

        assert success is True
        assert "3.12" in message

    def test_python_version_check_failure(self):
        """Test Python version check when version is insufficient."""
        from arcaneum.cli.doctor import check_python_version

        with patch.object(sys, 'version_info', (3, 10, 0)):
            success, message = check_python_version()

        assert success is False
        assert "3.10" in message

    def test_dependency_check_installed(self):
        """Test dependency check when package is installed."""
        from arcaneum.cli.doctor import check_dependency

        # Test with a package we know is installed
        success, message = check_dependency("pytest")

        assert success is True
        assert "pytest" in message

    def test_dependency_check_not_installed(self):
        """Test dependency check when package is not installed."""
        from arcaneum.cli.doctor import check_dependency

        success, message = check_dependency("nonexistent_package_xyz")

        assert success is False
        assert "not installed" in message

    def test_qdrant_connection_success(self):
        """Test Qdrant connection check when server is available."""
        from arcaneum.cli.doctor import check_qdrant_connection

        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(), MagicMock()]
        mock_client.get_collections.return_value = mock_collections

        with patch('arcaneum.cli.doctor.create_qdrant_client', return_value=mock_client):
            success, message = check_qdrant_connection()

        assert success is True
        assert "connected" in message.lower()
        assert "2 collections" in message

    def test_qdrant_connection_failure(self):
        """Test Qdrant connection check when server is not available."""
        from arcaneum.cli.doctor import check_qdrant_connection

        with patch('arcaneum.cli.doctor.create_qdrant_client') as mock_create:
            mock_create.side_effect = Exception("Connection refused")
            success, message = check_qdrant_connection()

        assert success is False
        assert "failed" in message.lower()

    def test_meilisearch_connection_success(self):
        """Test MeiliSearch connection check when server is available."""
        from arcaneum.cli.doctor import check_meilisearch_connection

        mock_client = MagicMock()
        mock_client.health.return_value = {'status': 'available'}
        mock_client.get_indexes.return_value = {'results': [MagicMock(), MagicMock()]}

        with patch('meilisearch.Client', return_value=mock_client):
            success, message = check_meilisearch_connection()

        assert success is True
        assert "connected" in message.lower()

    def test_meilisearch_connection_failure(self):
        """Test MeiliSearch connection check when server is not available."""
        from arcaneum.cli.doctor import check_meilisearch_connection

        with patch('meilisearch.Client') as mock_client_class:
            mock_client_class.side_effect = Exception("Connection refused")
            success, message = check_meilisearch_connection()

        # MeiliSearch is optional, so it returns None for status
        assert success is None or success is False
        assert "failed" in message.lower() or "optional" in message.lower()

    def test_embedding_models_check(self):
        """Test embedding models availability check."""
        from arcaneum.cli.doctor import check_embedding_models

        mock_models = {
            'stella': {'dimensions': 1024, 'available': True},
            'jina-code': {'dimensions': 768, 'available': True},
        }

        # Import happens inside function, so patch at the source
        with patch('arcaneum.embeddings.client.EMBEDDING_MODELS', mock_models):
            success, message = check_embedding_models()

        assert success is True
        assert "available" in message.lower()

    def test_temp_dir_writable(self, temp_dir):
        """Test temp directory writability check."""
        from arcaneum.cli.doctor import check_temp_dir_writable

        with patch('tempfile.gettempdir', return_value=str(temp_dir)):
            success, message = check_temp_dir_writable()

        assert success is True
        assert "writable" in message.lower()

    def test_environment_vars_check_with_vars(self, clean_env):
        """Test environment variables check when vars are set."""
        from arcaneum.cli.doctor import check_environment_vars
        import os

        os.environ['QDRANT_URL'] = 'http://localhost:6333'

        success, message = check_environment_vars(verbose=True)

        assert success is True
        assert 'QDRANT_URL' in message

    def test_environment_vars_check_no_vars(self, clean_env):
        """Test environment variables check when no vars are set."""
        from arcaneum.cli.doctor import check_environment_vars

        success, message = check_environment_vars()

        # Returns None when no env vars are set (using defaults)
        assert success is None
        assert "default" in message.lower()


class TestDoctorOutput:
    """Test doctor command output formats."""

    def test_json_format(self, capsys):
        """Test JSON output format."""
        from arcaneum.cli.doctor import doctor_command

        # Mock all checks to return success
        with patch('arcaneum.cli.doctor.check_python_version', return_value=(True, "Python 3.12")):
            with patch('arcaneum.cli.doctor.check_dependency', return_value=(True, "installed")):
                with patch('arcaneum.cli.doctor.check_qdrant_connection', return_value=(True, "connected")):
                    with patch('arcaneum.cli.doctor.check_meilisearch_connection', return_value=(True, "connected")):
                        with patch('arcaneum.cli.doctor.check_embedding_models', return_value=(True, "available")):
                            with patch('arcaneum.cli.doctor.check_temp_dir_writable', return_value=(True, "writable")):
                                with patch('arcaneum.cli.doctor.check_environment_vars', return_value=(None, "defaults")):
                                    doctor_command(verbose=False, output_json=True)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output['status'] == 'success'
        assert 'data' in output
        assert 'checks' in output['data']
        assert 'summary' in output['data']

    def test_table_format(self, capsys):
        """Test table output format."""
        from arcaneum.cli.doctor import doctor_command

        # Mock all checks to return success
        with patch('arcaneum.cli.doctor.check_python_version', return_value=(True, "Python 3.12")):
            with patch('arcaneum.cli.doctor.check_dependency', return_value=(True, "installed")):
                with patch('arcaneum.cli.doctor.check_qdrant_connection', return_value=(True, "connected")):
                    with patch('arcaneum.cli.doctor.check_meilisearch_connection', return_value=(True, "connected")):
                        with patch('arcaneum.cli.doctor.check_embedding_models', return_value=(True, "available")):
                            with patch('arcaneum.cli.doctor.check_temp_dir_writable', return_value=(True, "writable")):
                                with patch('arcaneum.cli.doctor.check_environment_vars', return_value=(None, "defaults")):
                                    doctor_command(verbose=False, output_json=False)

        captured = capsys.readouterr()
        # Table format should show check names
        assert 'Python' in captured.out or 'Diagnostics' in captured.out

    def test_exit_code_all_pass(self):
        """Test exit code when all checks pass."""
        from arcaneum.cli.doctor import doctor_command
        from arcaneum.cli.errors import EXIT_SUCCESS

        with patch('arcaneum.cli.doctor.check_python_version', return_value=(True, "Python 3.12")):
            with patch('arcaneum.cli.doctor.check_dependency', return_value=(True, "installed")):
                with patch('arcaneum.cli.doctor.check_qdrant_connection', return_value=(True, "connected")):
                    with patch('arcaneum.cli.doctor.check_meilisearch_connection', return_value=(None, "optional")):
                        with patch('arcaneum.cli.doctor.check_embedding_models', return_value=(True, "available")):
                            with patch('arcaneum.cli.doctor.check_temp_dir_writable', return_value=(True, "writable")):
                                with patch('arcaneum.cli.doctor.check_environment_vars', return_value=(None, "defaults")):
                                    result = doctor_command(verbose=False, output_json=True)

        assert result == EXIT_SUCCESS

    def test_exit_code_some_fail(self):
        """Test exit code when some checks fail."""
        from arcaneum.cli.doctor import doctor_command
        from arcaneum.cli.errors import EXIT_ERROR

        with patch('arcaneum.cli.doctor.check_python_version', return_value=(True, "Python 3.12")):
            with patch('arcaneum.cli.doctor.check_dependency', return_value=(False, "not installed")):
                with patch('arcaneum.cli.doctor.check_qdrant_connection', return_value=(False, "failed")):
                    with patch('arcaneum.cli.doctor.check_meilisearch_connection', return_value=(None, "optional")):
                        with patch('arcaneum.cli.doctor.check_embedding_models', return_value=(True, "available")):
                            with patch('arcaneum.cli.doctor.check_temp_dir_writable', return_value=(True, "writable")):
                                with patch('arcaneum.cli.doctor.check_environment_vars', return_value=(None, "defaults")):
                                    result = doctor_command(verbose=False, output_json=True)

        assert result == EXIT_ERROR


class TestDoctorVerboseMode:
    """Test doctor command verbose mode."""

    def test_verbose_shows_more_details(self, capsys):
        """Test that verbose mode shows more detailed information."""
        from arcaneum.cli.doctor import doctor_command

        with patch('arcaneum.cli.doctor.check_python_version', return_value=(True, "Python 3.12")):
            with patch('arcaneum.cli.doctor.check_dependency', return_value=(True, "qdrant-client 1.0.0")):
                with patch('arcaneum.cli.doctor.check_qdrant_connection', return_value=(True, "connected (5 collections)")):
                    with patch('arcaneum.cli.doctor.check_meilisearch_connection', return_value=(None, "optional")):
                        with patch('arcaneum.cli.doctor.check_embedding_models', return_value=(True, "3 models available")):
                            with patch('arcaneum.cli.doctor.check_temp_dir_writable', return_value=(True, "/tmp writable")):
                                with patch('arcaneum.cli.doctor.check_environment_vars', return_value=(True, "QDRANT_URL set")):
                                    doctor_command(verbose=True, output_json=False)

        captured = capsys.readouterr()
        # Verbose should show more detail
        assert 'Python' in captured.out or 'Diagnostics' in captured.out


class TestDoctorIntegration:
    """Integration-style tests for doctor command."""

    def test_full_doctor_command_json(self, capsys, mock_qdrant_client):
        """Test full doctor command with JSON output."""
        from arcaneum.cli.doctor import doctor_command

        with patch('arcaneum.cli.doctor.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('meilisearch.Client') as mock_meili:
                mock_meili_instance = MagicMock()
                mock_meili_instance.health.return_value = {'status': 'available'}
                mock_meili_instance.get_indexes.return_value = {'results': []}
                mock_meili.return_value = mock_meili_instance

                result = doctor_command(verbose=False, output_json=True)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert 'status' in output
        assert 'data' in output
        assert isinstance(output['data']['checks'], list)

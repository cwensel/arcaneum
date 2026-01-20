"""CLI tests for MeiliSearch indexes commands.

Tests for 'arc indexes' subcommands: create, list, info, delete, verify, items, export, import, list-projects, delete-project.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestIndexesCreate:
    """Test 'arc indexes create' command."""

    def test_create_with_type(self, mock_meili_client, capsys):
        """Test creating index with type."""
        from arcaneum.cli.fulltext import create_index
        from click.testing import CliRunner

        mock_meili_client.index_exists.return_value = False

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            with patch('arcaneum.cli.fulltext.get_index_settings', return_value={'searchableAttributes': ['content'], 'filterableAttributes': ['language']}):
                result = runner.invoke(create_index, ['TestIndex', '--type', 'code', '--json'])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output['status'] == 'success'
        mock_meili_client.create_index.assert_called_once()

    def test_create_already_exists(self, mock_meili_client):
        """Test error when index already exists."""
        from arcaneum.cli.fulltext import create_index
        from arcaneum.cli.errors import InvalidArgumentError
        from click.testing import CliRunner

        mock_meili_client.index_exists.return_value = True

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            result = runner.invoke(create_index, ['TestIndex', '--type', 'code'])

        # Command should fail with exception
        assert result.exit_code != 0
        # Exception is raised, check the exception type
        assert result.exception is not None

    def test_json_output(self, mock_meili_client, capsys):
        """Test JSON output format."""
        from arcaneum.cli.fulltext import create_index
        from click.testing import CliRunner

        mock_meili_client.index_exists.return_value = False

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            with patch('arcaneum.cli.fulltext.get_index_settings', return_value={'searchableAttributes': [], 'filterableAttributes': []}):
                result = runner.invoke(create_index, ['TestIndex', '--type', 'pdf', '--json'])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert 'status' in output
        assert 'data' in output


class TestIndexesList:
    """Test 'arc indexes list' command."""

    def test_empty_indexes(self, mock_meili_client, capsys):
        """Test listing when no indexes exist."""
        from arcaneum.cli.fulltext import list_indexes
        from click.testing import CliRunner

        mock_meili_client.list_indexes.return_value = []

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            result = runner.invoke(list_indexes, ['--json'])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output['status'] == 'success'
        assert output['data']['indexes'] == []

    def test_with_indexes(self, mock_meili_client, capsys):
        """Test listing when indexes exist."""
        from arcaneum.cli.fulltext import list_indexes
        from click.testing import CliRunner

        mock_meili_client.list_indexes.return_value = [
            {'uid': 'Index1', 'primaryKey': 'id', 'createdAt': '2024-01-01T00:00:00Z'},
            {'uid': 'Index2', 'primaryKey': 'id', 'createdAt': '2024-01-02T00:00:00Z'},
        ]

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            result = runner.invoke(list_indexes, ['--json'])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output['status'] == 'success'
        assert len(output['data']['indexes']) == 2

    def test_json_output(self, mock_meili_client):
        """Test JSON output format."""
        from arcaneum.cli.fulltext import list_indexes
        from click.testing import CliRunner

        mock_meili_client.list_indexes.return_value = [{'uid': 'Test', 'primaryKey': 'id'}]

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            result = runner.invoke(list_indexes, ['--json'])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert 'indexes' in output['data']


class TestIndexesInfo:
    """Test 'arc indexes info' command."""

    def test_shows_settings(self, mock_meili_client):
        """Test that info shows index settings."""
        from arcaneum.cli.fulltext import index_info
        from click.testing import CliRunner

        mock_meili_client.index_exists.return_value = True
        mock_meili_client.get_index.return_value = MagicMock(primary_key='id')
        mock_meili_client.get_index_stats.return_value = {'numberOfDocuments': 100, 'isIndexing': False}
        mock_meili_client.get_index_settings.return_value = {
            'searchableAttributes': ['content', 'title'],
            'filterableAttributes': ['language', 'project'],
            'sortableAttributes': ['created_at'],
            'typoTolerance': {'enabled': True},
        }

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            result = runner.invoke(index_info, ['TestIndex', '--json'])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output['status'] == 'success'
        assert 'settings' in output['data']
        assert 'stats' in output['data']

    def test_not_found(self, mock_meili_client):
        """Test error when index not found."""
        from arcaneum.cli.fulltext import index_info
        from click.testing import CliRunner

        mock_meili_client.index_exists.return_value = False

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            result = runner.invoke(index_info, ['NonExistent'])

        assert result.exit_code != 0


class TestIndexesDelete:
    """Test 'arc indexes delete' command."""

    def test_confirm_required(self, mock_meili_client):
        """Test that --confirm is required."""
        from arcaneum.cli.fulltext import delete_index
        from click.testing import CliRunner

        mock_meili_client.index_exists.return_value = True

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            result = runner.invoke(delete_index, ['TestIndex'], input='n\n')

        # Should prompt and user cancels
        assert 'Cancelled' in result.output or 'cancelled' in result.output.lower()

    def test_with_confirm(self, mock_meili_client):
        """Test deletion with --confirm flag."""
        from arcaneum.cli.fulltext import delete_index
        from click.testing import CliRunner

        mock_meili_client.index_exists.return_value = True

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            result = runner.invoke(delete_index, ['TestIndex', '--confirm'])

        assert result.exit_code == 0
        mock_meili_client.delete_index.assert_called_once_with('TestIndex')


class TestIndexesVerify:
    """Test 'arc indexes verify' command."""

    def test_healthy_index(self, mock_meili_client, mock_interaction_logger):
        """Test verifying a healthy index."""
        from arcaneum.cli.fulltext import verify_index
        from click.testing import CliRunner

        mock_meili_client.index_exists.return_value = True
        mock_meili_client.get_index_stats.return_value = {'numberOfDocuments': 100, 'isIndexing': False}
        mock_meili_client.get_index_settings.return_value = {
            'searchableAttributes': ['content'],
            'filterableAttributes': ['language'],
        }
        mock_meili_client.search.return_value = {'hits': [{'id': '1'}]}

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            result = runner.invoke(verify_index, ['TestIndex', '--json'])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output['data']['is_healthy'] is True

    def test_with_warnings(self, mock_meili_client, mock_interaction_logger):
        """Test verifying index with warnings."""
        from arcaneum.cli.fulltext import verify_index
        from click.testing import CliRunner

        mock_meili_client.index_exists.return_value = True
        mock_meili_client.get_index_stats.return_value = {'numberOfDocuments': 100, 'isIndexing': True}
        mock_meili_client.get_index_settings.return_value = {
            'searchableAttributes': ['*'],  # Wildcard triggers warning
            'filterableAttributes': [],  # Empty triggers warning
        }
        mock_meili_client.search.return_value = {'hits': [{'id': '1'}]}

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            result = runner.invoke(verify_index, ['TestIndex', '--json'])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert len(output['data']['warnings']) > 0


class TestIndexesItems:
    """Test 'arc indexes items' command."""

    def test_lists_files(self, mock_meili_client, mock_interaction_logger):
        """Test listing indexed files."""
        from arcaneum.cli.fulltext import list_items
        from click.testing import CliRunner

        mock_meili_client.index_exists.return_value = True
        mock_meili_client.search.return_value = {
            'hits': [
                {'file_path': '/path/to/file1.py', 'filename': 'file1.py', 'language': 'python'},
                {'file_path': '/path/to/file1.py', 'filename': 'file1.py', 'language': 'python'},
                {'file_path': '/path/to/file2.py', 'filename': 'file2.py', 'language': 'python'},
            ]
        }

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            result = runner.invoke(list_items, ['TestIndex', '--json'])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output['data']['total_items'] == 2  # Deduplicated

    def test_pagination(self, mock_meili_client, mock_interaction_logger):
        """Test pagination parameters."""
        from arcaneum.cli.fulltext import list_items
        from click.testing import CliRunner

        mock_meili_client.index_exists.return_value = True
        mock_meili_client.search.return_value = {'hits': []}

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            result = runner.invoke(list_items, ['TestIndex', '--limit', '50', '--json'])

        assert result.exit_code == 0


class TestIndexesExport:
    """Test 'arc indexes export' command."""

    def test_export_to_jsonl(self, mock_meili_client, mock_interaction_logger, temp_dir):
        """Test exporting index to JSONL."""
        from arcaneum.cli.fulltext import export_index
        from click.testing import CliRunner

        mock_meili_client.index_exists.return_value = True
        mock_meili_client.get_index.return_value = MagicMock(primary_key='id')
        mock_meili_client.get_index_stats.return_value = {'numberOfDocuments': 2}
        mock_meili_client.get_index_settings.return_value = {}
        mock_meili_client.search.side_effect = [
            {'hits': [{'id': '1', 'content': 'test1'}, {'id': '2', 'content': 'test2'}]},
            {'hits': []},
        ]

        output_file = temp_dir / 'export.jsonl'

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            result = runner.invoke(export_index, ['TestIndex', '-o', str(output_file), '--json'])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output['data']['exported_count'] == 2


class TestIndexesImport:
    """Test 'arc indexes import' command."""

    def test_import_from_jsonl(self, mock_meili_client, mock_interaction_logger, temp_dir):
        """Test importing index from JSONL."""
        from arcaneum.cli.fulltext import import_index
        from click.testing import CliRunner

        # Create a test export file
        export_file = temp_dir / 'export.jsonl'
        with open(export_file, 'w') as f:
            f.write('{"_type": "index_metadata", "name": "TestIndex", "primary_key": "id", "settings": {}}\n')
            f.write('{"_type": "document", "id": "1", "content": "test1"}\n')
            f.write('{"_type": "document", "id": "2", "content": "test2"}\n')

        mock_meili_client.index_exists.return_value = False

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            result = runner.invoke(import_index, [str(export_file), '--json'])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output['data']['imported_count'] == 2


class TestIndexesListProjects:
    """Test 'arc indexes list-projects' command."""

    def test_list_projects(self, mock_meili_client, mock_interaction_logger):
        """Test listing indexed git projects."""
        from arcaneum.cli.fulltext import list_projects
        from click.testing import CliRunner

        mock_meili_client.index_exists.return_value = True

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            with patch('arcaneum.indexing.fulltext.sync.GitCodeMetadataSync') as mock_sync_class:
                mock_sync = MagicMock()
                mock_sync.get_indexed_projects.return_value = {
                    'myrepo#main': 'abc123def456',
                    'myrepo#feature': 'def456abc123',
                }
                mock_sync_class.return_value = mock_sync

                result = runner.invoke(list_projects, ['TestIndex', '--json'])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output['data']['total_projects'] == 2


class TestIndexesDeleteProject:
    """Test 'arc indexes delete-project' command."""

    def test_delete_project(self, mock_meili_client, mock_interaction_logger):
        """Test deleting a git project."""
        from arcaneum.cli.fulltext import delete_project
        from click.testing import CliRunner

        mock_meili_client.index_exists.return_value = True

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            with patch('arcaneum.indexing.fulltext.sync.GitCodeMetadataSync') as mock_sync_class:
                mock_sync = MagicMock()
                mock_sync.get_project_document_count.return_value = 100
                mock_sync.delete_project_documents.return_value = 100
                mock_sync_class.return_value = mock_sync

                result = runner.invoke(delete_project, ['myrepo#main', '--index', 'TestIndex', '--confirm', '--json'])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output['data']['deleted_count'] == 100

    def test_delete_project_not_found(self, mock_meili_client, mock_interaction_logger):
        """Test deleting a project that doesn't exist."""
        from arcaneum.cli.fulltext import delete_project
        from click.testing import CliRunner

        mock_meili_client.index_exists.return_value = True

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            with patch('arcaneum.indexing.fulltext.sync.GitCodeMetadataSync') as mock_sync_class:
                mock_sync = MagicMock()
                mock_sync.get_project_document_count.return_value = 0
                mock_sync_class.return_value = mock_sync

                result = runner.invoke(delete_project, ['nonexistent#main', '--index', 'TestIndex', '--json'])

        output = json.loads(result.output)
        assert output['status'] == 'warning'
        assert output['data']['deleted_count'] == 0


class TestIndexesServerUnavailable:
    """Test error handling when MeiliSearch is unavailable."""

    def test_list_server_unavailable(self, mock_meili_client):
        """Test listing when server is unavailable."""
        from arcaneum.cli.fulltext import list_indexes
        from click.testing import CliRunner

        mock_meili_client.health_check.return_value = False

        runner = CliRunner()
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_meili_client):
            result = runner.invoke(list_indexes)

        # Should fail with non-zero exit code
        assert result.exit_code != 0
        # Error could be in output or as an exception
        error_info = result.output + (str(result.exception) if result.exception else '')
        assert 'not available' in error_info.lower() or 'MeiliSearch' in error_info

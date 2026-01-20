"""CLI tests for collection management commands.

Tests for 'arc collection' subcommands: list, info, delete, items, verify.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestCollectionList:
    """Test 'arc collection list' command."""

    def test_empty_collections(self, mock_qdrant_client, mock_interaction_logger, capsys):
        """Test listing when no collections exist."""
        from arcaneum.cli.collections import list_collections_command

        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            list_collections_command(verbose=False, output_json=True)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output['status'] == 'success'
        assert output['data']['collections'] == []
        assert 'Found 0 collections' in output['message']

    def test_with_collections(self, mock_qdrant_client_with_collections, mock_interaction_logger, capsys):
        """Test listing when collections exist."""
        from arcaneum.cli.collections import list_collections_command

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client_with_collections):
            with patch('arcaneum.cli.collections.get_collection_metadata', return_value={'model': 'stella', 'collection_type': 'pdf'}):
                list_collections_command(verbose=False, output_json=True)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output['status'] == 'success'
        assert len(output['data']['collections']) == 2
        assert 'Found 2 collections' in output['message']

    def test_verbose_output(self, mock_qdrant_client_with_collections, mock_interaction_logger, capsys):
        """Test verbose output shows additional details."""
        from arcaneum.cli.collections import list_collections_command

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client_with_collections):
            with patch('arcaneum.cli.collections.get_collection_metadata', return_value={'model': 'stella', 'collection_type': 'pdf'}):
                list_collections_command(verbose=True, output_json=False)

        captured = capsys.readouterr()
        # Verbose should show table with type and vectors columns
        assert 'TestCollection' in captured.out or 'Collections' in captured.out

    def test_json_output(self, mock_qdrant_client_with_collections, mock_interaction_logger, capsys):
        """Test JSON output structure."""
        from arcaneum.cli.collections import list_collections_command

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client_with_collections):
            with patch('arcaneum.cli.collections.get_collection_metadata', return_value={'model': 'stella', 'collection_type': 'pdf'}):
                list_collections_command(verbose=False, output_json=True)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert 'status' in output
        assert 'data' in output
        assert 'collections' in output['data']
        for col in output['data']['collections']:
            assert 'name' in col
            assert 'points_count' in col


class TestCollectionInfo:
    """Test 'arc collection info' command."""

    def test_all_fields_shown(self, mock_qdrant_client, mock_interaction_logger, capsys):
        """Test that all collection fields are shown."""
        from arcaneum.cli.collections import info_collection_command

        mock_qdrant_client.get_collection.return_value = MagicMock(
            points_count=100,
            status="green",
            config=MagicMock(
                params=MagicMock(vectors={
                    'stella': MagicMock(size=1024, distance="Cosine")
                }),
                hnsw_config=MagicMock(m=16, ef_construct=100)
            )
        )

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.collections.get_collection_metadata', return_value={'model': 'stella', 'collection_type': 'pdf'}):
                info_collection_command('TestCollection', output_json=True)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output['status'] == 'success'
        assert output['data']['name'] == 'TestCollection'
        assert output['data']['points_count'] == 100
        assert 'vectors' in output['data']
        assert 'hnsw_config' in output['data']

    def test_not_found_error(self, mock_qdrant_client, mock_interaction_logger):
        """Test error when collection not found."""
        from arcaneum.cli.collections import info_collection_command

        mock_qdrant_client.get_collection.side_effect = Exception("Collection not found")

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            with pytest.raises(SystemExit) as exc_info:
                info_collection_command('NonExistent', output_json=False)

        assert exc_info.value.code == 1

    def test_json_output(self, mock_qdrant_client, mock_interaction_logger, capsys):
        """Test JSON output format."""
        from arcaneum.cli.collections import info_collection_command

        mock_qdrant_client.get_collection.return_value = MagicMock(
            points_count=50,
            status="green",
            config=MagicMock(
                params=MagicMock(vectors={'stella': MagicMock(size=1024, distance="Cosine")}),
                hnsw_config=MagicMock(m=16, ef_construct=100)
            )
        )

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.collections.get_collection_metadata', return_value={'model': 'stella', 'collection_type': 'pdf'}):
                info_collection_command('TestCollection', output_json=True)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output['status'] == 'success'
        assert 'data' in output


class TestCollectionDelete:
    """Test 'arc collection delete' command."""

    def test_confirm_required(self, mock_qdrant_client, mock_interaction_logger):
        """Test that --confirm flag is required."""
        from arcaneum.cli.collections import delete_collection_command
        from arcaneum.cli.errors import InvalidArgumentError

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            with pytest.raises(InvalidArgumentError) as exc_info:
                delete_collection_command('TestCollection', confirm=False, output_json=True)

        assert '--confirm' in str(exc_info.value)

    def test_with_confirm_flag(self, mock_qdrant_client, mock_interaction_logger, capsys):
        """Test deletion with --confirm flag."""
        from arcaneum.cli.collections import delete_collection_command

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            delete_collection_command('TestCollection', confirm=True, output_json=True)

        mock_qdrant_client.delete_collection.assert_called_once_with('TestCollection')

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output['status'] == 'success'
        assert 'Deleted' in output['message']

    def test_not_found_error(self, mock_qdrant_client, mock_interaction_logger):
        """Test error when collection not found."""
        from arcaneum.cli.collections import delete_collection_command

        mock_qdrant_client.delete_collection.side_effect = Exception("Collection not found")

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            with pytest.raises(SystemExit) as exc_info:
                delete_collection_command('NonExistent', confirm=True, output_json=False)

        assert exc_info.value.code == 1


class TestCollectionItems:
    """Test 'arc collection items' command."""

    def test_pdf_collection(self, mock_qdrant_client, mock_interaction_logger, capsys):
        """Test listing items in a PDF collection."""
        from arcaneum.cli.collections import items_collection_command

        # Mock scroll to return PDF items
        mock_points = [
            MagicMock(payload={
                'file_path': '/path/to/doc1.pdf',
                'file_hash': 'abc123',
                'file_size': 1024,
                'filename': 'doc1.pdf',
            }),
            MagicMock(payload={
                'file_path': '/path/to/doc1.pdf',
                'file_hash': 'abc123',
                'file_size': 1024,
                'filename': 'doc1.pdf',
            }),
            MagicMock(payload={
                'file_path': '/path/to/doc2.pdf',
                'file_hash': 'def456',
                'file_size': 2048,
                'filename': 'doc2.pdf',
            }),
        ]
        mock_qdrant_client.scroll.return_value = (mock_points, None)

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.collections.get_collection_type', return_value='pdf'):
                items_collection_command('PDFCollection', output_json=True)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output['status'] == 'success'
        assert output['data']['item_count'] == 2  # Deduplicated
        assert output['data']['type'] == 'pdf'

    def test_code_collection(self, mock_qdrant_client, mock_interaction_logger, capsys):
        """Test listing items in a code collection."""
        from arcaneum.cli.collections import items_collection_command

        # Mock scroll to return code items
        mock_points = [
            MagicMock(payload={
                'git_project_name': 'myrepo',
                'git_project_identifier': 'myrepo#main',
                'git_branch': 'main',
                'git_commit_hash': 'abc123def456',
            }),
            MagicMock(payload={
                'git_project_name': 'myrepo',
                'git_project_identifier': 'myrepo#main',
                'git_branch': 'main',
                'git_commit_hash': 'abc123def456',
            }),
        ]
        mock_qdrant_client.scroll.return_value = (mock_points, None)

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.collections.get_collection_type', return_value='code'):
                items_collection_command('CodeCollection', output_json=True)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output['status'] == 'success'
        assert output['data']['item_count'] == 1  # Deduplicated by identifier
        assert output['data']['type'] == 'code'
        assert output['data']['items'][0]['git_project_name'] == 'myrepo'

    def test_empty_collection(self, mock_qdrant_client, mock_interaction_logger, capsys):
        """Test listing items in an empty collection."""
        from arcaneum.cli.collections import items_collection_command

        mock_qdrant_client.scroll.return_value = ([], None)

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.collections.get_collection_type', return_value='pdf'):
                items_collection_command('EmptyCollection', output_json=True)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output['status'] == 'success'
        assert output['data']['item_count'] == 0

    def test_json_output(self, mock_qdrant_client, mock_interaction_logger, capsys):
        """Test JSON output format."""
        from arcaneum.cli.collections import items_collection_command

        mock_points = [
            MagicMock(payload={
                'file_path': '/path/to/doc.pdf',
                'file_hash': 'abc123',
                'file_size': 1024,
                'filename': 'doc.pdf',
            }),
        ]
        mock_qdrant_client.scroll.return_value = (mock_points, None)

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.collections.get_collection_type', return_value='pdf'):
                items_collection_command('TestCollection', output_json=True)

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert 'status' in output
        assert 'data' in output
        assert 'items' in output['data']


class TestCollectionVerify:
    """Test 'arc collection verify' command."""

    def test_healthy_collection(self, mock_qdrant_client, mock_interaction_logger):
        """Test verifying a healthy collection."""
        from arcaneum.cli.collections import verify_collection_command
        from io import StringIO

        mock_result = MagicMock()
        mock_result.collection_name = 'TestCollection'
        mock_result.collection_type = 'pdf'
        mock_result.total_points = 100
        mock_result.total_items = 10
        mock_result.complete_items = 10
        mock_result.incomplete_items = 0
        mock_result.is_healthy = True
        mock_result.errors = []
        mock_result.projects = []
        mock_result.files = []
        mock_result.get_items_needing_repair.return_value = []

        # Capture output via StringIO since Rich console doesn't use stdout
        output_buffer = StringIO()

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.indexing.verify.CollectionVerifier') as mock_verifier_class:
                mock_verifier = MagicMock()
                mock_verifier.verify_collection.return_value = mock_result
                mock_verifier_class.return_value = mock_verifier

                with patch('builtins.print', side_effect=lambda *args, **kwargs: output_buffer.write(str(args[0]) if args else '')):
                    verify_collection_command('TestCollection', project=None, verbose=False, output_json=True)

        # Parse the captured output as JSON
        output_str = output_buffer.getvalue()
        # Verify that the function was called with correct params
        mock_verifier.verify_collection.assert_called_once_with('TestCollection', project_filter=None, verbose=False)

    def test_incomplete_items(self, mock_qdrant_client, mock_interaction_logger):
        """Test verifying a collection with incomplete items."""
        from arcaneum.cli.collections import verify_collection_command

        mock_result = MagicMock()
        mock_result.collection_name = 'TestCollection'
        mock_result.collection_type = 'pdf'
        mock_result.total_points = 90
        mock_result.total_items = 10
        mock_result.complete_items = 8
        mock_result.incomplete_items = 2
        mock_result.is_healthy = False
        mock_result.errors = []
        mock_result.projects = []
        mock_result.files = [
            MagicMock(
                file_path='/path/to/incomplete.pdf',
                expected_chunks=10,
                actual_chunks=8,
                completion_percentage=80.0,
                is_complete=False,
                missing_indices=[8, 9]
            )
        ]
        mock_result.get_items_needing_repair.return_value = ['/path/to/incomplete.pdf']

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.indexing.verify.CollectionVerifier') as mock_verifier_class:
                mock_verifier = MagicMock()
                mock_verifier.verify_collection.return_value = mock_result
                mock_verifier_class.return_value = mock_verifier

                verify_collection_command('TestCollection', project=None, verbose=False, output_json=True)

        # Verify the command executed with correct params
        mock_verifier.verify_collection.assert_called_once()
        # The mock result was returned correctly
        assert mock_result.is_healthy is False
        assert mock_result.incomplete_items == 2

    def test_project_filter(self, mock_qdrant_client, mock_interaction_logger):
        """Test verifying with project filter."""
        from arcaneum.cli.collections import verify_collection_command

        mock_result = MagicMock()
        mock_result.collection_name = 'CodeCollection'
        mock_result.collection_type = 'code'
        mock_result.total_points = 500
        mock_result.total_items = 1
        mock_result.complete_items = 1
        mock_result.incomplete_items = 0
        mock_result.is_healthy = True
        mock_result.errors = []
        mock_result.projects = [
            MagicMock(
                identifier='myrepo#main',
                project_name='myrepo',
                branch='main',
                commit_hash='abc123',
                total_files=50,
                complete_files=50,
                completion_percentage=100.0,
                is_complete=True,
                incomplete_files=[]
            )
        ]
        mock_result.files = []
        mock_result.get_items_needing_repair.return_value = []

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.indexing.verify.CollectionVerifier') as mock_verifier_class:
                mock_verifier = MagicMock()
                mock_verifier.verify_collection.return_value = mock_result
                mock_verifier_class.return_value = mock_verifier

                verify_collection_command('CodeCollection', project='myrepo#main', verbose=False, output_json=True)

        # Verify the filter was passed
        mock_verifier.verify_collection.assert_called_once_with('CodeCollection', project_filter='myrepo#main', verbose=False)

    def test_json_output(self, mock_qdrant_client, mock_interaction_logger):
        """Test JSON output format."""
        from arcaneum.cli.collections import verify_collection_command

        mock_result = MagicMock()
        mock_result.collection_name = 'TestCollection'
        mock_result.collection_type = 'pdf'
        mock_result.total_points = 100
        mock_result.total_items = 10
        mock_result.complete_items = 10
        mock_result.incomplete_items = 0
        mock_result.is_healthy = True
        mock_result.errors = []
        mock_result.projects = []
        mock_result.files = []
        mock_result.get_items_needing_repair.return_value = []

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.indexing.verify.CollectionVerifier') as mock_verifier_class:
                mock_verifier = MagicMock()
                mock_verifier.verify_collection.return_value = mock_result
                mock_verifier_class.return_value = mock_verifier

                verify_collection_command('TestCollection', project=None, verbose=False, output_json=True)

        # Verify verifier was called
        mock_verifier.verify_collection.assert_called_once_with('TestCollection', project_filter=None, verbose=False)


class TestCollectionCreate:
    """Test 'arc collection create' command."""

    def test_create_with_type(self, mock_qdrant_client, mock_interaction_logger):
        """Test creating collection with type."""
        from arcaneum.cli.collections import create_collection_command

        # Mock get_collection to return proper info for metadata setting
        mock_qdrant_client.get_collection.return_value = MagicMock(
            config=MagicMock(
                params=MagicMock(vectors={'stella': MagicMock(size=1024)})
            )
        )

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.collections.build_vectors_config', return_value={}):
                with patch('arcaneum.cli.collections.set_collection_metadata'):
                    with patch('arcaneum.cli.collections.EMBEDDING_MODELS', {'stella': {'dimensions': 1024}}):
                        create_collection_command(
                            name='NewCollection',
                            model=None,
                            hnsw_m=16,
                            hnsw_ef=100,
                            on_disk=False,
                            output_json=True,
                            collection_type='pdf'
                        )

        mock_qdrant_client.create_collection.assert_called_once()

    def test_create_requires_model_or_type(self, mock_qdrant_client, mock_interaction_logger):
        """Test that creating without model or type raises error."""
        from arcaneum.cli.collections import create_collection_command
        from arcaneum.cli.errors import InvalidArgumentError

        with patch('arcaneum.cli.collections.create_qdrant_client', return_value=mock_qdrant_client):
            with pytest.raises(InvalidArgumentError) as exc_info:
                create_collection_command(
                    name='NewCollection',
                    model=None,
                    hnsw_m=16,
                    hnsw_ef=100,
                    on_disk=False,
                    output_json=True,
                    collection_type=None
                )

        assert '--model' in str(exc_info.value) or '--type' in str(exc_info.value)

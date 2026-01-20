"""CLI tests for search commands.

Tests for 'arc search semantic' command.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestSearchSemantic:
    """Test 'arc search semantic' command."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mocked SearchEmbedder."""
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 1024
        return embedder

    @pytest.fixture
    def sample_search_results(self):
        """Sample search results from Qdrant."""
        return [
            MagicMock(
                id='doc1',
                score=0.95,
                payload={
                    'file_path': '/path/to/doc1.pdf',
                    'content': 'This is the first result about authentication.',
                    'page_number': 5,
                }
            ),
            MagicMock(
                id='doc2',
                score=0.85,
                payload={
                    'file_path': '/path/to/doc2.pdf',
                    'content': 'This is the second result about security.',
                    'page_number': 10,
                }
            ),
        ]

    def test_basic_search(self, mock_qdrant_client, mock_embedder, sample_search_results, capsys):
        """Test basic semantic search returns results."""
        from arcaneum.cli.search import search_command

        json_output = json.dumps({
            'status': 'success',
            'query': 'authentication',
            'collection': 'TestCollection',
            'results': [
                {'id': 'doc1', 'score': 0.95, 'file_path': '/path/to/doc1.pdf'},
                {'id': 'doc2', 'score': 0.85, 'file_path': '/path/to/doc2.pdf'},
            ]
        })

        with patch('arcaneum.cli.search.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.search.SearchEmbedder', return_value=mock_embedder):
                with patch('arcaneum.cli.search.search_collection', return_value=sample_search_results):
                    with patch('arcaneum.cli.search.format_json_results', return_value=json_output):
                        with patch('arcaneum.cli.search.interaction_logger'):
                            with pytest.raises(SystemExit) as exc_info:
                                search_command(
                                    query='authentication',
                                    collection='TestCollection',
                                    vector_name=None,
                                    filter_arg=None,
                                    limit=10,
                                    offset=0,
                                    score_threshold=None,
                                    output_json=True,
                                    verbose=False
                                )

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output['status'] == 'success'
        assert len(output['results']) == 2

    def test_with_filter(self, mock_qdrant_client, mock_embedder, sample_search_results, capsys):
        """Test search with metadata filter."""
        from arcaneum.cli.search import search_command

        json_output = json.dumps({'status': 'success', 'results': []})

        with patch('arcaneum.cli.search.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.search.SearchEmbedder', return_value=mock_embedder):
                with patch('arcaneum.cli.search.search_collection', return_value=sample_search_results) as mock_search:
                    with patch('arcaneum.cli.search.parse_filter') as mock_parse:
                        mock_parse.return_value = MagicMock()
                        with patch('arcaneum.cli.search.build_filter_description', return_value='test filter'):
                            with patch('arcaneum.cli.search.format_json_results', return_value=json_output):
                                with patch('arcaneum.cli.search.interaction_logger'):
                                    with pytest.raises(SystemExit) as exc_info:
                                        search_command(
                                            query='authentication',
                                            collection='TestCollection',
                                            vector_name=None,
                                            filter_arg='page_number > 3',
                                            limit=10,
                                            offset=0,
                                            score_threshold=None,
                                            output_json=True,
                                            verbose=False
                                        )

        assert exc_info.value.code == 0
        mock_parse.assert_called_once_with('page_number > 3')

    def test_pagination(self, mock_qdrant_client, mock_embedder, sample_search_results, capsys):
        """Test search with pagination (limit and offset)."""
        from arcaneum.cli.search import search_command

        json_output = json.dumps({'status': 'success', 'results': []})

        with patch('arcaneum.cli.search.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.search.SearchEmbedder', return_value=mock_embedder):
                with patch('arcaneum.cli.search.search_collection', return_value=sample_search_results[:1]) as mock_search:
                    with patch('arcaneum.cli.search.format_json_results', return_value=json_output):
                        with patch('arcaneum.cli.search.interaction_logger'):
                            with pytest.raises(SystemExit) as exc_info:
                                search_command(
                                    query='authentication',
                                    collection='TestCollection',
                                    vector_name=None,
                                    filter_arg=None,
                                    limit=5,
                                    offset=10,
                                    score_threshold=None,
                                    output_json=True,
                                    verbose=False
                                )

        assert exc_info.value.code == 0
        # Verify pagination params were passed
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs['limit'] == 5
        assert call_kwargs['offset'] == 10

    def test_score_threshold(self, mock_qdrant_client, mock_embedder, sample_search_results, capsys):
        """Test search with score threshold."""
        from arcaneum.cli.search import search_command

        json_output = json.dumps({'status': 'success', 'results': []})

        with patch('arcaneum.cli.search.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.search.SearchEmbedder', return_value=mock_embedder):
                with patch('arcaneum.cli.search.search_collection', return_value=sample_search_results) as mock_search:
                    with patch('arcaneum.cli.search.format_json_results', return_value=json_output):
                        with patch('arcaneum.cli.search.interaction_logger'):
                            with pytest.raises(SystemExit) as exc_info:
                                search_command(
                                    query='authentication',
                                    collection='TestCollection',
                                    vector_name=None,
                                    filter_arg=None,
                                    limit=10,
                                    offset=0,
                                    score_threshold=0.8,
                                    output_json=True,
                                    verbose=False
                                )

        assert exc_info.value.code == 0
        # Verify score threshold was passed
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs['score_threshold'] == 0.8

    def test_json_output(self, mock_qdrant_client, mock_embedder, sample_search_results, capsys):
        """Test JSON output format."""
        from arcaneum.cli.search import search_command

        json_output = json.dumps({
            'status': 'success',
            'query': 'authentication',
            'collection': 'TestCollection',
            'results': []
        })

        with patch('arcaneum.cli.search.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.search.SearchEmbedder', return_value=mock_embedder):
                with patch('arcaneum.cli.search.search_collection', return_value=sample_search_results):
                    with patch('arcaneum.cli.search.format_json_results', return_value=json_output):
                        with patch('arcaneum.cli.search.interaction_logger'):
                            with pytest.raises(SystemExit) as exc_info:
                                search_command(
                                    query='authentication',
                                    collection='TestCollection',
                                    vector_name=None,
                                    filter_arg=None,
                                    limit=10,
                                    offset=0,
                                    score_threshold=None,
                                    output_json=True,
                                    verbose=False
                                )

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert 'status' in output
        assert 'results' in output
        assert 'query' in output
        assert 'collection' in output

    def test_verbose_output(self, mock_qdrant_client, mock_embedder, sample_search_results, capsys):
        """Test verbose output shows additional info."""
        from arcaneum.cli.search import search_command

        with patch('arcaneum.cli.search.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.search.SearchEmbedder', return_value=mock_embedder):
                with patch('arcaneum.cli.search.search_collection', return_value=sample_search_results):
                    with patch('arcaneum.cli.search.interaction_logger'):
                        with pytest.raises(SystemExit) as exc_info:
                            search_command(
                                query='authentication',
                                collection='TestCollection',
                                vector_name=None,
                                filter_arg=None,
                                limit=10,
                                offset=0,
                                score_threshold=None,
                                output_json=False,
                                verbose=True
                            )

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        # Verbose mode should show query and collection info
        assert 'TestCollection' in captured.out or 'authentication' in captured.out.lower()

    def test_collection_not_found(self, mock_qdrant_client, mock_embedder):
        """Test error when collection not found."""
        from arcaneum.cli.search import search_command
        from arcaneum.cli.errors import ResourceNotFoundError

        with patch('arcaneum.cli.search.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.search.SearchEmbedder', return_value=mock_embedder):
                with patch('arcaneum.cli.search.search_collection') as mock_search:
                    mock_search.side_effect = ResourceNotFoundError("Collection 'NonExistent' not found")
                    with patch('arcaneum.cli.search.interaction_logger'):
                        with pytest.raises(ResourceNotFoundError) as exc_info:
                            search_command(
                                query='test',
                                collection='NonExistent',
                                vector_name=None,
                                filter_arg=None,
                                limit=10,
                                offset=0,
                                score_threshold=None,
                                output_json=False,
                                verbose=False
                            )

        assert 'not found' in str(exc_info.value).lower()

    def test_interaction_logging(self, mock_qdrant_client, mock_embedder, sample_search_results):
        """Test that interaction logging is called."""
        from arcaneum.cli.search import search_command

        mock_logger = MagicMock()

        with patch('arcaneum.cli.search.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.search.SearchEmbedder', return_value=mock_embedder):
                with patch('arcaneum.cli.search.search_collection', return_value=sample_search_results):
                    with patch('arcaneum.cli.search.interaction_logger', mock_logger):
                        with pytest.raises(SystemExit):
                            search_command(
                                query='test query',
                                collection='TestCollection',
                                vector_name=None,
                                filter_arg=None,
                                limit=10,
                                offset=0,
                                score_threshold=None,
                                output_json=True,
                                verbose=False
                            )

        # Verify logging was called
        mock_logger.start.assert_called_once()
        call_kwargs = mock_logger.start.call_args[1]
        assert call_kwargs['collection'] == 'TestCollection'
        assert call_kwargs['query'] == 'test query'

        mock_logger.finish.assert_called_once()


class TestSearchNoResults:
    """Test search with no results."""

    def test_empty_results(self, mock_qdrant_client, capsys):
        """Test search returning no results."""
        from arcaneum.cli.search import search_command

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 1024

        json_output = json.dumps({'status': 'success', 'results': []})

        with patch('arcaneum.cli.search.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.search.SearchEmbedder', return_value=mock_embedder):
                with patch('arcaneum.cli.search.search_collection', return_value=[]):
                    with patch('arcaneum.cli.search.format_json_results', return_value=json_output):
                        with patch('arcaneum.cli.search.interaction_logger'):
                            with pytest.raises(SystemExit) as exc_info:
                                search_command(
                                    query='nonexistent query xyz123',
                                    collection='TestCollection',
                                    vector_name=None,
                                    filter_arg=None,
                                    limit=10,
                                    offset=0,
                                    score_threshold=None,
                                    output_json=True,
                                    verbose=False
                                )

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output['results'] == []


class TestSearchInvalidFilter:
    """Test search with invalid filter."""

    def test_invalid_filter_syntax(self, mock_qdrant_client):
        """Test error on invalid filter syntax."""
        from arcaneum.cli.search import search_command
        from arcaneum.cli.errors import InvalidArgumentError

        mock_embedder = MagicMock()

        with patch('arcaneum.cli.search.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.search.SearchEmbedder', return_value=mock_embedder):
                with patch('arcaneum.cli.search.parse_filter') as mock_parse:
                    mock_parse.side_effect = ValueError("Invalid filter syntax")
                    with patch('arcaneum.cli.search.interaction_logger'):
                        with pytest.raises(InvalidArgumentError) as exc_info:
                            search_command(
                                query='test',
                                collection='TestCollection',
                                vector_name=None,
                                filter_arg='invalid @@@ filter',
                                limit=10,
                                offset=0,
                                score_threshold=None,
                                output_json=False,
                                verbose=False
                            )

        assert 'Invalid filter' in str(exc_info.value)

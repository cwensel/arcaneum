"""CLI tests for 'arc search text' command (RDR-012).

Tests for full-text search CLI command functionality using mocked MeiliSearch client.
"""

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from arcaneum.cli.errors import ResourceNotFoundError
from arcaneum.cli.fulltext import search_text_command


class TestSearchTextBasicExecution:
    """Test basic search text command execution."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked FullTextClient."""
        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        return client

    @pytest.fixture
    def sample_search_results(self):
        """Sample search results from MeiliSearch."""
        return {
            'hits': [
                {
                    'id': 'doc1',
                    'content': 'def authenticate(username, password):',
                    'filename': 'auth.py',
                    'file_path': '/src/auth/verify.py',
                    'line_number': 42,
                    'language': 'python',
                    'project': 'myapp',
                    '_formatted': {
                        'content': 'def <em>authenticate</em>(username, password):'
                    }
                }
            ],
            'estimatedTotalHits': 1,
            'processingTimeMs': 12,
            'query': 'def authenticate',
            'limit': 10,
            'offset': 0,
        }

    def test_basic_search_returns_results(self, mock_client, sample_search_results, capsys):
        """Test that basic search returns and displays results."""
        mock_client.search.return_value = sample_search_results

        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                # The command calls sys.exit(0) on success, so we need to catch it
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query='def authenticate',
                        index_name='MyCode-fulltext',
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=False,
                        verbose=False
                    )

                assert exc_info.value.code == 0

        # Verify search was called correctly
        mock_client.search.assert_called_once_with(
            'MyCode-fulltext',
            'def authenticate',
            filter=None,
            limit=10,
            offset=0,
            attributes_to_highlight=['content']
        )

    def test_search_with_json_output(self, mock_client, sample_search_results, capsys):
        """Test JSON output mode returns valid JSON."""
        mock_client.search.return_value = sample_search_results

        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query='def authenticate',
                        index_name='MyCode-fulltext',
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False
                    )

                assert exc_info.value.code == 0

        # Capture and parse JSON output
        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output['status'] == 'success'
        assert 'data' in output
        assert output['data']['query'] == 'def authenticate'
        assert output['data']['index'] == 'MyCode-fulltext'
        assert 'hits' in output['data']
        assert 'estimatedTotalHits' in output['data']
        assert 'processingTimeMs' in output['data']


class TestExactPhraseSearchInCode:
    """Scenario 1: Exact phrase search in code (RDR-012)."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked FullTextClient."""
        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        return client

    def test_exact_phrase_search_returns_document_with_match(self, mock_client, capsys):
        """Test exact phrase search returns document with exact match."""
        # Setup: Simulate code indexed with function 'authenticate'
        mock_client.search.return_value = {
            'hits': [
                {
                    'id': 'func_authenticate_1',
                    'content': '''def authenticate(username, password):
    """Verify user credentials using bcrypt."""
    if not username or not password:
        raise ValueError("Missing credentials")
    return bcrypt.checkpw(password, get_hash(username))''',
                    'filename': 'verify.py',
                    'file_path': '/src/auth/verify.py',
                    'start_line': 42,
                    'end_line': 67,
                    'function_name': 'authenticate',
                    'language': 'python',
                    'project': 'arcaneum',
                    '_formatted': {
                        'content': '''<em>def authenticate</em>(username, password):
    """Verify user credentials using bcrypt."""
    if not username or not password:
        raise ValueError("Missing credentials")
    return bcrypt.checkpw(password, get_hash(username))'''
                    }
                }
            ],
            'estimatedTotalHits': 1,
            'processingTimeMs': 8,
            'query': 'def authenticate',
            'limit': 10,
            'offset': 0,
        }

        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query='"def authenticate"',
                        index_name='MyCode-fulltext',
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False
                    )

                assert exc_info.value.code == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        # Verify exact match is returned
        assert output['status'] == 'success'
        hits = output['data']['hits']
        assert len(hits) == 1

        hit = hits[0]
        assert hit['function_name'] == 'authenticate'
        assert 'def authenticate' in hit['content']
        # Highlighting shows matched phrase
        assert '<em>def authenticate</em>' in hit['_formatted']['content']


class TestKeywordSearchWithFilters:
    """Scenario 2: Keyword search with filters (RDR-012)."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked FullTextClient."""
        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        return client

    def test_filter_by_language_returns_only_python(self, mock_client, capsys):
        """Test filtering by language returns only Python files."""
        # Setup: Multi-language code indexed, filter for Python only
        mock_client.search.return_value = {
            'hits': [
                {
                    'id': 'py_calc_1',
                    'content': 'def calculate_total(items): return sum(item.price for item in items)',
                    'filename': 'utils.py',
                    'file_path': '/src/utils.py',
                    'line_number': 15,
                    'language': 'python',
                    'project': 'myapp',
                    '_formatted': {
                        'content': 'def <em>calculate</em>_total(items): return sum(item.price for item in items)'
                    }
                },
                {
                    'id': 'py_calc_2',
                    'content': 'def calculate_discount(price, rate): return price * (1 - rate)',
                    'filename': 'pricing.py',
                    'file_path': '/src/pricing.py',
                    'line_number': 28,
                    'language': 'python',
                    'project': 'myapp',
                    '_formatted': {
                        'content': 'def <em>calculate</em>_discount(price, rate): return price * (1 - rate)'
                    }
                }
            ],
            'estimatedTotalHits': 2,
            'processingTimeMs': 5,
            'query': 'calculate',
            'limit': 10,
            'offset': 0,
        }

        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query='calculate',
                        index_name='MyCode-fulltext',
                        filter_arg='language = python',
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False
                    )

                assert exc_info.value.code == 0

        # Verify filter was passed to search
        mock_client.search.assert_called_once_with(
            'MyCode-fulltext',
            'calculate',
            filter='language = python',
            limit=10,
            offset=0,
            attributes_to_highlight=['content']
        )

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        # Verify all results are Python
        hits = output['data']['hits']
        assert len(hits) == 2
        for hit in hits:
            assert hit['language'] == 'python'
            assert 'calculate' in hit['content']


class TestPDFSearchWithPageFilter:
    """Scenario 3: PDF search with page filter (RDR-012)."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked FullTextClient."""
        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        return client

    def test_filter_by_page_number_returns_pages_6_plus(self, mock_client, capsys):
        """Test filtering by page number > 5 returns only pages 6+."""
        # Setup: PDFs indexed, filter for pages > 5
        mock_client.search.return_value = {
            'hits': [
                {
                    'id': 'pdf_page_7',
                    'content': 'The neural network architecture consists of multiple hidden layers...',
                    'filename': 'research-paper.pdf',
                    'file_path': '/docs/research-paper.pdf',
                    'page_number': 7,
                    '_formatted': {
                        'content': 'The <em>neural network</em> architecture consists of multiple hidden layers...'
                    }
                },
                {
                    'id': 'pdf_page_12',
                    'content': 'Training the neural network requires careful hyperparameter tuning...',
                    'filename': 'research-paper.pdf',
                    'file_path': '/docs/research-paper.pdf',
                    'page_number': 12,
                    '_formatted': {
                        'content': 'Training the <em>neural network</em> requires careful hyperparameter tuning...'
                    }
                }
            ],
            'estimatedTotalHits': 2,
            'processingTimeMs': 15,
            'query': 'neural network',
            'limit': 10,
            'offset': 0,
        }

        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query='neural network',
                        index_name='PDFs',
                        filter_arg='page_number > 5',
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False
                    )

                assert exc_info.value.code == 0

        # Verify filter was passed correctly
        mock_client.search.assert_called_once_with(
            'PDFs',
            'neural network',
            filter='page_number > 5',
            limit=10,
            offset=0,
            attributes_to_highlight=['content']
        )

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        # Verify all results are from pages > 5
        hits = output['data']['hits']
        assert len(hits) == 2
        for hit in hits:
            assert hit['page_number'] > 5
            assert 'neural network' in hit['content'].lower()


class TestErrorHandlingMissingIndex:
    """Scenario 4: Error handling for missing index (RDR-012)."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked FullTextClient with server healthy but index missing."""
        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = False  # Index does not exist
        return client

    def test_missing_index_raises_resource_not_found_error(self, mock_client):
        """Test that searching a non-existent index raises ResourceNotFoundError."""
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(ResourceNotFoundError) as exc_info:
                    search_text_command(
                        query='query',
                        index_name='NonExistent',
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=False,
                        verbose=False
                    )

                # Verify clear error message
                assert "Index 'NonExistent' not found" in str(exc_info.value)

    def test_missing_index_with_json_raises_resource_not_found(self, mock_client):
        """Test that missing index in JSON mode also raises ResourceNotFoundError."""
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(ResourceNotFoundError) as exc_info:
                    search_text_command(
                        query='query',
                        index_name='NonExistent',
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False
                    )

                assert "Index 'NonExistent' not found" in str(exc_info.value)


class TestErrorHandlingServerUnavailable:
    """Test error handling when MeiliSearch server is unavailable."""

    @pytest.fixture
    def mock_client_unhealthy(self):
        """Create a mocked FullTextClient with server unhealthy."""
        client = MagicMock()
        client.health_check.return_value = False  # Server not available
        return client

    def test_server_unavailable_raises_resource_not_found_error(self, mock_client_unhealthy):
        """Test that unavailable server raises ResourceNotFoundError with helpful message."""
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client_unhealthy):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(ResourceNotFoundError) as exc_info:
                    search_text_command(
                        query='query',
                        index_name='MyIndex',
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=False,
                        verbose=False
                    )

                error_msg = str(exc_info.value)
                assert "MeiliSearch server not available" in error_msg
                # Should suggest how to start the server
                assert "docker compose" in error_msg


class TestJSONOutputForProgrammaticUse:
    """Scenario 5: JSON output for programmatic use (RDR-012)."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked FullTextClient."""
        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        return client

    def test_json_output_contains_required_fields(self, mock_client, capsys):
        """Test JSON output contains all required fields for programmatic use."""
        mock_client.search.return_value = {
            'hits': [
                {
                    'id': 'test_doc',
                    'content': 'test content',
                    'file_path': '/test/file.py',
                    '_formatted': {'content': '<em>test</em> content'}
                }
            ],
            'estimatedTotalHits': 1,
            'processingTimeMs': 5,
            'query': 'test',
            'limit': 10,
            'offset': 0,
        }

        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query='test',
                        index_name='MyCode-fulltext',
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False
                    )

                assert exc_info.value.code == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        # Verify JSON structure according to RDR-012
        assert 'status' in output
        assert 'data' in output

        data = output['data']
        assert 'query' in data
        assert data['query'] == 'test'
        assert 'index' in data
        assert data['index'] == 'MyCode-fulltext'
        assert 'hits' in data
        assert 'estimatedTotalHits' in data
        assert 'processingTimeMs' in data
        assert 'limit' in data
        assert 'offset' in data

        # Verify hit structure
        hits = data['hits']
        assert len(hits) == 1
        hit = hits[0]
        assert 'file_path' in hit
        assert 'content' in hit
        assert '_formatted' in hit

    def test_json_output_is_parseable(self, mock_client, capsys):
        """Test JSON output is valid and parseable JSON."""
        mock_client.search.return_value = {
            'hits': [
                {'id': 'doc1', 'content': 'content with "quotes" and special chars <>&'},
            ],
            'estimatedTotalHits': 1,
            'processingTimeMs': 3,
            'query': 'test',
            'limit': 10,
            'offset': 0,
        }

        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query='test',
                        index_name='TestIndex',
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False
                    )

                assert exc_info.value.code == 0

        captured = capsys.readouterr()
        # Should not raise JSONDecodeError
        output = json.loads(captured.out)
        assert output is not None


class TestPaginationOptions:
    """Test pagination with limit and offset options."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked FullTextClient."""
        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        return client

    def test_custom_limit_is_passed_to_search(self, mock_client, capsys):
        """Test that custom limit is passed to MeiliSearch search."""
        mock_client.search.return_value = {
            'hits': [],
            'estimatedTotalHits': 0,
            'processingTimeMs': 2,
        }

        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(SystemExit):
                    search_text_command(
                        query='test',
                        index_name='TestIndex',
                        filter_arg=None,
                        limit=50,
                        offset=0,
                        output_json=True,
                        verbose=False
                    )

        mock_client.search.assert_called_once_with(
            'TestIndex',
            'test',
            filter=None,
            limit=50,
            offset=0,
            attributes_to_highlight=['content']
        )

    def test_custom_offset_is_passed_to_search(self, mock_client, capsys):
        """Test that custom offset is passed for pagination."""
        mock_client.search.return_value = {
            'hits': [],
            'estimatedTotalHits': 100,
            'processingTimeMs': 3,
        }

        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(SystemExit):
                    search_text_command(
                        query='test',
                        index_name='TestIndex',
                        filter_arg=None,
                        limit=10,
                        offset=20,
                        output_json=True,
                        verbose=False
                    )

        mock_client.search.assert_called_once_with(
            'TestIndex',
            'test',
            filter=None,
            limit=10,
            offset=20,
            attributes_to_highlight=['content']
        )


class TestVerboseOutput:
    """Test verbose output mode."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked FullTextClient."""
        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        return client

    def test_verbose_mode_shows_additional_metadata(self, mock_client, capsys):
        """Test that verbose mode displays additional metadata."""
        mock_client.search.return_value = {
            'hits': [
                {
                    'id': 'doc1',
                    'content': 'def test_function():',
                    'filename': 'test.py',
                    'file_path': '/src/test.py',
                    'line_number': 10,
                    'language': 'python',
                    'project': 'myproject',
                    'page_number': None,
                    '_formatted': {'content': 'def <em>test</em>_function():'}
                }
            ],
            'estimatedTotalHits': 1,
            'processingTimeMs': 5,
        }

        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query='test',
                        index_name='TestIndex',
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=False,
                        verbose=True
                    )

                assert exc_info.value.code == 0

        captured = capsys.readouterr()
        # Verbose mode should show language and project metadata
        assert 'python' in captured.out.lower() or 'Language' in captured.out
        assert 'myproject' in captured.out.lower() or 'Project' in captured.out


class TestNoResultsHandling:
    """Test handling of search queries with no results."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked FullTextClient."""
        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        return client

    def test_no_results_returns_empty_hits(self, mock_client, capsys):
        """Test that search with no results returns empty hits array."""
        mock_client.search.return_value = {
            'hits': [],
            'estimatedTotalHits': 0,
            'processingTimeMs': 2,
        }

        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query='nonexistentquery12345',
                        index_name='TestIndex',
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False
                    )

                assert exc_info.value.code == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output['data']['hits'] == []
        assert output['data']['estimatedTotalHits'] == 0


class TestInteractionLogging:
    """Test that interaction logging is properly invoked (RDR-018)."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked FullTextClient."""
        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        client.search.return_value = {
            'hits': [{'id': 'test'}],
            'estimatedTotalHits': 1,
            'processingTimeMs': 5,
        }
        return client

    def test_interaction_logger_is_called_on_success(self, mock_client):
        """Test that interaction logger is called with correct parameters on success."""
        mock_logger = MagicMock()

        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger', mock_logger):
                with pytest.raises(SystemExit):
                    search_text_command(
                        query='test query',
                        index_name='TestIndex',
                        filter_arg='language = python',
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False
                    )

        # Verify start was called with correct params
        mock_logger.start.assert_called_once()
        call_args = mock_logger.start.call_args
        assert call_args[0][0] == 'search'  # command
        assert call_args[0][1] == 'text'    # subcommand
        assert call_args[1]['index'] == 'TestIndex'
        assert call_args[1]['query'] == 'test query'
        assert call_args[1]['filters'] == 'language = python'

        # Verify finish was called with result count
        mock_logger.finish.assert_called_once()
        finish_args = mock_logger.finish.call_args
        assert finish_args[1]['result_count'] == 1

    def test_interaction_logger_captures_search_errors(self, mock_client):
        """Test that interaction logger captures search errors."""
        mock_client.search.side_effect = Exception("Search failed")
        mock_logger = MagicMock()

        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger', mock_logger):
                with pytest.raises(SystemExit):
                    search_text_command(
                        query='test',
                        index_name='TestIndex',
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=False,
                        verbose=False
                    )

        # Verify error was logged
        mock_logger.finish.assert_called_once()
        finish_args = mock_logger.finish.call_args
        assert 'error' in finish_args[1]
        assert 'Search failed' in finish_args[1]['error']


class TestComplexFilterExpressions:
    """Test handling of complex MeiliSearch filter expressions."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked FullTextClient."""
        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        client.search.return_value = {
            'hits': [],
            'estimatedTotalHits': 0,
            'processingTimeMs': 5,
        }
        return client

    def test_and_filter_expression(self, mock_client):
        """Test filter with AND operator."""
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(SystemExit):
                    search_text_command(
                        query='test',
                        index_name='TestIndex',
                        filter_arg='language = python AND git_branch = main',
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False
                    )

        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args[1]['filter'] == 'language = python AND git_branch = main'

    def test_in_operator_filter(self, mock_client):
        """Test filter with IN operator for array membership."""
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(SystemExit):
                    search_text_command(
                        query='test',
                        index_name='TestIndex',
                        filter_arg='programming_language IN [python, javascript, typescript]',
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False
                    )

        call_args = mock_client.search.call_args
        assert 'IN [python, javascript, typescript]' in call_args[1]['filter']

    def test_numeric_range_filter(self, mock_client):
        """Test filter with numeric range."""
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(SystemExit):
                    search_text_command(
                        query='test',
                        index_name='TestIndex',
                        filter_arg='page_number > 5 AND page_number < 20',
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False
                    )

        call_args = mock_client.search.call_args
        assert 'page_number > 5 AND page_number < 20' in call_args[1]['filter']

    def test_contains_filter_for_file_path(self, mock_client):
        """Test filter with CONTAINS for file path matching."""
        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(SystemExit):
                    search_text_command(
                        query='test',
                        index_name='TestIndex',
                        filter_arg='file_path CONTAINS /src/auth/',
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False
                    )

        call_args = mock_client.search.call_args
        assert 'file_path CONTAINS /src/auth/' in call_args[1]['filter']


class TestHighlightingFormatting:
    """Test handling of MeiliSearch highlighting in results."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked FullTextClient."""
        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        return client

    def test_highlighting_is_requested(self, mock_client, capsys):
        """Test that highlighting is requested for content attribute."""
        mock_client.search.return_value = {
            'hits': [],
            'estimatedTotalHits': 0,
            'processingTimeMs': 2,
        }

        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(SystemExit):
                    search_text_command(
                        query='test',
                        index_name='TestIndex',
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False
                    )

        # Verify highlighting was requested
        call_args = mock_client.search.call_args
        assert call_args[1]['attributes_to_highlight'] == ['content']

    def test_formatted_content_preserved_in_json(self, mock_client, capsys):
        """Test that _formatted field is preserved in JSON output."""
        mock_client.search.return_value = {
            'hits': [
                {
                    'id': 'doc1',
                    'content': 'def authenticate():',
                    '_formatted': {
                        'content': 'def <em>authenticate</em>():'
                    }
                }
            ],
            'estimatedTotalHits': 1,
            'processingTimeMs': 5,
        }

        with patch('arcaneum.cli.fulltext.create_meili_client', return_value=mock_client):
            with patch('arcaneum.cli.fulltext.interaction_logger'):
                with pytest.raises(SystemExit):
                    search_text_command(
                        query='authenticate',
                        index_name='TestIndex',
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False
                    )

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        hit = output['data']['hits'][0]
        assert '_formatted' in hit
        assert '<em>authenticate</em>' in hit['_formatted']['content']

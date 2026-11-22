"""CLI tests for 'arc search text' command (RDR-012).

Tests for full-text search CLI command functionality using mocked MeiliSearch client.
"""

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from arcaneum.cli.errors import ArcaneumError, ResourceNotFoundError, ServiceDownError
from arcaneum.cli.fulltext import search_text_command


@pytest.fixture(autouse=True)
def no_qdrant(monkeypatch):
    """Text search reads cite roots from Qdrant; keep these tests off the network.

    A bare mock holds no ``source_roots``, so cites use the absolute path.
    """
    monkeypatch.setattr("arcaneum.cli.fulltext.create_qdrant_client", lambda **kwargs: MagicMock())


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
            "hits": [
                {
                    "id": "doc1",
                    "content": "def authenticate(username, password):",
                    "filename": "auth.py",
                    "file_path": "/src/auth/verify.py",
                    "line_number": 42,
                    "language": "python",
                    "project": "myapp",
                    "_formatted": {"content": "def <em>authenticate</em>(username, password):"},
                }
            ],
            "estimatedTotalHits": 1,
            "processingTimeMs": 12,
            "query": "def authenticate",
            "limit": 10,
            "offset": 0,
        }

    def test_basic_search_returns_results(self, mock_client, sample_search_results, capsys):
        """Test that basic search returns and displays results."""
        mock_client.search.return_value = sample_search_results

        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client):
            with patch("arcaneum.cli.fulltext.interaction_logger"):
                # The command calls sys.exit(0) on success, so we need to catch it
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query="def authenticate",
                        corpora=["MyCode-fulltext"],
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=False,
                        verbose=False,
                    )

                assert exc_info.value.code == 0

        # Verify search was called correctly
        mock_client.search.assert_called_once_with(
            "MyCode-fulltext",
            "def authenticate",
            filter=None,
            limit=10,
            offset=0,
            attributes_to_highlight=["content"],
            show_ranking_score=True,
        )

    def test_search_with_json_output(self, mock_client, sample_search_results, capsys):
        """Test JSON output mode returns valid JSON."""
        mock_client.search.return_value = sample_search_results

        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client):
            with patch("arcaneum.cli.fulltext.interaction_logger"):
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query="def authenticate",
                        corpora=["MyCode-fulltext"],
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False,
                    )

                assert exc_info.value.code == 0

        # Capture and parse JSON output
        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output["status"] == "success"
        assert output["message"] == "Found 1 results"
        assert "data" in output
        assert output["data"]["schema_version"] == 1
        assert output["data"]["query"] == "def authenticate"
        assert output["data"]["corpora"] == ["MyCode-fulltext"]
        assert output["data"]["missing_corpora"] == []
        assert "hits" in output["data"]


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
            "hits": [
                {
                    "id": "func_authenticate_1",
                    "content": '''def authenticate(username, password):
    """Verify user credentials using bcrypt."""
    if not username or not password:
        raise ValueError("Missing credentials")
    return bcrypt.checkpw(password, get_hash(username))''',
                    "filename": "verify.py",
                    "file_path": "/src/auth/verify.py",
                    "start_line": 42,
                    "end_line": 67,
                    "function_name": "authenticate",
                    "language": "python",
                    "project": "arcaneum",
                    "_formatted": {
                        "content": '''<em>def authenticate</em>(username, password):
    """Verify user credentials using bcrypt."""
    if not username or not password:
        raise ValueError("Missing credentials")
    return bcrypt.checkpw(password, get_hash(username))'''
                    },
                }
            ],
            "estimatedTotalHits": 1,
            "processingTimeMs": 8,
            "query": "def authenticate",
            "limit": 10,
            "offset": 0,
        }

        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client):
            with patch("arcaneum.cli.fulltext.interaction_logger"):
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query='"def authenticate"',
                        corpora=["MyCode-fulltext"],
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False,
                    )

                assert exc_info.value.code == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        # Verify exact match is returned
        assert output["status"] == "success"
        hits = output["data"]["hits"]
        assert len(hits) == 1

        hit = hits[0]
        assert hit["cite"] == "MyCode-fulltext:/src/auth/verify.py#L42-67"
        assert hit["chunk_id"] == "func_authenticate_1"
        assert "def authenticate" in hit["snippet"]
        # Highlight markup stays out of the hit (RDR-026 Q2)
        assert "_formatted" not in hit


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
            "hits": [
                {
                    "id": "py_calc_1",
                    "content": "def calculate_total(items): return sum(item.price for item in items)",
                    "filename": "utils.py",
                    "file_path": "/src/utils.py",
                    "line_number": 15,
                    "language": "python",
                    "project": "myapp",
                    "_formatted": {
                        "content": "def <em>calculate</em>_total(items): return sum(item.price for item in items)"
                    },
                },
                {
                    "id": "py_calc_2",
                    "content": "def calculate_discount(price, rate): return price * (1 - rate)",
                    "filename": "pricing.py",
                    "file_path": "/src/pricing.py",
                    "line_number": 28,
                    "language": "python",
                    "project": "myapp",
                    "_formatted": {
                        "content": "def <em>calculate</em>_discount(price, rate): return price * (1 - rate)"
                    },
                },
            ],
            "estimatedTotalHits": 2,
            "processingTimeMs": 5,
            "query": "calculate",
            "limit": 10,
            "offset": 0,
        }

        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client):
            with patch("arcaneum.cli.fulltext.interaction_logger"):
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query="calculate",
                        corpora=["MyCode-fulltext"],
                        filter_arg="language = python",
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=True,
                    )

                assert exc_info.value.code == 0

        # Verify the neutral filter reached MeiliSearch with its value quoted
        mock_client.search.assert_called_once_with(
            "MyCode-fulltext",
            "calculate",
            filter='language = "python"',
            limit=10,
            offset=0,
            attributes_to_highlight=["content"],
            show_ranking_score=True,
        )

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        # Verify all results are Python
        hits = output["data"]["hits"]
        assert len(hits) == 2
        for hit in hits:
            assert hit["metadata"]["language"] == "python"
            assert "calculate" in hit["snippet"]
        assert [hit["cite"] for hit in hits] == [
            "MyCode-fulltext:/src/utils.py",
            "MyCode-fulltext:/src/pricing.py",
        ]


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
            "hits": [
                {
                    "id": "pdf_page_7",
                    "content": "The neural network architecture consists of multiple hidden layers...",
                    "filename": "research-paper.pdf",
                    "file_path": "/docs/research-paper.pdf",
                    "page_number": 7,
                    "_formatted": {
                        "content": "The <em>neural network</em> architecture consists of multiple hidden layers..."
                    },
                },
                {
                    "id": "pdf_page_12",
                    "content": "Training the neural network requires careful hyperparameter tuning...",
                    "filename": "research-paper.pdf",
                    "file_path": "/docs/research-paper.pdf",
                    "page_number": 12,
                    "_formatted": {
                        "content": "Training the <em>neural network</em> requires careful hyperparameter tuning..."
                    },
                },
            ],
            "estimatedTotalHits": 2,
            "processingTimeMs": 15,
            "query": "neural network",
            "limit": 10,
            "offset": 0,
        }

        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client):
            with patch("arcaneum.cli.fulltext.interaction_logger"):
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query="neural network",
                        corpora=["PDFs"],
                        filter_arg="page_number > 5",
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=True,
                    )

                assert exc_info.value.code == 0

        # The neutral filter is re-rendered with quoted values;
        # MeiliSearch still compares page_number > "5" numerically.
        mock_client.search.assert_called_once_with(
            "PDFs",
            "neural network",
            filter='page_number > "5"',
            limit=10,
            offset=0,
            attributes_to_highlight=["content"],
            show_ranking_score=True,
        )

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        # Verify all results are from pages > 5
        hits = output["data"]["hits"]
        assert len(hits) == 2
        for hit in hits:
            assert hit["metadata"]["page_number"] > 5
            assert "neural network" in hit["snippet"].lower()
        assert [hit["cite"] for hit in hits] == [
            "PDFs:/docs/research-paper.pdf#page=7",
            "PDFs:/docs/research-paper.pdf#page=12",
        ]


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
        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client):
            with patch("arcaneum.cli.fulltext.interaction_logger"):
                with pytest.raises(ResourceNotFoundError) as exc_info:
                    search_text_command(
                        query="query",
                        corpora=["NonExistent"],
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=False,
                        verbose=False,
                    )

                # Verify clear error message
                assert "Corpus 'NonExistent' not found" in str(exc_info.value)


class TestErrorHandlingServerUnavailable:
    """Test error handling when MeiliSearch server is unavailable."""

    @pytest.fixture
    def mock_client_unhealthy(self):
        """Create a mocked FullTextClient with server unhealthy."""
        client = MagicMock()
        client.health_check.return_value = False  # Server not available
        return client

    def test_server_unavailable_raises_service_down_error(self, mock_client_unhealthy):
        """An unavailable server is service-down (exit 1) with a helpful message."""
        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client_unhealthy):
            with patch("arcaneum.cli.fulltext.interaction_logger"):
                with pytest.raises(ServiceDownError) as exc_info:
                    search_text_command(
                        query="query",
                        corpora=["MyIndex"],
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=False,
                        verbose=False,
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
            "hits": [
                {
                    "id": "test_doc",
                    "content": "test content",
                    "file_path": "/test/file.py",
                    "_formatted": {"content": "<em>test</em> content"},
                }
            ],
            "estimatedTotalHits": 1,
            "processingTimeMs": 5,
            "query": "test",
            "limit": 10,
            "offset": 0,
        }

        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client):
            with patch("arcaneum.cli.fulltext.interaction_logger"):
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query="test",
                        corpora=["MyCode-fulltext"],
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False,
                    )

                assert exc_info.value.code == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        # Verify JSON structure according to RDR-012
        assert "status" in output
        assert "data" in output

        data = output["data"]
        assert set(data) == {
            "schema_version",
            "query",
            "corpora",
            "missing_corpora",
            "limit",
            "offset",
            "hits",
        }
        assert data["query"] == "test"
        assert data["corpora"] == ["MyCode-fulltext"]
        assert (data["limit"], data["offset"]) == (10, 0)

        # Verify hit structure (RDR-026 Q2)
        hits = data["hits"]
        assert len(hits) == 1
        assert hits[0] == {
            "corpus": "MyCode-fulltext",
            "chunk_id": "test_doc",
            "locator": "/test/file.py",
            "cite": "MyCode-fulltext:/test/file.py",
            "score": None,
            "snippet": "test content",
            "snippet_truncated": False,
        }


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
            "hits": [],
            "estimatedTotalHits": 0,
            "processingTimeMs": 2,
        }

        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client):
            with patch("arcaneum.cli.fulltext.interaction_logger"):
                with pytest.raises(SystemExit):
                    search_text_command(
                        query="test",
                        corpora=["TestIndex"],
                        filter_arg=None,
                        limit=50,
                        offset=0,
                        output_json=True,
                        verbose=False,
                    )

        mock_client.search.assert_called_once_with(
            "TestIndex",
            "test",
            filter=None,
            limit=50,
            offset=0,
            attributes_to_highlight=["content"],
            show_ranking_score=True,
        )

    def test_custom_offset_is_passed_to_search(self, mock_client, capsys):
        """Test that custom offset is passed for pagination."""
        mock_client.search.return_value = {
            "hits": [],
            "estimatedTotalHits": 100,
            "processingTimeMs": 3,
        }

        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client):
            with patch("arcaneum.cli.fulltext.interaction_logger"):
                with pytest.raises(SystemExit):
                    search_text_command(
                        query="test",
                        corpora=["TestIndex"],
                        filter_arg=None,
                        limit=10,
                        offset=20,
                        output_json=True,
                        verbose=False,
                    )

        mock_client.search.assert_called_once_with(
            "TestIndex",
            "test",
            filter=None,
            limit=30,  # limit + offset combined for internal merging
            offset=0,
            attributes_to_highlight=["content"],
            show_ranking_score=True,
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
            "hits": [
                {
                    "id": "doc1",
                    "content": "def test_function():",
                    "filename": "test.py",
                    "file_path": "/src/test.py",
                    "line_number": 10,
                    "language": "python",
                    "project": "myproject",
                    "page_number": None,
                    "_formatted": {"content": "def <em>test</em>_function():"},
                }
            ],
            "estimatedTotalHits": 1,
            "processingTimeMs": 5,
        }

        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client):
            with patch("arcaneum.cli.fulltext.interaction_logger"):
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query="test",
                        corpora=["TestIndex"],
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=False,
                        verbose=True,
                    )

                assert exc_info.value.code == 0

        captured = capsys.readouterr()
        # Verbose mode should show language and project metadata
        assert "python" in captured.out.lower() or "Language" in captured.out
        assert "myproject" in captured.out.lower() or "Project" in captured.out


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
            "hits": [],
            "estimatedTotalHits": 0,
            "processingTimeMs": 2,
        }

        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client):
            with patch("arcaneum.cli.fulltext.interaction_logger"):
                with pytest.raises(SystemExit) as exc_info:
                    search_text_command(
                        query="nonexistentquery12345",
                        corpora=["TestIndex"],
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False,
                    )

                assert exc_info.value.code == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output["data"]["hits"] == []
        # Zero hits name the next step agents take (RDR-027 Q5).
        assert output["message"] == (
            "Found 0 results. For a conceptual match try: "
            "arc search semantic nonexistentquery12345 --corpus TestIndex"
        )


class TestInteractionLogging:
    """Test that interaction logging is properly invoked (RDR-018)."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked FullTextClient."""
        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        client.search.return_value = {
            "hits": [{"id": "test"}],
            "estimatedTotalHits": 1,
            "processingTimeMs": 5,
        }
        return client

    def test_interaction_logger_is_called_on_success(self, mock_client):
        """Test that interaction logger is called with correct parameters on success."""
        mock_logger = MagicMock()
        # track() is a contextmanager — its __enter__ yields the mutable result
        # dict that the command updates with result_count/extras.
        track_ctx = MagicMock()
        ctx_dict = {}
        track_ctx.__enter__.return_value = ctx_dict
        track_ctx.__exit__.return_value = False
        mock_logger.track.return_value = track_ctx

        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client):
            with patch("arcaneum.cli.fulltext.interaction_logger", mock_logger):
                with pytest.raises(SystemExit):
                    search_text_command(
                        query="test query",
                        corpora=["TestIndex"],
                        filter_arg="language = python",
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False,
                    )

        # Verify track() was called with correct params
        mock_logger.track.assert_called_once()
        call_args = mock_logger.track.call_args
        assert call_args[0][0] == "search"  # command
        assert call_args[0][1] == "text"  # subcommand
        assert call_args[1]["corpora"] == ["TestIndex"]
        assert call_args[1]["query"] == "test query"
        assert call_args[1]["filters"] == "language = python"

        # Verify result_count was recorded on the yielded context
        assert ctx_dict.get("result_count") == 1

    def test_interaction_logger_captures_search_errors(self, mock_client):
        """Test that interaction logger captures search errors."""
        mock_client.search.side_effect = Exception("Search failed")
        mock_logger = MagicMock()
        # Simulate track()'s real behavior: on exception, record the error on
        # the yielded dict and propagate (returning False from __exit__).
        ctx_dict = {}

        def _exit(exc_type, exc, tb):
            if exc is not None and ctx_dict.get("error") is None:
                ctx_dict["error"] = str(exc)
            return False

        track_ctx = MagicMock()
        track_ctx.__enter__.return_value = ctx_dict
        track_ctx.__exit__.side_effect = _exit
        mock_logger.track.return_value = track_ctx

        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client):
            with patch("arcaneum.cli.fulltext.interaction_logger", mock_logger):
                with pytest.raises(ArcaneumError):
                    search_text_command(
                        query="test",
                        corpora=["TestIndex"],
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=False,
                        verbose=False,
                    )

        # Verify track() was used and the error propagated through __exit__
        mock_logger.track.assert_called_once()
        assert "Search failed" in ctx_dict.get("error", "")


class TestComplexFilterExpressions:
    """Test handling of complex MeiliSearch filter expressions."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked FullTextClient."""
        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        client.search.return_value = {
            "hits": [],
            "estimatedTotalHits": 0,
            "processingTimeMs": 5,
        }
        return client

    def test_and_filter_expression(self, mock_client):
        """Test filter with AND operator."""
        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client):
            with patch("arcaneum.cli.fulltext.interaction_logger"):
                with pytest.raises(SystemExit):
                    search_text_command(
                        query="test",
                        corpora=["TestIndex"],
                        filter_arg="language = python AND git_branch = main",
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False,
                    )

        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args[1]["filter"] == 'language = "python" AND git_branch = "main"'

    def test_native_or_filter_passes_through_verbatim(self, mock_client):
        """Test a native-only filter (OR) reaches MeiliSearch unchanged."""
        filter_arg = "language = python OR language = rust"
        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client):
            with patch("arcaneum.cli.fulltext.interaction_logger"):
                with pytest.raises(SystemExit):
                    search_text_command(
                        query="test",
                        corpora=["TestIndex"],
                        filter_arg=filter_arg,
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=False,
                    )

        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args[1]["filter"] == filter_arg


class TestHighlightingFormatting:
    """Test handling of MeiliSearch highlighting in results."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked FullTextClient."""
        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        return client

    @pytest.mark.parametrize("verbose", [False, True])
    def test_formatted_content_stays_out_of_json(self, mock_client, capsys, verbose):
        """MeiliSearch's _formatted highlight never reaches a hit (RDR-026 Q2)."""
        mock_client.search.return_value = {
            "hits": [
                {
                    "id": "doc1",
                    "content": "def authenticate():",
                    "_formatted": {"content": "def <em>authenticate</em>():"},
                }
            ],
            "estimatedTotalHits": 1,
            "processingTimeMs": 5,
        }

        with patch("arcaneum.cli.fulltext.create_meili_client", return_value=mock_client):
            with patch("arcaneum.cli.fulltext.interaction_logger"):
                with pytest.raises(SystemExit):
                    search_text_command(
                        query="authenticate",
                        corpora=["TestIndex"],
                        filter_arg=None,
                        limit=10,
                        offset=0,
                        output_json=True,
                        verbose=verbose,
                    )

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        hit = output["data"]["hits"][0]
        assert hit["snippet"] == "def authenticate():"
        assert "_formatted" not in hit
        assert "_formatted" not in hit.get("metadata", {})
        assert "<em>" not in json.dumps(output)


class TestJSONModeFailureEnvelope:
    """RDR-026 Q3: under --json every failure prints a coded envelope on stdout."""

    @pytest.fixture
    def run_main(self, monkeypatch, capsys):
        from arcaneum.cli import fulltext
        from arcaneum.cli.main import main

        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        client.get_index_settings.return_value = {"filterableAttributes": ["file_path"]}
        monkeypatch.setattr("arcaneum.cli.main.configure_ssl_from_env", lambda: None)
        monkeypatch.setattr(fulltext, "create_meili_client", lambda: client)
        monkeypatch.setattr(fulltext, "interaction_logger", MagicMock())

        def run(*args):
            monkeypatch.setattr("sys.argv", ["arc", "search", "text", "q", "--corpus", "DevRef", *args])
            exit_code = main()
            return exit_code, capsys.readouterr()

        return client, run

    @staticmethod
    def _filter_error():
        from types import SimpleNamespace

        from meilisearch.errors import MeilisearchApiError

        body = {"message": "Attribute `filename` is not filterable.", "code": "invalid_search_filter"}
        return MeilisearchApiError("bad", SimpleNamespace(status_code=400, text=json.dumps(body)))

    def test_bad_filter_json_prints_coded_envelope_and_exits_2(self, run_main):
        client, run = run_main
        client.search.side_effect = self._filter_error()

        exit_code, captured = run("--filter", '{"filename": "db(31).pdf"}', "--json")

        assert exit_code == 2
        payload = json.loads(captured.out)
        assert payload["status"] == "error"
        assert payload["errors"][0]["code"] == "bad-filter"
        assert "filename" in payload["errors"][0]["message"]
        assert "file_path" in payload["errors"][0]["message"]

    def test_bad_filter_text_mode_prints_to_stderr(self, run_main):
        client, run = run_main
        client.search.side_effect = self._filter_error()

        exit_code, captured = run("--filter", "filename = x")

        assert exit_code == 2
        assert captured.out == ""
        assert "[ERROR]" in captured.err
        assert "filename" in captured.err

    def test_unknown_corpus_json_exits_3_with_code(self, run_main):
        client, run = run_main
        client.index_exists.return_value = False

        exit_code, captured = run("--json")

        assert exit_code == 3
        assert json.loads(captured.out)["errors"] == [
            {"code": "unknown-corpus", "message": "Corpus 'DevRef' not found"}
        ]

    def test_service_down_json_exits_1_with_code(self, run_main):
        client, run = run_main
        client.health_check.return_value = False

        exit_code, captured = run("--json")

        assert exit_code == 1
        assert json.loads(captured.out)["errors"][0]["code"] == "service-down"


class TestQdrantMetadataUnreadable:
    """Qdrant down: MeiliSearch still answers, and the output says so (arcaneum#178)."""

    @pytest.fixture
    def run_main(self, monkeypatch, capsys):
        from arcaneum.cli import fulltext
        from arcaneum.cli.main import main

        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        client.search.return_value = {
            "hits": [{"id": "d1", "content": "lorem", "file_path": "/src/a.py"}],
            "estimatedTotalHits": 1,
            "processingTimeMs": 1,
        }
        qdrant = MagicMock()
        qdrant.retrieve.side_effect = ConnectionError("refused")
        monkeypatch.setattr("arcaneum.cli.main.configure_ssl_from_env", lambda: None)
        monkeypatch.setattr(fulltext, "create_meili_client", lambda: client)
        monkeypatch.setattr(fulltext, "create_qdrant_client", lambda **kwargs: qdrant)
        monkeypatch.setattr(fulltext, "interaction_logger", MagicMock())

        def run(*args):
            monkeypatch.setattr("sys.argv", ["arc", "search", "text", "q", "--corpus", "DevRef", *args])
            try:
                exit_code = main()
            except SystemExit as e:  # the command exits 0 itself on success
                exit_code = e.code
            return exit_code, capsys.readouterr()

        return client, run

    def test_text_mode_warns_on_stderr_and_prints_hits(self, run_main):
        _client, run = run_main

        exit_code, captured = run()

        assert exit_code == 0
        assert "DevRef:/src/a.py" in captured.out
        assert "[WARNING]" in captured.err
        assert "Qdrant" in captured.err
        assert "stale" in captured.err

    def test_json_carries_service_errors(self, run_main):
        _client, run = run_main

        exit_code, captured = run("--json")

        assert exit_code == 0
        data = json.loads(captured.out)["data"]
        assert data["service_errors"] == ["Qdrant: refused"]
        assert [hit["cite"] for hit in data["hits"]] == ["DevRef:/src/a.py"]

    def test_corpus_absent_from_meilisearch_is_service_down(self, run_main):
        client, run = run_main
        client.index_exists.return_value = False

        exit_code, captured = run("--json")

        assert exit_code == 1
        error = json.loads(captured.out)["errors"][0]
        assert error["code"] == "service-down"
        assert "Qdrant" in error["message"]


def test_source_roots_client_gets_a_short_timeout(monkeypatch):
    """Roots are optional for text search, so a slow Qdrant must not hold it up."""
    from arcaneum.cli import fulltext

    calls = []
    monkeypatch.setattr(
        fulltext, "create_qdrant_client", lambda **kwargs: calls.append(kwargs) or MagicMock()
    )

    fulltext._source_roots_client()

    assert calls[0]["timeout"] == fulltext.SOURCE_ROOTS_TIMEOUT_SECONDS
    assert fulltext.SOURCE_ROOTS_TIMEOUT_SECONDS <= 5

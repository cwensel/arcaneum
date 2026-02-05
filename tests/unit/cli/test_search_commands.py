"""CLI tests for search commands.

Tests for 'arc search semantic' command.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from arcaneum.search.searcher import SearchResult


def make_search_result(
    score: float,
    collection: str,
    file_path: str,
    content: str,
    point_id: str = "test-id",
    **extra_metadata
) -> SearchResult:
    """Helper to create SearchResult objects for testing."""
    metadata = {
        'file_path': file_path,
        'content': content,
        **extra_metadata
    }
    return SearchResult(
        score=score,
        collection=collection,
        location=file_path,
        content=content,
        metadata=metadata,
        point_id=point_id
    )


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
        """Sample search results as SearchResult dataclass objects."""
        return [
            make_search_result(
                score=0.95,
                collection='TestCollection',
                file_path='/path/to/doc1.pdf',
                content='This is the first result about authentication.',
                point_id='doc1',
                page_number=5,
            ),
            make_search_result(
                score=0.85,
                collection='TestCollection',
                file_path='/path/to/doc2.pdf',
                content='This is the second result about security.',
                point_id='doc2',
                page_number=10,
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
                                    corpora=['TestCollection'],
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
                                            corpora=['TestCollection'],
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
                                    corpora=['TestCollection'],
                                    vector_name=None,
                                    filter_arg=None,
                                    limit=5,
                                    offset=10,
                                    score_threshold=None,
                                    output_json=True,
                                    verbose=False
                                )

        assert exc_info.value.code == 0
        # For multi-corpus support, search_collection is called with limit+offset
        # and offset=0, then pagination is applied after merging results
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs['limit'] == 15  # limit + offset for merge buffer
        assert call_kwargs['offset'] == 0  # offset applied after merge

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
                                    corpora=['TestCollection'],
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
                                    corpora=['TestCollection'],
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
                                corpora=['TestCollection'],
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
                                corpora=['NonExistent'],
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
                                corpora=['TestCollection'],
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
        assert call_kwargs['corpora'] == ['TestCollection']
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
                                    corpora=['TestCollection'],
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
                                corpora=['TestCollection'],
                                vector_name=None,
                                filter_arg='invalid @@@ filter',
                                limit=10,
                                offset=0,
                                score_threshold=None,
                                output_json=False,
                                verbose=False
                            )

        assert 'Invalid filter' in str(exc_info.value)


class TestMultiCorpusSearch:
    """Test multi-corpus search functionality.

    These tests ensure SearchResult dataclass objects are handled correctly
    when merging results from multiple corpora.
    """

    @pytest.fixture
    def mock_embedder(self):
        """Create a mocked SearchEmbedder."""
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 1024
        return embedder

    @pytest.fixture
    def corpus1_results(self):
        """Results from first corpus."""
        return [
            make_search_result(
                score=0.95,
                collection='Corpus1',
                file_path='/path/to/corpus1/doc1.pdf',
                content='High relevance result from corpus 1.',
                point_id='c1-doc1',
            ),
            make_search_result(
                score=0.75,
                collection='Corpus1',
                file_path='/path/to/corpus1/doc2.pdf',
                content='Medium relevance result from corpus 1.',
                point_id='c1-doc2',
            ),
        ]

    @pytest.fixture
    def corpus2_results(self):
        """Results from second corpus."""
        return [
            make_search_result(
                score=0.85,
                collection='Corpus2',
                file_path='/path/to/corpus2/doc1.pdf',
                content='High relevance result from corpus 2.',
                point_id='c2-doc1',
            ),
            make_search_result(
                score=0.65,
                collection='Corpus2',
                file_path='/path/to/corpus2/doc2.pdf',
                content='Lower relevance result from corpus 2.',
                point_id='c2-doc2',
            ),
        ]

    def test_multi_corpus_search_merges_and_sorts_results(
        self, mock_qdrant_client, mock_embedder, corpus1_results, corpus2_results, capsys
    ):
        """Test that multi-corpus search merges results and sorts by score.

        This test would have caught the bug where SearchResult dataclass
        objects were incorrectly treated as dictionaries (using [] assignment
        and .get() method).
        """
        from arcaneum.cli.search import search_command

        def search_side_effect(client, embedder, query, collection_name, **kwargs):
            if collection_name == 'Corpus1':
                return corpus1_results
            elif collection_name == 'Corpus2':
                return corpus2_results
            raise Exception(f"Unknown collection: {collection_name}")

        with patch('arcaneum.cli.search.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.search.SearchEmbedder', return_value=mock_embedder):
                with patch('arcaneum.cli.search.search_collection', side_effect=search_side_effect):
                    with patch('arcaneum.cli.search.interaction_logger'):
                        with pytest.raises(SystemExit) as exc_info:
                            search_command(
                                query='test query',
                                corpora=['Corpus1', 'Corpus2'],
                                vector_name=None,
                                filter_arg=None,
                                limit=10,
                                offset=0,
                                score_threshold=None,
                                output_json=False,
                                verbose=False
                            )

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        # Verify results are shown (basic smoke test)
        assert 'Found 4 results' in captured.out

    def test_multi_corpus_search_json_output(
        self, mock_qdrant_client, mock_embedder, corpus1_results, corpus2_results, capsys
    ):
        """Test multi-corpus search with JSON output format."""
        from arcaneum.cli.search import search_command

        def search_side_effect(client, embedder, query, collection_name, **kwargs):
            if collection_name == 'Corpus1':
                return corpus1_results
            elif collection_name == 'Corpus2':
                return corpus2_results
            raise Exception(f"Unknown collection: {collection_name}")

        with patch('arcaneum.cli.search.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.search.SearchEmbedder', return_value=mock_embedder):
                with patch('arcaneum.cli.search.search_collection', side_effect=search_side_effect):
                    with patch('arcaneum.cli.search.interaction_logger'):
                        with pytest.raises(SystemExit) as exc_info:
                            search_command(
                                query='test query',
                                corpora=['Corpus1', 'Corpus2'],
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

        # Verify results are merged
        assert output['total_results'] == 4

        # Verify results are sorted by score (descending)
        scores = [r['score'] for r in output['results']]
        assert scores == sorted(scores, reverse=True)

    def test_multi_corpus_search_with_pagination(
        self, mock_qdrant_client, mock_embedder, corpus1_results, corpus2_results, capsys
    ):
        """Test multi-corpus search respects limit and offset."""
        from arcaneum.cli.search import search_command

        def search_side_effect(client, embedder, query, collection_name, **kwargs):
            if collection_name == 'Corpus1':
                return corpus1_results
            elif collection_name == 'Corpus2':
                return corpus2_results
            raise Exception(f"Unknown collection: {collection_name}")

        with patch('arcaneum.cli.search.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.search.SearchEmbedder', return_value=mock_embedder):
                with patch('arcaneum.cli.search.search_collection', side_effect=search_side_effect):
                    with patch('arcaneum.cli.search.interaction_logger'):
                        with pytest.raises(SystemExit) as exc_info:
                            search_command(
                                query='test query',
                                corpora=['Corpus1', 'Corpus2'],
                                vector_name=None,
                                filter_arg=None,
                                limit=2,
                                offset=1,
                                score_threshold=None,
                                output_json=True,
                                verbose=False
                            )

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)

        # Should return 2 results (limit) after skipping 1 (offset)
        assert output['total_results'] == 2

    def test_multi_corpus_partial_failure_continues(
        self, mock_qdrant_client, mock_embedder, corpus1_results, capsys
    ):
        """Test that search continues if one corpus is not found."""
        from arcaneum.cli.search import search_command

        def search_side_effect(client, embedder, query, collection_name, **kwargs):
            if collection_name == 'Corpus1':
                return corpus1_results
            elif collection_name == 'MissingCorpus':
                raise Exception("Collection 'MissingCorpus' doesn't exist")
            raise Exception(f"Unknown collection: {collection_name}")

        with patch('arcaneum.cli.search.create_qdrant_client', return_value=mock_qdrant_client):
            with patch('arcaneum.cli.search.SearchEmbedder', return_value=mock_embedder):
                with patch('arcaneum.cli.search.search_collection', side_effect=search_side_effect):
                    with patch('arcaneum.cli.search.interaction_logger'):
                        with pytest.raises(SystemExit) as exc_info:
                            search_command(
                                query='test query',
                                corpora=['Corpus1', 'MissingCorpus'],
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

        # Should still return results from Corpus1
        assert output['total_results'] == 2

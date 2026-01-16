"""Integration tests for full-text search workflow (RDR-012).

These tests verify the end-to-end search workflow including:
- Complete search workflow with real MeiliSearch indexes
- Cooperative workflow (semantic -> exact)
- Filter validation
- Result format validation

Requirements:
- Running MeiliSearch server (default: http://localhost:7700)
- Running Qdrant server (for cooperative workflow tests)
- Git installed (for git project fixtures)

Run with: pytest tests/integration/test_fulltext_search_workflow.py -v
"""

import json
import os
import pytest
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch

from arcaneum.fulltext.client import FullTextClient
from arcaneum.fulltext.indexes import SOURCE_CODE_FULLTEXT_SETTINGS
from arcaneum.indexing.fulltext.code_indexer import SourceCodeFullTextIndexer
from arcaneum.paths import get_meilisearch_api_key


# Environment configuration
MEILISEARCH_URL = os.environ.get("MEILISEARCH_URL", "http://localhost:7700")
MEILISEARCH_API_KEY = os.environ.get("MEILISEARCH_API_KEY") or get_meilisearch_api_key()
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def meili_client():
    """Provide MeiliSearch client connected to test server."""
    try:
        client = FullTextClient(url=MEILISEARCH_URL, api_key=MEILISEARCH_API_KEY)

        # Skip tests if server not available
        if not client.health_check():
            pytest.skip("MeiliSearch server not available")

        # Test that we can list indexes (verifies API key)
        client.list_indexes()

    except Exception as e:
        pytest.skip(f"MeiliSearch server not available: {e}")

    yield client

    # Cleanup: delete test indexes
    try:
        indexes = client.list_indexes()
        for idx in indexes:
            if idx['uid'].startswith("test_workflow_"):
                client.delete_index(idx['uid'])
    except Exception:
        pass


@pytest.fixture
def qdrant_client():
    """Provide Qdrant client connected to test server."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL)

        # Test connection
        client.get_collections()

    except Exception as e:
        pytest.skip(f"Qdrant server not available: {e}")

    yield client

    # Cleanup: delete test collections
    try:
        collections = client.get_collections()
        for col in collections.collections:
            if col.name.startswith("test_workflow_"):
                client.delete_collection(col.name)
    except Exception:
        pass


@pytest.fixture
def test_index(meili_client):
    """Create a test index with source code fulltext settings."""
    index_name = "test_workflow_fulltext"

    # Delete if exists
    if meili_client.index_exists(index_name):
        meili_client.delete_index(index_name)

    # Create with source code fulltext settings
    meili_client.create_index(
        name=index_name,
        primary_key='id',
        settings=SOURCE_CODE_FULLTEXT_SETTINGS
    )

    yield index_name

    # Cleanup
    try:
        meili_client.delete_index(index_name)
    except Exception:
        pass


@pytest.fixture
def git_project_with_auth():
    """Create a temporary git repository with authentication-related code.

    This fixture creates a realistic code structure with auth/ directory
    containing authentication code, suitable for testing the cooperative
    workflow (semantic -> exact search).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir) / "auth-project"
        project_path.mkdir()

        # Initialize git repository
        subprocess.run(["git", "init"], cwd=project_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=project_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=project_path, check=True, capture_output=True
        )

        # Create auth directory with authentication code
        auth_dir = project_path / "src" / "auth"
        auth_dir.mkdir(parents=True)

        # Create authentication module (Python)
        (auth_dir / "verify.py").write_text('''"""User authentication verification module."""

import hashlib
import secrets
from typing import Optional

from .models import User
from .exceptions import AuthenticationError


def authenticate(username: str, password: str) -> User:
    """Authenticate a user with username and password.

    Args:
        username: The user's username
        password: The user's plain text password

    Returns:
        User object if authentication succeeds

    Raises:
        AuthenticationError: If credentials are invalid
    """
    user = find_user_by_username(username)
    if user is None:
        raise AuthenticationError("User not found")

    if not verify_password(password, user.password_hash):
        raise AuthenticationError("Invalid password")

    return user


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    computed_hash = hashlib.sha256(password.encode()).hexdigest()
    return secrets.compare_digest(computed_hash, password_hash)


def find_user_by_username(username: str) -> Optional[User]:
    """Find a user by their username."""
    # Implementation would query database
    pass


class Authenticator:
    """Main authenticator class for managing user sessions."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self._sessions = {}

    def login(self, username: str, password: str) -> str:
        """Log in a user and return a session token."""
        user = authenticate(username, password)
        token = secrets.token_urlsafe(32)
        self._sessions[token] = user
        return token

    def logout(self, token: str) -> None:
        """Log out a user by invalidating their session token."""
        self._sessions.pop(token, None)

    def get_current_user(self, token: str) -> Optional[User]:
        """Get the currently logged in user from a session token."""
        return self._sessions.get(token)
''')

        # Create models module
        (auth_dir / "models.py").write_text('''"""Authentication data models."""

from dataclasses import dataclass


@dataclass
class User:
    """User data model."""
    id: int
    username: str
    email: str
    password_hash: str
    is_active: bool = True
''')

        # Create exceptions module
        (auth_dir / "exceptions.py").write_text('''"""Authentication exceptions."""


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class AuthorizationError(Exception):
    """Raised when user is not authorized."""
    pass
''')

        # Create __init__.py
        (auth_dir / "__init__.py").write_text('''"""Authentication package."""

from .verify import authenticate, Authenticator
from .models import User
from .exceptions import AuthenticationError, AuthorizationError

__all__ = [
    "authenticate",
    "Authenticator",
    "User",
    "AuthenticationError",
    "AuthorizationError",
]
''')

        # Create a utils module outside auth directory
        utils_dir = project_path / "src" / "utils"
        utils_dir.mkdir(parents=True)

        (utils_dir / "helpers.py").write_text('''"""General utility helpers."""


def calculate_total(items):
    """Calculate the total price of items."""
    return sum(item.price for item in items)


def format_currency(amount: float) -> str:
    """Format amount as currency string."""
    return f"${amount:.2f}"
''')

        # Git add and commit
        subprocess.run(["git", "add", "."], cwd=project_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit with auth module"],
            cwd=project_path, check=True, capture_output=True
        )

        yield str(project_path)


@pytest.fixture
def indexed_auth_project(meili_client, test_index, git_project_with_auth):
    """Index the authentication project into MeiliSearch."""
    indexer = SourceCodeFullTextIndexer(
        meili_client=meili_client,
        index_name=test_index,
        batch_size=100
    )

    stats = indexer.index_single_project(
        project_root=git_project_with_auth,
        force=True,
        verbose=False
    )

    # Verify indexing succeeded
    assert stats['indexed_projects'] == 1
    assert stats['indexed_files'] >= 4  # verify.py, models.py, exceptions.py, __init__.py, helpers.py

    yield {
        'index_name': test_index,
        'project_path': git_project_with_auth,
        'stats': stats,
    }


# -----------------------------------------------------------------------------
# Complete Search Workflow Tests
# -----------------------------------------------------------------------------

class TestFullTextSearchWorkflow:
    """Tests for the complete full-text search workflow."""

    def test_basic_text_search(self, meili_client, indexed_auth_project):
        """Test basic full-text search returns results."""
        index_name = indexed_auth_project['index_name']

        results = meili_client.search(
            index_name,
            "authenticate",
            limit=10
        )

        assert 'hits' in results
        assert 'processingTimeMs' in results
        assert 'estimatedTotalHits' in results
        assert results['estimatedTotalHits'] >= 1

    def test_exact_phrase_search(self, meili_client, indexed_auth_project):
        """Test exact phrase search with quotes."""
        index_name = indexed_auth_project['index_name']

        # Search for exact phrase "def authenticate"
        results = meili_client.search(
            index_name,
            '"def authenticate"',
            limit=10
        )

        assert results['estimatedTotalHits'] >= 1

        # The result should contain the exact phrase
        found_exact = False
        for hit in results['hits']:
            content = hit.get('content', '')
            if 'def authenticate' in content:
                found_exact = True
                break

        assert found_exact, "Expected to find 'def authenticate' in results"

    def test_search_with_highlighting(self, meili_client, indexed_auth_project):
        """Test search returns highlighted content."""
        index_name = indexed_auth_project['index_name']

        results = meili_client.search(
            index_name,
            "password",
            limit=10,
            attributes_to_highlight=['content']
        )

        assert results['estimatedTotalHits'] >= 1

        # Check that at least one result has highlighting
        has_highlighting = False
        for hit in results['hits']:
            if '_formatted' in hit and 'content' in hit['_formatted']:
                formatted_content = hit['_formatted']['content']
                # MeiliSearch wraps matches in <em> tags
                if '<em>' in formatted_content and '</em>' in formatted_content:
                    has_highlighting = True
                    break

        assert has_highlighting, "Expected highlighted content with <em> tags"

    def test_search_with_pagination(self, meili_client, indexed_auth_project):
        """Test search pagination with limit and offset."""
        index_name = indexed_auth_project['index_name']

        # Get all results
        all_results = meili_client.search(
            index_name,
            "",  # Empty query returns all
            limit=100
        )

        total = all_results['estimatedTotalHits']

        if total >= 2:
            # Get first result
            first_results = meili_client.search(
                index_name,
                "",
                limit=1,
                offset=0
            )

            # Get second result using offset
            second_results = meili_client.search(
                index_name,
                "",
                limit=1,
                offset=1
            )

            # Results should be different
            if first_results['hits'] and second_results['hits']:
                first_id = first_results['hits'][0].get('id')
                second_id = second_results['hits'][0].get('id')
                assert first_id != second_id, "Pagination should return different results"


# -----------------------------------------------------------------------------
# Filter Validation Tests (RDR-012)
# -----------------------------------------------------------------------------

class TestFilterValidation:
    """Tests for MeiliSearch filter expressions."""

    def test_filter_by_language(self, meili_client, indexed_auth_project):
        """Test filtering by programming language."""
        index_name = indexed_auth_project['index_name']

        results = meili_client.search(
            index_name,
            "",  # All documents
            filter='programming_language = "python"',
            limit=100
        )

        assert results['estimatedTotalHits'] >= 1

        # All results should be Python
        for hit in results['hits']:
            assert hit.get('programming_language') == 'python'

    def test_filter_by_code_type_function(self, meili_client, indexed_auth_project):
        """Test filtering for function definitions."""
        index_name = indexed_auth_project['index_name']

        results = meili_client.search(
            index_name,
            "",
            filter='code_type = "function"',
            limit=100
        )

        assert results['estimatedTotalHits'] >= 1

        # All results should be functions
        for hit in results['hits']:
            assert hit.get('code_type') == 'function'

    def test_filter_by_code_type_class(self, meili_client, indexed_auth_project):
        """Test filtering for class definitions."""
        index_name = indexed_auth_project['index_name']

        results = meili_client.search(
            index_name,
            "",
            filter='code_type = "class"',
            limit=100
        )

        # Should find Authenticator, User, AuthenticationError, AuthorizationError
        assert results['estimatedTotalHits'] >= 1

        for hit in results['hits']:
            assert hit.get('code_type') == 'class'

    def test_filter_file_path_contains(self, meili_client, indexed_auth_project):
        """Test filtering by file path using CONTAINS operator.

        Note: CONTAINS is an experimental feature in MeiliSearch and may not
        be enabled on all servers. This test will be skipped if the feature
        is not available.
        """
        index_name = indexed_auth_project['index_name']

        try:
            results = meili_client.search(
                index_name,
                "",
                filter='file_path CONTAINS "/auth/"',
                limit=100
            )
        except Exception as e:
            if 'feature_not_enabled' in str(e) or 'CONTAINS' in str(e):
                pytest.skip("CONTAINS filter requires experimental feature to be enabled")
            raise

        assert results['estimatedTotalHits'] >= 1

        # All results should have /auth/ in their path
        for hit in results['hits']:
            file_path = hit.get('file_path', '')
            assert '/auth/' in file_path, f"Expected '/auth/' in path: {file_path}"

    def test_filter_combined_conditions(self, meili_client, indexed_auth_project):
        """Test filtering with multiple conditions using AND."""
        index_name = indexed_auth_project['index_name']

        results = meili_client.search(
            index_name,
            "authenticate",
            filter='programming_language = "python" AND code_type = "function"',
            limit=100
        )

        # Should find the authenticate function
        assert results['estimatedTotalHits'] >= 1

        for hit in results['hits']:
            assert hit.get('programming_language') == 'python'
            assert hit.get('code_type') == 'function'

    def test_filter_or_conditions(self, meili_client, indexed_auth_project):
        """Test filtering with OR conditions."""
        index_name = indexed_auth_project['index_name']

        results = meili_client.search(
            index_name,
            "",
            filter='code_type = "function" OR code_type = "class"',
            limit=100
        )

        assert results['estimatedTotalHits'] >= 1

        # All results should be either function or class
        for hit in results['hits']:
            code_type = hit.get('code_type')
            assert code_type in ('function', 'class'), f"Unexpected code_type: {code_type}"


# -----------------------------------------------------------------------------
# Result Format Validation Tests (RDR-012)
# -----------------------------------------------------------------------------

class TestResultFormatValidation:
    """Tests for validating search result format."""

    def test_result_contains_file_path(self, meili_client, indexed_auth_project):
        """Test that results contain file_path."""
        index_name = indexed_auth_project['index_name']

        results = meili_client.search(index_name, "authenticate", limit=10)

        assert results['hits'], "Expected at least one result"

        for hit in results['hits']:
            assert 'file_path' in hit, "Result should contain file_path"
            assert hit['file_path'], "file_path should not be empty"

    def test_result_contains_line_numbers(self, meili_client, indexed_auth_project):
        """Test that results contain line number information."""
        index_name = indexed_auth_project['index_name']

        results = meili_client.search(index_name, "authenticate", limit=10)

        assert results['hits'], "Expected at least one result"

        for hit in results['hits']:
            # Should have start_line and end_line per RDR-011
            assert 'start_line' in hit, "Result should contain start_line"
            assert 'end_line' in hit, "Result should contain end_line"
            assert 'line_count' in hit, "Result should contain line_count"

            # Validate line number consistency
            assert hit['start_line'] >= 1, "start_line should be >= 1"
            assert hit['end_line'] >= hit['start_line'], "end_line should be >= start_line"
            assert hit['line_count'] == hit['end_line'] - hit['start_line'] + 1

    def test_result_contains_function_name(self, meili_client, indexed_auth_project):
        """Test that function results contain function_name."""
        index_name = indexed_auth_project['index_name']

        results = meili_client.search(
            index_name,
            "authenticate",
            filter='code_type = "function"',
            limit=10
        )

        assert results['hits'], "Expected at least one function result"

        for hit in results['hits']:
            assert 'function_name' in hit, "Function result should contain function_name"

    def test_result_contains_class_name(self, meili_client, indexed_auth_project):
        """Test that class results contain class_name."""
        index_name = indexed_auth_project['index_name']

        results = meili_client.search(
            index_name,
            "",
            filter='code_type = "class"',
            limit=10
        )

        assert results['hits'], "Expected at least one class result"

        for hit in results['hits']:
            assert 'class_name' in hit, "Class result should contain class_name"

    def test_result_contains_content(self, meili_client, indexed_auth_project):
        """Test that results contain code content."""
        index_name = indexed_auth_project['index_name']

        results = meili_client.search(index_name, "authenticate", limit=10)

        assert results['hits'], "Expected at least one result"

        for hit in results['hits']:
            assert 'content' in hit, "Result should contain content"
            assert hit['content'], "content should not be empty"

    def test_result_contains_git_metadata(self, meili_client, indexed_auth_project):
        """Test that results contain git metadata."""
        index_name = indexed_auth_project['index_name']

        results = meili_client.search(index_name, "authenticate", limit=10)

        assert results['hits'], "Expected at least one result"

        for hit in results['hits']:
            # Git metadata from RDR-011
            assert 'git_project_name' in hit, "Result should contain git_project_name"
            assert 'git_branch' in hit, "Result should contain git_branch"
            assert 'git_project_identifier' in hit, "Result should contain git_project_identifier"
            assert 'git_commit_hash' in hit, "Result should contain git_commit_hash"


# -----------------------------------------------------------------------------
# Cooperative Workflow Tests (RDR-012: Semantic -> Exact)
# -----------------------------------------------------------------------------

class TestCooperativeWorkflow:
    """Tests for cooperative workflow: semantic search followed by exact search.

    This workflow is documented in RDR-012:
    1. arc search semantic "authentication" --collection MyCode
    2. Note file path from results
    3. arc search text "def authenticate" --index MyCode-fulltext --filter "file_path CONTAINS /auth/"

    Expected:
    - Semantic search finds relevant files
    - Exact search verifies specific implementation
    - Both use same file metadata
    """

    @pytest.fixture
    def dual_indexed_project(self, meili_client, qdrant_client, test_index, git_project_with_auth):
        """Index project in both MeiliSearch (full-text) and Qdrant (semantic)."""
        # Index in MeiliSearch for full-text search
        ft_indexer = SourceCodeFullTextIndexer(
            meili_client=meili_client,
            index_name=test_index,
            batch_size=100
        )

        ft_stats = ft_indexer.index_single_project(
            project_root=git_project_with_auth,
            force=True,
            verbose=False
        )

        # For Qdrant semantic indexing, we'll use mocked results
        # since setting up embedding models in tests is complex
        yield {
            'fulltext_index': test_index,
            'project_path': git_project_with_auth,
            'ft_stats': ft_stats,
        }

    def test_step1_semantic_finds_auth_files(self, meili_client, dual_indexed_project):
        """Step 1: Verify semantic-style search finds authentication-related files.

        Note: This uses full-text search to simulate semantic search behavior
        since embedding models aren't available in tests.
        """
        index_name = dual_indexed_project['fulltext_index']

        # Simulate semantic search for "authentication patterns"
        # In real workflow, this would be: arc search semantic "authentication" --collection MyCode
        results = meili_client.search(
            index_name,
            "authentication user verify",  # Conceptual query
            limit=10
        )

        assert results['estimatedTotalHits'] >= 1, "Should find authentication-related results"

        # Collect file paths from results
        auth_file_paths = []
        for hit in results['hits']:
            file_path = hit.get('file_path', '')
            if '/auth/' in file_path:
                auth_file_paths.append(file_path)

        assert len(auth_file_paths) >= 1, "Should find files in /auth/ directory"

    def test_step2_exact_search_in_discovered_path(self, meili_client, dual_indexed_project):
        """Step 2: Verify exact search finds specific implementation in discovered path.

        This simulates:
        arc search text "def authenticate" --index MyCode-fulltext --filter "file_path CONTAINS /auth/"

        Note: Uses code_type filter as fallback if CONTAINS is not available.
        """
        index_name = dual_indexed_project['fulltext_index']

        # Try with CONTAINS first, fall back to code_type filter
        try:
            results = meili_client.search(
                index_name,
                '"def authenticate"',
                filter='file_path CONTAINS "/auth/"',
                limit=10
            )
        except Exception as e:
            if 'feature_not_enabled' in str(e) or 'CONTAINS' in str(e):
                # Fall back to code_type filter (which is filterable)
                results = meili_client.search(
                    index_name,
                    '"def authenticate"',
                    filter='code_type = "function"',
                    limit=10
                )
            else:
                raise

        assert results['estimatedTotalHits'] >= 1, \
            "Should find 'def authenticate' in /auth/ directory"

        # Verify we found the actual function definition
        found_authenticate = False
        for hit in results['hits']:
            function_name = hit.get('function_name', '')
            if function_name == 'authenticate':
                found_authenticate = True

                # Verify it's in the auth directory
                file_path = hit.get('file_path', '')
                assert '/auth/' in file_path
                assert file_path.endswith('verify.py')

                # Verify line numbers exist
                assert 'start_line' in hit
                assert 'end_line' in hit
                break

        assert found_authenticate, "Should find authenticate function"

    def test_cooperative_workflow_complete(self, meili_client, dual_indexed_project):
        """Test complete cooperative workflow as described in RDR-012.

        Workflow:
        1. Semantic search finds auth-related files
        2. Extract file paths from results
        3. Exact search verifies specific implementation

        Note: Uses code_type filter as fallback if CONTAINS is not available.
        """
        index_name = dual_indexed_project['fulltext_index']

        # Step 1: Conceptual search (simulating semantic)
        conceptual_results = meili_client.search(
            index_name,
            "authentication user credentials",
            limit=20
        )

        assert conceptual_results['estimatedTotalHits'] >= 1

        # Step 2: Extract relevant file paths
        relevant_paths = set()
        for hit in conceptual_results['hits']:
            file_path = hit.get('file_path', '')
            if file_path:
                # Extract directory path pattern for filter
                if '/auth/' in file_path:
                    relevant_paths.add('/auth/')

        assert relevant_paths, "Should discover /auth/ directory"

        # Step 3: Exact search in discovered paths (with fallback)
        exact_search_succeeded = False
        exact_results = None

        # Try CONTAINS filter first
        for path_pattern in relevant_paths:
            try:
                exact_results = meili_client.search(
                    index_name,
                    '"def authenticate"',
                    filter=f'file_path CONTAINS "{path_pattern}"',
                    limit=10
                )
                exact_search_succeeded = True
            except Exception as e:
                if 'feature_not_enabled' not in str(e) and 'CONTAINS' not in str(e):
                    raise
                # CONTAINS not available, try fallback
                break

            # Verify exact match found
            if exact_results and exact_results['estimatedTotalHits'] >= 1:
                hit = exact_results['hits'][0]
                assert 'file_path' in hit
                assert 'start_line' in hit
                assert 'end_line' in hit
                assert 'content' in hit
                assert 'git_project_name' in hit
                return  # Test passed

        # Fallback: Use code_type filter (which is filterable)
        if not exact_search_succeeded:
            exact_results = meili_client.search(
                index_name,
                '"def authenticate"',
                filter='code_type = "function"',
                limit=10
            )

            assert exact_results['estimatedTotalHits'] >= 1, \
                "Should find authenticate function with code_type filter"

            # Verify we found the function in /auth/ directory
            found_in_auth = False
            for hit in exact_results['hits']:
                if hit.get('function_name') == 'authenticate':
                    file_path = hit.get('file_path', '')
                    if '/auth/' in file_path:
                        found_in_auth = True
                        # Both searches use same metadata structure
                        assert 'file_path' in hit
                        assert 'start_line' in hit
                        assert 'end_line' in hit
                        assert 'content' in hit
                        assert 'git_project_name' in hit
                        break

            assert found_in_auth, "Should find authenticate function in /auth/ directory"

    def test_metadata_consistency_between_searches(self, meili_client, dual_indexed_project):
        """Verify both search types use same file metadata."""
        index_name = dual_indexed_project['fulltext_index']

        # Search for same function with different queries
        conceptual_results = meili_client.search(
            index_name,
            "authenticate user password",
            limit=10
        )

        exact_results = meili_client.search(
            index_name,
            '"def authenticate"',
            filter='code_type = "function"',
            limit=10
        )

        # Both should find the authenticate function
        assert conceptual_results['hits'], "Conceptual search should return results"
        assert exact_results['hits'], "Exact search should return results"

        # Find the authenticate function in both results
        conceptual_auth = None
        exact_auth = None

        for hit in conceptual_results['hits']:
            if hit.get('function_name') == 'authenticate':
                conceptual_auth = hit
                break

        for hit in exact_results['hits']:
            if hit.get('function_name') == 'authenticate':
                exact_auth = hit
                break

        # If both found, verify metadata matches
        if conceptual_auth and exact_auth:
            # Same file path
            assert conceptual_auth['file_path'] == exact_auth['file_path']

            # Same line numbers
            assert conceptual_auth['start_line'] == exact_auth['start_line']
            assert conceptual_auth['end_line'] == exact_auth['end_line']

            # Same git metadata
            assert conceptual_auth['git_project_name'] == exact_auth['git_project_name']
            assert conceptual_auth['git_branch'] == exact_auth['git_branch']


# -----------------------------------------------------------------------------
# Error Handling Tests (RDR-012)
# -----------------------------------------------------------------------------

class TestErrorHandling:
    """Tests for error handling in search workflow."""

    def test_search_nonexistent_index(self, meili_client):
        """Test searching a non-existent index raises appropriate error."""
        with pytest.raises(Exception):
            meili_client.search(
                "nonexistent_index_xyz123",
                "test query",
                limit=10
            )

    def test_invalid_filter_syntax(self, meili_client, indexed_auth_project):
        """Test that invalid filter syntax is handled gracefully."""
        index_name = indexed_auth_project['index_name']

        # Invalid filter syntax should raise an error
        with pytest.raises(Exception):
            meili_client.search(
                index_name,
                "test",
                filter='invalid syntax !!!',
                limit=10
            )

    def test_search_empty_index(self, meili_client):
        """Test searching an empty index returns no results."""
        index_name = "test_workflow_empty"

        # Create empty index
        if meili_client.index_exists(index_name):
            meili_client.delete_index(index_name)

        meili_client.create_index(name=index_name, primary_key='id')

        try:
            results = meili_client.search(
                index_name,
                "test query",
                limit=10
            )

            assert results['hits'] == []
            assert results['estimatedTotalHits'] == 0
        finally:
            meili_client.delete_index(index_name)


# -----------------------------------------------------------------------------
# JSON Output Format Tests (RDR-012)
# -----------------------------------------------------------------------------

class TestJsonOutputFormat:
    """Tests for JSON output format validation."""

    def test_search_result_structure(self, meili_client, indexed_auth_project):
        """Test that search results have expected JSON structure."""
        index_name = indexed_auth_project['index_name']

        results = meili_client.search(
            index_name,
            "authenticate",
            limit=10
        )

        # Verify top-level structure
        assert 'hits' in results
        assert 'processingTimeMs' in results
        assert 'estimatedTotalHits' in results
        assert 'query' in results

        # Verify hits is a list
        assert isinstance(results['hits'], list)

        # Verify processing time is numeric
        assert isinstance(results['processingTimeMs'], (int, float))

        # Verify estimated total is numeric
        assert isinstance(results['estimatedTotalHits'], int)

    def test_hit_structure(self, meili_client, indexed_auth_project):
        """Test that individual hits have expected structure."""
        index_name = indexed_auth_project['index_name']

        results = meili_client.search(
            index_name,
            "authenticate",
            attributes_to_highlight=['content'],
            limit=10
        )

        assert results['hits'], "Expected at least one hit"

        hit = results['hits'][0]

        # Required fields per RDR-011/012
        required_fields = [
            'id',
            'file_path',
            'content',
            'start_line',
            'end_line',
            'programming_language',
            'code_type',
        ]

        for field in required_fields:
            assert field in hit, f"Hit should contain '{field}'"

        # Verify _formatted when highlighting is requested
        assert '_formatted' in hit, "Hit should contain '_formatted' when highlighting"
        assert 'content' in hit['_formatted'], "_formatted should contain 'content'"

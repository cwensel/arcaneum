"""Integration tests for source code full-text indexing (RDR-011).

These tests require:
- Running MeiliSearch server
- Git installed

Run with: pytest tests/integration/test_fulltext_code_indexing.py -v
"""

import os
import pytest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from arcaneum.fulltext.client import FullTextClient
from arcaneum.fulltext.indexes import SOURCE_CODE_FULLTEXT_SETTINGS
from arcaneum.indexing.fulltext.code_indexer import SourceCodeFullTextIndexer
from arcaneum.indexing.fulltext.sync import GitCodeMetadataSync
from arcaneum.paths import get_meilisearch_api_key


MEILISEARCH_URL = os.environ.get("MEILISEARCH_URL", "http://localhost:7700")
MEILISEARCH_API_KEY = os.environ.get("MEILISEARCH_API_KEY") or get_meilisearch_api_key()


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
            if idx['uid'].startswith("test_code_"):
                client.delete_index(idx['uid'])
    except Exception:
        pass


@pytest.fixture
def test_index(meili_client):
    """Create a test index with source code fulltext settings."""
    index_name = "test_code_fulltext"

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
def git_project():
    """Create a temporary git repository with sample code."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir) / "test-project"
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

        # Create source directory
        src_dir = project_path / "src"
        src_dir.mkdir()

        # Create Python file with functions and classes
        (src_dir / "main.py").write_text('''"""Main module for test project."""

import os

CONSTANT = 42


def main():
    """Entry point."""
    print("Hello, World!")
    return CONSTANT


class Application:
    """Main application class."""

    def __init__(self, name):
        self.name = name

    def run(self):
        """Run the application."""
        print(f"Running {self.name}")


def helper_function():
    """Helper utility."""
    return main() * 2
''')

        # Create JavaScript file
        (src_dir / "utils.js").write_text('''// Utility functions

function formatDate(date) {
    return date.toISOString();
}

class DateFormatter {
    constructor(locale) {
        this.locale = locale;
    }

    format(date) {
        return date.toLocaleDateString(this.locale);
    }
}

function calculateSum(numbers) {
    return numbers.reduce((a, b) => a + b, 0);
}
''')

        # Create TypeScript file
        (src_dir / "types.ts").write_text('''// Type definitions

interface User {
    id: number;
    name: string;
    email: string;
}

function createUser(id: number, name: string, email: string): User {
    return { id, name, email };
}

class UserService {
    private users: User[] = [];

    addUser(user: User): void {
        this.users.push(user);
    }

    findById(id: number): User | undefined {
        return this.users.find(u => u.id === id);
    }
}
''')

        # Git add and commit
        subprocess.run(["git", "add", "."], cwd=project_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=project_path, check=True, capture_output=True
        )

        yield str(project_path)


class TestSourceCodeFullTextIndexerIntegration:
    """Integration tests for source code indexing to MeiliSearch."""

    def test_index_single_project(self, meili_client, test_index, git_project):
        """Test indexing a single git project."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=meili_client,
            index_name=test_index,
            batch_size=100
        )

        # Index the project
        stats = indexer.index_single_project(
            project_root=git_project,
            force=True,
            verbose=False
        )

        assert stats['indexed_projects'] == 1
        assert stats['indexed_files'] >= 3  # main.py, utils.js, types.ts
        assert stats['total_definitions'] >= 6  # Multiple functions/classes

        # Verify documents in index
        index_stats = meili_client.get_index_stats(test_index)
        assert index_stats['numberOfDocuments'] >= 6

    def test_search_function_name(self, meili_client, test_index, git_project):
        """Test searching by function name."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=meili_client,
            index_name=test_index,
            batch_size=100
        )

        indexer.index_single_project(project_root=git_project, force=True)

        # Search for function name
        results = meili_client.search(
            test_index,
            "helper_function",
            limit=10
        )

        assert results['estimatedTotalHits'] >= 1
        # Verify we found the helper_function
        hits = results['hits']
        assert any(
            h.get('function_name') == 'helper_function' or
            'helper_function' in h.get('content', '')
            for h in hits
        )

    def test_search_class_name(self, meili_client, test_index, git_project):
        """Test searching by class name."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=meili_client,
            index_name=test_index,
            batch_size=100
        )

        indexer.index_single_project(project_root=git_project, force=True)

        # Search for class name
        results = meili_client.search(
            test_index,
            "Application",
            limit=10
        )

        assert results['estimatedTotalHits'] >= 1
        hits = results['hits']
        assert any(
            h.get('class_name') == 'Application' or
            'Application' in h.get('content', '')
            for h in hits
        )

    def test_search_code_content(self, meili_client, test_index, git_project):
        """Test searching in code content."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=meili_client,
            index_name=test_index,
            batch_size=100
        )

        indexer.index_single_project(project_root=git_project, force=True)

        # Search for code content
        results = meili_client.search(
            test_index,
            "print Hello World",
            limit=10
        )

        assert results['estimatedTotalHits'] >= 1

    def test_filter_by_language(self, meili_client, test_index, git_project):
        """Test filtering by programming language."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=meili_client,
            index_name=test_index,
            batch_size=100
        )

        indexer.index_single_project(project_root=git_project, force=True)

        # Search with language filter
        results = meili_client.search(
            test_index,
            "",  # Empty query to get all
            filter='programming_language = "python"',
            limit=100
        )

        assert results['estimatedTotalHits'] >= 1
        for hit in results['hits']:
            assert hit['programming_language'] == 'python'

    def test_filter_by_code_type(self, meili_client, test_index, git_project):
        """Test filtering by code type (function, class, etc.)."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=meili_client,
            index_name=test_index,
            batch_size=100
        )

        indexer.index_single_project(project_root=git_project, force=True)

        # Search for classes only
        results = meili_client.search(
            test_index,
            "",
            filter='code_type = "class"',
            limit=100
        )

        assert results['estimatedTotalHits'] >= 1
        for hit in results['hits']:
            assert hit['code_type'] == 'class'

    def test_change_detection_skip_unchanged(self, meili_client, test_index, git_project):
        """Test that unchanged projects are skipped on re-index."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=meili_client,
            index_name=test_index,
            batch_size=100
        )

        # First index
        stats1 = indexer.index_single_project(project_root=git_project, force=True)
        assert stats1['indexed_projects'] == 1

        # Clear cache so second index sees the indexed project
        indexer.sync.clear_cache()

        # Second index without force should skip
        stats2 = indexer.index_single_project(project_root=git_project, force=False)
        assert stats2['skipped_projects'] == 1
        assert stats2['indexed_projects'] == 0

    def test_delete_project(self, meili_client, test_index, git_project):
        """Test deleting all documents for a project."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=meili_client,
            index_name=test_index,
            batch_size=100
        )

        # Index the project
        indexer.index_single_project(project_root=git_project, force=True)

        # Clear cache and get project identifier
        indexer.sync.clear_cache()
        projects = indexer.get_indexed_projects()
        assert len(projects) == 1
        project_identifier = list(projects.keys())[0]

        # Verify documents exist
        stats = meili_client.get_index_stats(test_index)
        initial_count = stats['numberOfDocuments']
        assert initial_count > 0

        # Delete project
        deleted = indexer.delete_project(project_identifier)
        assert deleted > 0

        # Verify documents deleted
        stats = meili_client.get_index_stats(test_index)
        assert stats['numberOfDocuments'] == 0

    def test_get_indexed_projects(self, meili_client, test_index, git_project):
        """Test getting list of indexed projects."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=meili_client,
            index_name=test_index,
            batch_size=100
        )

        # Initially empty
        projects = indexer.get_indexed_projects()
        # Clear any cache
        indexer.sync.clear_cache()

        # Index project
        indexer.index_single_project(project_root=git_project, force=True)

        # Clear cache to get fresh data
        indexer.sync.clear_cache()
        projects = indexer.get_indexed_projects()

        assert len(projects) >= 1
        # Project identifier format: "name#branch"
        identifier = list(projects.keys())[0]
        assert "#" in identifier

    def test_line_number_stored(self, meili_client, test_index, git_project):
        """Test that line numbers are stored correctly."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=meili_client,
            index_name=test_index,
            batch_size=100
        )

        indexer.index_single_project(project_root=git_project, force=True)

        # Get all documents
        results = meili_client.search(
            test_index,
            "",
            limit=100
        )

        # Verify all documents have line numbers
        for hit in results['hits']:
            assert 'start_line' in hit
            assert 'end_line' in hit
            assert 'line_count' in hit
            assert hit['start_line'] >= 1
            assert hit['end_line'] >= hit['start_line']
            assert hit['line_count'] == hit['end_line'] - hit['start_line'] + 1


class TestSourceCodeFullTextSettingsIntegration:
    """Tests for SOURCE_CODE_FULLTEXT_SETTINGS in MeiliSearch."""

    def test_code_settings_applied(self, meili_client, test_index):
        """Test that SOURCE_CODE_FULLTEXT_SETTINGS are correctly applied."""
        settings = meili_client.get_index_settings(test_index)

        # Check searchable attributes (RDR-011)
        assert "content" in settings["searchableAttributes"]
        assert "function_name" in settings["searchableAttributes"]
        assert "class_name" in settings["searchableAttributes"]
        assert "qualified_name" in settings["searchableAttributes"]
        assert "filename" in settings["searchableAttributes"]

        # Check filterable attributes (RDR-011)
        assert "programming_language" in settings["filterableAttributes"]
        assert "git_project_name" in settings["filterableAttributes"]
        assert "git_branch" in settings["filterableAttributes"]
        assert "git_project_identifier" in settings["filterableAttributes"]
        assert "git_commit_hash" in settings["filterableAttributes"]
        assert "file_path" in settings["filterableAttributes"]
        assert "code_type" in settings["filterableAttributes"]
        assert "file_extension" in settings["filterableAttributes"]

        # Check sortable (RDR-011)
        assert "start_line" in settings["sortableAttributes"]

        # Check pagination (RDR-011: high limit for function-level)
        assert settings["pagination"]["maxTotalHits"] == 10000


class TestMultiBranchSupport:
    """Tests for multi-branch indexing support."""

    @pytest.fixture
    def git_project_with_branches(self):
        """Create a git project with multiple branches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "multi-branch-project"
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

            # Create initial file on main
            (project_path / "main.py").write_text('''
def main():
    print("main branch")
''')

            subprocess.run(["git", "add", "."], cwd=project_path, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=project_path, check=True, capture_output=True
            )

            yield str(project_path)

    def test_branch_specific_identifier(self, meili_client, test_index, git_project_with_branches):
        """Test that branch is included in project identifier."""
        indexer = SourceCodeFullTextIndexer(
            meili_client=meili_client,
            index_name=test_index,
            batch_size=100
        )

        indexer.index_single_project(project_root=git_project_with_branches, force=True)

        # Get indexed projects
        indexer.sync.clear_cache()
        projects = indexer.get_indexed_projects()

        assert len(projects) == 1
        identifier = list(projects.keys())[0]

        # Identifier should be "project#branch" format
        assert "#" in identifier
        parts = identifier.split("#")
        assert len(parts) == 2
        project_name, branch = parts
        assert project_name == "multi-branch-project"
        # Branch could be 'main' or 'master' depending on git config
        assert branch in ['main', 'master']

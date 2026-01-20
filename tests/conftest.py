"""Shared pytest fixtures for Arcaneum CLI tests."""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Auto-mark tests based on directory
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Get the test file path relative to tests/
        test_path = str(item.fspath)

        if '/tests/unit/' in test_path or '\\tests\\unit\\' in test_path:
            item.add_marker(pytest.mark.unit)
        elif '/tests/integration/' in test_path or '\\tests\\integration\\' in test_path:
            item.add_marker(pytest.mark.integration)
        elif '/tests/e2e/' in test_path or '\\tests\\e2e\\' in test_path:
            item.add_marker(pytest.mark.e2e)


# ============================================================================
# Mock Qdrant Client Fixtures
# ============================================================================

@pytest.fixture
def mock_qdrant_client():
    """Create a mocked QdrantClient for testing."""
    client = MagicMock()

    # Mock common methods
    client.get_collections.return_value = MagicMock(collections=[])
    client.get_collection.return_value = MagicMock(
        points_count=0,
        status="green",
        config=MagicMock(
            params=MagicMock(vectors={}),
            hnsw_config=MagicMock(m=16, ef_construct=100)
        )
    )
    client.scroll.return_value = ([], None)
    client.create_collection.return_value = True
    client.delete_collection.return_value = True

    return client


@pytest.fixture
def mock_qdrant_client_with_collections(mock_qdrant_client):
    """Mock Qdrant client with sample collections."""
    from unittest.mock import MagicMock

    # Create sample collections
    collection1 = MagicMock()
    collection1.name = "TestCollection"

    collection2 = MagicMock()
    collection2.name = "CodeCollection"

    mock_qdrant_client.get_collections.return_value = MagicMock(
        collections=[collection1, collection2]
    )

    # Mock get_collection for each
    def get_collection_side_effect(name):
        if name == "TestCollection":
            return MagicMock(
                points_count=100,
                status="green",
                config=MagicMock(
                    params=MagicMock(vectors={
                        "stella": MagicMock(size=1024, distance="Cosine")
                    }),
                    hnsw_config=MagicMock(m=16, ef_construct=100)
                )
            )
        elif name == "CodeCollection":
            return MagicMock(
                points_count=500,
                status="green",
                config=MagicMock(
                    params=MagicMock(vectors={
                        "jina-code": MagicMock(size=768, distance="Cosine")
                    }),
                    hnsw_config=MagicMock(m=16, ef_construct=100)
                )
            )
        raise Exception(f"Collection {name} not found")

    mock_qdrant_client.get_collection.side_effect = get_collection_side_effect

    return mock_qdrant_client


# ============================================================================
# Mock MeiliSearch Client Fixtures
# ============================================================================

@pytest.fixture
def mock_meili_client():
    """Create a mocked FullTextClient for testing."""
    client = MagicMock()
    client.health_check.return_value = True
    client.index_exists.return_value = True
    client.get_indexes.return_value = []
    client.search.return_value = {
        'hits': [],
        'estimatedTotalHits': 0,
        'processingTimeMs': 5,
    }
    return client


@pytest.fixture
def mock_meili_client_with_indexes(mock_meili_client):
    """Mock MeiliSearch client with sample indexes."""
    mock_meili_client.get_indexes.return_value = [
        {'uid': 'TestIndex', 'primaryKey': 'id'},
        {'uid': 'CodeIndex', 'primaryKey': 'id'},
    ]
    return mock_meili_client


# ============================================================================
# Subprocess Mock Fixtures (for container commands)
# ============================================================================

@pytest.fixture
def mock_subprocess():
    """Mock subprocess.run for container command tests."""
    with patch('subprocess.run') as mock_run:
        # Default to successful command execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        yield mock_run


@pytest.fixture
def mock_docker_available(mock_subprocess):
    """Mock Docker being available."""
    with patch('shutil.which', return_value='/usr/bin/docker'):
        yield mock_subprocess


@pytest.fixture
def mock_docker_unavailable():
    """Mock Docker not being available."""
    with patch('shutil.which', return_value=None):
        yield


# ============================================================================
# Interaction Logger Fixtures
# ============================================================================

@pytest.fixture
def mock_interaction_logger():
    """Mock the interaction logger."""
    with patch('arcaneum.cli.interaction_logger.interaction_logger') as mock_logger:
        yield mock_logger


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_git_repo(temp_dir):
    """Create a temporary git repository."""
    import subprocess

    repo_dir = temp_dir / "test_repo"
    repo_dir.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_dir, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_dir, capture_output=True
    )

    # Create a sample file and commit
    test_file = repo_dir / "test.py"
    test_file.write_text("def hello():\n    print('Hello')\n")
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_dir, capture_output=True
    )

    yield repo_dir


@pytest.fixture
def sample_markdown_files(temp_dir):
    """Create sample markdown files for testing."""
    docs_dir = temp_dir / "docs"
    docs_dir.mkdir()

    # Create sample markdown files
    (docs_dir / "readme.md").write_text("""# Test Documentation

This is a test markdown file.

## Section 1

Some content here.

## Section 2

More content here.
""")

    (docs_dir / "guide.md").write_text("""# User Guide

Instructions for using the software.

## Getting Started

First, install the dependencies.
""")

    yield docs_dir


@pytest.fixture
def sample_pdf_file(temp_dir):
    """Create a sample PDF file path for testing (actual PDF creation requires external libs)."""
    # For unit tests, we just need the path
    pdf_path = temp_dir / "sample.pdf"
    # Create empty file to simulate PDF existence
    pdf_path.touch()
    yield pdf_path


# ============================================================================
# Health Check Fixtures
# ============================================================================

@pytest.fixture
def mock_qdrant_health():
    """Mock Qdrant health check."""
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def mock_services_healthy(mock_qdrant_health):
    """Mock both Qdrant and MeiliSearch as healthy."""
    with patch('arcaneum.cli.docker.check_qdrant_health', return_value=True):
        with patch('arcaneum.cli.docker.check_meilisearch_health', return_value=True):
            yield


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def clean_env():
    """Provide a clean environment without Arcaneum env vars."""
    env_vars = [
        'QDRANT_URL', 'QDRANT_API_KEY',
        'MEILISEARCH_URL', 'MEILISEARCH_API_KEY',
    ]
    old_values = {}
    for var in env_vars:
        old_values[var] = os.environ.pop(var, None)

    yield

    # Restore old values
    for var, value in old_values.items():
        if value is not None:
            os.environ[var] = value


# ============================================================================
# Output Capture Helpers
# ============================================================================

@pytest.fixture
def capture_json_output(capsys):
    """Helper to capture and parse JSON output."""
    def _capture():
        captured = capsys.readouterr()
        try:
            return json.loads(captured.out)
        except json.JSONDecodeError:
            return {"raw": captured.out, "error": "Not valid JSON"}
    return _capture


# ============================================================================
# Collection Metadata Fixtures
# ============================================================================

@pytest.fixture
def mock_collection_metadata():
    """Mock collection metadata functions."""
    with patch('arcaneum.cli.collections.get_collection_metadata') as mock_get:
        with patch('arcaneum.cli.collections.set_collection_metadata') as mock_set:
            with patch('arcaneum.cli.collections.get_collection_type') as mock_type:
                mock_get.return_value = {"model": "stella", "collection_type": "pdf"}
                mock_type.return_value = "pdf"
                yield {
                    'get': mock_get,
                    'set': mock_set,
                    'type': mock_type,
                }


# ============================================================================
# CLI Runner Fixture
# ============================================================================

@pytest.fixture
def cli_runner():
    """Provide Click's CliRunner for testing CLI commands."""
    from click.testing import CliRunner
    return CliRunner()

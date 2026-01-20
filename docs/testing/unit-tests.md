# Running Unit Tests

This document covers running the automated test suite for the Arcaneum CLI.

## Quick Start

```bash
# Run all unit tests
pytest -m unit

# Run CLI unit tests only
pytest -m unit tests/unit/cli/

# Run with verbose output
pytest -m unit -v
```

## Test Markers

Tests are automatically marked based on their directory location:

| Marker        | Directory             | Description                                    |
| ------------- | --------------------- | ---------------------------------------------- |
| `unit`        | `tests/unit/`         | Unit tests (no external dependencies)          |
| `integration` | `tests/integration/`  | Integration tests (require Qdrant/MeiliSearch) |
| `e2e`         | `tests/e2e/`          | End-to-end tests                               |
| `slow`        | N/A                   | Tests taking >5 seconds (manual marker)        |

## Running Tests by Category

```bash
# Unit tests only (fast, no services required)
pytest -m unit

# Integration tests (requires running services)
pytest -m integration

# All tests except slow ones
pytest -m "not slow"

# Specific test file
pytest tests/unit/cli/test_container_commands.py

# Specific test class
pytest tests/unit/cli/test_container_commands.py::TestContainerStart

# Specific test function
pytest tests/unit/cli/test_container_commands.py::TestContainerStart::test_docker_compose_up_called
```

## CLI Test Coverage

The CLI tests cover the following command groups:

| Test File                      | Commands Tested                                          |
| ------------------------------ | -------------------------------------------------------- |
| `test_collection_commands.py`  | `arc collection list/info/delete/items/verify`           |
| `test_config_commands.py`      | `arc config show-cache-dir/clear-cache`                  |
| `test_container_commands.py`   | `arc container start/stop/restart/status/logs/reset`     |
| `test_corpus_commands.py`      | `arc corpus create/list/delete/info/verify`              |
| `test_doctor_command.py`       | `arc doctor`                                             |
| `test_index_commands.py`       | `arc index pdf/code/markdown`                            |
| `test_indexes_commands.py`     | `arc indexes create/list/info/delete/verify/items/export`|
| `test_models_command.py`       | `arc models list`                                        |
| `test_search_commands.py`      | `arc search semantic`                                    |
| `test_store_command.py`        | `arc store`                                              |

## Test Configuration

Test configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
markers = [
    "unit: Unit tests (no external dependencies)",
    "integration: Integration tests (require services)",
    "e2e: End-to-end tests",
    "slow: Tests taking >5 seconds",
]
addopts = "-v --tb=short"
```

## Shared Fixtures

Common test fixtures are in `tests/conftest.py`:

- `mock_qdrant_client` - Mocked Qdrant client
- `mock_meili_client` - Mocked MeiliSearch client
- `mock_interaction_logger` - Mocked interaction logger
- `temp_dir` - Temporary directory for test files
- `temp_git_repo` - Temporary git repository
- `sample_markdown_files` - Sample markdown test files
- `sample_pdf_file` - Sample PDF test file

## Writing New Tests

1. Place unit tests in `tests/unit/cli/`
2. Use the naming convention `test_<module>_commands.py`
3. Use shared fixtures from `conftest.py`
4. Mock external dependencies (Qdrant, MeiliSearch, subprocess)

Example test structure:

```python
"""CLI tests for example command."""

import json
from unittest.mock import MagicMock, patch
import pytest


class TestExampleCommand:
    """Test 'arc example' command."""

    def test_basic_functionality(self, mock_qdrant_client, capsys):
        """Test basic command execution."""
        from arcaneum.cli.example import example_command

        with patch('arcaneum.cli.example.create_qdrant_client', return_value=mock_qdrant_client):
            example_command(arg1='value', output_json=True)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output['status'] == 'success'
```

## Troubleshooting

### Tests Not Selected with `-m unit`

Ensure `tests/conftest.py` contains the auto-marking hook:

```python
def pytest_collection_modifyitems(config, items):
    for item in items:
        test_path = str(item.fspath)
        if '/tests/unit/' in test_path:
            item.add_marker(pytest.mark.unit)
```

### Module Import Errors

Ensure the package is installed in development mode:

```bash
pip install -e .
```

### Mock Not Working

When mocking functions imported inside a CLI function, patch at the source module:

```python
# Wrong - patches at import location
with patch('arcaneum.cli.search.SearchEmbedder', ...):

# Right - patches at source module (if imported inside function)
with patch('arcaneum.search.SearchEmbedder', ...):
```

## Pre-Commit Testing

Run unit tests before committing:

```bash
pytest -m unit --tb=short
```

For comprehensive testing before a release:

```bash
# Full unit test suite
pytest -m unit -v

# With coverage report
pytest -m unit --cov=src/arcaneum/cli --cov-report=term-missing
```

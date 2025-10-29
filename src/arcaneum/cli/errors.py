"""Custom exception classes for CLI error handling (RDR-006).

This module defines exception classes that map to specific exit codes following
Beads best practices for structured error handling.

Exit Codes:
- 0: Success
- 1: General error
- 2: Invalid arguments
- 3: Resource not found
"""


# Exit codes (same as main.py)
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INVALID_ARGS = 2
EXIT_NOT_FOUND = 3


class ArcaneumError(Exception):
    """Base exception for Arcaneum CLI errors.

    All custom Arcaneum exceptions should inherit from this class.
    Default exit code is EXIT_ERROR (1).
    """
    exit_code = EXIT_ERROR


class InvalidArgumentError(ArcaneumError):
    """Invalid command line arguments or configuration.

    Examples:
    - Invalid model name
    - Invalid collection type
    - Mutually exclusive options used together
    - Required file/directory doesn't exist

    Exit code: 2
    """
    exit_code = EXIT_INVALID_ARGS


class ResourceNotFoundError(ArcaneumError):
    """Resource not found (collection, file, directory, etc.).

    Examples:
    - Collection doesn't exist in Qdrant
    - File path not found
    - Git repository not found
    - Model not downloaded

    Exit code: 3
    """
    exit_code = EXIT_NOT_FOUND


class CollectionNotFoundError(ResourceNotFoundError):
    """Specific case: Qdrant collection not found."""
    pass


class ModelNotFoundError(ResourceNotFoundError):
    """Specific case: Embedding model not found or not downloaded."""
    pass


class GitRepositoryNotFoundError(ResourceNotFoundError):
    """Specific case: Git repository not found at specified path."""
    pass

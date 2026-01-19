"""Command wrapper utilities for consistent error handling and logging.

This module provides decorators and context managers for CLI command functions
that standardize:
- Interaction logging (RDR-018)
- Error handling with proper exit codes
- Custom exception handling (InvalidArgumentError, ResourceNotFoundError)

Example:
    @with_error_handling("collection", "create")
    def create_collection_command(name: str, output_json: bool):
        # Just the business logic, no try/except needed
        client = create_qdrant_client()
        client.create_collection(name, ...)
"""

import sys
import functools
from contextlib import contextmanager
from typing import Callable, Any, Optional

from ..interaction_logger import interaction_logger
from ..output import print_error
from ..errors import InvalidArgumentError, ResourceNotFoundError


@contextmanager
def command_context(
    namespace: str,
    operation: str,
    output_json: bool = False,
    error_prefix: str = "Failed",
    **log_kwargs
):
    """Context manager for CLI command error handling and logging.

    Provides consistent error handling, interaction logging, and exit behavior
    for CLI commands.

    Args:
        namespace: Logging namespace (e.g., "collection", "corpus", "search")
        operation: Operation name (e.g., "create", "delete", "list")
        output_json: Whether JSON output mode is enabled
        error_prefix: Prefix for error messages (default: "Failed")
        **log_kwargs: Additional keyword arguments passed to interaction_logger.start()

    Yields:
        None

    Raises:
        InvalidArgumentError: Re-raised for CLI to handle
        ResourceNotFoundError: Re-raised for CLI to handle
        SystemExit: On unhandled exceptions (exits with code 1)

    Example:
        def my_command(name: str, output_json: bool):
            with command_context("collection", "create", output_json, collection=name):
                client = create_qdrant_client()
                client.create_collection(name, ...)
    """
    interaction_logger.start(namespace, operation, **log_kwargs)

    try:
        yield
        interaction_logger.finish()

    except (InvalidArgumentError, ResourceNotFoundError):
        interaction_logger.finish(error="invalid argument or resource not found")
        raise  # Re-raise for CLI main() to handle

    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"{error_prefix}: {e}", output_json)
        sys.exit(1)


def with_error_handling(
    namespace: str,
    operation: str,
    error_prefix: str = "Failed",
    log_params: Optional[list] = None,
):
    """Decorator for CLI command functions with consistent error handling.

    Wraps a command function with interaction logging and error handling.
    The decorated function should accept output_json as a parameter
    (either positional or keyword).

    Args:
        namespace: Logging namespace (e.g., "collection", "corpus")
        operation: Operation name (e.g., "create", "delete")
        error_prefix: Prefix for error messages (default: "Failed")
        log_params: List of parameter names to pass to interaction_logger.
                   If None, only namespace and operation are logged.
                   Parameter values are extracted from function arguments.

    Returns:
        Decorator function

    Example:
        @with_error_handling("collection", "create", log_params=["name"])
        def create_collection_command(name: str, model: str, output_json: bool):
            client = create_qdrant_client()
            client.create_collection(name, ...)

    Note:
        - The decorated function MUST have an 'output_json' parameter
        - On success, call interaction_logger.finish() with any result data
        - InvalidArgumentError and ResourceNotFoundError are re-raised
        - All other exceptions result in sys.exit(1)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract output_json from kwargs or try to find it in args
            # by inspecting the function signature
            output_json = kwargs.get('output_json', False)

            # Build log kwargs from specified parameters
            log_kwargs = {}
            if log_params:
                import inspect
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())

                for param_name in log_params:
                    if param_name in kwargs:
                        log_kwargs[param_name] = kwargs[param_name]
                    elif param_name in param_names:
                        idx = param_names.index(param_name)
                        if idx < len(args):
                            log_kwargs[param_name] = args[idx]

            with command_context(
                namespace,
                operation,
                output_json=output_json,
                error_prefix=error_prefix,
                **log_kwargs
            ):
                return func(*args, **kwargs)

        return wrapper
    return decorator

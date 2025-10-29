"""Output formatting utilities for CLI commands (RDR-006).

This module provides consistent JSON and text output formatting for all CLI commands
following Beads best practices.
"""

import json
import sys
from typing import Any, Dict, List, Optional


def format_json_response(
    status: str,
    message: str = "",
    data: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None
) -> str:
    """Format a consistent JSON response.

    Args:
        status: "success" or "error"
        message: Human-readable summary
        data: Command-specific data (optional)
        errors: List of error messages (optional)

    Returns:
        JSON string formatted for output

    Example:
        >>> format_json_response("success", "Created collection", {"name": "MyCollection"})
        '{"status": "success", "message": "Created collection", "data": {"name": "MyCollection"}, "errors": []}'
    """
    response = {
        "status": status,
        "message": message,
        "data": data or {},
        "errors": errors or []
    }
    return json.dumps(response, indent=2)


def print_json(
    status: str,
    message: str = "",
    data: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None
):
    """Print JSON response to stdout.

    Args:
        status: "success" or "error"
        message: Human-readable summary
        data: Command-specific data (optional)
        errors: List of error messages (optional)
    """
    output = format_json_response(status, message, data, errors)
    print(output)


def print_error(message: str, json_output: bool = False):
    """Print error message with [ERROR] prefix.

    Args:
        message: Error message
        json_output: If True, output JSON format instead

    Note:
        JSON errors include the [ERROR] prefix in the message for consistency
    """
    if json_output:
        print_json("error", f"[ERROR] {message}", errors=[message])
    else:
        print(f"[ERROR] {message}", file=sys.stderr)


def print_info(message: str, json_output: bool = False):
    """Print info message with [INFO] prefix.

    Args:
        message: Info message
        json_output: If True, skip output (info messages are text-only)

    Note:
        Info messages are typically progress updates and are not output in JSON mode
    """
    if not json_output:
        print(f"[INFO] {message}")


def print_success(message: str, json_output: bool = False, data: Optional[Dict[str, Any]] = None):
    """Print success message.

    Args:
        message: Success message
        json_output: If True, output JSON format
        data: Optional data to include in JSON output
    """
    if json_output:
        print_json("success", message, data=data)
    else:
        print(message)


def print_warning(message: str, json_output: bool = False):
    """Print warning message with [WARNING] prefix.

    Args:
        message: Warning message
        json_output: If True, skip output (warnings are text-only)

    Note:
        Warning messages are typically for non-fatal issues
    """
    if not json_output:
        print(f"[WARNING] {message}", file=sys.stderr)


def print_progress(current: int, total: int, message: str = "", json_output: bool = False):
    """Print progress update with percentage.

    Args:
        current: Current item number (1-indexed)
        total: Total number of items
        message: Optional message to append
        json_output: If True, skip output (progress is text-only)

    Example:
        >>> print_progress(10, 100, "files processed")
        [INFO] Processing 10/100 (10%) files processed
    """
    if not json_output:
        percentage = (current / total * 100) if total > 0 else 0
        msg = f"[INFO] Processing {current}/{total} ({percentage:.0f}%)"
        if message:
            msg += f" {message}"
        print(msg)


def print_complete(items: int, item_type: str, extra: str = "", json_output: bool = False, data: Optional[Dict[str, Any]] = None):
    """Print completion summary.

    Args:
        items: Number of items processed
        item_type: Type of items (files, chunks, etc.)
        extra: Extra information to append
        json_output: If True, output JSON format
        data: Optional data for JSON output

    Example:
        >>> print_complete(47, "files", "1247 chunks")
        [INFO] Complete: 47 files, 1247 chunks
    """
    msg = f"[INFO] Complete: {items} {item_type}"
    if extra:
        msg += f", {extra}"

    if json_output:
        print_json("success", msg, data=data)
    else:
        print(msg)

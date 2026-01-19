"""Formatting utilities for human-readable output."""

from typing import Union


def format_size(size_bytes: Union[int, float]) -> str:
    """Format bytes as human-readable size.

    Converts byte counts to human-readable strings using standard
    binary units (KB, MB, GB, etc.).

    Args:
        size_bytes: Size in bytes (int or float)

    Returns:
        Formatted string like "1.5 KB", "256.0 MB", etc.

    Examples:
        >>> format_size(1024)
        '1.0 KB'
        >>> format_size(1536)
        '1.5 KB'
        >>> format_size(1048576)
        '1.0 MB'
        >>> format_size(0)
        '0.0 B'
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds (can be float)

    Returns:
        Formatted string like "45s", "3m 45s", or "1h 23m"

    Examples:
        >>> format_duration(45)
        '45s'
        >>> format_duration(185)
        '3m 5s'
        >>> format_duration(3725)
        '1h 2m'
    """
    seconds = int(seconds)

    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        if secs == 0:
            return f"{minutes}m"
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        if minutes == 0:
            return f"{hours}h"
        return f"{hours}h {minutes}m"

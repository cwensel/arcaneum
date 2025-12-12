"""Formatting utilities for human-readable output."""


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

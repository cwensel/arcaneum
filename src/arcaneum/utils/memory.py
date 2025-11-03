"""Memory management utilities for Arcaneum."""

import psutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_available_memory_gb() -> float:
    """Get available system memory in GB.

    Returns:
        Available memory in gigabytes
    """
    mem = psutil.virtual_memory()
    return mem.available / (1024 ** 3)


def calculate_safe_workers(
    requested_workers: int,
    estimated_memory_per_worker_mb: int,
    max_memory_gb: Optional[float] = None,
    min_workers: int = 1
) -> tuple[int, str]:
    """Calculate safe number of workers based on available memory.

    Args:
        requested_workers: Number of workers requested
        estimated_memory_per_worker_mb: Estimated memory usage per worker in MB
        max_memory_gb: Maximum memory to use (None = auto-calculate from available)
        min_workers: Minimum number of workers to return

    Returns:
        Tuple of (safe_worker_count, warning_message)
        warning_message is empty string if no warning needed
    """
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)

    # Determine memory limit
    if max_memory_gb is not None:
        memory_limit_gb = max_memory_gb
        limit_source = "user-specified"
    else:
        # Use 80% of available memory to leave headroom
        memory_limit_gb = available_gb * 0.8
        limit_source = "auto-calculated"

    # Calculate maximum workers based on memory
    memory_per_worker_gb = estimated_memory_per_worker_mb / 1024
    max_workers_by_memory = int(memory_limit_gb / memory_per_worker_gb)

    # Ensure at least min_workers
    safe_workers = max(min_workers, min(requested_workers, max_workers_by_memory))

    # Generate warning if we had to reduce workers
    warning = ""
    if safe_workers < requested_workers:
        reduced_by = requested_workers - safe_workers
        warning = (
            f"⚠️  Reduced workers from {requested_workers} to {safe_workers} "
            f"due to memory constraints\n"
            f"   Available: {available_gb:.1f}GB / {total_gb:.1f}GB total, "
            f"Limit ({limit_source}): {memory_limit_gb:.1f}GB, "
            f"Est. per worker: {estimated_memory_per_worker_mb}MB"
        )

    return safe_workers, warning


def log_memory_stats(prefix: str = ""):
    """Log current memory statistics.

    Args:
        prefix: Optional prefix for log message
    """
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    used_gb = mem.used / (1024 ** 3)

    log_msg = f"{prefix}Memory: {used_gb:.1f}GB used / {total_gb:.1f}GB total ({mem.percent:.1f}%), {available_gb:.1f}GB available"

    if mem.percent > 90:
        logger.warning(log_msg)
    else:
        logger.info(log_msg)

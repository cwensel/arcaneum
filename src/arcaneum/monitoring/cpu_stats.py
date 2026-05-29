"""CPU utilization monitoring for indexing pipelines (RDR-013 Phase 1).

This module provides accurate CPU usage tracking including ONNX Runtime worker threads.
Standard process monitors (htop, top) only show the main Python process (~14% CPU)
while ONNX Runtime spawns worker threads that consume most of the actual CPU cycles.

psutil.Process.cpu_percent() aggregates all thread CPU usage to show true utilization.
"""

import logging
import time
from typing import Dict, Optional

import psutil

logger = logging.getLogger(__name__)


class CPUMonitor:
    """Monitor CPU usage including child processes and threads.

    Tracks true CPU utilization by aggregating main process + all worker threads.
    This is critical for validating performance optimizations in embedding pipelines
    where ONNX Runtime threads do the actual work.
    """

    def __init__(self):
        """Initialize CPU monitor."""
        self.process = psutil.Process()
        self.start_time: Optional[float] = None

    def start(self):
        """Begin monitoring CPU usage."""
        self.start_time = time.time()
        logger.debug("CPU monitoring started")

    def get_stats(self) -> Dict[str, float]:
        """Get current CPU statistics.

        Returns:
            Dict with:
            - cpu_percent: Overall CPU usage (0-100 per core, can exceed 100)
            - cpu_percent_per_core: Per-core average (0-100)
            - num_threads: Number of threads
            - num_cores: Number of CPU cores
            - elapsed_time: Seconds since start()
        """
        elapsed = time.time() - self.start_time if self.start_time else 0

        # Get CPU usage over interval (includes all threads)
        # interval=0.1 samples over 100ms for accurate measurement
        cpu_percent = self.process.cpu_percent(interval=0.1)

        # Get thread count
        num_threads = self.process.num_threads()

        # Calculate per-core average
        num_cores = psutil.cpu_count()
        cpu_percent_per_core = cpu_percent / num_cores if num_cores else cpu_percent

        return {
            "cpu_percent": cpu_percent,
            "cpu_percent_per_core": cpu_percent_per_core,
            "num_threads": num_threads,
            "num_cores": num_cores,
            "elapsed_time": elapsed,
        }

    def get_summary(self) -> str:
        """Get human-readable summary of CPU usage.

        Returns:
            Formatted string with CPU stats
        """
        stats = self.get_stats()
        return (
            f"CPU: {stats['cpu_percent']:.1f}% total "
            f"({stats['cpu_percent_per_core']:.1f}% per core avg) | "
            f"Threads: {stats['num_threads']} | "
            f"Cores: {stats['num_cores']}"
        )


def create_monitor() -> Optional[CPUMonitor]:
    """Create a CPU monitor.

    Returns:
        CPUMonitor instance, or None if construction fails.
    """
    try:
        return CPUMonitor()
    except Exception as e:
        logger.warning(f"Failed to create CPU monitor: {e}")
        return None

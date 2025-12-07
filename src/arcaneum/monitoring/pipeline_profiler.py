"""Pipeline profiling for indexing performance analysis (RDR-013).

Provides per-stage timing breakdown to identify bottlenecks in the
file processing → embedding → upload pipeline.

Usage:
    profiler = PipelineProfiler()

    with profiler.stage("file_processing", file_count=100):
        # process files...

    with profiler.stage("embedding", chunk_count=5000):
        # generate embeddings...

    print(profiler.report())
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from contextlib import contextmanager


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    items_processed: int = 0

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time

    @property
    def throughput(self) -> float:
        """Items per second."""
        if self.duration > 0:
            return self.items_processed / self.duration
        return 0.0


class PipelineProfiler:
    """Track timing and throughput across pipeline stages.

    Thread-safe profiler that can be used with parallel processing.
    Each stage is timed independently and results are aggregated.

    Example:
        >>> profiler = PipelineProfiler()
        >>> with profiler.stage("embedding", chunk_count=1000):
        ...     embeddings = model.encode(texts)
        >>> print(profiler.report())
        Pipeline Performance Report
        ========================================
        embedding: 5.23s (100.0%) - 191.2 items/s
        ----------------------------------------
        Total: 5.23s
    """

    def __init__(self):
        """Initialize the profiler."""
        self.stages: Dict[str, StageMetrics] = {}
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def start(self):
        """Mark the start of the pipeline run."""
        self._start_time = time.perf_counter()

    def stop(self):
        """Mark the end of the pipeline run."""
        self._end_time = time.perf_counter()

    @property
    def total_duration(self) -> float:
        """Total pipeline duration in seconds.

        Returns the sum of all stage durations for accurate percentage calculations.
        Use elapsed_time for actual wall-clock time.
        """
        return sum(s.duration for s in self.stages.values())

    @property
    def elapsed_time(self) -> float:
        """Actual wall-clock time from start() to stop()."""
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.perf_counter()
        return end - self._start_time

    @contextmanager
    def stage(self, name: str, item_count: int = 0):
        """Context manager for timing a pipeline stage.

        Args:
            name: Stage identifier (e.g., "file_processing", "embedding", "upload")
            item_count: Number of items processed in this stage (for throughput calculation)

        Yields:
            StageMetrics object that can be updated during processing

        Example:
            >>> with profiler.stage("embedding", chunk_count=1000):
            ...     # perform embedding
        """
        metrics = StageMetrics(name=name, start_time=time.perf_counter())
        try:
            yield metrics
        finally:
            metrics.end_time = time.perf_counter()
            if item_count > 0:
                metrics.items_processed = item_count
            with self._lock:
                self.stages[name] = metrics

    def record_stage(self, name: str, duration: float, item_count: int = 0):
        """Record a stage directly without context manager.

        Useful when timing is done externally (e.g., from timing collectors).
        Durations are accumulated across multiple calls for the same stage name.

        Args:
            name: Stage identifier
            duration: Duration in seconds
            item_count: Number of items processed
        """
        with self._lock:
            if name in self.stages:
                # Accumulate durations for same stage across multiple projects
                existing = self.stages[name]
                metrics = StageMetrics(
                    name=name,
                    start_time=0.0,
                    end_time=existing.duration + duration,
                    items_processed=existing.items_processed + item_count
                )
            else:
                metrics = StageMetrics(
                    name=name,
                    start_time=0.0,
                    end_time=duration,
                    items_processed=item_count
                )
            self.stages[name] = metrics

    def report(self, include_header: bool = True) -> str:
        """Generate profiling report.

        Args:
            include_header: Whether to include the header lines

        Returns:
            Formatted report string with stage breakdown
        """
        lines = []

        if include_header:
            lines.append("Pipeline Performance Report")
            lines.append("=" * 50)

        total = self.total_duration

        # Sort stages by start time if available, otherwise by name
        sorted_stages = sorted(
            self.stages.values(),
            key=lambda s: (s.start_time, s.name)
        )

        for metrics in sorted_stages:
            pct = (metrics.duration / total * 100) if total > 0 else 0
            if metrics.items_processed > 0:
                lines.append(
                    f"  {metrics.name}: {metrics.duration:.2f}s ({pct:.1f}%) "
                    f"- {metrics.throughput:.1f} items/s"
                )
            else:
                lines.append(f"  {metrics.name}: {metrics.duration:.2f}s ({pct:.1f}%)")

        lines.append("-" * 50)
        lines.append(f"  Total: {total:.2f}s")

        return "\n".join(lines)

    def get_stage_summary(self, name: str) -> Optional[str]:
        """Get summary for a specific stage.

        Args:
            name: Stage identifier

        Returns:
            Formatted summary string or None if stage not found
        """
        if name not in self.stages:
            return None

        metrics = self.stages[name]
        total = self.total_duration
        pct = (metrics.duration / total * 100) if total > 0 else 0

        if metrics.items_processed > 0:
            return (
                f"{metrics.name}: {metrics.duration:.2f}s ({pct:.1f}%) "
                f"- {metrics.throughput:.1f} items/s"
            )
        return f"{metrics.name}: {metrics.duration:.2f}s ({pct:.1f}%)"

    def get_compact_summary(self) -> str:
        """Get a single-line compact summary.

        Returns:
            Compact summary string for log output
        """
        parts = []
        total = self.total_duration

        for name in ["file_processing", "embedding", "upload"]:
            if name in self.stages:
                metrics = self.stages[name]
                pct = (metrics.duration / total * 100) if total > 0 else 0
                parts.append(f"{name[:5]}:{metrics.duration:.1f}s({pct:.0f}%)")

        if not parts:
            # Fallback to any stages we have
            for name, metrics in self.stages.items():
                pct = (metrics.duration / total * 100) if total > 0 else 0
                parts.append(f"{name[:8]}:{metrics.duration:.1f}s({pct:.0f}%)")

        return f"Profile: {' | '.join(parts)} | total:{total:.1f}s"

    def reset(self):
        """Clear all recorded stages."""
        with self._lock:
            self.stages.clear()
            self._start_time = None
            self._end_time = None


def create_profiler() -> PipelineProfiler:
    """Create a new pipeline profiler.

    Returns:
        Initialized PipelineProfiler instance
    """
    return PipelineProfiler()

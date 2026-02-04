"""Shared multiprocessing utilities for consistent fork/spawn and signal handling.

This module provides consistent multiprocessing configuration across the codebase:
- Fork context on Unix for better performance and signal handling
- Worker initializer that resets SIGINT to allow proper Ctrl-C termination
- Proper cleanup patterns for ProcessPoolExecutor
"""

import multiprocessing as mp
import os
import signal
from concurrent.futures import ProcessPoolExecutor
from typing import Optional


def get_mp_context():
    """Get the appropriate multiprocessing context for the platform.

    Uses 'fork' on Unix systems for:
    - Better performance (no pickling overhead)
    - Proper signal handling (forked processes inherit handlers)

    Falls back to default (spawn) on Windows or if fork unavailable.

    Returns:
        Multiprocessing context object
    """
    if hasattr(os, 'fork'):
        try:
            return mp.get_context('fork')
        except ValueError:
            pass
    return mp.get_context()


def worker_init():
    """Initialize worker process with proper signal handling.

    By default, Python's multiprocessing workers ignore SIGINT, causing them
    to continue running when the user presses Ctrl-C. This initializer resets
    SIGINT to the default behavior, allowing workers to be terminated cleanly.

    Use this as the 'initializer' argument to ProcessPoolExecutor or Pool.
    """
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def create_process_pool(
    max_workers: Optional[int] = None,
    initializer=None,
    initargs=()
) -> ProcessPoolExecutor:
    """Create a ProcessPoolExecutor with consistent configuration.

    Args:
        max_workers: Maximum number of worker processes
        initializer: Callable to run in each worker at startup (default: worker_init)
        initargs: Arguments to pass to initializer

    Returns:
        Configured ProcessPoolExecutor

    Note:
        The returned executor should be used with try/finally for proper cleanup:

        executor = create_process_pool(max_workers=4)
        try:
            # submit work...
        except KeyboardInterrupt:
            print("Interrupted")
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    """
    ctx = get_mp_context()

    # Use worker_init by default unless caller provides their own
    if initializer is None:
        initializer = worker_init

    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=initializer,
        initargs=initargs
    )

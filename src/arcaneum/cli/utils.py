"""CLI utility functions."""

import os
import sys
import logging

logger = logging.getLogger(__name__)


def set_process_priority(priority: str) -> None:
    """Set process scheduling priority.

    Args:
        priority: Priority level - 'low', 'normal', or 'high'
            - low: nice +10 (background processing, minimal impact on UI)
            - normal: no change (default)
            - high: nice -10 (requires root/admin privileges)

    Note:
        On Windows, this uses SetPriorityClass instead of nice.
        'high' priority may require administrator privileges.
    """
    if priority == "normal":
        return  # No change needed

    # Unix/Linux/macOS: use os.nice()
    if hasattr(os, 'nice'):
        try:
            if priority == "low":
                os.nice(10)  # Lower priority (background job)
                logger.info("Set process priority to LOW (nice +10)")
            elif priority == "high":
                os.nice(-10)  # Higher priority (may require privileges)
                logger.info("Set process priority to HIGH (nice -10)")
            else:
                logger.warning(f"Unknown priority '{priority}', using normal")
        except PermissionError:
            logger.warning(
                f"Cannot set priority to '{priority}' - requires elevated privileges. "
                "Run with sudo or as administrator."
            )
        except Exception as e:
            logger.warning(f"Failed to set process priority: {e}")

    # Windows: use SetPriorityClass via psutil
    elif sys.platform == 'win32':
        try:
            import psutil
            process = psutil.Process(os.getpid())

            if priority == "low":
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                logger.info("Set process priority to LOW (BELOW_NORMAL)")
            elif priority == "high":
                process.nice(psutil.HIGH_PRIORITY_CLASS)
                logger.info("Set process priority to HIGH")
            else:
                logger.warning(f"Unknown priority '{priority}', using normal")
        except ImportError:
            logger.warning(
                "psutil not available on Windows - cannot set process priority. "
                "Install with: pip install psutil"
            )
        except Exception as e:
            logger.warning(f"Failed to set process priority on Windows: {e}")

    else:
        logger.warning(f"Process priority not supported on platform: {sys.platform}")

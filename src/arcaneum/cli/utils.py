"""CLI utility functions."""

import os
import sys
import logging
from typing import Optional
from pathlib import Path
from qdrant_client import QdrantClient

from ..config import load_config, ArcaneumConfig, QdrantConfig

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


def create_qdrant_client(
    url: Optional[str] = None,
    timeout: Optional[int] = None,
    for_search: bool = False,
    config_path: Optional[Path] = None
) -> QdrantClient:
    """Create QdrantClient with proper timeout configuration.

    This helper ensures consistent client creation across all CLI commands,
    with proper timeout handling and environment variable overrides.

    Args:
        url: Qdrant server URL (defaults to config or localhost:6333)
        timeout: Timeout in seconds (overrides config)
        for_search: If True, use search_timeout from config (default: False)
        config_path: Path to config file (defaults to ~/.arcaneum/config.yaml)

    Returns:
        Configured QdrantClient instance

    Environment Variables:
        ARC_QDRANT_URL: Override Qdrant URL
        ARC_QDRANT_TIMEOUT: Override timeout value
    """
    # Try to load config, but don't fail if it doesn't exist
    qdrant_config = None
    if config_path is None:
        config_path = Path.home() / ".arcaneum" / "config.yaml"

    if config_path.exists():
        try:
            config = load_config(config_path)
            qdrant_config = config.qdrant
        except Exception as e:
            logger.debug(f"Could not load config: {e}")

    # Determine URL (priority: param > env > config > default)
    final_url = url
    if not final_url:
        final_url = os.environ.get("ARC_QDRANT_URL")
    if not final_url and qdrant_config:
        final_url = qdrant_config.url
    if not final_url:
        final_url = "http://localhost:6333"

    # Determine timeout (priority: param > env > config > default)
    final_timeout = timeout
    if final_timeout is None:
        env_timeout = os.environ.get("ARC_QDRANT_TIMEOUT")
        if env_timeout:
            try:
                final_timeout = int(env_timeout)
            except ValueError:
                logger.warning(f"Invalid ARC_QDRANT_TIMEOUT value: {env_timeout}")

    if final_timeout is None and qdrant_config:
        # Use search_timeout for search operations, regular timeout for indexing
        final_timeout = qdrant_config.search_timeout if for_search else qdrant_config.timeout

    if final_timeout is None:
        final_timeout = 60 if for_search else 30

    logger.debug(f"Creating QdrantClient: url={final_url}, timeout={final_timeout}s")

    return QdrantClient(url=final_url, timeout=final_timeout)

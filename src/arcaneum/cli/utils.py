"""CLI utility functions."""

import os
import sys
import logging
from typing import Optional, List, Set
from pathlib import Path
from qdrant_client import QdrantClient

from ..config import load_config, ArcaneumConfig, QdrantConfig

logger = logging.getLogger(__name__)


def read_file_list(
    from_file: str,
    allowed_extensions: Optional[Set[str]] = None
) -> List[Path]:
    """Read file paths from a list file or stdin.

    Args:
        from_file: Path to file containing list of files (one per line),
                   or "-" to read from stdin
        allowed_extensions: Optional set of allowed file extensions (e.g., {'.pdf', '.md'})
                          If provided, files with other extensions will be skipped with warning

    Returns:
        List of Path objects (absolute paths) for files that exist and match criteria

    Notes:
        - Empty lines and lines starting with '#' are skipped
        - Relative paths are resolved relative to current working directory
        - Non-existent files are skipped with warning
        - Files with wrong extensions (if allowed_extensions provided) are skipped with warning
    """
    paths = []

    # Read lines from stdin or file
    if from_file == '-':
        logger.debug("Reading file list from stdin")
        lines = sys.stdin.readlines()
    else:
        from_file_path = Path(from_file)
        if not from_file_path.exists():
            logger.error(f"File list not found: {from_file}")
            return []
        logger.debug(f"Reading file list from {from_file}")
        with open(from_file_path, 'r') as f:
            lines = f.readlines()

    # Process each line
    for line_num, line in enumerate(lines, 1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue

        # Convert to Path and make absolute
        path = Path(line)
        if not path.is_absolute():
            path = path.absolute()

        # Check if file exists
        if not path.exists():
            logger.warning(f"Line {line_num}: File not found, skipping: {line}")
            continue

        # Check if it's a file (not directory)
        if not path.is_file():
            logger.warning(f"Line {line_num}: Not a file, skipping: {line}")
            continue

        # Check extension if filtering is requested
        if allowed_extensions is not None:
            if path.suffix.lower() not in allowed_extensions:
                logger.warning(
                    f"Line {line_num}: Wrong file type (expected {', '.join(allowed_extensions)}), "
                    f"skipping: {line}"
                )
                continue

        paths.append(path)

    logger.info(f"Read {len(paths)} valid file paths from list")
    return paths


def set_process_priority(priority: str, disable_worker_nice: bool = False) -> None:
    """Set process scheduling priority.

    Args:
        priority: Priority level - 'low', 'normal', or 'high'
            - low: nice +10 (background processing, minimal impact on UI)
            - normal: no change (default)
            - high: nice -10 (requires root/admin privileges)
        disable_worker_nice: If True, disable nice settings for worker processes (arcaneum-mql4)

    Note:
        On Windows, this uses SetPriorityClass instead of nice.
        'high' priority may require administrator privileges.
    """
    # Store disable_worker_nice in environment for worker processes to access (arcaneum-mql4)
    if disable_worker_nice:
        os.environ['ARCANEUM_DISABLE_WORKER_NICE'] = '1'

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

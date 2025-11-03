"""Metadata-based sync for incremental indexing (RDR-004)."""

import hashlib
from pathlib import Path
from typing import List, Set, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import logging

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file content in binary mode.

    Uses binary mode to handle both text and binary files (PDFs, images, etc.)
    without encoding issues. For binary files (PDFs, images), this is the
    correct approach. For text files where normalized content matters
    (markdown), use compute_text_file_hash instead.

    Args:
        file_path: Path to file

    Returns:
        Full SHA256 hash (64 characters)
    """
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def compute_text_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of text file content with normalization.

    Reads file in text mode with UTF-8 encoding (latin-1 fallback), which
    normalizes line endings (CRLF -> LF). This ensures hash matches the
    actual content being indexed after parsing/normalization.

    Use this for text files (markdown, source code) where:
    - Line ending normalization is desired
    - Content is parsed/processed before indexing
    - Hash should match processed content

    Args:
        file_path: Path to text file

    Returns:
        Full SHA256 hash (64 characters)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decode failed for {file_path}, trying latin-1")
        content = file_path.read_text(encoding='latin-1')

    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def _compute_hashes_parallel(file_list: List[Path],
                             hash_fn: Callable,
                             num_workers: int = None) -> dict:
    """Compute file hashes in parallel using threading.

    Uses ThreadPoolExecutor for concurrent hash computation. Threading works
    well here because hashlib releases the GIL during hash computation, and
    file I/O also releases the GIL. This provides good parallelism without
    the overhead and platform issues of multiprocessing.

    Args:
        file_list: List of file paths to hash
        hash_fn: Hash function to use (compute_file_hash or compute_text_file_hash)
        num_workers: Number of worker threads (defaults to 2x CPU count for I/O-bound work)

    Returns:
        Dict mapping file_path to file_hash (absolute path as string)
    """
    import os

    if num_workers is None:
        # For large files (2-10MB), hashing is I/O-bound, not CPU-bound.
        # Need more threads to saturate disk I/O while other threads wait.
        # Use 8x CPU count for good I/O parallelism on modern SSDs.
        num_workers = (os.cpu_count() or 4) * 8

    total_files = len(file_list)
    file_hashes = {}

    # Show progress for large file sets
    show_progress = total_files > 100

    def hash_with_path(file_path: Path):
        """Hash a file and return (path, hash) tuple."""
        try:
            file_hash = hash_fn(file_path)
            return (str(file_path.absolute()), file_hash)
        except Exception as e:
            logger.warning(f"Failed to hash {file_path}: {e}")
            return (str(file_path.absolute()), None)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        if show_progress:
            # Submit all tasks and track completion
            futures = {executor.submit(hash_with_path, f): f for f in file_list}
            completed = 0

            for future in as_completed(futures):
                file_path_str, file_hash = future.result()
                if file_hash is not None:
                    file_hashes[file_path_str] = file_hash

                completed += 1
                if completed % 100 == 0:
                    print(f"\r  Computing hashes: {completed}/{total_files} files...", end="", flush=True)

            print(f"\r  Computing hashes: {total_files}/{total_files} files... done")
        else:
            # No progress for small file sets - just map and collect
            results = executor.map(hash_with_path, file_list)
            for file_path_str, file_hash in results:
                if file_hash is not None:
                    file_hashes[file_path_str] = file_hash

    return file_hashes


class MetadataBasedSync:
    """Check indexing status using Qdrant metadata queries.

    Follows chroma-embedded pattern: query file_path and file_hash
    from chunk metadata to determine if file is already indexed.
    """

    def __init__(self, qdrant_client: QdrantClient):
        """Initialize sync manager.

        Args:
            qdrant_client: Qdrant client instance
        """
        self.qdrant = qdrant_client

    def is_file_indexed(self, collection_name: str, file_path: Path,
                       file_hash: str) -> bool:
        """Check if file with current content hash is already indexed.

        Returns True if ANY chunks with this file_path AND file_hash exist.

        Args:
            collection_name: Qdrant collection name
            file_path: Path to file
            file_hash: Content hash of file

        Returns:
            True if file is already indexed with same content
        """
        try:
            points, _ = self.qdrant.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=str(file_path))
                        ),
                        FieldCondition(
                            key="file_hash",
                            match=MatchValue(value=file_hash)
                        )
                    ]
                ),
                limit=1,  # Just need to know if exists
                with_payload=False,  # Don't need payload, faster
                with_vectors=False   # Don't need vectors, faster
            )

            return len(points) > 0

        except Exception as e:
            logger.warning(f"Error querying collection: {e}")
            return False

    def get_indexed_file_paths(self, collection_name: str) -> Set[tuple]:
        """Get all (file_path, file_hash) pairs from collection.

        Returns set of tuples for fast lookup.

        Args:
            collection_name: Qdrant collection name

        Returns:
            Set of (file_path, file_hash) tuples
        """
        indexed = set()
        offset = None

        try:
            while True:
                points, offset = self.qdrant.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=["file_path", "file_hash"],
                    with_vectors=False
                )

                if not points:
                    break

                for point in points:
                    if point.payload:
                        path = point.payload.get("file_path")
                        hash_val = point.payload.get("file_hash")
                        if path and hash_val:
                            indexed.add((path, hash_val))

                if offset is None:
                    break

            return indexed

        except Exception as e:
            logger.warning(f"Error scrolling collection: {e}")
            return set()

    def get_unindexed_files(self, collection_name: str,
                            file_list: List[Path],
                            hash_fn=None) -> List[Path]:
        """Filter file list to only unindexed or modified files.

        Uses batch query for efficiency and parallel hash computation.
        Hash computation runs at low priority across all CPU cores.

        Args:
            collection_name: Qdrant collection name
            file_list: List of file paths to check
            hash_fn: Optional hash function(Path) -> str. Defaults to compute_file_hash.

        Returns:
            List of files that need indexing
        """
        if hash_fn is None:
            hash_fn = compute_file_hash

        try:
            # Get all indexed (path, hash) pairs
            indexed = self.get_indexed_file_paths(collection_name)

            # Compute hashes in parallel at low priority
            file_hashes = _compute_hashes_parallel(file_list, hash_fn)

            # Filter to files not in indexed set
            unindexed = []
            for file_path in file_list:
                file_path_str = str(file_path.absolute())
                file_hash = file_hashes.get(file_path_str)

                if file_hash is None:
                    # Hash computation failed, skip this file
                    continue

                if (file_path_str, file_hash) not in indexed:
                    unindexed.append(file_path)

            logger.info(f"Found {len(unindexed)}/{len(file_list)} "
                       f"files to index ({len(file_list) - len(unindexed)} "
                       f"already indexed)")
            return unindexed

        except Exception as e:
            logger.warning(f"Error querying collection: {e}, "
                          "processing all files")
            return file_list

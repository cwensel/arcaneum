"""Metadata-based sync for incremental indexing (RDR-004)."""

import hashlib
import os
import multiprocessing as mp
from pathlib import Path
from typing import List, Set, Callable, Tuple, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
import logging
import xxhash

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute xxHash (xxh3_128) of file content in binary mode with streaming.

    Uses xxHash for fast non-cryptographic hashing (20-50 GB/s vs SHA256's 500 MB/s).
    Streams file in 64KB chunks to avoid memory issues with large files.

    Uses binary mode to handle both text and binary files (PDFs, images, etc.)
    without encoding issues. For binary files (PDFs, images), this is the
    correct approach. For text files where normalized content matters
    (markdown), use compute_text_file_hash instead.

    Args:
        file_path: Path to file

    Returns:
        xxh3_128 hash (32 hex characters)
    """
    hasher = xxhash.xxh3_128()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):  # 64KB chunks
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_text_file_hash(file_path: Path) -> str:
    """Compute xxHash (xxh3_128) of text file content with normalization.

    Uses xxHash for fast non-cryptographic hashing (20-50 GB/s vs SHA256's 500 MB/s).

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
        xxh3_128 hash (32 hex characters)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decode failed for {file_path}, trying latin-1")
        content = file_path.read_text(encoding='latin-1')

    return xxhash.xxh3_128(content.encode('utf-8')).hexdigest()


def _compute_hash_worker(args: Tuple[Path, Callable]) -> Tuple[Path, str]:
    """Worker function for parallel hash computation.

    Sets low process priority and computes hash for a single file.

    Args:
        args: Tuple of (file_path, hash_function)

    Returns:
        Tuple of (file_path, file_hash)
    """
    file_path, hash_fn = args

    # Set low priority (nice level 19 on Unix, IDLE on Windows)
    try:
        if hasattr(os, 'nice'):
            os.nice(19)
    except Exception:
        pass  # Ignore if we can't set priority

    try:
        file_hash = hash_fn(file_path)
        return (file_path, file_hash)
    except Exception as e:
        logger.warning(f"Failed to hash {file_path}: {e}")
        return (file_path, None)


def _compute_hashes_parallel(file_list: List[Path],
                             hash_fn: Callable,
                             num_workers: int = None) -> dict:
    """Compute file hashes in parallel using all CPU cores at low priority.

    Args:
        file_list: List of file paths to hash
        hash_fn: Hash function to use (compute_file_hash or compute_text_file_hash)
        num_workers: Number of worker processes (defaults to CPU count)

    Returns:
        Dict mapping file_path to file_hash (absolute path as string)
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    total_files = len(file_list)
    file_hashes = {}

    # Show progress for large file sets
    show_progress = total_files > 100

    # Prepare work items
    work_items = [(f, hash_fn) for f in file_list]

    # Use multiprocessing pool with chunksize for better progress feedback
    chunksize = max(1, total_files // (num_workers * 10))

    # Use fork context on Unix for better performance and compatibility
    # (avoids spawn issues with pickling on macOS)
    ctx = mp.get_context('fork') if hasattr(os, 'fork') else mp.get_context()

    with ctx.Pool(processes=num_workers) as pool:
        if show_progress:
            # Use imap for incremental results with progress tracking
            results = pool.imap(_compute_hash_worker, work_items, chunksize=chunksize)

            for idx, (file_path, file_hash) in enumerate(results, 1):
                if file_hash is not None:
                    file_hashes[str(file_path.absolute())] = file_hash

                if idx % 100 == 0:
                    print(f"\r  Computing hashes: {idx}/{total_files} files...", end="", flush=True)

            print(f"\r  Computing hashes: {total_files}/{total_files} files... done")
        else:
            # No progress for small file sets
            results = pool.map(_compute_hash_worker, work_items, chunksize=chunksize)
            for file_path, file_hash in results:
                if file_hash is not None:
                    file_hashes[str(file_path.absolute())] = file_hash

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

    def get_indexed_file_paths(self, collection_name: str) -> Dict[str, List[str]]:
        """Get all indexed files organized by hash.

        Returns dict mapping file_hash to list of file_paths for rename detection.

        Args:
            collection_name: Qdrant collection name

        Returns:
            Dict mapping file_hash to list of file_paths
        """
        indexed_by_hash = {}
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
                            if hash_val not in indexed_by_hash:
                                indexed_by_hash[hash_val] = []
                            if path not in indexed_by_hash[hash_val]:
                                indexed_by_hash[hash_val].append(path)

                if offset is None:
                    break

            return indexed_by_hash

        except Exception as e:
            logger.warning(f"Error scrolling collection: {e}")
            return {}

    def get_unindexed_files(self, collection_name: str,
                            file_list: List[Path],
                            hash_fn=None) -> Tuple[List[Path], List[Tuple[str, str]], List[Path]]:
        """Filter file list to unindexed, renamed, or already-indexed files.

        Uses hash-first lookup to detect file renames without reindexing.
        When a file's hash exists but path differs, it's a rename candidate.

        Args:
            collection_name: Qdrant collection name
            file_list: List of file paths to check
            hash_fn: Optional hash function(Path) -> str. Defaults to compute_file_hash.

        Returns:
            Tuple of:
            - unindexed: Files that need full indexing
            - renames: List of (old_path, new_path) tuples for rename candidates
            - already_indexed: Files already indexed with matching path and hash
        """
        if hash_fn is None:
            hash_fn = compute_file_hash

        try:
            # Get indexed files organized by hash
            indexed_by_hash = self.get_indexed_file_paths(collection_name)

            # Compute hashes in parallel at low priority
            file_hashes = _compute_hashes_parallel(file_list, hash_fn)

            # Categorize files using hash-first lookup
            unindexed = []
            renames = []
            already_indexed = []

            for file_path in file_list:
                file_path_str = str(file_path.absolute())
                file_hash = file_hashes.get(file_path_str)

                if file_hash is None:
                    # Hash computation failed, skip this file
                    continue

                if file_hash not in indexed_by_hash:
                    # Hash not found - file needs full indexing
                    unindexed.append(file_path)
                else:
                    # Hash exists - check if path matches
                    stored_paths = indexed_by_hash[file_hash]

                    if file_path_str in stored_paths:
                        # Exact match - already indexed
                        already_indexed.append(file_path)
                    else:
                        # Hash exists but path differs - rename detected
                        if len(stored_paths) == 1:
                            # Single path - safe to update
                            renames.append((stored_paths[0], file_path_str))
                        else:
                            # Multiple paths with same hash - ambiguous, treat as new file
                            unindexed.append(file_path)

            logger.info(f"Found {len(unindexed)} new, {len(renames)} renamed, "
                       f"{len(already_indexed)} already indexed "
                       f"(total {len(file_list)} files)")
            return (unindexed, renames, already_indexed)

        except Exception as e:
            logger.warning(f"Error querying collection: {e}, "
                          "processing all files")
            return (file_list, [], [])

    def handle_renames(self, collection_name: str,
                      renames: List[Tuple[str, str]]) -> int:
        """Update file_path metadata for renamed files.

        Updates all chunks with old_path to use new_path. This avoids
        reindexing the entire file when only the path changed.

        Args:
            collection_name: Qdrant collection name
            renames: List of (old_path, new_path) tuples

        Returns:
            Number of files successfully renamed
        """
        if not renames:
            return 0

        renamed_count = 0

        for old_path, new_path in renames:
            try:
                # Update all chunks with old_path to use new_path
                self.qdrant.set_payload(
                    collection_name=collection_name,
                    payload={"file_path": new_path},
                    points=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="file_path",
                                    match=MatchValue(value=old_path)
                                )
                            ]
                        )
                    )
                )
                renamed_count += 1
                logger.info(f"Renamed: {old_path} -> {new_path}")

            except Exception as e:
                logger.warning(f"Failed to rename {old_path} -> {new_path}: {e}")

        if renamed_count > 0:
            logger.info(f"Successfully renamed {renamed_count}/{len(renames)} files")

        return renamed_count

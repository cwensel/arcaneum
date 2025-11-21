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


def compute_quick_hash(file_path: Path) -> str:
    """Compute metadata-only hash using mtime and size (no file I/O).

    Pure metadata-based fast change detection for Pass 1 of incremental sync:
    - Combines file mtime + size as change indicator
    - Zero file I/O overhead (~1-2µs per file)
    - Detects mtime-based changes immediately
    - Content verification deferred to Pass 2 (full hash) if needed

    Design rationale:
    - Pass 1 goal: Quick gate to skip obviously unchanged files
    - If mtime+size same → file unchanged with high confidence
    - If mtime changed → content may have changed → verify in Pass 2
    - This approach: minimum I/O overhead, defers expensive checks
    - Alternative (old): Read first+last chunks, catches middle-only edits (rare)

    Uses xxh64 for maximum speed (30-40 GB/s on modern CPUs).

    Args:
        file_path: Path to file

    Returns:
        Metadata hash (16 hex characters, xxh64 of "mtime:size")
    """
    stat = file_path.stat()
    # Only hash metadata: mtime (millisecond precision) + size
    mtime = int(stat.st_mtime * 1000)  # Milliseconds for precision
    size = stat.st_size

    hasher = xxhash.xxh64()
    hasher.update(f"{mtime}:{size}".encode())

    return hasher.hexdigest()


def compute_file_hash(file_path: Path) -> str:
    """Compute xxHash (xxh64) of file content in binary mode with streaming.

    Uses xxHash for extremely fast non-cryptographic hashing:
    - xxh64: ~30-40 GB/s on modern CPUs
    - Much faster than xxh3_128 for full file hashing
    - Streams file in 256KB chunks for optimal throughput

    Uses binary mode to handle both text and binary files (PDFs, images, etc.)
    without encoding issues. For binary files (PDFs, images), this is the
    correct approach. For text files where normalized content matters
    (markdown), use compute_text_file_hash instead.

    Args:
        file_path: Path to file

    Returns:
        xxh64 hash (16 hex characters, 64-bit digest)
    """
    hasher = xxhash.xxh64()
    with open(file_path, 'rb') as f:
        while chunk := f.read(262144):  # 256KB chunks for better throughput
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

    # Set low priority UNLESS disabled by --not-nice flag (arcaneum-mql4)
    if os.environ.get('ARCANEUM_DISABLE_WORKER_NICE') != '1':
        try:
            if hasattr(os, 'nice'):
                os.nice(19)  # Lowest priority for hash workers
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
                             num_workers: int = None,
                             show_progress: bool = True) -> dict:
    """Compute file hashes in parallel using all CPU cores at low priority.

    Args:
        file_list: List of file paths to hash
        hash_fn: Hash function to use (compute_file_hash or compute_text_file_hash)
        num_workers: Number of worker processes (defaults to CPU count)
        show_progress: If True, show progress for large file sets (default: True)

    Returns:
        Dict mapping file_path to file_hash (absolute path as string)
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    total_files = len(file_list)
    file_hashes = {}

    # Show progress for large file sets (unless explicitly disabled)
    show_progress = show_progress and total_files > 100

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

    def is_file_indexed_by_quick_hash(self, collection_name: str, file_path: Path,
                                      quick_hash: str) -> bool:
        """Check if file with metadata hash (mtime+size) exists in collection.

        Fast gate check using only mtime+size hash. Used in Pass 1 of incremental sync.

        Performance: ~0.003ms per file (metadata-only, no file I/O)
        - Queries Qdrant index for exact match on file_path + quick_hash
        - If found → file unchanged at same location, skip indexing
        - If miss → proceed to Pass 2 (full content hash) for deep verification

        Args:
            collection_name: Qdrant collection name
            file_path: Path to file
            quick_hash: Metadata hash (mtime+size, from compute_quick_hash)

        Returns:
            True if file_path + quick_hash exists in collection
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
                            key="quick_hash",  # New metadata field
                            match=MatchValue(value=quick_hash)
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False
            )

            return len(points) > 0

        except Exception as e:
            logger.warning(f"Error querying collection (quick_hash): {e}")
            return False

    def find_file_by_content_hash(self, collection_name: str, file_hash: str) -> List[str]:
        """Find all file_paths in collection with given content hash.

        Used in Pass 3 rename detection to find if file moved.

        Args:
            collection_name: Qdrant collection name
            file_hash: Content hash to search for

        Returns:
            List of file_paths that have this content hash
        """
        try:
            paths = []
            offset = None

            while True:
                points, offset = self.qdrant.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="file_hash",
                                match=MatchValue(value=file_hash)
                            )
                        ]
                    ),
                    limit=100,
                    offset=offset,
                    with_payload=["file_path"],
                    with_vectors=False
                )

                if not points:
                    break

                for point in points:
                    if point.payload and "file_path" in point.payload:
                        path = point.payload["file_path"]
                        if path not in paths:
                            paths.append(path)

                if offset is None:
                    break

            return paths

        except Exception as e:
            logger.warning(f"Error finding file by content hash: {e}")
            return []

    def check_if_synced_two_pass(self, collection_name: str, file_path: Path) -> tuple:
        """Three-pass sync check: metadata gate + content verification + rename detection.

        **Pass 1 (Ultra-fast ~0.003ms)**: Check if file_path + quick_hash exists
        - Computes mtime+size hash only (metadata, no file I/O)
        - Queries Qdrant index for match
        - If found → file unchanged at same location → skip indexing
        - Zero file I/O overhead for unchanged files

        **Pass 2 (Slower ~50ms)**: If Pass 1 misses, compute and check content hash
        - Only computed for files with changed metadata (rare)
        - Confirms if file content truly changed
        - Handles edge case: mtime changed but content same (e.g., touch command)
        - ~50ms per file (full xxh64 hash of content)

        **Pass 3 (Rename detection ~5ms)**: If Pass 2 misses, check for renames
        - File may have been moved but content unchanged
        - Searches for file_hash elsewhere in collection
        - Returns old_path if found (for metadata-only update)

        Args:
            collection_name: Qdrant collection name
            file_path: Path to file

        Returns:
            Tuple of (is_synced, old_path_if_renamed)
            - (True, None): File unchanged at same location → skip indexing
            - (True, old_path): File moved but content same → update metadata only
            - (False, None): File new or changed → reindex
        """
        file_path_str = str(file_path)

        # Pass 1: Ultra-fast metadata check (~0.003ms, no file I/O)
        quick_hash = compute_quick_hash(file_path)
        if self.is_file_indexed_by_quick_hash(collection_name, file_path, quick_hash):
            logger.debug(f"Pass 1 HIT: {file_path} unchanged (mtime+size match)")
            return (True, None)

        # Pass 1 miss - file either new, metadata changed, or moved
        # Pass 2: Content verification (~50ms per file, only if Pass 1 misses)
        content_hash = compute_file_hash(file_path)
        if self.is_file_indexed(collection_name, file_path, content_hash):
            logger.debug(f"Pass 2 HIT: {file_path} unchanged (content match, metadata changed)")
            return (True, None)

        # Pass 2 miss - check if file was moved/renamed (Pass 3)
        # Find all paths with this content hash
        old_paths = self.find_file_by_content_hash(collection_name, content_hash)

        if old_paths:
            # Content hash found elsewhere - file was moved/renamed
            if len(old_paths) == 1:
                old_path = old_paths[0]
                logger.debug(f"Pass 3 HIT: {file_path} is rename from {old_path} (content unchanged)")
                return (True, old_path)
            else:
                # Multiple matches - ambiguous, treat as new file to be safe
                logger.debug(f"Pass 3 AMBIGUOUS: {len(old_paths)} files with same content, treating as new")
                return (False, None)

        # All passes miss - file is new or changed
        logger.debug(f"All passes miss: {file_path} is new or changed, needs indexing")
        return (False, None)

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

    def _get_indexed_quick_hashes(self, collection_name: str) -> set:
        """Get all indexed (file_path, quick_hash) pairs for Pass 1 matching.

        Used during incremental sync to detect unchanged files without reindexing.
        Returns a set of (file_path, quick_hash) tuples for O(1) lookup.

        Args:
            collection_name: Qdrant collection name

        Returns:
            Set of (file_path, quick_hash) tuples
        """
        indexed_pairs = set()
        offset = None

        try:
            while True:
                points, offset = self.qdrant.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=["file_path", "quick_hash"],
                    with_vectors=False
                )

                if not points:
                    break

                for point in points:
                    if point.payload:
                        path = point.payload.get("file_path")
                        quick_hash = point.payload.get("quick_hash")
                        if path and quick_hash:
                            indexed_pairs.add((path, quick_hash))

                if offset is None:
                    break

            return indexed_pairs

        except Exception as e:
            logger.warning(f"Error querying quick_hashes: {e}")
            return set()

    def get_unindexed_files(self, collection_name: str,
                            file_list: List[Path],
                            hash_fn=None) -> Tuple[List[Path], List[Tuple[str, str]], List[Path]]:
        """Filter file list using metadata gate to identify files needing reindexing.

        **Metadata-only gate strategy** (highly optimized incremental sync):

        Pass 1: Ultra-fast metadata check (~0.003ms per file)
        - Compute quick_hash for all files (mtime+size only, no file I/O)
        - Query Qdrant for (file_path, quick_hash) matches
        - Files matching → already indexed (skip reindexing)
        - Files NOT matching → candidates for reindexing

        Full content hashing deferred to indexing pipeline:
        - NOT computed here (expensive, ~50ms per file)
        - Computed during indexing when pre-deletion and new chunks needed
        - Allows natural rename detection (different path + same content hash = rename)

        **Performance**: Unchanged files (99%) skip expensive hashing entirely.
        Only files with metadata changes get full hashing during indexing.

        Args:
            collection_name: Qdrant collection name
            file_list: List of file paths to check
            hash_fn: Deprecated parameter, ignored (kept for backward compatibility)

        Returns:
            Tuple of:
            - unindexed: Files to reindex (metadata changed or new)
            - renames: Empty list (rename detection handled via full hash during indexing)
            - already_indexed: Files with matching (path, quick_hash)
        """
        try:
            # Pass 1: Metadata gate (mtime+size) - ultra fast
            quick_hashes = _compute_hashes_parallel(file_list, compute_quick_hash, show_progress=False)

            # Query Qdrant for (file_path, quick_hash) matches
            indexed_quick_hashes = self._get_indexed_quick_hashes(collection_name)

            # Also get indexed files by content hash (for rename detection)
            indexed_by_hash = self.get_indexed_file_paths(collection_name)

            # Categorize files
            unindexed = []
            renames = []
            already_indexed = []

            for file_path in file_list:
                file_path_str = str(file_path.absolute())
                quick_hash = quick_hashes.get(file_path_str)

                if quick_hash is None:
                    # Hash computation failed, skip this file
                    continue

                # Check if (file_path, quick_hash) exists in collection
                if (file_path_str, quick_hash) in indexed_quick_hashes:
                    # Metadata unchanged → already indexed
                    already_indexed.append(file_path)
                else:
                    # Metadata changed or new → candidate for reindexing
                    # Attempt to detect rename by checking if file content exists elsewhere
                    # This requires computing full hash during indexing, but we can flag it here
                    # For now, treat as unindexed (full hash will be computed during indexing)
                    unindexed.append(file_path)

            logger.info(f"Found {len(unindexed)} candidates for reindexing, "
                       f"{len(already_indexed)} already indexed "
                       f"(total {len(file_list)} files)")
            return (unindexed, renames, already_indexed)

        except Exception as e:
            logger.warning(f"Error querying collection: {e}, "
                          "processing all files")
            return (file_list, [], [])

    def delete_chunks_by_file_hash(self, collection_name: str, file_hash: str) -> int:
        """Delete all chunks with a specific file_hash from collection.

        Used before reindexing a file to remove old/partial chunks.
        Prevents partial data if indexing is interrupted mid-file.

        Args:
            collection_name: Qdrant collection name
            file_hash: Content hash of file to delete chunks for

        Returns:
            Number of points deleted (0 if no chunks found)
        """
        try:
            # Count chunks before deletion
            points_before, _ = self.qdrant.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_hash",
                            match=MatchValue(value=file_hash)
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False
            )

            if not points_before:
                return 0  # No chunks to delete

            # Delete all points with this file_hash
            result = self.qdrant.delete(
                collection_name=collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="file_hash",
                                match=MatchValue(value=file_hash)
                            )
                        ]
                    )
                )
            )

            # UpdateResult only has operation_id and status, count the deleted chunks from before
            deleted_count = len(points_before) if points_before else 0
            logger.debug(f"Deleted {deleted_count} chunks for file_hash {file_hash} from {collection_name}")
            return deleted_count

        except Exception as e:
            logger.warning(f"Error deleting chunks by file_hash {file_hash}: {e}")
            return 0

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

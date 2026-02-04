"""Metadata-based sync for incremental indexing (RDR-004)."""

import hashlib
import os
from pathlib import Path
from typing import List, Set, Callable, Tuple, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, FilterSelector
import logging
import xxhash

from .multiprocessing import get_mp_context, worker_init

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

    # Use shared context and signal handler for proper Ctrl-C handling
    ctx = get_mp_context()
    pool = None

    try:
        pool = ctx.Pool(
            processes=num_workers,
            initializer=worker_init
        )
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

    except KeyboardInterrupt:
        logger.warning("Interrupted - terminating hash workers...")
        raise
    finally:
        if pool:
            pool.terminate()
            pool.join()

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
    def _get_indexed_quick_hashes(self, collection_name: str) -> set:
        """Get all indexed (file_path, quick_hash) pairs for Pass 1 matching.

        Used during incremental sync to detect unchanged files without reindexing.
        Returns a set of (file_path, quick_hash) tuples for O(1) lookup.

        For chunks with file_quick_hashes dict, includes all alternate paths.
        For old format chunks, includes only the primary file_path.

        Args:
            collection_name: Qdrant collection name

        Returns:
            Set of (file_path, quick_hash) tuples
        """
        indexed_pairs = set()
        offset = None
        chunks_with_dict = 0
        chunks_with_array = 0
        total_chunks = 0
        dict_entries = 0

        try:
            while True:
                points, offset = self.qdrant.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=["file_path", "quick_hash", "file_quick_hashes", "file_paths"],
                    with_vectors=False
                )

                if not points:
                    break

                for point in points:
                    total_chunks += 1
                    if point.payload:
                        # New format: dict of path → quick_hash for all locations
                        # Use ONLY dict format if it exists (skip old format to avoid conflicts)
                        # Check for key existence, not truthiness, to handle empty dicts correctly
                        if "file_quick_hashes" in point.payload:
                            file_quick_hashes = point.payload.get("file_quick_hashes", {})
                            chunks_with_dict += 1
                            for dict_path, dict_quick_hash in file_quick_hashes.items():
                                dict_entries += 1
                                if dict_path and dict_quick_hash:
                                    indexed_pairs.add((dict_path, dict_quick_hash))
                        else:
                            # Old format: single path + quick_hash (only if dict doesn't exist)
                            path = point.payload.get("file_path")
                            quick_hash = point.payload.get("quick_hash")
                            if path and quick_hash:
                                indexed_pairs.add((path, quick_hash))

                        # Check if file_paths array exists
                        file_paths = point.payload.get("file_paths", [])
                        if file_paths:
                            chunks_with_array += 1

                if offset is None:
                    break

            logger.debug(f"Pass 1: Loaded {len(indexed_pairs)} unique (path, quick_hash) pairs from {total_chunks} chunks ({chunks_with_dict} with dict)")

            return indexed_pairs

        except Exception as e:
            logger.warning(f"Error querying quick_hashes: {e}")
            return set()

    def _get_indexed_file_paths_set(self, collection_name: str) -> set:
        """Get all indexed file_path values for deduplication in Pass 2.

        Used to determine which Pass 1 misses are new files vs existing files with changed metadata.
        Returns a set of file_path strings for O(1) lookup.

        Args:
            collection_name: Qdrant collection name

        Returns:
            Set of file_path strings
        """
        indexed_paths = set()
        offset = None

        try:
            while True:
                points, offset = self.qdrant.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=["file_path"],
                    with_vectors=False
                )

                if not points:
                    break

                for point in points:
                    if point.payload:
                        path = point.payload.get("file_path")
                        if path:
                            indexed_paths.add(path)

                if offset is None:
                    break

            return indexed_paths

        except Exception as e:
            logger.warning(f"Error querying indexed paths: {e}")
            return set()

    def get_chunk_counts_by_file(self, collection_name: str) -> Dict[str, int]:
        """Get chunk counts per file_path from Qdrant collection.

        Args:
            collection_name: Qdrant collection name

        Returns:
            Dict mapping file_path to chunk count
        """
        chunk_counts: Dict[str, int] = {}
        offset = None

        try:
            while True:
                points, offset = self.qdrant.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=["file_path"],
                    with_vectors=False
                )

                if not points:
                    break

                for point in points:
                    if point.payload:
                        path = point.payload.get("file_path")
                        if path:
                            chunk_counts[path] = chunk_counts.get(path, 0) + 1

                if offset is None:
                    break

            return chunk_counts

        except Exception as e:
            logger.warning(f"Error querying chunk counts: {e}")
            return {}

    def get_unindexed_files(self, collection_name: str,
                            file_list: List[Path],
                            hash_fn=None) -> Tuple[List[Path], List[Path]]:
        """Filter file list using fast metadata check to identify files needing processing.

        **Single-pass strategy** (deferred content hashing):

        Pass 1 (Ultra-fast ~0.003ms): Metadata check (mtime+size)
        - Compute quick_hash for all files
        - Query Qdrant for (file_path, quick_hash) matches
        - If found → file unchanged at same location → skip
        - If not found → needs processing (duplicate detection deferred to indexing)

        **Performance**:
        - Unchanged files (99%): ~0.003ms per file
        - Changed/new files: ~0.003ms + deferred to indexing phase

        **Content hashing and duplicate detection** now happen during indexing phase:
        - Eliminates double hashing (was: sync + indexing)
        - Each file hashed only once (during indexing)
        - Duplicate/rename detection at start of indexing

        Args:
            collection_name: Qdrant collection name
            file_list: List of file paths to check
            hash_fn: Deprecated parameter, ignored (kept for backward compatibility)

        Returns:
            Tuple of:
            - needs_processing: Files that need indexing/duplicate-check (Pass 1 miss)
            - already_indexed: Files unchanged (Pass 1 hit)
        """
        try:
            import time
            start_time = time.time()

            # Brief delay to ensure Qdrant consistency (in case previous run just finished)
            # This prevents race conditions where chunks are uploaded but not yet queryable
            time.sleep(1.0)

            # Pass 1: Metadata gate (mtime+size) - ultra fast
            pass1_start = time.time()
            quick_hashes = _compute_hashes_parallel(file_list, compute_quick_hash, show_progress=False)
            quick_hash_time = time.time() - pass1_start

            pass1_qdrant_start = time.time()
            indexed_quick_hashes = self._get_indexed_quick_hashes(collection_name)
            pass1_qdrant_time = time.time() - pass1_qdrant_start

            # Categorize files
            needs_processing = []
            already_indexed = []

            # Pass 1: Quick metadata check
            pass1_check_start = time.time()
            for file_path in file_list:
                file_path_str = str(file_path.absolute())
                quick_hash = quick_hashes.get(file_path_str)

                if quick_hash is None:
                    # Hash computation failed, treat as needs processing
                    needs_processing.append(file_path)
                    continue

                # Check if (file_path, quick_hash) exists in collection
                if (file_path_str, quick_hash) in indexed_quick_hashes:
                    # Pass 1 HIT: Metadata unchanged → skip
                    already_indexed.append(file_path)
                else:
                    # Pass 1 MISS: Metadata changed or new → defer to indexing phase
                    needs_processing.append(file_path)

            pass1_check_time = time.time() - pass1_check_start
            pass1_total = time.time() - pass1_start

            logger.info(f"Pass 1 (metadata gate): {len(file_list)} files "
                       f"→ {len(already_indexed)} hits, {len(needs_processing)} need processing "
                       f"(hash: {quick_hash_time:.3f}s, qdrant: {pass1_qdrant_time:.3f}s, check: {pass1_check_time:.3f}s, total: {pass1_total:.3f}s)")

            total_time = time.time() - start_time
            logger.info(f"Sync complete: {len(needs_processing)} files to process, "
                       f"{len(already_indexed)} already indexed "
                       f"(total {len(file_list)} files, {total_time:.3f}s)")
            return (needs_processing, already_indexed)

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

    def find_file_by_content_hash(self, collection_name: str, file_hash: str) -> List[str]:
        """Find all file_paths in collection with given content hash.

        Returns ALL paths including both the primary file_path and any paths
        in the file_paths array (for duplicates).

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
                    with_payload=["file_path", "file_paths"],
                    with_vectors=False
                )

                if not points:
                    break

                for point in points:
                    if point.payload:
                        # Primary path (old format)
                        if "file_path" in point.payload:
                            path = point.payload["file_path"]
                            if path not in paths:
                                paths.append(path)

                        # All paths (new format with duplicates)
                        file_paths_array = point.payload.get("file_paths", [])
                        for path in file_paths_array:
                            if path and path not in paths:
                                paths.append(path)

                if offset is None:
                    break

            return paths

        except Exception as e:
            logger.warning(f"Error finding file by content hash: {e}")
            return []

    def filter_existing_paths(self, paths: List[str]) -> List[str]:
        """Filter a list of paths to only those that exist on the filesystem.

        Args:
            paths: List of file paths to check

        Returns:
            List of paths that exist on disk
        """
        from pathlib import Path
        return [p for p in paths if Path(p).exists()]

    def handle_renames(self, collection_name: str,
                      renames: List[Tuple]) -> int:
        """Update metadata for renamed/moved files.

        Updates all chunks with old_path to use new_path and updates
        all path-dependent metadata. This avoids reindexing when only
        the path changed.

        Args:
            collection_name: Qdrant collection name
            renames: List of tuples, either:
                - (old_path, new_path) for backward compatibility
                - (old_path, new_path, metadata_dict) for full update

        Returns:
            Number of files successfully renamed
        """
        if not renames:
            return 0

        renamed_count = 0

        for rename_info in renames:
            try:
                # Support both old format (2-tuple) and new format (3-tuple)
                if len(rename_info) == 2:
                    old_path, new_path = rename_info
                    new_metadata = {}
                elif len(rename_info) == 3:
                    old_path, new_path, new_metadata = rename_info
                else:
                    logger.warning(f"Invalid rename tuple length: {len(rename_info)}")
                    continue

                # Build payload with all path-dependent metadata
                payload = {"file_path": new_path}

                # Add optional metadata fields if provided
                if "filename" in new_metadata:
                    payload["filename"] = new_metadata["filename"]
                if "quick_hash" in new_metadata:
                    payload["quick_hash"] = new_metadata["quick_hash"]
                # Note: file_hash stays same (it's the key for finding renames)
                # Note: file_size stays same

                # Update all chunks with old_path
                self.qdrant.set_payload(
                    collection_name=collection_name,
                    payload=payload,
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

                if new_metadata:
                    logger.info(f"Renamed: {old_path} -> {new_path} (updated {len(payload)} fields)")
                else:
                    logger.info(f"Renamed: {old_path} -> {new_path}")

            except Exception as e:
                logger.warning(f"Failed to rename {old_path} -> {new_path}: {e}")

        if renamed_count > 0:
            logger.info(f"Successfully renamed {renamed_count}/{len(renames)} files")

        return renamed_count

    def add_alternate_path(self, collection_name: str, file_hash: str, new_path: str, quick_hash: str) -> int:
        """Add an alternate file path to existing chunks with same content.

        When a duplicate file is found (same content, different path), this adds
        the new path to the file_paths array and stores its quick_hash for Pass 1 sync.

        Args:
            collection_name: Qdrant collection name
            file_hash: Content hash to find existing chunks
            new_path: New file path to add to file_paths array
            quick_hash: Quick hash (mtime+size) for this specific file

        Returns:
            Number of chunks updated
        """
        try:
            # First, get one chunk to see current file_paths state
            points, _ = self.qdrant.scroll(
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
                with_payload=True,
                with_vectors=False
            )

            if not points:
                logger.warning(f"No chunks found with file_hash {file_hash}")
                return 0

            # Get current state
            current_payload = points[0].payload
            file_paths = current_payload.get("file_paths", [])
            file_quick_hashes = current_payload.get("file_quick_hashes", {})

            # Lazy migration: if file_paths doesn't exist, create from file_path
            if not file_paths and "file_path" in current_payload:
                file_paths = [current_payload["file_path"]]
                logger.debug(f"Migrated old format: created file_paths array from file_path")

            # IMPORTANT: Also migrate primary path's quick_hash if missing from dict
            primary_path = current_payload.get("file_path")
            if primary_path and primary_path not in file_quick_hashes and "quick_hash" in current_payload:
                file_quick_hashes[primary_path] = current_payload["quick_hash"]
                logger.debug(f"Migrated primary path quick_hash to file_quick_hashes dict")

            # Ensure all paths in array have dict entries (migration for partially-migrated chunks)
            for existing_path in file_paths:
                if existing_path not in file_quick_hashes:
                    # Auto-fix: populate missing dict entry from old field if possible
                    if existing_path == primary_path and "quick_hash" in current_payload:
                        file_quick_hashes[existing_path] = current_payload["quick_hash"]
                        logger.debug(f"Auto-fixed: migrated missing dict entry for {existing_path}")
                    else:
                        logger.warning(f"Path {existing_path} missing dict entry and cannot auto-fix")

            # Add new path if not already present
            new_path_abs = str(new_path) if not isinstance(new_path, str) else new_path
            path_existed = new_path_abs in file_paths
            dict_entry_existed = new_path_abs in file_quick_hashes

            if not path_existed:
                file_paths.append(new_path_abs)

            # Check if quick_hash changed (file was touched/modified)
            quick_hash_changed = file_quick_hashes.get(new_path_abs) != quick_hash

            # Always update dict entry (creates new or updates existing)
            file_quick_hashes[new_path_abs] = quick_hash

            # If both already existed with same quick_hash, nothing to do
            if path_existed and dict_entry_existed and not quick_hash_changed:
                logger.debug(f"Path {new_path} already fully tracked with same quick_hash")
                return 0

            # Update all chunks with this file_hash
            # Keep old quick_hash field in sync with primary path's dict entry for consistency
            update_payload = {
                "file_paths": file_paths,
                "file_quick_hashes": file_quick_hashes
            }
            if primary_path and primary_path in file_quick_hashes:
                update_payload["quick_hash"] = file_quick_hashes[primary_path]

            result = self.qdrant.set_payload(
                collection_name=collection_name,
                payload=update_payload,
                points=FilterSelector(
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

            logger.debug(f"Added alternate path {new_path} to {len(file_paths)} total paths for hash {file_hash[:8]}")
            return len(file_paths)

        except Exception as e:
            logger.warning(f"Failed to add alternate path {new_path}: {e}")
            return 0

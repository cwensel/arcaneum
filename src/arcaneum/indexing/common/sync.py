"""Metadata-based sync for incremental indexing (RDR-004)."""

import hashlib
from pathlib import Path
from typing import List, Set
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import logging

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file content.

    Matches the hash computation in discovery.py for consistency.
    Uses text mode with UTF-8 encoding and latin-1 fallback.

    Args:
        file_path: Path to file

    Returns:
        Full SHA256 hash (64 characters)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decode failed for {file_path}, trying latin-1")
        content = file_path.read_text(encoding='latin-1')

    return hashlib.sha256(content.encode('utf-8')).hexdigest()


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
                            file_list: List[Path]) -> List[Path]:
        """Filter file list to only unindexed or modified files.

        Uses batch query for efficiency instead of per-file queries.

        Args:
            collection_name: Qdrant collection name
            file_list: List of file paths to check

        Returns:
            List of files that need indexing
        """
        try:
            # Get all indexed (path, hash) pairs
            indexed = self.get_indexed_file_paths(collection_name)

            # Filter to files not in indexed set
            unindexed = []
            for file_path in file_list:
                file_hash = compute_file_hash(file_path)
                # Use absolute path to match how paths are stored during indexing
                if (str(file_path.absolute()), file_hash) not in indexed:
                    unindexed.append(file_path)

            logger.info(f"Found {len(unindexed)}/{len(file_list)} "
                       f"files to index ({len(file_list) - len(unindexed)} "
                       f"already indexed)")
            return unindexed

        except Exception as e:
            logger.warning(f"Error querying collection: {e}, "
                          "processing all files")
            return file_list

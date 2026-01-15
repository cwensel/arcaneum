"""Change detection utilities for full-text indexing (RDR-010).

Provides file hash-based change detection for idempotent re-indexing.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Set, Optional

from ...fulltext.client import FullTextClient

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Streams file in 8KB chunks for memory efficiency.

    Args:
        file_path: Path to file

    Returns:
        SHA-256 hex digest
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_indexed_files(
    meili_client: FullTextClient,
    index_name: str,
    limit: int = 10000
) -> Dict[str, str]:
    """Get mapping of indexed file paths to their hashes.

    Queries MeiliSearch for all indexed documents and returns
    a mapping of file_path -> file_hash for change detection.

    Args:
        meili_client: MeiliSearch client instance
        index_name: Index name to query
        limit: Maximum documents to retrieve

    Returns:
        Dict mapping file_path to file_hash
    """
    indexed_files: Dict[str, str] = {}

    try:
        # Get all unique file paths and hashes
        # MeiliSearch doesn't support GROUP BY, so we get all docs
        results = meili_client.search(
            index_name=index_name,
            query='',
            limit=limit
        )

        hits = results.get('hits', [])

        for hit in hits:
            file_path = hit.get('file_path')
            file_hash = hit.get('file_hash')
            if file_path and file_hash:
                indexed_files[file_path] = file_hash

    except Exception as e:
        logger.warning(f"Failed to get indexed files: {e}")

    return indexed_files


def find_files_to_index(
    pdf_files: list[Path],
    indexed_files: Dict[str, str],
    force_reindex: bool = False
) -> tuple[list[Path], list[Path], list[Path]]:
    """Determine which files need indexing based on change detection.

    Compares file hashes to identify new, modified, and unchanged files.

    Args:
        pdf_files: List of PDF files to check
        indexed_files: Mapping of file_path -> file_hash from index
        force_reindex: If True, return all files as needing indexing

    Returns:
        Tuple of (new_files, modified_files, unchanged_files)
    """
    new_files: list[Path] = []
    modified_files: list[Path] = []
    unchanged_files: list[Path] = []

    for pdf_path in pdf_files:
        file_path_str = str(pdf_path.absolute())

        if force_reindex:
            # Force mode: treat all as new
            new_files.append(pdf_path)
            continue

        if file_path_str not in indexed_files:
            # New file (not in index)
            new_files.append(pdf_path)
        else:
            # File exists in index, check hash
            current_hash = compute_file_hash(pdf_path)
            indexed_hash = indexed_files[file_path_str]

            if current_hash != indexed_hash:
                # File modified (hash changed)
                modified_files.append(pdf_path)
            else:
                # File unchanged
                unchanged_files.append(pdf_path)

    return new_files, modified_files, unchanged_files


def get_orphaned_files(
    indexed_files: Dict[str, str],
    pdf_files: list[Path]
) -> Set[str]:
    """Find indexed files that no longer exist on disk.

    Args:
        indexed_files: Mapping of file_path -> file_hash from index
        pdf_files: List of PDF files currently on disk

    Returns:
        Set of file paths that are indexed but don't exist
    """
    current_paths = {str(p.absolute()) for p in pdf_files}
    indexed_paths = set(indexed_files.keys())

    return indexed_paths - current_paths

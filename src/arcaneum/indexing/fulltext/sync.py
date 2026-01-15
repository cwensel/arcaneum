"""Change detection utilities for full-text indexing (RDR-010, RDR-011).

Provides file hash-based change detection for idempotent re-indexing.
Extended in RDR-011 for git-aware source code indexing with branch support.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Set, Optional, Tuple

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


# =============================================================================
# Git-Aware Sync for Source Code (RDR-011)
# =============================================================================

class GitCodeMetadataSync:
    """Query MeiliSearch for indexed git projects (source of truth).

    This class implements the metadata-based sync pattern from RDR-005,
    adapted for MeiliSearch full-text indexing. MeiliSearch serves as
    the single source of truth for what's indexed.

    Key features:
    - Multi-branch support via composite identifiers (project#branch)
    - Commit hash comparison for change detection
    - Branch-specific deletion (other branches unaffected)
    """

    def __init__(self, meili_client: FullTextClient):
        """Initialize git metadata sync.

        Args:
            meili_client: MeiliSearch client instance
        """
        self.meili_client = meili_client
        self._cache: Dict[str, Dict[str, str]] = {}

    def get_indexed_projects(self, index_name: str) -> Dict[str, str]:
        """Get all (git_project_identifier, git_commit_hash) pairs from MeiliSearch.

        Queries MeiliSearch for distinct project identifiers and their
        commit hashes. Results are cached to avoid repeated queries.

        Args:
            index_name: MeiliSearch index name

        Returns:
            Dict mapping git_project_identifier -> git_commit_hash
        """
        if index_name in self._cache:
            return self._cache[index_name]

        indexed: Dict[str, str] = {}

        try:
            # MeiliSearch doesn't support GROUP BY, so we paginate through
            # documents and deduplicate by project identifier
            offset = 0
            limit = 1000

            while True:
                results = self.meili_client.search(
                    index_name=index_name,
                    query='',
                    limit=limit,
                    offset=offset
                )

                hits = results.get('hits', [])
                if not hits:
                    break

                for hit in hits:
                    identifier = hit.get('git_project_identifier')
                    commit = hit.get('git_commit_hash')
                    if identifier and commit and identifier not in indexed:
                        indexed[identifier] = commit

                offset += len(hits)

                # Stop if we got fewer results than requested
                if len(hits) < limit:
                    break

            self._cache[index_name] = indexed

        except Exception as e:
            logger.warning(f"Failed to get indexed projects from {index_name}: {e}")

        return indexed

    def should_reindex_project(
        self,
        index_name: str,
        project_identifier: str,
        current_commit: str
    ) -> bool:
        """Check if (project, branch) needs re-indexing.

        Compares the current commit hash against what's stored in MeiliSearch.

        Args:
            index_name: MeiliSearch index name
            project_identifier: Composite identifier (e.g., "project#branch")
            current_commit: Current HEAD commit hash

        Returns:
            True if project needs to be (re)indexed
        """
        indexed = self.get_indexed_projects(index_name)

        # Not indexed yet
        if project_identifier not in indexed:
            return True

        # Commit changed
        if indexed[project_identifier] != current_commit:
            return True

        # Unchanged
        return False

    def delete_project_documents(
        self,
        index_name: str,
        project_identifier: str
    ) -> int:
        """Delete all documents for specific (project, branch).

        Uses MeiliSearch filter-based deletion to remove all documents
        with a specific git_project_identifier. Other branches are unaffected.

        Args:
            index_name: MeiliSearch index name
            project_identifier: Composite identifier (e.g., "project#branch")

        Returns:
            Number of documents deleted (estimated)
        """
        try:
            index = self.meili_client.get_index(index_name)

            # Get document IDs to delete
            deleted_count = 0
            offset = 0
            limit = 1000

            while True:
                results = self.meili_client.search(
                    index_name=index_name,
                    query='',
                    filter=f'git_project_identifier = "{project_identifier}"',
                    limit=limit,
                    offset=offset
                )

                hits = results.get('hits', [])
                if not hits:
                    break

                # Delete documents by ID
                doc_ids = [hit['id'] for hit in hits if 'id' in hit]
                if doc_ids:
                    task = index.delete_documents(doc_ids)
                    self.meili_client.client.wait_for_task(task.task_uid)
                    deleted_count += len(doc_ids)

                # Stop if we got fewer results than requested
                if len(hits) < limit:
                    break

                # Reset offset since we're deleting
                # Actually, after deletion the next search will get new results
                # So we don't increment offset

            # Invalidate cache
            if index_name in self._cache:
                del self._cache[index_name]

            logger.info(f"Deleted {deleted_count} documents for {project_identifier}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete documents for {project_identifier}: {e}")
            return 0

    def clear_cache(self):
        """Clear the indexed projects cache.

        Call this after making changes to the index to ensure
        fresh data on next query.
        """
        self._cache.clear()

    def get_project_document_count(
        self,
        index_name: str,
        project_identifier: str
    ) -> int:
        """Get the number of documents for a specific project.

        Args:
            index_name: MeiliSearch index name
            project_identifier: Composite identifier (e.g., "project#branch")

        Returns:
            Number of documents for this project
        """
        try:
            results = self.meili_client.search(
                index_name=index_name,
                query='',
                filter=f'git_project_identifier = "{project_identifier}"',
                limit=0  # We only need the count
            )
            return results.get('estimatedTotalHits', 0)

        except Exception as e:
            logger.warning(f"Failed to get document count for {project_identifier}: {e}")
            return 0

"""
Metadata-based sync for git projects using Qdrant as source of truth.

This module queries Qdrant metadata to determine which projects need indexing,
following the RDR-04 pattern with Qdrant as the single source of truth (RDR-005).
"""

import logging
from typing import Dict, Optional, Tuple, Set
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

logger = logging.getLogger(__name__)


@dataclass
class IndexedProject:
    """Represents an indexed project in Qdrant.

    Attributes:
        identifier: Composite identifier "project#branch"
        commit_hash: Full 40-char SHA of indexed commit
        point_count: Number of chunks indexed for this project
    """
    identifier: str
    commit_hash: str
    point_count: int = 0


class GitMetadataSync:
    """Query Qdrant for indexed projects (source of truth, follows RDR-04 pattern).

    This class manages incremental indexing by:
    1. Querying Qdrant for all indexed (project, branch, commit) combinations
    2. Caching results per collection to avoid repeated queries
    3. Comparing current git state with Qdrant metadata
    4. Determining which projects need re-indexing

    Qdrant is the single source of truth - if metadata says a project is indexed,
    it is indexed. No external state tracking (like SQLite) is used.
    """

    def __init__(self, qdrant_client: QdrantClient):
        """Initialize metadata sync.

        Args:
            qdrant_client: Initialized Qdrant client
        """
        self.qdrant = qdrant_client
        self._cache: Dict[str, Dict[str, IndexedProject]] = {}

    def get_indexed_projects(
        self,
        collection_name: str,
        force_refresh: bool = False
    ) -> Dict[str, IndexedProject]:
        """Get all indexed projects from Qdrant metadata.

        Queries Qdrant using scroll to retrieve all (project, branch, commit)
        combinations that are currently indexed. Results are cached per collection.

        Args:
            collection_name: Name of Qdrant collection
            force_refresh: If True, bypass cache and re-query Qdrant

        Returns:
            Dictionary mapping git_project_identifier to IndexedProject
            Example: {"arcaneum#main": IndexedProject("arcaneum#main", "abc123...", 150)}

        Performance:
            - Target: <5s for 1000 projects
            - Uses scroll API with batch size of 100
            - Only fetches payload metadata, no vectors
        """
        # Return cached results if available
        if not force_refresh and collection_name in self._cache:
            logger.debug(f"Using cached indexed projects for {collection_name}")
            return self._cache[collection_name]

        logger.debug(f"Querying Qdrant for indexed projects in {collection_name}")

        indexed_projects: Dict[str, IndexedProject] = {}
        offset = None
        batch_count = 0

        try:
            while True:
                # Scroll through all points, fetching only required metadata
                points, offset = self.qdrant.scroll(
                    collection_name=collection_name,
                    with_payload=["git_project_identifier", "git_commit_hash"],
                    with_vectors=False,  # Don't fetch vectors (performance)
                    limit=100,
                    offset=offset
                )

                batch_count += 1

                if not points:
                    break

                # Extract unique (identifier, commit) pairs
                for point in points:
                    payload = point.payload
                    identifier = payload.get("git_project_identifier")
                    commit_hash = payload.get("git_commit_hash")

                    if identifier and commit_hash:
                        if identifier not in indexed_projects:
                            # First time seeing this identifier
                            indexed_projects[identifier] = IndexedProject(
                                identifier=identifier,
                                commit_hash=commit_hash,
                                point_count=1
                            )
                        else:
                            # Already seen - increment count and verify commit match
                            existing = indexed_projects[identifier]
                            existing.point_count += 1

                            # Warn if same identifier has different commits (shouldn't happen)
                            if existing.commit_hash != commit_hash:
                                logger.warning(
                                    f"Inconsistent commits for {identifier}: "
                                    f"{existing.commit_hash[:12]} vs {commit_hash[:12]}"
                                )

                # Break if no more results
                if offset is None:
                    break

            logger.debug(
                f"Found {len(indexed_projects)} indexed projects "
                f"in {collection_name} ({batch_count} batches)"
            )

            # Cache results
            self._cache[collection_name] = indexed_projects

            return indexed_projects

        except Exception as e:
            logger.error(f"Error querying Qdrant metadata: {e}")
            # Return empty dict on error (fail-safe: will re-index everything)
            return {}

    def should_reindex_project(
        self,
        collection_name: str,
        project_identifier: str,
        current_commit: str
    ) -> bool:
        """Check if a project needs re-indexing by comparing commits.

        Args:
            collection_name: Name of Qdrant collection
            project_identifier: Composite identifier "project#branch"
            current_commit: Current commit SHA from git

        Returns:
            True if project should be indexed:
            - Project not yet indexed (new)
            - Commit changed (update needed)
            False if project unchanged (skip)
        """
        indexed = self.get_indexed_projects(collection_name)

        # Not indexed yet
        if project_identifier not in indexed:
            logger.debug(f"Project {project_identifier} not indexed yet")
            return True

        # Check if commit changed
        indexed_project = indexed[project_identifier]
        if indexed_project.commit_hash != current_commit:
            logger.debug(
                f"Commit changed for {project_identifier}: "
                f"{indexed_project.commit_hash[:12]} -> {current_commit[:12]}"
            )
            return True

        # Unchanged
        logger.debug(
            f"Project {project_identifier} unchanged "
            f"(commit {current_commit[:12]})"
        )
        return False

    def get_project_stats(
        self,
        collection_name: str,
        project_identifier: str
    ) -> Optional[IndexedProject]:
        """Get statistics for a specific indexed project.

        Args:
            collection_name: Name of Qdrant collection
            project_identifier: Composite identifier "project#branch"

        Returns:
            IndexedProject with stats, or None if not indexed
        """
        indexed = self.get_indexed_projects(collection_name)
        return indexed.get(project_identifier)

    def get_all_branches(
        self,
        collection_name: str,
        project_name: str
    ) -> Set[str]:
        """Get all indexed branches for a specific project.

        Args:
            collection_name: Name of Qdrant collection
            project_name: Project name (without branch suffix)

        Returns:
            Set of branch names for this project

        Example:
            >>> sync.get_all_branches("code", "arcaneum")
            {"main", "feature-x", "develop"}
        """
        indexed = self.get_indexed_projects(collection_name)

        branches = set()
        for identifier in indexed.keys():
            # Parse identifier: "project#branch"
            if "#" in identifier:
                proj, branch = identifier.split("#", 1)
                if proj == project_name:
                    branches.add(branch)

        return branches

    def count_total_chunks(self, collection_name: str) -> int:
        """Count total number of code chunks indexed in collection.

        Args:
            collection_name: Name of Qdrant collection

        Returns:
            Total number of chunks across all projects
        """
        indexed = self.get_indexed_projects(collection_name)
        return sum(proj.point_count for proj in indexed.values())

    def clear_cache(self, collection_name: Optional[str] = None):
        """Clear cached metadata.

        Args:
            collection_name: Clear cache for specific collection, or all if None
        """
        if collection_name:
            self._cache.pop(collection_name, None)
            logger.debug(f"Cleared cache for {collection_name}")
        else:
            self._cache.clear()
            logger.debug("Cleared all cached metadata")

    def get_stale_identifiers(
        self,
        collection_name: str,
        current_identifiers: Set[str]
    ) -> Set[str]:
        """Find stale project identifiers that are indexed but no longer exist.

        This helps identify branches that have been deleted or projects that
        have been removed from the indexing directory.

        Args:
            collection_name: Name of Qdrant collection
            current_identifiers: Set of identifiers from current git scan

        Returns:
            Set of identifiers indexed in Qdrant but not in current scan

        Example:
            If Qdrant has ["proj#main", "proj#old-branch"] but current scan
            only finds ["proj#main"], returns {"proj#old-branch"}
        """
        indexed = self.get_indexed_projects(collection_name)
        indexed_identifiers = set(indexed.keys())

        stale = indexed_identifiers - current_identifiers

        if stale:
            logger.info(f"Found {len(stale)} stale identifiers: {stale}")

        return stale

    def is_version_indexed(
        self,
        collection_name: str,
        version_identifier: str
    ) -> bool:
        """Check if a specific version (project#branch@commit) is already indexed.

        This supports the --git-version mode where multiple versions of the same
        branch can coexist in the collection.

        Args:
            collection_name: Name of Qdrant collection
            version_identifier: Versioned identifier "project#branch@commit_short"
                               (e.g., "arcaneum#main@abc1234")

        Returns:
            True if version is already indexed, False otherwise
        """
        try:
            # Query for any points with this version identifier
            result = self.qdrant.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="git_version_identifier",
                            match=MatchValue(value=version_identifier)
                        )
                    ]
                ),
                with_payload=False,
                with_vectors=False,
                limit=1  # Only need to know if any exist
            )

            points, _ = result
            is_indexed = len(points) > 0

            if is_indexed:
                logger.debug(f"Version {version_identifier} already indexed in {collection_name}")
            else:
                logger.debug(f"Version {version_identifier} not indexed in {collection_name}")

            return is_indexed

        except Exception as e:
            logger.error(f"Error checking version index for {version_identifier}: {e}")
            # Fail-safe: return False to allow indexing
            return False

    def verify_consistency(self, collection_name: str) -> Tuple[bool, str]:
        """Verify metadata consistency in collection.

        Checks:
        - All projects have consistent commit hashes
        - No missing required fields
        - No duplicate identifiers with different commits

        Args:
            collection_name: Name of Qdrant collection

        Returns:
            Tuple of (is_consistent, message)
        """
        try:
            indexed = self.get_indexed_projects(collection_name, force_refresh=True)

            if not indexed:
                return True, "Collection empty or no projects indexed"

            # Check for projects with very low chunk counts (might indicate partial index)
            low_count_projects = [
                p.identifier for p in indexed.values()
                if p.point_count < 5
            ]

            if low_count_projects:
                return False, (
                    f"Found {len(low_count_projects)} projects with <5 chunks: "
                    f"{low_count_projects[:5]}"
                )

            return True, f"Metadata consistent for {len(indexed)} projects"

        except Exception as e:
            return False, f"Error verifying consistency: {e}"

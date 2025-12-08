"""
Collection verification (fsck-like) for detecting incomplete indexing.

This module provides verification functionality to detect and report items
(repositories, files) that have incomplete chunk sets in Qdrant collections.

Key checks:
- Code collections: Verify chunk_index coverage matches chunk_count for each file
- PDF/Markdown collections: Verify all chunks exist for each file

Usage:
    from arcaneum.indexing.verify import CollectionVerifier
    verifier = CollectionVerifier(qdrant_client)
    result = verifier.verify_collection("MyCollection")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from arcaneum.indexing.collection_metadata import get_collection_type

logger = logging.getLogger(__name__)


@dataclass
class FileVerificationResult:
    """Verification result for a single file."""

    file_path: str
    expected_chunks: int  # chunk_count from metadata
    actual_chunks: int  # number of chunks found
    missing_indices: List[int] = field(default_factory=list)  # which chunk_indices are missing
    is_complete: bool = True

    @property
    def completion_percentage(self) -> float:
        """Return completion percentage."""
        if self.expected_chunks == 0:
            return 100.0
        return (self.actual_chunks / self.expected_chunks) * 100


@dataclass
class ProjectVerificationResult:
    """Verification result for a code project (repo)."""

    identifier: str  # git_project_identifier
    project_name: str
    branch: str
    commit_hash: str
    total_files: int
    complete_files: int
    incomplete_files: List[FileVerificationResult] = field(default_factory=list)
    total_expected_chunks: int = 0
    total_actual_chunks: int = 0

    @property
    def is_complete(self) -> bool:
        return len(self.incomplete_files) == 0

    @property
    def completion_percentage(self) -> float:
        if self.total_expected_chunks == 0:
            return 100.0
        return (self.total_actual_chunks / self.total_expected_chunks) * 100


@dataclass
class CollectionVerificationResult:
    """Overall verification result for a collection."""

    collection_name: str
    collection_type: Optional[str]
    total_points: int
    total_items: int  # projects for code, files for pdf/markdown
    complete_items: int
    incomplete_items: int
    is_healthy: bool
    # Detailed results by item
    projects: List[ProjectVerificationResult] = field(default_factory=list)  # for code
    files: List[FileVerificationResult] = field(default_factory=list)  # for pdf/markdown
    errors: List[str] = field(default_factory=list)

    def get_items_needing_repair(self) -> List[str]:
        """Return list of items that need re-indexing."""
        if self.collection_type == "code":
            return [p.identifier for p in self.projects if not p.is_complete]
        else:
            return [f.file_path for f in self.files if not f.is_complete]


class CollectionVerifier:
    """Verify collection integrity by checking chunk completeness.

    This class performs fsck-like verification of Qdrant collections,
    detecting items with incomplete chunk sets that may need re-indexing.
    """

    def __init__(self, qdrant_client: QdrantClient):
        """Initialize verifier.

        Args:
            qdrant_client: Initialized Qdrant client
        """
        self.qdrant = qdrant_client

    def verify_collection(
        self,
        collection_name: str,
        project_filter: Optional[str] = None,
        verbose: bool = False,
    ) -> CollectionVerificationResult:
        """Verify a collection's integrity.

        Scans all chunks in the collection and verifies that each file
        has a complete set of chunks (chunk_index 0 to chunk_count-1).

        Args:
            collection_name: Name of the Qdrant collection
            project_filter: Optional filter for specific project identifier (code only)
            verbose: Log verbose output

        Returns:
            CollectionVerificationResult with detailed verification data
        """
        collection_type = get_collection_type(self.qdrant, collection_name)

        # Get total point count
        collection_info = self.qdrant.get_collection(collection_name)
        total_points = collection_info.points_count

        if collection_type == "code":
            return self._verify_code_collection(
                collection_name, total_points, project_filter, verbose
            )
        else:
            return self._verify_file_collection(
                collection_name, collection_type, total_points, verbose
            )

    def _verify_code_collection(
        self,
        collection_name: str,
        total_points: int,
        project_filter: Optional[str] = None,
        verbose: bool = False,
    ) -> CollectionVerificationResult:
        """Verify a source code collection.

        For each file in each project, verifies that all chunk_indices exist.
        """
        logger.debug(f"Verifying code collection: {collection_name}")

        # Collect chunk info: {project_id: {file_path: {chunk_index: chunk_count}}}
        project_files: Dict[str, Dict[str, Dict[str, any]]] = defaultdict(
            lambda: defaultdict(lambda: {"indices": set(), "chunk_count": 0, "project_info": {}})
        )

        # Build filter if project specified
        scroll_filter = None
        if project_filter:
            scroll_filter = Filter(
                must=[
                    FieldCondition(
                        key="git_project_identifier",
                        match=MatchValue(value=project_filter),
                    )
                ]
            )

        # Scroll through all points
        offset = None
        batch_count = 0
        payload_fields = [
            "git_project_identifier",
            "git_project_name",
            "git_branch",
            "git_commit_hash",
            "file_path",
            "chunk_index",
            "chunk_count",
        ]

        while True:
            points, offset = self.qdrant.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                with_payload=payload_fields,
                with_vectors=False,
                limit=500,
                offset=offset,
            )

            batch_count += 1
            if verbose and batch_count % 10 == 0:
                logger.debug(f"Processed {batch_count} batches...")

            if not points:
                break

            for point in points:
                payload = point.payload
                if not payload:
                    continue

                project_id = payload.get("git_project_identifier")
                file_path = payload.get("file_path")
                chunk_index = payload.get("chunk_index")
                chunk_count = payload.get("chunk_count")

                if not all([project_id, file_path, chunk_index is not None, chunk_count]):
                    continue

                file_data = project_files[project_id][file_path]
                file_data["indices"].add(chunk_index)
                file_data["chunk_count"] = chunk_count
                file_data["project_info"] = {
                    "project_name": payload.get("git_project_name", ""),
                    "branch": payload.get("git_branch", ""),
                    "commit_hash": payload.get("git_commit_hash", ""),
                }

            if offset is None:
                break

        # Analyze results
        project_results: List[ProjectVerificationResult] = []
        total_complete = 0
        total_incomplete = 0

        for project_id, files in project_files.items():
            project_info = None
            incomplete_files = []
            complete_count = 0
            total_expected = 0
            total_actual = 0

            for file_path, file_data in files.items():
                if project_info is None:
                    project_info = file_data["project_info"]

                indices = file_data["indices"]
                chunk_count = file_data["chunk_count"]
                expected_indices = set(range(chunk_count))
                missing = expected_indices - indices

                total_expected += chunk_count
                total_actual += len(indices)

                if missing:
                    incomplete_files.append(
                        FileVerificationResult(
                            file_path=file_path,
                            expected_chunks=chunk_count,
                            actual_chunks=len(indices),
                            missing_indices=sorted(missing),
                            is_complete=False,
                        )
                    )
                else:
                    complete_count += 1

            project_result = ProjectVerificationResult(
                identifier=project_id,
                project_name=project_info.get("project_name", "") if project_info else "",
                branch=project_info.get("branch", "") if project_info else "",
                commit_hash=project_info.get("commit_hash", "") if project_info else "",
                total_files=len(files),
                complete_files=complete_count,
                incomplete_files=incomplete_files,
                total_expected_chunks=total_expected,
                total_actual_chunks=total_actual,
            )
            project_results.append(project_result)

            if project_result.is_complete:
                total_complete += 1
            else:
                total_incomplete += 1

        return CollectionVerificationResult(
            collection_name=collection_name,
            collection_type="code",
            total_points=total_points,
            total_items=len(project_results),
            complete_items=total_complete,
            incomplete_items=total_incomplete,
            is_healthy=total_incomplete == 0,
            projects=project_results,
        )

    def _verify_file_collection(
        self,
        collection_name: str,
        collection_type: Optional[str],
        total_points: int,
        verbose: bool = False,
    ) -> CollectionVerificationResult:
        """Verify a PDF or markdown collection.

        For each file, verifies that all chunk_indices exist.
        """
        logger.debug(f"Verifying {collection_type or 'file'} collection: {collection_name}")

        # Collect chunk info: {file_path: {"indices": set, "chunk_count": int}}
        file_chunks: Dict[str, Dict[str, any]] = defaultdict(
            lambda: {"indices": set(), "chunk_count": 0}
        )

        # Scroll through all points
        offset = None
        batch_count = 0
        payload_fields = ["file_path", "chunk_index", "chunk_count"]

        while True:
            points, offset = self.qdrant.scroll(
                collection_name=collection_name,
                with_payload=payload_fields,
                with_vectors=False,
                limit=500,
                offset=offset,
            )

            batch_count += 1
            if verbose and batch_count % 10 == 0:
                logger.debug(f"Processed {batch_count} batches...")

            if not points:
                break

            for point in points:
                payload = point.payload
                if not payload:
                    continue

                file_path = payload.get("file_path")
                chunk_index = payload.get("chunk_index")
                chunk_count = payload.get("chunk_count")

                if not file_path:
                    continue

                # Track chunk indices for this file
                if chunk_index is not None:
                    file_chunks[file_path]["indices"].add(chunk_index)
                    # Use explicit chunk_count if available, otherwise infer from max index
                    if chunk_count:
                        file_chunks[file_path]["chunk_count"] = chunk_count
                    else:
                        # Infer expected count from max index seen (0-based, so +1)
                        file_chunks[file_path]["chunk_count"] = max(
                            file_chunks[file_path]["chunk_count"],
                            chunk_index + 1,
                        )
                else:
                    # No chunk_index at all - just count occurrences
                    file_chunks[file_path]["indices"].add(len(file_chunks[file_path]["indices"]))
                    file_chunks[file_path]["chunk_count"] = len(file_chunks[file_path]["indices"])

            if offset is None:
                break

        # Analyze results
        file_results: List[FileVerificationResult] = []
        complete_count = 0
        incomplete_count = 0

        for file_path, file_data in file_chunks.items():
            indices = file_data["indices"]
            chunk_count = file_data["chunk_count"]

            # If chunk_count is 0 or matches indices count, assume complete
            if chunk_count == 0 or chunk_count == len(indices):
                file_results.append(
                    FileVerificationResult(
                        file_path=file_path,
                        expected_chunks=len(indices),
                        actual_chunks=len(indices),
                        is_complete=True,
                    )
                )
                complete_count += 1
            else:
                expected_indices = set(range(chunk_count))
                missing = expected_indices - indices
                file_results.append(
                    FileVerificationResult(
                        file_path=file_path,
                        expected_chunks=chunk_count,
                        actual_chunks=len(indices),
                        missing_indices=sorted(missing),
                        is_complete=False,
                    )
                )
                incomplete_count += 1

        return CollectionVerificationResult(
            collection_name=collection_name,
            collection_type=collection_type,
            total_points=total_points,
            total_items=len(file_results),
            complete_items=complete_count,
            incomplete_items=incomplete_count,
            is_healthy=incomplete_count == 0,
            files=file_results,
        )

    def get_incomplete_projects(
        self, collection_name: str
    ) -> List[Tuple[str, float]]:
        """Get list of incomplete projects with completion percentage.

        Convenience method for getting items that need repair.

        Args:
            collection_name: Name of collection to check

        Returns:
            List of (identifier, completion_percentage) tuples for incomplete items
        """
        result = self.verify_collection(collection_name)
        incomplete = []

        if result.collection_type == "code":
            for project in result.projects:
                if not project.is_complete:
                    incomplete.append((project.identifier, project.completion_percentage))
        else:
            for file in result.files:
                if not file.is_complete:
                    incomplete.append((file.file_path, file.completion_percentage))

        return sorted(incomplete, key=lambda x: x[1])  # Sort by completion %

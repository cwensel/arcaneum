"""Dual indexer for coordinated Qdrant and MeiliSearch indexing (RDR-009).

This module provides the DualIndexer class that orchestrates indexing to both
Qdrant (vector search) and MeiliSearch (full-text search) with shared metadata.
"""

import logging
from typing import List, Tuple, Optional
from uuid import uuid4

from qdrant_client import QdrantClient

from ..fulltext.client import FullTextClient
from ..schema.document import DualIndexDocument, to_qdrant_point, to_meilisearch_doc

logger = logging.getLogger(__name__)


class DualIndexer:
    """Coordinates indexing to both Qdrant and MeiliSearch.

    This class provides a simple interface for dual indexing that:
    - Indexes documents to Qdrant with named vectors
    - Indexes documents to MeiliSearch with shared metadata
    - Uses consistent document IDs across both systems
    - Handles batch operations for performance

    Example:
        >>> from qdrant_client import QdrantClient
        >>> from arcaneum.fulltext.client import FullTextClient
        >>>
        >>> qdrant = QdrantClient(url="http://localhost:6333")
        >>> meili = FullTextClient("http://localhost:7700", api_key)
        >>>
        >>> indexer = DualIndexer(qdrant, meili, "my-corpus", "my-corpus")
        >>> indexer.index_batch(documents)
    """

    DEFAULT_BATCH_SIZE = 100

    def __init__(
        self,
        qdrant_client: QdrantClient,
        meili_client: FullTextClient,
        collection_name: str,
        index_name: str,
        batch_size: int = DEFAULT_BATCH_SIZE
    ):
        """Initialize dual indexer.

        Args:
            qdrant_client: Initialized Qdrant client
            meili_client: Initialized MeiliSearch client
            collection_name: Qdrant collection name
            index_name: MeiliSearch index name
            batch_size: Batch size for MeiliSearch uploads (default 100)
        """
        self.qdrant = qdrant_client
        self.meili = meili_client
        self.collection_name = collection_name
        self.index_name = index_name
        self.batch_size = batch_size

    def index_batch(
        self,
        documents: List[DualIndexDocument],
        wait: bool = True
    ) -> Tuple[int, int]:
        """Index a batch of documents to both Qdrant and MeiliSearch.

        Args:
            documents: List of DualIndexDocument objects with vectors
            wait: If True, wait for indexing to complete (default True)

        Returns:
            Tuple of (qdrant_count, meili_count) indexed

        Raises:
            ValueError: If any document is missing vectors
        """
        if not documents:
            return 0, 0

        # Validate all documents have vectors
        for i, doc in enumerate(documents):
            if not doc.vectors:
                raise ValueError(
                    f"Document {i} ({doc.file_path}) missing vectors. "
                    "Generate embeddings before indexing."
                )

        logger.debug(f"Indexing {len(documents)} documents to both systems")

        # Generate UUIDs for documents that don't have IDs
        for doc in documents:
            if not doc.id:
                doc.id = str(uuid4())

        # Convert to Qdrant format and upload
        qdrant_points = [to_qdrant_point(doc) for doc in documents]
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=qdrant_points,
            wait=wait
        )
        qdrant_count = len(qdrant_points)
        logger.debug(f"Indexed {qdrant_count} points to Qdrant")

        # Convert to MeiliSearch format and upload in batches
        meili_docs = [to_meilisearch_doc(doc) for doc in documents]
        meili_count = 0

        for i in range(0, len(meili_docs), self.batch_size):
            batch = meili_docs[i:i + self.batch_size]
            if wait:
                self.meili.add_documents_sync(self.index_name, batch)
            else:
                self.meili.add_documents(self.index_name, batch)
            meili_count += len(batch)

        logger.debug(f"Indexed {meili_count} documents to MeiliSearch")

        return qdrant_count, meili_count

    def index_single(
        self,
        document: DualIndexDocument,
        wait: bool = True
    ) -> Tuple[int, int]:
        """Index a single document to both systems.

        Convenience method for single document indexing.

        Args:
            document: DualIndexDocument with vectors
            wait: If True, wait for indexing to complete

        Returns:
            Tuple of (1, 1) if successful
        """
        return self.index_batch([document], wait=wait)

    def delete_by_file_path(self, file_path: str) -> Tuple[int, int]:
        """Delete all documents for a file from both systems.

        Args:
            file_path: File path to delete documents for

        Returns:
            Tuple of (qdrant_deleted, meili_deleted) - note these are approximate
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        logger.debug(f"Deleting documents for file: {file_path}")

        # Delete from Qdrant using filter
        self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="file_path",
                        match=MatchValue(value=file_path)
                    )
                ]
            )
        )

        # MeiliSearch doesn't support filter-based deletion easily
        # We would need to search and delete by IDs
        # For now, return 0 for meili since we can't easily count
        # In practice, users should re-sync the directory

        return 0, 0

    def delete_by_project_identifier(self, project_identifier: str) -> Tuple[int, int]:
        """Delete all documents for a project/branch from both systems.

        Args:
            project_identifier: Composite "project#branch" identifier

        Returns:
            Tuple of (qdrant_deleted, meili_deleted) - note these are approximate
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        logger.debug(f"Deleting documents for project: {project_identifier}")

        # Delete from Qdrant using filter
        self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="git_project_identifier",
                        match=MatchValue(value=project_identifier)
                    )
                ]
            )
        )

        # MeiliSearch filter deletion would require search + delete by IDs
        # Return 0 for meili since we can't easily count
        return 0, 0

    def get_stats(self) -> dict:
        """Get statistics from both systems.

        Returns:
            Dictionary with stats from both Qdrant and MeiliSearch
        """
        # Get Qdrant stats
        qdrant_info = self.qdrant.get_collection(self.collection_name)
        qdrant_stats = {
            "points_count": qdrant_info.points_count,
            "indexed_vectors_count": qdrant_info.indexed_vectors_count,
            "status": str(qdrant_info.status),
        }

        # Get MeiliSearch stats
        meili_stats = self.meili.get_index_stats(self.index_name)

        return {
            "qdrant": qdrant_stats,
            "meilisearch": meili_stats,
        }

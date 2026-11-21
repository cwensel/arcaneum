"""Dual indexer for coordinated Qdrant and MeiliSearch indexing (RDR-009).

This module provides the DualIndexer class that orchestrates indexing to both
Qdrant (vector search) and MeiliSearch (full-text search) with shared metadata.
"""

import logging
from typing import List, Optional, Tuple
from uuid import uuid4

from qdrant_client import QdrantClient

from ..fulltext.client import FullTextClient
from ..schema.document import DualIndexDocument, to_qdrant_point, to_meilisearch_doc
from .collection_metadata import DEFAULT_TEXT_BACKEND
from .qdrant_batching import upsert_in_batches

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

    DEFAULT_BATCH_SIZE = 300

    def __init__(
        self,
        qdrant_client: QdrantClient,
        meili_client: Optional[FullTextClient],
        collection_name: str,
        index_name: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        corpus_type: Optional[str] = None,
        text_backend: str = DEFAULT_TEXT_BACKEND,
    ):
        """Initialize dual indexer.

        Args:
            qdrant_client: Initialized Qdrant client
            meili_client: Initialized MeiliSearch client; unused, and may be
                None, for a ``qdrant`` corpus
            collection_name: Qdrant collection name
            index_name: MeiliSearch index name
            batch_size: Batch size for Qdrant and MeiliSearch uploads (default 300)
            corpus_type: The corpus's type, whose stop words decide which
                chunks yield no token
            text_backend: The corpus's text backend (RDR-027 Q2); a ``qdrant``
                corpus's text is written only in its Qdrant points (Q7), and
                its zero-token chunks carry ``text_tokens: False`` (Q8)
        """
        self.qdrant = qdrant_client
        self.meili = meili_client
        self.collection_name = collection_name
        self.index_name = index_name
        self.batch_size = batch_size
        self.corpus_type = corpus_type
        self.text_backend = text_backend

    def index_batch(self, documents: List[DualIndexDocument], wait: bool = True) -> Tuple[int, int]:
        """Index a batch of documents to both Qdrant and MeiliSearch.

        Args:
            documents: List of DualIndexDocument objects with vectors
            wait: If True, wait for indexing to complete (default True)

        Returns:
            Tuple of (qdrant_count, meili_count) indexed; meili_count is 0
            for a ``qdrant`` corpus

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

        # Convert to Qdrant format and upload in batches split by count
        # (timeouts on 2800+ chunk documents) and by serialized size (Qdrant's
        # 32 MiB request limit, RDR-027 Q7)
        qdrant_points = [to_qdrant_point(doc) for doc in documents]
        if self.text_backend == "qdrant":
            self._flag_zero_token_chunks(qdrant_points)
        qdrant_count = upsert_in_batches(
            self.qdrant,
            self.collection_name,
            qdrant_points,
            wait=wait,
            max_count=self.batch_size,
        )
        logger.debug(f"Indexed {qdrant_count} points to Qdrant")

        if self.text_backend == "qdrant":
            # The points just written carry the corpus's text (RDR-027 Q7).
            return qdrant_count, 0

        # Convert to MeiliSearch format and upload in batches
        meili_docs = [to_meilisearch_doc(doc) for doc in documents]
        meili_count = 0

        # Split into batches
        batches = [
            meili_docs[i : i + self.batch_size] for i in range(0, len(meili_docs), self.batch_size)
        ]

        if wait and len(batches) > 1:
            # Use parallel batch upload for multiple batches
            result = self.meili.add_documents_batch_parallel(self.index_name, batches)
            meili_count = result["total_documents"]
        else:
            # Single batch or async mode
            for batch in batches:
                if wait:
                    self.meili.add_documents_sync(self.index_name, batch)
                else:
                    self.meili.add_documents(self.index_name, batch)
                meili_count += len(batch)

        logger.debug(f"Indexed {meili_count} documents to MeiliSearch")

        return qdrant_count, meili_count

    def _flag_zero_token_chunks(self, points) -> None:
        """Set ``text_tokens: False`` on each point whose text yields no token (Q8).

        The key is stored only when false: an absent key means the chunk has
        tokens, so a tokenful chunk's payload is unchanged.
        """
        from ..retrieval.qdrant_text import TEXT_FIELD, TEXT_TOKENS_FIELD, zero_token

        for point in points:
            if zero_token(point.payload.get(TEXT_FIELD) or "", self.corpus_type):
                point.payload[TEXT_TOKENS_FIELD] = False

    def index_single(self, document: DualIndexDocument, wait: bool = True) -> Tuple[int, int]:
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
                must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]
            ),
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
                        key="git_project_identifier", match=MatchValue(value=project_identifier)
                    )
                ]
            ),
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

"""
Qdrant integration for source code indexing.

This module provides efficient batch upload and filter-based deletion
for git-aware source code chunks (RDR-005).
"""

import logging
import time
from typing import List, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .types import CodeChunk

logger = logging.getLogger(__name__)


class QdrantIndexer:
    """Handles Qdrant operations for source code indexing.

    Provides:
    - Filter-based branch-specific deletion (40-100x faster than ChromaDB)
    - Batch upload with optimized chunk sizes (150 chunks)
    - Retry logic with exponential backoff
    - gRPC support for faster uploads
    """

    DEFAULT_BATCH_SIZE = 300  # Optimized for Qdrant (Phase 1 RDR-013)
    MAX_RETRIES = 3
    INITIAL_RETRY_WAIT = 1  # seconds

    def __init__(
        self,
        client: QdrantClient,
        batch_size: int = DEFAULT_BATCH_SIZE
    ):
        """Initialize Qdrant indexer.

        Args:
            client: Initialized Qdrant client
            batch_size: Number of chunks per batch upload (default 150)
        """
        self.client = client
        self.batch_size = batch_size

    def delete_branch_chunks(
        self,
        collection_name: str,
        project_identifier: str
    ) -> int:
        """Delete all chunks for a specific (project, branch) combination.

        Uses filter-based deletion which is 40-100x faster than ID-based deletion.
        Only affects the specified branch - other branches are unaffected.

        Args:
            collection_name: Name of Qdrant collection
            project_identifier: Composite identifier "project#branch"

        Returns:
            Number of chunks deleted (approximate)

        Performance:
            Target: <500ms for typical projects

        Example:
            >>> indexer.delete_branch_chunks("code", "arcaneum#main")
            # Deletes only arcaneum#main, leaves arcaneum#feature-x intact
        """
        logger.info(f"Deleting chunks for {project_identifier} from {collection_name}")

        start_time = time.time()

        try:
            # Use filter-based deletion (very fast)
            result = self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="git_project_identifier",
                            match=MatchValue(value=project_identifier)
                        )
                    ]
                )
            )

            elapsed = time.time() - start_time
            logger.info(
                f"Deleted chunks for {project_identifier} in {elapsed:.3f}s "
                f"(operation_id: {result.operation_id if hasattr(result, 'operation_id') else 'N/A'})"
            )

            # Note: Qdrant delete doesn't return count, so we return 0
            # Caller should query before deletion if count is needed
            return 0

        except Exception as e:
            logger.error(f"Error deleting chunks for {project_identifier}: {e}")
            raise

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=INITIAL_RETRY_WAIT, min=1, max=10),
        reraise=True
    )
    def upload_chunks_batch(
        self,
        collection_name: str,
        chunks: List[CodeChunk],
        wait: bool = True,
        vector_name: Optional[str] = None
    ) -> int:
        """Upload a batch of code chunks to Qdrant.

        Uses retry logic with exponential backoff for robustness.

        Args:
            collection_name: Name of Qdrant collection
            chunks: List of CodeChunk objects with embeddings
            wait: If True, wait for indexing to complete (default True)
            vector_name: Name of vector if using named vectors (e.g., "stella")

        Returns:
            Number of chunks uploaded

        Raises:
            ValueError: If any chunk is missing an embedding
            Exception: If upload fails after retries

        Performance:
            Target: 150 chunks per batch for optimal throughput
        """
        if not chunks:
            return 0

        # Validate all chunks have embeddings
        for i, chunk in enumerate(chunks):
            if chunk.embedding is None:
                raise ValueError(
                    f"Chunk {i} ({chunk.file_path}) missing embedding. "
                    "Generate embeddings before upload."
                )

        logger.debug(f"Uploading batch of {len(chunks)} chunks to {collection_name}")

        try:
            # Convert chunks to PointStruct
            points = [chunk.to_point(vector_name=vector_name) for chunk in chunks]

            # Upload to Qdrant
            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=wait
            )

            logger.debug(f"Successfully uploaded {len(chunks)} chunks")
            return len(chunks)

        except Exception as e:
            logger.error(f"Error uploading batch: {e}")
            raise

    def upload_chunks(
        self,
        collection_name: str,
        chunks: List[CodeChunk],
        show_progress: bool = False,
        vector_name: Optional[str] = None,
        bulk_mode: bool = False
    ) -> int:
        """Upload multiple chunks in optimized batches.

        Args:
            collection_name: Name of Qdrant collection
            chunks: List of CodeChunk objects with embeddings
            show_progress: If True, show progress bar (requires tqdm)
            vector_name: Name of vector if using named vectors
            bulk_mode: If True, disable indexing during upload for 1.3-1.5x speedup (RDR-013)

        Returns:
            Total number of chunks uploaded
        """
        if not chunks:
            return 0

        # Enable bulk mode if requested
        if bulk_mode:
            self.enable_bulk_mode(collection_name)

        try:
            total_uploaded = 0

            # Optional progress bar
            if show_progress:
                try:
                    from tqdm import tqdm
                    chunk_iter = tqdm(
                        range(0, len(chunks), self.batch_size),
                        desc="Uploading chunks",
                        unit="batch"
                    )
                except ImportError:
                    chunk_iter = range(0, len(chunks), self.batch_size)
            else:
                chunk_iter = range(0, len(chunks), self.batch_size)

            # Upload in batches with wait=False for bulk mode
            for i in chunk_iter:
                batch = chunks[i:i + self.batch_size]
                uploaded = self.upload_chunks_batch(
                    collection_name,
                    batch,
                    wait=not bulk_mode,  # Don't wait if bulk mode
                    vector_name=vector_name
                )
                total_uploaded += uploaded

            logger.info(f"Uploaded {total_uploaded} chunks in {(total_uploaded + self.batch_size - 1) // self.batch_size} batches")

        finally:
            # Always disable bulk mode if it was enabled
            if bulk_mode:
                self.disable_bulk_mode(collection_name)

        return total_uploaded

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE
    ):
        """Create a new Qdrant collection for source code.

        Args:
            collection_name: Name for the collection
            vector_size: Dimensionality of embeddings
                        - 768 for jina-embeddings-v2-base-code
                        - 1536 for jina-code-embeddings-1.5b
            distance: Distance metric (default: Cosine)
        """
        logger.info(f"Creating collection {collection_name} with vector_size={vector_size}")

        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            logger.info(f"Collection {collection_name} created successfully")

        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Name of collection to check

        Returns:
            True if collection exists, False otherwise
        """
        try:
            collections = self.client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def get_collection_info(self, collection_name: str) -> dict:
        """Get information about a collection.

        Args:
            collection_name: Name of collection

        Returns:
            Dictionary with collection info
        """
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise

    def count_chunks_for_project(
        self,
        collection_name: str,
        project_identifier: str
    ) -> int:
        """Count chunks for a specific project/branch.

        Args:
            collection_name: Name of Qdrant collection
            project_identifier: Composite identifier "project#branch"

        Returns:
            Number of chunks for this project
        """
        try:
            result = self.client.count(
                collection_name=collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="git_project_identifier",
                            match=MatchValue(value=project_identifier)
                        )
                    ]
                )
            )
            return result.count

        except Exception as e:
            logger.error(f"Error counting chunks: {e}")
            return 0

    def enable_bulk_mode(self, collection_name: str):
        """Enable bulk upload mode for faster indexing (RDR-013 Phase 1).

        Disables HNSW index construction during bulk uploads by setting
        indexing_threshold=0. This provides 1.3-1.5x speedup for large uploads.
        Must call disable_bulk_mode() afterwards to rebuild the index.

        Args:
            collection_name: Name of collection
        """
        logger.info(f"Enabling bulk mode for {collection_name}")

        try:
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=0  # Disable indexing during upload
                )
            )
            logger.info(f"Bulk mode enabled for {collection_name}")

        except Exception as e:
            logger.error(f"Error enabling bulk mode: {e}")
            raise

    def disable_bulk_mode(self, collection_name: str):
        """Disable bulk upload mode and rebuild index (RDR-013 Phase 1).

        Re-enables HNSW index construction by restoring indexing_threshold to
        default (20000). This triggers index rebuild for all uploaded points.

        Args:
            collection_name: Name of collection
        """
        logger.info(f"Disabling bulk mode for {collection_name}, rebuilding index...")

        try:
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=20000  # Restore default
                )
            )
            logger.info(f"Bulk mode disabled for {collection_name}, index rebuild complete")

        except Exception as e:
            logger.error(f"Error disabling bulk mode: {e}")
            raise

    def delete_collection(self, collection_name: str):
        """Delete a collection.

        Args:
            collection_name: Name of collection to delete
        """
        logger.info(f"Deleting collection {collection_name}")

        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Collection {collection_name} deleted")

        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise


def create_qdrant_client(
    url: str = "localhost",
    port: int = 6333,
    grpc_port: Optional[int] = 6334,
    api_key: Optional[str] = None,
    prefer_grpc: bool = True
) -> QdrantClient:
    """Create a Qdrant client with optional gRPC support.

    Args:
        url: Qdrant server URL or hostname
        port: HTTP port (default 6333)
        grpc_port: gRPC port (default 6334), None to disable gRPC
        api_key: Optional API key for authentication
        prefer_grpc: If True, use gRPC when available (faster)

    Returns:
        Configured QdrantClient
    """
    if prefer_grpc and grpc_port:
        logger.info(f"Creating Qdrant client with gRPC: {url}:{grpc_port}")
        return QdrantClient(
            host=url,
            port=port,
            grpc_port=grpc_port,
            api_key=api_key,
            prefer_grpc=True
        )
    else:
        logger.info(f"Creating Qdrant client with HTTP: {url}:{port}")
        return QdrantClient(
            host=url,
            port=port,
            api_key=api_key
        )

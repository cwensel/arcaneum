"""Core search functionality for Qdrant collections (RDR-007)."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from .embedder import SearchEmbedder

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Standardized search result format.

    Attributes:
        score: Similarity score (0.0 to 1.0, higher is better)
        collection: Name of collection this result came from
        location: File path or document location (formatted for display)
        content: Text content of the matched chunk
        metadata: Full payload metadata dictionary
        point_id: Qdrant point ID (for debugging)
    """
    score: float
    collection: str
    location: str
    content: str
    metadata: Dict[str, Any]
    point_id: str


def format_location(metadata: Dict[str, Any]) -> str:
    """Format location string for a search result.

    For source code: /path/to/file.py
    For PDFs: /path/to/file.pdf:page12

    Args:
        metadata: Result payload metadata

    Returns:
        Formatted location string
    """
    file_path = metadata.get("file_path", "")

    # PDF: Include page number if available
    if "page_number" in metadata:
        return f"{file_path}:page{metadata['page_number']}"

    # Source code or other: Just file path
    return file_path or f"[id:{metadata.get('id', '?')}]"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, OSError, TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def search_collection(
    client: QdrantClient,
    embedder: SearchEmbedder,
    query: str,
    collection_name: str,
    vector_name: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    query_filter: Optional[models.Filter] = None,
    score_threshold: Optional[float] = None
) -> List[SearchResult]:
    """Search single collection with semantic query.

    This is the core search function. It:
    1. Auto-detects the embedding model from collection metadata
    2. Generates a query embedding using that model
    3. Executes the semantic search with optional filters
    4. Returns formatted results

    Retries up to 3 times on connection/timeout errors with exponential backoff.

    Args:
        client: Qdrant client instance
        embedder: SearchEmbedder for query embeddings
        query: Search query text
        collection_name: Name of collection to search
        vector_name: Optional specific vector to use (auto-detects if None)
        limit: Maximum number of results to return
        offset: Number of results to skip (for pagination)
        query_filter: Optional metadata filter
        score_threshold: Optional minimum similarity score (0.0 to 1.0)

    Returns:
        List of SearchResult objects, sorted by relevance (highest score first)

    Raises:
        ValueError: If collection doesn't exist, has no vectors, or model not found
        ConnectionError, OSError, TimeoutError: After 3 retry attempts
        Exception: For other Qdrant connection errors
    """
    # Step 1: Generate query embedding with auto-detected or specified model
    # This will validate the collection exists and has vectors
    model_key, query_vector = embedder.generate_query_embedding(
        query, collection_name, client, vector_name
    )

    # Step 2: Execute Qdrant search using query_points API (qdrant-client 1.16+)
    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=model_key,  # Named vector to search
            query_filter=query_filter,
            limit=limit,
            offset=offset,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False  # Don't return vectors (saves bandwidth)
        )
        results = response.points
    except Exception as e:
        raise Exception(f"Search failed: {e}")

    # Step 3: Convert Qdrant results to SearchResult format
    search_results = []
    for r in results:
        search_results.append(
            SearchResult(
                score=r.score,
                collection=collection_name,
                location=format_location(r.payload),
                content=r.payload.get("content", r.payload.get("text", "")),
                metadata=r.payload,
                point_id=str(r.id)
            )
        )

    return search_results


def explain_search(
    client: QdrantClient,
    embedder: SearchEmbedder,
    query: str,
    collection_name: str,
    vector_name: Optional[str] = None
) -> Dict[str, Any]:
    """Explain how a search would be executed (without running it).

    Useful for debugging and understanding which model/vector will be used.

    Args:
        client: Qdrant client instance
        embedder: SearchEmbedder for model detection
        query: Search query text
        collection_name: Name of collection
        vector_name: Optional specific vector name

    Returns:
        Dictionary with execution plan details:
            - collection: Collection name
            - query: Query text
            - model_key: Which model will be used
            - vector_name: Which named vector will be searched
            - dimensions: Embedding dimensions
            - available_vectors: All vectors in collection
    """
    # Detect model
    model_key = embedder.detect_collection_model(client, collection_name, vector_name)

    # Get collection info
    collection_info = client.get_collection(collection_name)
    available_vectors = list(collection_info.config.params.vectors.keys())

    # Get dimensions
    dimensions = embedder.get_model_dimensions(model_key)

    return {
        "collection": collection_name,
        "query": query,
        "model_key": model_key,
        "vector_name": model_key,  # Same as model_key in our architecture
        "dimensions": dimensions,
        "available_vectors": available_vectors,
        "vector_count": collection_info.vectors_count,
        "points_count": collection_info.points_count
    }

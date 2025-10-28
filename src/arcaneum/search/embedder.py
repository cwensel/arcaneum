"""Query embedding pipeline for semantic search (RDR-007)."""

from functools import lru_cache
from typing import Tuple, List
from pathlib import Path
from qdrant_client import QdrantClient

from ..embeddings.client import EmbeddingClient, EMBEDDING_MODELS


class SearchEmbedder:
    """Manages embedding models for search queries.

    Automatically detects which embedding model a collection uses by inspecting
    its named vectors, then generates query embeddings using that model.
    """

    def __init__(self, cache_dir: Path, verify_ssl: bool = True):
        """Initialize search embedder.

        Args:
            cache_dir: Directory to cache downloaded models
            verify_ssl: Whether to verify SSL certificates
        """
        self.cache_dir = cache_dir
        self.verify_ssl = verify_ssl
        self._embedding_client = EmbeddingClient(
            cache_dir=str(cache_dir),
            verify_ssl=verify_ssl
        )

    @staticmethod
    def detect_collection_model(
        client: QdrantClient,
        collection_name: str,
        vector_name: str = None
    ) -> str:
        """Detect which embedding model a collection uses.

        Collections use named vectors where the vector name IS the model key
        (e.g., "stella", "jina", "modernbert", "bge").

        Args:
            client: Qdrant client instance
            collection_name: Name of collection to inspect
            vector_name: Optional specific vector name to validate

        Returns:
            Model key (e.g., "stella", "jina-code")

        Raises:
            ValueError: If collection has no vectors or vector_name not found
        """
        collection_info = client.get_collection(collection_name)
        available_vectors = list(collection_info.config.params.vectors.keys())

        if not available_vectors:
            raise ValueError(f"Collection '{collection_name}' has no vectors")

        if vector_name:
            # User specified - validate it exists
            if vector_name not in available_vectors:
                available = ", ".join(available_vectors)
                raise ValueError(
                    f"Vector '{vector_name}' not found in collection.\n"
                    f"Available vectors: {available}"
                )
            return vector_name

        # Auto-select first vector (alphabetically for consistency)
        return sorted(available_vectors)[0]

    def generate_query_embedding(
        self,
        query: str,
        collection_name: str,
        client: QdrantClient,
        vector_name: str = None
    ) -> Tuple[str, List[float]]:
        """Generate query embedding with auto-detected or specified model.

        Args:
            query: Search query text
            collection_name: Name of collection to search
            client: Qdrant client instance
            vector_name: Optional vector name to use (auto-detects if not specified)

        Returns:
            Tuple of (model_key, query_vector) where:
                - model_key: The model identifier used (e.g., "stella")
                - query_vector: List of floats representing the embedding

        Raises:
            ValueError: If model detection fails or model not configured
        """
        # Detect which model to use from collection's vector configuration
        model_key = self.detect_collection_model(client, collection_name, vector_name)

        # Validate model is configured
        if model_key not in EMBEDDING_MODELS:
            available = ", ".join(EMBEDDING_MODELS.keys())
            raise ValueError(
                f"Model '{model_key}' not configured.\n"
                f"Available models: {available}"
            )

        # Generate embedding using the detected model
        # Note: query_embed() is for queries, embed() is for documents
        model = self._embedding_client.get_model(model_key)

        # Handle different backends (fastembed vs sentence-transformers)
        backend = EMBEDDING_MODELS[model_key].get("backend", "fastembed")

        if backend == "fastembed":
            # FastEmbed: Use query_embed for queries
            query_vector = list(model.query_embed([query]))[0]
            return (model_key, query_vector.tolist())
        elif backend == "sentence-transformers":
            # SentenceTransformers: Use encode (no separate query_embed)
            query_vector = model.encode([query], convert_to_numpy=True)[0]
            return (model_key, query_vector.tolist())
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def get_model_dimensions(self, model_key: str) -> int:
        """Get embedding dimensions for a model.

        Args:
            model_key: Model identifier (e.g., "stella", "jina-code")

        Returns:
            Number of dimensions in embedding vectors

        Raises:
            ValueError: If model not configured
        """
        if model_key not in EMBEDDING_MODELS:
            available = ", ".join(EMBEDDING_MODELS.keys())
            raise ValueError(
                f"Model '{model_key}' not configured.\n"
                f"Available models: {available}"
            )
        return EMBEDDING_MODELS[model_key]["dimensions"]

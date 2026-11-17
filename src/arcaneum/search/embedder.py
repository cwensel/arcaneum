"""Query embedding pipeline for semantic search (RDR-007)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

from ..embeddings.client import EMBEDDING_MODELS, EmbeddingClient
from ..indexing.collection_metadata import (
    backfill_embedding_prompt_policy,
    get_collection_metadata,
    get_collection_type,
    prompt_policy_can_be_backfilled,
    prompt_policy_issues,
)

if TYPE_CHECKING:
    # qdrant_client is imported where it is used, keeping it off the `arc` start
    # path (RDR-026 Q11).
    from qdrant_client import QdrantClient


class ModelNotCachedError(Exception):
    """The query model is missing from the local cache and downloads are off."""

    def __init__(self, model_key: str):
        self.model_key = model_key
        super().__init__(
            f"Embedding model '{model_key}' is not in the local model cache and downloads "
            "are disabled; cache it with `arc models`, or use text search"
        )


class QueryEmbeddingError(Exception):
    """Loading the query model (including a download) or embedding the query failed.

    Kept apart from Qdrant failures, so an unreachable model host is not
    reported as an unreachable Qdrant.
    """

    def __init__(self, model_key: str, cause: BaseException):
        self.model_key = model_key
        super().__init__(f"Embedding model '{model_key}' could not embed the query: {cause}")


class SearchEmbedder:
    """Manages embedding models for search queries.

    Automatically detects which embedding model a collection uses by inspecting
    its named vectors, then generates query embeddings using that model.
    """

    # Class default, so an instance built without __init__ keeps download-on-use.
    _allow_download = True

    def __init__(self, cache_dir: Path, allow_download: bool = True):
        """Initialize search embedder.

        Args:
            cache_dir: Directory to cache downloaded models
            allow_download: When False, a model missing from the local cache
                raises ModelNotCachedError instead of downloading (RDR-026 Q9)
        """
        self._allow_download = allow_download
        self._embedding_client = EmbeddingClient(
            cache_dir=str(cache_dir),
        )

    @staticmethod
    def detect_collection_model(
        client: QdrantClient, collection_name: str, vector_name: str = None
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
        self, query: str, collection_name: str, client: QdrantClient, vector_name: str = None
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
            raise ValueError(f"Model '{model_key}' not configured.\nAvailable models: {available}")

        # Refuse before the policy check, whose backfill writes to Qdrant, and
        # before get_model, which downloads when the local load fails.
        if not self._allow_download and not self._embedding_client.is_model_cached(model_key):
            raise ModelNotCachedError(model_key)

        metadata = get_collection_metadata(client, collection_name)
        policy_issues = prompt_policy_issues(metadata, model_key)
        if policy_issues and prompt_policy_can_be_backfilled(metadata, [model_key]):
            collection_type = get_collection_type(client, collection_name) or "pdf"
            metadata = backfill_embedding_prompt_policy(
                client,
                collection_name,
                collection_type,
                model_key,
            )
            policy_issues = prompt_policy_issues(metadata, model_key)
        if policy_issues:
            raise ValueError(policy_issues[0])

        try:
            return self._embed_query(query, model_key)
        except Exception as e:
            raise QueryEmbeddingError(model_key, e) from e

    def _embed_query(self, query: str, model_key: str) -> Tuple[str, List[float]]:
        """Load the query model (downloading it if allowed) and embed ``query``."""
        # Generate embedding using the detected model
        # Note: query_embed() is for queries, embed() is for documents
        model = self._embedding_client.get_model(model_key)

        # Handle different backends (fastembed vs sentence-transformers)
        backend = EMBEDDING_MODELS[model_key].get("backend", "fastembed")

        if backend == "fastembed":
            # FastEmbed: Use query_embed for queries
            query_vector = list(model.query_embed([query]))[0]
            # Return numpy array directly - Qdrant accepts numpy arrays natively
            return (model_key, query_vector)
        elif backend == "sentence-transformers":
            # SentenceTransformers: use the same prompt-aware path as indexing,
            # with query semantics selected from the model registry.
            query_vector = self._embedding_client.embed(
                [query],
                model_key,
                prompt_type="query",
            )[0]
            # Return numpy array directly - Qdrant accepts numpy arrays natively (arcaneum-zfch)
            return (model_key, query_vector)
        else:
            raise ValueError(f"Unknown backend: {backend}")

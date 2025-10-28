"""
Collection metadata management for type enforcement.

This module provides utilities to store and validate collection types,
ensuring PDFs and source code are not mixed in the same collection.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import CollectionInfo

logger = logging.getLogger(__name__)

# Reserved UUID for collection metadata point
# Using a fixed UUID so we can always find it
METADATA_POINT_ID = "00000000-0000-0000-0000-000000000001"


class CollectionType:
    """Valid collection types."""
    PDF = "pdf"
    CODE = "code"

    @classmethod
    def values(cls):
        """Get all valid types."""
        return [cls.PDF, cls.CODE]

    @classmethod
    def validate(cls, collection_type: str):
        """Validate collection type is valid."""
        if collection_type not in cls.values():
            raise ValueError(
                f"Invalid collection type: {collection_type}. "
                f"Must be one of: {', '.join(cls.values())}"
            )


def set_collection_metadata(
    client: QdrantClient,
    collection_name: str,
    collection_type: str,
    model: str,
    **extra_metadata
) -> None:
    """Set collection-level metadata including type.

    Stores metadata as a special point in the collection with a reserved ID.

    Args:
        client: Qdrant client
        collection_name: Name of collection
        collection_type: Type of collection ("pdf" or "code")
        model: Embedding model name
        **extra_metadata: Additional metadata to store

    Raises:
        ValueError: If collection_type is invalid
    """
    CollectionType.validate(collection_type)

    metadata = {
        "collection_type": collection_type,
        "model": model,
        "created_at": datetime.now().isoformat(),
        "created_by": "arcaneum",
        "is_metadata": True,  # Flag to identify metadata point
        **extra_metadata
    }

    try:
        # Store metadata as a special point with reserved ID
        # We use a zero vector since this is not for search
        from qdrant_client.models import PointStruct

        # Get vector size from collection
        info = client.get_collection(collection_name)

        # Handle both single vector and named vectors
        if hasattr(info.config.params, 'vectors'):
            if isinstance(info.config.params.vectors, dict):
                # Named vectors - use first one
                vector_name = list(info.config.params.vectors.keys())[0]
                vector_size = info.config.params.vectors[vector_name].size
                vectors = {vector_name: [0.0] * vector_size}
            else:
                # Single unnamed vector
                vector_size = info.config.params.vectors.size
                vectors = [0.0] * vector_size
        else:
            raise ValueError("Could not determine vector configuration")

        # Upsert metadata point with reserved UUID
        metadata_point = PointStruct(
            id=METADATA_POINT_ID,
            vector=vectors,
            payload=metadata
        )

        client.upsert(
            collection_name=collection_name,
            points=[metadata_point]
        )

        logger.info(f"Set collection metadata for {collection_name}: type={collection_type}")

    except Exception as e:
        logger.error(f"Failed to set collection metadata: {e}")
        raise


def get_collection_metadata(
    client: QdrantClient,
    collection_name: str
) -> Dict[str, Any]:
    """Get collection-level metadata.

    Retrieves metadata from the special reserved point.

    Args:
        client: Qdrant client
        collection_name: Name of collection

    Returns:
        Dictionary of metadata, empty if none set

    Raises:
        Exception: If collection doesn't exist
    """
    try:
        # Retrieve the special metadata point
        points = client.retrieve(
            collection_name=collection_name,
            ids=[METADATA_POINT_ID],
            with_payload=True,
            with_vectors=False
        )

        if points and len(points) > 0:
            payload = points[0].payload
            # Return payload without the is_metadata flag
            return {k: v for k, v in payload.items() if k != "is_metadata"}

        return {}

    except Exception as e:
        # If retrieve fails, collection might not have metadata yet
        logger.debug(f"No metadata found for collection {collection_name}: {e}")
        return {}


def get_collection_type(
    client: QdrantClient,
    collection_name: str
) -> Optional[str]:
    """Get the type of a collection.

    Args:
        client: Qdrant client
        collection_name: Name of collection

    Returns:
        Collection type ("pdf" or "code"), or None if untyped

    Raises:
        Exception: If collection doesn't exist
    """
    try:
        metadata = get_collection_metadata(client, collection_name)
        return metadata.get("collection_type")

    except Exception as e:
        logger.error(f"Failed to get collection type: {e}")
        raise


def validate_collection_type(
    client: QdrantClient,
    collection_name: str,
    expected_type: str,
    allow_untyped: bool = True
) -> None:
    """Validate that collection type matches expected type.

    Args:
        client: Qdrant client
        collection_name: Name of collection
        expected_type: Expected type ("pdf" or "code")
        allow_untyped: If True, allow untyped collections with warning

    Raises:
        TypeError: If collection type doesn't match expected
        ValueError: If expected_type is invalid
    """
    CollectionType.validate(expected_type)

    actual_type = get_collection_type(client, collection_name)

    if actual_type is None:
        # Untyped collection
        if allow_untyped:
            logger.warning(
                f"Collection '{collection_name}' has no type. "
                f"Allowing {expected_type} indexing. "
                "Consider recreating with --type flag."
            )
            return
        else:
            raise TypeError(
                f"Collection '{collection_name}' has no type set. "
                "Create collection with --type flag."
            )

    if actual_type != expected_type:
        raise TypeError(
            f"Collection '{collection_name}' is type '{actual_type}', "
            f"cannot index {expected_type} content. "
            f"Create a new collection with --type {expected_type}."
        )

    logger.debug(f"Collection '{collection_name}' type validated: {actual_type}")


def is_typed_collection(
    client: QdrantClient,
    collection_name: str
) -> bool:
    """Check if collection has a type set.

    Args:
        client: Qdrant client
        collection_name: Name of collection

    Returns:
        True if collection has type metadata, False otherwise
    """
    try:
        collection_type = get_collection_type(client, collection_name)
        return collection_type is not None
    except Exception:
        return False


def get_vector_names(
    client: QdrantClient,
    collection_name: str
) -> list:
    """Get list of vector names in collection.

    Args:
        client: Qdrant client
        collection_name: Name of collection

    Returns:
        List of vector names (e.g., ["stella", "bge"])
        Empty list if collection has no named vectors
    """
    try:
        info = client.get_collection(collection_name)

        if hasattr(info.config.params, 'vectors'):
            if isinstance(info.config.params.vectors, dict):
                return list(info.config.params.vectors.keys())

        return []

    except Exception as e:
        logger.error(f"Failed to get vector names: {e}")
        return []

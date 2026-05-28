"""
Collection metadata management for type enforcement.

This module provides utilities to store and validate collection types,
ensuring PDFs, source code, and markdown are not mixed in the same collection.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Condition, FieldCondition, Filter, MatchValue

logger = logging.getLogger(__name__)

# Reserved UUID for collection metadata point
# Using a fixed UUID so we can always find it
METADATA_POINT_ID = "00000000-0000-0000-0000-000000000001"
METADATA_PAYLOAD_KEY = "is_metadata"


def _condition_list(
    conditions: Optional[Condition | list[Condition]],
) -> list[Condition]:
    if conditions is None:
        return []
    if isinstance(conditions, list):
        return list(conditions)
    return [conditions]


def metadata_exclusion_filter(query_filter: Optional[Filter] = None) -> Filter:
    """Return a Qdrant filter that excludes reserved metadata points."""
    metadata_condition = FieldCondition(
        key=METADATA_PAYLOAD_KEY,
        match=MatchValue(value=True),
    )

    if query_filter is None:
        return Filter(must_not=[metadata_condition])

    must_not = _condition_list(query_filter.must_not)
    if metadata_condition not in must_not:
        must_not.append(metadata_condition)

    return Filter(
        must=query_filter.must,
        should=query_filter.should,
        must_not=must_not,
        min_should=query_filter.min_should,
    )


class CollectionType:
    """Valid collection types."""
    PDF = "pdf"
    CODE = "code"
    MARKDOWN = "markdown"

    @classmethod
    def values(cls):
        """Get all valid types."""
        return [cls.PDF, cls.CODE, cls.MARKDOWN]

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
        collection_type: Type of collection ("pdf", "code", or "markdown")
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
        METADATA_PAYLOAD_KEY: True,  # Flag to identify metadata point
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


def update_collection_metadata(
    client: QdrantClient,
    collection_name: str,
    **updates
) -> Dict[str, Any]:
    """Update collection metadata in place without touching indexed documents.

    Args:
        client: Qdrant client
        collection_name: Name of collection
        **updates: Metadata keys to update. Values of None remove the key.

    Returns:
        Updated metadata payload without the internal metadata flag.

    Raises:
        ValueError: If the collection has no metadata point to update
    """
    existing = get_collection_metadata(client, collection_name)
    if not existing:
        raise ValueError(
            f"Collection '{collection_name}' has no metadata point to update"
        )

    metadata = {**existing}
    for key, value in updates.items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value

    collection_type = metadata.get("collection_type")
    if collection_type is not None:
        CollectionType.validate(collection_type)

    try:
        from qdrant_client.models import PointStruct

        info = client.get_collection(collection_name)

        if hasattr(info.config.params, 'vectors'):
            if isinstance(info.config.params.vectors, dict):
                vector_name = list(info.config.params.vectors.keys())[0]
                vector_size = info.config.params.vectors[vector_name].size
                vectors = {vector_name: [0.0] * vector_size}
            else:
                vector_size = info.config.params.vectors.size
                vectors = [0.0] * vector_size
        else:
            raise ValueError("Could not determine vector configuration")

        client.upsert(
            collection_name=collection_name,
            points=[PointStruct(
                id=METADATA_POINT_ID,
                vector=vectors,
                payload={**metadata, METADATA_PAYLOAD_KEY: True},
            )],
        )
        logger.info(f"Updated collection metadata for {collection_name}")
        return metadata

    except Exception as e:
        logger.error(f"Failed to update collection metadata: {e}")
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
            return {k: v for k, v in payload.items() if k != METADATA_PAYLOAD_KEY}

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
        Collection type ("pdf", "code", or "markdown"), or None if untyped

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
        expected_type: Expected type ("pdf", "code", or "markdown")
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

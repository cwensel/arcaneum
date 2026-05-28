"""
Collection metadata management for type enforcement.

This module provides utilities to store and validate collection types,
ensuring PDFs, source code, and markdown are not mixed in the same collection.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Condition, FieldCondition, Filter, MatchValue

from arcaneum.embeddings.client import (
    get_embedding_prompt_policies,
    get_embedding_prompt_policy,
    model_key_for_name,
)
from arcaneum.schema.document import (
    PERSISTED_SCHEMA_VERSION,
    PERSISTED_SCHEMA_VERSION_FIELD,
    persisted_metadata_fields,
)

logger = logging.getLogger(__name__)

# Reserved UUID for collection metadata point
# Using a fixed UUID so we can always find it
METADATA_POINT_ID = "00000000-0000-0000-0000-000000000001"
METADATA_PAYLOAD_KEY = "is_metadata"


def persisted_schema_defaults() -> Dict[str, Any]:
    """Return metadata fields required for persisted collection compatibility."""
    return persisted_metadata_fields()


def persisted_schema_issues(metadata: Dict[str, Any]) -> list[str]:
    """Return compatibility issues for persisted collection metadata.

    Missing schema_version is legacy v0. Versions older than the current schema
    need repair/backfill; newer versions require a newer Arcaneum before use.
    """
    version = metadata.get(PERSISTED_SCHEMA_VERSION_FIELD)
    if version is None:
        return [
            "collection metadata is legacy schema v0; reindex or backfill "
            "schema_version/app_version before relying on persisted compatibility"
        ]

    if isinstance(version, bool):
        return [f"collection metadata has invalid schema_version {version!r}"]
    if isinstance(version, int):
        version_int = version
    elif isinstance(version, str) and version.isdecimal():
        version_int = int(version)
    else:
        return [f"collection metadata has invalid schema_version {version!r}"]

    if version_int < PERSISTED_SCHEMA_VERSION:
        return [
            f"collection metadata schema_version {version_int} is older than "
            f"supported v{PERSISTED_SCHEMA_VERSION}; reindex or run a backfill"
        ]
    if version_int > PERSISTED_SCHEMA_VERSION:
        return [
            f"collection metadata schema_version {version_int} is newer than "
            f"this Arcaneum supports (v{PERSISTED_SCHEMA_VERSION})"
        ]

    if not metadata.get("app_version"):
        return ["collection metadata is missing app_version"]
    return []


def prompt_policy_issues(metadata: Dict[str, Any], model_name: str) -> list[str]:
    """Return issues when stored prompt policy differs from current model policy."""
    model_key = model_key_for_name(model_name)
    if model_key is None:
        return [f"embedding model '{model_name}' is not configured"]

    stored_policies = metadata.get("embedding_prompt_policy")
    if not stored_policies:
        return [
            "collection metadata is missing embedding_prompt_policy; "
            "reindex before semantic search with prompt-aware models"
        ]

    stored_policy = stored_policies.get(model_key)
    current_policy = get_embedding_prompt_policy(model_key)
    if stored_policy is None:
        return [
            f"collection metadata is missing prompt policy for model '{model_key}'; "
            "reindex before semantic search"
        ]
    if stored_policy != current_policy:
        return [
            f"collection prompt policy for model '{model_key}' differs from "
            "the current embedding registry; reindex the corpus"
        ]
    return []


def should_stamp_prompt_policy(
    force: bool,
    file_list: Optional[list],
    stats: Dict[str, Any],
    orphans_remaining: int,
) -> bool:
    """Decide whether a run may (re)stamp the collection's prompt policy.

    Stamping certifies that EVERY vector in the collection was produced under
    the collection's recorded embedding prompt policy. It is therefore only
    valid for a force, full-directory run that succeeded cleanly and left no
    orphan vectors behind. Shared by all force paths (pdf/markdown/source CLI
    and the dual-index sync force path) so the gate is enforced identically.

    The stamp is allowed ONLY when ALL hold:
      - force is True (incremental runs never re-certify the whole collection)
      - file_list is None (a partial --file-list run never covers the corpus)
      - stats["errors"] == 0 (a failed file may have left stale chunks)
      - stats["files"] > 0 (something was actually indexed)
      - orphans_remaining == 0 (no indexed file is missing from disk)

    Callers that can perform scope-limited work pass ``covered_paths`` to
    :func:`prune_orphans_and_stamp`, which withholds certification when any
    still-existing indexed file was not actually processed by the run.

    Args:
        force: Whether the run used force/full reindex.
        file_list: Explicit file list for the run, or None for full-directory.
        stats: Run statistics dict with "files" and "errors" counts.
        orphans_remaining: Count of indexed files no longer on disk after the run.

    Returns:
        True if the run may stamp the prompt policy, False otherwise.
    """
    if not force:
        return False
    if file_list is not None:
        return False
    if stats.get("errors", 0) != 0:
        return False
    if stats.get("files", 0) <= 0:
        return False
    if orphans_remaining != 0:
        return False
    return True


def stamp_embedding_prompt_policy(
    qdrant: QdrantClient,
    collection_name: str,
    collection_type: str,
    model: str,
) -> Dict[str, Any]:
    """Write/refresh the collection's embedding_prompt_policy metadata.

    Records the current prompt policy for ``model`` (via
    :func:`get_embedding_prompt_policies`) onto the collection metadata point,
    certifying that the indexed vectors match the active embedding registry.
    Mirrors how create-time metadata records the policy. Only call this after
    :func:`should_stamp_prompt_policy` returns True.

    Args:
        qdrant: Qdrant client.
        collection_name: Name of the collection to stamp.
        collection_type: Collection type ("pdf", "code", or "markdown").
        model: Embedding model name (or names) backing the collection.

    Returns:
        The updated metadata payload (without the internal metadata flag).
    """
    CollectionType.validate(collection_type)
    policy = get_embedding_prompt_policies(model)

    metadata = _retrieve_collection_metadata(qdrant, collection_name)
    if metadata is None:
        set_collection_metadata(qdrant, collection_name, collection_type, model)
        logger.info(
            f"Initialized metadata point for legacy collection {collection_name} "
            f"before stamping embedding prompt policy (model={model})"
        )
        return get_collection_metadata(qdrant, collection_name)

    updated = update_collection_metadata(
        qdrant,
        collection_name,
        embedding_prompt_policy=policy,
    )
    logger.info(f"Stamped embedding prompt policy for {collection_name} (model={model})")
    return updated


class MultiRootPruneError(ValueError):
    """Raised when --prune is requested on a multi-root collection.

    Pruning is single-directory only: an orphan is an indexed file that no
    longer exists under the directory being indexed. When the collection's
    indexed paths span multiple directory trees, a single-directory force run
    cannot distinguish a genuine orphan from a file that lives under another
    indexed tree, so pruning would risk deleting still-on-disk files (job-1921).
    """


def _is_under(path: str, directory: str) -> bool:
    """Return True if ``path`` is the directory itself or lives under it.

    Uses absolute, normalized paths and ``os.path.commonpath`` so trailing
    slashes and ``..`` segments are handled robustly. Returns False when the
    paths share no common base (e.g. different drives on Windows) or on any
    path error.
    """
    try:
        p = os.path.abspath(path)
        d = os.path.abspath(directory)
        if p == d:
            return True
        return os.path.commonpath([p, d]) == d
    except (ValueError, TypeError):
        return False


def _orphan_chunks_remain(sync: Any, collection_name: str, file_path: str) -> bool:
    """Return True if chunks for ``file_path`` still exist after a prune attempt.

    Prefers the sync's state query (``has_chunks_for_file_path``). If the sync
    does not expose it, conservatively assume chunks remain so a path that could
    not be verified clean does not get counted as pruned (preserves the stamp's
    integrity guarantee).
    """
    checker = getattr(sync, "has_chunks_for_file_path", None)
    if checker is None:
        return True
    return bool(checker(collection_name, file_path))


def _collection_is_multi_root(pre_run_paths: set, indexed_dir: str) -> bool:
    """True if any indexed path lies outside the directory being indexed.

    A collection is "multi-root" relative to ``indexed_dir`` when its
    pre-run indexed paths include at least one file not under ``indexed_dir``.
    Such a collection spans multiple directory trees, so a single-directory
    force run must not treat the other trees' files as orphans.
    """
    return any(not _is_under(p, indexed_dir) for p in pre_run_paths)


def prune_orphans_and_stamp(
    qdrant: QdrantClient,
    sync: Any,
    collection_name: str,
    collection_type: str,
    model: str,
    force: bool,
    file_list: Optional[list],
    stats: Dict[str, Any],
    on_disk_paths: set,
    pre_run_paths: set,
    prune: bool,
    indexed_dir: Optional[str] = None,
    covered_paths: Optional[set] = None,
    warn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Shared post-indexing orphan handling + prompt-policy stamp for force runs.

    Orphans are indexed files that no longer exist on disk; their vectors may
    have been produced under a now-stale prompt policy. Orphan detection is only
    meaningful for a full-directory force run (``force`` true, ``file_list``
    None). When ``prune`` is set, orphan chunks are deleted by ``file_path``
    (the C1 primitive) so the collection can be certified. The prompt policy is
    stamped only when :func:`should_stamp_prompt_policy` allows it (no orphans
    remain). If orphans remain and ``prune`` is off, the stamp is skipped and a
    warning is emitted.

    Multi-root guard (job-1921 Fix A): when ``indexed_dir`` is provided (the
    single-directory CLIs — markdown, PDF — pass it), the collection is checked
    for paths outside that directory. Orphan detection compares a collection-
    wide pre-run set against a directory-scoped on-disk set, so on a collection
    spanning multiple trees the other trees' files look like orphans. To avoid
    deleting still-on-disk files:
      - ``prune`` set + multi-root → raise :class:`MultiRootPruneError`
        (delete nothing, stamp nothing).
      - bare force (no prune) + multi-root → skip the stamp and warn; do NOT
        classify the other directories' files as orphans.
      - single-root (or ``indexed_dir`` is None) → unchanged behavior.
    Paths that thread a collection-wide on-disk set (source, dual-index sync)
    are symmetric and pass ``indexed_dir=None`` to keep their behavior intact.

    Args:
        qdrant: Qdrant client.
        sync: MetadataBasedSync (provides delete_chunks_by_file_path).
        collection_name: Collection being indexed.
        collection_type: Collection type ("pdf", "code", "markdown").
        model: Embedding model name(s).
        force: Whether this was a force/full reindex run.
        file_list: Explicit file list, or None for full-directory runs.
        stats: Run statistics dict with "files" and "errors".
        on_disk_paths: Absolute path strings the run covered that exist on disk.
        pre_run_paths: Indexed file_path set captured BEFORE indexing.
        prune: Whether to delete orphan chunks.
        indexed_dir: Directory being indexed for single-directory CLIs; enables
            the multi-root guard. None disables it (collection-wide on-disk
            paths are already symmetric).
        covered_paths: Absolute path strings the run actually (re)embedded under
            the current prompt policy. When provided, the stamp is additionally
            withheld if any still-existing pre-run indexed path was NOT covered
            (a scope-limited run such as --no-recursive or a depth-limited source
            reindex leaves stale vectors that must not be certified). None
            disables the coverage gate (back-compat / callers that always cover
            the full collection).
        warn: Optional callable(str) used to surface the stale-policy warning.

    Returns:
        Dict with keys: orphans (list), orphans_pruned (int),
        orphans_remaining (int), stamped (bool).

    Raises:
        MultiRootPruneError: If ``prune`` is requested on a multi-root
            collection (single-directory CLIs only).
    """
    result = {
        "orphans": [],
        "orphans_pruned": 0,
        "orphans_remaining": 0,
        "stamped": False,
    }

    # Orphan detection only applies to full-directory force runs.
    if force and file_list is None:
        # Multi-root guard (markdown/PDF single-directory CLIs pass indexed_dir).
        if indexed_dir is not None and _collection_is_multi_root(pre_run_paths, indexed_dir):
            if prune:
                raise MultiRootPruneError(
                    f"Cannot --prune: collection {collection_name} contains "
                    f"indexed files outside {indexed_dir} (spans multiple "
                    "directories). Prune is single-directory only."
                )
            message = (
                f"Collection {collection_name} spans multiple directories; "
                "prompt-policy not certified from a single-directory run."
            )
            if warn is not None:
                warn(message)
            else:
                logger.warning(message)
            # Do not classify other-directory files as orphans; skip stamp.
            return result

        orphans = sorted(set(pre_run_paths) - set(on_disk_paths))
        result["orphans"] = orphans

        if orphans and prune:
            pruned = 0
            for orphan in orphans:
                try:
                    removed = sync.delete_chunks_by_file_path(collection_name, orphan)
                    # An orphan is resolved when its chunks are gone from the
                    # collection. A positive delete count proves removal. A
                    # zero count is ambiguous: it may mean the chunks were
                    # already cleared earlier in the run (the source pipeline
                    # deletes a project's whole branch before re-upload, so a
                    # deleted file's chunks vanish before this per-file prune) OR
                    # that the delete matched/removed nothing because of an
                    # error. Resolve the ambiguity by checking actual state so a
                    # legitimately-clean orphan counts as pruned while a path
                    # whose chunks still remain is correctly withheld
                    # (job-1921 Fix B + source reconciliation).
                    if removed > 0:
                        pruned += 1
                    elif not _orphan_chunks_remain(sync, collection_name, orphan):
                        pruned += 1
                except Exception as exc:
                    logger.warning(f"Failed to prune orphan {orphan}: {exc}")
            result["orphans_pruned"] = pruned
            result["orphans_remaining"] = len(orphans) - pruned
        else:
            # Without --prune, an orphan only blocks certification if its
            # chunks are actually still present (stale vectors). Orphans whose
            # chunks were already cleared earlier in the run (e.g. the source
            # pipeline's branch-delete before re-upload) leave no stale vectors,
            # so they must NOT withhold the stamp on an otherwise-clean reindex.
            result["orphans_remaining"] = sum(
                1 for orphan in orphans if _orphan_chunks_remain(sync, collection_name, orphan)
            )

    orphans_remaining = result["orphans_remaining"]

    # Coverage gate: a stamp certifies that EVERY vector in the collection was
    # produced under the current prompt policy. A scope-limited force (e.g.
    # markdown --no-recursive, a depth-limited source reindex) only re-embeds a
    # subset, so any still-existing indexed file that this run did NOT cover
    # retains potentially-stale vectors. Such uncovered files are NOT orphans
    # (they still exist, so they must not be pruned) but they DO bar
    # certification. Only enforced when callers supply covered_paths.
    uncovered_present: list = []
    if force and file_list is None and covered_paths is not None:
        still_present = set(pre_run_paths) & set(on_disk_paths)
        uncovered_present = sorted(still_present - set(covered_paths))

    if uncovered_present:
        message = (
            f"{len(uncovered_present)} indexed file(s) were not re-indexed by "
            "this run (scope-limited); their vectors may use a stale prompt "
            "policy. Collection not certified. Re-run a full reindex to certify."
        )
        if warn is not None:
            warn(message)
        else:
            logger.warning(message)
    elif should_stamp_prompt_policy(force, file_list, stats, orphans_remaining):
        stamp_embedding_prompt_policy(qdrant, collection_name, collection_type, model)
        result["stamped"] = True
    elif force and file_list is None and orphans_remaining > 0 and not prune:
        message = (
            f"{orphans_remaining} indexed file(s) no longer on disk; their "
            "vectors may use a stale prompt policy. Re-run with --prune to "
            "remove them and certify the collection."
        )
        if warn is not None:
            warn(message)
        else:
            logger.warning(message)

    return result


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
    client: QdrantClient, collection_name: str, collection_type: str, model: str, **extra_metadata
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
        **persisted_schema_defaults(),
        "collection_type": collection_type,
        "model": model,
        "embedding_prompt_policy": get_embedding_prompt_policies(model),
        "created_at": datetime.now().isoformat(),
        "created_by": "arcaneum",
        METADATA_PAYLOAD_KEY: True,  # Flag to identify metadata point
        **extra_metadata,
    }

    try:
        # Store metadata as a special point with reserved ID
        # We use a zero vector since this is not for search
        from qdrant_client.models import PointStruct

        # Get vector size from collection
        info = client.get_collection(collection_name)

        # Handle both single vector and named vectors
        if hasattr(info.config.params, "vectors"):
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
        metadata_point = PointStruct(id=METADATA_POINT_ID, vector=vectors, payload=metadata)

        client.upsert(collection_name=collection_name, points=[metadata_point])

        logger.info(f"Set collection metadata for {collection_name}: type={collection_type}")

    except Exception as e:
        logger.error(f"Failed to set collection metadata: {e}")
        raise


def update_collection_metadata(
    client: QdrantClient, collection_name: str, **updates
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
        raise ValueError(f"Collection '{collection_name}' has no metadata point to update")

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

        if hasattr(info.config.params, "vectors"):
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
            points=[
                PointStruct(
                    id=METADATA_POINT_ID,
                    vector=vectors,
                    payload={**metadata, METADATA_PAYLOAD_KEY: True},
                )
            ],
        )
        logger.info(f"Updated collection metadata for {collection_name}")
        return metadata

    except Exception as e:
        logger.error(f"Failed to update collection metadata: {e}")
        raise


def _retrieve_collection_metadata(
    client: QdrantClient, collection_name: str
) -> Optional[Dict[str, Any]]:
    """Return metadata payload, or None when the metadata point is absent.

    Unlike ``get_collection_metadata``, retrieval errors propagate so callers
    can distinguish a legacy collection from a failed metadata read.
    """
    points = client.retrieve(
        collection_name=collection_name,
        ids=[METADATA_POINT_ID],
        with_payload=True,
        with_vectors=False,
    )

    if points and len(points) > 0:
        payload = points[0].payload
        return {k: v for k, v in payload.items() if k != METADATA_PAYLOAD_KEY}

    return None


def get_collection_metadata(client: QdrantClient, collection_name: str) -> Dict[str, Any]:
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
        return _retrieve_collection_metadata(client, collection_name) or {}

    except Exception as e:
        # If retrieve fails, collection might not have metadata yet
        logger.debug(f"No metadata found for collection {collection_name}: {e}")
        return {}


def get_collection_type(client: QdrantClient, collection_name: str) -> Optional[str]:
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
    client: QdrantClient, collection_name: str, expected_type: str, allow_untyped: bool = True
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


def get_vector_names(client: QdrantClient, collection_name: str) -> list:
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

        if hasattr(info.config.params, "vectors"):
            if isinstance(info.config.params.vectors, dict):
                return list(info.config.params.vectors.keys())

        return []

    except Exception as e:
        logger.error(f"Failed to get vector names: {e}")
        return []

"""Corpus listing for the retrieval core (RDR-026 Q10).

``list_corpora`` pairs Qdrant collections with MeiliSearch indexes and reports
each corpus's type, sync status, item counts, models and description. A corpus
whose text backend is qdrant has no MeiliSearch side: its status is its text
coverage (RDR-027 Q8), and any MeiliSearch index of its name is ignored. It is
read-only: counts come from persisted metadata (stamped by full directory sync
and by repair) or are computed on demand from bounded Qdrant facets, and are
never written back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..embeddings.client import EMBEDDING_MODELS
from ..indexing.collection_metadata import (
    DEFAULT_TEXT_BACKEND,
    get_collection_metadata,
    metadata_exclusion_filter,
    text_backend_of,
    user_point_count,
)

logger = logging.getLogger(__name__)

UNKNOWN_LEGACY = "unknown/legacy"
LIST_COUNT_FACET_LIMIT = 10_000
COUNTABLE_CORPUS_TYPES = {"code", "markdown", "pdf"}


def _item_unit_for_corpus_type(collection_type):
    """Return the user-facing unit for corpus list item counts."""
    if collection_type == "code":
        return "repositories"
    if collection_type == "markdown":
        return "files"
    if collection_type == "pdf":
        return "documents"
    return UNKNOWN_LEGACY


def _last_sync_state(metadata, chunk_count: int = 0):
    """Normalize sync metadata into stable JSON fields."""
    if not metadata:
        return None, "unknown"
    last_sync = metadata.get("last_sync")
    if not last_sync:
        if chunk_count > 0:
            return None, "unknown"
        return None, "never_synced"
    return last_sync, "synced"


def _split_metadata_models(model_metadata):
    """Normalize stored model metadata into an ordered list of model names."""
    if not model_metadata:
        return []
    if isinstance(model_metadata, str):
        return [part.strip() for part in model_metadata.split(",") if part.strip()]
    if isinstance(model_metadata, (list, tuple)):
        return [str(part).strip() for part in model_metadata if str(part).strip()]
    return [str(model_metadata)]


def _embedding_model_config(model_name):
    """Look up an embedding model by alias or canonical model name."""
    if model_name in EMBEDDING_MODELS:
        return model_name, EMBEDDING_MODELS[model_name]
    for alias, config in EMBEDDING_MODELS.items():
        if config.get("name") == model_name:
            return alias, config
    return model_name, {}


def _extract_vectors_config(collection_info):
    """Return stable vector config records for single or named Qdrant vectors."""
    vectors = getattr(collection_info.config.params, "vectors", None)
    if isinstance(vectors, dict):
        return [
            {
                "vector_name": name,
                "dimensions": vector_params.size,
                "distance": str(vector_params.distance),
            }
            for name, vector_params in vectors.items()
        ]
    if vectors is not None:
        return [
            {
                "vector_name": None,
                "dimensions": getattr(vectors, "size", None),
                "distance": str(getattr(vectors, "distance", UNKNOWN_LEGACY)),
            }
        ]
    return []


def _extract_policy_metadata(metadata):
    """Expose prompt/schema policy metadata when collections recorded it."""
    policy = {key: metadata.get(key, UNKNOWN_LEGACY) for key in ("prompt_policy", "schema_policy")}
    for optional_key in ("prompt_schema_policy", "context_enrichment"):
        if optional_key in metadata:
            policy[optional_key] = metadata[optional_key]
    return policy


def _build_corpus_model_info(metadata, collection_info):
    """Build JSON-stable model/vector/backend details for a corpus."""
    model_names = _split_metadata_models(metadata.get("model"))
    vector_configs = _extract_vectors_config(collection_info)
    named_vectors = {
        vector["vector_name"]: vector
        for vector in vector_configs
        if vector.get("vector_name") is not None
    }
    positional_vectors = [vector for vector in vector_configs if vector.get("vector_name") is None]

    if not model_names and not vector_configs:
        return [
            {
                "alias": UNKNOWN_LEGACY,
                "name": UNKNOWN_LEGACY,
                "backend": UNKNOWN_LEGACY,
                "vector_name": UNKNOWN_LEGACY,
                "dimensions": None,
                "distance": UNKNOWN_LEGACY,
                "policy": _extract_policy_metadata(metadata),
            }
        ]

    models = []
    for index, model_name in enumerate(model_names):
        alias, config = _embedding_model_config(model_name)
        canonical_name = config.get("name", model_name)
        candidates = (model_name, alias, canonical_name)
        vector = next(
            (
                named_vectors.pop(candidate)
                for candidate in candidates
                if candidate in named_vectors
            ),
            positional_vectors[index] if index < len(positional_vectors) else {},
        )
        vector_name = vector.get("vector_name")

        models.append(
            {
                "alias": alias,
                "name": canonical_name,
                "backend": config.get("backend", UNKNOWN_LEGACY),
                "vector_name": vector_name or UNKNOWN_LEGACY,
                "dimensions": vector.get("dimensions", config.get("dimensions")),
                "distance": vector.get("distance", UNKNOWN_LEGACY),
                "policy": _extract_policy_metadata(metadata),
            }
        )

    if model_names:
        unmatched_vectors = named_vectors.values()
    else:
        unmatched_vectors = vector_configs

    for vector in unmatched_vectors:
        vector_name = vector.get("vector_name")
        lookup_name = vector_name

        if lookup_name:
            alias, config = _embedding_model_config(lookup_name)
            canonical_name = config.get("name", lookup_name)
            backend = config.get("backend", UNKNOWN_LEGACY)
        else:
            alias = UNKNOWN_LEGACY
            canonical_name = UNKNOWN_LEGACY
            backend = UNKNOWN_LEGACY

        models.append(
            {
                "alias": alias,
                "name": canonical_name,
                "backend": backend,
                "vector_name": vector_name or UNKNOWN_LEGACY,
                "dimensions": vector.get(
                    "dimensions",
                    config.get("dimensions") if lookup_name else None,
                ),
                "distance": vector.get("distance", UNKNOWN_LEGACY),
                "policy": _extract_policy_metadata(metadata),
            }
        )
    return models


def _format_model_summary(models):
    """Format concise model info for the corpus table."""
    if not models:
        return UNKNOWN_LEGACY
    parts = []
    for model in models:
        alias = model.get("alias") or UNKNOWN_LEGACY
        vector_name = model.get("vector_name")
        backend = model.get("backend")
        suffix = []
        if vector_name and vector_name != UNKNOWN_LEGACY and vector_name != alias:
            suffix.append(vector_name)
        if backend and backend != UNKNOWN_LEGACY:
            suffix.append(backend)
        parts.append(f"{alias} ({', '.join(suffix)})" if suffix else alias)
    return ", ".join(parts)


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    """Return a non-negative integer metadata value, if present."""
    value = metadata.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def _facet_hit_count(result) -> int:
    """Return the number of unique facet values in a Qdrant facet response."""
    hits = getattr(result, "hits", None)
    if hits is None and isinstance(result, dict):
        hits = result.get("hits")
    return len(hits or [])


def _bounded_qdrant_facet_count(client, collection_name: str, key: str) -> tuple[int | None, str]:
    """Count unique payload values without scrolling every point.

    Qdrant facets are limited but not paginated. A response that reaches the cap
    may be truncated, so callers treat it as unavailable instead of exact.
    """
    result = client.facet(
        collection_name=collection_name,
        key=key,
        facet_filter=metadata_exclusion_filter(),
        limit=LIST_COUNT_FACET_LIMIT,
        exact=True,
    )
    count = _facet_hit_count(result)
    if count >= LIST_COUNT_FACET_LIMIT:
        return None, "unavailable"
    return count, "facet"


def _scroll_qdrant_list_item_count(client, collection_name: str, collection_type: str):
    """Exact legacy count path used once to populate missing list counters."""
    unique_items = set()
    unique_files = set()
    offset = None
    payload_fields = (
        ["git_project_identifier", "file_path", "file_paths"]
        if collection_type == "code"
        else ["file_path", "file_paths"]
    )

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=metadata_exclusion_filter(),
            limit=1000,
            offset=offset,
            with_payload=payload_fields,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            if point.payload:
                if collection_type == "code":
                    item_id = point.payload.get("git_project_identifier")
                    if item_id:
                        unique_items.add(item_id)
                    for file_path in _payload_file_paths(point.payload):
                        unique_files.add(file_path)
                else:
                    for item_id in _payload_file_paths(point.payload):
                        unique_items.add(item_id)

        if offset is None:
            break

    return {
        "item_count": len(unique_items),
        "item_unit": _item_unit_for_corpus_type(collection_type),
        "file_count": len(unique_files) if collection_type == "code" else None,
        "file_unit": "source files" if collection_type == "code" else None,
        "item_count_status": "scroll",
    }


def _facet_based_list_item_count(client, collection_name: str, collection_type: str):
    """Count list items using bounded Qdrant facets only."""
    item_key = "git_project_identifier" if collection_type == "code" else "file_path"
    try:
        item_count, item_status = _bounded_qdrant_facet_count(client, collection_name, item_key)
        if collection_type == "code" and item_status != "unavailable":
            file_count, file_status = _bounded_qdrant_facet_count(
                client, collection_name, "file_path"
            )
            if file_status == "unavailable":
                item_status = "unavailable"
                item_count = None
                file_count = None
        else:
            file_count = None
    except Exception:
        item_count = None
        file_count = None
        item_status = "unavailable"

    return {
        "item_count": item_count,
        "item_unit": _item_unit_for_corpus_type(collection_type),
        "file_count": file_count if collection_type == "code" else None,
        "file_unit": "source files" if collection_type == "code" else None,
        "item_count_status": item_status,
    }


def _get_qdrant_list_item_count(
    client, collection_name: str, collection_type: str, metadata: dict[str, Any] | None = None
):
    """Count corpus-list items using persisted metadata or bounded facets."""
    if collection_type not in COUNTABLE_CORPUS_TYPES:
        return {
            "item_count": None,
            "item_unit": UNKNOWN_LEGACY,
            "file_count": None,
            "file_unit": None,
            "item_count_status": "unavailable",
        }

    metadata = metadata or {}
    item_count = _metadata_int(metadata, "item_count")
    file_count = _metadata_int(metadata, "file_count") if collection_type == "code" else None
    if item_count is not None and (collection_type != "code" or file_count is not None):
        return {
            "item_count": item_count,
            "item_unit": _item_unit_for_corpus_type(collection_type),
            "file_count": file_count,
            "file_unit": "source files" if collection_type == "code" else None,
            "item_count_status": metadata.get("count_source") or "metadata",
        }

    item_counts = _facet_based_list_item_count(client, collection_name, collection_type)

    if item_counts["item_count_status"] == "unavailable":
        try:
            return _scroll_qdrant_list_item_count(client, collection_name, collection_type)
        except Exception:
            return item_counts

    return item_counts


def _get_bounded_qdrant_list_item_count(
    client, collection_name: str, collection_type: str, metadata: dict[str, Any] | None = None
):
    """Count list items without unbounded point scrolling.

    Used by the default list view to fill missing count metadata for newer
    corpora synced from file lists or individual files.
    """
    metadata = metadata or {}
    cached = _cached_qdrant_list_item_count(collection_type, metadata)
    if cached["item_count"] is not None:
        return cached
    if collection_type not in COUNTABLE_CORPUS_TYPES:
        return cached

    return _facet_based_list_item_count(client, collection_name, collection_type)


def _cached_qdrant_list_item_count(collection_type: str, metadata: dict[str, Any] | None = None):
    """Return persisted corpus-list item counts without querying indexed points."""
    if collection_type not in COUNTABLE_CORPUS_TYPES:
        return {
            "item_count": None,
            "item_unit": UNKNOWN_LEGACY,
            "file_count": None,
            "file_unit": None,
            "item_count_status": "unavailable",
        }

    metadata = metadata or {}
    item_count = _metadata_int(metadata, "item_count")
    file_count = _metadata_int(metadata, "file_count") if collection_type == "code" else None
    count_status = (
        (metadata.get("count_source") or "metadata")
        if item_count is not None or file_count is not None
        else "not_requested"
    )

    return {
        "item_count": item_count,
        "item_unit": _item_unit_for_corpus_type(collection_type),
        "file_count": file_count,
        "file_unit": "source files" if collection_type == "code" else None,
        "item_count_status": count_status,
    }


def _payload_file_paths(payload):
    """Yield every indexed file path represented by a point payload."""
    file_paths = payload.get("file_paths")
    if isinstance(file_paths, list):
        emitted = False
        for file_path in file_paths:
            if file_path:
                emitted = True
                yield file_path
        if emitted:
            return
    file_path = payload.get("file_path")
    if file_path:
        yield file_path


def count_corpus_items(client, collection_name: str, collection_type: str) -> dict[str, Any]:
    """Count a corpus's items from the index, ignoring persisted counts.

    Uses bounded Qdrant facets and falls back to an exact scroll when a facet
    reaches its cap. Repair stamps the result; listing never persists it.
    """
    return _get_qdrant_list_item_count(client, collection_name, collection_type, metadata=None)


def _unknown_models() -> List[Dict[str, Any]]:
    """Model info for a corpus Qdrant does not hold (a MeiliSearch-only corpus)."""
    return [
        {
            "alias": UNKNOWN_LEGACY,
            "name": UNKNOWN_LEGACY,
            "backend": UNKNOWN_LEGACY,
            "vector_name": UNKNOWN_LEGACY,
            "dimensions": None,
            "distance": UNKNOWN_LEGACY,
            "policy": {
                "prompt_policy": UNKNOWN_LEGACY,
                "schema_policy": UNKNOWN_LEGACY,
            },
        }
    ]


@dataclass
class CorpusListing:
    """Every corpus found in Qdrant and MeiliSearch, with its parity status."""

    corpora: List[Dict[str, Any]]
    service_errors: List[str] = field(default_factory=list)

    def to_data(self) -> Dict[str, Any]:
        """The JSON ``data`` the CLI and the MCP server emit.

        ``corpora[].name`` is a stable contract (``research doctor`` reads it).
        """
        return {"corpora": self.corpora, "service_errors": list(self.service_errors)}


def _qdrant_collections(
    qdrant, details: bool, collections: Dict[str, Dict[str, Any]]
) -> None:
    """Fill ``collections`` in place, so a later collection's failure keeps the earlier ones."""
    for col in qdrant.get_collections().collections:
        metadata = get_collection_metadata(qdrant, col.name)
        col_info = qdrant.get_collection(col.name)
        chunk_count = user_point_count(qdrant, col.name, col_info)
        models = _build_corpus_model_info(metadata, col_info)
        collection_type = metadata.get("collection_type")
        last_sync, last_sync_status = _last_sync_state(metadata, chunk_count)
        # --details may scroll when a facet caps; the default view stays
        # bounded. Neither persists what it counts (RDR-026 Q10).
        count = _get_qdrant_list_item_count if details else _get_bounded_qdrant_list_item_count
        item_counts = count(qdrant, col.name, collection_type, metadata)
        text_backend = text_backend_of(metadata)
        coverage = None
        if text_backend == "qdrant":
            # A Qdrant corpus's text lives here: its coverage is its parity (RDR-027 Q8).
            from .qdrant_text import QdrantTextBackend

            coverage = QdrantTextBackend(qdrant, {col.name: collection_type}).coverage(
                col.name, files=False
            )
        collections[col.name] = {
            "type": collection_type,
            "text_backend": text_backend,
            "coverage": coverage,
            "model": metadata.get("model"),
            "description": metadata.get("description"),
            "models": models,
            "model_summary": _format_model_summary(models),
            "chunks": chunk_count,
            "last_sync": last_sync,
            "last_sync_status": last_sync_status,
            **item_counts,
        }


def _text_indexes(backend) -> Dict[str, Dict[str, Any]]:
    from ..cli.errors import ServiceDownError

    try:
        backend.health()
    except ServiceDownError as e:
        raise ConnectionError("server not available") from e
    return {name: {"chunks": chunks} for name, chunks in backend.corpora().items()}


def _service_error(service: str, exc: Exception) -> str:
    return f"{service}: {exc}"


def _default_qdrant_client():
    from ..cli.utils import create_qdrant_client

    return create_qdrant_client()


def _default_text_backend():
    from ..cli.utils import create_meili_client
    from ..fulltext.backend import MeiliTextBackend

    return MeiliTextBackend(create_meili_client())


def list_corpora(
    *,
    details: bool = False,
    qdrant_client=None,
    text_backend=None,
    qdrant_factory=None,
    text_factory=None,
) -> CorpusListing:
    """List every corpus with its parity status, writing nothing.

    Args:
        details: Allow an exact scroll when a bounded facet count caps
        qdrant_client: Qdrant client (built by ``qdrant_factory`` when None)
        text_backend: The ``TextBackend`` (built by ``text_factory`` when None)
        qdrant_factory: Builds the Qdrant client (from config when None)
        text_factory: Builds the text backend (MeiliSearch from config when None)

    A client or backend is built inside its service's error handling, so one
    service failing to construct does not hide the other's corpora.

    Returns:
        The listing. When one service fails, the listing holds the other's
        corpora (and any of the failing service's read before it failed) and
        names the failure in ``service_errors``.

    Raises:
        ServiceDownError: neither Qdrant nor MeiliSearch is reachable
    """
    service_errors: List[str] = []

    qdrant_collections: Dict[str, Dict[str, Any]] = {}
    try:
        if qdrant_client is None:
            qdrant_client = (qdrant_factory or _default_qdrant_client)()
        _qdrant_collections(qdrant_client, details, qdrant_collections)
    except Exception as e:
        logger.debug("listing Qdrant collections failed", exc_info=True)
        service_errors.append(_service_error("Qdrant", e))

    meili_indexes: Dict[str, Dict[str, Any]] = {}
    try:
        if text_backend is None:
            text_backend = (text_factory or _default_text_backend)()
        meili_indexes = _text_indexes(text_backend)
    except Exception as e:
        logger.debug("listing MeiliSearch indexes failed", exc_info=True)
        service_errors.append(_service_error("MeiliSearch", e))

    # Qdrant answered if it yielded any collection before failing.
    if len(service_errors) == 2 and not qdrant_collections:
        from ..cli.errors import ServiceDownError

        raise ServiceDownError(
            "Qdrant and MeiliSearch are unreachable: " + "; ".join(service_errors)
        )

    corpora = []
    for name in sorted(set(qdrant_collections) | set(meili_indexes)):
        q_info = qdrant_collections.get(name)
        m_info = meili_indexes.get(name)
        coverage = q_info.get("coverage") if q_info else None
        if coverage is not None:
            # A MeiliSearch index kept beside a Qdrant corpus is stale and unused.
            m_info = None

        q_chunks = q_info.get("chunks", 0) if q_info else 0
        m_chunks = m_info.get("chunks", 0) if m_info else 0
        if coverage is not None:
            status = coverage.status
        elif q_info and m_info:
            status = "synced" if q_chunks == m_chunks else "chunk_mismatch"
        elif q_info:
            status = "qdrant_only"
        else:
            status = "meili_only"

        q_info = q_info or {}
        entry = {
            "name": name,
            "type": q_info.get("type"),
            # A corpus Qdrant does not hold is a MeiliSearch index (RDR-027 Q2).
            "text_backend": q_info.get("text_backend", DEFAULT_TEXT_BACKEND),
            "model": q_info.get("model"),
            "description": q_info.get("description"),
            "models": q_info.get("models") or _unknown_models(),
            "model_summary": q_info.get("model_summary", UNKNOWN_LEGACY),
            "status": status,
            "qdrant_chunks": q_chunks,
            "meili_chunks": m_chunks,
            "text_chunks": m_chunks,
            "last_sync": q_info.get("last_sync"),
            "last_sync_status": q_info.get("last_sync_status", "unknown"),
            "item_count": q_info.get("item_count"),
            "item_unit": q_info.get("item_unit", UNKNOWN_LEGACY),
            "file_count": q_info.get("file_count"),
            "file_unit": q_info.get("file_unit"),
            "item_count_status": q_info.get("item_count_status", "unavailable"),
        }
        if coverage is not None:
            del entry["meili_chunks"]
            entry["text_chunks"] = coverage.indexed or 0
            entry["zero_token_chunks"] = coverage.zero_token_chunks
        corpora.append(entry)

    return CorpusListing(corpora=corpora, service_errors=service_errors)

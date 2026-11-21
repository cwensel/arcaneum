"""Qdrant upserts split by serialized size as well as count (RDR-027 Q7).

Qdrant refuses a REST request body over 32 MiB (33554432 bytes) with HTTP 400.
A count cap alone does not bound that: 300 PDF chunks of about 120 KB each
are over it. ``iter_upsert_batches`` closes each batch before either the count
cap or a byte budget is reached, measuring each point the way the REST client
serializes it, so callers send consecutive, ordered slices.
"""

import logging
from typing import Any, Iterator, List, Optional, Sequence

logger = logging.getLogger(__name__)

QDRANT_REQUEST_LIMIT_BYTES = 33554432
"""Qdrant's default limit on a REST request body (``service.max_request_size_mb``)."""

QDRANT_UPSERT_BUDGET_BYTES = 30 * 1024 * 1024
"""Bytes of points per upsert request, a margin below the server limit."""

POINT_ALLOWANCE_BYTES = 16
"""Per-point allowance for the list separator and envelope around each point."""


class OversizedPointError(ValueError):
    """A single point serializes larger than the upsert budget."""

    def __init__(self, file_path: Optional[str], chunk_index: Any, size: int, budget: int):
        self.file_path = file_path
        self.chunk_index = chunk_index
        self.size = size
        self.budget = budget
        where = file_path or "<unknown file>"
        if chunk_index is not None:
            where = f"{where} (chunk {chunk_index})"
        super().__init__(
            f"Chunk of {where} is {size} bytes serialized, over the {budget}-byte "
            f"Qdrant upsert budget (server limit {QDRANT_REQUEST_LIMIT_BYTES} bytes)"
        )


def point_request_bytes(point: Any) -> int:
    """Bytes ``point`` adds to an upsert request body.

    The REST client sends ``model_dump_json(by_alias=True, exclude_unset=True,
    exclude_none=True)`` of the points list as UTF-8, so the vectors count as
    well as the payload.
    """
    serialized = point.model_dump_json(by_alias=True, exclude_unset=True, exclude_none=True)
    return len(serialized.encode("utf-8")) + POINT_ALLOWANCE_BYTES


def _payload_field(point: Any, key: str) -> Any:
    payload = getattr(point, "payload", None) or {}
    return payload.get(key)


def iter_upsert_batches(
    points: Sequence[Any], max_count: int, max_bytes: Optional[int] = None
) -> Iterator[List[Any]]:
    """Yield consecutive slices of ``points``, each within both limits.

    A batch closes before it would exceed ``max_count`` points or
    ``max_bytes`` (default ``QDRANT_UPSERT_BUDGET_BYTES``, read at call time)
    serialized bytes. Order is preserved.

    Raises:
        OversizedPointError: a single point is over ``max_bytes``.
    """
    if max_count < 1:
        raise ValueError(f"max_count must be at least 1, got {max_count}")
    budget = QDRANT_UPSERT_BUDGET_BYTES if max_bytes is None else max_bytes
    batch: List[Any] = []
    batch_bytes = 0
    for point in points:
        size = point_request_bytes(point)
        if size > budget:
            raise OversizedPointError(
                _payload_field(point, "file_path"), _payload_field(point, "chunk_index"), size, budget
            )
        if batch and (len(batch) >= max_count or batch_bytes + size > budget):
            yield batch
            batch, batch_bytes = [], 0
        batch.append(point)
        batch_bytes += size
    if batch:
        yield batch


def upsert_in_batches(
    client: Any,
    collection_name: str,
    points: Sequence[Any],
    wait: bool = True,
    max_count: int = 300,
    max_bytes: Optional[int] = None,
    delete_sent_on_failure: bool = False,
) -> int:
    """Upsert ``points`` in requests within the count cap and byte budget.

    Every batch is planned before the first request, so an oversized point
    raises ``OversizedPointError`` with nothing sent. Returns the number of
    points sent.

    With ``delete_sent_on_failure``, a request that fails after earlier ones
    succeeded deletes the points already sent before the error propagates, so
    a file is never left with part of its chunks and no manifest. Only for
    freshly generated ids: deleting an id that existed before the call would
    remove its earlier content. A failed cleanup is logged; the upsert error
    is the one raised.
    """
    batches = list(iter_upsert_batches(points, max_count=max_count, max_bytes=max_bytes))
    sent_ids: List[Any] = []
    for batch in batches:
        try:
            client.upsert(collection_name=collection_name, points=batch, wait=wait)
        except Exception:
            if delete_sent_on_failure and sent_ids:
                _delete_sent(client, collection_name, sent_ids)
            raise
        sent_ids.extend(point.id for point in batch)
    return len(sent_ids)


def _delete_sent(client: Any, collection_name: str, ids: List[Any]) -> None:
    from qdrant_client.models import PointIdsList

    try:
        client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=ids),
            wait=True,
        )
    except Exception as cleanup_error:
        logger.warning(
            f"Could not remove {len(ids)} partially upserted points from "
            f"{collection_name}: {cleanup_error}"
        )

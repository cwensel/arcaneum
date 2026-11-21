"""Qdrant upserts split by serialized size as well as count (RDR-027 Q7).

Qdrant refuses a REST request body over 32 MiB with HTTP 400. These tests use
a recording fake client and check that every request stays under the byte
budget and the count cap, that ids and order survive the split, and that a
single point over the budget fails naming its file and size before anything
is sent.
"""

from uuid import UUID

import pytest
from qdrant_client.models import PointStruct

from arcaneum.indexing import qdrant_batching
from arcaneum.indexing.qdrant_batching import (
    QDRANT_REQUEST_LIMIT_BYTES,
    QDRANT_UPSERT_BUDGET_BYTES,
    OversizedPointError,
    iter_upsert_batches,
    point_request_bytes,
    upsert_in_batches,
)


class RecordingClient:
    def __init__(self):
        self.requests = []

    def upsert(self, collection_name, points, wait=True):
        self.requests.append({"collection_name": collection_name, "points": list(points), "wait": wait})


def _point(n: int, text_bytes: int, file_path: str = "/docs/big.pdf") -> PointStruct:
    return PointStruct(
        id=str(UUID(int=n + 1)),
        vector={"stub": [1.0, float(n)]},
        payload={"text": "x" * text_bytes, "file_path": file_path, "chunk_index": n},
    )


def _request_bytes(points) -> int:
    return sum(point_request_bytes(p) for p in points)


def test_budget_sits_below_the_server_limit():
    assert QDRANT_REQUEST_LIMIT_BYTES == 33554432
    assert QDRANT_UPSERT_BUDGET_BYTES < QDRANT_REQUEST_LIMIT_BYTES


def test_point_size_covers_the_serialized_point():
    point = _point(0, 1000)
    serialized = point.model_dump_json(by_alias=True, exclude_unset=True, exclude_none=True)
    assert point_request_bytes(point) >= len(serialized.encode("utf-8"))


def test_non_ascii_text_is_measured_in_bytes():
    ascii_point = _point(0, 1000)
    wide_point = PointStruct(
        id=ascii_point.id,
        vector=ascii_point.vector,
        payload={**ascii_point.payload, "text": "é" * 1000},
    )
    assert point_request_bytes(wide_point) >= point_request_bytes(ascii_point) + 1000


def test_requests_stay_under_the_byte_budget_and_keep_order():
    points = [_point(i, 1000) for i in range(40)]
    budget = 5 * point_request_bytes(points[0]) + 10
    client = RecordingClient()

    sent = upsert_in_batches(client, "corpus", points, wait=True, max_count=300, max_bytes=budget)

    assert sent == 40
    assert len(client.requests) > 1
    for request in client.requests:
        assert _request_bytes(request["points"]) <= budget
        assert len(request["points"]) <= 300
        assert request["collection_name"] == "corpus"
        assert request["wait"] is True
    sent_ids = [p.id for request in client.requests for p in request["points"]]
    assert sent_ids == [p.id for p in points]


def test_requests_stay_under_the_count_cap():
    points = [_point(i, 10) for i in range(7)]
    client = RecordingClient()

    upsert_in_batches(client, "corpus", points, wait=False, max_count=3)

    assert [len(r["points"]) for r in client.requests] == [3, 3, 1]
    assert all(r["wait"] is False for r in client.requests)


def test_count_and_size_limits_combine():
    points = [_point(i, 1000) for i in range(10)]
    budget = 3 * point_request_bytes(points[0]) + 10

    batches = list(iter_upsert_batches(points, max_count=2, max_bytes=budget))

    assert [len(b) for b in batches] == [2] * 5
    batches = list(iter_upsert_batches(points, max_count=10, max_bytes=budget))
    assert [len(b) for b in batches] == [3, 3, 3, 1]


def test_a_batch_under_both_limits_is_one_request():
    points = [_point(i, 100) for i in range(5)]
    client = RecordingClient()

    upsert_in_batches(client, "corpus", points, wait=True, max_count=300)

    assert len(client.requests) == 1
    assert [p.id for p in client.requests[0]["points"]] == [p.id for p in points]


def test_an_empty_list_sends_nothing():
    client = RecordingClient()

    assert upsert_in_batches(client, "corpus", [], wait=True, max_count=300) == 0
    assert client.requests == []


def test_an_oversized_point_names_its_file_and_size_and_sends_nothing():
    small = _point(0, 100)
    big = _point(1, 5000, file_path="/docs/huge.pdf")
    budget = point_request_bytes(big) - 1
    client = RecordingClient()

    with pytest.raises(OversizedPointError) as excinfo:
        upsert_in_batches(client, "corpus", [small, big], wait=True, max_count=300, max_bytes=budget)

    assert client.requests == []
    err = excinfo.value
    assert err.file_path == "/docs/huge.pdf"
    assert err.size == point_request_bytes(big)
    assert "/docs/huge.pdf" in str(err)
    assert str(point_request_bytes(big)) in str(err)


def test_the_default_budget_is_read_at_call_time(monkeypatch):
    points = [_point(i, 1000) for i in range(4)]
    monkeypatch.setattr(
        qdrant_batching, "QDRANT_UPSERT_BUDGET_BYTES", 2 * point_request_bytes(points[0]) + 10
    )

    batches = list(iter_upsert_batches(points, max_count=300))

    assert [len(b) for b in batches] == [2, 2]


class FailingClient(RecordingClient):
    """Accepts the first ``ok`` upserts, then refuses; records deletes."""

    def __init__(self, ok: int, delete_error: Exception = None):
        super().__init__()
        self.ok = ok
        self.deletes = []
        self.delete_error = delete_error

    def upsert(self, collection_name, points, wait=True):
        if len(self.requests) >= self.ok:
            raise RuntimeError("upsert refused")
        super().upsert(collection_name, points, wait)

    def delete(self, collection_name, points_selector, wait=True):
        self.deletes.append(
            {"collection_name": collection_name, "ids": list(points_selector.points), "wait": wait}
        )
        if self.delete_error is not None:
            raise self.delete_error


def test_a_failed_later_batch_deletes_the_points_already_sent():
    points = [_point(i, 10) for i in range(7)]
    client = FailingClient(ok=2)

    with pytest.raises(RuntimeError, match="upsert refused"):
        upsert_in_batches(client, "corpus", points, max_count=3, delete_sent_on_failure=True)

    assert len(client.deletes) == 1
    assert client.deletes[0]["collection_name"] == "corpus"
    assert client.deletes[0]["ids"] == [p.id for p in points[:6]]
    assert client.deletes[0]["wait"] is True


def test_a_failed_first_batch_deletes_nothing():
    client = FailingClient(ok=0)

    with pytest.raises(RuntimeError, match="upsert refused"):
        upsert_in_batches(
            client, "corpus", [_point(0, 10)], max_count=3, delete_sent_on_failure=True
        )

    assert client.deletes == []


def test_a_failed_cleanup_still_raises_the_upsert_error():
    points = [_point(i, 10) for i in range(4)]
    client = FailingClient(ok=1, delete_error=RuntimeError("delete refused"))

    with pytest.raises(RuntimeError, match="upsert refused"):
        upsert_in_batches(client, "corpus", points, max_count=2, delete_sent_on_failure=True)

    assert len(client.deletes) == 1


def test_without_cleanup_a_failed_batch_leaves_sent_points():
    points = [_point(i, 10) for i in range(4)]
    client = FailingClient(ok=1)

    with pytest.raises(RuntimeError, match="upsert refused"):
        upsert_in_batches(client, "corpus", points, max_count=2)

    assert client.deletes == []

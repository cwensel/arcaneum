import threading

import numpy as np
import pytest

from arcaneum.embeddings.batch_scheduler import (
    BatchBudget,
    BatchResultCollector,
    BudgetCatalog,
    OversizePolicy,
    SchedulingCancelled,
    SchedulingFailed,
    create_batch_queue,
    schedule_batches,
    submit_batches,
)


def tokens(text: str) -> int:
    return len(text.split())


def test_buckets_lengths_and_obeys_actual_padded_and_count_budgets():
    texts = ["a " * 5, "b", "c " * 4, "d " * 2, "e " * 3]
    batches = schedule_batches(
        texts,
        budget=BatchBudget(6, 6, 10, max_batch_size=2),
        count_tokens=tokens,
    )

    assert [batch.original_indices for batch in batches] == [(1, 3), (4,), (2,), (0,)]
    assert all(batch.actual_tokens <= 6 for batch in batches)
    assert all(batch.padded_tokens <= 6 for batch in batches)
    assert [(b.actual_tokens, b.padded_tokens) for b in batches] == [(3, 4), (3, 3), (4, 4), (5, 5)]


def test_count_option_is_only_an_additional_compatibility_cap():
    batches = schedule_batches(
        ["a", "b", "c"],
        budget=BatchBudget(100, 100, 10, max_batch_size=2),
        count_tokens=tokens,
    )
    assert [len(batch.items) for batch in batches] == [2, 1]


def test_catalog_prefers_model_budget_then_backend_default():
    default = BatchBudget(10, 10, 10)
    large = BatchBudget(2, 2, 2)
    catalog = BudgetCatalog({("mps", None): default, ("mps", "large"): large})
    assert catalog.resolve("mps", "large") is large
    assert catalog.resolve("mps", "small") is default
    with pytest.raises(KeyError, match="backend='cuda'"):
        catalog.resolve("cuda", "large")


def test_empty_input_has_no_batches_and_empty_result():
    assert schedule_batches([], budget=BatchBudget(1, 1, 1), count_tokens=tokens) == []
    assert BatchResultCollector(0).finalize().shape == (0, 0)


def test_oversize_error_singleton_and_sequence_limit_are_explicit():
    with pytest.raises(ValueError, match="cannot fit"):
        schedule_batches(["a b c"], budget=BatchBudget(2, 2, 4), count_tokens=tokens)
    singleton = schedule_batches(
        ["a b c"],
        budget=BatchBudget(2, 2, 4, oversize_policy=OversizePolicy.SINGLETON),
        count_tokens=tokens,
    )
    assert singleton[0].exceeds_budget
    with pytest.raises(ValueError, match="model limit"):
        schedule_batches(
            ["a b c d e"],
            budget=BatchBudget(10, 10, 4, oversize_policy=OversizePolicy.SINGLETON),
            count_tokens=tokens,
        )


def test_truncate_recounts_and_marks_metadata():
    batches = schedule_batches(
        ["a b c d", "e"],
        budget=BatchBudget(5, 6, 3, oversize_policy=OversizePolicy.TRUNCATE),
        count_tokens=tokens,
        truncate=lambda text, limit: " ".join(text.split()[:limit]),
    )
    item = next(item for batch in batches for item in batch.items if item.original_index == 0)
    assert (item.text, item.tokens, item.truncated) == ("a b c", 3, True)


def test_collector_restores_order_from_out_of_order_batches_and_owns_rows():
    batches = schedule_batches(
        ["one two", "three", "four five six"],
        budget=BatchBudget(3, 3, 4),
        count_tokens=tokens,
    )
    collector = BatchResultCollector(3)
    for batch in reversed(batches):
        rows = np.array(
            [[item.original_index, item.tokens] for item in batch.items], dtype=np.float32
        )
        collector.add(batch, rows)
        rows[:] = -1
    assert collector.finalize().tolist() == [[0, 2], [1, 1], [2, 3]]
    assert collector.completed_items == 3


def test_collector_rejects_missing_duplicate_and_failed_batch_shapes():
    batch = schedule_batches(["one"], budget=BatchBudget(2, 2, 2), count_tokens=tokens)[0]
    collector = BatchResultCollector(1)
    with pytest.raises(SchedulingFailed, match="row count"):
        collector.add(batch, np.empty((0, 2)))
    with pytest.raises(SchedulingFailed, match="only 0/1"):
        collector.finalize()
    collector.add(batch, np.ones((1, 2)))
    with pytest.raises(SchedulingFailed, match="completed twice"):
        collector.add(batch, np.ones((1, 2)))


def test_submit_uses_bounded_queue_and_reports_cancel_or_sink_failure():
    batches = schedule_batches(
        ["one", "two"],
        budget=BatchBudget(1, 1, 2),
        count_tokens=tokens,
    )
    sink = create_batch_queue(BatchBudget(1, 1, 2, max_queued_batches=1))
    assert sink.maxsize == 1
    cancel = threading.Event()
    sink.put(batches[0])
    cancel.set()
    with pytest.raises(SchedulingCancelled, match="0 batches"):
        submit_batches(batches, sink, cancel=cancel, put_timeout=0.001)

    class BrokenSink:
        def put(self, item, timeout=None):
            raise OSError("closed")

    with pytest.raises(SchedulingFailed, match="batch 0 submission failed"):
        submit_batches(batches, BrokenSink())

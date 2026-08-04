"""Token-budgeted scheduling primitives for embedding workers.

The scheduler is deliberately independent of an inference runtime.  It turns text
requests into length-bucketed batches and records enough metadata to restore the
caller's order after batches complete in any order.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Protocol

import numpy as np


class OversizePolicy(StrEnum):
    ERROR = "error"
    TRUNCATE = "truncate"
    SINGLETON = "singleton"


class SchedulingCancelled(RuntimeError):
    """Scheduling stopped before all batches entered the bounded queue."""


class SchedulingFailed(RuntimeError):
    """A scheduled batch failed or result bookkeeping was inconsistent."""


@dataclass(frozen=True)
class BatchBudget:
    """Backend/model memory limits.

    Token limits are authoritative. ``max_batch_size`` is retained as a count-based
    compatibility cap; it may make batches smaller but can never relax token limits.
    """

    max_actual_tokens: int
    max_padded_tokens: int
    max_sequence_tokens: int
    max_batch_size: int | None = None
    max_queued_batches: int = 1
    oversize_policy: OversizePolicy = OversizePolicy.ERROR

    def __post_init__(self) -> None:
        for name in (
            "max_actual_tokens",
            "max_padded_tokens",
            "max_sequence_tokens",
            "max_queued_batches",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one")
        if self.max_batch_size is not None and self.max_batch_size < 1:
            raise ValueError("max_batch_size must be at least one")


@dataclass(frozen=True)
class BudgetCatalog:
    """Resolve explicit backend/model budgets with an optional backend default."""

    budgets: Mapping[tuple[str, str | None], BatchBudget]

    def resolve(self, backend: str, model: str) -> BatchBudget:
        try:
            return self.budgets[(backend, model)]
        except KeyError:
            try:
                return self.budgets[(backend, None)]
            except KeyError as exc:
                raise KeyError(
                    f"no scheduling budget for backend={backend!r}, model={model!r}"
                ) from exc


@dataclass(frozen=True)
class ScheduledText:
    original_index: int
    text: str
    tokens: int
    truncated: bool = False


@dataclass(frozen=True)
class ScheduledBatch:
    batch_index: int
    items: tuple[ScheduledText, ...]
    actual_tokens: int
    padded_tokens: int
    exceeds_budget: bool = False

    @property
    def texts(self) -> list[str]:
        return [item.text for item in self.items]

    @property
    def original_indices(self) -> tuple[int, ...]:
        return tuple(item.original_index for item in self.items)


TokenCounter = Callable[[str], int]
TokenTruncator = Callable[[str, int], str]


def schedule_batches(
    texts: Iterable[str],
    *,
    budget: BatchBudget,
    count_tokens: TokenCounter,
    truncate: TokenTruncator | None = None,
) -> list[ScheduledBatch]:
    """Return deterministic, length-bucketed batches.

    Empty requests produce no batches.  Inputs longer than the model sequence limit
    are rejected unless ``truncate`` policy and callback are supplied.  A singleton
    policy permits one item to exceed aggregate token/shape budgets, but never the
    model sequence limit.
    """
    prepared: list[ScheduledText] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError("texts must contain only strings")
        tokens = count_tokens(text)
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            raise ValueError("token counter must return a non-negative integer")
        truncated = False
        if tokens > budget.max_sequence_tokens:
            if budget.oversize_policy is not OversizePolicy.TRUNCATE or truncate is None:
                raise ValueError(
                    f"text {index} has {tokens} tokens, exceeding model limit "
                    f"{budget.max_sequence_tokens}"
                )
            text = truncate(text, budget.max_sequence_tokens)
            tokens = count_tokens(text)
            truncated = True
            if tokens > budget.max_sequence_tokens:
                raise ValueError("token truncator did not satisfy the model sequence limit")
        prepared.append(ScheduledText(index, text, tokens, truncated))

    # Stable length sorting reduces padding while original_index breaks equal-length ties.
    prepared.sort(key=lambda item: (item.tokens, item.original_index))
    batches: list[ScheduledBatch] = []
    current: list[ScheduledText] = []

    def fits(items: list[ScheduledText]) -> bool:
        actual = sum(item.tokens for item in items)
        padded = max((item.tokens for item in items), default=0) * len(items)
        count_ok = budget.max_batch_size is None or len(items) <= budget.max_batch_size
        return (
            actual <= budget.max_actual_tokens and padded <= budget.max_padded_tokens and count_ok
        )

    def append(items: list[ScheduledText], *, exceeds: bool = False) -> None:
        batches.append(
            ScheduledBatch(
                batch_index=len(batches),
                items=tuple(items),
                actual_tokens=sum(item.tokens for item in items),
                padded_tokens=max(item.tokens for item in items) * len(items),
                exceeds_budget=exceeds,
            )
        )

    for item in prepared:
        if not fits([item]):
            if budget.oversize_policy is OversizePolicy.SINGLETON:
                if current:
                    append(current)
                    current = []
                append([item], exceeds=True)
                continue
            raise ValueError(
                f"text {item.original_index} cannot fit token/shape budget as a singleton"
            )
        if current and not fits([*current, item]):
            append(current)
            current = []
        current.append(item)
    if current:
        append(current)
    return batches


class BatchSink(Protocol):
    def put(self, item: ScheduledBatch, timeout: float | None = None) -> None: ...


def create_batch_queue(budget: BatchBudget) -> queue.Queue[ScheduledBatch]:
    """Create the bounded producer/worker handoff declared by a budget."""
    return queue.Queue(maxsize=budget.max_queued_batches)


def submit_batches(
    batches: Iterable[ScheduledBatch],
    sink: BatchSink,
    *,
    cancel: threading.Event | None = None,
    put_timeout: float = 0.1,
) -> int:
    """Submit with bounded-queue backpressure and cooperative cancellation."""
    submitted = 0
    for batch in batches:
        while True:
            if cancel is not None and cancel.is_set():
                raise SchedulingCancelled(f"cancelled after submitting {submitted} batches")
            try:
                sink.put(batch, timeout=put_timeout)
                submitted += 1
                break
            except queue.Full:
                continue
            except BaseException as exc:
                raise SchedulingFailed(
                    f"batch {batch.batch_index} submission failed after {submitted} batches"
                ) from exc
    return submitted


class BatchResultCollector:
    """Collect batch arrays and restore original request order exactly once."""

    def __init__(self, expected_items: int) -> None:
        if expected_items < 0:
            raise ValueError("expected_items cannot be negative")
        self._rows: list[np.ndarray | None] = [None] * expected_items
        self.completed_batches = 0
        self.completed_items = 0

    def add(self, batch: ScheduledBatch, rows: np.ndarray) -> None:
        value = np.asarray(rows)
        if value.ndim != 2 or value.shape[0] != len(batch.items):
            raise SchedulingFailed(f"batch {batch.batch_index} returned an invalid row count")
        for item, row in zip(batch.items, value, strict=True):
            if (
                item.original_index >= len(self._rows)
                or self._rows[item.original_index] is not None
            ):
                raise SchedulingFailed("result index is out of range or was completed twice")
            self._rows[item.original_index] = np.array(row, copy=True)
            self.completed_items += 1
        self.completed_batches += 1

    def finalize(self) -> np.ndarray:
        if not self._rows:
            return np.empty((0, 0), dtype=np.float32)
        if any(row is None for row in self._rows):
            raise SchedulingFailed(
                f"only {self.completed_items}/{len(self._rows)} result rows completed"
            )
        return np.stack([row for row in self._rows if row is not None])

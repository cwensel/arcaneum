"""Executable specification for RDR-024 cross-file index batching.

These tests pin the two load-bearing invariants from RDR-024 before the
implementation exists. They are expected to fail against `main` (the
`batched_flush` module is not yet written) and are the red half of the
red/green cycle.

Invariant 1 — no straddle: a single file's chunks are never split across two
flushes. A mid-file flush would leave a partially indexed file whose manifest
is never published, and nothing downstream would signal it.

Invariant 2 — deferred publication: manifests are published only after the
flush containing their chunks succeeds. An abort mid-batch must publish no
manifest for any file in that batch, so every affected file is re-detected as
new on the next run.
"""

import pytest

pytestmark = pytest.mark.rdr_spec

# The module under specification does not exist until RDR-024 Step 2 lands.
# Skip at collection rather than raising ModuleNotFoundError, so an explicit
# `-m rdr_spec` run reports "not yet implemented" instead of a collection error.
batched_flush = pytest.importorskip(
    "arcaneum.indexing.common.batched_flush",
    reason="RDR-024 not yet implemented",
)

FlushBuffer = batched_flush.FlushBuffer
adaptive_threshold = batched_flush.adaptive_threshold


class _Recorder:
    """Captures flush and manifest calls in order for assertion."""

    def __init__(self, fail_on_flush=None):
        self.flushes = []
        self.manifests = []
        self._fail_on_flush = fail_on_flush

    def index_batch(self, documents):
        if self._fail_on_flush is not None and len(self.flushes) == self._fail_on_flush:
            raise RuntimeError("MeiliSearch unavailable")
        self.flushes.append(list(documents))
        return len(documents), len(documents)

    def upsert_file_manifest(self, path, chunk_count):
        self.manifests.append((path, chunk_count))


def _file(name, chunks):
    """A file contributing `chunks` documents, each tagged with its origin."""
    return name, [f"{name}:{i}" for i in range(chunks)]


# --- Invariant 1: no file straddles a flush -------------------------------


@pytest.mark.parametrize(
    "chunk_counts,threshold",
    [
        ([10, 10, 10], 25),  # boundary falls mid-file if computed naively
        ([1, 1, 1, 1, 1], 2),  # many tiny files
        ([400], 300),  # single file larger than the threshold
        ([299, 2], 300),  # first file just under, second crosses
        ([300], 300),  # file exactly at the threshold
        ([50, 300, 50], 100),  # oversized file between small ones
    ],
)
def test_no_file_straddles_a_flush(chunk_counts, threshold):
    recorder = _Recorder()
    buffer = FlushBuffer(threshold=threshold, indexer=recorder)

    for i, count in enumerate(chunk_counts):
        name, docs = _file(f"f{i}", count)
        buffer.add_file(name, docs, chunk_count=count)
    buffer.finish()

    for batch in recorder.flushes:
        origins = [doc.split(":")[0] for doc in batch]
        # Every file appearing in this flush must appear in full.
        for origin in set(origins):
            expected = chunk_counts[int(origin[1:])]
            assert origins.count(origin) == expected, (
                f"file {origin} was split across flushes: "
                f"{origins.count(origin)} of {expected} chunks in this batch"
            )


def test_every_chunk_is_flushed_exactly_once():
    recorder = _Recorder()
    buffer = FlushBuffer(threshold=100, indexer=recorder)

    expected = []
    for i, count in enumerate([30, 80, 5, 200]):
        name, docs = _file(f"f{i}", count)
        expected.extend(docs)
        buffer.add_file(name, docs, chunk_count=count)
    buffer.finish()

    flushed = [doc for batch in recorder.flushes for doc in batch]
    assert sorted(flushed) == sorted(expected)
    assert len(flushed) == len(set(flushed)), "a chunk was flushed twice"


def test_trailing_partial_batch_is_flushed_on_finish():
    recorder = _Recorder()
    buffer = FlushBuffer(threshold=1000, indexer=recorder)

    name, docs = _file("only", 5)
    buffer.add_file(name, docs, chunk_count=5)
    assert recorder.flushes == [], "should not flush before the threshold"

    buffer.finish()
    assert len(recorder.flushes) == 1
    assert recorder.manifests == [("only", 5)]


# --- Invariant 2: manifests published only after a successful flush -------


def test_manifest_is_not_published_before_its_flush():
    recorder = _Recorder()
    buffer = FlushBuffer(threshold=1000, indexer=recorder)

    name, docs = _file("pending", 10)
    buffer.add_file(name, docs, chunk_count=10)

    # Chunks are buffered but not durable, so no manifest may exist yet.
    assert recorder.manifests == []


def test_abort_mid_batch_publishes_no_manifest_for_that_batch():
    # Fail on the second flush; the first batch's manifests must survive.
    recorder = _Recorder(fail_on_flush=1)
    buffer = FlushBuffer(threshold=10, indexer=recorder)

    name_a, docs_a = _file("a", 10)
    buffer.add_file(name_a, docs_a, chunk_count=10)  # triggers flush 0, succeeds

    name_b, docs_b = _file("b", 10)
    with pytest.raises(RuntimeError):
        buffer.add_file(name_b, docs_b, chunk_count=10)  # triggers flush 1, fails

    published = [p for p, _ in recorder.manifests]
    assert "a" in published, "a completed flush must publish its manifests"
    assert "b" not in published, (
        "a file whose flush failed must have no manifest, so the next run "
        "re-detects it as new"
    )


def test_manifests_follow_their_flush_in_order():
    recorder = _Recorder()
    order = []
    original_index = recorder.index_batch
    original_manifest = recorder.upsert_file_manifest

    def tracked_index(documents):
        order.append("flush")
        return original_index(documents)

    def tracked_manifest(path, chunk_count):
        order.append(f"manifest:{path}")
        return original_manifest(path, chunk_count)

    recorder.index_batch = tracked_index
    recorder.upsert_file_manifest = tracked_manifest

    buffer = FlushBuffer(threshold=10, indexer=recorder)
    for i, count in enumerate([6, 6]):
        name, docs = _file(f"f{i}", count)
        buffer.add_file(name, docs, chunk_count=count)
    buffer.finish()

    # No manifest may precede the flush that made its chunks durable.
    assert order[0] == "flush"
    for i, event in enumerate(order):
        if event.startswith("manifest"):
            assert "flush" in order[:i]


# --- Adaptive threshold ---------------------------------------------------


def test_threshold_is_monotonic_non_decreasing_in_index_size():
    sizes = [0, 10_000, 50_000, 100_000, 200_000, 300_000, 500_000, 1_000_000]
    thresholds = [adaptive_threshold(s) for s in sizes]
    assert thresholds == sorted(thresholds)


def test_threshold_is_bounded():
    # Must never exceed what a single flush can safely carry.
    assert adaptive_threshold(10_000_000) <= 10_000
    # Must stay positive so a flush always makes progress.
    assert adaptive_threshold(0) >= 1


def test_threshold_grows_where_merges_are_expensive():
    # RDR-024: merges are cheap below ~50k docs and dominate past ~250k.
    assert adaptive_threshold(300_000) > adaptive_threshold(25_000)

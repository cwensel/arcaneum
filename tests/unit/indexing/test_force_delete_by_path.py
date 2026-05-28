"""Regression tests for force-reindex delete-by-file_path (C1).

On force reindex, a file's prior chunks must be removed by stable identity
(file_path), not by the freshly-computed file_hash. A content change yields a
new hash, so a hash-based delete would leave the old chunks behind with a
potentially stale prompt policy.
"""

from types import SimpleNamespace

from qdrant_client.models import FilterSelector

from arcaneum.indexing.common.sync import MetadataBasedSync


class FakeQdrant:
    """Minimal Qdrant stand-in keyed on file_path payloads."""

    def __init__(self, points):
        # points: list of (point_id, file_path, file_hash)
        self._points = list(points)
        self.deleted_selectors = []

    def scroll(self, collection_name, scroll_filter=None, limit=10,
               offset=None, with_payload=False, with_vectors=False):
        matched = self._match(scroll_filter)
        return ([SimpleNamespace(id=p[0]) for p in matched[:limit]], None)

    def delete(self, collection_name, points_selector):
        self.deleted_selectors.append(points_selector)
        # Apply the delete so a follow-up scroll observes the removal.
        filt = points_selector.filter
        keep = []
        for p in self._points:
            if not self._point_matches(p, filt):
                keep.append(p)
        self._points = keep

    def _match(self, scroll_filter):
        if scroll_filter is None:
            return list(self._points)
        return [p for p in self._points if self._point_matches(p, scroll_filter)]

    @staticmethod
    def _point_matches(point, filt):
        _, file_path, file_hash = point
        for cond in (filt.must or []):
            key = cond.key
            want = cond.match.value
            have = {"file_path": file_path, "file_hash": file_hash}.get(key)
            if have != want:
                return False
        return True


def test_delete_chunks_by_file_path_removes_regardless_of_hash():
    """A changed-content file (new hash) must still have its old chunks removed
    when deleting by file_path."""
    path = "/abs/docs/changed.md"
    qdrant = FakeQdrant([
        (1, path, "OLD_HASH"),
        (2, path, "OLD_HASH"),
        (3, "/abs/docs/other.md", "OTHER_HASH"),
    ])
    sync = MetadataBasedSync(qdrant)

    deleted = sync.delete_chunks_by_file_path("docs", path)

    assert deleted == 2
    # The selector targets file_path, not file_hash.
    assert len(qdrant.deleted_selectors) == 1
    selector = qdrant.deleted_selectors[0]
    assert isinstance(selector, FilterSelector)
    keys = {c.key for c in selector.filter.must}
    assert keys == {"file_path"}
    # Unrelated file untouched; target file fully removed.
    remaining_paths = {p[1] for p in qdrant._points}
    assert path not in remaining_paths
    assert "/abs/docs/other.md" in remaining_paths


def test_delete_chunks_by_file_path_no_match_returns_zero():
    qdrant = FakeQdrant([(1, "/abs/keep.md", "H")])
    sync = MetadataBasedSync(qdrant)

    deleted = sync.delete_chunks_by_file_path("docs", "/abs/missing.md")

    assert deleted == 0
    assert qdrant.deleted_selectors == []

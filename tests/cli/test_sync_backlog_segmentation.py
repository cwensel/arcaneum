"""Tests for phase segmentation of a ``--order newest`` sync.

A resumed ``--order newest`` sync works through a pending list that is not one
run but several, and the useful distinction is not file age in the abstract —
it is *why* a file is still pending:

  - written after the last run's newest file (it appeared while that run was
    working, or since it stopped),
  - falling inside the range the last run already covered (it was skipped or
    written mid-run), or
  - below everything the last run reached (the true backlog).

Measured against a real corpus: an interrupted sync over an actively-appended
transcript directory leaves 246 indexed files spanning 08-12..08-17, only 7 of
which form a clean newest-first prefix. Files keep arriving while the sync
runs, so "one interruption, one clean cut" does not hold, and any label that
calls the freshest files in the tree "backlog" is inverted.
"""

import os
from pathlib import Path

from arcaneum.cli.sync import (
    PendingPhase,
    phase_label,
    segment_pending_by_phase,
)


def _touch(path: Path, mtime: float) -> Path:
    """Create a file with an explicit mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("content")
    os.utime(path, (mtime, mtime))
    return path


def _names(segment):
    return [p.name for p in segment.files]


class TestSegmentPendingByPhase:
    """Classify pending files against the mtime range the last run covered."""

    def test_arrived_covered_and_backlog_are_distinguished(self, tmp_path: Path):
        """The three reasons a file can still be pending, in index order."""
        indexed = [
            _touch(tmp_path / "i_new.md", 5_000),
            _touch(tmp_path / "i_old.md", 3_000),
        ]
        pending = [
            _touch(tmp_path / "arrived.md", 6_000),   # above the indexed range
            _touch(tmp_path / "covered.md", 4_000),   # inside the indexed range
            _touch(tmp_path / "backlog.md", 1_000),   # below the indexed range
        ]

        segments = segment_pending_by_phase(pending, indexed)

        assert [s.phase for s in segments] == [
            PendingPhase.ARRIVED,
            PendingPhase.COVERED,
            PendingPhase.BACKLOG,
        ]
        assert [_names(s) for s in segments] == [["arrived.md"], ["covered.md"], ["backlog.md"]]

    def test_real_corpus_shape_names_fresh_files_as_arrived(self, tmp_path: Path):
        """The screenshot case: files written during the sync are not 'backlog'.

        The previous run indexed 08-12..08-17; twelve files landed at 19:51-19:52
        while it was finishing. Those are the newest content in the tree.
        """
        indexed = [
            _touch(tmp_path / "i_top.md", 1_755_400_339),   # 08-17 19:52:19
            _touch(tmp_path / "i_bottom.md", 1_754_900_187),  # 08-12 10:56
        ]
        pending = [
            _touch(tmp_path / "during_a.md", 1_755_400_500),
            _touch(tmp_path / "during_b.md", 1_755_400_400),
            _touch(tmp_path / "old_a.md", 1_750_000_000),
            _touch(tmp_path / "old_b.md", 1_749_000_000),
        ]

        segments = segment_pending_by_phase(pending, indexed)

        assert segments[0].phase is PendingPhase.ARRIVED
        assert _names(segments[0]) == ["during_a.md", "during_b.md"]
        assert segments[1].phase is PendingPhase.BACKLOG
        assert _names(segments[1]) == ["old_a.md", "old_b.md"]

    def test_covered_files_group_into_one_run(self, tmp_path: Path):
        """Everything inside the indexed range is one phase, not a gap per file.

        A non-contiguous indexed set (239 of 246 files sat below the first
        unindexed one) must not shatter the display into dozens of segments.
        """
        indexed = [
            _touch(tmp_path / f"i{i}.md", 3_000 + i * 100) for i in range(10)
        ]
        pending = [
            _touch(tmp_path / f"p{i}.md", 3_050 + i * 100) for i in range(9)
        ]

        segments = segment_pending_by_phase(pending, indexed)

        assert len(segments) == 1
        assert segments[0].phase is PendingPhase.COVERED
        assert len(segments[0].files) == 9

    def test_no_indexed_files_is_a_single_arrived_run(self, tmp_path: Path):
        """A first sync has no prior run to compare against."""
        pending = [_touch(tmp_path / "a.md", 2_000), _touch(tmp_path / "b.md", 1_000)]

        segments = segment_pending_by_phase(pending, [])

        assert len(segments) == 1
        assert segments[0].phase is PendingPhase.ARRIVED

    def test_empty_pending_yields_no_segments(self, tmp_path: Path):
        assert segment_pending_by_phase([], [_touch(tmp_path / "i.md", 1_000)]) == []

    def test_boundaries_are_inclusive_of_the_indexed_range(self, tmp_path: Path):
        """A file tied with an indexed mtime sits inside the covered range."""
        indexed = [_touch(tmp_path / "hi.md", 5_000), _touch(tmp_path / "lo.md", 2_000)]
        pending = [_touch(tmp_path / "at_hi.md", 5_000), _touch(tmp_path / "at_lo.md", 2_000)]

        segments = segment_pending_by_phase(pending, indexed)

        assert len(segments) == 1
        assert segments[0].phase is PendingPhase.COVERED

    def test_preserves_caller_order_within_a_segment(self, tmp_path: Path):
        """Segmentation classifies; it never reorders."""
        indexed = [_touch(tmp_path / "i.md", 5_000)]
        pending = [
            _touch(tmp_path / "z.md", 1_000),
            _touch(tmp_path / "a.md", 900),
        ]

        segments = segment_pending_by_phase(pending, indexed)

        assert _names(segments[0]) == ["z.md", "a.md"]

    def test_repeated_phase_after_an_excursion_starts_a_new_segment(self, tmp_path: Path):
        """Order drives the display, so a return to a phase is a new run."""
        indexed = [_touch(tmp_path / "i.md", 3_000)]
        pending = [
            _touch(tmp_path / "old1.md", 1_000),
            _touch(tmp_path / "new1.md", 4_000),
            _touch(tmp_path / "old2.md", 900),
        ]

        segments = segment_pending_by_phase(pending, indexed)

        assert [s.phase for s in segments] == [
            PendingPhase.BACKLOG,
            PendingPhase.ARRIVED,
            PendingPhase.BACKLOG,
        ]

    def test_vanished_pending_file_joins_the_current_segment(self, tmp_path: Path):
        """A file removed between discovery and segmentation must not abort a sync."""
        indexed = [_touch(tmp_path / "i.md", 5_000)]
        pending = [
            _touch(tmp_path / "a.md", 1_000),
            _touch(tmp_path / "gone.md", 900),
            _touch(tmp_path / "b.md", 800),
        ]
        pending[1].unlink()

        segments = segment_pending_by_phase(pending, indexed)

        assert len(segments) == 1
        assert _names(segments[0]) == ["a.md", "gone.md", "b.md"]

    def test_vanished_indexed_file_is_ignored(self, tmp_path: Path):
        """A stat failure on an indexed file must not distort the range."""
        indexed = [_touch(tmp_path / "gone.md", 9_000), _touch(tmp_path / "here.md", 2_000)]
        indexed[0].unlink()
        pending = [_touch(tmp_path / "new.md", 3_000)]

        segments = segment_pending_by_phase(pending, indexed)

        assert segments[0].phase is PendingPhase.ARRIVED

    def test_segment_carries_its_own_progress_weight(self, tmp_path: Path):
        indexed = [_touch(tmp_path / "i.md", 2_000)]
        new_file = tmp_path / "new.md"
        new_file.write_text("x" * 100)
        os.utime(new_file, (3_000, 3_000))
        old_file = tmp_path / "old.md"
        old_file.write_text("y" * 50)
        os.utime(old_file, (1_000, 1_000))

        segments = segment_pending_by_phase([new_file, old_file], indexed)

        assert segments[0].weight == 100
        assert segments[1].weight == 50


class TestPhaseLabel:
    """Labels must say why a file is pending, in words that survive a glance."""

    def test_labels_state_the_reason_not_the_age(self, tmp_path: Path):
        indexed = [_touch(tmp_path / "i_hi.md", 5_000), _touch(tmp_path / "i_lo.md", 3_000)]
        pending = [
            _touch(tmp_path / "a.md", 6_000),
            _touch(tmp_path / "c.md", 4_000),
            _touch(tmp_path / "b.md", 1_000),
        ]

        labels = [phase_label(s) for s in segment_pending_by_phase(pending, indexed)]

        assert labels == [
            "New since last sync (1 file)",
            "Missed by last sync (1 file)",
            "Older backlog (1 file)",
        ]

    def test_label_pluralizes(self, tmp_path: Path):
        indexed = [_touch(tmp_path / "i.md", 5_000)]
        pending = [_touch(tmp_path / "a.md", 1_000), _touch(tmp_path / "b.md", 900)]

        segments = segment_pending_by_phase(pending, indexed)

        assert phase_label(segments[0]) == "Older backlog (2 files)"


class TestPhaseTransitionSequence:
    """The descriptions the sync loop emits as it crosses phase boundaries."""

    def _descriptions(self, pending, indexed):
        segments = segment_pending_by_phase(pending, indexed)
        starts = {}
        if len(segments) > 1:
            position = 0
            for segment in segments:
                starts[position] = segment
                position += len(segment.files)

        label = ""
        out = []
        for index, path in enumerate(pending):
            entering = starts.get(index)
            if entering is not None:
                label = phase_label(entering)
            prefix = f"{label} · " if label else ""
            out.append(f"{prefix}Processing {path.name}...")
        return out

    def test_description_changes_when_crossing_into_the_backlog(self, tmp_path: Path):
        indexed = [_touch(tmp_path / "i.md", 2_000)]
        pending = [
            _touch(tmp_path / "new.md", 3_000),
            _touch(tmp_path / "old_a.md", 1_500),
            _touch(tmp_path / "old_b.md", 1_000),
        ]

        assert self._descriptions(pending, indexed) == [
            "New since last sync (1 file) · Processing new.md...",
            "Older backlog (2 files) · Processing old_a.md...",
            "Older backlog (2 files) · Processing old_b.md...",
        ]

    def test_single_phase_run_carries_no_prefix(self, tmp_path: Path):
        """Nothing to distinguish, so the display stays as it was before."""
        pending = [_touch(tmp_path / "a.md", 2_000), _touch(tmp_path / "b.md", 1_000)]

        assert self._descriptions(pending, []) == [
            "Processing a.md...",
            "Processing b.md...",
        ]

"""Tests for parameterized index ordering in arcaneum.cli.sync.

Discovery has always returned files sorted lexicographically by path. The
``order`` parameter lets a sync process recently-modified files first, so an
interrupted or long-running sync surfaces the most valuable content early.
Every order is total and deterministic — ties fall back to path so repeated
runs over an unchanged tree produce identical sequences.
"""

import os
from pathlib import Path

import pytest

from arcaneum.cli.sync import discover_files, order_files


def _touch(path: Path, mtime: float) -> Path:
    """Create a file with an explicit mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("content")
    os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def dated_tree(tmp_path: Path) -> Path:
    """Three markdown files whose mtime order is the reverse of path order."""
    _touch(tmp_path / "a_oldest.md", 1_000)
    _touch(tmp_path / "b_middle.md", 2_000)
    _touch(tmp_path / "c_newest.md", 3_000)
    return tmp_path


class TestOrderFiles:
    """order_files applies a named ordering to an arbitrary file list."""

    def test_path_order_is_lexicographic(self, dated_tree: Path):
        files = list(dated_tree.glob("*.md"))
        ordered = order_files(files, "path")
        assert [f.name for f in ordered] == ["a_oldest.md", "b_middle.md", "c_newest.md"]

    def test_newest_order_puts_recent_files_first(self, dated_tree: Path):
        files = list(dated_tree.glob("*.md"))
        ordered = order_files(files, "newest")
        assert [f.name for f in ordered] == ["c_newest.md", "b_middle.md", "a_oldest.md"]

    def test_oldest_order_puts_stale_files_first(self, dated_tree: Path):
        files = list(dated_tree.glob("*.md"))
        ordered = order_files(files, "oldest")
        assert [f.name for f in ordered] == ["a_oldest.md", "b_middle.md", "c_newest.md"]

    def test_mtime_ties_fall_back_to_path(self, tmp_path: Path):
        """Equal mtimes must still yield a stable, deterministic sequence."""
        for name in ("z.md", "m.md", "a.md"):
            _touch(tmp_path / name, 5_000)
        files = list(tmp_path.glob("*.md"))

        ordered = order_files(files, "newest")

        assert [f.name for f in ordered] == ["a.md", "m.md", "z.md"]
        assert ordered == order_files(list(reversed(files)), "newest")

    def test_unknown_order_is_rejected(self, dated_tree: Path):
        with pytest.raises(ValueError, match="nonsense"):
            order_files(list(dated_tree.glob("*.md")), "nonsense")

    def test_missing_file_sorts_last_without_raising(self, tmp_path: Path):
        """A file deleted between discovery and ordering must not abort the sync."""
        present = _touch(tmp_path / "present.md", 9_000)
        missing = tmp_path / "vanished.md"

        ordered = order_files([missing, present], "newest")

        assert ordered[0] == present
        assert ordered[-1] == missing

    def test_returns_new_list_without_mutating_input(self, dated_tree: Path):
        files = list(dated_tree.glob("*.md"))
        original = list(files)

        order_files(files, "newest")

        assert files == original


class TestDiscoverFilesOrdering:
    """discover_files threads the order through to its returned file list."""

    def test_defaults_to_path_order(self, dated_tree: Path):
        files, _ = discover_files(dated_tree, None, "markdown")
        assert [f.name for f in files] == ["a_oldest.md", "b_middle.md", "c_newest.md"]

    def test_newest_first_reverses_by_mtime(self, dated_tree: Path):
        files, _ = discover_files(dated_tree, None, "markdown", order="newest")
        assert [f.name for f in files] == ["c_newest.md", "b_middle.md", "a_oldest.md"]

    def test_ordering_spans_subdirectories(self, tmp_path: Path):
        """Ordering is global across the tree, not per-directory."""
        _touch(tmp_path / "zdir" / "new.md", 8_000)
        _touch(tmp_path / "adir" / "old.md", 1_000)

        files, _ = discover_files(tmp_path, None, "markdown", order="newest")

        assert [f.name for f in files] == ["new.md", "old.md"]

    def test_ordering_does_not_change_the_discovered_set(self, dated_tree: Path):
        by_path, _ = discover_files(dated_tree, None, "markdown", order="path")
        by_newest, _ = discover_files(dated_tree, None, "markdown", order="newest")

        assert set(by_path) == set(by_newest)

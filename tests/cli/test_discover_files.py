"""Tests for corpus-type-aware file discovery in arcaneum.cli.sync.discover_files.

The key behavior under test: git-tracked discovery only applies to ``code``
corpora. For ``markdown`` and ``pdf`` corpora the filesystem is scanned
directly so untracked research notes and downloaded documents are picked up.
"""

import subprocess
from pathlib import Path

import pytest

from arcaneum.cli.sync import discover_files


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def git_repo_with_untracked(tmp_path: Path) -> Path:
    """Create a git repo with one tracked and one untracked markdown file."""
    repo = tmp_path / "repo"
    repo.mkdir()

    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")

    (repo / "tracked.md").write_text("# tracked\n")
    _git(repo, "add", "tracked.md")
    _git(repo, "commit", "-q", "-m", "initial")

    (repo / "untracked.md").write_text("# untracked\n")

    return repo


class TestDiscoverFilesMarkdown:
    def test_markdown_discovers_untracked_in_git_repo(self, git_repo_with_untracked):
        """Markdown corpora ignore git and pick up untracked files."""
        files = discover_files(git_repo_with_untracked, None, "markdown")
        names = {f.name for f in files}
        assert names == {"tracked.md", "untracked.md"}

    def test_markdown_respects_skip_dir_prefixes(self, git_repo_with_untracked):
        """Directories matching skip prefixes are still excluded."""
        (git_repo_with_untracked / "_excluded").mkdir()
        (git_repo_with_untracked / "_excluded" / "skip.md").write_text("# skip\n")

        files = discover_files(git_repo_with_untracked, None, "markdown")
        names = {f.name for f in files}
        assert "skip.md" not in names
        assert "untracked.md" in names


class TestDiscoverFilesPdf:
    def test_pdf_discovers_untracked_in_git_repo(self, git_repo_with_untracked):
        """PDF corpora also use filesystem discovery, not git."""
        tracked_pdf = git_repo_with_untracked / "tracked.pdf"
        tracked_pdf.write_bytes(b"%PDF-1.4\n")
        _git(git_repo_with_untracked, "add", "tracked.pdf")
        _git(git_repo_with_untracked, "commit", "-q", "-m", "add pdf")

        (git_repo_with_untracked / "untracked.pdf").write_bytes(b"%PDF-1.4\n")

        files = discover_files(git_repo_with_untracked, None, "pdf")
        names = {f.name for f in files}
        assert names == {"tracked.pdf", "untracked.pdf"}


class TestDiscoverFilesCode:
    def test_code_uses_git_ls_files_in_git_repo(self, git_repo_with_untracked):
        """Code corpora still respect git tracking — untracked files are skipped."""
        (git_repo_with_untracked / "tracked.py").write_text("x = 1\n")
        _git(git_repo_with_untracked, "add", "tracked.py")
        _git(git_repo_with_untracked, "commit", "-q", "-m", "add py")

        (git_repo_with_untracked / "untracked.py").write_text("y = 2\n")

        files = discover_files(git_repo_with_untracked, None, "code")
        names = {f.name for f in files}
        assert "tracked.py" in names
        assert "untracked.py" not in names

    def test_code_falls_back_to_rglob_when_no_tracked_files(self, tmp_path):
        """Code corpora in a non-git directory use rglob."""
        (tmp_path / "loose.py").write_text("z = 3\n")

        files = discover_files(tmp_path, None, "code")
        names = {f.name for f in files}
        assert names == {"loose.py"}

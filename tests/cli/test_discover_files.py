"""Tests for corpus-type-aware file discovery in arcaneum.cli.sync.discover_files.

The key behavior under test:
- Code corpora: git repos are treated atomically via git ls-files (respects
  .gitignore) regardless of whether the input is a single repo or a
  folder-of-repos.
- Markdown and PDF corpora: filesystem discovery (rglob) so untracked research
  notes and downloaded documents are picked up.
- discover_files returns (files, git_roots); git_roots is non-empty only for
  code corpora that contain at least one git repo.
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


def _make_repo(path: Path) -> Path:
    """Create a minimal git repo at path."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    return path


@pytest.fixture
def git_repo_with_untracked(tmp_path: Path) -> Path:
    """Create a git repo with one tracked and one untracked markdown file."""
    repo = _make_repo(tmp_path / "repo")

    (repo / "tracked.md").write_text("# tracked\n")
    _git(repo, "add", "tracked.md")
    _git(repo, "commit", "-q", "-m", "initial")

    (repo / "untracked.md").write_text("# untracked\n")

    return repo


@pytest.fixture
def folder_of_repos(tmp_path: Path) -> Path:
    """Create a folder containing two independent git repos."""
    parent = tmp_path / "repos"
    parent.mkdir()

    for name in ("repo-a", "repo-b"):
        repo = _make_repo(parent / name)
        (repo / f"{name}_tracked.py").write_text("x = 1\n")
        _git(repo, "add", f"{name}_tracked.py")
        _git(repo, "commit", "-q", "-m", "initial")
        # Untracked file — should NOT be discovered
        (repo / f"{name}_untracked.py").write_text("y = 2\n")
        # Gitignored file — should NOT be discovered
        (repo / ".gitignore").write_text("ignored.py\n")
        (repo / "ignored.py").write_text("z = 3\n")
        _git(repo, "add", ".gitignore")
        _git(repo, "commit", "-q", "-m", "add gitignore")

    return parent


@pytest.fixture
def nested_folder_of_repos(tmp_path: Path) -> Path:
    """Create an org/repo layout (repos nested one level deeper)."""
    parent = tmp_path / "orgs"
    parent.mkdir()

    for org, proj in [("org-a", "proj-1"), ("org-b", "proj-2")]:
        org_dir = parent / org
        repo = _make_repo(org_dir / proj)
        (repo / "main.go").write_text("package main\n")
        _git(repo, "add", "main.go")
        _git(repo, "commit", "-q", "-m", "initial")

    return parent


class TestDiscoverFilesMarkdown:
    def test_markdown_discovers_untracked_in_git_repo(self, git_repo_with_untracked):
        """Markdown corpora ignore git and pick up untracked files."""
        files, roots = discover_files(git_repo_with_untracked, None, "markdown")
        names = {f.name for f in files}
        assert names == {"tracked.md", "untracked.md"}
        assert roots == []

    def test_markdown_respects_skip_dir_prefixes(self, git_repo_with_untracked):
        """Directories matching skip prefixes are still excluded."""
        (git_repo_with_untracked / "_excluded").mkdir()
        (git_repo_with_untracked / "_excluded" / "skip.md").write_text("# skip\n")

        files, _ = discover_files(git_repo_with_untracked, None, "markdown")
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

        files, roots = discover_files(git_repo_with_untracked, None, "pdf")
        names = {f.name for f in files}
        assert names == {"tracked.pdf", "untracked.pdf"}
        assert roots == []


class TestDiscoverFilesCode:
    def test_code_uses_git_ls_files_in_single_repo(self, git_repo_with_untracked):
        """Code corpora respect git tracking when input is a single repo."""
        (git_repo_with_untracked / "tracked.py").write_text("x = 1\n")
        _git(git_repo_with_untracked, "add", "tracked.py")
        _git(git_repo_with_untracked, "commit", "-q", "-m", "add py")

        (git_repo_with_untracked / "untracked.py").write_text("y = 2\n")

        files, roots = discover_files(git_repo_with_untracked, None, "code")
        names = {f.name for f in files}
        assert "tracked.py" in names
        assert "untracked.py" not in names
        assert len(roots) == 1

    def test_code_falls_back_to_rglob_when_no_git(self, tmp_path):
        """Code corpora with no git repos at all fall back to rglob."""
        (tmp_path / "loose.py").write_text("z = 3\n")

        files, roots = discover_files(tmp_path, None, "code")
        names = {f.name for f in files}
        assert names == {"loose.py"}
        assert roots == []

    def test_code_folder_of_repos_uses_git_ls_files_per_repo(self, folder_of_repos):
        """Folder-of-repos: each sub-repo is discovered via git ls-files."""
        files, roots = discover_files(folder_of_repos, None, "code")
        names = {f.name for f in files}

        # Only tracked files from each repo
        assert "repo-a_tracked.py" in names
        assert "repo-b_tracked.py" in names

        # Untracked and gitignored files must be absent
        assert "repo-a_untracked.py" not in names
        assert "repo-b_untracked.py" not in names
        assert "ignored.py" not in names

        # Two repos discovered
        assert len(roots) == 2

    def test_code_folder_of_repos_returns_correct_git_roots(self, folder_of_repos):
        """git_roots contains the actual repo root paths, not the parent dir."""
        _, roots = discover_files(folder_of_repos, None, "code")
        root_names = {Path(r).name for r in roots}
        assert root_names == {"repo-a", "repo-b"}

    def test_code_folder_of_repos_respects_skip_git_roots(self, folder_of_repos):
        """Repos in skip_git_roots are excluded entirely."""
        repo_a_root = str(folder_of_repos / "repo-a")
        files, roots = discover_files(
            folder_of_repos, None, "code",
            skip_git_roots={repo_a_root},
        )
        names = {f.name for f in files}
        assert "repo-a_tracked.py" not in names
        assert "repo-b_tracked.py" in names
        assert len(roots) == 1
        assert Path(roots[0]).name == "repo-b"

    def test_code_nested_folder_of_repos(self, nested_folder_of_repos):
        """Org/repo layout: repos nested one level deep are discovered."""
        files, roots = discover_files(nested_folder_of_repos, None, "code")
        names = {f.name for f in files}
        assert "main.go" in names
        assert len(roots) == 2

    def test_code_gitignore_respected_in_folder_of_repos(self, folder_of_repos):
        """.gitignore inside each sub-repo is honoured."""
        files, _ = discover_files(folder_of_repos, None, "code")
        names = {f.name for f in files}
        assert "ignored.py" not in names

    def test_code_includes_git_tracked_dot_and_underscore_paths(self, git_repo_with_untracked):
        """Code corpora include tracked files under dot/underscore directories.

        git ls-files is the source of truth for code corpora, so the default
        skip_dir_prefixes=('_',) and dot-prefix conventions must NOT drop
        tracked paths like .github/workflows/ci.py or _internal/foo.py.
        """
        repo = git_repo_with_untracked
        (repo / ".github").mkdir()
        (repo / ".github" / "ci.py").write_text("ci = 1\n")
        (repo / "_internal").mkdir()
        (repo / "_internal" / "helper.py").write_text("h = 1\n")
        _git(repo, "add", ".github/ci.py", "_internal/helper.py")
        _git(repo, "commit", "-q", "-m", "add tracked dot/underscore dirs")

        files, _ = discover_files(repo, None, "code")
        names = {f.name for f in files}
        assert "ci.py" in names
        assert "helper.py" in names

    def test_code_markdown_still_skips_underscore_dirs(self, git_repo_with_untracked):
        """Non-code corpora keep the legacy underscore-prefix exclusion."""
        repo = git_repo_with_untracked
        (repo / "_internal").mkdir()
        (repo / "_internal" / "notes.md").write_text("# notes\n")
        _git(repo, "add", "_internal/notes.md")
        _git(repo, "commit", "-q", "-m", "add tracked underscore dir")

        files, _ = discover_files(repo, None, "markdown")
        names = {f.name for f in files}
        assert "notes.md" not in names

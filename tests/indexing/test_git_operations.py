"""Tests for git operations module."""

import os
import tempfile
import shutil
from pathlib import Path

import pytest
import git

from arcaneum.indexing.git_operations import GitProjectDiscovery, apply_git_metadata
from arcaneum.indexing.types import GitMetadata
from arcaneum.schema.document import DualIndexDocument


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def simple_repo(temp_dir):
    """Create a simple git repository for testing."""
    repo_path = os.path.join(temp_dir, "test-repo")
    os.makedirs(repo_path)

    # Initialize repo
    repo = git.Repo.init(repo_path)

    # Configure user
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@example.com").release()

    # Create a file and commit
    test_file = os.path.join(repo_path, "test.py")
    with open(test_file, 'w') as f:
        f.write("print('hello')\n")

    repo.index.add(["test.py"])
    repo.index.commit("Initial commit")

    # Add remote
    repo.create_remote("origin", "https://github.com/user/test-repo.git")

    return repo_path


@pytest.fixture
def detached_head_repo(temp_dir):
    """Create a repo with detached HEAD."""
    repo_path = os.path.join(temp_dir, "detached-repo")
    os.makedirs(repo_path)

    repo = git.Repo.init(repo_path)
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@example.com").release()

    # Create commits
    test_file = os.path.join(repo_path, "test.txt")
    with open(test_file, 'w') as f:
        f.write("content\n")

    repo.index.add(["test.txt"])
    commit = repo.index.commit("First commit")

    # Detach HEAD
    repo.git.checkout(commit.hexsha)

    return repo_path


@pytest.fixture
def nested_repos(temp_dir):
    """Create nested git repositories."""
    # Create structure:
    # temp_dir/
    #   repo1/
    #   projects/
    #     repo2/
    #     nested/
    #       repo3/

    repos = []

    # repo1 at depth 0
    repo1_path = os.path.join(temp_dir, "repo1")
    os.makedirs(repo1_path)
    repo1 = git.Repo.init(repo1_path)
    repo1.config_writer().set_value("user", "name", "Test").release()
    repo1.config_writer().set_value("user", "email", "test@example.com").release()
    repos.append(repo1_path)

    # repo2 at depth 1
    repo2_path = os.path.join(temp_dir, "projects", "repo2")
    os.makedirs(repo2_path)
    repo2 = git.Repo.init(repo2_path)
    repo2.config_writer().set_value("user", "name", "Test").release()
    repo2.config_writer().set_value("user", "email", "test@example.com").release()
    repos.append(repo2_path)

    # repo3 at depth 2
    repo3_path = os.path.join(temp_dir, "projects", "nested", "repo3")
    os.makedirs(repo3_path)
    repo3 = git.Repo.init(repo3_path)
    repo3.config_writer().set_value("user", "name", "Test").release()
    repo3.config_writer().set_value("user", "email", "test@example.com").release()
    repos.append(repo3_path)

    return temp_dir, repos


class TestGitProjectDiscovery:
    """Tests for GitProjectDiscovery class."""

    def test_find_single_repo(self, simple_repo):
        """Test finding a single repository."""
        discovery = GitProjectDiscovery()
        parent_dir = os.path.dirname(simple_repo)

        projects = discovery.find_git_projects(parent_dir)

        assert len(projects) == 1
        assert projects[0] == simple_repo

    def test_find_nested_repos_no_depth(self, nested_repos):
        """Test finding all nested repos without depth limit."""
        temp_dir, expected_repos = nested_repos
        discovery = GitProjectDiscovery()

        projects = discovery.find_git_projects(temp_dir)

        assert len(projects) == 3
        assert set(projects) == set(expected_repos)

    def test_find_nested_repos_with_depth(self, nested_repos):
        """Test finding repos with depth limit."""
        temp_dir, expected_repos = nested_repos
        discovery = GitProjectDiscovery()

        # Depth 0: only repo1
        projects_d0 = discovery.find_git_projects(temp_dir, depth=0)
        assert len(projects_d0) == 1
        assert projects_d0[0] == expected_repos[0]  # repo1

        # Depth 1: repo1 and repo2
        projects_d1 = discovery.find_git_projects(temp_dir, depth=1)
        assert len(projects_d1) == 2
        assert set(projects_d1) == {expected_repos[0], expected_repos[1]}

        # Depth 2: all three repos
        projects_d2 = discovery.find_git_projects(temp_dir, depth=2)
        assert len(projects_d2) == 3
        assert set(projects_d2) == set(expected_repos)

    def test_find_invalid_directory(self):
        """Test error handling for invalid directory."""
        discovery = GitProjectDiscovery()

        with pytest.raises(ValueError, match="does not exist"):
            discovery.find_git_projects("/nonexistent/path")

    def test_extract_metadata_basic(self, simple_repo):
        """Test basic metadata extraction."""
        discovery = GitProjectDiscovery()

        metadata = discovery.extract_metadata(simple_repo)

        assert metadata is not None
        assert isinstance(metadata, GitMetadata)
        assert metadata.project_root == simple_repo
        assert len(metadata.commit_hash) == 40  # Full SHA
        assert metadata.branch == "master" or metadata.branch == "main"
        assert metadata.project_name == "test-repo"
        assert "github.com/user/test-repo" in metadata.remote_url

    def test_extract_metadata_detached_head(self, detached_head_repo):
        """Test metadata extraction with detached HEAD."""
        discovery = GitProjectDiscovery()

        metadata = discovery.extract_metadata(detached_head_repo)

        assert metadata is not None
        assert metadata.branch.startswith("(detached-")
        assert len(metadata.commit_hash) == 40

    def test_extract_metadata_no_remote(self, temp_dir):
        """Test metadata extraction without remote."""
        repo_path = os.path.join(temp_dir, "no-remote-repo")
        os.makedirs(repo_path)

        repo = git.Repo.init(repo_path)
        repo.config_writer().set_value("user", "name", "Test").release()
        repo.config_writer().set_value("user", "email", "test@example.com").release()

        # Create commit without remote
        test_file = os.path.join(repo_path, "file.txt")
        with open(test_file, 'w') as f:
            f.write("content")
        repo.index.add(["file.txt"])
        repo.index.commit("Commit")

        discovery = GitProjectDiscovery()
        metadata = discovery.extract_metadata(repo_path)

        assert metadata is not None
        assert metadata.remote_url is None
        assert metadata.project_name == "no-remote-repo"  # Falls back to dirname

    def test_extract_metadata_invalid_repo(self, temp_dir):
        """Test metadata extraction from non-git directory."""
        non_repo = os.path.join(temp_dir, "not-a-repo")
        os.makedirs(non_repo)

        discovery = GitProjectDiscovery()
        metadata = discovery.extract_metadata(non_repo)

        assert metadata is None

    def test_sanitize_url_with_credentials(self):
        """Test URL sanitization removes credentials."""
        discovery = GitProjectDiscovery()

        # HTTPS with credentials
        url = "https://user:password@github.com/user/repo.git"
        sanitized = discovery._sanitize_url(url)
        assert "user:" not in sanitized
        assert "password" not in sanitized
        assert "github.com" in sanitized

    def test_sanitize_url_ssh(self):
        """Test SSH URLs are unchanged."""
        discovery = GitProjectDiscovery()

        url = "git@github.com:user/repo.git"
        sanitized = discovery._sanitize_url(url)
        assert sanitized == url

    def test_derive_project_name_from_url(self):
        """Test project name derivation from various URL formats."""
        discovery = GitProjectDiscovery()

        # HTTPS GitHub
        name1 = discovery._derive_project_name(
            "https://github.com/user/my-project.git",
            "/tmp/repo"
        )
        assert name1 == "my-project"

        # SSH GitHub
        name2 = discovery._derive_project_name(
            "git@github.com:user/another-project.git",
            "/tmp/repo"
        )
        assert name2 == "another-project"

        # No URL - fallback to dirname
        name3 = discovery._derive_project_name(None, "/tmp/fallback-repo")
        assert name3 == "fallback-repo"

    def test_composite_identifier(self, simple_repo):
        """Test composite identifier generation."""
        discovery = GitProjectDiscovery()
        metadata = discovery.extract_metadata(simple_repo)

        assert metadata is not None
        identifier = metadata.identifier
        assert "#" in identifier
        assert identifier.startswith("test-repo#")
        assert metadata.project_name in identifier
        assert metadata.branch in identifier

    def test_get_tracked_files_all(self, simple_repo):
        """Test getting all tracked files."""
        # Add more files
        repo = git.Repo(simple_repo)
        for filename in ["file1.py", "file2.java", "README.md"]:
            filepath = os.path.join(simple_repo, filename)
            with open(filepath, 'w') as f:
                f.write("content")
            repo.index.add([filename])
        repo.index.commit("Add files")

        discovery = GitProjectDiscovery()
        tracked_files = discovery.get_tracked_files(simple_repo)

        assert len(tracked_files) >= 4  # test.py + 3 new files
        assert all(os.path.isabs(f) for f in tracked_files)
        assert any("test.py" in f for f in tracked_files)

    def test_get_tracked_files_filtered(self, simple_repo):
        """Test getting tracked files with extension filter."""
        # Add files with different extensions
        repo = git.Repo(simple_repo)
        files = {
            "script.py": "python",
            "app.java": "java",
            "main.js": "javascript",
            "README.md": "markdown"
        }
        for filename, content in files.items():
            filepath = os.path.join(simple_repo, filename)
            with open(filepath, 'w') as f:
                f.write(content)
            repo.index.add([filename])
        repo.index.commit("Add multiple files")

        discovery = GitProjectDiscovery()

        # Filter for Python only
        py_files = discovery.get_tracked_files(simple_repo, extensions=['.py'])
        assert len(py_files) >= 2  # test.py + script.py
        assert all(f.endswith('.py') for f in py_files)

        # Filter for Java only
        java_files = discovery.get_tracked_files(simple_repo, extensions=['.java'])
        assert len(java_files) >= 1
        assert all(f.endswith('.java') for f in java_files)

    def test_get_tracked_files_respects_gitignore(self, simple_repo):
        """Test that get_tracked_files respects .gitignore."""
        repo = git.Repo(simple_repo)

        # Create .gitignore
        gitignore_path = os.path.join(simple_repo, ".gitignore")
        with open(gitignore_path, 'w') as f:
            f.write("*.ignored\n")
        repo.index.add([".gitignore"])
        repo.index.commit("Add gitignore")

        # Create ignored file (not added to git)
        ignored_file = os.path.join(simple_repo, "secret.ignored")
        with open(ignored_file, 'w') as f:
            f.write("secret content")

        # Create tracked file
        tracked_file = os.path.join(simple_repo, "tracked.txt")
        with open(tracked_file, 'w') as f:
            f.write("tracked content")
        repo.index.add(["tracked.txt"])
        repo.index.commit("Add tracked file")

        discovery = GitProjectDiscovery()
        tracked_files = discovery.get_tracked_files(simple_repo)

        # Ignored file should not be in the list
        assert not any("secret.ignored" in f for f in tracked_files)
        # Tracked file should be in the list
        assert any("tracked.txt" in f for f in tracked_files)


class TestApplyGitMetadata:
    """Tests for apply_git_metadata utility function."""

    def test_apply_git_metadata_all_fields(self):
        """Test that all git metadata fields are applied to document."""
        doc = DualIndexDocument(content="test content")
        git_meta = GitMetadata(
            project_root="/path/to/repo",
            commit_hash="abc123def456789012345678901234567890abcd",
            branch="main",
            project_name="test-project",
            remote_url="https://github.com/user/test-project.git"
        )

        apply_git_metadata(doc, git_meta)

        assert doc.project == "test-project"
        assert doc.branch == "main"
        assert doc.git_project_identifier == "test-project#main"
        assert doc.git_commit_hash == "abc123def456789012345678901234567890abcd"
        assert doc.git_remote_url == "https://github.com/user/test-project.git"

    def test_apply_git_metadata_no_remote_url(self):
        """Test applying metadata when remote_url is None."""
        doc = DualIndexDocument(content="test content")
        git_meta = GitMetadata(
            project_root="/path/to/repo",
            commit_hash="abc123def456789012345678901234567890abcd",
            branch="feature-branch",
            project_name="local-project",
            remote_url=None
        )

        apply_git_metadata(doc, git_meta)

        assert doc.project == "local-project"
        assert doc.branch == "feature-branch"
        assert doc.git_project_identifier == "local-project#feature-branch"
        assert doc.git_commit_hash == "abc123def456789012345678901234567890abcd"
        assert doc.git_remote_url is None

    def test_apply_git_metadata_with_extracted_metadata(self, simple_repo):
        """Test applying real extracted metadata to a document."""
        discovery = GitProjectDiscovery()
        git_meta = discovery.extract_metadata(simple_repo)
        assert git_meta is not None

        doc = DualIndexDocument(content="test content")
        apply_git_metadata(doc, git_meta)

        assert doc.project == git_meta.project_name
        assert doc.branch == git_meta.branch
        assert doc.git_project_identifier == git_meta.identifier
        assert doc.git_commit_hash == git_meta.commit_hash
        assert doc.git_remote_url == git_meta.remote_url

    def test_apply_git_metadata_with_version_identifier(self):
        """Test applying git metadata with include_version_id=True."""
        git_meta = GitMetadata(
            project_root="/path/to/repo",
            commit_hash="abc123def456abc123def456abc123def456abc1",
            branch="main",
            project_name="myproject",
            remote_url="https://github.com/org/myproject.git"
        )

        doc = DualIndexDocument(content="test content")
        apply_git_metadata(doc, git_meta, include_version_id=True)

        assert doc.project == "myproject"
        assert doc.branch == "main"
        assert doc.git_project_identifier == "myproject#main"
        assert doc.git_commit_hash == "abc123def456abc123def456abc123def456abc1"
        assert doc.git_remote_url == "https://github.com/org/myproject.git"
        assert doc.git_version_identifier == "myproject#main@abc123d"

    def test_apply_git_metadata_without_version_identifier(self):
        """Test applying git metadata with include_version_id=False (default)."""
        git_meta = GitMetadata(
            project_root="/path/to/repo",
            commit_hash="abc123def456abc123def456abc123def456abc1",
            branch="main",
            project_name="myproject",
            remote_url=None
        )

        doc = DualIndexDocument(content="test content")
        apply_git_metadata(doc, git_meta)  # Default is include_version_id=False

        assert doc.git_version_identifier is None


class TestGitSyncHelpers:
    """Tests for git-aware sync helper functions in cli/sync.py."""

    def test_find_git_root_in_repo(self, simple_repo):
        """Test _find_git_root returns correct root for path inside repo."""
        from arcaneum.cli.sync import _find_git_root

        # Test with the repo root
        assert _find_git_root(Path(simple_repo)) == simple_repo

        # Test with a file inside the repo
        test_file = Path(simple_repo) / "test.txt"
        assert _find_git_root(test_file) == simple_repo

    def test_find_git_root_outside_repo(self, temp_dir):
        """Test _find_git_root returns None for non-git path."""
        from arcaneum.cli.sync import _find_git_root

        # temp_dir is not a git repo
        result = _find_git_root(Path(temp_dir))
        assert result is None

    def test_group_paths_by_git_root(self, simple_repo, temp_dir):
        """Test _group_paths_by_git_root groups correctly."""
        from arcaneum.cli.sync import _group_paths_by_git_root

        # Create some paths
        repo_file = Path(simple_repo) / "test.txt"
        non_git_file = Path(temp_dir) / "other.txt"
        non_git_file.touch()

        paths = [repo_file, non_git_file]
        groups = _group_paths_by_git_root(paths)

        # Should have two groups: git repo and None (non-git)
        assert simple_repo in groups
        assert repo_file in groups[simple_repo]
        assert None in groups
        assert non_git_file in groups[None]

    def test_group_paths_multiple_repos(self, simple_repo, temp_dir):
        """Test _group_paths_by_git_root with multiple git repos."""
        from arcaneum.cli.sync import _group_paths_by_git_root

        # Create a second git repo
        second_repo = Path(temp_dir) / "second-repo"
        second_repo.mkdir()
        git.Repo.init(second_repo)

        paths = [
            Path(simple_repo) / "file1.py",
            Path(simple_repo) / "file2.py",
            second_repo / "file3.py",
        ]

        groups = _group_paths_by_git_root(paths)

        # Should have two groups, one for each repo
        assert len([k for k in groups.keys() if k is not None]) == 2
        assert simple_repo in groups
        assert str(second_repo) in groups
        assert len(groups[simple_repo]) == 2
        assert len(groups[str(second_repo)]) == 1

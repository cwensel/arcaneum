"""
Git operations for source code indexing.

This module provides git-aware project discovery and metadata extraction
with robust error handling for edge cases (RDR-005).
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse, urlunparse
import logging

import git
from git.exc import GitCommandError, InvalidGitRepositoryError

from .types import GitMetadata

logger = logging.getLogger(__name__)


def apply_git_metadata(doc, git_metadata: GitMetadata) -> None:
    """Apply GitMetadata to a document object.

    Centralizes git metadata assignment to prevent inconsistencies between indexing paths.
    Works with DualIndexDocument (sync.py) which uses attribute assignment.

    Args:
        doc: Document object with git metadata attributes (e.g., DualIndexDocument)
        git_metadata: GitMetadata containing project info, branch, commit hash, etc.
    """
    doc.project = git_metadata.project_name
    doc.branch = git_metadata.branch
    doc.git_project_identifier = git_metadata.identifier
    doc.git_commit_hash = git_metadata.commit_hash
    doc.git_remote_url = git_metadata.remote_url


class GitProjectDiscovery:
    """Discovers git projects and extracts metadata with robust error handling.

    Handles edge cases:
    - Detached HEAD states
    - Shallow clones
    - Missing remotes
    - Submodules
    - Credential sanitization
    - Timeout protection
    """

    DEFAULT_TIMEOUT = 5.0  # seconds

    def find_git_projects(
        self,
        input_path: str,
        depth: Optional[int] = None
    ) -> List[str]:
        """Find all .git directories with optional depth control.

        Args:
            input_path: Directory to search for git repositories
            depth: Maximum directory depth to search (None = unlimited)
                  Note: depth=0 means repos directly in input_path
                        depth=1 means repos in input_path or one level down
                  (maxdepth = depth + 2 because we search for .git dirs)

        Returns:
            Sorted list of absolute paths to git project roots

        Example:
            >>> discovery = GitProjectDiscovery()
            >>> projects = discovery.find_git_projects("/home/code", depth=2)
            ['/home/code/project-a', '/home/code/projects/project-b']
        """
        input_path = os.path.abspath(input_path)

        if not os.path.isdir(input_path):
            raise ValueError(f"Input path does not exist or is not a directory: {input_path}")

        # Build find command
        find_cmd = ["find", input_path]
        if depth is not None:
            # Add 2 to depth: +1 for project dir, +1 for .git dir
            # Example: depth=0 means repos directly in input_path
            #   input_path/repo/.git requires maxdepth=2
            find_cmd.extend(["-maxdepth", str(depth + 2)])
        find_cmd.extend(["-name", ".git", "-type", "d"])

        try:
            result = subprocess.run(
                find_cmd,
                capture_output=True,
                text=True,
                timeout=self.DEFAULT_TIMEOUT * 10  # Generous timeout for find
            )

            if result.returncode != 0:
                logger.warning(f"Find command failed: {result.stderr}")
                return []

            # Extract project roots (parent directories of .git)
            git_dirs = result.stdout.strip().split('\n')
            project_roots = []

            for git_dir in git_dirs:
                if git_dir:  # Skip empty lines
                    project_root = os.path.dirname(git_dir)
                    project_roots.append(project_root)

            return sorted(set(project_roots))

        except subprocess.TimeoutExpired:
            logger.error(f"Find command timed out searching {input_path}")
            return []
        except Exception as e:
            logger.error(f"Error finding git projects: {e}")
            return []

    def extract_metadata(self, project_root: str) -> Optional[GitMetadata]:
        """Extract git metadata with robust error handling.

        Args:
            project_root: Path to git repository root

        Returns:
            GitMetadata object or None if extraction fails

        Handles:
            - Detached HEAD: Uses git describe --tags or commit SHA
            - Shallow clones: Detects via .git/shallow file
            - Missing remotes: Falls back to "unknown"
            - Credential sanitization: Removes user:pass from URLs
        """
        try:
            repo = git.Repo(project_root)

            # Extract commit hash
            try:
                commit_hash = repo.head.commit.hexsha
            except ValueError:
                logger.error(f"Cannot get commit hash for {project_root}")
                return None

            # Extract branch with detached HEAD handling
            branch = self._extract_branch(repo)

            # Extract remote URL with credential sanitization
            remote_url = self._extract_remote_url(repo)

            # Derive project name
            project_name = self._derive_project_name(remote_url, project_root)

            # Detect shallow clone
            is_shallow = self._is_shallow_clone(repo)
            if is_shallow:
                logger.debug(f"Shallow clone detected: {project_root}")

            return GitMetadata(
                project_root=project_root,
                commit_hash=commit_hash,
                branch=branch,
                project_name=project_name,
                remote_url=remote_url
            )

        except InvalidGitRepositoryError:
            logger.warning(f"Not a valid git repository: {project_root}")
            return None
        except Exception as e:
            logger.error(f"Error extracting metadata from {project_root}: {e}")
            return None

    def _extract_branch(self, repo: git.Repo) -> str:
        """Extract branch name with detached HEAD handling.

        Returns:
            - Branch name for normal HEAD
            - (tag)TAG_NAME for detached HEAD at tag
            - (detached-SHA) for detached HEAD without tag
        """
        try:
            # Normal branch
            return repo.active_branch.name
        except TypeError:
            # Detached HEAD - try to find tag
            try:
                tag = repo.git.describe('--tags', '--exact-match')
                return f"(tag){tag}"
            except GitCommandError:
                # No exact tag match - use commit SHA
                commit_sha = repo.head.commit.hexsha[:12]
                return f"(detached-{commit_sha})"

    def _extract_remote_url(self, repo: git.Repo) -> Optional[str]:
        """Extract remote URL with credential sanitization.

        Priority order:
        1. origin remote
        2. upstream remote
        3. First available remote
        4. None if no remotes

        Sanitizes credentials from URLs like:
        - https://user:pass@github.com/repo -> https://github.com/repo
        - git@github.com:user/repo.git -> git@github.com:user/repo.git (unchanged)
        """
        try:
            # Try origin first
            try:
                remote_url = repo.remote('origin').url
            except ValueError:
                # Try upstream
                try:
                    remote_url = repo.remote('upstream').url
                except ValueError:
                    # Use first available remote
                    if repo.remotes:
                        remote_url = repo.remotes[0].url
                    else:
                        return None

            # Sanitize credentials
            return self._sanitize_url(remote_url)

        except Exception as e:
            logger.debug(f"Could not extract remote URL: {e}")
            return None

    def _sanitize_url(self, url: str) -> str:
        """Remove credentials from URL.

        Args:
            url: Git remote URL (possibly with credentials)

        Returns:
            Sanitized URL without credentials

        Examples:
            https://user:pass@github.com/repo -> https://github.com/repo
            git@github.com:user/repo.git -> git@github.com:user/repo.git
        """
        if not url:
            return url

        # Handle HTTPS URLs with credentials
        if url.startswith('https://') or url.startswith('http://'):
            try:
                parsed = urlparse(url)
                # Remove username and password
                sanitized = urlunparse((
                    parsed.scheme,
                    parsed.hostname + (f":{parsed.port}" if parsed.port else ""),
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment
                ))
                return sanitized
            except Exception:
                return url

        # SSH URLs (git@...) don't typically expose credentials
        return url

    def _derive_project_name(self, remote_url: Optional[str], project_root: str) -> str:
        """Derive project name from remote URL or directory name.

        Priority:
        1. Extract from remote URL (e.g., github.com/user/PROJECT.git -> PROJECT)
        2. Use directory basename as fallback

        Args:
            remote_url: Git remote URL (may be None)
            project_root: Path to git repository

        Returns:
            Project name string
        """
        if remote_url:
            # Try to extract project name from URL
            # Examples:
            # - https://github.com/user/my-project.git -> my-project
            # - git@github.com:user/my-project.git -> my-project

            # Remove .git suffix
            url_no_git = remote_url.removesuffix('.git')

            # Extract last path component
            if '/' in url_no_git:
                project_name = url_no_git.rsplit('/', 1)[-1]
            elif ':' in url_no_git:  # SSH format git@host:path
                project_name = url_no_git.rsplit(':', 1)[-1]
            else:
                project_name = os.path.basename(project_root)

            # Clean up
            project_name = project_name.rstrip('/')
            if project_name:
                return project_name

        # Fallback to directory basename
        return os.path.basename(project_root)

    def _is_shallow_clone(self, repo: git.Repo) -> bool:
        """Detect if repository is a shallow clone.

        Args:
            repo: GitPython repository object

        Returns:
            True if shallow clone, False otherwise
        """
        shallow_file = os.path.join(repo.git_dir, 'shallow')
        return os.path.exists(shallow_file)

    def get_tracked_files(
        self,
        project_root: str,
        extensions: Optional[List[str]] = None
    ) -> List[str]:
        """Get list of git-tracked files respecting .gitignore.

        Uses git ls-files to respect:
        - .gitignore patterns
        - .git/info/exclude
        - Global gitignore

        Args:
            project_root: Path to git repository root
            extensions: List of file extensions to filter (e.g., ['.py', '.java'])
                       If None, returns all tracked files

        Returns:
            List of absolute paths to tracked files

        Example:
            >>> discovery = GitProjectDiscovery()
            >>> files = discovery.get_tracked_files('/repo', extensions=['.py', '.java'])
        """
        try:
            repo = git.Repo(project_root)

            # Build git ls-files command
            if extensions:
                # Build pattern for each extension
                patterns = []
                for ext in extensions:
                    patterns.append(f"*{ext}")

                # Get tracked files matching patterns
                tracked_files = []
                for pattern in patterns:
                    try:
                        files = repo.git.ls_files(pattern).split('\n')
                        tracked_files.extend([f for f in files if f])
                    except GitCommandError:
                        continue
            else:
                # Get all tracked files
                tracked_files = repo.git.ls_files().split('\n')
                tracked_files = [f for f in tracked_files if f]

            # Convert relative paths to absolute
            absolute_paths = [
                os.path.join(project_root, rel_path)
                for rel_path in tracked_files
                if rel_path
            ]

            return absolute_paths

        except Exception as e:
            logger.error(f"Error getting tracked files from {project_root}: {e}")
            return []

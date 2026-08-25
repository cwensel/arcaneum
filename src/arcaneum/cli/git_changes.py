"""Compute the files a git revision or range touched (kata vq0n).

`arc corpus sync --changed-since <rev>` and the installed git hook both need
the same question answered: which paths did this commit (or this range) add,
modify, or delete? Asking git is far cheaper than walking a large working tree
just to have mtime+size change detection reject almost every file.

`git diff-tree -z --name-status` is the source of truth. NUL separation means
paths containing spaces, quotes, or newlines survive intact — git's default
output would quote and escape them.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .errors import InvalidArgumentError

logger = logging.getLogger(__name__)

GIT_TIMEOUT_SECONDS = 60

# Statuses that carry two paths: <old> then <new>.
_TWO_PATH_STATUSES = ("R", "C")


@dataclass(frozen=True)
class GitChanges:
    """Paths a revision touched, split by what sync must do with them."""

    changed: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.changed or self.removed)


def parse_name_status(raw: str) -> GitChanges:
    """Parse NUL-separated `git diff-tree --name-status -z` output.

    The stream is a flat sequence of records: a status field followed by one
    path, or by two paths for renames (R) and copies (C).

    A path can appear more than once across a multi-commit range — added then
    deleted, or deleted then re-added. The last record wins, matching the tree
    state at the end of the range, so a file resurrected in a later commit is
    re-indexed rather than dropped.
    """
    fields = [f for f in raw.split("\0") if f != ""]
    # Ordered map path -> is_removed; re-assignment lets a later record override.
    verdicts: Dict[str, bool] = {}

    i = 0
    while i < len(fields):
        status = fields[i]
        code = status[:1]
        i += 1

        if code in _TWO_PATH_STATUSES:
            if i + 1 >= len(fields):
                logger.debug("Truncated %s record in diff output; ignoring", status)
                break
            old_path, new_path = fields[i], fields[i + 1]
            i += 2
            # A rename moves content: the new path needs indexing and the old
            # one must be dropped. A copy leaves the source in place, so only
            # the new path is new work.
            verdicts[new_path] = False
            if code == "R":
                verdicts[old_path] = True
            continue

        if i >= len(fields):
            logger.debug("Truncated %s record in diff output; ignoring", status)
            break
        path = fields[i]
        i += 1

        if code == "D":
            verdicts[path] = True
        elif code in ("A", "M", "T"):
            verdicts[path] = False
        else:
            # U (unmerged) and anything git adds later: not safe to act on.
            logger.debug("Ignoring unhandled diff status %r for %s", status, path)

    return GitChanges(
        changed=[p for p, removed in verdicts.items() if not removed],
        removed=[p for p, removed in verdicts.items() if removed],
    )


def _run_git(repo: Path, args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
    )


def repo_root(path: Union[str, Path]) -> Optional[Path]:
    """Return the working-tree root containing `path`, or None if not a repo."""
    target = Path(path)
    if not target.is_dir():
        target = target.parent
    try:
        result = _run_git(target, ["rev-parse", "--show-toplevel"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("git rev-parse failed for %s: %s", path, exc)
        return None
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    return Path(root) if root else None


def _validate_revision(revision: str) -> None:
    if revision.startswith("-"):
        # git would read this as an option; reject rather than let a caller
        # smuggle flags like --output through a revision argument.
        raise InvalidArgumentError(
            f"Invalid revision {revision!r}: revisions cannot start with '-'."
        )


def changes_between(
    repo: Union[str, Path], old: str, new: str
) -> GitChanges:
    """Compare two commits' trees directly, ignoring the path between them.

    This is what a history rewrite needs. `A..B` walks the commits in the range
    and diffs each against its own parent, so a file whose only commit was
    *dropped* by a rebase appears in no walked diff -- yet it has just vanished
    from the working tree and must be de-indexed. Comparing the two tip trees
    reports it as a deletion.

    It is also cheaper and more truthful for the common case: a rebase that
    only re-signs commits rewrites every SHA but changes no tree, so the
    endpoint diff is empty where a walk would report every touched file.

    Raises:
        InvalidArgumentError: not a repo, or either revision is unknown.
    """
    _validate_revision(old)
    _validate_revision(new)

    root = repo_root(repo)
    if root is None:
        raise InvalidArgumentError(f"Not a git repository: {repo}")

    for revision in (old, new):
        verify = _run_git(root, ["rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}"])
        if verify.returncode != 0:
            raise InvalidArgumentError(f"Unknown git revision {revision!r} in {root}.")

    # Two revisions as separate arguments (not "A..B") makes diff-tree compare
    # the trees rather than walk the commits between them.
    args = ["diff-tree", "-z", "--no-commit-id", "--name-status", "-r", "-M", old, new]
    try:
        result = _run_git(root, args)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise InvalidArgumentError(f"Could not read git changes in {root}: {exc}") from exc

    if result.returncode != 0:
        raise InvalidArgumentError(
            f"Could not diff {old!r}..{new!r} in {root}: {result.stderr.strip()}"
        )

    relative = parse_name_status(result.stdout)
    return GitChanges(
        changed=[str(root / p) for p in relative.changed],
        removed=[str(root / p) for p in relative.removed],
    )


def changes_since(repo: Union[str, Path], revision: str) -> GitChanges:
    """List paths touched by `revision` (a single commit or an A..B range).

    Args:
        repo: Any path inside the working tree.
        revision: A commit-ish ("HEAD"), or a range ("ORIG_HEAD..HEAD").

    Returns:
        GitChanges with absolute paths, in the order git reported them.

    Raises:
        InvalidArgumentError: not a git repository, or the revision is unknown.
    """
    if revision.startswith("-"):
        # git would read this as an option; reject rather than let a caller
        # smuggle flags like --output through a revision argument.
        raise InvalidArgumentError(
            f"Invalid revision {revision!r}: revisions cannot start with '-'."
        )

    root = repo_root(repo)
    if root is None:
        raise InvalidArgumentError(f"Not a git repository: {repo}")

    verify = _run_git(root, ["rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}"])
    is_range = ".." in revision
    if not is_range and verify.returncode != 0:
        raise InvalidArgumentError(
            f"Unknown git revision {revision!r} in {root}. "
            "Pass a commit, tag, branch, or range like 'ORIG_HEAD..HEAD'."
        )

    # --root makes the initial commit diff against the empty tree, so a hook
    # firing on a repo's first commit still reports every file as added
    # instead of silently indexing nothing.
    args = [
        "diff-tree",
        "-z",
        "--no-commit-id",
        "--name-status",
        "-r",
        "-M",
        "--root",
        revision,
    ]
    try:
        result = _run_git(root, args)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise InvalidArgumentError(f"Could not read git changes in {root}: {exc}") from exc

    if result.returncode != 0:
        raise InvalidArgumentError(
            f"Unknown git revision {revision!r} in {root}: {result.stderr.strip()}"
        )

    relative = parse_name_status(result.stdout)
    return GitChanges(
        changed=[str(root / p) for p in relative.changed],
        removed=[str(root / p) for p in relative.removed],
    )

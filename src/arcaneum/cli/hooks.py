"""Install git hooks that keep a corpus in sync with a working repo (kata vq0n).

Without this, an indexed repo drifts behind its source tree between manual
`arc corpus sync` runs. The installed hook asks git which paths the commit
touched — far cheaper than walking a large tree — spools them, and hands off to
a background worker (`arc corpus sync --drain-spool`).

Two constraints shape the generated script:

- **It must never break git.** Every path exits 0, output is suppressed, and
  the work happens detached. A missing `arc` on PATH is not an error.
- **It must coexist.** The block is delimited by marker comments and appended
  to whatever hook already exists, so a user's own hook keeps working and
  uninstall removes only our block. `core.hooksPath` is respected.
"""

from __future__ import annotations

import logging
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from .errors import InvalidArgumentError

logger = logging.getLogger(__name__)

BLOCK_START = "# >>> arcaneum corpus sync ({corpus}) >>>"
BLOCK_END = "# <<< arcaneum corpus sync ({corpus}) <<<"

SHEBANG = "#!/bin/sh"

# Hook points that fire *after* the ref moved, so HEAD (or ORIG_HEAD..HEAD)
# names the change we want to index. Pre-hooks would index the wrong tree, and
# a failing pre-hook aborts the git operation — which this must never do.
SUPPORTED_HOOKS = ("post-commit", "post-merge", "post-checkout", "post-rewrite")

# The revision each hook point should diff. post-merge and post-rewrite can
# move HEAD by several commits, so they span from ORIG_HEAD when it exists.
_HOOK_REVISIONS = {
    "post-commit": "HEAD",
    "post-merge": "ORIG_HEAD..HEAD",
    "post-checkout": "ORIG_HEAD..HEAD",
    "post-rewrite": "ORIG_HEAD..HEAD",
}


@dataclass(frozen=True)
class InstalledHook:
    """One arcaneum block found in one hook script."""

    corpus: str
    hook: str
    path: Path


def _run_git(repo: Union[str, Path], args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30
    )


def resolve_hooks_dir(repo: Union[str, Path]) -> Path:
    """Return the directory git will actually look in for hook scripts.

    Honors `core.hooksPath`, which relocates hooks away from `.git/hooks`;
    installing into `.git/hooks` on such a repo would silently do nothing.
    """
    try:
        toplevel = _run_git(repo, ["rev-parse", "--show-toplevel"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise InvalidArgumentError(f"Could not run git in {repo}: {exc}") from exc
    if toplevel.returncode != 0:
        raise InvalidArgumentError(
            f"Not a git repository: {repo}. Run this inside a repo, or pass --repo."
        )
    root = Path(toplevel.stdout.strip())

    configured = _run_git(root, ["config", "--get", "core.hooksPath"])
    if configured.returncode == 0 and configured.stdout.strip():
        hooks_path = Path(configured.stdout.strip())
        return hooks_path if hooks_path.is_absolute() else root / hooks_path

    git_dir = _run_git(root, ["rev-parse", "--git-common-dir"])
    if git_dir.returncode != 0 or not git_dir.stdout.strip():
        raise InvalidArgumentError(f"Could not resolve the git directory for {root}")
    resolved = Path(git_dir.stdout.strip())
    if not resolved.is_absolute():
        resolved = root / resolved
    return resolved / "hooks"


def repo_root(repo: Union[str, Path]) -> Path:
    result = _run_git(repo, ["rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        raise InvalidArgumentError(f"Not a git repository: {repo}")
    return Path(result.stdout.strip())


def render_block(corpus: str, hook: str, repo: Path, *, spawn: bool = True) -> str:
    """Render the guarded shell block for one corpus in one hook script.

    The script does only the cheap half inline: ask git for the touched paths
    and spool them. Indexing is handed to a detached worker so the commit
    returns immediately, and every failure mode is swallowed — a hook that
    exits non-zero would break the user's git workflow.
    """
    revision = _HOOK_REVISIONS.get(hook, "HEAD")
    start = BLOCK_START.format(corpus=corpus)
    end = BLOCK_END.format(corpus=corpus)

    # Single-quoted in the script, so a literal quote must be escaped for sh.
    corpus_sh = corpus.replace("'", "'\\''")
    repo_sh = str(repo).replace("'", "'\\''")

    if spawn:
        # setsid detaches from the terminal so the worker outlives the commit;
        # where it is unavailable (macOS lacks it by default) a plain
        # background nohup is equivalent for our purposes.
        spawn_lines = """    if command -v setsid >/dev/null 2>&1; then
        setsid "$_arc_bin" corpus sync "$_arc_corpus" --drain-spool \\
            >>"$_arc_log" 2>&1 < /dev/null &
    else
        nohup "$_arc_bin" corpus sync "$_arc_corpus" --drain-spool \\
            >>"$_arc_log" 2>&1 < /dev/null &
    fi"""
    else:
        # --no-spawn: queue only. Used by tests and by anyone who prefers to
        # drain on a schedule or via the OS watcher (`--service`).
        spawn_lines = "    : # --no-spawn: leave draining to the service or a manual run"

    return f"""{start}
# Managed by `arc corpus hook`. Edit via that command, not by hand.
# Queues the paths this operation touched, then drains them in the background.
# Every path exits 0: a hook must never fail the git command that ran it.
_arc_corpus='{corpus_sh}'
_arc_repo='{repo_sh}'
_arc_rev='{revision}'
_arc_log="${{XDG_STATE_HOME:-$HOME/.local/state}}/arcaneum/hook.log"

_arc_sync() {{
    _arc_bin=$(command -v arc 2>/dev/null) || return 0
    mkdir -p "$(dirname "$_arc_log")" 2>/dev/null || true

    # ORIG_HEAD is absent on a fresh clone's first merge; fall back to HEAD.
    case "$_arc_rev" in
        *ORIG_HEAD*)
            git -C "$_arc_repo" rev-parse --verify --quiet ORIG_HEAD >/dev/null 2>&1 \\
                || _arc_rev='HEAD'
            ;;
    esac

    "$_arc_bin" corpus hook spool "$_arc_corpus" \\
        --repo "$_arc_repo" --changed-since "$_arc_rev" >>"$_arc_log" 2>&1 || return 0

{spawn_lines}
}}

_arc_sync || true
{end}
"""


def _block_pattern(corpus: str) -> re.Pattern:
    return re.compile(
        re.escape(BLOCK_START.format(corpus=corpus))
        + r".*?"
        + re.escape(BLOCK_END.format(corpus=corpus))
        + r"\n?",
        re.DOTALL,
    )


def install(
    corpus: str,
    repo: Union[str, Path],
    hook: str = "post-commit",
    *,
    spawn: bool = True,
) -> Path:
    """Install (or refresh) the block for `corpus` in `hook`. Returns its path."""
    if hook not in SUPPORTED_HOOKS:
        raise InvalidArgumentError(
            f"Unsupported hook {hook!r}. Choose one of: {', '.join(SUPPORTED_HOOKS)}."
        )

    root = repo_root(repo)
    hooks_dir = resolve_hooks_dir(root)
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / hook

    block = render_block(corpus, hook, root, spawn=spawn)

    if hook_path.exists():
        existing = hook_path.read_text()
        pattern = _block_pattern(corpus)
        if pattern.search(existing):
            # Refresh in place so re-installing never duplicates the block.
            body = pattern.sub(block, existing, count=1)
        else:
            separator = "" if existing.endswith("\n") else "\n"
            body = f"{existing}{separator}\n{block}"
    else:
        body = f"{SHEBANG}\n\n{block}"

    hook_path.write_text(body)
    mode = hook_path.stat().st_mode
    hook_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return hook_path


def uninstall(corpus: str, repo: Union[str, Path], hook: Optional[str] = None) -> List[Path]:
    """Remove the block for `corpus`, leaving any other content intact.

    Returns the hook files that changed. Removing a block we never installed is
    a no-op, not an error, so uninstall is safe to run twice.
    """
    root = repo_root(repo)
    hooks_dir = resolve_hooks_dir(root)
    targets = [hook] if hook else list(SUPPORTED_HOOKS)
    pattern = _block_pattern(corpus)
    changed: List[Path] = []

    for name in targets:
        hook_path = hooks_dir / name
        if not hook_path.is_file():
            continue
        existing = hook_path.read_text()
        if not pattern.search(existing):
            continue

        remaining = pattern.sub("", existing)
        # If nothing but a shebang and blank lines is left, the file is ours
        # alone — remove it rather than leaving an inert stub behind.
        meaningful = [
            line
            for line in remaining.splitlines()
            if line.strip() and not line.startswith("#!")
        ]
        if meaningful:
            hook_path.write_text(remaining.rstrip("\n") + "\n")
        else:
            hook_path.unlink()
        changed.append(hook_path)

    return changed


def list_installed(repo: Union[str, Path]) -> List[InstalledHook]:
    """Every arcaneum block installed in this repo, across hook points."""
    root = repo_root(repo)
    hooks_dir = resolve_hooks_dir(root)
    if not hooks_dir.is_dir():
        return []

    # BLOCK_START with the corpus name recovered from the marker itself.
    marker = re.compile(
        re.escape(BLOCK_START.split("{corpus}")[0])
        + r"(?P<corpus>.+?)"
        + re.escape(BLOCK_START.split("{corpus}")[1])
    )

    found: List[InstalledHook] = []
    for name in SUPPORTED_HOOKS:
        hook_path = hooks_dir / name
        if not hook_path.is_file():
            continue
        try:
            body = hook_path.read_text()
        except OSError as exc:
            logger.debug("Could not read %s: %s", hook_path, exc)
            continue
        for match in marker.finditer(body):
            found.append(
                InstalledHook(corpus=match.group("corpus"), hook=name, path=hook_path)
            )
    return found

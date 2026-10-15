"""Install git hooks that keep a corpus in sync with a working repo (kata vq0n).

Without this, an indexed repo drifts behind its source tree between manual
`arc corpus sync` runs. The installed hook asks git which paths changed — far
cheaper than walking a large tree — spools them, and hands off to a background
worker (`arc corpus sync --drain-spool`).

Each corpus is pinned to one branch (kata vr9b). Every hook point runs the same
gate: when the repo has that branch checked out and its tip differs from the
last spooled commit (the mark), spool the tree diff mark..tip and advance the
mark. Asking "where is the branch now?" instead of "what did this event do?"
makes every way a branch moves look alike -- commit, merge, rebase, reset, or a
tool's `update-ref` -- and lets the next event catch up any that were missed.

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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from . import spool
from .errors import InvalidArgumentError
from .git_changes import (
    GitChanges,
    branch_tip,
    changes_between,
    changes_since,
    current_branch,
    is_commit,
)

logger = logging.getLogger(__name__)

BLOCK_START = "# >>> arcaneum corpus sync ({corpus}) >>>"
BLOCK_END = "# <<< arcaneum corpus sync ({corpus}) <<<"

SHEBANG = "#!/bin/sh"

# Hook points that fire *after* refs moved, so the branch tip is final. Pre-hooks
# would read a tree about to change, and a failing pre-hook aborts the git
# operation -- which this must never do.
#
# reference-transaction (git 2.28+) is the only hook that fires for every way a
# branch moves, including plumbing (`commit-tree` + `update-ref`), `reset` and
# `branch -f`, none of which run a post-* hook.
SUPPORTED_HOOKS = (
    "reference-transaction",
    "post-commit",
    "post-merge",
    "post-checkout",
    "post-rewrite",
)

REFERENCE_TRANSACTION_MIN_GIT = (2, 28)


@dataclass(frozen=True)
class InstalledHook:
    """One arcaneum block found in one hook script."""

    corpus: str
    hook: str
    path: Path
    branch: Optional[str] = None


# Extensions that decide a repo's corpus type. Not exhaustive -- this only has
# to be right often enough to make a good default that the user can override.
_CODE_EXTENSIONS = frozenset(
    {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".scala",
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".m",
        ".sh",
        ".bash",
        ".zsh",
        ".sql",
        ".proto",
        ".ex",
        ".exs",
        ".clj",
        ".hs",
    }
)
_PDF_EXTENSIONS = frozenset({".pdf"})

# Enough files to judge the mix without walking a huge tree.
_INFERENCE_SAMPLE_LIMIT = 2000


def infer_corpus_type(repo: Union[str, Path]) -> str:
    """Guess whether a tree is best indexed as code, markdown, or pdf.

    Counts file extensions and picks the most common category. Dot-directories
    are pruned from the walk, not merely excluded from the count: .git or a
    vendored .venv holds far more files than most working trees, and walking
    one in full would both skew nothing and cost everything — the sample limit
    would bound counted work while traversal ran on.

    Falls back to "code", the type whose chunker degrades most gracefully on
    unexpected input.
    """
    from ..indexing.common.text_source import MARKDOWN_EXTENSIONS

    root = Path(repo)
    counts: Counter = Counter()
    seen = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Mutating dirnames in place is what prunes the walk.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for filename in filenames:
            if seen >= _INFERENCE_SAMPLE_LIMIT:
                return counts.most_common(1)[0][0] if counts else "code"
            if filename.startswith("."):
                continue
            seen += 1
            suffix = Path(filename).suffix.lower()
            if suffix in MARKDOWN_EXTENSIONS:
                counts["markdown"] += 1
            elif suffix in _PDF_EXTENSIONS:
                counts["pdf"] += 1
            elif suffix in _CODE_EXTENSIONS:
                counts["code"] += 1

    if not counts:
        return "code"
    return counts.most_common(1)[0][0]


def git_version() -> tuple:
    """Installed git's version as a tuple of ints, or () when unknown."""
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ()
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", result.stdout)
    if not match:
        return ()
    return tuple(int(part) for part in match.groups(default="0"))


def default_hooks(repo: Union[str, Path]) -> List[str]:
    """Hook points that together see every way the pinned branch can move.

    reference-transaction sees each ref update. post-checkout and post-rewrite
    back it up at the two moments HEAD re-attaches to the branch (checking it
    out again, finishing a rebase), which older git may not run through a
    transaction. Without reference-transaction, every post-* hook is needed;
    plumbing and resets then go unseen until the next of them fires.
    """
    if git_version() >= REFERENCE_TRANSACTION_MIN_GIT:
        return ["reference-transaction", "post-checkout", "post-rewrite"]
    return ["post-commit", "post-merge", "post-checkout", "post-rewrite"]


def default_branch(repo: Union[str, Path]) -> str:
    """The branch to index when none is named: the remote's default, else the current one.

    Raises:
        InvalidArgumentError: neither is known (no origin/HEAD, detached HEAD).
    """
    remote_head = _run_git(repo, ["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"])
    prefix = "refs/remotes/origin/"
    ref = remote_head.stdout.strip()
    if remote_head.returncode == 0 and ref.startswith(prefix):
        return ref[len(prefix) :]

    branch = current_branch(repo)
    if branch:
        return branch
    raise InvalidArgumentError(
        f"Could not tell which branch to index in {repo}: HEAD is detached and "
        "origin/HEAD is not set. Pass --branch."
    )


def list_corpus_names() -> List[str]:
    """Names of corpora that exist in Qdrant, for the interactive picker.

    Returns an empty list when Qdrant is unreachable: the caller offers to
    create a corpus, which is the right move either way.
    """
    try:
        from .utils import create_qdrant_client

        client = create_qdrant_client()
        return sorted(c.name for c in client.get_collections().collections)
    except Exception as exc:
        logger.debug("Could not list corpora: %s", exc)
        return []


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


def _render_prelude(hook: str) -> str:
    """Shell that returns early when this invocation cannot have moved a branch."""
    if hook == "reference-transaction":
        # git runs this hook once per transaction phase. Only "committed"
        # means refs moved; "prepared" may still abort.
        return """    [ "$1" = committed ] || return 0
"""
    return ""


def _sh_quote(value: str) -> str:
    """Escape for a single-quoted sh string."""
    return value.replace("'", "'\\''")


def render_block(
    corpus: str, hook: str, repo: Path, *, spawn: bool = True, branch: str = "main"
) -> str:
    """Render the guarded shell block for one corpus in one hook script.

    The script does only the cheap half inline: check the branch, and spool
    when its tip moved past the mark. Indexing is handed to a detached worker
    so the git command returns immediately, and every failure mode is
    swallowed — a hook that exits non-zero would break the user's git workflow.
    """
    start = BLOCK_START.format(corpus=corpus)
    end = BLOCK_END.format(corpus=corpus)

    # Single-quoted in the script, so a literal quote must be escaped for sh.
    corpus_sh = _sh_quote(corpus)
    repo_sh = _sh_quote(str(repo))
    branch_sh = _sh_quote(branch)
    # Data-dir relative, rebuilt from $XDG_DATA_HOME at run time exactly as
    # the Python side does. Built from [A-Za-z0-9_.-] only, so safe unquoted.
    mark_rel = spool.mark_relpath(corpus, repo, branch).as_posix()
    queue_rel = spool.queue_relpath(corpus).as_posix()
    deferred_rel = spool.deferred_relpath(corpus).as_posix()
    lock_rel = (spool.queue_relpath(corpus).parent / "worker.lock").as_posix()
    repo_id = spool.repo_id(repo)

    if spawn:
        # Two gates before spawning, both measured against a real rebase that
        # produced 1299 declined spawns for 51 useful batches:
        #
        #   1. If a worker already holds the lock, spawning is pure waste --
        #      ~1.1s of interpreter startup to discover the lock is taken, run
        #      concurrently with the worker that is actually embedding. The
        #      entry is already spooled, so the running worker picks it up on
        #      its next loop.
        #   2. Otherwise debounce: git runs each hook in its own shell, so a
        #      burst would still stampede past gate 1 if every invocation
        #      spawned at once. Sleep first, then re-check.
        #
        # The probe runs synchronously on every commit, so it must stay far
        # cheaper than the spawn it prevents. Two tiers:
        #   - flock(1) where it exists (Linux): a sub-millisecond C binary.
        #   - otherwise a bare interpreter holding only fcntl (~25ms). Stock
        #     macOS ships no flock(1), and macOS is where the waste was
        #     measured, so the gate cannot depend on it.
        # Deliberately not `arc ...`: a full arc import costs ~700ms, which
        # would reintroduce per-commit startup at the gate meant to remove it.
        #
        # The launchd/systemd watcher (`hook install --service`) is the safety
        # net for the narrow race where a worker exits between the probe and
        # its own final spool read.
        spawn_lines = """    # Spawn only for queued work. Most gate passes queue nothing, and
    # entries the worker deferred while the branch was away are picked up
    # here once it is checked out again.
    _arc_queued=''
    for _arc_entry in "$_arc_queue"/*."$_arc_repo_id".json; do
        [ -e "$_arc_entry" ] && _arc_queued=1
        break
    done
    [ -n "$_arc_queued" ] || return 0

    # Exit 0 when a worker holds the lock, 1 otherwise.
    _arc_locked() {
        [ -e "$_arc_lock" ] || return 1
        if command -v flock >/dev/null 2>&1; then
            flock -n "$_arc_lock" true >/dev/null 2>&1 && return 1
            return 0
        fi
        "$_arc_py" "$_arc_probe" "$_arc_lock" 2>/dev/null
    }

    _arc_locked && return 0

    # One body for both branches: setsid gets a fresh shell that inherits no
    # functions, so it repeats the probe inline rather than a copy of the logic.
    _arc_spawn_body="sleep $_arc_debounce; \\
if command -v flock >/dev/null 2>&1; then \\
  flock -n '$_arc_lock' true >/dev/null 2>&1 || exit 0; \\
else \\
  '$_arc_py' '$_arc_probe' '$_arc_lock' 2>/dev/null && exit 0; \\
fi; \\
'$_arc_bin' corpus sync '$_arc_corpus' --drain-spool --background >>'$_arc_log' 2>&1"

    if command -v setsid >/dev/null 2>&1; then
        setsid sh -c "$_arc_spawn_body" >/dev/null 2>&1 < /dev/null &
    else
        sh -c "$_arc_spawn_body" >/dev/null 2>&1 < /dev/null &
    fi"""
    else:
        # --no-spawn: queue only. Used by tests and by anyone who prefers to
        # drain on a schedule or via the OS watcher (`--service`).
        spawn_lines = "    : # --no-spawn: leave draining to the service or a manual run"

    prelude = _render_prelude(hook)

    return f"""{start}
# Managed by `arc corpus hook`. Edit via that command, not by hand.
# Indexes one branch. When this repo has it checked out and its tip moved past
# the last spooled commit, queues the files that changed in between, then
# drains them in the background.
# Every path exits 0: a hook must never fail the git command that ran it.
_arc_corpus='{corpus_sh}'
_arc_repo='{repo_sh}'
_arc_branch='{branch_sh}'
_arc_log="${{XDG_STATE_HOME:-$HOME/.local/state}}/arcaneum/hook.log"
_arc_data="${{XDG_DATA_HOME:-$HOME/.local/share}}/arcaneum"
# Last spooled commit of the branch, the entry queue and parked entries (named
# by repo id), and the worker lock.
_arc_mark="$_arc_data/{mark_rel}"
_arc_queue="$_arc_data/{queue_rel}"
_arc_deferred="$_arc_data/{deferred_rel}"
_arc_repo_id='{repo_id}'
_arc_lock="$_arc_data/{lock_rel}"
# Seconds to wait before starting a worker, so a burst of commits coalesces
# into one worker instead of a stampede of processes that each die on the lock.
_arc_debounce="${{ARC_HOOK_DEBOUNCE:-3}}"
# Interpreter and probe script for the lock check on platforms without
# flock(1). Written next to the spool at install time.
_arc_py="${{ARC_HOOK_PYTHON:-python3}}"
_arc_probe="$_arc_data/hook-probe-lock.py"

_arc_sync() {{
{prelude}    # git exports variables such as GIT_DIR to some hooks. Left set, they
    # would point the commands below at whichever repository fired the hook.
    unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_COMMON_DIR GIT_PREFIX \\
        GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES

    # Indexing reads the working tree, which holds the branch only while it is
    # checked out. A move made elsewhere is caught up when it is checked out.
    [ "$(git -C "$_arc_repo" symbolic-ref -q HEAD 2>/dev/null)" = "refs/heads/$_arc_branch" ] \\
        || return 0
    _arc_bin=$(command -v arc 2>/dev/null) || return 0
    mkdir -p "$(dirname "$_arc_log")" 2>/dev/null || true

    # Compared in shell first: most invocations (other refs, a second hook for
    # the same commit) find the mark current and must not pay arc's startup.
    _arc_tip=$(git -C "$_arc_repo" rev-parse -q --verify "refs/heads/$_arc_branch" 2>/dev/null) \\
        || return 0
    _arc_seen=''
    [ -r "$_arc_mark" ] && IFS= read -r _arc_seen < "$_arc_mark"
    if [ "$_arc_tip" != "$_arc_seen" ]; then
        "$_arc_bin" corpus hook spool "$_arc_corpus" --repo "$_arc_repo" \\
            --branch "$_arc_branch" >>"$_arc_log" 2>&1 || return 0
    fi

    # The worker parks entries outside the queue while the branch is away, so
    # the --service watcher goes quiet. The branch is back: requeue them.
    for _arc_entry in "$_arc_deferred"/*."$_arc_repo_id".json; do
        [ -e "$_arc_entry" ] && mv -f "$_arc_entry" "$_arc_queue/" 2>/dev/null
    done

{spawn_lines}
}}

# A subshell, so the unset above cannot reach hook content after this block.
( _arc_sync "$@" ) || true
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


# Standalone lock probe for platforms without flock(1). Kept out of the hook
# script so neither the hook nor setsid has to carry it through nested quoting.
# Exits 0 when a worker holds the lock, 1 when it is free or absent.
_PROBE_SCRIPT = """import fcntl
import os
import sys

try:
    fd = os.open(sys.argv[1], os.O_RDWR)
except OSError:
    sys.exit(1)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(0)
sys.exit(1)
"""


def probe_script_path() -> Path:
    from ..paths import get_data_dir

    return get_data_dir() / "hook-probe-lock.py"


def write_probe_script() -> Path:
    """Install the lock probe used by hooks on flock-less platforms."""
    path = probe_script_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_PROBE_SCRIPT)
    return path


def _write_block(hook_path: Path, corpus: str, block: str) -> None:
    """Put `block` into `hook_path`, replacing this corpus's block if present."""
    if hook_path.exists():
        existing = hook_path.read_text()
        pattern = _block_pattern(corpus)
        if pattern.search(existing):
            # Refresh in place so re-installing never duplicates the block. A
            # function replacement, so the block's backslashes stay literal.
            body = pattern.sub(lambda _match: block, existing, count=1)
        else:
            separator = "" if existing.endswith("\n") else "\n"
            body = f"{existing}{separator}\n{block}"
    else:
        body = f"{SHEBANG}\n\n{block}"

    hook_path.write_text(body)
    mode = hook_path.stat().st_mode
    hook_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def install(
    corpus: str,
    repo: Union[str, Path],
    hook: str = "reference-transaction",
    *,
    spawn: bool = True,
    branch: Optional[str] = None,
) -> Path:
    """Install (or refresh) the block for `corpus` in `hook`. Returns its path.

    `branch` defaults to `default_branch`. Every other block this corpus has
    in the repo is re-rendered with the same branch, so no two hook points
    ever disagree about what is indexed.

    The mark starts at the branch tip: a hook indexes what changes from here
    on, and history is a backfill's job. An existing mark is kept, so a
    reinstall still catches up whatever moved since it was written.
    """
    if hook not in SUPPORTED_HOOKS:
        raise InvalidArgumentError(
            f"Unsupported hook {hook!r}. Choose one of: {', '.join(SUPPORTED_HOOKS)}."
        )

    root = repo_root(repo)
    branch = branch or default_branch(root)
    hooks_dir = resolve_hooks_dir(root)
    hooks_dir.mkdir(parents=True, exist_ok=True)

    if spawn:
        write_probe_script()

    with spool.mark_lock(corpus, root, branch):
        seen = spool.read_mark(corpus, root, branch)
        if seen is None or not is_commit(root, seen):
            tip = branch_tip(root, branch)
            if tip:
                spool.write_mark(corpus, root, branch, tip)

    siblings = {h.hook for h in list_installed(root) if h.corpus == corpus}
    for point in sorted(siblings | {hook}):
        block = render_block(corpus, point, root, spawn=spawn, branch=branch)
        _write_block(hooks_dir / point, corpus, block)
    return hooks_dir / hook


def catch_up(corpus: str, repo: Union[str, Path], branch: str) -> Optional[GitChanges]:
    """Spool what `branch` gained since its mark, then advance the mark.

    Returns None when the repo does not have `branch` checked out: its working
    tree holds other content, so nothing is spooled and the mark stays put for
    the next event to catch up from. Otherwise returns the spooled changes,
    empty when the mark was already current.

    With no usable mark (never set, or naming a commit gc has dropped), falls
    back to the tip commit's own changes.
    """
    from .git_changes import repo_root as git_repo_root

    root = git_repo_root(repo)
    if root is None:
        raise InvalidArgumentError(f"Not a git repository: {repo}")

    with spool.mark_lock(corpus, root, branch):
        if current_branch(root) != branch:
            return None
        tip = branch_tip(root, branch)
        if tip is None:
            return None
        seen = spool.read_mark(corpus, root, branch)
        if seen == tip:
            return GitChanges()

        if seen and is_commit(root, seen):
            changes = changes_between(root, seen, tip)
        else:
            changes = changes_since(root, tip)
        # Entry before mark: a crash between them re-spools, never loses.
        spool.write_entry(
            corpus, root, changed=changes.changed, removed=changes.removed, branch=branch
        )
        spool.write_mark(corpus, root, branch, tip)
        return changes


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
            line for line in remaining.splitlines() if line.strip() and not line.startswith("#!")
        ]
        if meaningful:
            hook_path.write_text(remaining.rstrip("\n") + "\n")
        else:
            hook_path.unlink()
        changed.append(hook_path)

    # The mark only means something while a hook keeps it current. Left behind,
    # a later install would resume from it and spool everything in between.
    if not any(h.corpus == corpus for h in list_installed(root)):
        spool.clear_marks(corpus, root)

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
            corpus = match.group("corpus")
            block = _block_pattern(corpus).search(body, match.start())
            branch = _BRANCH_LINE.search(block.group(0)) if block else None
            found.append(
                InstalledHook(
                    corpus=corpus,
                    hook=name,
                    path=hook_path,
                    branch=branch.group(1).replace("'\\''", "'") if branch else None,
                )
            )
    return found


# The `_arc_branch='...'` assignment in a rendered block; `'\''` is an escaped quote.
_BRANCH_LINE = re.compile(r"^_arc_branch='((?:[^']|'\\\\'')*)'$", re.MULTILINE)

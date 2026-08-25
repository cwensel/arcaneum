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
from collections import Counter
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

# What each hook point should diff. Where git hands the hook the SHAs that
# moved, use them: guessing with ORIG_HEAD..HEAD under-reports a branch switch
# spanning several commits and misreports a rebase entirely (ORIG_HEAD there
# points at the pre-rebase tip, not at what was rewritten).
#
#   post-commit    no arguments;        HEAD is the new commit
#   post-merge     $1=squash flag;      ORIG_HEAD..HEAD is the range that moved
#   post-checkout  $1=old $2=new $3=branch-flag (0 for `git checkout -- file`)
#   post-rewrite   $1=reason;           old/new SHA pairs arrive on stdin
_HOOK_REVISIONS = {
    "post-commit": "HEAD",
    "post-merge": "ORIG_HEAD..HEAD",
}


@dataclass(frozen=True)
class InstalledHook:
    """One arcaneum block found in one hook script."""

    corpus: str
    hook: str
    path: Path


# Extensions that decide a repo's corpus type. Not exhaustive -- this only has
# to be right often enough to make a good default that the user can override.
_CODE_EXTENSIONS = frozenset(
    {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".scala",
        ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".m",
        ".sh", ".bash", ".zsh", ".sql", ".proto", ".ex", ".exs", ".clj", ".hs",
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


def has_remote(repo: Union[str, Path]) -> bool:
    """True when the repo has at least one configured remote."""
    try:
        result = _run_git(repo, ["remote"])
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def suggested_hooks(repo: Union[str, Path]) -> List[str]:
    """Hook points that will actually keep this repo current.

    post-commit alone silently misses everything that arrives by `git pull`,
    which is the failure people are least likely to notice -- the index just
    quietly lags. So a repo with a remote gets post-merge as well.
    """
    if has_remote(repo):
        return ["post-commit", "post-merge"]
    return ["post-commit"]


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


def _render_driver(hook: str) -> str:
    """Shell that turns this hook's arguments into revisions to queue.

    Each hook point learns what moved differently, so each gets its own reader
    rather than a single guess that is wrong for half of them.
    """
    if hook == "post-checkout":
        # $1=old $2=new $3=1 for a branch switch, 0 for `git checkout -- file`.
        # A file checkout changes working-tree content with no commit range to
        # diff, so there is nothing to queue; the next commit picks it up.
        return """    [ "$3" = "1" ] || return 0
    [ "$1" != "$2" ] || return 0
    _arc_spool "$1..$2"
"""

    if hook == "post-rewrite":
        # Old/new SHA pairs arrive on stdin, one per rewritten commit. Queue
        # each new SHA: a rebase rewrites several, and ORIG_HEAD points at the
        # pre-rebase tip rather than at any of them.
        return """    while read -r _arc_old _arc_new _arc_rest; do
        [ -n "$_arc_new" ] || continue
        _arc_spool "$_arc_new"
    done"""

    revision = _HOOK_REVISIONS.get(hook, "HEAD")
    if "ORIG_HEAD" in revision:
        # ORIG_HEAD is absent on a fresh clone's first merge; fall back to HEAD.
        return f"""    _arc_rev='{revision}'
    git -C "$_arc_repo" rev-parse --verify --quiet ORIG_HEAD >/dev/null 2>&1 \\
        || _arc_rev='HEAD'
    _arc_spool "$_arc_rev"
"""

    return f"""    _arc_spool '{revision}'"""


def render_block(corpus: str, hook: str, repo: Path, *, spawn: bool = True) -> str:
    """Render the guarded shell block for one corpus in one hook script.

    The script does only the cheap half inline: ask git for the touched paths
    and spool them. Indexing is handed to a detached worker so the commit
    returns immediately, and every failure mode is swallowed — a hook that
    exits non-zero would break the user's git workflow.
    """
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

    driver = _render_driver(hook)

    return f"""{start}
# Managed by `arc corpus hook`. Edit via that command, not by hand.
# Queues the paths this operation touched, then drains them in the background.
# Every path exits 0: a hook must never fail the git command that ran it.
_arc_corpus='{corpus_sh}'
_arc_repo='{repo_sh}'
_arc_log="${{XDG_STATE_HOME:-$HOME/.local/state}}/arcaneum/hook.log"

# Queue one revision or range. Called once per range the hook was told about.
_arc_spool() {{
    [ -n "$1" ] || return 0
    "$_arc_bin" corpus hook spool "$_arc_corpus" \\
        --repo "$_arc_repo" --changed-since "$1" >>"$_arc_log" 2>&1 || return 0
}}

_arc_sync() {{
    _arc_bin=$(command -v arc 2>/dev/null) || return 0
    mkdir -p "$(dirname "$_arc_log")" 2>/dev/null || true

{driver}

{spawn_lines}
}}

_arc_sync "$@" || true
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

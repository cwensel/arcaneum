"""Readable process titles for long-running arc work.

A drain worker shows up in ``ps``/``top`` as the interpreter path followed by
the script path::

    /opt/local/.../Python.app/Contents/MacOS/Python /Users/x/.../bin/arc corpus...

which is unidentifiable at a glance, and during a rebase burst there can be
several at once with no way to tell the worker holding the lock from the ones
exiting immediately on it. Rewriting the title to ``arc corpus sync <corpus>
--drain-spool`` makes the process list answer "which arc is this, and what is
it doing".

``setproctitle`` is an optional extra (``arcaneum[proctitle]``). It is a small
C extension with no runtime dependencies, but this is a cosmetic feature, so
every helper here degrades to a no-op when it is not installed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

# ps/top show the title inline; a very long one wraps and buries the lines
# around it. Real invocations sit well under this -- the cap only guards
# against a pathological argv.
MAX_TITLE_CHARS = 160

_ELLIPSIS = "..."


def _load():
    """Return the setproctitle module, or None when it is not installed."""
    try:
        import setproctitle
    except ImportError:
        return None
    return setproctitle


def title_from_argv(argv: Optional[Sequence[str]] = None) -> str:
    """Build a title from this invocation: ``arc`` plus its subcommand and flags.

    Only the leading path is replaced -- the arguments are what make one worker
    distinguishable from another, so they are kept verbatim.
    """
    parts: List[str] = list(argv if argv is not None else [])
    if not parts:
        return "arc"

    title = " ".join(["arc", *parts[1:]]).rstrip()
    if len(title) <= MAX_TITLE_CHARS:
        return title
    return title[: MAX_TITLE_CHARS - len(_ELLIPSIS)].rstrip() + _ELLIPSIS


def set_title(title: str) -> None:
    """Set this process's title, prefixing ``arc`` when absent. Never raises."""
    backend = _load()
    if backend is None:
        return

    text = title if title == "arc" or title.startswith("arc ") else f"arc {title}"
    if len(text) > MAX_TITLE_CHARS:
        text = text[: MAX_TITLE_CHARS - len(_ELLIPSIS)].rstrip() + _ELLIPSIS

    try:
        backend.setproctitle(text)
    except Exception as exc:
        # Cosmetics must never take down an indexing run.
        logger.debug("Could not set process title: %s", exc)


def set_title_from_argv(argv: Optional[Sequence[str]] = None) -> None:
    """Convenience: derive the title from argv and apply it."""
    import sys

    set_title(title_from_argv(argv if argv is not None else sys.argv))

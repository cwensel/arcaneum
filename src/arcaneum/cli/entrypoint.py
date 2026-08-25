"""Console-script entry point that names the process before doing real work.

`arcaneum.cli.main` transitively imports qdrant_client and fastembed, which
costs roughly 700ms. Setting the process title from inside `main()` meant a
short-lived `arc` -- the hook's spool call, or a drain that exits immediately
on a held lock -- finished before it was ever renamed, so it showed up in
ps/top as the interpreter and script paths.

This module is the entry point instead. It imports only `proctitle` (which
pulls `arcaneum.paths` and nothing heavy), sets the title, and only then
imports `main`. Keep it free of heavy imports: anything added here is paid
before the title lands, which defeats the point.
"""

from __future__ import annotations


def _load_main():
    """Import the real CLI. Separated so tests can assert import ordering."""
    from .main import main

    return main


def run() -> int:
    """Title the process, then hand off to the CLI."""
    from . import proctitle

    proctitle.set_title_from_argv()
    return _load_main()()

"""Relay fd-level stderr through a Rich console so it cannot corrupt a live display.

Rich's `redirect_stderr` wraps `sys.stderr`, so it only intercepts Python-level
writes. Native runtimes write straight to file descriptor 2 and bypass it
entirely: PyTorch's C++ layer emits its `PyInterpreter.cpp` teardown diagnostics
that way during PDF layout analysis. Those bytes land in the middle of a
rendered progress frame and mangle it.

Capturing fd 2 and re-emitting each line through `progress.console` keeps the
diagnostics fully intact - see `docs/pdf-layout-warning-investigation.md`, which
records that suppressing this evidence is not an acceptable resolution - while
letting Rich serialize the output against its own frames.
"""

import contextlib
import os
import threading
from typing import Iterator, Protocol


class _Console(Protocol):
    def print(self, text: str, *args, **kwargs) -> None: ...


@contextlib.contextmanager
def relay_native_stderr(console: _Console) -> Iterator[None]:
    """Route fd-level stderr writes through `console` for the duration of the block.

    Args:
        console: Typically `progress.console`, which serializes printing against
            the live display.
    """
    try:
        original_stderr = os.dup(2)
    except OSError:
        # Without a usable stderr there is nothing to relay; never let display
        # plumbing break the sync itself.
        yield
        return

    read_fd, write_fd = os.pipe()

    def drain() -> None:
        # Buffer partial reads so a line split across chunks prints once, whole.
        pending = b""
        with os.fdopen(read_fd, "rb", closefd=True) as reader:
            while True:
                chunk = reader.read(4096)
                if not chunk:
                    break
                pending += chunk
                *lines, pending = pending.split(b"\n")
                for line in lines:
                    _emit(console, line)
        if pending:
            _emit(console, pending)

    drainer = threading.Thread(target=drain, name="arcaneum-stderr-relay", daemon=True)
    drainer.start()

    os.dup2(write_fd, 2)
    os.close(write_fd)
    try:
        yield
    finally:
        # Restore first so late writers reach the real stderr, then close the
        # pipe so the drain thread observes EOF and finishes.
        os.dup2(original_stderr, 2)
        os.close(original_stderr)
        drainer.join(timeout=5.0)


def _emit(console: _Console, raw: bytes) -> None:
    text = raw.decode("utf-8", errors="replace").rstrip("\r")
    if not text.strip():
        return
    # Markup is disabled: these are third-party diagnostics that may contain
    # square brackets ("[W815 ...]") which Rich would otherwise parse as tags.
    console.print(text, markup=False, highlight=False)

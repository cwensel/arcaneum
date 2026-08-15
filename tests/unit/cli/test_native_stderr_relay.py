"""Native (fd-level) stderr must reach the console without corrupting the display."""

import os

from arcaneum.cli.native_stderr import relay_native_stderr


class _RecordingConsole:
    """Stand-in for `progress.console`, which serializes against the live display."""

    def __init__(self):
        self.lines = []

    def print(self, text, *args, **kwargs):
        self.lines.append(text)


def test_fd_level_writes_are_relayed_through_the_console():
    """PyTorch's C++ runtime writes to fd 2 directly, bypassing sys.stderr.

    Rich's redirect_stderr only wraps the Python-level stream, so such writes
    land mid-frame and mangle the progress bar. They must be captured at the fd
    layer and re-emitted through the console instead.
    """
    console = _RecordingConsole()

    with relay_native_stderr(console):
        os.write(2, b"[W815 09:29:16] Warning: Deallocating Tensor\n")

    assert any("Deallocating Tensor" in line for line in console.lines)


def test_diagnostics_are_preserved_not_suppressed():
    """The investigation doc is explicit that hiding this evidence is not a fix."""
    console = _RecordingConsole()

    with relay_native_stderr(console):
        os.write(2, b"first\nsecond\n")

    relayed = "\n".join(console.lines)
    assert "first" in relayed
    assert "second" in relayed


def test_stderr_is_restored_after_the_display_ends():
    console = _RecordingConsole()
    before = os.dup(2)
    try:
        with relay_native_stderr(console):
            pass
        after = os.dup(2)
        try:
            assert os.fstat(after).st_ino == os.fstat(before).st_ino
        finally:
            os.close(after)
    finally:
        os.close(before)


def test_restores_stderr_even_when_the_body_raises():
    console = _RecordingConsole()
    before = os.dup(2)
    try:
        try:
            with relay_native_stderr(console):
                raise RuntimeError("sync failed")
        except RuntimeError:
            pass
        after = os.dup(2)
        try:
            assert os.fstat(after).st_ino == os.fstat(before).st_ino
        finally:
            os.close(after)
    finally:
        os.close(before)


def test_adaptive_progress_relays_native_stderr_while_live():
    """The live display owns the relay, so every progress call site is covered."""
    import io

    from rich.console import Console

    from arcaneum.cli.sync import AdaptiveProgress

    buffer = io.StringIO()
    recording = Console(file=buffer, force_terminal=False, width=200)

    with AdaptiveProgress(console=recording) as progress:
        progress.add_task("Indexing...", total=10)
        os.write(2, b"[W815 09:29:16] Warning: Deallocating Tensor\n")

    assert "Deallocating Tensor" in buffer.getvalue()


def test_adaptive_progress_leaves_stderr_alone_when_disabled():
    """Under --json the bar is disabled, so there is no frame to protect."""
    import io

    from rich.console import Console

    from arcaneum.cli.sync import AdaptiveProgress

    recording = Console(file=io.StringIO(), force_terminal=False)
    before = os.dup(2)
    try:
        with AdaptiveProgress(console=recording, disable=True):
            inside = os.dup(2)
            try:
                assert os.fstat(inside).st_ino == os.fstat(before).st_ino
            finally:
                os.close(inside)
    finally:
        os.close(before)


def test_quiet_runs_add_no_console_noise():
    console = _RecordingConsole()

    with relay_native_stderr(console):
        pass

    assert console.lines == []

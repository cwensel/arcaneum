"""Persistent spawned worker for PyMuPDF4LLM layout extraction.

The parent deliberately does not import pymupdf4llm, pymupdf-layout, or Torch.
Those native runtimes are owned by a spawned child so a timeout or native crash
can be contained and fully reaped before the caller uses plain PyMuPDF fallback.
"""

from __future__ import annotations

import atexit
import contextlib
import itertools
import multiprocessing
import os
import threading
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from typing import Any, Callable


@dataclass(frozen=True)
class LayoutRequest:
    """Serializable options for one whole-document conversion."""

    pdf_path: str
    layout: bool
    ignore_images: bool
    preserve_images: bool
    use_ocr: bool


class LayoutWorkerError(RuntimeError):
    """Base class for failures reported by the layout worker."""


class LayoutWorkerTimeout(LayoutWorkerError):
    """The worker did not complete a request before its deadline."""


class LayoutWorkerCrashed(LayoutWorkerError):
    """The worker exited or disconnected while handling a request."""


class LayoutWorkerContainmentError(LayoutWorkerError):
    """A child could not be proven dead, so in-process fallback is unsafe."""


class LayoutConversionError(LayoutWorkerError):
    """PyMuPDF4LLM rejected a document without crashing its worker."""

    def __init__(self, message: str, *, font_error: bool = False):
        super().__init__(message)
        self.font_error = font_error


def _silence_native_output() -> None:
    """Keep third-party parser chatter and C++ teardown diagnostics in the child."""
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
    finally:
        os.close(null_fd)


def _worker_main(connection: Connection, silence_output: bool) -> None:
    """Own the native layout runtime and serve serializable requests."""
    if silence_output:
        _silence_native_output()

    try:
        import pymupdf
        import pymupdf4llm

        pymupdf.TOOLS.mupdf_display_errors(False)
        connection.send({"type": "ready", "pid": os.getpid()})
    except BaseException as exc:
        with contextlib.suppress(Exception):
            connection.send({"type": "startup_error", "message": repr(exc)})
        connection.close()
        return

    while True:
        try:
            message = connection.recv()
        except EOFError:
            break

        operation = message.get("operation")
        request_id = message.get("request_id")
        if operation == "shutdown":
            connection.send({"type": "stopped", "request_id": request_id})
            break
        if operation == "health":
            connection.send({"type": "healthy", "request_id": request_id, "pid": os.getpid()})
            continue
        if operation != "convert":
            connection.send(
                {
                    "type": "error",
                    "request_id": request_id,
                    "message": f"unknown operation: {operation!r}",
                    "font_error": False,
                }
            )
            continue

        request = message["request"]
        try:
            pymupdf4llm.use_layout(bool(request["layout"]))
            pages = pymupdf4llm.to_markdown(
                request["pdf_path"],
                page_chunks=True,
                ignore_images=bool(request["ignore_images"]),
                write_images=bool(request["preserve_images"]),
                force_text=True,
                table_strategy="lines_strict",
                use_ocr=bool(request["use_ocr"]),
            )
            # Materialize only ordinary Python containers across the boundary.
            serialized_pages = []
            for page_number, page in enumerate(pages, start=1):
                metadata = dict(page.get("metadata") or {})
                metadata.setdefault("page_number", page_number)
                serialized_pages.append(
                    {"text": str(page.get("text", "")), "metadata": metadata}
                )
            with pymupdf.open(request["pdf_path"]) as document:
                page_count = len(document)
            connection.send(
                {
                    "type": "result",
                    "request_id": request_id,
                    "pages": serialized_pages,
                    "page_count": page_count,
                    "worker_pid": os.getpid(),
                }
            )
        except BaseException as exc:
            error_text = str(exc)
            connection.send(
                {
                    "type": "error",
                    "request_id": request_id,
                    "message": error_text,
                    "font_error": "font" in error_text.lower() or "code=4" in error_text,
                }
            )

    connection.close()


class PDFLayoutWorker:
    """Single-owner client for one persistent spawned layout process."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 300.0,
        startup_timeout_seconds: float = 60.0,
        silence_output: bool = True,
        process_target: Callable[[Connection, bool], None] = _worker_main,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.startup_timeout_seconds = startup_timeout_seconds
        self.silence_output = silence_output
        self._process_target = process_target
        self._context = multiprocessing.get_context("spawn")
        self._process: multiprocessing.Process | None = None
        self._connection: Connection | None = None
        self._lock = threading.RLock()
        self._request_ids = itertools.count(1)
        self._generation = 0
        self._completed_requests = 0

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None and process.is_alive() else None

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def completed_requests(self) -> int:
        return self._completed_requests

    def _start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._reap(force=True)
        parent_connection, child_connection = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=self._process_target,
            args=(child_connection, self.silence_output),
            name="arcaneum-pdf-layout",
            daemon=True,
        )
        try:
            process.start()
        except BaseException:
            parent_connection.close()
            child_connection.close()
            with contextlib.suppress(ValueError):
                process.close()
            raise
        child_connection.close()
        self._process = process
        self._connection = parent_connection
        if not parent_connection.poll(self.startup_timeout_seconds):
            self._reap(force=True)
            raise LayoutWorkerTimeout("PDF layout worker startup timed out")
        try:
            response = parent_connection.recv()
        except EOFError as exc:
            exit_code = process.exitcode
            self._reap(force=True)
            raise LayoutWorkerCrashed(
                f"PDF layout worker exited during startup (exit code {exit_code})"
            ) from exc
        if response.get("type") != "ready":
            message = response.get("message", repr(response))
            self._reap(force=True)
            raise LayoutWorkerCrashed(f"PDF layout worker failed to start: {message}")
        self._generation += 1

    def _exchange(self, operation: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        self._start()
        assert self._connection is not None
        assert self._process is not None
        request_id = next(self._request_ids)
        message: dict[str, Any] = {"operation": operation, "request_id": request_id}
        if payload:
            message.update(payload)
        try:
            self._connection.send(message)
        except (BrokenPipeError, EOFError, OSError) as exc:
            exit_code = self._process.exitcode
            self._reap(force=True)
            raise LayoutWorkerCrashed(
                f"PDF layout worker disconnected (exit code {exit_code})"
            ) from exc

        if not self._connection.poll(self.timeout_seconds):
            self._reap(force=True)
            raise LayoutWorkerTimeout(
                f"PDF layout worker exceeded {self.timeout_seconds:g}s timeout"
            )
        try:
            response = self._connection.recv()
        except EOFError as exc:
            exit_code = self._process.exitcode
            self._reap(force=True)
            raise LayoutWorkerCrashed(
                f"PDF layout worker exited (exit code {exit_code})"
            ) from exc
        if response.get("request_id") != request_id:
            self._reap(force=True)
            raise LayoutWorkerCrashed("PDF layout worker returned a mismatched response")
        return response

    def convert(self, request: LayoutRequest) -> dict[str, Any]:
        """Convert one document, reusing the same child across successful calls."""
        with self._lock:
            response = self._exchange("convert", payload={"request": asdict(request)})
            if response.get("type") == "error":
                raise LayoutConversionError(
                    response.get("message", "PDF layout conversion failed"),
                    font_error=bool(response.get("font_error")),
                )
            if response.get("type") != "result":
                self._reap(force=True)
                raise LayoutWorkerCrashed(f"unexpected PDF layout response: {response!r}")
            self._completed_requests += 1
            return response

    def health(self) -> dict[str, Any]:
        """Verify that the worker event loop is responsive."""
        with self._lock:
            response = self._exchange("health")
            if response.get("type") != "healthy":
                self._reap(force=True)
                raise LayoutWorkerCrashed(f"unexpected PDF layout health response: {response!r}")
            return response

    def _reap(self, *, force: bool) -> None:
        process = self._process
        connection = self._connection
        self._connection = None
        if connection is not None:
            connection.close()
        if process is None:
            return
        if force and process.is_alive():
            process.terminate()
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)
        if process.is_alive():
            # Retain the process handle so callers can retry containment. Most
            # importantly, do not let the extractor mistake this for a completed
            # reap and begin local PyMuPDF fallback concurrently with the child.
            self._process = process
            raise LayoutWorkerContainmentError(
                f"PDF layout worker pid {process.pid} remained alive after terminate and kill"
            )
        process.close()
        self._process = None

    def close(self) -> None:
        """Request a clean shutdown, then guarantee that the child is reaped."""
        with self._lock:
            if self._process is None:
                return
            try:
                if self._process.is_alive() and self._connection is not None:
                    request_id = next(self._request_ids)
                    self._connection.send({"operation": "shutdown", "request_id": request_id})
                    if self._connection.poll(min(self.timeout_seconds, 5.0)):
                        self._connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                # Reaping below is authoritative; a shutdown acknowledgement is
                # optional once the connection has failed.
                pass
            self._reap(force=False)

    def __enter__(self) -> PDFLayoutWorker:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


_SHARED_WORKER: PDFLayoutWorker | None = None
_SHARED_WORKER_LOCK = threading.Lock()


def shared_layout_worker() -> PDFLayoutWorker:
    """Return the process-wide client; its child remains persistent across PDFs."""
    global _SHARED_WORKER
    with _SHARED_WORKER_LOCK:
        if _SHARED_WORKER is None:
            _SHARED_WORKER = PDFLayoutWorker()
        return _SHARED_WORKER


def shutdown_shared_layout_worker() -> None:
    global _SHARED_WORKER
    with _SHARED_WORKER_LOCK:
        worker, _SHARED_WORKER = _SHARED_WORKER, None
    if worker is not None:
        worker.close()


atexit.register(shutdown_shared_layout_worker)

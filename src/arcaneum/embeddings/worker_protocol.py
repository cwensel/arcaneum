"""Spawned accelerator worker protocol and lifecycle primitives.

This module deliberately does not import an inference runtime.  Accelerator imports
and model construction belong to the child-side backend factory so the parent never
initializes Torch, MPS, CUDA, or another native runtime before spawning the worker.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Protocol

import numpy as np

PROTOCOL_VERSION = 1


class MessageType(StrEnum):
    INITIALIZE = "initialize"
    INITIALIZED = "initialized"
    ENCODE = "encode"
    ENCODED = "encoded"
    HEARTBEAT = "heartbeat"
    HEALTH = "health"
    SHUTDOWN = "shutdown"
    SHUTDOWN_COMPLETE = "shutdown_complete"
    ERROR = "error"


class WorkerProtocolError(RuntimeError):
    """The worker sent a malformed or unexpected reply."""


class WorkerTimeoutError(TimeoutError):
    """A worker request exceeded its deadline and the worker was reaped."""


class WorkerCrashedError(RuntimeError):
    """The worker exited before replying."""


class WorkerBackend(Protocol):
    def encode(self, texts: list[str], **options: Any) -> np.ndarray: ...

    def health(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class WorkerConfig:
    """Serializable child initialization configuration."""

    backend_factory: str
    backend_config: dict[str, Any]
    queue_size: int = 1

    def __post_init__(self) -> None:
        if self.queue_size < 1:
            raise ValueError("queue_size must be at least one")


def make_message(kind: MessageType, request_id: str, **payload: Any) -> dict[str, Any]:
    """Create a pickle/JSON-compatible protocol envelope."""
    return {
        "version": PROTOCOL_VERSION,
        "type": kind.value,
        "request_id": request_id,
        "payload": payload,
    }


def _validate_message(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkerProtocolError("worker reply is not a mapping")
    required = {"version", "type", "request_id", "payload"}
    if set(value) != required:
        raise WorkerProtocolError("worker reply has an invalid envelope")
    if value["version"] != PROTOCOL_VERSION:
        raise WorkerProtocolError(f"unsupported worker protocol version: {value['version']!r}")
    try:
        MessageType(value["type"])
    except (TypeError, ValueError) as exc:
        raise WorkerProtocolError(f"unknown worker message type: {value['type']!r}") from exc
    if not isinstance(value["request_id"], str) or not value["request_id"]:
        raise WorkerProtocolError("worker reply has no request identifier")
    if not isinstance(value["payload"], dict):
        raise WorkerProtocolError("worker reply payload is not a mapping")
    return value


def _load_factory(path: str):
    module_name, separator, attribute = path.rpartition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("backend_factory must have 'module:attribute' form")
    return getattr(importlib.import_module(module_name), attribute)


def _error(request_id: str, code: str, exc: BaseException) -> dict[str, Any]:
    return make_message(
        MessageType.ERROR,
        request_id,
        code=code,
        message=str(exc),
        exception_type=type(exc).__name__,
    )


def _worker_main(
    commands: mp.Queue[dict[str, Any]], replies: mp.Queue[dict[str, Any]], config: WorkerConfig
) -> None:
    """Child entry point. Keep all backend/native runtime state in this function."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    backend: WorkerBackend | None = None
    try:
        init = _validate_message(commands.get())
        if init["type"] != MessageType.INITIALIZE:
            raise WorkerProtocolError("first worker command must initialize the backend")
        try:
            backend = _load_factory(config.backend_factory)(config.backend_config)
        except BaseException as exc:
            replies.put(_error(init["request_id"], "initialization_failed", exc))
            return
        replies.put(make_message(MessageType.INITIALIZED, init["request_id"], pid=os.getpid()))

        while True:
            command = _validate_message(commands.get())
            request_id = command["request_id"]
            kind = MessageType(command["type"])
            try:
                if kind is MessageType.ENCODE:
                    texts = command["payload"].get("texts")
                    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                        raise ValueError("encode texts must be a list of strings")
                    options = command["payload"].get("options", {})
                    if not isinstance(options, dict):
                        raise ValueError("encode options must be a mapping")
                    result = np.array(
                        backend.encode(texts, **options), dtype=np.float32, copy=True, order="C"
                    )
                    # A plain nested list is independently owned CPU data on the wire.
                    replies.put(
                        make_message(
                            MessageType.ENCODED,
                            request_id,
                            embeddings=result.tolist(),
                            shape=list(result.shape),
                            dtype=str(result.dtype),
                        )
                    )
                elif kind is MessageType.HEARTBEAT:
                    replies.put(
                        make_message(
                            MessageType.HEALTH,
                            request_id,
                            pid=os.getpid(),
                            state="ready",
                            backend=backend.health(),
                        )
                    )
                elif kind is MessageType.SHUTDOWN:
                    backend.close()
                    backend = None
                    replies.put(make_message(MessageType.SHUTDOWN_COMPLETE, request_id))
                    return
                else:
                    raise WorkerProtocolError(f"unexpected command: {kind.value}")
            except BaseException as exc:
                replies.put(_error(request_id, "backend_error", exc))
    except (EOFError, KeyboardInterrupt):
        pass
    except BaseException as exc:
        try:
            replies.put(_error("worker", "protocol_error", exc))
        except BaseException:
            pass
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException:
                pass


class AcceleratorWorkerSession:
    """Own one persistent, bounded, spawn-created accelerator process.

    Calls are intentionally synchronous: exactly one request can be in flight per
    session. On timeout, crash, malformed output, or interruption the child is
    terminated and joined before the exception escapes.
    """

    def __init__(self, config: WorkerConfig, *, startup_timeout: float = 10.0) -> None:
        self.config = config
        self.startup_timeout = startup_timeout
        self._context = mp.get_context("spawn")
        self._commands: mp.Queue[dict[str, Any]] | None = None
        self._replies: mp.Queue[dict[str, Any]] | None = None
        self._process: BaseProcess | None = None
        self._in_flight = False
        self._request_lock = threading.RLock()

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self) -> "AcceleratorWorkerSession":
        if self._process is not None:
            raise RuntimeError("worker session has already been started")
        self._commands = self._context.Queue(maxsize=self.config.queue_size)
        self._replies = self._context.Queue(maxsize=self.config.queue_size)
        self._process = self._context.Process(
            target=_worker_main,
            args=(self._commands, self._replies, self.config),
            name="arcaneum-accelerator",
        )
        self._process.start()
        request_id = self._send(MessageType.INITIALIZE)
        reply = self._receive(request_id, MessageType.INITIALIZED, self.startup_timeout)
        if reply["type"] == MessageType.ERROR:
            self._reap()
            raise WorkerCrashedError(reply["payload"].get("message", "worker init failed"))
        return self

    def encode(self, texts: list[str], *, timeout: float, **options: Any) -> np.ndarray:
        with self._request_lock:
            request_id = self._send(MessageType.ENCODE, texts=texts, options=options)
            reply = self._receive(request_id, MessageType.ENCODED, timeout)
        payload = reply["payload"]
        try:
            shape = tuple(payload["shape"])
            result = np.array(payload["embeddings"], dtype=np.dtype(payload["dtype"]), copy=True)
        except (KeyError, TypeError, ValueError) as exc:
            self._reap()
            raise WorkerProtocolError("encoded reply has invalid array data") from exc
        if result.shape != shape or result.dtype != np.float32:
            self._reap()
            raise WorkerProtocolError("encoded reply metadata does not match array data")
        return np.array(result, copy=True, order="C")

    def health(self, *, timeout: float = 1.0) -> dict[str, Any]:
        with self._request_lock:
            request_id = self._send(MessageType.HEARTBEAT)
            return self._receive(request_id, MessageType.HEALTH, timeout)["payload"]

    def shutdown(self, *, timeout: float = 5.0) -> None:
        with self._request_lock:
            if self._process is None:
                return
            if self._process.is_alive():
                try:
                    request_id = self._send(MessageType.SHUTDOWN)
                    self._receive(request_id, MessageType.SHUTDOWN_COMPLETE, timeout)
                except (WorkerCrashedError, WorkerProtocolError, WorkerTimeoutError):
                    pass
            self._reap(graceful=True)

    def _send(self, kind: MessageType, **payload: Any) -> str:
        if self._commands is None or self._process is None or not self._process.is_alive():
            self._reap()
            raise WorkerCrashedError("accelerator worker is not running")
        if self._in_flight:
            raise RuntimeError("only one request may be in flight per worker")
        request_id = uuid.uuid4().hex
        self._commands.put_nowait(make_message(kind, request_id, **payload))
        self._in_flight = True
        return request_id

    def _receive(self, request_id: str, expected: MessageType, timeout: float) -> dict[str, Any]:
        assert self._replies is not None
        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                try:
                    raw = self._replies.get(timeout=min(remaining, 0.05))
                    break
                except queue.Empty:
                    if self._process is None or not self._process.is_alive():
                        raise WorkerCrashedError("accelerator worker exited without a reply")
            reply = _validate_message(raw)
            if reply["request_id"] != request_id:
                raise WorkerProtocolError("worker reply request identifier does not match")
            if reply["type"] == MessageType.ERROR:
                raise WorkerCrashedError(reply["payload"].get("message", "worker error"))
            if reply["type"] != expected:
                raise WorkerProtocolError(
                    f"expected {expected.value!r}, received {reply['type']!r}"
                )
            return reply
        except queue.Empty as exc:
            self._reap()
            raise WorkerTimeoutError(f"worker request exceeded {timeout:.3f}s") from exc
        except (KeyboardInterrupt, WorkerCrashedError, WorkerProtocolError):
            self._reap()
            raise
        finally:
            self._in_flight = False

    def _reap(self, *, graceful: bool = False) -> None:
        process = self._process
        if process is not None:
            if process.is_alive() and not graceful:
                process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
            process.close()
        for channel in (self._commands, self._replies):
            if channel is not None:
                channel.close()
                channel.join_thread()
        self._process = None
        self._commands = None
        self._replies = None

    def __enter__(self) -> "AcceleratorWorkerSession":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.shutdown()


class DeterministicFakeBackend:
    """No-runtime backend for protocol tests and downstream scheduler tests."""

    def __init__(self, config: dict[str, Any]) -> None:
        if config.get("fail_init"):
            raise RuntimeError("requested fake initialization failure")
        self.config = config
        self.dimension = int(config.get("dimension", 3))
        self.delay = float(config.get("delay", 0.0))
        self.crash = bool(config.get("crash", False))
        self._encodes = 0

    def encode(self, texts: list[str], **options: Any) -> np.ndarray:
        if self.crash:
            os._exit(86)
        if self.delay:
            time.sleep(self.delay)
        if marker := self.config.get("completion_marker"):
            Path(marker).write_text("completed")
        self._encodes += 1
        return np.array(
            [
                [float(len(text)), float(sum(map(ord, text)) % 997), float(i)]
                for i, text in enumerate(texts)
            ],
            dtype=np.float32,
        )[:, : self.dimension]

    def health(self) -> dict[str, Any]:
        return {"model_loads": 1, "encodes": self._encodes}

    def close(self) -> None:
        return None


def create_deterministic_fake_backend(config: dict[str, Any]) -> DeterministicFakeBackend:
    return DeterministicFakeBackend(config)

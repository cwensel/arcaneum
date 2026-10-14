"""Shared fakes for remote embedding server and pool tests (RDR-025)."""

from __future__ import annotations

import threading
import time
from multiprocessing.connection import Client

import numpy as np

from arcaneum.embeddings.client import EMBEDDING_MODELS
from arcaneum.embeddings.remote.identity import local_identity
from arcaneum.embeddings.worker_protocol import (
    ConnectionChannel,
    DeterministicFakeBackend,
    MessageType,
    make_message,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeHost:
    """Model host with gates so tests decide when loads and encodes finish."""

    def __init__(self, *, per_text: bool = False, dimension: int = 3) -> None:
        # per_text vectors depend only on each text, not on its position in a batch,
        # so results are independent of how a pool splits sub-batches.
        self.per_text = per_text
        self.dimension = dimension
        self.backend = DeterministicFakeBackend({"dimension": 3})
        self.loads: list[str] = []
        self.unloads: list[str] = []
        self.encodes: list[tuple[str, list[str]]] = []
        self.load_gate: threading.Event | None = None
        self.encode_permits: dict[str, threading.Semaphore] = {}
        self.encode_started = threading.Event()
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def load(self, model: str) -> None:
        self.loads.append(model)
        if self.load_gate is not None:
            assert self.load_gate.wait(5)

    def encode(self, model: str, texts: list[str]) -> np.ndarray:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.encode_started.set()
            permits = self.encode_permits.get(model)
            if permits is not None:
                assert permits.acquire(timeout=5)
            self.encodes.append((model, list(texts)))
            if self.per_text:
                dimension = EMBEDDING_MODELS[model]["dimensions"]
                return np.array([text_vector(text, dimension) for text in texts], np.float32)
            return self.backend.encode(texts)
        finally:
            with self._lock:
                self.active -= 1

    def unload(self, model: str) -> None:
        self.unloads.append(model)


class RawClient:
    """Minimal protocol client that buffers replies by request ID."""

    def __init__(self, address, model=None, *, authkey=None, identity=None) -> None:
        self.channel = ConnectionChannel(Client(address, authkey=authkey))
        self._replies: dict[str, dict] = {}
        self._counter = 0
        payload = local_identity(model) if identity is None else identity
        self.initialized = self.wait(self.send(MessageType.INITIALIZE, **payload))

    def send(self, kind: MessageType, **payload) -> str:
        self._counter += 1
        request_id = f"r{self._counter}"
        self.channel.send(make_message(kind, request_id, **payload))
        return request_id

    def wait(self, request_id: str, timeout: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout
        while request_id not in self._replies:
            reply = self.channel.receive(timeout=max(0.0, deadline - time.monotonic()))
            if reply is None:
                raise TimeoutError(request_id)
            self._replies[reply["request_id"]] = reply
        return self._replies.pop(request_id)

    def heartbeat(self) -> dict:
        reply = self.wait(self.send(MessageType.HEARTBEAT))
        assert reply["type"] == MessageType.HEALTH
        return reply["payload"]

    def encode(self, texts: list[str]) -> str:
        return self.send(MessageType.ENCODE, texts=texts)

    def close(self) -> None:
        self.channel.close()


def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.005)


def text_vector(text: str, dimension: int = 3) -> list[float]:
    """Deterministic, non-degenerate vector for one text."""
    seed = sum(map(ord, text))
    return [1.0 + len(text)] + [
        float((seed * (i + 7)) % 997) / 997.0 + 0.01 for i in range(dimension - 1)
    ]

"""Embedding server behind `arc models serve` (RDR-025).

Each connection is read on its own thread, so heartbeats are answered at any time.
Model loads and encodes run on one executor thread, so encodes from several
clients never overlap. At most one model is resident: requests for another model
wait (``queued``) until the resident model yields, which it does after its current
encode once another model is waiting and either it has been idle for the grace
period or it has been resident for the turn length.
"""

from __future__ import annotations

import gc
import ipaddress
import logging
import os
import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.connection import Listener, answer_challenge, deliver_challenge
from typing import Any, Optional, Protocol

import numpy as np

from arcaneum.embeddings.client import EMBEDDING_MODELS, EmbeddingClient, model_revision
from arcaneum.embeddings.remote import DEFAULT_PORT
from arcaneum.embeddings.remote.identity import (
    dependency_versions,
    identity_mismatches,
    local_identity,
)
from arcaneum.embeddings.worker_protocol import (
    ConnectionChannel,
    MessageType,
    WorkerProtocolError,
    encode_request,
    encoded_message,
    make_message,
)

logger = logging.getLogger(__name__)

TAILSCALE_NETWORK = ipaddress.ip_network("100.64.0.0/10")
HANDSHAKE_TIMEOUT = 30.0


class ModelHost(Protocol):
    """Loads, runs and unloads embedding models for the server's executor."""

    def load(self, model: str) -> None: ...

    def encode(self, model: str, texts: list[str]) -> np.ndarray: ...

    def unload(self, model: str) -> None: ...


class EmbeddingClientHost:
    """Model host backed by arcaneum's own CPU embedding path.

    Running the workstation's ``EmbeddingClient`` keeps preprocessing, truncation
    and prompt policy identical to local embedding.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        # The server always embeds locally, whatever router its process has.
        self.client = EmbeddingClient(cache_dir=cache_dir, use_gpu=False, remote_routing=False)

    def load(self, model: str) -> None:
        self.client.get_model(model)

    def encode(self, model: str, texts: list[str]) -> np.ndarray:
        return np.asarray(self.client.embed(texts, model, prompt_type="document"))

    def unload(self, model: str) -> None:
        self.client.unload_model(model)
        gc.collect()


def default_encode_batch_size(model: str) -> int:
    """arcaneum's CPU batching: 8 chunks for SentenceTransformers, up to 100 for fastembed."""
    if EMBEDDING_MODELS[model].get("backend") == "sentence-transformers":
        return EmbeddingClient._CPU_FALLBACK_INNER_BATCH
    return 100


@dataclass
class ServerSettings:
    port: int = DEFAULT_PORT
    bind: str = "all"
    authkey: Optional[bytes] = None
    # None serves every remote-eligible (revision-pinned) model.
    models: Optional[tuple[str, ...]] = None
    batch_sizes: dict[str, int] = field(default_factory=dict)
    idle_grace: float = 60.0
    turn_length: float = 600.0
    cache_dir: Optional[str] = None

    def serves(self, model: Optional[str]) -> bool:
        if model not in EMBEDDING_MODELS or model_revision(model) is None:
            return False
        return self.models is None or model in self.models

    def encode_batch_size(self, model: str) -> int:
        return int(self.batch_sizes.get(model) or default_encode_batch_size(model))


# --- Listening address -----------------------------------------------------------


def find_tailscale_address(interfaces: dict[str, list[Any]]) -> Optional[str]:
    """The host's Tailscale IPv4 address: on ``tailscale0``, else on any interface."""
    if "tailscale0" in interfaces:
        candidates = interfaces["tailscale0"]
    else:
        candidates = [address for addresses in interfaces.values() for address in addresses]
    for address in candidates:
        if address.family != socket.AF_INET:
            continue
        try:
            if ipaddress.ip_address(address.address) in TAILSCALE_NETWORK:
                return address.address
        except ValueError:
            continue
    return None


def _has_address(interfaces: dict[str, list[Any]], wanted: str) -> bool:
    return any(
        address.address == wanted for addresses in interfaces.values() for address in addresses
    )


def _net_if_addrs() -> dict[str, list[Any]]:
    import psutil

    return psutil.net_if_addrs()


def resolve_bind_address(
    bind: str,
    *,
    interfaces: Callable[[], dict[str, list[Any]]] = _net_if_addrs,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    poll_interval: float = 2.0,
    log_interval: float = 60.0,
) -> str:
    """Return the address to listen on, waiting until it exists on this host.

    ``all`` listens on every interface. ``tailscale`` and a literal address wait,
    polling every two seconds and logging once a minute, because at boot the
    address may not be configured yet (``net.ipv4.ip_nonlocal_bind`` is 0).
    """
    if bind == "all":
        return "0.0.0.0"
    if bind == "tailscale":
        describe = "the Tailscale address"

        def lookup() -> Optional[str]:
            return find_tailscale_address(interfaces())

    else:
        try:
            literal = str(ipaddress.ip_address(bind))
        except ValueError as exc:
            raise ValueError(
                f"bind must be 'all', 'tailscale' or an IP address, not {bind!r}"
            ) from exc
        describe = f"address {literal}"

        def lookup() -> Optional[str]:
            return literal if _has_address(interfaces(), literal) else None

    started = clock()
    next_log = started
    waited = False
    while True:
        address = lookup()
        if address is not None:
            if waited:
                logger.info(
                    "Binding %s after waiting %.0f s for it to appear", address, clock() - started
                )
            else:
                logger.info("Binding %s", address)
            return address
        if clock() >= next_log:
            logger.info("Waiting for %s to appear on this host before binding", describe)
            next_log += log_interval
        sleep(poll_interval)
        waited = True


# --- Scheduling --------------------------------------------------------------------


@dataclass(eq=False)
class _Request:
    connection: "_ServerConnection"
    request_id: str
    model: str
    texts: list[str]
    sequence: int = 0


class _Scheduler:
    """Single executor: per-model FIFO queues that take turns on one resident model."""

    def __init__(
        self,
        host: ModelHost,
        settings: ServerSettings,
        clock: Callable[[], float],
    ) -> None:
        self._host = host
        self._settings = settings
        self._clock = clock
        self._cond = threading.Condition()
        self._queues: dict[str, deque[_Request]] = {}
        self._resident: Optional[str] = None
        self._loading: Optional[str] = None
        self._resident_since = 0.0
        self._last_work = 0.0
        self._running: Optional[_Request] = None
        self._running_state = "idle"
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="arc-embed-executor", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join(timeout=5)

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def submit(self, request: _Request) -> None:
        with self._cond:
            self._queues.setdefault(request.model, deque()).append(request)
            self._cond.notify_all()

    def drop(self, connection: "_ServerConnection") -> None:
        """Forget a closed connection's queued requests; a running one finishes unseen."""
        with self._cond:
            for queue in self._queues.values():
                for request in [r for r in queue if r.connection is connection]:
                    queue.remove(request)
            self._cond.notify_all()

    def queued_count(self) -> int:
        with self._cond:
            return sum(len(queue) for queue in self._queues.values())

    def busy(self) -> bool:
        with self._cond:
            return self._running is not None or any(self._queues.values())

    def resident(self) -> Optional[str]:
        with self._cond:
            return self._resident

    def loading(self) -> Optional[str]:
        with self._cond:
            return self._loading

    def state_of(self, connection: "_ServerConnection") -> str:
        with self._cond:
            if self._running is not None and self._running.connection is connection:
                return self._running_state
            for model, queue in self._queues.items():
                if any(request.connection is connection for request in queue):
                    return "loading" if model == self._loading else "queued"
            return "idle"

    def _next(self, now: float) -> tuple[Optional[_Request], Optional[float]]:
        """Pick the next request, or how long to wait before deciding again."""
        resident_queue = self._queues.get(self._resident) if self._resident else None
        others = [
            model for model, queue in self._queues.items() if queue and model != self._resident
        ]
        turn_over = now - self._resident_since >= self._settings.turn_length
        if resident_queue and (not others or not turn_over):
            return resident_queue.popleft(), None
        if not others:
            return None, None
        idle_for = now - self._last_work
        if self._resident is not None and not resident_queue and not turn_over:
            if idle_for < self._settings.idle_grace:
                return None, self._settings.idle_grace - idle_for
        # Hand the endpoint to the model whose oldest request has waited longest.
        model = min(others, key=lambda m: self._queues[m][0].sequence)
        return self._queues[model].popleft(), None

    def _run(self) -> None:
        while True:
            with self._cond:
                while True:
                    if self._stopped:
                        return
                    request, wait = self._next(self._clock())
                    if request is not None:
                        break
                    # Decisions use the injected clock; re-check at least once a second.
                    self._cond.wait(timeout=min(wait, 1.0) if wait is not None else 1.0)
                self._running = request
                switching = request.model != self._resident
                if switching:
                    previous, self._resident = self._resident, None
                    self._loading = request.model
                    self._running_state = "loading"
                else:
                    previous = None
                    self._running_state = "encoding"
            if switching:
                if not self._switch(previous, request):
                    continue
            self._encode(request)

    def _switch(self, previous: Optional[str], request: _Request) -> bool:
        if previous is not None:
            logger.info("Unloading %s to serve %s", previous, request.model)
            try:
                self._host.unload(previous)
            except Exception:
                logger.exception("Unloading %s failed", previous)
        logger.info("Loading %s", request.model)
        try:
            self._host.load(request.model)
        except Exception as exc:
            logger.exception("Loading %s failed", request.model)
            with self._cond:
                failed = [request, *self._queues.pop(request.model, deque())]
                self._loading = None
                self._running = None
                self._running_state = "idle"
                self._last_work = self._clock()
                self._cond.notify_all()
            for each in failed:
                each.connection.reply_error(each, "load_failed", exc)
            return False
        with self._cond:
            self._resident = request.model
            self._loading = None
            self._resident_since = self._clock()
            self._running_state = "encoding"
        return True

    def _encode(self, request: _Request) -> None:
        started = time.monotonic()
        try:
            result = self._host.encode(request.model, request.texts)
            reply = encoded_message(
                request.request_id, result, encode_seconds=time.monotonic() - started
            )
        except Exception as exc:
            logger.exception("Encoding %d texts for %s failed", len(request.texts), request.model)
            reply = None
            error = exc
        with self._cond:
            self._running = None
            self._running_state = "idle"
            self._last_work = self._clock()
            self._cond.notify_all()
        if reply is not None:
            request.connection.reply(request, reply)
        else:
            request.connection.reply_error(request, "backend_error", error)


# --- Connections -------------------------------------------------------------------


class _ServerConnection:
    def __init__(self, server: "EmbeddingServer", channel: ConnectionChannel, peer: str) -> None:
        self._server = server
        self.channel = channel
        self.peer = peer
        self.model: Optional[str] = None
        self.serves = False
        self.compatible = False
        self._pending: Optional[_Request] = None
        self._lock = threading.Lock()

    def run(self) -> None:
        try:
            init = self.channel.receive(timeout=HANDSHAKE_TIMEOUT)
            if init is None or init["type"] != MessageType.INITIALIZE:
                logger.warning("Connection from %s did not initialize; closing", self.peer)
                return
            self._initialize(init)
            while True:
                message = self.channel.receive()
                if message is None:
                    continue
                if not self._dispatch(message):
                    return
        except (EOFError, OSError):
            pass
        except WorkerProtocolError as exc:
            logger.warning("Closing connection from %s: %s", self.peer, exc)
        finally:
            self._server._scheduler.drop(self)
            self.channel.close()
            self._server._forget(self)

    def _initialize(self, init: dict[str, Any]) -> None:
        payload = init["payload"]
        model = payload.get("model")
        status_only = model is None
        known = isinstance(model, str) and model in EMBEDDING_MODELS
        identity = self._server.identity(model if known else None)
        if status_only or not known:
            self.compatible = False
        else:
            self.compatible = not identity_mismatches(identity, payload)
        self.model = model if known else None
        self.serves = known and self._server.settings.serves(model)
        reply: dict[str, Any] = {
            **identity,
            "serves": self.serves,
            "pid": os.getpid(),
            "dependencies": dependency_versions(),
        }
        if known:
            reply["encode_batch_size"] = self._server.settings.encode_batch_size(model)
        if not status_only:
            logger.info(
                "Connection from %s for %s (serves=%s, compatible=%s)",
                self.peer,
                model,
                self.serves,
                self.compatible,
            )
        self.channel.send(make_message(MessageType.INITIALIZED, init["request_id"], **reply))

    def _dispatch(self, message: dict[str, Any]) -> bool:
        kind = MessageType(message["type"])
        request_id = message["request_id"]
        if kind is MessageType.HEARTBEAT:
            self.channel.send(make_message(MessageType.HEALTH, request_id, **self._health()))
        elif kind is MessageType.ENCODE:
            self._submit(request_id, message["payload"])
        elif kind is MessageType.SHUTDOWN:
            self.channel.send(make_message(MessageType.SHUTDOWN_COMPLETE, request_id))
            return False
        else:
            self._send_error(request_id, "protocol_error", f"unexpected command: {kind.value}")
        return True

    def _submit(self, request_id: str, payload: dict[str, Any]) -> None:
        if self.model is None or not self.serves:
            self._send_error(request_id, "not_serving", "this connection cannot encode")
            return
        if not self.compatible:
            self._send_error(
                request_id, "incompatible", "handshake mismatch; this connection cannot encode"
            )
            return
        try:
            texts, _ = encode_request(payload)
        except ValueError as exc:
            self._send_error(request_id, "backend_error", str(exc), type(exc).__name__)
            return
        with self._lock:
            if self._pending is not None:
                self._send_error(request_id, "busy", "one request may be in flight per connection")
                return
            request = _Request(
                self, request_id, self.model, texts, sequence=self._server._next_sequence()
            )
            self._pending = request
        self._server._scheduler.submit(request)

    def _health(self) -> dict[str, Any]:
        identity = self._server.identity(None)
        scheduler = self._server._scheduler
        return {
            "pid": os.getpid(),
            "arcaneum_version": identity["arcaneum_version"],
            "code_fingerprint": identity["code_fingerprint"],
            "resident_model": scheduler.resident(),
            "loading_model": scheduler.loading(),
            "busy": scheduler.busy(),
            "state": scheduler.state_of(self),
            "dependencies": dependency_versions(),
        }

    def _finish(self, request: _Request) -> bool:
        with self._lock:
            if self._pending is not request:
                return False
            self._pending = None
            return True

    def reply(self, request: _Request, message: dict[str, Any]) -> None:
        if self._finish(request):
            self._safe_send(message)

    def reply_error(self, request: _Request, code: str, exc: BaseException) -> None:
        if self._finish(request):
            self._send_error(request.request_id, code, str(exc), type(exc).__name__)

    def _send_error(
        self, request_id: str, code: str, message: str, exception_type: str = "RemoteEmbeddingError"
    ) -> None:
        self._safe_send(
            make_message(
                MessageType.ERROR,
                request_id,
                code=code,
                message=message,
                exception_type=exception_type,
            )
        )

    def _safe_send(self, message: dict[str, Any]) -> None:
        try:
            self.channel.send(message)
        except (OSError, ValueError, EOFError):
            logger.debug("Dropping reply to closed connection from %s", self.peer)


# --- Server ------------------------------------------------------------------------


class EmbeddingServer:
    def __init__(
        self,
        settings: ServerSettings,
        *,
        host: Optional[ModelHost] = None,
        clock: Callable[[], float] = time.monotonic,
        resolve_address: Callable[[str], str] = resolve_bind_address,
        identity: Callable[[Optional[str]], dict[str, Any]] = local_identity,
    ) -> None:
        self.settings = settings
        # Handshake fields as this server's code defines them (injectable for tests).
        self.identity = identity
        self._host = host or EmbeddingClientHost(settings.cache_dir)
        self._resolve_address = resolve_address
        self._scheduler = _Scheduler(self._host, settings, clock)
        self._listener: Optional[Listener] = None
        self._connections: set[_ServerConnection] = set()
        self._connections_lock = threading.Lock()
        self._sequence = 0
        self._stopped = threading.Event()

    @property
    def address(self) -> tuple[str, int]:
        assert self._listener is not None, "server has not started"
        return self._listener.address

    def start(self) -> tuple[str, int]:
        """Resolve and bind the listening address, then accept connections."""
        address = self._resolve_address(self.settings.bind)
        # Authentication runs on each connection's own thread (below), so one slow
        # client cannot stall accept().
        if address == "0.0.0.0" and not self.settings.authkey:
            logger.warning(
                "Listening on all interfaces without an authkey: anyone who can reach port "
                "%d can send this server pickled data. Use --bind tailscale or an address, "
                "or set ARC_EMBED_AUTHKEY, unless this network is trusted.",
                self.settings.port,
            )
        self._listener = Listener((address, self.settings.port), family="AF_INET")
        self._scheduler.start()
        threading.Thread(target=self._accept_loop, name="arc-embed-accept", daemon=True).start()
        logger.info("Serving embeddings on %s:%d", *self.address)
        return self.address

    def serve_forever(self) -> None:
        self.start()
        self._stopped.wait()

    def stop(self) -> None:
        self._stopped.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        with self._connections_lock:
            connections = list(self._connections)
        for connection in connections:
            connection.channel.close()
        self._scheduler.stop()

    def wake(self) -> None:
        """Re-evaluate scheduling now (tests advance an injected clock)."""
        self._scheduler.wake()

    def queued_count(self) -> int:
        return self._scheduler.queued_count()

    def _next_sequence(self) -> int:
        with self._connections_lock:
            self._sequence += 1
            return self._sequence

    def _forget(self, connection: _ServerConnection) -> None:
        with self._connections_lock:
            self._connections.discard(connection)

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stopped.is_set():
            try:
                raw = self._listener.accept()
            except OSError:
                if self._stopped.is_set():
                    return
                logger.exception("accept failed")
                continue
            peer = str(getattr(self._listener, "last_accepted", "") or "unknown")
            threading.Thread(
                target=self._serve_connection,
                args=(raw, peer),
                name=f"arc-embed-conn-{peer}",
                daemon=True,
            ).start()

    def _serve_connection(self, raw: Any, peer: str) -> None:
        if self.settings.authkey:
            try:
                deliver_challenge(raw, self.settings.authkey)
                answer_challenge(raw, self.settings.authkey)
            except Exception as exc:
                logger.warning("Rejected connection from %s: %s", peer, exc)
                raw.close()
                return
        connection = _ServerConnection(self, ConnectionChannel(raw), peer)
        with self._connections_lock:
            self._connections.add(connection)
        connection.run()

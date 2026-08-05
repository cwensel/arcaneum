"""Spawn-process lifecycle tests for the persistent PDF layout worker."""

from __future__ import annotations

import os
import time

import pytest

from arcaneum.indexing.pdf.layout_worker import (
    LayoutRequest,
    LayoutWorkerContainmentError,
    LayoutWorkerCrashed,
    LayoutWorkerTimeout,
    PDFLayoutWorker,
)


def _request() -> LayoutRequest:
    return LayoutRequest(
        pdf_path="/tmp/serializable.pdf",
        layout=True,
        ignore_images=True,
        preserve_images=False,
        use_ocr=False,
    )


def _pid_is_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _healthy_target(connection, silence_output):
    connection.send({"type": "ready", "pid": os.getpid()})
    while True:
        message = connection.recv()
        request_id = message["request_id"]
        if message["operation"] == "shutdown":
            connection.send({"type": "stopped", "request_id": request_id})
            break
        if message["operation"] == "health":
            connection.send({"type": "healthy", "request_id": request_id, "pid": os.getpid()})
            continue
        request = message["request"]
        connection.send(
            {
                "type": "result",
                "request_id": request_id,
                "pages": [{"text": request["pdf_path"], "metadata": {"page_number": 1}}],
                "page_count": 1,
                "worker_pid": os.getpid(),
            }
        )
    connection.close()


def _crashing_target(connection, silence_output):
    connection.send({"type": "ready", "pid": os.getpid()})
    connection.recv()
    os._exit(17)


def _hanging_target(connection, silence_output):
    connection.send({"type": "ready", "pid": os.getpid()})
    while True:
        message = connection.recv()
        if message["operation"] == "health":
            connection.send(
                {"type": "healthy", "request_id": message["request_id"], "pid": os.getpid()}
            )
        else:
            time.sleep(60)


def _startup_error_target(connection, silence_output):
    connection.send({"type": "startup_error", "message": "layout import failed"})
    connection.close()


class _UnkillableProcess:
    pid = 424242

    def __init__(self):
        self.terminated = False
        self.killed = False
        self.closed = False

    def is_alive(self):
        return True

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def join(self, timeout=None):
        pass

    def close(self):
        self.closed = True


class _FakeConnection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _ExitCodeAfterJoinProcess:
    """Model platforms where exitcode is unavailable until the child is joined."""

    pid = 424243

    def __init__(self):
        self.joined = False
        self.closed = False

    @property
    def exitcode(self):
        return 17 if self.joined else None

    def is_alive(self):
        return False

    def join(self, timeout=None):
        self.joined = True

    def close(self):
        self.closed = True


def test_worker_is_spawned_healthy_and_reused_across_documents():
    worker = PDFLayoutWorker(process_target=_healthy_target, timeout_seconds=2)
    try:
        first = worker.convert(_request())
        first_pid = first["worker_pid"]
        assert worker.health()["pid"] == first_pid

        second = worker.convert(_request())

        assert second["worker_pid"] == first_pid
        assert worker.generation == 1
        assert worker.completed_requests == 2
    finally:
        pid = worker.pid
        worker.close()
    assert pid is not None
    assert worker._process is None
    assert not _pid_is_alive(pid)


def test_startup_error_is_structured_and_process_is_reaped():
    worker = PDFLayoutWorker(process_target=_startup_error_target, timeout_seconds=2)

    with pytest.raises(LayoutWorkerCrashed, match="layout import failed"):
        worker.health()

    assert worker.pid is None


def test_crashed_worker_is_reaped_and_replaced_on_next_request():
    worker = PDFLayoutWorker(process_target=_crashing_target, timeout_seconds=2)
    with pytest.raises(LayoutWorkerCrashed, match="exit code 17"):
        worker.convert(_request())
    assert worker.pid is None

    worker._process_target = _healthy_target
    try:
        result = worker.convert(_request())
        assert result["worker_pid"] == worker.pid
        assert worker.generation == 2
    finally:
        worker.close()


def test_reap_returns_exit_code_observed_after_join():
    worker = PDFLayoutWorker()
    process = _ExitCodeAfterJoinProcess()
    worker._process = process
    worker._connection = _FakeConnection()

    assert worker._reap(force=True) == 17
    assert process.closed
    assert worker._process is None


def test_repeated_crash_restart_and_close_leaves_no_children_or_handles():
    worker = PDFLayoutWorker(process_target=_crashing_target, timeout_seconds=2)
    crashed_pids = []
    for _ in range(3):
        worker._start()
        crashed_pids.append(worker.pid)
        with pytest.raises(LayoutWorkerCrashed, match="exited"):
            worker.convert(_request())
        assert worker._process is None

    worker._process_target = _healthy_target
    final_pid = worker.convert(_request())["worker_pid"]
    worker.close()

    assert worker._process is None
    assert worker._connection is None
    assert all(pid is not None and not _pid_is_alive(pid) for pid in crashed_pids)
    assert not _pid_is_alive(final_pid)


def test_timed_out_worker_is_terminated_and_reaped():
    worker = PDFLayoutWorker(process_target=_hanging_target, timeout_seconds=0.1)
    assert worker.health()["type"] == "healthy"
    pid = worker.pid

    with pytest.raises(LayoutWorkerTimeout, match="exceeded"):
        worker.convert(_request())

    assert pid is not None
    assert not _pid_is_alive(pid)
    assert worker._process is None
    assert worker.pid is None


def test_repeated_timeout_restart_and_close_leaves_no_children_or_handles():
    worker = PDFLayoutWorker(process_target=_hanging_target, timeout_seconds=0.05)
    timed_out_pids = []
    for _ in range(3):
        worker.health()
        timed_out_pids.append(worker.pid)
        with pytest.raises(LayoutWorkerTimeout):
            worker.convert(_request())
        assert worker._process is None

    worker._process_target = _healthy_target
    result = worker.convert(_request())
    final_pid = result["worker_pid"]
    worker.close()

    assert worker._process is None
    assert worker._connection is None
    assert all(pid is not None and not _pid_is_alive(pid) for pid in timed_out_pids)
    assert not _pid_is_alive(final_pid)


def test_unconfirmed_reap_retains_process_handle_and_raises_containment_error():
    worker = PDFLayoutWorker(process_target=_healthy_target, timeout_seconds=0.05)
    process = _UnkillableProcess()
    connection = _FakeConnection()
    worker._process = process
    worker._connection = connection

    with pytest.raises(LayoutWorkerContainmentError, match="remained alive"):
        worker._reap(force=True)

    assert worker._process is process
    assert worker._connection is None
    assert connection.closed
    assert process.terminated
    assert process.killed
    assert not process.closed

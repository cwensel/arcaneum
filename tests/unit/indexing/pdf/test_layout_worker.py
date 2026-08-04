"""Spawn-process lifecycle tests for the persistent PDF layout worker."""

from __future__ import annotations

import os
import time

import pytest

from arcaneum.indexing.pdf.layout_worker import (
    LayoutRequest,
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
        process = worker._process
        worker.close()
    assert process is not None
    assert not process.is_alive()
    assert process.exitcode == 0


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


def test_timed_out_worker_is_terminated_and_reaped():
    worker = PDFLayoutWorker(process_target=_hanging_target, timeout_seconds=0.1)
    assert worker.health()["type"] == "healthy"
    process = worker._process

    with pytest.raises(LayoutWorkerTimeout, match="exceeded"):
        worker.convert(_request())

    assert process is not None
    assert not process.is_alive()
    assert process.exitcode is not None
    assert worker.pid is None

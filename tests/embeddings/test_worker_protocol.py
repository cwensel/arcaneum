import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
import pytest

from arcaneum.embeddings.worker_protocol import (
    AcceleratorWorkerSession,
    WorkerConfig,
    WorkerContainmentError,
    WorkerCrashedError,
    WorkerProtocolError,
    WorkerTimeoutError,
    _validate_message,
)

FACTORY = "arcaneum.embeddings.worker_protocol:create_deterministic_fake_backend"


def session(**backend_config):
    return AcceleratorWorkerSession(WorkerConfig(FACTORY, backend_config), startup_timeout=3)


def test_spawn_worker_loads_model_once_and_returns_owned_numpy_arrays():
    with session() as worker:
        first = worker.encode(["alpha", "b"], timeout=2)
        second = worker.encode(["gamma"], timeout=2)
        health = worker.health(timeout=2)

        assert first.dtype == np.float32
        assert first.flags.owndata
        assert first.flags.c_contiguous
        assert second.shape == (1, 3)
        assert health["backend"] == {"model_loads": 1, "encodes": 2}
        assert worker._context.get_start_method() == "spawn"


def test_initialization_failure_is_reported_and_reaped():
    worker = session(fail_init=True)
    with pytest.raises(WorkerCrashedError, match="initialization failure"):
        worker.start()
    assert not worker.is_alive
    assert worker.pid is None


def test_crash_is_reported_and_reaped():
    worker = session(crash=True).start()
    with pytest.raises(WorkerCrashedError, match="exited"):
        worker.encode(["boom"], timeout=2)
    assert not worker.is_alive


def test_timeout_terminates_and_reaps_worker():
    worker = session(delay=2).start()
    pid = worker.pid
    with pytest.raises(WorkerTimeoutError):
        worker.encode(["slow"], timeout=0.05)
    assert not worker.is_alive
    assert worker.pid is None
    assert pid not in {child.pid for child in mp.active_children()}


def test_timed_out_encode_cannot_continue_after_reap(tmp_path):
    marker = tmp_path / "encode-completed"
    worker = session(delay=0.3, completion_marker=str(marker)).start()

    with pytest.raises(WorkerTimeoutError):
        worker.encode(["slow"], timeout=0.02)
    time.sleep(0.4)

    assert not marker.exists()
    assert not worker.is_alive


def test_orderly_shutdown_is_idempotent_and_leaves_no_child():
    worker = session().start()
    pid = worker.pid
    worker.shutdown()
    worker.shutdown()
    assert not worker.is_alive
    assert pid not in {child.pid for child in mp.active_children()}


def test_interrupt_reaps_worker(monkeypatch):
    worker = session().start()
    pid = worker.pid

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(worker._replies, "get", interrupt)
    with pytest.raises(KeyboardInterrupt):
        worker.encode(["interrupt"], timeout=2)
    assert not worker.is_alive
    assert pid not in {child.pid for child in mp.active_children()}


def test_malformed_reply_reaps_worker(monkeypatch):
    worker = session().start()
    pid = worker.pid

    monkeypatch.setattr(worker._replies, "get", lambda **kwargs: {"not": "an envelope"})
    with pytest.raises(WorkerProtocolError, match="envelope"):
        worker.encode(["malformed"], timeout=2)
    assert not worker.is_alive
    assert pid not in {child.pid for child in mp.active_children()}


@pytest.mark.parametrize(
    "reply",
    [None, {}, {"version": 1, "type": "wat", "request_id": "x", "payload": {}}],
)
def test_malformed_replies_are_rejected(reply):
    with pytest.raises(WorkerProtocolError):
        _validate_message(reply)


def test_invalid_queue_bound_is_rejected():
    with pytest.raises(ValueError, match="queue_size"):
        WorkerConfig(FACTORY, {}, queue_size=0)


def test_concurrent_callers_are_serialized_without_crossed_replies():
    with session(delay=0.01) as worker:
        inputs = [[f"file-{index}"] for index in range(8)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda value: worker.encode(value, timeout=2), inputs))
        assert [row[0, 0] for row in results] == [len(value[0]) for value in inputs]
        assert worker.health(timeout=2)["backend"]["encodes"] == len(inputs)


def test_unkillable_process_retains_handle_and_never_closes_it():
    worker = session()
    process = MagicMock()
    process.is_alive.return_value = True
    process.pid = 42
    commands, replies = MagicMock(), MagicMock()
    worker._process, worker._commands, worker._replies = process, commands, replies

    with pytest.raises(WorkerContainmentError, match="survived terminate and kill"):
        worker._reap()

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    process.close.assert_not_called()
    assert worker._process is process
    for channel in (commands, replies):
        channel.cancel_join_thread.assert_called_once_with()
        channel.close.assert_called_once_with()
        channel.join_thread.assert_not_called()


def test_forced_reap_does_not_join_blocked_queue_feeders():
    worker = session()
    process = MagicMock()
    process.is_alive.side_effect = [True, False, False]
    commands, replies = MagicMock(), MagicMock()
    worker._process, worker._commands, worker._replies = process, commands, replies

    worker._reap()

    for channel in (commands, replies):
        channel.cancel_join_thread.assert_called_once_with()
        channel.join_thread.assert_not_called()


def test_repeated_timeouts_and_crashes_leave_no_children():
    for _ in range(2):
        timed = session(delay=0.2).start()
        with pytest.raises(WorkerTimeoutError):
            timed.encode(["slow"], timeout=0.01)
        crashed = session(crash=True).start()
        with pytest.raises(WorkerCrashedError):
            crashed.encode(["boom"], timeout=1)
    assert not [child for child in mp.active_children() if child.name == "arcaneum-accelerator"]

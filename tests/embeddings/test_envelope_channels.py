"""Envelope channels shared by the spawn-queue and socket transports (RDR-025)."""

import queue
import threading
from multiprocessing.connection import Client, Listener, Pipe

import pytest

from arcaneum.embeddings.worker_protocol import (
    ConnectionChannel,
    MessageType,
    QueueChannel,
    WorkerProtocolError,
    make_message,
)


def queue_pair():
    commands, replies = queue.Queue(), queue.Queue()
    return QueueChannel(commands, replies), QueueChannel(replies, commands)


def pipe_pair():
    left, right = Pipe()
    return ConnectionChannel(left), ConnectionChannel(right)


@pytest.fixture(params=[queue_pair, pipe_pair], ids=["queue", "connection"])
def channels(request):
    parent, child = request.param()
    yield parent, child
    parent.close()
    child.close()


def test_channels_carry_validated_envelopes_both_ways(channels):
    parent, child = channels
    request = make_message(MessageType.ENCODE, "req-1", texts=["alpha"])

    parent.send(request)
    received = child.receive(timeout=1)
    child.send(make_message(MessageType.ENCODED, "req-1", embeddings=[[1.0]]))
    reply = parent.receive(timeout=1)

    assert received == request
    assert reply["type"] == MessageType.ENCODED
    assert reply["payload"] == {"embeddings": [[1.0]]}


def test_receive_returns_none_when_nothing_arrives_in_time(channels):
    parent, _ = channels

    assert parent.receive(timeout=0.01) is None


def test_receive_rejects_malformed_envelopes(channels):
    parent, child = channels
    if isinstance(parent, QueueChannel):
        child._outgoing.put({"not": "an envelope"})
    else:
        child._connection.send({"not": "an envelope"})

    with pytest.raises(WorkerProtocolError, match="envelope"):
        parent.receive(timeout=1)


def test_connection_channel_reports_a_closed_peer_as_eof():
    parent, child = pipe_pair()
    child.close()

    with pytest.raises(EOFError):
        parent.receive(timeout=1)
    parent.close()


def test_connection_channel_serializes_concurrent_senders():
    parent, child = pipe_pair()
    count = 200

    def send_many(prefix):
        for index in range(count):
            parent.send(make_message(MessageType.HEARTBEAT, f"{prefix}-{index}"))

    threads = [threading.Thread(target=send_many, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    received = [child.receive(timeout=2)["request_id"] for _ in range(2 * count)]
    for thread in threads:
        thread.join()

    for prefix in ("a", "b"):
        ordered = [rid for rid in received if rid.startswith(f"{prefix}-")]
        assert ordered == [f"{prefix}-{index}" for index in range(count)]
    parent.close()
    child.close()


def test_connection_channel_works_over_an_authenticated_socket():
    with Listener(("127.0.0.1", 0), authkey=b"secret") as listener:
        accepted = {}

        def accept():
            accepted["channel"] = ConnectionChannel(listener.accept())

        thread = threading.Thread(target=accept)
        thread.start()
        client = ConnectionChannel(Client(listener.address, authkey=b"secret"))
        thread.join(timeout=5)
        server = accepted["channel"]

        client.send(make_message(MessageType.HEARTBEAT, "beat"))
        assert server.receive(timeout=1)["request_id"] == "beat"
        server.send(make_message(MessageType.HEALTH, "beat", state="idle"))
        assert client.receive(timeout=1)["payload"] == {"state": "idle"}
        client.close()
        server.close()

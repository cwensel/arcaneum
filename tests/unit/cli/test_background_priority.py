"""--background: run a sync or repair below interactive work on this machine."""

import os
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from arcaneum.cli import background
from arcaneum.cli.main import cli


class FakeSystem:
    """Records priority changes instead of making them."""

    def __init__(self, nice=None):
        self.nice = dict(nice or {})
        self.set_calls = []
        self.scheduler_calls = []

    def setpriority(self, which, who, value):
        self.set_calls.append((which, who, value))
        self.nice[who] = value

    def getpriority(self, which, who):
        return self.nice.get(who, 0)

    def sched_setscheduler(self, tid, policy, param):
        self.scheduler_calls.append((tid, policy))


@pytest.fixture
def system(monkeypatch):
    fake = FakeSystem()
    monkeypatch.setattr(background.os, "setpriority", fake.setpriority)
    monkeypatch.setattr(background.os, "getpriority", fake.getpriority)
    return fake


def test_macos_lowers_arc_with_nice_only_and_leaves_docker_alone(system):
    # No darwin background band: it throttles disk I/O, swap page-ins included,
    # and under memory pressure a drain in it stalled for 25h at zero CPU. No
    # Docker VM renice: it outlives the run and slows every later search.
    notes = background.enter_background("darwin")

    assert system.set_calls == [(os.PRIO_PROCESS, 0, background.MACOS_NICE)]
    assert any(f"nice {background.MACOS_NICE}" in note for note in notes)


def test_macos_keeps_a_lower_priority_already_set(system):
    system.nice[0] = 15

    notes = background.enter_background("darwin")

    assert system.set_calls == []
    assert any("already at nice 15" in note for note in notes)


def linux_scheduler(monkeypatch, system):
    """Linux scheduling calls, which macOS Python lacks."""
    monkeypatch.setattr(
        background.os, "sched_setscheduler", system.sched_setscheduler, raising=False
    )
    monkeypatch.setattr(background.os, "sched_param", lambda priority: priority, raising=False)
    monkeypatch.setattr(background.os, "SCHED_IDLE", 5, raising=False)


def test_linux_idles_every_arc_thread_and_leaves_docker_alone(system, monkeypatch):
    monkeypatch.setattr(background, "_thread_ids", lambda: [100, 101])
    linux_scheduler(monkeypatch, system)

    notes = background.enter_background("linux")

    assert system.scheduler_calls == [(100, 5), (101, 5)]
    assert (os.PRIO_PROCESS, 100, 19) in system.set_calls
    assert (os.PRIO_PROCESS, 101, 19) in system.set_calls
    assert any("Docker unchanged" in note for note in notes)


def test_linux_keeps_a_lower_priority_already_set(system, monkeypatch):
    monkeypatch.setattr(background, "_thread_ids", lambda: [100])
    linux_scheduler(monkeypatch, system)
    system.nice[100] = 20

    background.enter_background("linux")

    assert not [call for call in system.set_calls if call[0] == os.PRIO_PROCESS]


def test_other_platforms_change_nothing(system):
    notes = background.enter_background("win32")

    assert system.set_calls == []
    assert any("not supported" in note for note in notes)


def record_background(order):
    return patch(
        "arcaneum.cli.background.enter_background",
        side_effect=lambda: order.append("bg") or [],
    )


def test_corpus_sync_background_enters_background_before_syncing(tmp_path):
    order = []
    with (
        record_background(order),
        patch(
            "arcaneum.cli.sync.sync_directory_command",
            side_effect=lambda *a, **k: order.append("sync"),
        ),
    ):
        result = CliRunner().invoke(
            cli, ["corpus", "sync", "MyCorpus", str(tmp_path), "--background"]
        )

    assert result.exit_code == 0, result.output
    assert order == ["bg", "sync"]


def test_corpus_repair_background_enters_background_before_repairing():
    order = []
    with (
        record_background(order),
        patch(
            "arcaneum.cli.sync.repair_corpus_command",
            side_effect=lambda *a, **k: order.append("repair"),
        ),
    ):
        result = CliRunner().invoke(cli, ["corpus", "repair", "MyCorpus", "--background"])

    assert result.exit_code == 0, result.output
    assert order == ["bg", "repair"]


def test_corpus_sync_without_background_leaves_priority_alone(tmp_path):
    with (
        patch("arcaneum.cli.background.enter_background") as enter,
        patch("arcaneum.cli.sync.sync_directory_command"),
    ):
        result = CliRunner().invoke(cli, ["corpus", "sync", "MyCorpus", str(tmp_path)])

    assert result.exit_code == 0, result.output
    enter.assert_not_called()


def test_corpus_sync_drain_spool_background_enters_background_before_draining():
    order = []
    with (
        record_background(order),
        patch(
            "arcaneum.cli.main._drain_corpus_spool",
            side_effect=lambda *a, **k: order.append("drain"),
        ),
    ):
        result = CliRunner().invoke(
            cli, ["corpus", "sync", "MyCorpus", "--drain-spool", "--background"]
        )

    assert result.exit_code == 0, result.output
    assert order == ["bg", "drain"]

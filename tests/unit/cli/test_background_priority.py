"""--background: run a sync or repair below interactive work on this machine."""

import os
import subprocess
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from arcaneum.cli import background
from arcaneum.cli.main import cli


class FakeSystem:
    """Records priority changes instead of making them."""

    def __init__(self, vm_pids=(4242,), docker_running=True, nice=None, refuse=()):
        self.vm_pids = list(vm_pids)
        self.docker_running = docker_running
        self.nice = dict(nice or {})
        self.refuse = set(refuse)
        self.set_calls = []
        self.scheduler_calls = []

    def setpriority(self, which, who, value):
        if (which, who) in self.refuse:
            raise PermissionError("not permitted")
        self.set_calls.append((which, who, value))
        self.nice[who] = value

    def getpriority(self, which, who):
        return self.nice.get(who, 0)

    def sched_setscheduler(self, tid, policy, param):
        self.scheduler_calls.append((tid, policy))

    def run(self, argv, **kwargs):
        pattern = argv[-1]
        if pattern == background.DOCKER_DESKTOP_PROCESS:
            pids = [1] if self.docker_running else []
        else:
            pids = self.vm_pids
        stdout = "".join(f"{pid}\n" for pid in pids)
        return subprocess.CompletedProcess(argv, 0 if pids else 1, stdout, "")


@pytest.fixture
def system(monkeypatch):
    fake = FakeSystem()
    monkeypatch.setattr(background.os, "setpriority", fake.setpriority)
    monkeypatch.setattr(background.os, "getpriority", fake.getpriority)
    monkeypatch.setattr(background.subprocess, "run", fake.run)
    return fake


def test_macos_puts_arc_in_the_darwin_background_band(system):
    background.enter_background("darwin")

    assert (background.PRIO_DARWIN_PROCESS, 0, background.PRIO_DARWIN_BG) in system.set_calls


def test_macos_lowers_the_docker_vm_slightly(system):
    notes = background.enter_background("darwin")

    assert (os.PRIO_PROCESS, 4242, background.DOCKER_VM_NICE) in system.set_calls
    assert any("Docker VM" in note and "nice 5" in note for note in notes)


def test_macos_never_raises_a_docker_vm_already_set_lower(system):
    system.nice[4242] = 10

    notes = background.enter_background("darwin")

    assert not [call for call in system.set_calls if call[1] == 4242]
    assert any("already at nice 10" in note for note in notes)


@pytest.mark.parametrize("vm_pids", [(), (4242, 4343)])
def test_macos_leaves_vms_alone_unless_exactly_one_runs(system, vm_pids):
    system.vm_pids = list(vm_pids)

    notes = background.enter_background("darwin")

    assert [call for call in system.set_calls if call[1] != 0] == []
    assert any("Docker VM unchanged" in note for note in notes)


def test_macos_leaves_a_vm_alone_when_docker_desktop_is_not_running(system):
    system.docker_running = False

    background.enter_background("darwin")

    assert [call for call in system.set_calls if call[1] != 0] == []


def test_macos_reports_a_vm_it_may_not_renice_and_carries_on(system):
    system.refuse.add((os.PRIO_PROCESS, 4242))

    notes = background.enter_background("darwin")

    assert (background.PRIO_DARWIN_PROCESS, 0, background.PRIO_DARWIN_BG) in system.set_calls
    assert any("Docker VM unchanged" in note for note in notes)


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

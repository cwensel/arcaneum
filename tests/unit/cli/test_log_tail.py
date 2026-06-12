from datetime import datetime, timezone
from unittest.mock import patch

from click.testing import CliRunner

from arcaneum.cli.logs import CurrentLogTailer, current_interaction_log_path
from arcaneum.cli.main import cli


def test_current_interaction_log_path_uses_utc_date(tmp_path):
    def now():
        return datetime(2026, 6, 12, 23, 30, tzinfo=timezone.utc)

    assert current_interaction_log_path(tmp_path, now) == (
        tmp_path / "arc-interactions-2026-06-12.log"
    )


def test_tailer_starts_at_eof_by_default(tmp_path):
    log_file = tmp_path / "arc-interactions-2026-06-12.log"
    log_file.write_text("old\n", encoding="utf-8")

    def now():
        return datetime(2026, 6, 12, tzinfo=timezone.utc)

    tailer = CurrentLogTailer(log_dir=tmp_path, now=now)

    assert tailer.poll() == ""

    with log_file.open("a", encoding="utf-8") as f:
        f.write("new\n")

    assert tailer.poll() == "new\n"


def test_tailer_can_emit_initial_lines(tmp_path):
    log_file = tmp_path / "arc-interactions-2026-06-12.log"
    log_file.write_text("one\ntwo\nthree\n", encoding="utf-8")

    def now():
        return datetime(2026, 6, 12, tzinfo=timezone.utc)

    tailer = CurrentLogTailer(log_dir=tmp_path, lines=2, now=now)

    assert tailer.poll() == "two\nthree\n"


def test_tailer_fills_initial_lines_from_previous_logs(tmp_path):
    first_log = tmp_path / "arc-interactions-2026-06-11.log"
    first_log.write_text("one\ntwo\n", encoding="utf-8")
    second_log = tmp_path / "arc-interactions-2026-06-12.log"
    second_log.write_text("three\nfour\n", encoding="utf-8")

    def now():
        return datetime(2026, 6, 12, tzinfo=timezone.utc)

    tailer = CurrentLogTailer(log_dir=tmp_path, lines=3, now=now)

    assert tailer.poll() == "two\nthree\nfour\n"


def test_tailer_fills_initial_lines_when_current_log_is_missing(tmp_path):
    old_log = tmp_path / "arc-interactions-2026-06-11.log"
    old_log.write_text("one\ntwo\n", encoding="utf-8")

    def now():
        return datetime(2026, 6, 12, tzinfo=timezone.utc)

    tailer = CurrentLogTailer(log_dir=tmp_path, lines=2, now=now)

    assert tailer.poll() == "one\ntwo\n"

    current_log = tmp_path / "arc-interactions-2026-06-12.log"
    current_log.write_text("three\n", encoding="utf-8")

    assert tailer.poll() == "three\n"


def test_tailer_switches_to_new_log_on_utc_cutover(tmp_path):
    clock = [datetime(2026, 6, 12, 23, 59, tzinfo=timezone.utc)]

    def now():
        return clock[0]

    old_log = tmp_path / "arc-interactions-2026-06-12.log"
    old_log.write_text("before\n", encoding="utf-8")
    tailer = CurrentLogTailer(log_dir=tmp_path, now=now)

    assert tailer.poll() == ""

    old_log.write_text("before\nlast-old\n", encoding="utf-8")
    assert tailer.poll() == "last-old\n"

    clock[0] = datetime(2026, 6, 13, 0, 0, tzinfo=timezone.utc)
    new_log = tmp_path / "arc-interactions-2026-06-13.log"
    new_log.write_text("first-new\n", encoding="utf-8")

    assert tailer.poll() == "first-new\n"


def test_log_tail_entrypoint_dispatches():
    called = {}

    def fake_tail_current_log(*, lines, poll_interval):
        called["lines"] = lines
        called["poll_interval"] = poll_interval

    runner = CliRunner()
    with patch("arcaneum.cli.logs.tail_current_log", fake_tail_current_log):
        result = runner.invoke(cli, ["log", "tail", "--lines", "5"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert called == {"lines": 5, "poll_interval": 1.0}

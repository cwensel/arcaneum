"""OS-level spool watchers behind `arc corpus hook install --service` (kata vq0n).

launchd (macOS) and systemd (Linux) can watch the spool directory and drain it
after a reboot or a failed spawn, the way `git maintenance start` registers
itself. These tests exercise the file generation, not the OS registration.
"""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from arcaneum.cli import spool_service


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path))
    yield tmp_path


# --- launchd ------------------------------------------------------------------


def test_launchd_plist_is_valid_and_watches_the_spool(isolated):
    from arcaneum.cli import spool

    body = spool_service.render_launchd_plist("Docs", arc_bin="/usr/local/bin/arc")
    parsed = plistlib.loads(body)

    assert parsed["Label"] == spool_service.service_label("Docs")
    assert str(spool.corpus_spool_dir("Docs")) in parsed["QueueDirectories"]
    assert "--drain-spool" in parsed["ProgramArguments"]
    assert "Docs" in parsed["ProgramArguments"]


def test_launchd_plist_does_not_run_at_load(isolated):
    """QueueDirectories should trigger it; running on load would sync needlessly."""
    parsed = plistlib.loads(spool_service.render_launchd_plist("Docs", arc_bin="/bin/arc"))
    assert parsed.get("RunAtLoad", False) is False


def test_launchd_label_is_stable_and_corpus_scoped(isolated):
    assert spool_service.service_label("Docs") == spool_service.service_label("Docs")
    assert spool_service.service_label("Docs") != spool_service.service_label("Code")


# --- systemd ------------------------------------------------------------------


def test_systemd_path_unit_watches_the_spool_directory(isolated):
    from arcaneum.cli import spool

    unit = spool_service.render_systemd_path_unit("Docs")
    assert "DirectoryNotEmpty=" in unit
    assert str(spool.corpus_spool_dir("Docs")) in unit


def test_systemd_service_unit_runs_the_drain(isolated):
    unit = spool_service.render_systemd_service_unit("Docs", arc_bin="/usr/bin/arc")
    assert "--drain-spool" in unit
    assert "/usr/bin/arc" in unit
    assert "Type=oneshot" in unit


def test_systemd_units_are_corpus_scoped(isolated):
    assert "Docs" in spool_service.render_systemd_service_unit("Docs", arc_bin="/usr/bin/arc")
    assert "Code" in spool_service.render_systemd_service_unit("Code", arc_bin="/usr/bin/arc")


# --- corpus names with awkward characters -------------------------------------


def test_a_corpus_name_with_quotes_does_not_corrupt_the_plist(isolated):
    body = spool_service.render_launchd_plist("It's Docs", arc_bin="/bin/arc")
    parsed = plistlib.loads(body)
    assert "It's Docs" in parsed["ProgramArguments"]


def test_service_label_is_filename_safe(isolated):
    label = spool_service.service_label("../../evil")
    assert "/" not in label


# --- install / uninstall are safe without the OS present ----------------------


def test_install_on_an_unsupported_platform_is_reported_not_fatal(isolated, monkeypatch):
    monkeypatch.setattr(spool_service.platform, "system", lambda: "Plan9")
    result = spool_service.install("Docs")
    assert result is None


def test_uninstall_when_nothing_installed_is_not_an_error(isolated, monkeypatch):
    monkeypatch.setattr(spool_service.platform, "system", lambda: "Plan9")
    spool_service.uninstall("Docs")


def test_install_writes_the_plist_on_darwin(isolated, monkeypatch):
    monkeypatch.setattr(spool_service.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(spool_service.shutil, "which", lambda _: "/usr/local/bin/arc")
    loaded = []
    monkeypatch.setattr(spool_service, "_run", lambda args: loaded.append(args))

    path = spool_service.install("Docs")

    assert path is not None
    assert Path(path).exists()
    assert plistlib.loads(Path(path).read_bytes())["Label"] == spool_service.service_label("Docs")
    assert loaded, "should have asked launchctl to load the job"


def test_install_without_arc_on_path_is_reported_not_fatal(isolated, monkeypatch):
    monkeypatch.setattr(spool_service.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(spool_service.shutil, "which", lambda _: None)
    assert spool_service.install("Docs") is None


def test_uninstall_removes_the_plist_on_darwin(isolated, monkeypatch):
    monkeypatch.setattr(spool_service.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(spool_service.shutil, "which", lambda _: "/usr/local/bin/arc")
    monkeypatch.setattr(spool_service, "_run", lambda args: None)

    path = Path(spool_service.install("Docs"))
    assert path.exists()

    spool_service.uninstall("Docs")
    assert not path.exists()


# --- OS-driven drains must be memory-bounded ---------------------------------


def test_launchd_plist_caps_the_embedding_batch(isolated):
    """An unbounded OS drain drove swap to 13GB of 14.3GB on a real burst.

    A single large markdown file added 4.4GB of RSS; nothing capped the batch
    because the plist ran a bare --drain-spool.
    """
    parsed = plistlib.loads(spool_service.render_launchd_plist("Docs", arc_bin="/bin/arc"))
    args = parsed["ProgramArguments"]
    assert "--max-embedding-batch" in args
    assert args[args.index("--max-embedding-batch") + 1] == str(
        spool_service.DEFAULT_SERVICE_EMBEDDING_BATCH
    )


def test_systemd_service_caps_the_embedding_batch(isolated):
    unit = spool_service.render_systemd_service_unit("Docs", arc_bin="/bin/arc")
    assert f"--max-embedding-batch {spool_service.DEFAULT_SERVICE_EMBEDDING_BATCH}" in unit


def test_the_cap_is_conservative_enough_to_matter(isolated):
    """A cap only helps if it is well under the auto-tuned default."""
    assert 1 <= spool_service.DEFAULT_SERVICE_EMBEDDING_BATCH <= 32


def test_the_cap_is_overridable(isolated):
    body = spool_service.render_launchd_plist("Docs", arc_bin="/bin/arc", embedding_batch=4)
    args = plistlib.loads(body)["ProgramArguments"]
    assert args[args.index("--max-embedding-batch") + 1] == "4"


def test_the_cap_can_be_disabled(isolated):
    """Passing None restores the auto-tuned behavior for anyone who wants it."""
    args = plistlib.loads(
        spool_service.render_launchd_plist("Docs", arc_bin="/bin/arc", embedding_batch=None)
    )["ProgramArguments"]
    assert "--max-embedding-batch" not in args

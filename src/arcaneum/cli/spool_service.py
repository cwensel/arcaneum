"""OS-managed spool watchers for `arc corpus hook install --service` (kata vq0n).

The git hook spawns a drain worker itself, which covers the normal case. Two
cases it cannot cover: the machine rebooted with entries still spooled, and the
spawn failed (no `arc` on PATH at hook time, a killed worker). Registering the
spool directory with the OS closes both — launchd's `QueueDirectories` and
systemd's `DirectoryNotEmpty=` both fire when the directory stops being empty,
which is exactly the condition the hook creates. This is the same pattern
`git maintenance start` uses to register itself.

Registration is best-effort: an unsupported platform, a missing `arc`, or a
launchctl/systemctl failure is reported to the caller as "not registered"
rather than raised. The hook still works without it.
"""

from __future__ import annotations

import logging
import platform
import plistlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from ..paths import get_config_dir
from . import spool
from .hook_log import hook_log_path

logger = logging.getLogger(__name__)

LABEL_PREFIX = "net.arcaneum.spool"

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_name(corpus: str) -> str:
    return _UNSAFE_NAME_CHARS.sub("_", corpus)[:64] or "corpus"


def service_label(corpus: str) -> str:
    """Stable, filename-safe identifier for this corpus's watcher."""
    return f"{LABEL_PREFIX}.{_safe_name(corpus)}"


def _run(args: List[str]) -> None:
    subprocess.run(args, capture_output=True, text=True, timeout=30)


# --- launchd (macOS) ----------------------------------------------------------


def launchd_plist_path(corpus: str) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{service_label(corpus)}.plist"


def render_launchd_plist(corpus: str, *, arc_bin: str) -> bytes:
    """Render the LaunchAgent that drains this corpus's spool.

    plistlib does the escaping, so a corpus name containing quotes or angle
    brackets cannot corrupt the XML.
    """
    job = {
        "Label": service_label(corpus),
        "ProgramArguments": [arc_bin, "corpus", "sync", corpus, "--drain-spool"],
        # Fire when the hook drops an entry in, not when the agent loads:
        # RunAtLoad would sync on every login for no reason.
        "QueueDirectories": [str(spool.corpus_spool_dir(corpus))],
        "RunAtLoad": False,
        "LowPriorityIO": True,
        "Nice": 5,
        # Without these, a launchd-driven drain writes nowhere: the worker's
        # output is discarded and only the hook's own spool lines survive.
        "StandardOutPath": str(hook_log_path()),
        "StandardErrorPath": str(hook_log_path()),
    }
    return plistlib.dumps(job)


def _install_launchd(corpus: str, arc_bin: str) -> Optional[str]:
    path = launchd_plist_path(corpus)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The directory must exist before launchd watches it.
    spool.corpus_spool_dir(corpus).mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_launchd_plist(corpus, arc_bin=arc_bin))

    try:
        # Unload first so a re-install picks up a changed plist.
        _run(["launchctl", "unload", str(path)])
        _run(["launchctl", "load", str(path)])
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Wrote %s but could not load it with launchctl: %s", path, exc)
    return str(path)


def _uninstall_launchd(corpus: str) -> None:
    path = launchd_plist_path(corpus)
    if not path.exists():
        return
    try:
        _run(["launchctl", "unload", str(path)])
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("launchctl unload failed for %s: %s", path, exc)
    try:
        path.unlink()
    except OSError as exc:
        logger.warning("Could not remove %s: %s", path, exc)


# --- systemd (Linux) ----------------------------------------------------------


def systemd_unit_dir() -> Path:
    return get_config_dir().parent / "systemd" / "user"


def systemd_path_unit_path(corpus: str) -> Path:
    return systemd_unit_dir() / f"{service_label(corpus)}.path"


def systemd_service_unit_path(corpus: str) -> Path:
    return systemd_unit_dir() / f"{service_label(corpus)}.service"


def render_systemd_path_unit(corpus: str) -> str:
    # Sanitized in the free-text field too: a name carrying a newline could
    # otherwise inject a directive into the unit file.
    safe = _safe_name(corpus)
    return f"""[Unit]
Description=Watch the Arcaneum sync spool for corpus {safe}

[Path]
DirectoryNotEmpty={spool.corpus_spool_dir(corpus)}
Unit={service_label(corpus)}.service

[Install]
WantedBy=default.target
"""


def render_systemd_service_unit(corpus: str, *, arc_bin: str) -> str:
    # systemd splits ExecStart on whitespace; quote the corpus name so one
    # containing spaces stays a single argument.
    quoted = corpus.replace('"', '\\"').replace("\n", " ").replace("\r", " ")
    safe = _safe_name(corpus)
    return f"""[Unit]
Description=Drain the Arcaneum sync spool for corpus {safe}

[Service]
Type=oneshot
Nice=5
IOSchedulingClass=idle
ExecStart={arc_bin} corpus sync "{quoted}" --drain-spool
StandardOutput=append:{hook_log_path()}
StandardError=append:{hook_log_path()}
"""


def _install_systemd(corpus: str, arc_bin: str) -> Optional[str]:
    unit_dir = systemd_unit_dir()
    unit_dir.mkdir(parents=True, exist_ok=True)
    spool.corpus_spool_dir(corpus).mkdir(parents=True, exist_ok=True)

    service_path = systemd_service_unit_path(corpus)
    path_path = systemd_path_unit_path(corpus)
    service_path.write_text(render_systemd_service_unit(corpus, arc_bin=arc_bin))
    path_path.write_text(render_systemd_path_unit(corpus))

    try:
        _run(["systemctl", "--user", "daemon-reload"])
        _run(["systemctl", "--user", "enable", "--now", path_path.name])
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Wrote %s but could not enable it: %s", path_path, exc)
    return str(path_path)


def _uninstall_systemd(corpus: str) -> None:
    path_path = systemd_path_unit_path(corpus)
    service_path = systemd_service_unit_path(corpus)
    if path_path.exists():
        try:
            _run(["systemctl", "--user", "disable", "--now", path_path.name])
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("systemctl disable failed: %s", exc)
    for path in (path_path, service_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Could not remove %s: %s", path, exc)
    try:
        _run(["systemctl", "--user", "daemon-reload"])
    except (OSError, subprocess.SubprocessError):
        pass


# --- public API ---------------------------------------------------------------


def install(corpus: str) -> Optional[str]:
    """Register an OS watcher for this corpus's spool.

    Returns the path of the unit/plist written, or None when the platform is
    unsupported or `arc` is not on PATH — the hook works either way, so this
    never raises.
    """
    arc_bin = shutil.which("arc")
    if not arc_bin:
        logger.warning("`arc` is not on PATH; skipping spool service registration.")
        return None

    system = platform.system()
    if system == "Darwin":
        return _install_launchd(corpus, arc_bin)
    if system == "Linux":
        return _install_systemd(corpus, arc_bin)

    logger.warning("No spool service integration for %s; skipping.", system)
    return None


def uninstall(corpus: str) -> None:
    """Remove this corpus's OS watcher, if one was registered."""
    system = platform.system()
    if system == "Darwin":
        _uninstall_launchd(corpus)
    elif system == "Linux":
        _uninstall_systemd(corpus)

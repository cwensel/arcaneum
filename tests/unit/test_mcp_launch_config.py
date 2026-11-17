"""`arc mcp --print-config`: launch snippets that survive launchd's fixed PATH (RDR-026 Q1).

The review worker runs under launchd with a fixed PATH that leaves the
directory holding `arc` off it, so a client config must name an absolute
command. These tests parse each snippet the way its client does and check that
the command it names is an absolute path to an existing executable, that the
allowlist rides along, and that the snippet launches under a launchd-like PATH.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from arcaneum.mcp_server import launch_command, render_config

ROOT = Path(__file__).resolve().parents[2]
LAUNCHD_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"


# Importing arcaneum creates the empty model cache directory (paths.py's
# configure_model_cache_env, for every arc command); nothing else may appear.
CACHE_DIRS = [".cache", ".cache/arcaneum", ".cache/arcaneum/models"]


def _env(home: Path, pythonpath: bool = True, **extra):
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "PATH": LAUNCHD_PATH,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_DATA_HOME": str(home / ".local" / "share"),
        "XDG_STATE_HOME": str(home / ".local" / "state"),
    }
    env.pop("ARC_MCP_CORPORA", None)
    env.pop("HF_HOME", None)
    if not pythonpath:
        # As launchd starts a client: nothing points Python at this checkout.
        env.pop("PYTHONPATH")
    env.update(extra)
    return env


def _script(tmp_path: Path, shebang: str, executable: bool = True) -> Path:
    script = tmp_path / "bin" / "arc"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        f"{shebang}\nimport sys\nfrom arcaneum.cli.entrypoint import run\nsys.exit(run())\n"
    )
    script.chmod(0o755 if executable else 0o644)
    return script


def _assert_absolute_executable(command: str) -> None:
    assert os.path.isabs(command), command
    assert os.path.isfile(command), command
    assert os.access(command, os.X_OK), command


def _codex_overrides(snippet: str) -> dict:
    """Each `-c key=value` override the codex snippet lists, parsed as codex does (TOML)."""
    overrides = {}
    for line in snippet.splitlines():
        words = shlex.split(line.lstrip("#").strip()) if line.startswith("#") else []
        if len(words) == 2 and words[0] == "-c":
            key, _, value = words[1].partition("=")
            overrides[key] = tomllib.loads(f"value = {value}")["value"]
    return overrides


# ---------------------------------------------------------------------------
# The absolute command
# ---------------------------------------------------------------------------


class TestLaunchCommand:
    def test_a_script_with_an_absolute_interpreter_is_its_own_command(self, tmp_path):
        script = _script(tmp_path, f"#!{sys.executable}")
        assert launch_command(argv0=str(script), executable=sys.executable) == [
            os.path.realpath(script)
        ]

    def test_a_relative_argv0_resolves_to_an_absolute_path(self, tmp_path, monkeypatch):
        script = _script(tmp_path, f"#!{sys.executable}")
        monkeypatch.chdir(tmp_path)
        command = launch_command(argv0="bin/arc", executable=sys.executable)
        assert command == [os.path.realpath(script)]

    def test_an_env_shebang_runs_the_script_under_the_interpreter(self, tmp_path):
        # The script keeps its own sys.path setup (a checkout's bin/arc adds src).
        script = _script(tmp_path, "#!/usr/bin/env python3")
        assert launch_command(argv0=str(script), executable=sys.executable) == [
            sys.executable,
            os.path.realpath(script),
        ]

    def test_the_checkout_launcher_runs_under_the_interpreter(self):
        launcher = ROOT / "bin" / "arc"
        assert launch_command(argv0=str(launcher), executable=sys.executable) == [
            sys.executable,
            os.path.realpath(launcher),
        ]

    def test_a_relative_interpreter_runs_the_script_under_the_interpreter(self, tmp_path):
        script = _script(tmp_path, "#!python3")
        assert launch_command(argv0=str(script), executable=sys.executable) == [
            sys.executable,
            os.path.realpath(script),
        ]

    def test_a_missing_interpreter_runs_the_script_under_the_interpreter(self, tmp_path):
        script = _script(tmp_path, f"#!{tmp_path}/no-such-python")
        assert launch_command(argv0=str(script), executable=sys.executable) == [
            sys.executable,
            os.path.realpath(script),
        ]

    def test_an_executable_without_a_shebang_falls_back(self, tmp_path):
        script = tmp_path / "arc"
        script.write_bytes(b"\x7fELF")
        script.chmod(0o755)
        assert launch_command(argv0=str(script), executable=sys.executable) == [
            sys.executable,
            "-m",
            "arcaneum.cli.main",
        ]

    def test_a_non_executable_argv0_falls_back(self, tmp_path):
        script = _script(tmp_path, f"#!{sys.executable}", executable=False)
        assert launch_command(argv0=str(script), executable=sys.executable)[1:] == [
            "-m",
            "arcaneum.cli.main",
        ]

    @pytest.mark.parametrize("argv0", ["", "-c", "arc-not-on-disk"])
    def test_no_script_on_disk_falls_back(self, argv0):
        assert launch_command(argv0=argv0, executable=sys.executable) == [
            sys.executable,
            "-m",
            "arcaneum.cli.main",
        ]


# ---------------------------------------------------------------------------
# The snippets
# ---------------------------------------------------------------------------


class TestRenderConfig:
    command = [sys.executable, "-m", "arcaneum.cli.main"]

    def test_claude_snippet_is_mcp_config_json(self):
        snippet = render_config("claude", self.command, corpora=["DevRef", "Notes"])
        server = json.loads(snippet)["mcpServers"]["arc"]
        _assert_absolute_executable(server["command"])
        assert server["args"] == [
            "-m",
            "arcaneum.cli.main",
            "mcp",
            "--corpus",
            "DevRef",
            "--corpus",
            "Notes",
        ]

    def test_codex_snippet_is_a_toml_table_and_matching_overrides(self):
        snippet = render_config("codex", self.command, corpora=["DevRef"])
        server = tomllib.loads(snippet)["mcp_servers"]["arc"]
        _assert_absolute_executable(server["command"])
        assert server["args"] == ["-m", "arcaneum.cli.main", "mcp", "--corpus", "DevRef"]
        assert _codex_overrides(snippet) == {
            "mcp_servers.arc.command": server["command"],
            "mcp_servers.arc.args": server["args"],
        }

    def test_no_allowlist_serves_every_corpus(self):
        server = json.loads(render_config("claude", self.command))["mcpServers"]["arc"]
        assert server["args"] == ["-m", "arcaneum.cli.main", "mcp"]

    def test_non_default_limits_are_echoed(self):
        snippet = render_config("codex", self.command, timeout=20, max_output_chars=8000)
        assert tomllib.loads(snippet)["mcp_servers"]["arc"]["args"][3:] == [
            "--timeout",
            "20",
            "--max-output-chars",
            "8000",
        ]

    @pytest.mark.parametrize("timeout", [30.00001, 1.23456789, 2.5, 1e-07])
    def test_a_fractional_timeout_is_echoed_exactly(self, timeout):
        snippet = render_config("claude", self.command, timeout=timeout)
        args = json.loads(snippet)["mcpServers"]["arc"]["args"]
        assert float(args[args.index("--timeout") + 1]) == timeout

    def test_default_limits_are_not_echoed(self):
        snippet = render_config("claude", self.command, timeout=30.0, max_output_chars=40000)
        assert json.loads(snippet)["mcpServers"]["arc"]["args"] == [
            "-m",
            "arcaneum.cli.main",
            "mcp",
        ]

    def test_quotes_and_non_ascii_survive_both_formats(self):
        command = ['/opt/My "arc"/bin/arc']
        corpora = ["Café's"]
        claude = json.loads(render_config("claude", command, corpora=corpora))["mcpServers"]["arc"]
        codex_snippet = render_config("codex", command, corpora=corpora)
        codex = tomllib.loads(codex_snippet)["mcp_servers"]["arc"]
        assert claude["command"] == codex["command"] == command[0]
        assert claude["args"] == codex["args"] == ["mcp", "--corpus", "Café's"]
        assert _codex_overrides(codex_snippet)["mcp_servers.arc.args"] == codex["args"]

    def test_unknown_target_is_refused(self):
        with pytest.raises(ValueError):
            render_config("vscode", self.command)


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def _run(command, home, pythonpath=True, **extra_env):
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=_env(home, pythonpath, **extra_env),
        cwd=str(ROOT),
        timeout=60,
        stdin=subprocess.DEVNULL,
    )


def _tree(root: Path):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def test_print_config_from_an_installed_script_names_that_script(tmp_path):
    script = _script(tmp_path, f"#!{sys.executable}")
    home = tmp_path / "home"
    home.mkdir()
    result = _run([str(script), "mcp", "--print-config", "claude", "--corpus", "DevRef"], home)
    assert result.returncode == 0, result.stderr
    server = json.loads(result.stdout)["mcpServers"]["arc"]
    assert server == {"command": os.path.realpath(script), "args": ["mcp", "--corpus", "DevRef"]}
    # Printing a config writes nothing: no migration, no config, no key.
    assert _tree(home) == CACHE_DIRS


def test_print_config_under_python_m_names_the_interpreter(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = _run(
        [sys.executable, "-m", "arcaneum.cli.main", "mcp", "--print-config", "codex"],
        home,
        ARC_MCP_CORPORA="DevRef,Notes",
    )
    assert result.returncode == 0, result.stderr
    server = tomllib.loads(result.stdout)["mcp_servers"]["arc"]
    assert server["command"] == sys.executable
    # The environment allowlist becomes flags: launchd passes no ARC_MCP_CORPORA.
    assert server["args"] == [
        "-m",
        "arcaneum.cli.main",
        "mcp",
        "--corpus",
        "DevRef",
        "--corpus",
        "Notes",
    ]
    assert _tree(home) == CACHE_DIRS


def test_the_printed_command_launches_under_launchd_path(tmp_path):
    script = _script(tmp_path, "#!/usr/bin/env python3")
    home = tmp_path / "home"
    home.mkdir()
    printed = _run(
        [sys.executable, str(script), "mcp", "--print-config", "claude", "--corpus", "DevRef"], home
    )
    assert printed.returncode == 0, printed.stderr
    server = json.loads(printed.stdout)["mcpServers"]["arc"]
    _assert_absolute_executable(server["command"])
    # The printed command, run with launchd's PATH, reaches the same arc.
    relaunched = _run([server["command"], *server["args"], "--print-config", "claude"], home)
    assert relaunched.returncode == 0, relaunched.stderr
    assert json.loads(relaunched.stdout)["mcpServers"]["arc"] == server


def test_the_checkout_launcher_relaunches_this_checkout_without_pythonpath(tmp_path):
    # bin/arc puts the checkout's src first on sys.path; the printed command
    # must keep that, not import whatever arcaneum the interpreter has installed.
    launcher = ROOT / "bin" / "arc"
    home = tmp_path / "home"
    home.mkdir()
    printed = _run(
        [sys.executable, str(launcher), "mcp", "--print-config", "claude", "--corpus", "DevRef"],
        home,
        pythonpath=False,
    )
    assert printed.returncode == 0, printed.stderr
    server = json.loads(printed.stdout)["mcpServers"]["arc"]
    assert server["command"] == sys.executable
    assert server["args"] == [os.path.realpath(launcher), "mcp", "--corpus", "DevRef"]
    relaunched = _run(
        [server["command"], *server["args"], "--print-config", "claude"], home, pythonpath=False
    )
    assert relaunched.returncode == 0, relaunched.stderr
    assert json.loads(relaunched.stdout)["mcpServers"]["arc"] == server


def test_unknown_print_config_target_is_refused(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = _run(
        [sys.executable, "-m", "arcaneum.cli.main", "mcp", "--print-config", "vscode"], home
    )
    assert result.returncode == 2
    assert "vscode" in result.stderr
    assert result.stdout == ""

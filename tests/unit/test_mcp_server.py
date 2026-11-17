"""The `arc mcp` stdio server and its list_corpora tool (RDR-026 Q1, Q2, Q3, Q10).

Review sessions launch the server fresh for every review, so these tests drive
it as a child process over pipes, the way a client does. `mcp_harness.py`
runs `arc` with fake backends selected by a scenario name.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from arcaneum import __version__
from arcaneum.cli.errors import InvalidArgumentError, UnknownCorpusError

ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(__file__).resolve().with_name("mcp_harness.py")
PROTOCOL_VERSION = "2025-06-18"


def _env(**extra):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    env.pop("ARC_MCP_CORPORA", None)
    env.update(extra)
    return env


class McpProcess:
    """A JSON-RPC client over a child's stdin and stdout pipes."""

    def __init__(self, scenario, *args, env=None):
        self.proc = subprocess.Popen(
            [sys.executable, str(HARNESS), scenario, "mcp", *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env or _env(),
            cwd=str(ROOT),
        )
        self.lines: list[bytes] = []
        self._queue: queue.Queue = queue.Queue()
        self._next_id = 0
        threading.Thread(target=self._read, daemon=True).start()
        self._stderr: list[bytes] = []
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line)
            self._queue.put(line)
        self._queue.put(None)

    def _read_stderr(self):
        for line in self.proc.stderr:
            self._stderr.append(line)

    @property
    def stderr(self) -> str:
        return b"".join(self._stderr).decode("utf-8", "replace")

    def send(self, message):
        self.proc.stdin.write((json.dumps(message) + "\n").encode())
        self.proc.stdin.flush()

    def request(self, method, params=None, timeout=20.0):
        self._next_id += 1
        request_id = self._next_id
        message = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"no response to {method}; stderr:\n{self.stderr}")
            try:
                line = self._queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                raise AssertionError(f"server exited during {method}; stderr:\n{self.stderr}")
            response = json.loads(line)
            if response.get("id") == request_id:
                return response

    def initialize(self):
        response = self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "arc-tests", "version": "0"},
            },
        )
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return response

    def call(self, name, arguments=None, timeout=20.0):
        response = self.request(
            "tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout
        )
        assert "result" in response, response
        return response["result"]

    def close(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


@pytest.fixture
def server(request):
    started = []

    def start(scenario="fake", *args, env=None):
        proc = McpProcess(scenario, *args, env=env)
        started.append(proc)
        return proc

    yield start
    for proc in started:
        proc.close()


def _cli_list_data(scenario="fake"):
    result = subprocess.run(
        [sys.executable, str(HARNESS), scenario, "corpus", "list", "--json"],
        capture_output=True,
        text=True,
        env=_env(),
        cwd=str(ROOT),
        timeout=60,
        check=True,
    )
    return json.loads(result.stdout)["data"]


def _error(result):
    assert result["isError"] is True, result
    structured = result["structuredContent"]
    assert set(structured) == {"code", "message"}
    assert result["content"] == [
        {"type": "text", "text": f"{structured['code']}: {structured['message']}"}
    ]
    return structured


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_initialize_reports_arcaneum_version(server):
    proc = server()
    response = proc.initialize()
    assert response["result"]["serverInfo"]["name"] == "arc"
    assert response["result"]["serverInfo"]["version"] == __version__


def test_list_corpora_is_the_only_tool_and_read_only(server):
    proc = server()
    proc.initialize()
    tools = proc.request("tools/list")["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["list_corpora"]
    assert tools[0]["annotations"]["readOnlyHint"] is True
    assert tools[0]["inputSchema"]["type"] == "object"


def test_stdout_carries_only_json_rpc_when_a_dependency_prints(server):
    proc = server("noisy")
    proc.initialize()
    result = proc.call("list_corpora")
    assert result.get("isError") in (None, False), result
    proc.close()

    assert proc.lines, "no protocol output"
    for line in proc.lines:
        message = json.loads(line)
        assert message["jsonrpc"] == "2.0"
    for chatter in ("print chatter", "raw fd chatter", "log chatter", "warning chatter"):
        assert chatter in proc.stderr
    assert "rich chatter" in proc.stderr


# ---------------------------------------------------------------------------
# list_corpora
# ---------------------------------------------------------------------------


def test_list_corpora_matches_cli_data(server):
    proc = server()
    proc.initialize()
    result = proc.call("list_corpora")

    expected = _cli_list_data()
    assert result["structuredContent"] == expected
    assert json.loads(result["content"][0]["text"]) == expected
    assert [corpus["name"] for corpus in expected["corpora"]] == ["Alpha", "Beta", "Gamma"]


def test_corpus_flag_limits_the_listing(server):
    proc = server("fake", "--corpus", "Gamma", "--corpus", "Alpha")
    proc.initialize()
    data = proc.call("list_corpora")["structuredContent"]

    expected = _cli_list_data()
    expected["corpora"] = [c for c in expected["corpora"] if c["name"] in {"Alpha", "Gamma"}]
    assert data == expected


def test_environment_allowlist_limits_the_listing(server):
    proc = server("fake", env=_env(ARC_MCP_CORPORA=" Beta , Gamma,"))
    proc.initialize()
    data = proc.call("list_corpora")["structuredContent"]
    assert [corpus["name"] for corpus in data["corpora"]] == ["Beta", "Gamma"]


def test_corpus_flag_wins_over_the_environment(server):
    proc = server("fake", "--corpus", "Alpha", env=_env(ARC_MCP_CORPORA="Beta,Gamma"))
    proc.initialize()
    data = proc.call("list_corpora")["structuredContent"]
    assert [corpus["name"] for corpus in data["corpora"]] == ["Alpha"]


def test_qdrant_down_is_service_down(server):
    proc = server("qdrant-down")
    proc.initialize()
    error = _error(proc.call("list_corpora"))
    assert error["code"] == "service-down"
    assert "Qdrant" in error["message"]
    assert "MeiliSearch" not in error["message"]


def test_meilisearch_down_is_service_down_naming_meilisearch(server):
    proc = server("meili-down")
    proc.initialize()
    error = _error(proc.call("list_corpora"))
    assert error["code"] == "service-down"
    assert "MeiliSearch" in error["message"]


def test_stalled_call_is_timeout(server):
    proc = server("stall", "--timeout", "1")
    proc.initialize()
    started = time.monotonic()
    error = _error(proc.call("list_corpora", timeout=15))
    assert error["code"] == "timeout"
    assert time.monotonic() - started < 10


def test_client_cancelled_stall_does_not_hold_the_exit(server):
    proc = server("stall", "--timeout", "120")
    proc.initialize()
    call = {"name": "list_corpora", "arguments": {}}
    proc.send({"jsonrpc": "2.0", "id": 99, "method": "tools/call", "params": call})
    time.sleep(1.0)  # the call reaches its worker thread and blocks there
    proc.send(
        {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 99}}
    )
    time.sleep(0.5)
    proc.proc.stdin.close()
    # The stalled read sleeps 60 s; the server must exit without waiting for it.
    assert proc.proc.wait(timeout=15) == 0


def test_output_cap_drops_whole_entries_and_says_so(server):
    proc = server("many", "--max-output-chars", "4000")
    proc.initialize()
    result = proc.call("list_corpora")
    data = result["structuredContent"]

    assert data["output_truncated"] is True
    assert data["omitted_corpora"] == 200 - len(data["corpora"])
    assert 0 < len(data["corpora"]) < 200
    assert len(result["content"][0]["text"]) <= 4000
    assert json.loads(result["content"][0]["text"]) == data


def test_unknown_argument_is_bad_argument(server):
    proc = server()
    proc.initialize()
    error = _error(proc.call("list_corpora", {"corpus": "Alpha"}))
    assert error["code"] == "bad-argument"
    assert "corpus" in error["message"]


def test_output_cap_bounds_an_error_result(server):
    proc = server("fake", "--max-output-chars", "4000")
    proc.initialize()
    error = _error(proc.call("list_corpora", {"x" * 5000: 1}))

    assert error["code"] == "bad-argument"
    assert len(json.dumps(error, ensure_ascii=False, separators=(",", ":"))) <= 4000
    assert error["message"].startswith("list_corpora got unexpected arguments: xxx")


def test_max_output_chars_below_the_floor_is_refused():
    result = subprocess.run(
        [sys.executable, str(HARNESS), "real", "mcp", "--max-output-chars", "3999"],
        capture_output=True,
        text=True,
        env=_env(),
        cwd=str(ROOT),
        timeout=60,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 2
    assert "3999" in result.stderr


# ---------------------------------------------------------------------------
# Start-up
# ---------------------------------------------------------------------------


def _tree(root: Path):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def test_start_runs_no_migration_and_writes_no_config(server, tmp_path):
    home = tmp_path / "home"
    legacy = home / ".arcaneum"
    (legacy / "data").mkdir(parents=True)
    (legacy / "data" / "store.json").write_text("{}")  # a migration would move it
    before = _tree(home)

    env = _env(
        HOME=str(home),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_CACHE_HOME=str(home / ".cache"),
        XDG_DATA_HOME=str(home / ".local" / "share"),
        XDG_STATE_HOME=str(home / ".local" / "state"),
    )
    proc = server("real", env=env)
    proc.initialize()
    proc.request("tools/list")
    proc.close()

    # Importing arcaneum creates the empty model cache directory (paths.py's
    # configure_model_cache_env, for every arc command). Nothing else may
    # appear: no migrated legacy data, no config, no state, no downloaded model.
    cache_dirs = [".cache", ".cache/arcaneum", ".cache/arcaneum/models"]
    assert _tree(home) == sorted(before + cache_dirs)
    assert (legacy / "data" / "store.json").read_text() == "{}"


def test_list_corpora_migrates_no_config_and_generates_no_key(server, tmp_path):
    home = tmp_path / "home"
    legacy = home / ".arcaneum"
    legacy.mkdir(parents=True)
    (legacy / "config.yaml").write_text("qdrant:\n  url: http://127.0.0.1:9\n")
    before = _tree(home)

    env = _env(
        HOME=str(home),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_CACHE_HOME=str(home / ".cache"),
        XDG_DATA_HOME=str(home / ".local" / "share"),
        XDG_STATE_HOME=str(home / ".local" / "state"),
        ARC_QDRANT_URL="http://127.0.0.1:9",
        MEILISEARCH_URL="http://127.0.0.1:9",
    )
    env.pop("MEILISEARCH_API_KEY", None)
    proc = server("real", env=env)
    proc.initialize()
    error = _error(proc.call("list_corpora"))
    proc.close()

    assert error["code"] == "service-down"
    # Neither the legacy config is migrated nor a MeiliSearch key generated.
    cache_dirs = [".cache", ".cache/arcaneum", ".cache/arcaneum/models"]
    assert _tree(home) == sorted(before + cache_dirs)


# ---------------------------------------------------------------------------
# Allowlist and cap helpers
# ---------------------------------------------------------------------------


class TestAllowlist:
    def test_flag_wins_over_environment(self):
        from arcaneum.mcp_server import resolve_allowlist

        assert resolve_allowlist(("A",), {"ARC_MCP_CORPORA": "B"}) == frozenset({"A"})

    def test_environment_is_comma_separated(self):
        from arcaneum.mcp_server import resolve_allowlist

        assert resolve_allowlist((), {"ARC_MCP_CORPORA": " A, B ,,"}) == frozenset({"A", "B"})

    def test_no_flag_and_no_environment_allows_every_corpus(self):
        from arcaneum.mcp_server import resolve_allowlist

        assert resolve_allowlist((), {}) is None
        assert resolve_allowlist((), {"ARC_MCP_CORPORA": " , "}) is None

    def test_a_name_outside_the_allowlist_is_unknown_corpus(self):
        from arcaneum.mcp_server import require_allowed

        require_allowed(frozenset({"A"}), ["A"])
        require_allowed(None, ["anything"])
        with pytest.raises(UnknownCorpusError) as exc_info:
            require_allowed(frozenset({"A"}), ["A", "B"])
        assert "B" in str(exc_info.value)


class TestListingCap:
    def _data(self, count):
        return {
            "corpora": [{"name": f"C{i}", "description": "x" * 100} for i in range(count)],
            "service_errors": [],
        }

    def test_a_listing_under_the_cap_is_unchanged(self):
        from arcaneum.mcp_server import cap_listing

        data = self._data(3)
        assert cap_listing(data, 4000) == data

    def test_a_listing_over_the_cap_drops_tail_entries(self):
        from arcaneum.mcp_server import cap_listing, serialize

        capped = cap_listing(self._data(100), 4000)
        assert len(serialize(capped)) <= 4000
        kept = len(capped["corpora"])
        assert capped["corpora"] == self._data(100)["corpora"][:kept]
        assert capped["omitted_corpora"] == 100 - kept
        assert capped["output_truncated"] is True

    def test_a_cap_no_entry_fits_is_bad_argument(self):
        from arcaneum.mcp_server import cap_listing

        with pytest.raises(InvalidArgumentError) as exc_info:
            cap_listing({"corpora": [], "service_errors": ["x" * 5000]}, 4000)
        assert "4000" in str(exc_info.value)

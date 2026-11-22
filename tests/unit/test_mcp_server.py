"""The `arc mcp` stdio server and its list_corpora, search_text, search_semantic and
get_passage tools.

RDR-026 Q1, Q2, Q3, Q8, Q9 and Q10.

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

import mcp_harness
import pytest

from arcaneum import __version__
from arcaneum.cli.errors import InvalidArgumentError, UnknownCorpusError

ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(__file__).resolve().with_name("mcp_harness.py")
PROTOCOL_VERSION = "2025-06-18"


def _env(**extra):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "ARC_INTERACTION_LOG": "0"}
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


def _cli_text_data(
    query, corpora, *, limit, max_content_chars, offset=0, filter=None, scenario="fake"
):
    args = ["search", "text", query, "--json", "--limit", str(limit), "--offset", str(offset)]
    args += ["--max-content-chars", str(max_content_chars)]
    for name in corpora:
        args += ["--corpus", name]
    if filter is not None:
        args += ["--filter", filter]
    result = subprocess.run(
        [sys.executable, str(HARNESS), scenario, *args],
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


def test_every_tool_is_read_only(server):
    proc = server()
    proc.initialize()
    tools = proc.request("tools/list")["result"]["tools"]
    assert [tool["name"] for tool in tools] == [
        "list_corpora",
        "search_text",
        "search_semantic",
        "get_passage",
    ]
    for tool in tools:
        assert tool["annotations"]["readOnlyHint"] is True
        assert tool["inputSchema"]["type"] == "object"


def test_search_text_schema_advertises_its_bounds(server):
    proc = server()
    proc.initialize()
    tools = {tool["name"]: tool for tool in proc.request("tools/list")["result"]["tools"]}
    schema = tools["search_text"]["inputSchema"]
    assert schema["required"] == ["query", "corpora"]
    assert schema["additionalProperties"] is False
    properties = schema["properties"]
    assert set(properties) == {"query", "corpora", "limit", "offset", "filter", "max_content_chars"}
    assert properties["limit"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 25,
        "default": 10,
        "description": properties["limit"]["description"],
    }
    assert properties["max_content_chars"]["default"] == 1200
    assert properties["corpora"]["minItems"] == 1


@pytest.mark.parametrize("tool", ["SEARCH_TEXT_TOOL", "SEARCH_SEMANTIC_TOOL"])
def test_filter_description_names_the_neutral_forms(tool):
    # The neutral forms parse_filter takes in both modes (RDR-027 Q6).
    import arcaneum.mcp_server as mcp_server

    description = getattr(mcp_server, tool)["input_schema"]["properties"]["filter"]["description"]
    for form in ("key=value", "key:op:value", ">=", "<=", "AND", "quote"):
        assert form in description, f"{tool} filter description omits {form!r}"


def test_filter_descriptions_name_each_backends_native_syntax():
    from arcaneum.mcp_server import SEARCH_SEMANTIC_TOOL, SEARCH_TEXT_TOOL

    text = SEARCH_TEXT_TOOL["input_schema"]["properties"]["filter"]["description"]
    semantic = SEARCH_SEMANTIC_TOOL["input_schema"]["properties"]["filter"]["description"]
    assert "MeiliSearch corpora" in text
    assert "Qdrant corpora" in text
    assert "must_not" in semantic


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
# search_text
# ---------------------------------------------------------------------------


def test_search_text_matches_cli_data(server):
    proc = server()
    proc.initialize()
    arguments = {
        "query": "lorem",
        "corpora": ["Alpha", "Beta"],
        "limit": 5,
        "offset": 1,
        "filter": "language=python",
        "max_content_chars": 300,
    }
    result = proc.call("search_text", arguments)

    expected = _cli_text_data(
        "lorem",
        ["Alpha", "Beta"],
        limit=5,
        offset=1,
        filter="language=python",
        max_content_chars=300,
    )
    assert result.get("isError") in (None, False), result
    assert result["structuredContent"] == expected
    assert json.loads(result["content"][0]["text"]) == expected
    hits = expected["hits"]
    assert len(hits) == 5
    assert {hit["corpus"] for hit in hits} == {"Alpha", "Beta"}
    for hit in hits:
        # The filter reached the backend, and the source roots shorten each cite.
        assert int(hit["chunk_id"].split("-")[1]) % 2 == 1
        assert hit["cite"] == f"{hit['corpus']}:{hit['locator']}"
        assert hit["locator"].startswith("doc")
        assert len(hit["snippet"]) <= 300 and hit["snippet_truncated"] is True


def test_search_text_defaults_to_ten_hits_of_1200_characters(server):
    proc = server()
    proc.initialize()
    data = proc.call("search_text", {"query": "lorem", "corpora": ["Gamma"]})["structuredContent"]

    assert data == _cli_text_data("lorem", ["Gamma"], limit=10, max_content_chars=1200)
    assert data["limit"] == 10 and data["offset"] == 0
    assert len(data["hits"]) == 10
    assert "output_truncated" not in data
    for hit in data["hits"]:
        assert len(hit["snippet"]) <= 1200 and hit["snippet_truncated"] is True


def test_search_text_refuses_a_limit_above_25(server):
    proc = server()
    proc.initialize()
    error = _error(proc.call("search_text", {"query": "lorem", "corpora": ["Alpha"], "limit": 26}))
    assert error["code"] == "bad-argument"
    assert "limit" in error["message"] and "25" in error["message"]

    data = proc.call("search_text", {"query": "lorem", "corpora": ["Alpha"], "limit": 25})[
        "structuredContent"
    ]
    assert len(data["hits"]) == 25


def test_search_text_corpus_outside_the_allowlist_is_unknown_corpus(server):
    proc = server("fake", "--corpus", "Alpha")
    proc.initialize()
    for corpora in (["Beta"], ["Alpha", "Beta"]):
        error = _error(proc.call("search_text", {"query": "lorem", "corpora": corpora}))
        assert error["code"] == "unknown-corpus"
        assert "Beta" in error["message"]


def test_search_text_no_existing_corpus_is_unknown_corpus(server):
    proc = server()
    proc.initialize()
    error = _error(proc.call("search_text", {"query": "lorem", "corpora": ["Nope"]}))
    assert error["code"] == "unknown-corpus"

    data = proc.call("search_text", {"query": "lorem", "corpora": ["Alpha", "Nope"]})[
        "structuredContent"
    ]
    assert data["missing_corpora"] == ["Nope"]
    assert data["hits"]


def test_search_text_output_cap_drops_whole_hits_and_says_so(server):
    proc = server("fake", "--max-output-chars", "4000")
    proc.initialize()
    arguments = {"query": "lorem", "corpora": ["Alpha", "Beta"], "limit": 25}
    result = proc.call("search_text", arguments)
    data = result["structuredContent"]

    full = _cli_text_data("lorem", ["Alpha", "Beta"], limit=25, max_content_chars=1200)
    kept = len(data["hits"])
    assert 0 < kept < 25
    assert data["hits"] == full["hits"][:kept]
    assert data["output_truncated"] is True
    assert data["omitted_hits"] == 25 - kept
    assert len(result["content"][0]["text"]) <= 4000
    assert json.loads(result["content"][0]["text"]) == data


def test_search_text_query_longer_than_the_cap_is_bad_argument(server):
    proc = server("fake", "--max-output-chars", "4000")
    proc.initialize()
    error = _error(proc.call("search_text", {"query": "q" * 5000, "corpora": ["Alpha"]}))
    assert error["code"] == "bad-argument"
    assert "4000" in error["message"]


def test_search_text_qdrant_json_filter_is_bad_filter(server):
    proc = server()
    proc.initialize()
    arguments = {"query": "lorem", "corpora": ["Alpha"], "filter": '{"must": []}'}
    error = _error(proc.call("search_text", arguments))
    assert error["code"] == "bad-filter"


def test_search_text_meilisearch_down_is_service_down(server):
    proc = server("meili-down")
    proc.initialize()
    error = _error(proc.call("search_text", {"query": "lorem", "corpora": ["Alpha"]}))
    assert error["code"] == "service-down"
    assert "MeiliSearch" in error["message"]


def test_search_text_qdrant_down_names_qdrant_like_the_cli(server):
    # MeiliSearch still answers, but a migrated corpus's index may be stale (arcaneum#178).
    proc = server("qdrant-down")
    proc.initialize()
    result = proc.call("search_text", {"query": "lorem", "corpora": ["Alpha"], "limit": 2})

    assert result.get("isError") in (None, False), result
    data = result["structuredContent"]
    assert [entry.split(": ", 1)[0] for entry in data["service_errors"]] == ["Qdrant"]
    assert len(data["hits"]) == 2
    assert data == _cli_text_data(
        "lorem", ["Alpha"], limit=2, max_content_chars=1200, scenario="qdrant-down"
    )


def test_search_text_stall_is_timeout(server):
    proc = server("stall", "--timeout", "1")
    proc.initialize()
    error = _error(proc.call("search_text", {"query": "lorem", "corpora": ["Alpha"]}, timeout=15))
    assert error["code"] == "timeout"


# ---------------------------------------------------------------------------
# search_semantic
# ---------------------------------------------------------------------------


def _semantic_env(tmp_path, **extra):
    """An environment whose embedder slots live under ``tmp_path``, not the user's."""
    return _env(XDG_DATA_HOME=str(tmp_path / "data"), **extra)


def _cli_semantic(args, env, scenario="fake"):
    result = subprocess.run(
        [sys.executable, str(HARNESS), scenario, "search", "semantic", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
        timeout=60,
    )
    return result


def _cli_semantic_data(query, corpora, env, *, limit, max_content_chars, offset=0, **options):
    args = [query, "--json", "--limit", str(limit), "--offset", str(offset)]
    args += ["--max-content-chars", str(max_content_chars)]
    for name in corpora:
        args += ["--corpus", name]
    if options.get("filter") is not None:
        args += ["--filter", options["filter"]]
    if options.get("score_threshold") is not None:
        args += ["--score-threshold", str(options["score_threshold"])]
    result = _cli_semantic(args, env)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)["data"]


def _stderr_once_exited(proc, needle=None):
    """The server's stderr after it exits, so every line it wrote is read."""
    proc.close()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and needle is not None and needle not in proc.stderr:
        time.sleep(0.05)
    return proc.stderr


@pytest.fixture
def held_slots(tmp_path):
    """Hold every embedder slot (ARCANEUM_SEARCH_CONCURRENCY=1) from this process."""
    import fcntl

    locks = tmp_path / "data" / "arcaneum" / "locks"
    locks.mkdir(parents=True)
    fd = os.open(locks / "embedder-slot-0.lock", os.O_CREAT | os.O_WRONLY, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    yield
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def test_search_semantic_schema_advertises_its_bounds(server):
    proc = server()
    proc.initialize()
    tools = {tool["name"]: tool for tool in proc.request("tools/list")["result"]["tools"]}
    schema = tools["search_semantic"]["inputSchema"]
    assert schema["required"] == ["query", "corpora"]
    assert schema["additionalProperties"] is False
    properties = schema["properties"]
    assert set(properties) == {
        "query",
        "corpora",
        "limit",
        "offset",
        "filter",
        "max_content_chars",
        "score_threshold",
    }
    assert properties["limit"]["maximum"] == 25 and properties["limit"]["default"] == 10
    assert properties["max_content_chars"]["default"] == 1200
    assert properties["score_threshold"]["type"] == "number"


def test_search_semantic_matches_cli_data(server, tmp_path):
    env = _semantic_env(tmp_path)
    proc = server(env=env)
    proc.initialize()
    arguments = {
        "query": "lorem",
        "corpora": ["Alpha", "Beta"],
        "limit": 5,
        "offset": 1,
        "filter": "language=python",
        "max_content_chars": 300,
        "score_threshold": 0.5,
    }
    result = proc.call("search_semantic", arguments)

    expected = _cli_semantic_data(
        "lorem",
        ["Alpha", "Beta"],
        env,
        limit=5,
        offset=1,
        filter="language=python",
        max_content_chars=300,
        score_threshold=0.5,
    )
    assert result.get("isError") in (None, False), result
    assert result["structuredContent"] == expected
    assert json.loads(result["content"][0]["text"]) == expected
    hits = expected["hits"]
    assert len(hits) == 5
    assert {hit["corpus"] for hit in hits} == {"Alpha", "Beta"}
    assert [hit["score"] for hit in hits] == sorted((hit["score"] for hit in hits), reverse=True)
    for hit in hits:
        assert int(hit["chunk_id"].split("-")[1]) % 2 == 1
        assert hit["score"] >= 0.5
        assert hit["cite"] == f"{hit['corpus']}:{hit['locator']}"
        assert hit["locator"].startswith("doc")
        assert len(hit["snippet"]) <= 300 and hit["snippet_truncated"] is True


def test_search_semantic_score_threshold_drops_weak_hits(server, tmp_path):
    proc = server(env=_semantic_env(tmp_path))
    proc.initialize()
    arguments = {"query": "lorem", "corpora": ["Alpha"], "limit": 25}
    every = proc.call("search_semantic", arguments)["structuredContent"]["hits"]
    strong = proc.call("search_semantic", {**arguments, "score_threshold": 0.8})[
        "structuredContent"
    ]["hits"]

    assert len(every) == 25
    assert strong == [hit for hit in every if hit["score"] >= 0.8]
    assert 0 < len(strong) < len(every)


def test_search_semantic_keeps_its_embedder_warm_across_calls(server, tmp_path):
    proc = server(env=_semantic_env(tmp_path))
    proc.initialize()
    for _ in range(3):
        result = proc.call("search_semantic", {"query": "lorem", "corpora": ["Alpha"]})
        assert result.get("isError") in (None, False), result

    stderr = _stderr_once_exited(proc, "fake model load")
    assert stderr.count("fake model load: arctic-m") == 1


def test_search_semantic_writes_no_prompt_policy_stamp(server, tmp_path):
    env = _semantic_env(tmp_path)
    proc = server(env=env)
    proc.initialize()
    result = proc.call("search_semantic", {"query": "lorem", "corpora": ["Alpha"]})
    assert result.get("isError") in (None, False), result
    assert result["structuredContent"]["hits"]

    assert "fake qdrant write" not in _stderr_once_exited(proc)
    # The CLI stamps the same legacy collection, so the fake does see writes.
    cli = _cli_semantic(["lorem", "--json", "--corpus", "Alpha"], env)
    assert cli.returncode == 0, cli.stderr
    assert "fake qdrant write: upsert Alpha" in cli.stderr


def test_search_semantic_uncached_model_is_semantic_unavailable(server, tmp_path):
    env = _semantic_env(tmp_path)
    proc = server("uncached", env=env)
    proc.initialize()
    error = _error(proc.call("search_semantic", {"query": "lorem", "corpora": ["Alpha"]}))
    assert error["code"] == "semantic-unavailable"
    assert "arctic-m" in error["message"]
    assert "fake model" not in _stderr_once_exited(proc)

    offline = _cli_semantic(["lorem", "--json", "--offline", "--corpus", "Alpha"], env, "uncached")
    assert offline.returncode == 1
    envelope = json.loads(offline.stdout)
    assert envelope["status"] == "error"
    assert envelope["errors"][0]["code"] == "semantic-unavailable"
    assert "fake model download" not in offline.stderr

    # The CLI's default keeps download-on-first-use.
    online = _cli_semantic(["lorem", "--json", "--corpus", "Alpha"], env, "uncached")
    assert online.returncode == 0, online.stderr
    assert "fake model download: arctic-m" in online.stderr


def test_search_semantic_busy_slot_is_semantic_unavailable(server, tmp_path, held_slots):
    env = _semantic_env(tmp_path, ARCANEUM_SEARCH_CONCURRENCY="1")
    proc = server("fake", "--timeout", "2", env=env)
    proc.initialize()
    started = time.monotonic()
    error = _error(proc.call("search_semantic", {"query": "lorem", "corpora": ["Alpha"]}))
    assert error["code"] == "semantic-unavailable"
    assert "slot" in error["message"]
    assert time.monotonic() - started < 10

    cli = _cli_semantic(
        ["lorem", "--json", "--corpus", "Alpha"],
        {**env, "ARCANEUM_SEARCH_WAIT_SECONDS": "0.5"},
    )
    assert cli.returncode == 1
    assert json.loads(cli.stdout)["errors"][0]["code"] == "semantic-unavailable"


def test_search_semantic_stalled_embed_is_timeout(server, tmp_path):
    proc = server("embed-stall", "--timeout", "1", env=_semantic_env(tmp_path))
    proc.initialize()
    error = _error(
        proc.call("search_semantic", {"query": "lorem", "corpora": ["Alpha"]}, timeout=15)
    )
    assert error["code"] == "timeout"


def test_search_semantic_corpus_outside_the_allowlist_is_unknown_corpus(server, tmp_path):
    proc = server("fake", "--corpus", "Alpha", env=_semantic_env(tmp_path))
    proc.initialize()
    error = _error(proc.call("search_semantic", {"query": "lorem", "corpora": ["Alpha", "Beta"]}))
    assert error["code"] == "unknown-corpus"
    assert "Beta" in error["message"]


def test_search_semantic_output_cap_drops_whole_hits_and_says_so(server, tmp_path):
    proc = server("fake", "--max-output-chars", "4000", env=_semantic_env(tmp_path))
    proc.initialize()
    result = proc.call("search_semantic", {"query": "lorem", "corpora": ["Alpha"], "limit": 25})
    data = result["structuredContent"]

    assert 0 < len(data["hits"]) < 25
    assert data["output_truncated"] is True
    assert data["omitted_hits"] == 25 - len(data["hits"])
    assert len(result["content"][0]["text"]) <= 4000


# ---------------------------------------------------------------------------
# get_passage
# ---------------------------------------------------------------------------


def _cli_passage(chunk_id, corpus, *, scenario="fake", **options):
    args = ["search", "passage", chunk_id, "--corpus", corpus, "--json"]
    for name in ("before", "after", "max_chars"):
        if options.get(name) is not None:
            args += [f"--{name.replace('_', '-')}", str(options[name])]
    return subprocess.run(
        [sys.executable, str(mcp_harness.__file__), scenario, *args],
        capture_output=True,
        text=True,
        env=_env(),
        cwd=str(ROOT),
        timeout=60,
    )


def _cli_passage_data(chunk_id, corpus, **options):
    result = _cli_passage(chunk_id, corpus, **options)
    assert result.returncode == 0, result.stderr
    envelope = json.loads(result.stdout)
    assert envelope["status"] == "success"
    return envelope["data"]


def _book_hit(hits, i, corpus="Alpha"):
    """The chunk id of book chunk ``i`` among search hits."""
    ids = [hit["chunk_id"] for hit in hits]
    chunk_id = mcp_harness.book_chunk_id(corpus, i)
    assert chunk_id in ids, ids
    return chunk_id


def test_get_passage_schema_advertises_its_bounds(server):
    proc = server()
    proc.initialize()
    tools = {tool["name"]: tool for tool in proc.request("tools/list")["result"]["tools"]}
    schema = tools["get_passage"]["inputSchema"]
    assert schema["required"] == ["corpus", "chunk_id"]
    assert schema["additionalProperties"] is False
    properties = schema["properties"]
    assert set(properties) == {"corpus", "chunk_id", "before", "after", "max_chars"}
    for name in ("before", "after"):
        assert properties[name]["minimum"] == 0
        assert properties[name]["maximum"] == 5
        assert properties[name]["default"] == 1
    assert properties["max_chars"]["minimum"] == 0
    assert properties["max_chars"]["default"] == 8000


def test_get_passage_from_a_text_hit_matches_cli_data(server):
    proc = server()
    proc.initialize()
    hits = proc.call(
        "search_text",
        {
            "query": "w0300",
            "corpora": ["Alpha"],
            "filter": f"file_path={mcp_harness.book_path('Alpha')}",
        },
    )["structuredContent"]["hits"]
    chunk_id = _book_hit(hits, 2)

    arguments = {"corpus": "Alpha", "chunk_id": chunk_id, "before": 1, "after": 2, "max_chars": 0}
    result = proc.call("get_passage", arguments)
    assert result.get("isError") in (None, False), result

    expected = _cli_passage_data(chunk_id, "Alpha", before=1, after=2, max_chars=0)
    assert result["structuredContent"] == expected
    assert json.loads(result["content"][0]["text"]) == expected
    assert expected["chunks"] == [
        {"chunk_id": mcp_harness.book_chunk_id("Alpha", i), "chunk_index": i} for i in (1, 2, 3, 4)
    ]
    assert expected["first_chunk_id"] == mcp_harness.book_chunk_id("Alpha", 1)
    assert expected["last_chunk_id"] == mcp_harness.book_chunk_id("Alpha", 4)
    assert expected["text"] == mcp_harness.book_text(1, 4)
    assert expected["truncated"] is False
    assert expected["cite"] == "Alpha:book.md#book"


def test_get_passage_from_a_semantic_hit_equals_the_text_hit_window(server, tmp_path):
    proc = server(env=_semantic_env(tmp_path))
    proc.initialize()
    book = f"file_path={mcp_harness.book_path('Alpha')}"
    semantic = proc.call(
        "search_semantic", {"query": "w0300", "corpora": ["Alpha"], "filter": book}
    )
    text = proc.call("search_text", {"query": "w0300", "corpora": ["Alpha"], "filter": book})
    from_semantic = _book_hit(semantic["structuredContent"]["hits"], 2)
    from_text = _book_hit(text["structuredContent"]["hits"], 2)

    windows = [
        proc.call("get_passage", {"corpus": "Alpha", "chunk_id": chunk_id})["structuredContent"]
        for chunk_id in (from_semantic, from_text)
    ]
    assert windows[0] == windows[1]
    assert windows[0]["text"] == mcp_harness.book_text(1, 3)
    assert "fake qdrant write" not in _stderr_once_exited(proc)


def test_get_passage_defaults_match_the_cli_at_8000_characters(server):
    proc = server()
    proc.initialize()
    chunk_id = mcp_harness.book_chunk_id("Alpha", 0)
    data = proc.call("get_passage", {"corpus": "Alpha", "chunk_id": chunk_id})["structuredContent"]
    assert data == _cli_passage_data(chunk_id, "Alpha", max_chars=8000)
    assert [chunk["chunk_index"] for chunk in data["chunks"]] == [0, 1]

    capped = proc.call("get_passage", {"corpus": "Alpha", "chunk_id": chunk_id, "max_chars": 100})
    assert capped["structuredContent"]["text"] == mcp_harness.book_text(0, 1)[:100]
    assert capped["structuredContent"]["truncated"] is True


def test_get_passage_unknown_chunk_is_unknown_chunk(server):
    proc = server()
    proc.initialize()
    unknown = mcp_harness.book_chunk_id("Alpha", 99)
    error = _error(proc.call("get_passage", {"corpus": "Alpha", "chunk_id": unknown}))
    assert error["code"] == "unknown-chunk"
    assert unknown in error["message"]

    cli = _cli_passage(unknown, "Alpha")
    assert cli.returncode == 3
    envelope = json.loads(cli.stdout)
    assert envelope["status"] == "error"
    assert envelope["errors"][0]["code"] == "unknown-chunk"


def test_get_passage_qdrant_down_takes_the_window_from_meilisearch(server):
    proc = server("qdrant-down")
    proc.initialize()
    chunk_id = mcp_harness.book_chunk_id("Alpha", 2)
    result = proc.call("get_passage", {"corpus": "Alpha", "chunk_id": chunk_id})
    assert result.get("isError") in (None, False), result
    data = result["structuredContent"]
    assert data["text"] == mcp_harness.book_text(1, 3)
    # No Qdrant, so no recorded root: the cite falls back to the absolute path.
    assert data["cite"] == "Alpha:/src/Alpha/book.md#book"
    assert data == _cli_passage_data(chunk_id, "Alpha", scenario="qdrant-down", max_chars=8000)


def test_get_passage_corpus_outside_the_allowlist_is_unknown_corpus(server):
    proc = server("fake", "--corpus", "Alpha")
    proc.initialize()
    chunk_id = mcp_harness.book_chunk_id("Beta", 2)
    error = _error(proc.call("get_passage", {"corpus": "Beta", "chunk_id": chunk_id}))
    assert error["code"] == "unknown-corpus"
    assert "Beta" in error["message"]


def test_get_passage_output_cap_shortens_the_text_and_says_so(server):
    proc = server("fake", "--max-output-chars", "4000")
    proc.initialize()
    chunk_id = mcp_harness.book_chunk_id("Alpha", 2)
    arguments = {"corpus": "Alpha", "chunk_id": chunk_id, "before": 5, "after": 5, "max_chars": 0}
    result = proc.call("get_passage", arguments)
    data = result["structuredContent"]

    full = mcp_harness.book_text(0, mcp_harness.BOOK_CHUNKS - 1)
    assert len(full) > 4000
    assert 0 < len(data["text"]) < len(full)
    assert full.startswith(data["text"])
    assert data["truncated"] is True
    assert len(data["chunks"]) == mcp_harness.BOOK_CHUNKS
    assert len(result["content"][0]["text"]) <= 4000
    assert json.loads(result["content"][0]["text"]) == data


def test_get_passage_writes_nothing(server):
    proc = server()
    proc.initialize()
    chunk_id = mcp_harness.book_chunk_id("Alpha", 2)
    result = proc.call("get_passage", {"corpus": "Alpha", "chunk_id": chunk_id, "before": 5})
    assert result.get("isError") in (None, False), result
    stderr = _stderr_once_exited(proc)
    assert "fake qdrant write" not in stderr and "fake meili write" not in stderr


def test_cli_passage_migrates_no_config_and_generates_no_key(tmp_path):
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
    chunk_id = mcp_harness.book_chunk_id("Alpha", 2)
    result = subprocess.run(
        [sys.executable, str(HARNESS), "real", "search", "passage", chunk_id, "--corpus", "Alpha"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
        timeout=60,
    )
    assert result.returncode == 1, result.stderr
    # Neither the legacy config is migrated nor a MeiliSearch key generated.
    cache_dirs = [".cache", ".cache/arcaneum", ".cache/arcaneum/models"]
    assert _tree(home) == sorted(before + cache_dirs)


def test_get_passage_hands_the_core_its_8000_character_default(monkeypatch):
    import arcaneum.mcp_server as mcp_server
    import arcaneum.retrieval as retrieval
    from arcaneum.retrieval.passage import PassageChunk, PassageResult

    calls = []

    def get_passage(corpus, chunk_id, **options):
        calls.append(options)
        chunk = PassageChunk(chunk_id=chunk_id, chunk_index=0, text="x", payload={})
        return PassageResult(corpus, chunk_id, f"{corpus}:doc.md", [chunk], "x", False, "qdrant")

    monkeypatch.setattr(retrieval, "get_passage", get_passage)
    monkeypatch.setattr(mcp_server, "_qdrant_search_client", lambda: object())
    monkeypatch.setattr(mcp_server, "_meili_client", lambda: object())
    tools = mcp_server.ArcTools(allowlist=frozenset({"A"}))

    tools.get_passage({"corpus": "A", "chunk_id": "c"})
    tools.get_passage({"corpus": "A", "chunk_id": "c", "max_chars": 0})
    assert [call["max_chars"] for call in calls] == [8000, 0]
    assert [(call["before"], call["after"]) for call in calls] == [(1, 1), (1, 1)]


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


@pytest.mark.parametrize(
    "tool, arguments",
    [("list_corpora", {}), ("search_text", {"query": "lorem", "corpora": ["Alpha"]})],
)
def test_a_call_migrates_no_config_and_generates_no_key(server, tmp_path, tool, arguments):
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
    error = _error(proc.call(tool, arguments))
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


class TestSearchTextArguments:
    """Argument checks run before any client or core call."""

    @pytest.fixture
    def tools(self, monkeypatch):
        import arcaneum.mcp_server as mcp_server

        def refuse(*_args, **_kwargs):
            raise AssertionError("a refused call reached a client")

        monkeypatch.setattr(mcp_server, "_meili_client", refuse)
        monkeypatch.setattr(mcp_server, "_source_roots_client", refuse)
        return mcp_server.ArcTools(allowlist=frozenset({"A"}))

    @pytest.mark.parametrize(
        "arguments, field",
        [
            ({"corpora": ["A"]}, "query"),
            ({"query": 3, "corpora": ["A"]}, "query"),
            ({"query": "q"}, "corpora"),
            ({"query": "q", "corpora": []}, "corpora"),
            ({"query": "q", "corpora": "A"}, "corpora"),
            ({"query": "q", "corpora": ["A", ""]}, "corpora"),
            ({"query": "q", "corpora": ["A"], "limit": 26}, "limit"),
            ({"query": "q", "corpora": ["A"], "limit": 0}, "limit"),
            ({"query": "q", "corpora": ["A"], "limit": "5"}, "limit"),
            ({"query": "q", "corpora": ["A"], "limit": True}, "limit"),
            ({"query": "q", "corpora": ["A"], "limit": 2.5}, "limit"),
            ({"query": "q", "corpora": ["A"], "offset": -1}, "offset"),
            ({"query": "q", "corpora": ["A"], "max_content_chars": -1}, "max_content_chars"),
            ({"query": "q", "corpora": ["A"], "filter": 1}, "filter"),
            ({"query": "q", "corpora": ["A"], "verbose": True}, "verbose"),
        ],
    )
    def test_a_bad_argument_names_its_field(self, tools, arguments, field):
        with pytest.raises(InvalidArgumentError) as exc_info:
            tools.search_text(arguments)
        assert exc_info.value.code == "bad-argument"
        assert field in str(exc_info.value)

    def test_a_corpus_outside_the_allowlist_is_refused_before_any_client(self, tools):
        with pytest.raises(UnknownCorpusError) as exc_info:
            tools.search_text({"query": "q", "corpora": ["A", "B"]})
        assert "B" in str(exc_info.value)


class TestSearchSemanticArguments:
    """Argument checks run before any client, embedder or core call."""

    @pytest.fixture
    def tools(self, monkeypatch):
        import arcaneum.mcp_server as mcp_server

        def refuse(*_args, **_kwargs):
            raise AssertionError("a refused call reached a client")

        monkeypatch.setattr(mcp_server, "_qdrant_search_client", refuse)
        monkeypatch.setattr(mcp_server.ArcTools, "_semantic_session", property(refuse))
        return mcp_server.ArcTools(allowlist=frozenset({"A"}))

    @pytest.mark.parametrize(
        "arguments, field",
        [
            ({"corpora": ["A"]}, "query"),
            ({"query": "q", "corpora": []}, "corpora"),
            ({"query": "q", "corpora": ["A"], "limit": 26}, "limit"),
            ({"query": "q", "corpora": ["A"], "offset": -1}, "offset"),
            ({"query": "q", "corpora": ["A"], "max_content_chars": -1}, "max_content_chars"),
            ({"query": "q", "corpora": ["A"], "filter": 1}, "filter"),
            ({"query": "q", "corpora": ["A"], "score_threshold": "0.5"}, "score_threshold"),
            ({"query": "q", "corpora": ["A"], "score_threshold": True}, "score_threshold"),
            ({"query": "q", "corpora": ["A"], "vector_name": "x"}, "vector_name"),
        ],
    )
    def test_a_bad_argument_names_its_field(self, tools, arguments, field):
        with pytest.raises(InvalidArgumentError) as exc_info:
            tools.search_semantic(arguments)
        assert exc_info.value.code == "bad-argument"
        assert field in str(exc_info.value)

    def test_a_corpus_outside_the_allowlist_is_refused_before_any_client(self, tools):
        with pytest.raises(UnknownCorpusError) as exc_info:
            tools.search_semantic({"query": "q", "corpora": ["A", "B"]})
        assert "B" in str(exc_info.value)


class TestGetPassageArguments:
    """Argument checks run before any client or core call."""

    @pytest.fixture
    def tools(self, monkeypatch):
        import arcaneum.mcp_server as mcp_server

        def refuse(*_args, **_kwargs):
            raise AssertionError("a refused call reached a client")

        monkeypatch.setattr(mcp_server, "_meili_client", refuse)
        monkeypatch.setattr(mcp_server, "_qdrant_search_client", refuse)
        return mcp_server.ArcTools(allowlist=frozenset({"A"}))

    @pytest.mark.parametrize(
        "arguments, field",
        [
            ({"chunk_id": "c"}, "corpus"),
            ({"corpus": "", "chunk_id": "c"}, "corpus"),
            ({"corpus": 3, "chunk_id": "c"}, "corpus"),
            ({"corpus": "A"}, "chunk_id"),
            ({"corpus": "A", "chunk_id": ""}, "chunk_id"),
            ({"corpus": "A", "chunk_id": 7}, "chunk_id"),
            ({"corpus": "A", "chunk_id": "c", "before": 6}, "before"),
            ({"corpus": "A", "chunk_id": "c", "before": -1}, "before"),
            ({"corpus": "A", "chunk_id": "c", "after": 6}, "after"),
            ({"corpus": "A", "chunk_id": "c", "after": True}, "after"),
            ({"corpus": "A", "chunk_id": "c", "max_chars": -1}, "max_chars"),
            ({"corpus": "A", "chunk_id": "c", "corpora": ["A"]}, "corpora"),
        ],
    )
    def test_a_bad_argument_names_its_field(self, tools, arguments, field):
        with pytest.raises(InvalidArgumentError) as exc_info:
            tools.get_passage(arguments)
        assert exc_info.value.code == "bad-argument"
        assert field in str(exc_info.value)

    def test_a_corpus_outside_the_allowlist_is_refused_before_any_client(self, tools):
        with pytest.raises(UnknownCorpusError) as exc_info:
            tools.get_passage({"corpus": "B", "chunk_id": "c"})
        assert "B" in str(exc_info.value)


class TestPassageCap:
    def _data(self, text):
        return {
            "schema_version": 1,
            "corpus": "C",
            "chunk_id": "c2",
            "cite": "C:doc.md",
            "first_chunk_id": "c1",
            "last_chunk_id": "c3",
            "chunks": [{"chunk_id": f"c{i}", "chunk_index": i} for i in (1, 2, 3)],
            "text": text,
            "truncated": False,
        }

    def test_a_passage_under_the_cap_is_unchanged(self):
        from arcaneum.mcp_server import cap_passage

        data = self._data("x" * 100)
        assert cap_passage(data, 4000) == data

    def test_a_passage_over_the_cap_shortens_its_text(self):
        from arcaneum.mcp_server import cap_passage, serialize

        text = '\u00e9"' * 3000  # characters JSON escapes, so length is not linear
        capped = cap_passage(self._data(text), 4000)
        assert len(serialize(capped)) <= 4000
        assert text.startswith(capped["text"]) and capped["text"]
        assert capped["truncated"] is True
        # The longest prefix that fits: one more character would not.
        longer = {**capped, "text": text[: len(capped["text"]) + 1]}
        assert len(serialize(longer)) > 4000
        assert capped["chunks"] == self._data(text)["chunks"]

    def test_a_cap_no_text_fits_is_bad_argument(self):
        from arcaneum.mcp_server import cap_passage

        data = {**self._data("text"), "cite": "C:" + "x" * 5000}
        with pytest.raises(InvalidArgumentError) as exc_info:
            cap_passage(data, 4000)
        assert "4000" in str(exc_info.value)


class TestSearchCap:
    def _data(self, count):
        return {
            "schema_version": 1,
            "query": "q",
            "corpora": ["C"],
            "missing_corpora": [],
            "limit": count,
            "offset": 0,
            "hits": [{"cite": f"C:doc{i}.md", "snippet": "x" * 300} for i in range(count)],
        }

    def test_a_result_under_the_cap_is_unchanged(self):
        from arcaneum.mcp_server import cap_search

        data = self._data(3)
        assert cap_search(data, 4000) == data

    def test_a_result_over_the_cap_drops_tail_hits(self):
        from arcaneum.mcp_server import cap_search, serialize

        capped = cap_search(self._data(25), 4000)
        assert len(serialize(capped)) <= 4000
        kept = len(capped["hits"])
        assert 0 < kept < 25
        assert capped["hits"] == self._data(25)["hits"][:kept]
        assert capped["omitted_hits"] == 25 - kept
        assert capped["output_truncated"] is True

    def test_a_cap_no_hit_fits_is_bad_argument(self):
        from arcaneum.mcp_server import cap_search

        data = {**self._data(1), "query": "q" * 5000}
        with pytest.raises(InvalidArgumentError) as exc_info:
            cap_search(data, 4000)
        assert "4000" in str(exc_info.value)

#!/usr/bin/env python3
"""Live probe: `arc mcp` under `claude -p` and `codex exec` against a corpus (RDR-026).

Live, so never part of the service-free suite: it needs Qdrant and MeiliSearch
up, the corpus indexed, and `claude` and `codex` on PATH and signed in. Every
call it makes is read-only. Its session-log parsers are unit-tested over
fixture lines (tests/unit/test_probe_mcp_live.py).

usage:
    scripts/probe-mcp-live.py [--agent claude|codex]... [--corpus DevRef]
                              [--query TEXT] [--out-dir DIR] [--timeout SECONDS]
    scripts/probe-mcp-live.py --parse claude|codex LOG [--corpus DevRef]

1. **Configs.** `bin/arc mcp --corpus <corpus> --print-config claude|codex`
   under this interpreter. `bin/arc` puts this checkout's `src` first, and the
   printed command names `bin/arc` by absolute path, so every session below
   runs this checkout's arc, not an installed one.
2. **Direct.** One stdio session on the printed command: `initialize` records
   `serverInfo.version`; `tools/list` must list exactly the four tools, each
   `readOnlyHint: true`; a `search_text` call for the query must return hits
   (the services are up and the corpus answers). A failure here stops the
   probe before any model run.
3. **Sessions.** Each `--agent` (default both) gets one prompt that needs the
   corpus: `claude -p --strict-mcp-config --mcp-config <claude config>
   --allowedTools mcp__arc__search_text --output-format stream-json`, and
   `codex exec --json --sandbox read-only --ignore-user-config` with the codex
   config's `-c mcp_servers.arc.*` overrides. A session passes when its log
   holds a `search_text` call on `arc` whose result carries hits from the
   corpus, each with a cite, the session did not end in failure (an error
   `result` from claude, `turn.failed` from codex), and the client exited 0
   within `--timeout`.

Each session's raw log is kept under `--out-dir` (`claude.jsonl`,
`codex.jsonl`, with stderr beside it). `--parse` re-judges a kept log.

stdout: a JSON report (server version, tools, session ids, outcomes).
exit: 0 every check passed; 1 a check failed; 2 usage.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

REPO = Path(__file__).resolve().parents[1]
ARC = REPO / "bin" / "arc"

SERVER = "arc"
TOOL = "search_text"
CLAUDE_TOOL = f"mcp__{SERVER}__{TOOL}"
EXPECTED_TOOLS = frozenset({"list_corpora", "search_text", "search_semantic", "get_passage"})
PROTOCOL_VERSION = "2025-06-18"

PROMPT = (
    "Use the arc MCP server's search_text tool exactly once, with query {query} and "
    'corpora ["{corpus}"]. Then reply with only the cite of the first hit.'
)

# Set by a running Claude Code session; a child claude that inherits them acts
# as a nested session of the caller's, so the probe starts a fresh one.
CLAUDE_SESSION_ENV = (
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_PID",
    "CLAUDE_EFFORT",
)


class ProbeFailure(Exception):
    """A check the probe makes did not hold."""


# ---------------------------------------------------------------------------
# Checks (pure)
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    """What one session's log shows."""

    agent: str
    session_id: str | None = None
    passed: bool = False
    reason: str = ""
    calls: int = 0
    cites: list[str] = field(default_factory=list)
    answer: str | None = None
    log: str | None = None
    exit_code: int | None = None

    def to_data(self) -> dict[str, Any]:
        return asdict(self)


def parse_jsonl(lines: Iterable[str]) -> list[dict[str, Any]]:
    """The JSON objects among ``lines``; blank lines and anything else are skipped."""
    events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def check_tools_list(result: dict[str, Any]) -> list[str]:
    """The tool names of a ``tools/list`` result: exactly the four, each read-only."""
    tools = [tool for tool in result.get("tools") or [] if isinstance(tool, dict)]
    names = [tool.get("name") for tool in tools]
    extra = sorted(str(name) for name in set(names) - EXPECTED_TOOLS)
    missing = sorted(EXPECTED_TOOLS - set(names))
    if extra or missing or len(names) != len(set(names)):
        raise ProbeFailure(
            f"tools/list: expected exactly {sorted(EXPECTED_TOOLS)}; got {names} "
            f"(extra {extra}, missing {missing})"
        )
    not_read_only = sorted(
        tool["name"]
        for tool in tools
        if (tool.get("annotations") or {}).get("readOnlyHint") is not True
    )
    if not_read_only:
        raise ProbeFailure(f"tools/list: readOnlyHint is not true on {not_read_only}")
    return sorted(names)


def search_hits(data: Any, corpus: str) -> list[str]:
    """The cites of a ``search_text`` result's hits: at least one, all from ``corpus``."""
    if not isinstance(data, dict) or not isinstance(data.get("hits"), list):
        raise ProbeFailure(f"{TOOL} result has no hits list")
    hits = data["hits"]
    if not hits:
        raise ProbeFailure(f"{TOOL} returned no hits")
    cites = []
    for hit in hits:
        hit = hit if isinstance(hit, dict) else {}
        if hit.get("corpus") != corpus:
            raise ProbeFailure(
                f"{TOOL} returned a hit from corpus {hit.get('corpus')!r}, not {corpus}"
            )
        cite = hit.get("cite")
        if not isinstance(cite, str) or not cite.startswith(f"{corpus}:"):
            raise ProbeFailure(f"{TOOL} returned a hit without a {corpus} cite: {cite!r}")
        cites.append(cite)
    return cites


def _text(content: Any) -> str:
    """The text of a tool result's content: a string, or a list of text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        raise ProbeFailure(f"{TOOL} result is not JSON: {text[:200]!r}") from None


def _judge(
    outcome: Outcome,
    replies: list[Callable[[], list[str]]],
    problems: list[str],
    fatal: list[str],
) -> Outcome:
    """Pass ``outcome`` on the first reply that yields cites, else gather why not.

    ``fatal`` holds the session's terminal failures: any one fails the session
    even when a reply yields cites. ``problems`` only explain a session that
    has no such reply.
    """
    outcome.calls = len(replies)
    reasons = list(dict.fromkeys(problems))
    if not replies:
        reasons.append(f"no {TOOL} call on {SERVER}")
    for reply in replies:
        try:
            outcome.cites = reply()
        except ProbeFailure as exc:
            reasons.append(str(exc))
            continue
        if fatal:
            outcome.reason = "; ".join(dict.fromkeys(fatal))
            return outcome
        outcome.passed, outcome.reason = True, ""
        return outcome
    outcome.reason = "; ".join(dict.fromkeys(fatal + reasons))
    return outcome


def claude_outcome(lines: Iterable[str], corpus: str) -> Outcome:
    """Judge a ``claude -p --output-format stream-json --verbose`` log."""
    outcome = Outcome("claude")
    uses: list[Any] = []
    results: dict[Any, dict[str, Any]] = {}
    problems: list[str] = []
    fatal: list[str] = []
    for event in parse_jsonl(lines):
        kind = event.get("type")
        outcome.session_id = outcome.session_id or event.get("session_id")
        if kind == "system" and event.get("subtype") == "init":
            for server in event.get("mcp_servers") or []:
                if isinstance(server, dict) and server.get("name") == SERVER:
                    if server.get("status") != "connected":
                        problems.append(f"MCP server {SERVER} is {server.get('status')}")
        if kind == "result":
            if isinstance(event.get("result"), str):
                outcome.answer = event["result"][:500]
            if event.get("is_error"):
                fatal.append(f"session ended in error ({event.get('subtype')})")
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            if kind == "assistant" and block.get("type") == "tool_use":
                if block.get("name") == CLAUDE_TOOL:
                    uses.append(block.get("id"))
            elif kind == "user" and block.get("type") == "tool_result":
                results[block.get("tool_use_id")] = block

    def reply(use_id: Any) -> Callable[[], list[str]]:
        def judge() -> list[str]:
            block = results.get(use_id)
            if block is None:
                raise ProbeFailure(f"{TOOL} call got no result")
            text = _text(block.get("content"))
            if block.get("is_error"):
                raise ProbeFailure(f"{TOOL} returned an error: {text[:300]}")
            return search_hits(_json(text), corpus)

        return judge

    return _judge(outcome, [reply(use_id) for use_id in uses], problems, fatal)


def codex_outcome(lines: Iterable[str], corpus: str) -> Outcome:
    """Judge a ``codex exec --json`` log."""
    outcome = Outcome("codex")
    calls: dict[Any, tuple[str, dict[str, Any]]] = {}
    problems: list[str] = []
    fatal: list[str] = []
    for event in parse_jsonl(lines):
        kind = event.get("type")
        if kind == "thread.started":
            outcome.session_id = outcome.session_id or event.get("thread_id")
        elif kind == "error":
            problems.append(f"codex error: {event.get('message')}")
        elif kind == "turn.failed":
            error = event.get("error") or {}
            fatal.append(
                f"codex error: {error.get('message') if isinstance(error, dict) else error}"
            )
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
            outcome.answer = item["text"][:500]
        if (
            item.get("type") == "mcp_tool_call"
            and item.get("server") == SERVER
            and item.get("tool") == TOOL
        ):
            calls[item.get("id")] = (kind, item)

    def reply(kind: str, item: dict[str, Any]) -> Callable[[], list[str]]:
        def judge() -> list[str]:
            if kind != "item.completed":
                raise ProbeFailure(f"{TOOL} call got no result (status {item.get('status')})")
            error = item.get("error")
            if item.get("status") == "failed" or error:
                message = error.get("message") if isinstance(error, dict) else error
                raise ProbeFailure(f"{TOOL} call failed: {message}")
            result = item.get("result")
            if not isinstance(result, dict):
                raise ProbeFailure(f"{TOOL} call got no result")
            text = _text(result.get("content"))
            if result.get("is_error") or result.get("isError"):
                raise ProbeFailure(f"{TOOL} returned an error: {text[:300]}")
            data = result.get("structured_content", result.get("structuredContent"))
            return search_hits(data if isinstance(data, dict) else _json(text), corpus)

        return judge

    return _judge(outcome, [reply(*call) for call in calls.values()], problems, fatal)


OUTCOMES = {"claude": claude_outcome, "codex": codex_outcome}


def apply_exit(outcome: Outcome, exit_code: int | None, timeout: float) -> Outcome:
    """Fold the client's exit into ``outcome``: a timeout (``None``) or a nonzero
    exit fails the session whatever its log shows."""
    outcome.exit_code = exit_code
    if exit_code is None:
        failure = f"timed out after {timeout:g} s"
    elif exit_code != 0:
        failure = f"{outcome.agent} exited {exit_code}"
    else:
        return outcome
    outcome.passed = False
    outcome.reason = "; ".join(filter(None, [failure, outcome.reason]))
    return outcome


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


def print_config(target: str, corpus: str) -> str:
    """This checkout's ``arc mcp --print-config <target>`` for ``corpus``."""
    proc = subprocess.run(
        [sys.executable, str(ARC), "mcp", "--corpus", corpus, "--print-config", target],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise ProbeFailure(
            f"--print-config {target} exited {proc.returncode}: {proc.stderr[-500:]}"
        )
    return proc.stdout


def _require_checkout(command: str, args: list[str]) -> None:
    arc = os.path.realpath(ARC)
    if arc not in (os.path.realpath(word) for word in [command, *args] if os.path.isabs(word)):
        raise ProbeFailure(f"the printed config does not launch {arc}: {[command, *args]}")


def claude_server(config: str) -> tuple[str, list[str]]:
    """The command and arguments the claude ``--mcp-config`` JSON launches."""
    server = json.loads(config)["mcpServers"][SERVER]
    _require_checkout(server["command"], server["args"])
    return server["command"], list(server["args"])


def codex_overrides(config: str) -> list[str]:
    """The ``-c`` arguments the codex config's comments give, checked against its table."""
    table = tomllib.loads(config)["mcp_servers"][SERVER]
    _require_checkout(table["command"], table["args"])
    overrides: list[str] = []
    for line in config.splitlines():
        words = shlex.split(line.lstrip("#").strip()) if line.startswith("#") else []
        if len(words) == 2 and words[0] == "-c":
            overrides += words
    parsed = {}
    for value in overrides[1::2]:
        key, _, toml_value = value.partition("=")
        parsed[key] = tomllib.loads(f"v = {toml_value}")["v"]
    expected = {f"mcp_servers.{SERVER}.{k}": table[k] for k in ("command", "args")}
    if parsed != expected:
        raise ProbeFailure(f"codex -c overrides {parsed} differ from the table {expected}")
    return overrides


# ---------------------------------------------------------------------------
# Direct stdio session
# ---------------------------------------------------------------------------


class StdioClient:
    """A minimal MCP client over the server's stdio, for the direct checks."""

    def __init__(self, command: list[str], stderr_path: Path):
        self._stderr = open(stderr_path, "wb")
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr
        )
        self._lines: queue.Queue = queue.Queue()
        self._next_id = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})

    def _send(self, message: dict[str, Any]) -> None:
        self.proc.stdin.write((json.dumps(message) + "\n").encode())
        self.proc.stdin.flush()

    def request(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                break
            if line is None:
                raise ProbeFailure(f"arc mcp exited during {method}")
            response = json.loads(line)
            if response.get("id") == self._next_id:
                if "error" in response:
                    raise ProbeFailure(f"{method}: {response['error']}")
                return response["result"]
        raise ProbeFailure(f"no response to {method} within {timeout:g} s")

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self._stderr.close()


def direct_check(command: list[str], corpus: str, query: str, out_dir: Path) -> dict[str, Any]:
    """initialize, tools/list and one search_text call over ``command``'s stdio."""
    client = StdioClient(command, out_dir / "server-direct.stderr")
    try:
        init = client.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "arc-mcp-probe", "version": "0"},
            },
            timeout=60,
        )
        client.notify("notifications/initialized")
        tools = check_tools_list(client.request("tools/list", {}, timeout=30))
        result = client.request(
            "tools/call",
            {"name": TOOL, "arguments": {"query": query, "corpora": [corpus], "limit": 3}},
            timeout=60,
        )
        if result.get("isError"):
            raise ProbeFailure(f"direct {TOOL}: {_text(result.get('content'))}")
        cites = search_hits(
            result.get("structuredContent") or _json(_text(result.get("content"))), corpus
        )
    finally:
        client.close()
    info = init.get("serverInfo") or {}
    return {
        "name": info.get("name"),
        "version": info.get("version"),
        "protocol_version": init.get("protocolVersion"),
        "tools": tools,
        "direct_cites": cites,
    }


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def _version(binary: str) -> str | None:
    try:
        proc = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() or None


def run_session(
    agent: str,
    argv: list[str],
    prompt_stdin: str | None,
    env: dict[str, str],
    *,
    corpus: str,
    out_dir: Path,
    timeout: float,
) -> Outcome:
    """Run one agent session, keep its log, and judge it."""
    log = out_dir / f"{agent}.jsonl"
    with open(log, "wb") as stdout, open(out_dir / f"{agent}.stderr", "wb") as stderr:
        try:
            proc = subprocess.run(
                argv,
                input=prompt_stdin.encode() if prompt_stdin is not None else None,
                stdin=subprocess.DEVNULL if prompt_stdin is None else None,
                stdout=stdout,
                stderr=stderr,
                env=env,
                cwd=REPO,
                timeout=timeout,
            )
            exit_code: int | None = proc.returncode
        except subprocess.TimeoutExpired:
            exit_code = None
    outcome = OUTCOMES[agent](
        log.read_text(encoding="utf-8", errors="replace").splitlines(), corpus
    )
    outcome.log = str(log)
    return apply_exit(outcome, exit_code, timeout)


def claude_argv(config: str) -> list[str]:
    return [
        "claude",
        "-p",
        "--strict-mcp-config",
        "--mcp-config",
        config,
        "--allowedTools",
        CLAUDE_TOOL,
        "--tools",
        "",
        "--output-format",
        "stream-json",
        "--verbose",
    ]


def codex_argv(overrides: list[str], prompt: str) -> list[str]:
    return [
        "codex",
        "exec",
        "--json",
        "--sandbox",
        "read-only",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "-C",
        str(REPO),
        *overrides,
        prompt,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", action="append", choices=sorted(OUTCOMES))
    parser.add_argument("--corpus", default="DevRef")
    parser.add_argument("--query", default="hash join")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per session")
    parser.add_argument("--parse", nargs=2, metavar=("AGENT", "LOG"))
    args = parser.parse_args(argv)

    if args.parse:
        agent, path = args.parse
        if agent not in OUTCOMES:
            parser.error(f"--parse: unknown agent {agent!r}")
        outcome = OUTCOMES[agent](Path(path).read_text(encoding="utf-8").splitlines(), args.corpus)
        print(json.dumps(outcome.to_data(), indent=2))
        return 0 if outcome.passed else 1

    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="arc-mcp-probe-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    agents = list(dict.fromkeys(args.agent or sorted(OUTCOMES)))
    report: dict[str, Any] = {"corpus": args.corpus, "query": args.query, "out_dir": str(out_dir)}
    try:
        configs = {target: print_config(target, args.corpus) for target in ("claude", "codex")}
        command, server_args = claude_server(configs["claude"])
        overrides = codex_overrides(configs["codex"])
        report["launch"] = [command, *server_args]
        report["server"] = direct_check([command, *server_args], args.corpus, args.query, out_dir)
    except ProbeFailure as exc:
        report.update(passed=False, reason=str(exc))
        print(json.dumps(report, indent=2))
        return 1

    prompt = PROMPT.format(query=json.dumps(args.query), corpus=args.corpus)
    limits = {"corpus": args.corpus, "out_dir": out_dir, "timeout": args.timeout}
    sessions = []
    for agent in agents:
        if agent == "claude":
            env = {k: v for k, v in os.environ.items() if k not in CLAUDE_SESSION_ENV}
            outcome = run_session(agent, claude_argv(configs["claude"]), prompt, env, **limits)
        else:
            outcome = run_session(
                agent, codex_argv(overrides, prompt), None, dict(os.environ), **limits
            )
        data = outcome.to_data()
        data["client_version"] = _version(agent)
        sessions.append(data)
    report["sessions"] = sessions
    report["passed"] = all(session["passed"] for session in sessions)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

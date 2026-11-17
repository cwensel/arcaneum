"""The session-log parsers of `scripts/probe-mcp-live.py` (arcaneum#144, RDR-026).

The probe itself is live (it needs Qdrant, MeiliSearch, DevRef, `claude` and
`codex`), so it never runs in this suite. What it decides from a session's log
does: these tests feed its parsers fixture lines and check that only a
`search_text` call that returned cite-carrying hits from the probed corpus
passes. Positive fixtures under `fixtures/mcp_probe/` are excerpts of a real
run's logs; the negatives are built from them by hand.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mcp_probe"


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "probe_mcp_live", ROOT / "scripts" / "probe-mcp-live.py"
    )
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()

HIT = {
    "corpus": "DevRef",
    "chunk_id": "31d24db8-7fd9-4509-b606-801b8db8bf78",
    "locator": "~/Documents/References/book.pdf#page=7",
    "cite": "DevRef:~/Documents/References/book.pdf#page=7",
    "score": 0.98,
    "snippet": "a hash join builds a table",
}


def _data(hits):
    return {
        "schema_version": 1,
        "query": "hash join",
        "corpora": ["DevRef"],
        "missing_corpora": [],
        "limit": 3,
        "offset": 0,
        "hits": hits,
    }


def _lines(*events):
    return [json.dumps(event) for event in events]


# ---------------------------------------------------------------------------
# tools/list
# ---------------------------------------------------------------------------


def _tool(name, read_only=True):
    tool = {"name": name, "inputSchema": {"type": "object"}}
    if read_only is not None:
        tool["annotations"] = {"readOnlyHint": read_only}
    return tool


FOUR = ["list_corpora", "search_text", "search_semantic", "get_passage"]


def test_tools_list_accepts_exactly_the_four_read_only_tools():
    result = {"tools": [_tool(name) for name in FOUR]}
    assert probe.check_tools_list(result) == sorted(FOUR)


def test_tools_list_rejects_a_fifth_tool():
    result = {"tools": [_tool(name) for name in [*FOUR, "write_note"]]}
    with pytest.raises(probe.ProbeFailure, match="write_note"):
        probe.check_tools_list(result)


def test_tools_list_rejects_a_missing_tool():
    result = {"tools": [_tool(name) for name in FOUR[:3]]}
    with pytest.raises(probe.ProbeFailure, match="get_passage"):
        probe.check_tools_list(result)


@pytest.mark.parametrize("read_only", [None, False])
def test_tools_list_rejects_a_tool_not_marked_read_only(read_only):
    tools = [_tool(name) for name in FOUR[1:]] + [_tool("list_corpora", read_only)]
    with pytest.raises(probe.ProbeFailure, match="readOnlyHint.*list_corpora"):
        probe.check_tools_list({"tools": tools})


# ---------------------------------------------------------------------------
# search_text data
# ---------------------------------------------------------------------------


def test_hits_returns_the_cites_of_the_probed_corpus():
    assert probe.search_hits(_data([HIT]), "DevRef") == [HIT["cite"]]


def test_hits_rejects_zero_hits():
    with pytest.raises(probe.ProbeFailure, match="no hits"):
        probe.search_hits(_data([]), "DevRef")


def test_hits_rejects_a_hit_without_a_cite():
    hit = {k: v for k, v in HIT.items() if k != "cite"}
    with pytest.raises(probe.ProbeFailure, match="cite"):
        probe.search_hits(_data([hit]), "DevRef")


def test_hits_rejects_a_hit_from_another_corpus():
    hit = {**HIT, "corpus": "Other", "cite": "Other:x.md"}
    with pytest.raises(probe.ProbeFailure, match="Other"):
        probe.search_hits(_data([hit]), "DevRef")


def test_parse_jsonl_skips_blank_and_non_json_lines():
    assert probe.parse_jsonl(["", "not json", '{"type": "x"}', "[1]"]) == [{"type": "x"}]


# ---------------------------------------------------------------------------
# claude -p --output-format stream-json
# ---------------------------------------------------------------------------

CLAUDE_SESSION = "0b6f5c1e-3f4e-4b7a-9d55-1c2f3a4b5c6d"


def _claude_init(status="connected"):
    return {
        "type": "system",
        "subtype": "init",
        "session_id": CLAUDE_SESSION,
        "mcp_servers": [{"name": "arc", "status": status}],
    }


def _claude_use(name="mcp__arc__search_text", tool_id="toolu_1", corpora=("DevRef",)):
    return {
        "type": "assistant",
        "session_id": CLAUDE_SESSION,
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": {"query": "hash join", "corpora": list(corpora)},
                }
            ]
        },
    }


def _claude_result(content, tool_id="toolu_1", is_error=False):
    return {
        "type": "user",
        "session_id": CLAUDE_SESSION,
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content,
                    "is_error": is_error,
                }
            ]
        },
    }


def _claude_text(data):
    return [{"type": "text", "text": json.dumps(data)}]


def _claude_end():
    return {"type": "result", "subtype": "success", "is_error": False, "session_id": CLAUDE_SESSION}


def test_claude_session_with_hits_passes():
    lines = _lines(
        _claude_init(),
        _claude_use(),
        _claude_result(_claude_text(_data([HIT]))),
        _claude_end(),
    )
    outcome = probe.claude_outcome(lines, "DevRef")
    assert outcome.passed, outcome.reason
    assert outcome.session_id == CLAUDE_SESSION
    assert outcome.cites == [HIT["cite"]]


def test_claude_tool_result_as_a_plain_string_passes():
    lines = _lines(_claude_init(), _claude_use(), _claude_result(json.dumps(_data([HIT]))))
    assert probe.claude_outcome(lines, "DevRef").passed


def test_claude_search_with_zero_hits_fails():
    lines = _lines(_claude_init(), _claude_use(), _claude_result(_claude_text(_data([]))))
    outcome = probe.claude_outcome(lines, "DevRef")
    assert not outcome.passed
    assert "no hits" in outcome.reason


def test_claude_tool_result_carrying_an_error_fails():
    content = [{"type": "text", "text": "service-down: MeiliSearch is unreachable"}]
    lines = _lines(_claude_init(), _claude_use(), _claude_result(content, is_error=True))
    outcome = probe.claude_outcome(lines, "DevRef")
    assert not outcome.passed
    assert "service-down" in outcome.reason


def test_claude_call_to_another_tool_fails():
    lines = _lines(
        _claude_init(),
        _claude_use(name="mcp__arc__list_corpora"),
        _claude_result(_claude_text({"corpora": [], "service_errors": []})),
    )
    outcome = probe.claude_outcome(lines, "DevRef")
    assert not outcome.passed
    assert "no search_text call" in outcome.reason


def test_claude_search_with_no_result_fails():
    outcome = probe.claude_outcome(_lines(_claude_init(), _claude_use()), "DevRef")
    assert not outcome.passed
    assert "no result" in outcome.reason


def test_claude_result_matched_to_its_own_call():
    # The hits answer a list_corpora call, not the search_text call.
    lines = _lines(
        _claude_init(),
        _claude_use(name="mcp__arc__list_corpora", tool_id="toolu_0"),
        _claude_use(tool_id="toolu_1"),
        _claude_result(_claude_text(_data([HIT])), tool_id="toolu_0"),
    )
    assert not probe.claude_outcome(lines, "DevRef").passed


def test_claude_server_not_connected_is_reported():
    lines = _lines(_claude_init(status="failed"), _claude_end())
    outcome = probe.claude_outcome(lines, "DevRef")
    assert not outcome.passed
    assert "arc" in outcome.reason and "failed" in outcome.reason


# ---------------------------------------------------------------------------
# codex exec --json
# ---------------------------------------------------------------------------

CODEX_THREAD = "019a0000-aaaa-7bbb-8ccc-000000000001"


def _codex_call(status="completed", tool="search_text", result=None, error=None, item_id="item_1"):
    return {
        "type": "item.completed",
        "item": {
            "id": item_id,
            "type": "mcp_tool_call",
            "server": "arc",
            "tool": tool,
            "arguments": {"query": "hash join", "corpora": ["DevRef"]},
            "result": result,
            "error": error,
            "status": status,
        },
    }


def _codex_result(data):
    return {"content": [{"type": "text", "text": json.dumps(data)}], "structured_content": data}


def _codex_events(*middle):
    return _lines(
        {"type": "thread.started", "thread_id": CODEX_THREAD},
        {"type": "turn.started"},
        *middle,
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    )


def test_codex_session_with_hits_passes():
    outcome = probe.codex_outcome(
        _codex_events(_codex_call(result=_codex_result(_data([HIT])))), "DevRef"
    )
    assert outcome.passed, outcome.reason
    assert outcome.session_id == CODEX_THREAD
    assert outcome.cites == [HIT["cite"]]


def test_codex_result_with_only_text_content_passes():
    result = {"content": [{"type": "text", "text": json.dumps(_data([HIT]))}]}
    assert probe.codex_outcome(_codex_events(_codex_call(result=result)), "DevRef").passed


def test_codex_search_with_zero_hits_fails():
    outcome = probe.codex_outcome(
        _codex_events(_codex_call(result=_codex_result(_data([])))), "DevRef"
    )
    assert not outcome.passed
    assert "no hits" in outcome.reason


def test_codex_tool_result_carrying_an_error_fails():
    result = {
        "content": [{"type": "text", "text": "timeout: search_text took longer than 20 s"}],
        "structured_content": {"code": "timeout", "message": "took too long"},
        "is_error": True,
    }
    outcome = probe.codex_outcome(_codex_events(_codex_call(result=result)), "DevRef")
    assert not outcome.passed
    assert "timeout" in outcome.reason


def test_codex_failed_call_reports_its_error():
    call = _codex_call(status="failed", error={"message": "user cancelled MCP tool call"})
    outcome = probe.codex_outcome(_codex_events(call), "DevRef")
    assert not outcome.passed
    assert "user cancelled MCP tool call" in outcome.reason


def test_codex_call_to_another_tool_fails():
    call = _codex_call(tool="list_corpora", result=_codex_result({"corpora": []}))
    outcome = probe.codex_outcome(_codex_events(call), "DevRef")
    assert not outcome.passed
    assert "no search_text call" in outcome.reason


def test_codex_call_to_another_server_fails():
    event = _codex_call(result=_codex_result(_data([HIT])))
    event["item"]["server"] = "other"
    outcome = probe.codex_outcome(_codex_events(event), "DevRef")
    assert not outcome.passed


def test_codex_started_but_never_completed_call_fails():
    started = _codex_call(status="in_progress")
    started["type"] = "item.started"
    outcome = probe.codex_outcome(_codex_events(started), "DevRef")
    assert not outcome.passed
    assert "no result" in outcome.reason


def test_codex_turn_failure_is_reported():
    lines = _lines(
        {"type": "thread.started", "thread_id": CODEX_THREAD},
        {"type": "error", "message": "stream disconnected"},
        {"type": "turn.failed", "error": {"message": "stream disconnected"}},
    )
    outcome = probe.codex_outcome(lines, "DevRef")
    assert not outcome.passed
    assert "stream disconnected" in outcome.reason


# ---------------------------------------------------------------------------
# Real run excerpts (DevRef, 2026-11-17; each hit list cut to two, snippets
# shortened). In the real claude log the tool_result content is a plain string.
# ---------------------------------------------------------------------------

REAL = {
    "claude": ("060d8bb5-694c-484d-abc5-be431cfa078e", probe.claude_outcome),
    "codex": ("01a1e8dd-9957-7821-a9fc-b1fe6b5c2cdf", probe.codex_outcome),
}


def _real_lines(agent):
    return (FIXTURES / f"{agent}-session.jsonl").read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("agent", sorted(REAL))
def test_real_session_passes_with_its_session_id(agent):
    session_id, judge = REAL[agent]
    outcome = judge(_real_lines(agent), "DevRef")
    assert outcome.passed, outcome.reason
    assert outcome.session_id == session_id
    assert outcome.calls == 1
    assert len(outcome.cites) == 2
    assert all(cite.startswith("DevRef:") for cite in outcome.cites)


def _without_hits(agent):
    events = [json.loads(line) for line in _real_lines(agent)]
    for event in events:
        for block in (event.get("message") or {}).get("content") or []:
            if block.get("type") == "tool_result":
                block["content"] = json.dumps({**json.loads(block["content"]), "hits": []})
        result = (event.get("item") or {}).get("result")
        if result:
            data = {**result["structured_content"], "hits": []}
            result.update(
                structured_content=data, content=[{"type": "text", "text": json.dumps(data)}]
            )
    return [json.dumps(event) for event in events]


@pytest.mark.parametrize("agent", sorted(REAL))
def test_real_session_emptied_of_hits_fails(agent):
    outcome = REAL[agent][1](_without_hits(agent), "DevRef")
    assert not outcome.passed
    assert "no hits" in outcome.reason


TERMINAL_FAILURE = {
    "claude": {
        "type": "result",
        "subtype": "error_during_execution",
        "is_error": True,
        "session_id": REAL["claude"][0],
    },
    "codex": {"type": "turn.failed", "error": {"message": "stream disconnected"}},
}


@pytest.mark.parametrize("agent", sorted(REAL))
def test_real_session_ending_in_failure_fails_despite_hits(agent):
    # The search returned hits, but the session itself ended in failure.
    lines = _real_lines(agent)[:-1] + [json.dumps(TERMINAL_FAILURE[agent])]
    outcome = REAL[agent][1](lines, "DevRef")
    assert not outcome.passed
    assert outcome.reason


def test_codex_transient_error_event_does_not_fail_a_completed_session():
    # An error event the turn recovers from (no turn.failed) is not terminal.
    lines = _real_lines("codex")
    lines.insert(2, json.dumps({"type": "error", "message": "Reconnecting... 1/5"}))
    assert probe.codex_outcome(lines, "DevRef").passed


@pytest.mark.parametrize("agent", sorted(REAL))
def test_real_session_judged_for_another_corpus_fails(agent):
    assert not REAL[agent][1](_real_lines(agent), "Other").passed


# ---------------------------------------------------------------------------
# The client process's exit
# ---------------------------------------------------------------------------


# How many leading lines of each real log run through the search_text result.
THROUGH_TOOL_RESULT = {"claude": 3, "codex": 5}


@pytest.mark.parametrize("agent", sorted(REAL))
def test_real_session_with_a_clean_exit_passes(agent):
    outcome = probe.apply_exit(REAL[agent][1](_real_lines(agent), "DevRef"), 0, 60)
    assert outcome.passed, outcome.reason
    assert outcome.exit_code == 0


@pytest.mark.parametrize("agent", sorted(REAL))
@pytest.mark.parametrize("exit_code", [1, -15])
def test_real_session_with_hits_but_a_nonzero_exit_fails(agent, exit_code):
    # The client stopped after the hits, before any terminal event of its own.
    lines = _real_lines(agent)[: THROUGH_TOOL_RESULT[agent]]
    outcome = probe.apply_exit(REAL[agent][1](lines, "DevRef"), exit_code, 60)
    assert not outcome.passed
    assert f"exited {exit_code}" in outcome.reason
    assert outcome.exit_code == exit_code


@pytest.mark.parametrize("agent", sorted(REAL))
def test_real_session_that_timed_out_fails(agent):
    outcome = probe.apply_exit(REAL[agent][1](_real_lines(agent), "DevRef"), None, 60)
    assert not outcome.passed
    assert "timed out after 60 s" in outcome.reason

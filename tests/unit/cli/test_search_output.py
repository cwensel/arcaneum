"""Search output contracts exercised through both real Click command paths."""

import json
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from arcaneum.cli.main import cli, main
from arcaneum.search.searcher import SearchResult
from arcaneum.utils.search_output import limit_json_content


@pytest.fixture(params=["semantic", "text"])
def search_cli(request, monkeypatch):
    from arcaneum.cli import fulltext, search

    mode = request.param
    state = {"content": "    [bold]雪[/bold] 'quoted'\n" + "x" * 1500, "empty": False}
    client = MagicMock()
    client.collection_exists.return_value = True
    client.index_exists.return_value = True

    def semantic_results(**kwargs):
        if state["empty"]:
            return []
        if state.get("error"):
            raise RuntimeError(state["error"])
        return [
            SearchResult(
                score=0.875,
                collection=kwargs["collection_name"],
                location="src/example.py:42-68",
                content=state["content"],
                metadata={"file_path": "src/example.py", "content": state["content"]},
                point_id="one",
            )
        ]

    def text_results(*args, **kwargs):
        if state.get("error"):
            raise RuntimeError(state["error"])
        return {
            "hits": (
                []
                if state["empty"]
                else [
                    {
                        "file_path": "src/example.py",
                        "start_line": 42,
                        "end_line": 68,
                        "content": state["content"],
                        "_formatted": {"content": "<em>" + state["content"] + "</em>"},
                    }
                ]
            ),
            "estimatedTotalHits": 0 if state["empty"] else 1,
            "processingTimeMs": 1,
        }

    monkeypatch.setattr(search, "create_qdrant_client", lambda **kwargs: client)
    monkeypatch.setattr(search, "SearchEmbedder", MagicMock())
    monkeypatch.setattr(search, "acquire_embedder_slot", nullcontext)
    monkeypatch.setattr(search, "search_collection", semantic_results)
    monkeypatch.setattr(search, "interaction_logger", MagicMock())
    monkeypatch.setattr(fulltext, "_require_meili", lambda: client)
    monkeypatch.setattr(fulltext, "interaction_logger", MagicMock())
    client.search.side_effect = text_results

    def run(*args):
        return CliRunner().invoke(cli, ["search", mode, "a 'query'", "--corpus", "Code", *args])

    return mode, state, run


@pytest.mark.parametrize("verbose", [False, True])
def test_compact_preserves_source_and_default_budget(search_cli, verbose):
    mode, state, run = search_cli
    result = run("--format", "compact", *(["--verbose"] if verbose else []))
    assert result.exit_code == 0, result.output
    score = " score=0.875" if mode == "semantic" else ""
    assert result.stdout == (
        "Q: a 'query' | corpus: Code | returned: 1\n---\n"
        f"[1] src/example.py:42-68{score}\n{state['content'][:500]}\n[truncated]\n"
    )
    assert "\x1b" not in result.stdout
    assert "<em>" not in result.stdout


@pytest.mark.parametrize("budget", [0, 1, 1200, 5000])
def test_compact_explicit_budget(search_cli, budget):
    _, state, run = search_cli
    result = run("--format", "compact", "--max-content-chars", str(budget))
    assert result.exit_code == 0, result.output
    expected = state["content"][:budget] if budget else state["content"]
    assert expected in result.stdout
    assert ("[truncated]" in result.stdout) == (0 < budget < len(state["content"]))


def test_compact_empty_results(search_cli):
    _, state, run = search_cli
    state["empty"] = True
    result = run("--format", "compact", "--offset", "10")
    assert result.exit_code == 0, result.output
    assert result.stdout == "Q: a 'query' | corpus: Code | returned: 0\n"


def test_compact_multiple_corpora_and_offset(search_cli):
    _, _, run = search_cli
    result = run("--format", "compact", "--corpus", "Other", "--offset", "1")
    assert result.exit_code == 0, result.output
    assert "corpus: Code, Other | returned: 1" in result.stdout
    assert "[2] src/example.py:42-68 | corpus: Other" in result.stdout


def test_compact_exact_budget_and_empty_content(search_cli):
    _, state, run = search_cli
    for content in ("", "雪" * 10):
        state["content"] = content
        result = run("--format", "compact", "--max-content-chars", "10")
        assert result.exit_code == 0, result.output
        assert "[truncated]" not in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        ["--max-content-chars", "-1"],
        ["--max-content-chars", "nope"],
        ["--json", "--format", "compact"],
        ["--json", "--format", "text"],
    ],
)
def test_invalid_options_rejected_before_search(search_cli, args):
    _, state, run = search_cli
    state["error"] = "backend should not be reached"
    result = run(*args)
    assert result.exit_code == 2
    assert "backend should not be reached" not in result.output


@pytest.mark.parametrize(
    "json_args", [["--json"], ["--format", "json"], ["--json", "--format", "json"]]
)
def test_legacy_json_defaults_and_alias(search_cli, json_args):
    mode, state, run = search_cli
    result = run(*json_args)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    hit = payload["results"][0] if mode == "semantic" else payload["data"]["hits"][0]
    assert hit["content"] == (state["content"][:500] if mode == "semantic" else state["content"])
    assert "content_truncated" not in hit


@pytest.mark.parametrize("budget", [0, 10, 1200])
def test_json_budget_overrides_defaults_and_verbose(search_cli, budget):
    mode, state, run = search_cli
    result = run("--json", "--verbose", "--max-content-chars", str(budget))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    hit = payload["results"][0] if mode == "semantic" else payload["data"]["hits"][0]
    assert hit["content"] == (state["content"][:budget] if budget else state["content"])
    assert hit["content_truncated"] == bool(budget)
    nested = hit["metadata"] if mode == "semantic" else hit["_formatted"]
    if budget:
        assert len(nested["content"]) == budget


def test_json_budget_can_exceed_legacy_semantic_cap(search_cli):
    mode, state, run = search_cli
    result = run("--json", "--max-content-chars", "1200")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    hit = payload["results"][0] if mode == "semantic" else payload["data"]["hits"][0]
    assert hit["content"] == state["content"][:1200]


def test_terminal_budget_overrides_line_limit(search_cli):
    _, state, run = search_cli
    state["content"] = "\n".join(f"    line {n}" for n in range(30))
    result = run("--max-content-chars", "0")
    assert result.exit_code == 0, result.output
    assert "line 29" in result.stdout
    result = run("--max-content-chars", "10")
    assert result.exit_code == 0, result.output
    assert "[truncated]" in result.stdout
    assert "line 29" not in result.stdout


@pytest.mark.parametrize("output_args", [[], ["--format", "compact"], ["--json"]])
def test_backend_failure_is_nonzero_and_on_stderr(search_cli, output_args):
    _, state, run = search_cli
    state["error"] = "backend unavailable"
    result = run(*output_args)
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "backend unavailable" in result.stderr


def test_json_limit_does_not_mutate_backend_payload():
    source = {"content": "abcd", "metadata": {"content": "abcd", "other": "keep"}}
    result = limit_json_content(source, 2)
    assert result["content"] == result["metadata"]["content"] == "ab"
    assert result["metadata"]["other"] == "keep"
    assert source["content"] == source["metadata"]["content"] == "abcd"


@pytest.mark.parametrize("format_args", [["--format", "json"], ["--format=json"]])
def test_json_alias_formats_entrypoint_errors(monkeypatch, capsys, format_args):
    monkeypatch.setattr("sys.argv", ["arc", "search", "text", "query", *format_args])
    monkeypatch.setattr("arcaneum.cli.main.configure_ssl_from_env", lambda: None)
    assert main() == 2  # Missing corpus
    assert json.loads(capsys.readouterr().out)["status"] == "error"

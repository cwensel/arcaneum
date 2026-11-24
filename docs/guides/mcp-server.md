# arc MCP Server

`arc mcp` serves read-only arc tools to an MCP client over stdio. It is meant for sessions that have no shell, such
as review sessions, where a reviewer cannot run `arc` directly. Sessions that do have a shell keep using the `arc`
CLI.

The server serves four tools, each returning the same `data` as an `arc` command's `--json` output:
`list_corpora` (`arc corpus list`), `search_text` (`arc search text`), `search_semantic` (`arc search semantic`) and
`get_passage` (`arc search passage`). Search hits follow the [cite grammar](#cite-grammar) below.

Design record: [RDR-026](../rdr/RDR-026-arc-mcp-server.md). Option reference: [CLI Reference](cli-reference.md#mcp-server).

## Launch

A client starts the server itself, as a child process. A client started by launchd (a review worker, for one) runs
with a fixed PATH that usually leaves out the directory holding `arc`, such as `~/Library/Python/3.12/bin`. So the
client config must name `arc` by absolute path. `--print-config` prints a config that does:

```bash
arc mcp --corpus DevRef --print-config claude    # Claude Code --mcp-config JSON
arc mcp --corpus DevRef --print-config codex     # codex mcp_servers.arc TOML, plus -c overrides
```

It prints the snippet to stdout and exits; it starts no server and writes no file. The snippet's `command` is the
running `arc` script, resolved to an absolute path, when that script's shebang names its interpreter by absolute path
(as an installed `arc` does). A script whose shebang needs PATH, such as a checkout's `bin/arc` with
`#!/usr/bin/env python3`, runs under the running Python interpreter: the command is that interpreter and the script's
absolute path is the first argument, so `bin/arc` still puts its checkout's `src` first on the import path. Under
`python -m arcaneum.cli.main`, the command is the running interpreter with `-m arcaneum.cli.main`.

The snippet's `args` carry the options given with `--print-config`: the allowlist as `--corpus` flags (from
`--corpus`, or from `ARC_MCP_CORPORA` when no flag is given), and `--timeout` and `--max-output-chars` when they
differ from the defaults. A launched server then needs no environment of its own.

### Claude Code

```bash
arc mcp --corpus DevRef --print-config claude > arc-mcp.json
claude -p --strict-mcp-config --mcp-config arc-mcp.json "..."
```

The snippet has this shape:

```json
{
  "mcpServers": {
    "arc": {
      "command": "/Users/me/Library/Python/3.12/bin/arc",
      "args": ["mcp", "--corpus", "DevRef"]
    }
  }
}
```

### Codex

Add the `[mcp_servers.arc]` table to `~/.codex/config.toml`, or pass the same settings as the `-c` overrides that
the snippet lists in its trailing comments:

```toml
[mcp_servers.arc]
command = "/Users/me/Library/Python/3.12/bin/arc"
args = ["mcp", "--corpus", "DevRef"]

# The same settings as codex -c overrides:
#   -c 'mcp_servers.arc.command="/Users/me/Library/Python/3.12/bin/arc"'
#   -c 'mcp_servers.arc.args=["mcp", "--corpus", "DevRef"]'
```

```bash
codex exec --sandbox read-only \
  -c 'mcp_servers.arc.command="/Users/me/Library/Python/3.12/bin/arc"' \
  -c 'mcp_servers.arc.args=["mcp", "--corpus", "DevRef"]' \
  "..."
```

### Start-up

The server starts with no prompts, migrations, config writes, container checks or model downloads. stdout carries
only protocol messages; logs, warnings and anything a library prints go to stderr, which both clients capture.
`initialize` reports `serverInfo` with the name `arc` and arcaneum's version, which names the build when diagnosing a
session.

## Allowlist

`--corpus NAME` (repeatable) limits the server to those corpora. `ARC_MCP_CORPORA`, a comma-separated list, does the
same from the environment; the flag wins. With neither, every corpus is served.

With an allowlist, `list_corpora` shows only the allowed corpora, and a call naming any other corpus fails
`unknown-corpus`, with no partial result.

## Per-call limits

- `--timeout SECONDS` (default 30): each call runs under this timeout. Every tool is read-only, so the server can
  abandon a stalled call; it fails `timeout`.
- `--max-output-chars N` (default 40000, at least 4000): bounds the serialized result of every call. A listing over it
  drops whole entries from the tail and adds `output_truncated: true` and `omitted_corpora`; a search drops whole hits
  and adds `output_truncated: true` and `omitted_hits`; a passage shortens its `text` and sets `truncated`. A result
  that does not fit even with no entries, no hits or an empty text fails `bad-argument`, naming the cap.

## Results and error codes

Every tool carries `readOnlyHint: true`. A success puts `data` in `structuredContent` and the same JSON in one text
content block. A failure sets `isError`, puts `{code, message}` in `structuredContent`, and the text block reads
`<code>: <message>`.

The codes are the ones the CLI's `--json` errors carry. Exit is the CLI's exit status for the same failure.

| Code | Exit | Raised when |
| --- | --- | --- |
| `unknown-corpus` | 3 | No requested corpus exists, or one is outside the server allowlist |
| `unknown-chunk` | 3 | `get_passage` finds the chunk id in neither system |
| `bad-filter` | 2 | The filter does not parse, names an unfilterable field, or is a native form for the other mode |
| `bad-argument` | 2 | Any other bad argument, e.g. `limit` above 25 on the server |
| `service-down` | 1 | Qdrant (or MeiliSearch, when enabled) is unreachable; the message names which |
| `semantic-unavailable` | 1 | Q9's causes |
| `timeout` | 1 | The call exceeds the per-call timeout |

A multi-corpus search with some corpora missing still succeeds and lists them in `missing_corpora`. The server
refuses `limit` above 25 rather than clamping it, since its `inputSchema` advertises `maximum: 25`. The refusal names
the field. The CLI's `click.IntRange` options already refuse in the same way.

`semantic-unavailable`'s causes (RDR-026 Q9) are an embedding model missing from the local cache, since the server
never downloads one, and an embedder still busy when the call's time runs out. A reviewer who gets it falls back to
`search_text`. A failure no code classifies carries the code `error`.

## Semantic search

The first `search_semantic` call loads the query embedding model, so it pays the cold load against the call's
timeout; later calls reuse the loaded model. While the model is loaded the server holds one of the embedder slots
that `arc search semantic` shares (`ARCANEUM_SEARCH_CONCURRENCY`, default 2), and it runs one semantic call at a
time. After 5 minutes with no semantic call it unloads the model and frees the slot.

The server only uses models already in the local cache: one that is missing, or whose cache is incomplete, fails
`semantic-unavailable` rather than downloading. `arc search semantic --offline` behaves the same way. Where
`arc search semantic` stamps the current prompt policy on a legacy collection that records none, the server
searches under that policy without stamping it, so a call writes nothing.

## Passages

`get_passage` takes a hit's `corpus` and `chunk_id`, from either search tool, and returns the hit's chunk plus up to
`before` and `after` neighbouring chunks (default 1, at most 5) of the same file and version, in order, with chunk
overlap removed. Its `max_chars` bounds the text (default 8000; `0` is unlimited) and `truncated` says whether it was
cut. The result carries the hit's `cite` and the window's `first_chunk_id` and `last_chunk_id`, so a reviewer can
page further from either end.

The lookup tries Qdrant, then MeiliSearch when it is enabled, and takes the window from the system that holds the
chunk, so a corpus still on MeiliSearch works while Qdrant is down. It creates no payload index and changes no
MeiliSearch setting. An id that no enabled system holds, such as one a
resync replaced, fails `unknown-chunk`; search again for a current id.

## Cite grammar

Each search hit carries a `cite`, a human-facing label for the hit that a reviewer writes in a finding. The machine
handle is the hit's `chunk_id`.

```text
cite     = corpus ":" locator
corpus   = 1*( ALPHA / DIGIT / "-" / "_" )          ; a MeiliSearch index uid, never ":"
locator  = path [ "#" fragment ]
path     = relative-path / "~/" home-relative-path / absolute-path
fragment = "page=" 1*DIGIT                          ; PDF
         / "L" 1*DIGIT "-" 1*DIGIT                   ; code hit that carries start_line and end_line
         / slug                                      ; markdown: slug of the deepest heading in `section`
```

- **Paths** are relative to a sync root recorded at sync time. `arc corpus sync` records each resolved directory
  argument, plus each git toplevel it discovers for a code corpus, as `source_roots` in the corpus's Qdrant collection
  metadata. The cite builder strips the longest root that prefixes `file_path`. Without a matching root (every corpus
  until its next sync, a lone-file sync, or Qdrant down during a text call), the path is absolute with `$HOME` folded
  to `~`.
- **Escaping**: in the path, `#`, `%` and space are percent-encoded (`%23`, `%25`, `%20`). Nothing else is, so
  `db(31).pdf` stays readable. A cite therefore contains no whitespace and splits at its first `:` and its only
  unescaped `#`.
- **Slug**: the last heading of `section`, lowercased; letters, digits and `-` kept, spaces turned into `-`, all else
  dropped. A chunk before any heading cites the path with no fragment.
- **Code** (owner, 2026-11-16, on `arcaneum#145`): a code hit that carries `start_line` and `end_line` cites
  `#L<start>-<end>`. Today those are the MeiliSearch code documents that `arc index text code` writes. Every other
  code hit, including every semantic code hit, cites the path with no fragment, since corpus code chunks record no
  line range. Once `arcaneum#148` records line ranges at index time, `#L` extends to semantic hits and corpus text
  hits with no grammar change.
- **Examples**: `DevRef:Database-Books/db(31).pdf#page=89` (root `~/Documents/References/DevRefGit`), or
  `DevRef:~/Documents/References/DevRefGit/Database-Books/db(31).pdf#page=89` before the root is recorded;
  `Arcaneum:src/arcaneum/cli/errors.py` (a semantic code hit); `Libs:pkg/parse.py#L12-48` (a text hit with a line
  range); `Notes:guides/setup.md#install-qdrant`.
- **Role**: a cite is a human-facing label that `review#49` detects in findings. The machine handle is `chunk_id`.
  Reviewers write a cite in backticks, and review's parser matches `<corpus>:` for the corpora the session's arc
  server listed. Durable references that survive reindexing are `arcaneum#80`'s.

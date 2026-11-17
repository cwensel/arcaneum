# Recommendation 026: Read-Only arc MCP Server for Review Sessions

> Revise during planning; lock at implementation.
> If wrong, abandon code and iterate RDR.

## Metadata

- **Date**: 2026-11-16
- **Status**: Final
- **Type**: Architecture
- **Priority**: High
- **Related Issues**: umbrella `arcaneum#136`; this RDR is `arcaneum#137`; implemented by `arcaneum#138`-`#146`;
  consumers `review#47`, `review#49`, `review#50`; first slice of `arcaneum#77`/`#81`, with `#80` (durable source
  references) and `#82` (whole-unit retrieval) extending it; `arcaneum#92` (injected point ids); narrows RDR-006;
  relies on RDR-025 (query embedding stays local)

## Problem Statement

Review sessions cannot run arc. The review templates drop user MCP config, and the shell in a review session is
either off or sandboxed (`review#47`). A reviewer therefore cannot pick a corpus, search it, read around a hit, or
cite it in a finding. `review#50` is blocked on that.

The fix is an `arc mcp` stdio server with four read-only tools: `list_corpora`, `search_text`, `search_semantic`
and `get_passage`. The owner approved changing the CLI so that it and the server share one retrieval contract
(2026-11-16: "i'm ok modifying the existing cli to better match reuse by the mcp"). Each operation then has a CLI
command and an MCP tool, and both return the same JSON `data`.

This RDR settles the choices that would cost a rewrite if made wrong: the adapter shape and transport, the result
schema, the error codes, the cite grammar, chunk identity, the filter dialect, text ranking, passage lookup, semantic
availability, read-only listing, start time, the relation to `arcaneum#81`, and the relation to RDR-006.

## Context

### Background

RDR-006 chose direct Bash execution over an MCP server for interactive Claude Code. Its decision factors end with
"MCP can be added later if structured type hints become critical". That decision still governs sessions that have a
shell. This RDR adds an MCP server for sessions without one. The plugin's slash commands stay, and they gain the same
hits and cites. RDR-006 itself is unchanged; its FastMCP sketch under "Optional MCP Server Wrapper (Future RDR)" binds
nothing.

The server is a thin adapter. It adds only what a review session needs: stdio, a corpus allowlist, a per-call timeout
and an output cap. Everything is read-only: no index, sync, store, delete, collection or container tools. If Qdrant or
MeiliSearch is down, a call fails with an error that says so.

The only programmatic consumer of today's JSON is `research doctor` in the monotone checkout, which reads
`.data.corpora[].name` from `arc corpus list --json`. That path is kept.

### Technical Environment

- Python >= 3.12; `qdrant-client>=1.18.0` against Qdrant v1.18.2; `meilisearch>=0.43.0` against MeiliSearch v1.54.0
  (`deploy/docker-compose.yml`).
- `mcp` 2.3.0 (latest protocol revision 2026-07-28), not yet a dependency.
- Clients: Claude Code 2.1.326 (`claude -p --strict-mcp-config --mcp-config …`) and codex-cli 0.164.0
  (`codex exec --sandbox read-only -c mcp_servers.arc…`).
- The review worker runs under launchd with a fixed PATH that omits `~/Library/Python/3.12/bin`, where `arc` lives
  (monotone `site/bin/review-worker::worker_path`).

## Research Findings

### Investigation

**Output shapes.** `search/formatter.py::format_json_results` prints a bare `{query, collection, limit, offset,
total_results, results[]}` object with no envelope. That contradicts `docs/reference/cli-output-format.md`
§"`search --json`", which documents the `{status, message, data, errors}` envelope for semantic search.
`cli/fulltext.py::search_text_command` wraps raw MeiliSearch hits in the envelope: every stored field, plus
`_formatted` and `_corpus`, with `data` = `{query, index, corpora, hits, estimatedTotalHits, processingTimeMs, limit,
offset}`. PDF hits carry the whole `quality_manifest`. No cite or locator builder exists in `src/arcaneum`.

**Errors.** `cli/errors.py` defines `ArcaneumError` subclasses with exit codes 1 (general), 2 (invalid arguments) and
3 (not found). The JSON `errors` list holds strings without a code. The generic `except` in `search_text_command`
prints to stderr and calls `sys.exit(1)`, so it bypasses `main()`'s envelope. A bad text filter under `--json` prints
plain text to stderr and nothing to stdout.

**Chunk identity.** `indexing/dual_indexer.py::DualIndexer` writes both systems from one `DualIndexDocument`, and
`schema/document.py::to_qdrant_point` and `::to_meilisearch_doc` both use `doc.id` (no caller passes `point_id`).
The other write paths:

- Qdrant to MeiliSearch backfill (`cli/sync.py::_backfill_qdrant_to_meili`) reuses `str(point.id)`.
- MeiliSearch to Qdrant backfill (`cli/sync.py::_backfill_meili_to_qdrant`, and the in-sync backfill of
  `qdrant_backfill_paths` in `cli/sync.py::_sync_directory_locked`) re-chunks the file and mints `uuid4()` Qdrant ids.
  For a backfilled file the two systems then hold different ids.
- Injected content (`indexing/markdown/pipeline.py::inject_content`) is Qdrant-only, with integer ids from
  `_get_next_point_id` (collisions tracked by `arcaneum#92`).
- `arc index text code` (`indexing/fulltext/code_indexer.py::SourceCodeFullTextIndexer._build_document`) writes
  MeiliSearch-only indexes with ids of the form `<identifier>_<path>_<name>_<start_line>`, which are not valid Qdrant
  ids. Corpus sync does not use this path.
- A resync replaces a file's chunks under fresh `uuid4()` ids. A chunk id is valid until its file is reindexed.

**Locator metadata.** PDF chunks persist `page_number`. Markdown chunks persist `section`, the heading path joined
with `" > "` (`indexing/markdown/chunker.py`). Code corpus chunks persist no line numbers:
`cli/sync.py::_chunk_code_file_worker` emits `file_path`, `filename`, `language`, `chunk_index` and `method`, and the
`ASTCodeChunker` `Chunk` holds only content and method. The `start_line` branch of `cli/fulltext.py::format_location`
fires only for `arc index text code` indexes. The PDF and markdown chunkers compute `chunk_start_char` and
`chunk_end_char`, but neither `to_qdrant_point` nor sync persists them. Every corpus stores an absolute `file_path`
and records no sync root. Corpus names are MeiliSearch index uids, limited to `[A-Za-z0-9_-]`, so they never contain
`:`.

**Filters.** Semantic parses `key=value`, `key:op:value` and Qdrant JSON (`search/filters.py::parse_filter`). Text
hands the string to MeiliSearch. The flat JSON form `{"filename": "db(31).pdf"}` that the arc-search skill documents
for both modes fails in both. Text exits 1 with "Invalid filter attribute". Semantic raises "Invalid filter
structure", because `parse_json_filter` calls `models.Filter(**…)` and Qdrant's model rejects the key as
`extra_forbidden`. `parse_extended_filter` silently drops terms it cannot parse (a non-numeric range value, an
unknown op). It also splits `in` values on the same comma that separates terms, so `language:in:python,java` yields
a filter on `python` alone.

**Text ranking.** `fulltext/client.py::FullTextClient.search` requests no ranking score, and multi-corpus text search
merges round-robin by position. The comment in `search_text_command` anticipates a switch to score order. The
MeiliSearch settings differ by corpus type (`fulltext/indexes.py`): code, PDF and markdown indexes have different
searchable attributes, and only PDF and markdown use stop words.

**Passage neighbours.** `chunk_index` is in no `filterableAttributes`, but `file_path` is filterable in the code,
PDF and markdown settings. Qdrant has one payload index, on `metadata_type`
(`indexing/common/sync.py::ensure_file_manifest_payload_index`). Chunkers overlap: PDF by `overlap_percent=0.15` of
the chunk size (`indexing/pdf/chunker.py`), code by `chunk_lines_overlap` (`indexing/ast_chunker.py`).

**Semantic availability.** `cli/search.py::search_command` builds `SearchEmbedder` inside
`cli/concurrency.py::acquire_embedder_slot`, a flock-backed semaphore that waits `DEFAULT_SEARCH_WAIT_SECONDS` (60 s)
and then raises `SearchSlotUnavailable`. `embeddings/client.py::EmbeddingClient.get_model` falls back to
`local_files_only=False`, a download. `search/embedder.py::SearchEmbedder.generate_query_embedding` can call
`indexing/collection_metadata.py::backfill_embedding_prompt_policy`, a Qdrant metadata write. Queries never route to
a remote worker (`embeddings/remote/routing.py::RemoteRouter.routes`: "Queries never do"; RDR-025 Technical Design,
Routing), so the Wake-on-LAN path (`embeddings/remote/wake.py`) is unreachable from search.

**Listing.** `cli/corpus.py::list_corpora_command` calls `_persist_list_item_count_metadata`, which writes item counts
back to Qdrant on every call. Full directory sync already stamps counts
(`cli/sync.py::_stamp_full_directory_count_metadata`, `count_source: sync`). Repair stamps none.

**Start time.** `cli/utils.py`, `indexing/common/sync.py` and `indexing/collection_metadata.py` import `qdrant_client`
at module level, and `embeddings/client.py` imports `fastembed`. All four sit on the `arc` start path (`utils` reaches
`indexing/common/sync.py` through `text_source`).

**MCP SDK and clients.** `mcp.server.lowlevel.Server` plus `mcp.server.stdio.stdio_server(stdin=, stdout=)` accept
injected streams. The SDK pulls in starlette, uvicorn, cryptography, opentelemetry, httpx and jsonschema. semble 0.7.2
ships its server on `mcp>=2.0,<3.0`. In openai/codex, `codex-rs/core/src/mcp_tool_call.rs::requires_mcp_tool_approval`
requires approval for a tool without annotations and exempts tools marked `readOnlyHint: true`.
`codex-rs/rmcp-client/src/stdio_server_launcher.rs::LocalStdioServerLauncher` spawns local stdio servers as direct
children of codex, outside its seatbelt. Both were read at codex HEAD, not at the 0.164 tag.

**review#49.** The review repo has no cite parser. `lib/review/agents.py` parses session logs for usage and exposure
only. `review#49` will record "which findings cite a corpus passage", so this RDR defines the grammar and review
adopts it.

#### Dependency Source Verification

| Dependency | Source Searched? | Key Findings |
| --- | --- | --- |
| `mcp` 2.3.0 | Yes (scratch venv, 2026-11-16) | Low-level `Server` and `stdio_server` take injected streams; import 0.32-0.40 s warm |
| qdrant-client / Qdrant 1.18 | Yes, plus spike | `models.Filter` forbids extra keys; scroll filters on unindexed payload fields |
| MeiliSearch 1.54 | Spike | `file_path` filter works; `showRankingScore` returns `_rankingScore`; `chunk_index` unfilterable |
| openai/codex | Yes (HEAD) | `readOnlyHint` exempts a tool from approval; stdio servers run outside the seatbelt |
| fastembed / HF hub | Yes (repo code) | `get_model` downloads when the local-only load fails |

### Key Discoveries

- **Verified**: `import arcaneum.cli.main` takes 0.70-0.83 s warm and 1.32 s cold (scope review measurement), and
  0.75 s warm on re-check (2026-11-16). `qdrant_client` accounts for 0.58 s cumulative, `fastembed` for 0.15 s of
  that.
- **Verified**: an unindexed Qdrant scroll on DevRef (103,203 points), filtered on `file_path` plus a `chunk_index`
  range, returned chunks 4-6 of one PDF in 1.7 s warm and 7.5 s cold. The same lookup in MeiliSearch on `file_path`
  alone took 4 ms warm and 409 ms cold, returning all 344 chunks of the file. Chunk 5 had the same id in both systems.
- **Verified**: a DevRef text hit's `file_path` is `~/Documents/References/DevRefGit/Database-Books/db(31).pdf`
  (absolute in storage), with `page_number`. The MeiliSearch form `filename = "db(31).pdf"` filters correctly.
- **Verified**: DevRef `_rankingScore` values for a two-word query cluster in 0.9992-0.9999 for the top three hits.
- **Documented**: MeiliSearch-to-Qdrant backfill breaks chunk-id identity; resync rotates ids; code corpus chunks have
  no line range; chunk char offsets are not persisted (source reading above).
- **Documented**: `qwen3-embed` cold load is about 10 s and `arctic-m` about 0.3 s (RDR-025 Background), so a cold
  load fits a 30 s call.
- **Documented**: the bare semantic JSON contradicts `docs/reference/cli-output-format.md`.

### Critical Assumptions

- [x] A Qdrant scroll filters on unindexed payload fields without creating an index — **Status**: Verified
  — **Method**: Spike
- [x] MeiliSearch filters on `file_path` in every corpus type and returns `_rankingScore` on request
  — **Status**: Verified — **Method**: Spike (PDF), Source Search (code and markdown settings)
- [x] `stdio_server` accepts injected streams, so a saved stdout fd can carry the protocol — **Status**: Verified
  — **Method**: Source Search
- [x] Query embedding never routes remote — **Status**: Verified — **Method**: Source Search
- [x] codex exempts `readOnlyHint: true` tools from approval — **Status**: Verified (codex HEAD) — **Method**: Source
  Search; `arcaneum#144` confirms against the installed codex

## Proposed Solution

### Approach

A print-free core package, `src/arcaneum/retrieval/`, implements the four operations. It has no rich and no click,
writes nothing to stdout, starts no container, and imports `qdrant_client` lazily. The CLI commands become formatters
over it. The MCP server calls the same core in process, so the embedder stays warm across calls. Each decision below
answers one question of `arcaneum#137`.

### Technical Design

#### Q1. Adapter shape and transport

The server calls the core in process. It is built on the official `mcp` SDK at its low level
(`mcp.server.lowlevel.Server` with `mcp.server.stdio.stdio_server`), pinned `mcp>=2,<3` as a core dependency, because
the launchd review worker needs it present. Tool `inputSchema`s are written explicitly. The click command lives in
`cli/main.py`; the server in `src/arcaneum/mcp_server.py`.

```text
arc mcp [--corpus NAME]... [--timeout SECONDS=30] [--max-output-chars N=40000]
        [--print-config claude|codex]
ARC_MCP_CORPORA=A,B   # environment form of --corpus; the flag wins
```

- **stdout guard**: before any import that might print, the server saves fd 1 with `os.dup`, points fd 1 at stderr
  with `os.dup2(2, 1)`, and hands the saved fd to `stdio_server` as its writer. Logging, rich consoles, warnings,
  progress bars and stray prints all land on stderr.
- **Start**: no prompts, migrations, config writes, container checks or model downloads. `initialize` returns
  `serverInfo` with name `arc` and arcaneum's `__version__`.
- **Allowlist**: with one, `list_corpora` shows only the allowed corpora and any other name is `unknown-corpus`, with
  no partial result.
- **Per call**: the core runs in a worker thread under the timeout. Core calls are read-only, so the server can
  abandon a stalled one.
- **Annotations**: every tool carries `readOnlyHint: true`.
- **Results**: a success puts `data` in `structuredContent` and the same JSON as one text content block. A failure
  sets `isError`, puts `{code, message}` in `structuredContent`, and the text block reads `<code>: <message>`.
- `--print-config` prints a Claude `--mcp-config` JSON or a codex `mcp_servers.arc` TOML snippet that names an
  absolute command, then exits before any server start (`arcaneum#143`).

#### Q2. One result schema per operation

Both search modes, `arc search passage` and `arc corpus list` emit the envelope `{status, message, data, errors}`
under `--json`, and their `data` carries `schema_version: 1`. Search `data`, identical for both modes:

```text
data = {schema_version, query, corpora, missing_corpora, limit, offset, hits[]}
hit  = {corpus, chunk_id, locator, cite, score, snippet, snippet_truncated[, metadata]}
```

- `snippet` is the chunk content, bounded by `max_content_chars`; `snippet_truncated` says whether it was cut.
- `score` is Qdrant's similarity for semantic hits and `_rankingScore` for text hits. A score compares only within one
  mode, and for text only within one corpus.
- `metadata` (every stored field except vectors and content) appears only with `--verbose`. Without it no hit carries
  `quality_manifest`, `_formatted` or `_corpus`.
- Text and compact output print each hit's cite, so a Bash user cites the same way an MCP user does.
- Semantic hits sort by score across corpora; text hits merge per Q7.
- In the MCP result, the output cap may add `output_truncated: true` with `omitted_hits` or `omitted_corpora` to
  `data`, or shorten a passage (see Output cap).
- `arc corpus list --json` keeps `data.corpora[].name`; each entry carries name, type, sync status, item count and
  description.

**Output cap.** `--max-output-chars` bounds the serialized result of every call. It applies after
`max_content_chars` and `max_chars`, and never cuts a hit or an entry mid-way:

- A search drops whole hits from the tail and adds `output_truncated: true` and `omitted_hits`.
- `get_passage` shortens `text` until the result fits and sets `truncated`.
- `list_corpora` drops whole entries from the tail and adds `output_truncated: true` and `omitted_corpora`.

The flag takes at least 4,000 (`click.IntRange`), which leaves only the echoed arguments (the query and the corpus
names) able to outgrow the cap alone. A result that still exceeds it with no hits, no entries or an empty passage
text fails `bad-argument`, naming the cap. The default 40,000 holds the largest permitted search (25 hits of
1,200-character snippets plus their fields).

#### Q3. Error codes and exit codes

`ArcaneumError` gains a stable `code`. JSON `errors` becomes `[{code, message}]`. In JSON mode every failure prints
the envelope on stdout and exits with the mapped code; text mode prints the message to stderr, as today.

| Code | Exit | Raised when |
| --- | --- | --- |
| `unknown-corpus` | 3 | No requested corpus exists, or one is outside the server allowlist |
| `unknown-chunk` | 3 | `get_passage` finds the chunk id in neither system |
| `bad-filter` | 2 | The filter does not parse, names an unfilterable field, or is a native form for the other mode |
| `bad-argument` | 2 | Any other bad argument, e.g. `limit` above 25 on the server |
| `service-down` | 1 | Qdrant or MeiliSearch is unreachable; the message names which |
| `semantic-unavailable` | 1 | Q9's causes |
| `timeout` | 1 | The call exceeds the per-call timeout |

A multi-corpus search with some corpora missing still succeeds and lists them in `missing_corpora`. The server
refuses `limit` above 25 rather than clamping it, since its `inputSchema` advertises `maximum: 25`. The refusal names
the field. The CLI's `click.IntRange` options already refuse in the same way.

#### Q4. Cite grammar

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

#### Q5. Chunk id

`chunk_id` is the Qdrant point id, which equals the MeiliSearch document id on the `DualIndexer` path. Identity is not
load-bearing: `get_passage` resolves an id through whichever system holds it (Q8). That covers MeiliSearch-to-Qdrant
backfill, which mints new Qdrant ids (`arcaneum#147`), and MeiliSearch-only `arc index text code` ids, which are not
valid Qdrant ids and carry no `chunk_index`. An id that a resync has rotated returns `unknown-chunk`.

#### Q6. One filter dialect

One parser in the core produces a backend-neutral filter, translated to a Qdrant `Filter` and to a MeiliSearch
expression. Neutral forms:

- `key=value[,key=value]`, whitespace around `=` trimmed;
- `key:op:value`, where op is `match`, `in`, `gte`, `gt`, `lte`, `lt` or `contains`. `in` takes `|`-separated values
  (`language:in:python|java`), because `,` separates terms;
- flat JSON `{"field": value | [values]}`, where a list means `in`.

For MeiliSearch every value is quoted, with `"` and `\` escaped, so `(`, spaces and quotes survive. `in` becomes
`IN [...]`, and `gte`/`gt`/`lte`/`lt` become `>=`/`>`/`<=`/`<`. `contains` is `bad-filter` in text, because MeiliSearch's
CONTAINS sits behind an experimental flag.

Native forms pass through, documented as advanced. JSON with a top-level `must`, `should` or `must_not` is Qdrant
native and accepted by semantic only. A string is MeiliSearch native, and accepted by text only, when it contains
one of the words `AND`, `OR`, `NOT`, `TO`, `IN`, `EXISTS`, `IS NULL` or `IS EMPTY`, an operator `!=`, `>` or `<`, or a
double quote. A native form sent to the other mode is `bad-filter`. A term that does not parse is `bad-filter` in
both modes, never dropped. A field the index cannot filter on is `bad-filter`, naming the field and listing the
filterable ones.

#### Q7. Text ranking

Text search requests `showRankingScore` and reports `_rankingScore` as the hit's `score`. Multi-corpus text results
keep the positional round-robin merge, since scores have not been shown comparable across indexes: each corpus type
has its own searchable attributes and stop words, and DevRef's top scores saturate near 1.0. A switch to score order
needs a probe across same-type indexes showing comparability, and a decision of its own.

#### Q8. Passage neighbours

`get_passage(corpus, chunk_id, before=1, after=1, max_chars)` with `before` and `after` each at most 5:

1. Resolve the id. Use Qdrant `retrieve` when the id is a valid point id, else (or on a miss) MeiliSearch
   `get_document`. The system that holds the chunk is the seed's source. From the chunk, take `file_path`,
   `chunk_index` and the version keys it carries (`git_project_identifier`, `git_version_identifier`). A miss in both
   is `unknown-chunk`. A chunk without `chunk_index` (an `arc index text code` definition) is its own passage:
   `before` and `after` do not apply, and the one `chunks` entry has a null `chunk_index`.
2. Fetch the window from the seed's source only, matching `file_path`, `chunk_index` in `[i - before, i + after]`, and
   each version key the seed carries. MeiliSearch-to-Qdrant backfill re-chunks the file under the current chunk
   settings, so one system's ordinals do not select the other's content, and `--git-version` keeps several versions
   under one `file_path`. A Qdrant seed uses a `scroll` with those conditions, creating no payload index. A
   MeiliSearch seed, which is every seed while Qdrant is unavailable, pages a `file_path = "…"` search and selects
   client-side: MeiliSearch filters on neither `chunk_index` nor `git_version_identifier`, and its documents carry no
   `chunk_count`. The window never crosses `file_path` or version.
3. Stitch in `chunk_index` order. Where the payload carries `chunk_start_char`/`chunk_end_char`, splice by offset.
   No corpus persists offsets until `arcaneum#148`, so the default trims the longest suffix of each chunk that
   equals a prefix of the next, bounded by the chunker's overlap. Without a match, chunks join with a blank line.
4. Bound the text by `max_chars` (CLI default 0 = unlimited; MCP default 8,000) and set `truncated`.

```text
data = {schema_version, corpus, chunk_id, cite, first_chunk_id, last_chunk_id,
        chunks[{chunk_id, chunk_index}], text, truncated}
```

`cite` is the hit chunk's cite. The CLI command is `arc search passage CHUNK_ID --corpus C [--before N] [--after N]
[--max-chars N] [--json]`.

#### Q9. Semantic availability

`search_semantic(…, allow_download, deadline, stamp_policy)` in the core:

- `allow_download`: the CLI defaults it on, keeping download-on-first-use; the server always passes off. A model
  missing from the local cache (`EmbeddingClient.is_model_cached`) with downloads off is `semantic-unavailable`.
- `acquire_embedder_slot(timeout=…)` waits only for the time left in the call. A slot still busy then is
  `semantic-unavailable`.
- A cold load counts against the 30 s call; past the deadline the result is `timeout`.
- `stamp_policy=False` on the server path skips `backfill_embedding_prompt_policy`. The search still runs under the
  policy it would have stamped.
- The server loads its `SearchEmbedder` on the first semantic call, holds the slot while it is resident, serializes
  semantic calls on it, and drops both after 5 minutes idle, matching the remote worker's idle exit.
- There is no remote-worker cause: queries never route remote.

A reviewer who gets `semantic-unavailable` falls back to `search_text`.

#### Q10. Read-only listing

`list_corpora` never writes. It may count on demand but persists nothing. Sync already stamps counts; repair gains
the same stamp at the end of `repair_corpus_command`.

#### Q11. Start time

The four modules that import `qdrant_client` or `fastembed` at module level (`cli/utils.py`,
`indexing/common/sync.py`, `indexing/collection_metadata.py`, `embeddings/client.py`) import them lazily. A test
asserts that neither is in `sys.modules` after `import arcaneum.cli.main`. Every short `arc` command gets faster too.
`initialize` must answer within 2 s.

#### Q12. arcaneum#81

The Q2 hit is `arcaneum#81`'s first slice: per-hit corpus and source identity (`corpus`, `locator`, `chunk_id`), an
anchor (`cite`), an excerpt (`snippet`) and truncation (`snippet_truncated`). `#81` adds fields (source version,
content type, indexing state) under a higher `schema_version` without renaming these. `#80` and `#82` likewise extend
the locator and the passage.

#### Q13. Supersession

RDR-006's Bash-over-MCP decision is narrowed, not amended: it still governs sessions that have a shell, and this
RDR's server serves sessions without one. RDR-006 is not superseded.

### Existing Infrastructure Audit

| Proposed Component | Existing Module | Decision |
| --- | --- | --- |
| Core operations | `cli/search.py`, `cli/fulltext.py`, `cli/corpus.py` | Replace bodies: they become formatters over `retrieval/` |
| Multi-corpus fetch | `cli/search_merge.py::fetch_from_corpora` | Reuse behind the core |
| Error codes | `cli/errors.py` | Extend: `code` on `ArcaneumError` |
| Filter parser | `search/filters.py` | Replace with the neutral parser; reuse the Qdrant model mapping |
| Content cap | `utils/search_output.py::limit_json_content` | Reuse for `max_content_chars` |
| Embedder slot | `cli/concurrency.py::acquire_embedder_slot` | Reuse with a call-scoped timeout |
| Count stamping | `cli/sync.py::_stamp_full_directory_count_metadata` | Reuse in repair |
| Collection metadata | `indexing/collection_metadata.py` | Extend with `source_roots` |
| Location strings | `cli/fulltext.py::format_location`, `search/searcher.py::format_location` | Replace with the cite builder |

### Decision Rationale

In process keeps the embedder warm and avoids a 0.75 s import per call. The low-level SDK tracks protocol churn
without arcaneum owning version negotiation, and it keeps tool schemas explicit and the streams under the server's
control. Its 0.4 s import fits the 2 s budget once heavy imports are deferred. One envelope and one hit schema give
the CLI, the server and `#81` a single shape. Root-relative cites stay short and portable while degrading to a
readable absolute form. Making chunk identity non-load-bearing tolerates the write paths that already break it. The
neutral filter dialect makes one MCP `filter` parameter mean one thing while keeping the documented native examples
working.

## Alternatives Considered

### Alternative 1: Run `arc search … --json` per call (Q1)

**Description**: the server shells out to the CLI for every tool call and relays its JSON.

**Pros**:

- Maximum reuse; no core extraction.

**Cons**:

- Every call pays the 0.75 s import, and every semantic call a cold embedder (about 10 s for `qwen3-embed`).
- The CLI's stdout becomes an API, with prints and exits as failure modes.

**Reason for rejection**: per-call latency and fragility; the owner approved changing the CLI instead.

### Alternative 2: FastMCP decorators (Q1)

**Description**: the SDK's high-level FastMCP API derives tool schemas from type hints.

**Pros**:

- Less code per tool.

**Cons**:

- The schemas (bounds such as `maximum: 25`) derive from pydantic instead of being written down.
- Less direct control over the streams the stdout guard depends on.

**Reason for rejection**: the low-level API gives explicit schemas and stream control at the same dependency cost.

### Alternative 3: Hand-rolled JSON-RPC stdio loop (Q1)

**Description**: about 200 lines covering `initialize`, `ping`, `tools/list` and `tools/call`.

**Pros**:

- No dependency and no import time.

**Cons**:

- arcaneum owns protocol-version negotiation as the spec moves (latest revision 2026-07-28).

**Reason for rejection**: the SDK's 0.4 s import is affordable, and protocol churn is not arcaneum's to track.

### Alternative 4: Neutral filter dialect only (Q6)

**Description**: accept only the neutral forms; drop native MeiliSearch and Qdrant input.

**Pros**:

- The cleanest contract.

**Cons**:

- Breaks the documented text examples (`language = python AND project = myapp`, `page_number > 10`) and has no OR.

**Reason for rejection**: passthrough keeps them working at the cost of a documented detection rule.

### Briefly Rejected

- **Wrapping the click commands in process and capturing stdout (Q1)**: keeps the print and `sys.exit` coupling.
- **Keeping each mode's own JSON shape with a documented mapping (Q2)**: consumers parse two shapes, and semantic's
  bare object contradicts the documented envelope.
- **Clamping `limit` (Q3)**: contradicts the advertised `maximum`.
- **Absolute paths only in cites (Q4)**: long, machine-specific and noisy; kept only as the fallback.
- **Git-relative paths for every corpus (Q4)**: PDF and markdown corpora are not in git, and payloads store no git
  root.
- **Full RFC 3986 percent-encoding (Q4)**: renders `db(31).pdf` as `db%2831%29.pdf`.
- **A `#chunk=N` fragment for code hits without lines (Q4)**: adds a grammar term that `review#49` must adopt, and a
  chunk ordinal gives a reader no line to open.
- **Recording line ranges before code cites ship (Q4)**: holds `arcaneum#145` on an index-time change and a resync of
  every code corpus; `arcaneum#148` does it separately, and the path-only cite is a strict subset of the grammar.
- **Computing line ranges at query time (Q4)**: locating the chunk text in the file on disk reads arbitrary paths from
  a read-only call and gives wrong lines once the file has changed since indexing.
- **Requiring id identity across systems (Q5)**: a text-hit id for a backfilled file would fail its passage lookup.
- **Keeping the per-mode filter split and fixing only the docs (Q6)**: one MCP parameter would mean two dialects.
- **Score-ordered text merge (Q7)**: comparability across indexes with different settings is not shown.
- **MeiliSearch-only passage lookup on `chunk_index` (Q8)**: making it filterable changes index settings, which
  triggers a reindex at query time.
- **A Qdrant payload index on `file_path` at query time (Q8)**: creates an index from a read-only call.
- **Qdrant neighbours for every seed (Q8)**: a MeiliSearch seed for a backfilled file carries ordinals from another
  chunking, so a Qdrant window could hold unrelated text.
- **Neighbours by `file_path` alone (Q8)**: `--git-version` keeps several versions under one path, and a window would
  stitch them together.
- **Remote query embedding as a semantic-unavailable cause (Q9)**: queries stay local under RDR-025.
- **Download-on-first-use in the server (Q9)**: a download hangs a review call.
- **Keeping list's count write (Q10)**: listing stays slow and writes from a read path.
- **An `arc mcp` fast path in `cli/entrypoint.py` alone (Q11)**: speeds only the server; lazy imports speed every
  command. A fast path may still be added on top.
- **Waiting for `arcaneum#81`'s full contract (Q12)**: blocks `review#50` on a deferred design that can extend this
  slice instead.
- **Amending or superseding RDR-006 (Q13)**: locked RDRs are not edited, and Bash stays the right path where a shell
  exists.

## Trade-offs

### Consequences

- Positive: one retrieval contract for the CLI, the server and `#81`'s later slices.
- Positive: every `arc` command starts faster.
- Positive: `arc corpus list` becomes read-only.
- Negative: the search JSON changes shape for both modes. Pre-1.0 this needs a release note
  (`docs/reference/release-compatibility-policy.md` §"Migration and Deprecation Rules").
- Negative: `mcp` and its transitive dependencies join the core install.
- Negative: cites stay in the absolute fallback form until each corpus's next sync records its roots.

### Risks and Mitigations

- **Risk**: two roots of one corpus can yield the same relative path, so two cites collide.
  **Mitigation**: a cite is a label; `chunk_id` stays the handle, and `#80` owns durable references.
- **Risk**: the unindexed Qdrant scroll costs 1.7 s warm and 7.5 s cold on a 100k-point corpus.
  **Mitigation**: it fits the 30 s call; a sync-time payload index on `file_path` is the remedy if it does not.
- **Risk**: the MeiliSearch window for a file with more chunks than `maxTotalHits` (default 1,000), across its
  versions, returns a partial file. **Mitigation**: it serves only seeds Qdrant does not hold; a window past the
  returned chunks reports `truncated`.
- **Risk**: native filter detection misreads an unusual neutral value (one containing `>` or a quote).
  **Mitigation**: such values use the JSON form; detection is documented.
- **Risk**: codex at 0.164 sandboxes the server or ignores `readOnlyHint`.
  **Mitigation**: `#144`'s live probe records it and files a follow-up instead of weakening the probe.

### Failure Modes

- A service down gives `service-down` naming the service, both on the CLI (exit 1) and in the tool (`isError`).
- A library that writes to stdout during a call writes to stderr instead, so the protocol stream survives.
- A stale chunk id after resync gives `unknown-chunk`; the reader searches again.
- A missing model gives `semantic-unavailable`; the reviewer uses `search_text`, and the operator runs
  `arc models` to cache the model.
- Diagnosis: the server logs to stderr, which both clients capture; `serverInfo.version` names the build.

## Implementation Plan

### Prerequisites

- [x] All Critical Assumptions verified
- [ ] `arcaneum#138` lands before the items that consume the core

### Minimum Viable Validation

`arcaneum#144`'s live probe: a `claude -p --strict-mcp-config` session and a `codex exec --sandbox read-only`
session each call `search_text` against DevRef through `arc mcp`, get hits carrying cites, and record
`serverInfo.version`.

### Phase 1: Code Implementation

#### Step 1: Core, error codes, read-only listing (`arcaneum#138`)

The `retrieval/` package with `list_corpora`, `search_text` and `search_semantic(allow_download=…)`, returning
dataclasses with `to_data()`. `code` on `ArcaneumError` and the JSON-mode envelope on every failure. List stops
writing; repair stamps counts.

#### Step 2: Hit schema and cite builder (`arcaneum#145`)

The Q2 envelope and hit for both modes, the Q4 builder in the core, `source_roots` recorded at sync,
`showRankingScore`, and the docs that describe search output.

#### Step 3: Filter dialect (`arcaneum#146`)

The Q6 parser and both translations, with the skill and `docs/guides/cli-reference.md` updated.

#### Step 4: Server and list_corpora (`arcaneum#139`)

`arc mcp`, the stdout guard, the allowlist, timeout and cap, lazy imports with the `sys.modules` test, and
`list_corpora`.

#### Step 5: Search tools (`arcaneum#140`, `arcaneum#141`)

`search_text` and `search_semantic` over the core, with the Q9 causes and the CLI-to-tool contract tests.

#### Step 6: Passage (`arcaneum#142`)

`get_passage` in the core, `arc search passage`, and the tool.

#### Step 7: Launch config and docs (`arcaneum#143`)

`--print-config claude|codex` and a guide covering launch, the allowlist, the error codes and the cite grammar.

### Phase 2: Operational Activation

#### Activation Step 1: Live probes (`arcaneum#144`)

Run the probe on this machine and record the version and session evidence; then `review#50` wires arc into reviewers.

### Day 2 Operations

| Resource | List | Info | Delete | Verify | Backup |
| --- | --- | --- | --- | --- | --- |
| `source_roots` collection metadata | N/A | Deferred | N/A | N/A | N/A |

`source_roots` is advisory metadata on an existing collection. It is deleted with its corpus and exported with its
collection metadata. A wrong root yields only a longer cite, so showing it in `arc corpus info` can wait.

### New Dependencies

- `mcp>=2,<3` (MIT), core dependency. Its transitive dependencies (starlette, uvicorn, cryptography, opentelemetry,
  httpx, jsonschema) carry permissive licenses; no legal review is needed.

## Validation

### Testing Strategy

1. **Scenario**: drive the server over stdio pipes: `initialize`, `tools/list`, a call while a dependency prints.
   **Expected**: `serverInfo.version` is set; exactly four tools, each `readOnlyHint: true`; every stdout line parses
   as JSON-RPC.
2. **Scenario**: for each tool, the same arguments to the tool and to its CLI command with `--json`.
   **Expected**: the tool's `structuredContent` equals the command's `data`.
3. **Scenario**: each error code from the core, in CLI JSON mode and in the tool.
   **Expected**: the envelope or `isError` result carries the code; the CLI exits with the mapped code.
4. **Scenario**: fixture hits from PDF, markdown and code corpora in both modes.
   **Expected**: identical hit keys; each cite parses back to corpus and locator; `max_content_chars` bounds the
   snippet and sets `snippet_truncated`.
5. **Scenario**: filter table tests over every neutral form and both native forms.
   **Expected**: the expected Qdrant filter and MeiliSearch expression; `(`, spaces and `"` survive; unparseable,
   unfilterable and cross-mode native input give `bad-filter`.
6. **Scenario**: `get_passage` on a middle, first and last chunk; an id from a text hit and from a semantic hit; a
   text-hit id for a backfilled file; two indexed versions sharing one `file_path`; an `arc index text code`
   definition; an unknown id; a cap. **Expected**: ordered windows within one file; equal windows; a window from
   MeiliSearch alone; no chunk from the other version; the definition alone; `unknown-chunk`; `truncated`.
7. **Scenario**: semantic with the model uncached and downloads off; slot busy past the budget; a stalled embed.
   **Expected**: `semantic-unavailable`, `semantic-unavailable`, `timeout`; no download and no Qdrant write on the
   server path.
8. **Scenario**: `arc corpus list --json` with a mocked client. **Expected**: no Qdrant write; `data.corpora[].name`
   present.
9. **Scenario**: `import arcaneum.cli.main`. **Expected**: `qdrant_client` and `fastembed` absent from `sys.modules`.
10. **Scenario**: each tool under `--max-output-chars 4000` with a result larger than the cap; a 5,000-character
    query; the flag at 3,999. **Expected**: every serialized result fits, marked by `output_truncated` or
    `truncated`; the query fails `bad-argument`; 3,999 is refused.

### Performance Expectations

Measured: `import arcaneum.cli.main` 0.75 s warm before deferral, of which `qdrant_client` is 0.58 s; `mcp` import
0.32-0.40 s; Qdrant neighbour scroll 1.7 s warm on DevRef. `#139` records start time before and after the deferral.

## Finalization Gate

### Contradiction Check

No contradictions found between research findings, design principles, and proposed solution. Where the research
contradicts a fact the questions assumed, the design follows the research: code hits without a line range cite the
path alone, stitching defaults to overlap trimming, and passage lookup tolerates broken id identity.

### Assumption Verification

All Critical Assumptions are verified. codex's `readOnlyHint` handling was read at codex HEAD; `#144` confirms it
against the installed 0.164.

#### API Verification

| API Call | Library | Verification |
| --- | --- | --- |
| `Server`, `stdio_server(stdin=, stdout=)` | mcp 2.3.0 | Source Search |
| `scroll` with `FieldCondition` on unindexed keys | Qdrant 1.18 | Spike |
| `retrieve` | qdrant-client | Source Search |
| `search(filter=, showRankingScore=)`, `get_document` | MeiliSearch 1.54 | Spike |
| `requires_mcp_tool_approval` | openai/codex | Source Search (HEAD) |

### Scope Verification

The Minimum Viable Validation is `arcaneum#144`, a child of the umbrella and in scope. It runs both live sessions
against DevRef.

### Cross-Cutting Concerns

- **Versioning**: `schema_version: 1` in retrieval `data`; `serverInfo.version`; a release note for the search JSON
  change.
- **Build tool compatibility**: `mcp` as a core dependency in `pyproject.toml`.
- **Licensing**: MIT and permissive transitive licenses.
- **Deployment model**: launchd's fixed PATH is handled by `--print-config`'s absolute command.
- **IDE compatibility**: N/A.
- **Incremental adoption**: each child lands on its own; cites improve as corpora resync.
- **Secret/credential lifecycle**: the MeiliSearch key is read as the CLI reads it today; nothing new.
- **Memory management**: the server holds at most one embedder, inside the slot, and releases it after 5 minutes
  idle.

### Proportionality

The document answers thirteen questions that seven implementation items depend on. Each section states a decision,
its interface and its rejected alternatives; implementation detail stays in the items.

## References

- `arcaneum#136`-`#146`, `#77`, `#80`, `#81`, `#82`, `#92`; `review#47`, `review#49`, `review#50`
- RDR-006 §"Slash Commands vs MCP Servers" and §"Optional MCP Server Wrapper (Future RDR)"
- RDR-025 Technical Design (Routing) and Background
- `docs/reference/cli-output-format.md` §"`search --json`"
- `docs/reference/release-compatibility-policy.md` §"Migration and Deprecation Rules"
- `src/arcaneum/search/formatter.py`, `search/filters.py`, `search/embedder.py`, `cli/fulltext.py`, `cli/search.py`,
  `cli/corpus.py`, `cli/sync.py`, `cli/errors.py`, `cli/concurrency.py`, `schema/document.py`,
  `indexing/dual_indexer.py`, `indexing/fulltext/code_indexer.py`, `indexing/markdown/pipeline.py`,
  `fulltext/indexes.py`, `embeddings/client.py`, `embeddings/remote/routing.py`
- modelcontextprotocol/python-sdk 2.3.0; openai/codex `codex-rs/core/src/mcp_tool_call.rs`,
  `codex-rs/rmcp-client/src/stdio_server_launcher.rs`; semble 0.7.2

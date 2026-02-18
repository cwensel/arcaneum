# Post-Mortem: RDR-012 Full-Text Search Claude Integration

## RDR Summary

RDR-012 proposed exposing MeiliSearch full-text search capabilities through CLI
commands and slash commands, complementary to semantic search from RDR-007. The
recommended approach was a CLI-first integration following RDR-006 patterns, with
a unified `arc search` group containing `semantic` and `text` subcommands, plus
index management commands under `arc indexes`.

## Implementation Status

Implemented

The core search command (`arc search text`), slash command (`commands/search.md`),
and all index management commands are in production. The implementation evolved
beyond the original plan due to the project-wide "corpus" unification, which
renamed the primary option from `--index` to `--corpus` and added multi-corpus
search support. Location formatting enhancement (listed as remaining work in the
RDR) and integration tests (also listed as remaining) were both completed.

---

## Implementation vs. Plan

### What Was Implemented as Planned

- **Unified search group**: `arc search text` and `arc search semantic` under a
  single `search` command group in `src/arcaneum/cli/main.py:502-544`, exactly as
  proposed
- **CLI-first approach (no MCP)**: Consistent with RDR-006, the implementation
  uses direct CLI execution without MCP server wrappers
- **MeiliSearch native filter pass-through**: Filters are passed directly to
  MeiliSearch without transformation, as specified in the RDR's filter syntax
  section (`src/arcaneum/cli/fulltext.py:168-174`)
- **Highlighting via `_formatted` field**: Search requests include
  `attributes_to_highlight=['content']` and results display highlighted content
  with MeiliSearch `<em>` tags converted to Rich formatting
  (`fulltext.py:279-286`)
- **JSON and Rich console output modes**: Both modes implemented with `--json`
  flag, JSON output includes status/data/errors structure
  (`fulltext.py:233-293`)
- **Inline implementation in fulltext.py**: The RDR's Alternative 4 (separate
  modules) was rejected in favor of keeping search logic inline in
  `src/arcaneum/cli/fulltext.py`, which is what was built
- **All index management commands**: `list`, `create`, `info`, `delete`, `verify`,
  `items`, `export`, `import`, `list-projects`, `delete-project` all implemented
  in `fulltext.py:342-1253`
- **Interaction logging (RDR-018)**: All search and index management commands log
  interactions via `interaction_logger.start()` and `interaction_logger.finish()`
- **Health checks and error handling**: Server availability checks, index
  existence validation, and helpful error messages with docker compose suggestions
  (`fulltext.py:124-128, 157-166`)
- **Slash command**: `commands/search.md` documents both semantic and text
  subcommands with usage examples and guidance on when to use each

### What Diverged from the Plan

- **--index renamed to --corpus**: The RDR planned `--index` as the required
  option for specifying the MeiliSearch index to search. The implementation uses
  `--corpus` as the primary option (repeatable for multi-corpus search) with
  `--index` retained as a hidden/deprecated fallback
  (`main.py:533-534`). This happened because the project-wide "corpus"
  unification (from RDR-009) introduced the concept of a corpus as a dual-indexed
  entity spanning both Qdrant and MeiliSearch. The naming was unified so users
  only need to learn one concept.

- **Function signature changed from single to multi-target**: The RDR planned
  `search_text_command(query, index_name, ...)` with a single index name. The
  implementation uses `search_text_command(query, corpora: List[str], ...)`
  accepting multiple corpora (`fulltext.py:92-99`). This required adding
  multi-corpus search logic with round-robin interleaving of results
  (`fulltext.py:192-209`) and a `resolve_corpora()` backwards compatibility
  function (`fulltext.py:24-47`).

- **Result display format simplified**: The RDR proposed a score-based format:
  `[1] Score: 95% | Language: python | Project: arcaneum | Branch: main` followed
  by location and content. The implementation uses a simpler location-first
  format: `1. /path/file.py:42-67 (authenticate function) [CorpusName]` with
  metadata (language, project) shown only in verbose mode (`fulltext.py:259-293`).
  MeiliSearch does not expose relevance scores in the same way Qdrant does, making
  the score-based format impractical.

- **Location formatting completed (was listed as remaining)**: The RDR listed
  location formatting enhancement as Step 1 of remaining work with estimated 2
  hours effort. The `format_location()` function in `fulltext.py:50-89` was
  fully implemented with start_line-end_line ranges, function/class name context,
  PDF page numbers, and legacy single line_number fallback.

### What Was Added Beyond the Plan

- **`update-settings` command**: An additional index management command
  (`fulltext.py:577-617`) that applies preset type settings to an existing index,
  not mentioned in the RDR
- **Filter attribute error handling with suggestions**: When a filter references a
  non-filterable attribute, the implementation shows available filterable
  attributes and suggests similar ones (`fulltext.py:302-332`), going beyond the
  RDR's "show examples in error message" mitigation
- **`resolve_corpora` backwards compatibility layer**: A shared function
  (`fulltext.py:24-47`, duplicated in `search.py:29-52`) that handles the
  `--corpus` to `--index` migration with clear error messages
- **Multi-corpus result interleaving**: Round-robin interleaving algorithm for
  merging results from multiple corpora (`fulltext.py:192-209`)
- **Corpus-oriented slash command references**: The `commands/search.md` file
  references `/arc:corpus create`, `/arc:corpus sync`, and `/arc:corpus list` as
  related commands, reflecting the corpus-centric user workflow that emerged after
  the RDR was written

### What Was Planned but Not Implemented

- **CLI tests use outdated API**: `tests/cli/test_search_text.py` calls
  `search_text_command()` with keyword argument `index_name='MyCode-fulltext'`,
  but the actual function parameter is `corpora: List[str]`. The tests appear to
  be broken relative to the corpus migration and have not been updated to match
  the current function signature.
- **Score percentage in result display**: The RDR specified showing
  `Score: 95%` for each result. MeiliSearch results do not include a normalized
  relevance score like Qdrant's similarity score, so this was not implemented.
  The omission was pragmatic but was not documented as a known gap.

---

## Drift Classification

| Category | Count | Examples |
| -------- | ----- | -------- |
| **Unvalidated assumption** | 1 | RDR assumed MeiliSearch would provide a displayable relevance score analogous to Qdrant's similarity score; MeiliSearch ranks by relevance but does not expose a normalized score |
| **Framework API detail** | 0 | |
| **Missing failure mode** | 0 | |
| **Missing Day 2 operation** | 1 | Tests were not migrated when the function signature changed from `index_name` to `corpora`; no test migration plan existed |
| **Deferred critical constraint** | 0 | |
| **Over-specified code** | 1 | RDR specified a detailed score-based result format with per-result metadata headers that was substantially simplified in implementation |
| **Under-specified architecture** | 0 | |
| **Scope underestimation** | 1 | Multi-corpus search (searching multiple indexes simultaneously with result interleaving) was not anticipated but required for parity with semantic search |
| **Internal contradiction** | 0 | |
| **Missing cross-cutting concern** | 1 | The project-wide corpus unification (from RDR-009) renamed `--index` to `--corpus` and required backwards compatibility; this cross-project naming evolution was not anticipated |

### Drift Category Definitions

- **Unvalidated assumption** -- a claim presented as fact but never verified by
  spike/POC
- **Framework API detail** -- method signatures, interface contracts, or config
  syntax wrong
- **Missing failure mode** -- what breaks, what fails silently, recovery path not
  considered
- **Missing Day 2 operation** -- bootstrap, CI/CD, removal, rollback, migration
  not planned
- **Deferred critical constraint** -- downstream use case that validates the
  approach was out of scope
- **Over-specified code** -- implementation code that was substantially rewritten
- **Under-specified architecture** -- architectural decision that should have been
  made but wasn't
- **Scope underestimation** -- sub-feature that grew into its own major effort
- **Internal contradiction** -- research findings or stated principles conflicting
  with the proposal
- **Missing cross-cutting concern** -- versioning, licensing, config cache,
  deployment model, etc.

---

## RDR Quality Assessment

### What the RDR Got Right

- **Unified search group architecture**: The decision to place `text` and
  `semantic` as subcommands under a single `search` group was correct and survived
  implementation unchanged. This is the project's primary user-facing search
  interface.
- **CLI-first, no MCP**: Following RDR-006's pattern kept the implementation
  simple and consistent. The slash command delegates directly to the CLI, avoiding
  the complexity of a separate MCP server.
- **MeiliSearch native filter pass-through**: The decision to pass filter
  expressions directly to MeiliSearch (rather than building a custom parser like
  RDR-007's Qdrant filter parser) was pragmatic and correct. The comprehensive
  filter syntax documentation in the RDR was valuable for users.
- **Inline implementation choice**: Rejecting separate `fulltext_searcher.py` and
  `fulltext_formatter.py` modules in favor of a single `fulltext.py` was the
  right call. The search logic is straightforward enough that separate modules
  would have added complexity without benefit.
- **Identifying remaining work explicitly**: The RDR clearly labeled location
  formatting enhancement and integration tests as remaining work with effort
  estimates. Both were subsequently completed. This transparency was more useful
  than pretending the RDR covered everything.
- **Decision tree for semantic vs full-text**: The guidance on when to use each
  search type was carried through to the slash command and remains useful for both
  Claude and human users.

### What the RDR Missed

- **Cross-project naming evolution**: The RDR was written around `--index` as the
  primary option name, but the project's corpus unification (RDR-009) later
  renamed this to `--corpus`. RDRs that define CLI interfaces should note
  dependencies on naming conventions that other RDRs may change.
- **Multi-corpus search requirement**: The need to search across multiple indexes
  simultaneously was not anticipated. This was likely driven by the corpus
  abstraction, where a single logical corpus maps to both a Qdrant collection and
  a MeiliSearch index, and users may want to search several corpora at once.
- **MeiliSearch score availability**: The RDR specified a `Score: 95%` display
  format without verifying that MeiliSearch exposes relevance scores in a
  consumable form. A quick spike against the MeiliSearch search API would have
  revealed this gap.
- **Test maintenance path**: The RDR did not plan for how tests should be updated
  if the function signature changes. When `index_name` became `corpora`, the test
  file was not updated, leaving broken tests.

### What the RDR Over-specified

- **Detailed result format**: The RDR specified exact output formatting including
  score percentages, metadata headers, and line formatting. The implementation
  used a simpler format because (a) MeiliSearch does not provide scores and (b)
  the location-first format is more useful for code search. The detailed format
  specification consumed RDR space without guiding implementation.
- **Exhaustive filter syntax documentation**: The RDR included six tables
  documenting MeiliSearch filter operators (comparison, logical, collection,
  existence, string pattern). While useful as reference, this is MeiliSearch
  documentation that could have been linked rather than duplicated. The
  implementation passes filters through without transformation, making the
  detailed syntax tables unnecessary for implementation guidance.
- **Component 2 code sample for format_location**: The RDR included a complete
  Python function for `format_location()` labeled as "Required Enhancement." The
  actual implementation closely matches this code, but the RDR could have
  described the requirements without prescribing the exact implementation.

---

## Key Takeaways for RDR Process Improvement

1. **Verify external API capabilities with a spike before specifying output
   formats**: The RDR assumed MeiliSearch would provide a displayable relevance
   score, leading to a result format specification that could not be implemented
   as written. A 15-minute spike calling the MeiliSearch search API would have
   revealed that scores are not exposed in the same way as Qdrant, avoiding
   over-specification of the output format.

2. **Flag CLI option names as cross-RDR dependencies when other RDRs may rename
   them**: The `--index` to `--corpus` migration was driven by RDR-009's corpus
   unification, which RDR-012 did not reference as a dependency despite sharing
   the same CLI surface. When an RDR defines CLI options, it should list other
   active or planned RDRs that may affect the same command namespace and note the
   risk of naming changes.

3. **Link to upstream documentation instead of duplicating reference material**:
   The six filter syntax tables consumed significant RDR space duplicating
   MeiliSearch documentation. Since the implementation passes filters through
   without transformation, the RDR needed only to state "use MeiliSearch native
   filter syntax" with a documentation link. Reserve RDR space for decisions and
   rationale, not reference material available elsewhere.

4. **Include a test migration checklist when specifying function signatures**: The
   RDR specified function signatures (e.g., `search_text_command(query,
   index_name, ...)`) but did not plan for updating tests when signatures change.
   RDRs that define API contracts should include a note about which test files
   reference those contracts, so signature changes trigger test updates.

5. **Anticipate multi-target patterns when designing single-target commands**: The
   RDR designed `arc search text` for a single `--index` target, but the natural
   evolution to multi-corpus search required reworking the function signature,
   adding interleaving logic, and changing the CLI option to be repeatable. When
   designing commands that target a named resource, consider whether users will
   eventually need to target multiple resources and design the interface
   accordingly from the start.

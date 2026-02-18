# Post-Mortem: RDR-018 Claude Interaction Logging

## RDR Summary

RDR-018 proposed a lightweight interaction logger that captures all Arc CLI
invocations to `~/.arcaneum/logs/` in JSONL format with UTC date-based
filenames. The logger would detect whether Arc was invoked under Claude Code
or from a terminal, record query metadata and timing, and fail silently to
avoid disrupting the CLI.

## Implementation Status

Implemented

The implementation matches the RDR closely at the core logger level. The
`InteractionLogger` class, log schema, detection logic, and disable mechanism
all shipped as designed. The scope of integration significantly exceeded the
plan, covering all CLI commands (not just the six files the RDR identified),
and a reusable `command_wrapper` abstraction emerged to reduce boilerplate
across the broadened integration surface.

---

## Implementation vs. Plan

### What Was Implemented as Planned

- **InteractionLogger class** (`src/arcaneum/cli/interaction_logger.py`):
  The class is nearly line-for-line identical to the RDR code sample,
  including `start()`, `finish()`, `_write_entry()`, `_reset()`,
  `_detect_claude_environment()`, `_get_project()`, `_get_log_dir()`,
  `_get_claude_env()`, and the `execution_context` property.

- **Global logger instance**: `interaction_logger = InteractionLogger()`
  at module level, imported across all CLI modules.

- **Log location and rotation**: `~/.arcaneum/logs/` with
  `arc-interactions-YYYY-MM-DD.log` filenames using UTC dates.

- **JSONL format**: Entries are appended as one-JSON-object-per-line.

- **Log entry schema**: All fields from the RDR schema are present --
  `timestamp`, `context`, `project`, `command`, `subcommand`, `duration_ms`,
  `cwd`, `error`, `claude_env`, `result_count`.

- **Claude environment detection**: Checks `CLAUDECODE` (primary) and
  `CLAUDE_PLUGIN_ROOT` (secondary), exactly matching the RDR.

- **Future-proof Claude env vars**: `CLAUDE_SESSION_ID` and
  `CLAUDE_AGENT_ID` are captured when present (forward-looking).

- **Disable mechanism**: `ARC_INTERACTION_LOG=0` environment variable
  disables all logging.

- **Silent failure**: All logging wrapped in `try/except` with `pass` --
  logging never breaks the CLI.

- **No new dependencies**: Uses only standard library (json, os, time,
  datetime, pathlib).

- **Search command integration** (`src/arcaneum/cli/search.py`): Semantic
  search logs query, corpora, limit, offset, filters, score_threshold,
  and result_count.

- **Collection command integration** (`src/arcaneum/cli/collections.py`):
  All four planned operations (list, create, delete, info) are instrumented.

- **Index command integration**: `index_pdfs.py`, `index_source.py`, and
  `index_markdown.py` all log command parameters, result counts, and errors.

- **Unit tests** (`tests/cli/test_interaction_logger.py`): All six test
  scenarios from the RDR are covered -- Claude detection (CLAUDECODE and
  CLAUDE_PLUGIN_ROOT), log file creation, entry format validation, project
  field, silent failure, and ARC_INTERACTION_LOG=0.

### What Diverged from the Plan

- **Explicit start/finish instead of track() context manager**: The RDR
  showed search commands using `with interaction_logger.track(...)`, but all
  production code uses the explicit `start()`/`finish()` pair. The `track()`
  context manager exists in the logger class but is never called from
  production code. The explicit pattern was chosen because commands need
  fine-grained control over what gets logged in the `finish()` call (e.g.,
  different extra fields per command), which the `track()` yield-dict
  approach made awkward.

- **track() error handling improved**: The RDR code sample for `track()`
  initialized its result dict as `{"result_count": None}` and did not
  capture exceptions. The actual implementation adds
  `"error": None` to the result dict and wraps the yield in
  `except Exception as e: result["error"] = str(e); raise`, ensuring errors
  are always logged even through the context manager path.

- **Schema field names evolved**: The RDR showed `collection` as a top-level
  field. The implementation uses `corpora` (a list) in search commands,
  reflecting multi-corpus support that matured after the RDR was written.
  Index commands use `collection` (singular), matching their single-target
  semantics.

- **Click result_callback not used**: RDR Research Finding #3 noted that
  `@cli.result_callback()` could intercept all commands. The implementation
  chose per-command instrumentation instead, providing richer per-command
  context at the cost of more integration points.

### What Was Added Beyond the Plan

- **command_wrapper module** (`src/arcaneum/cli/core/command_wrapper.py`):
  An entirely new abstraction not anticipated by the RDR. Provides
  `command_context()` (context manager) and `@with_error_handling()`
  (decorator) that bundle interaction logging with standardized error
  handling, reducing boilerplate for the many commands that needed
  instrumentation. Includes its own test file
  (`tests/cli/core/test_command_wrapper.py`) with 7 tests.

- **Full-text search command logging** (`src/arcaneum/cli/fulltext.py`):
  The `search text` command, MeiliSearch index management commands (verify,
  items, export, import, list-projects, delete-project), and
  `create`/`list`/`info`/`delete` commands are all instrumented. The RDR
  did not mention fulltext commands.

- **Full-text indexing command logging** (`src/arcaneum/cli/index_text.py`):
  `index text-pdf`, `index text-code` (both git-aware and simple modes), and
  `index text-markdown` are all logged. Not mentioned in the RDR.

- **Corpus command logging** (`src/arcaneum/cli/corpus.py`): `corpus create`,
  `corpus list`, `corpus delete`, `corpus info`, `corpus items`, and
  `corpus verify` all have interaction logging. The RDR did not mention
  corpus commands (which were likely added by a later RDR).

- **Sync command logging** (`src/arcaneum/cli/sync.py`): Corpus sync and
  parity operations are instrumented. Not in the RDR scope.

- **Store command logging** (`src/arcaneum/cli/index_markdown.py`
  `store_command`): Agent-generated content storage is logged. Not in the
  RDR scope.

- **Global test fixture** (`tests/conftest.py`): A `mock_interaction_logger`
  fixture was added for use across all test modules. The RDR only specified
  tests for the logger itself, not for mocking it across the test suite.

- **Broad test coverage**: 24 files reference `interaction_logger` (274
  occurrences total). Unit tests for search, collection, index, corpus,
  store, and indexes commands all mock and verify logger calls.

### What Was Planned but Not Implemented

- **No items from the RDR were skipped**. Every planned feature was
  implemented. The only difference is the non-use of the `track()` context
  manager in production, though the method itself was implemented.

---

## Drift Classification

| Category | Count | Examples |
| -------- | ----- | -------- |
| **Unvalidated assumption** | 0 | |
| **Framework API detail** | 1 | Schema field `collection` became `corpora` (list) in search commands to match multi-corpus support |
| **Missing failure mode** | 0 | |
| **Missing Day 2 operation** | 0 | |
| **Deferred critical constraint** | 0 | |
| **Over-specified code** | 1 | `track()` context manager code sample was implemented but not used; all commands use explicit `start()`/`finish()` |
| **Under-specified architecture** | 1 | `command_wrapper.py` abstraction was needed but not anticipated |
| **Scope underestimation** | 1 | RDR scoped 7 files to modify; actual implementation touched 12+ source files and 12+ test files (24 files total reference the logger) |
| **Internal contradiction** | 0 | |
| **Missing cross-cutting concern** | 0 | |

---

## RDR Quality Assessment

### What the RDR Got Right

- **Core class design was production-ready**: The `InteractionLogger` class
  shipped almost unchanged from the RDR code sample. The method signatures,
  field names, silent failure pattern, and disable mechanism all survived
  contact with implementation.

- **Detection strategy was correct**: Checking `CLAUDECODE` and
  `CLAUDE_PLUGIN_ROOT` environment variables works as designed. The
  two-variable approach (primary + secondary) proved robust.

- **Technology selections were sound**: JSONL format, date-based rotation,
  `~/.arcaneum/logs/` location, UTC timestamps, and zero external
  dependencies were all good choices that required no revision.

- **Risk mitigations held**: Silent failure, date-based cleanup, and
  user-home log location all worked as the RDR anticipated.

- **Test scenario coverage was sufficient**: The six test scenarios defined
  in the RDR mapped directly to the test classes that were written.

### What the RDR Missed

- **Scope of integration was severely underestimated**: The "Files to Modify"
  section listed 7 files. The actual implementation required changes to 12+
  source files and 12+ test files. The RDR was written before full-text
  search, corpus management, and sync commands existed (or at least before
  they were considered in scope). A more thorough inventory of "all Arc CLI
  commands" at RDR time would have revealed the true integration surface.

- **Need for a command_wrapper abstraction**: With 20+ command functions
  needing identical start/try/finish/except patterns, the repetitive
  boilerplate became a maintenance concern. The `command_wrapper.py` module
  with `command_context()` and `@with_error_handling()` emerged during
  implementation. The RDR could have anticipated this by noting that
  identical error-handling patterns across N commands warrant a shared
  abstraction.

- **The track() pattern does not fit real commands**: The RDR proposed
  `track()` as the primary integration pattern but actual commands need
  to pass varying extra fields to `finish()` (e.g., `chunks=`, `errors=`,
  `exported_count=`, `is_healthy=`). The yield-dict pattern
  (`result["result_count"] = 10`) scales poorly when each command has
  different result metadata. The explicit `start()`/`finish()` pattern is
  more flexible and was universally preferred.

### What the RDR Over-specified

- **Implementation code for track()**: The RDR included a full code sample
  showing search commands using `with interaction_logger.track(...)`. This
  code was not used. The `track()` method was still implemented in the class
  (with improved error handling), but no production command calls it. The
  RDR would have been better served by showing only the `start()`/`finish()`
  pattern and noting `track()` as an optional convenience.

- **Integration code sample for search**: The "Integration in Search Command"
  section showed exact Click decorator signatures and function bodies. The
  actual search command structure differed (multi-corpus support, resolve
  corpora, different parameter names). The code sample created a false
  impression of how easy integration would be, while the per-command
  complexity was higher than shown.

---

## Key Takeaways for RDR Process Improvement

1. **Inventory all integration points before estimating scope**: RDR-018
   listed 7 files because it enumerated known commands at the time. A
   systematic `grep` or `find` for all Click command functions would have
   revealed the true integration surface (20+ commands across 12+ files).
   For any cross-cutting feature, the RDR should include a discovery step
   that programmatically identifies all affected code paths.

2. **Prefer showing the flexible pattern over the convenient one**: The RDR
   showed `track()` (context manager) as the integration pattern, but
   `start()`/`finish()` (explicit) was what every command actually used. When
   a feature will be integrated into many call sites with varying needs,
   the RDR should lead with the most flexible pattern and note convenience
   wrappers as optional.

3. **Anticipate abstraction needs when integration count exceeds 5**: The
   RDR could have predicted that instrumenting 7+ command functions with
   identical try/except/log patterns would create boilerplate pressure. A
   simple heuristic: if the RDR's integration plan touches more than 5
   call sites with the same pattern, include an abstraction layer (decorator,
   context manager, or base class) in the design.

4. **Do not embed code samples that assume stable external APIs**: The search
   command code sample assumed specific Click decorator signatures and
   parameter names that had already changed by implementation time. RDR code
   samples for integration points should focus on the new code being
   introduced, not the surrounding code that may evolve independently.

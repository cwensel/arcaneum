# Post-Mortem: RDR-006 Claude Code Marketplace Plugin and CLI Integration

## RDR Summary

RDR-006 proposed building a Claude Code marketplace plugin that exposes all
Arcaneum operations through discoverable slash commands, using a CLI-first
architecture (direct Bash execution via `python -m arcaneum.cli.main`) rather
than an MCP server. The RDR planned five slash commands (index-pdfs,
index-source, create-collection, list-collections, search) backed by a flat
Click-based CLI dispatcher, with JSON output, structured error codes, and a
doctor command for setup verification.

## Implementation Status

Implemented -- with substantial scope expansion and architectural evolution.

The core RDR-006 deliverables (plugin metadata, slash commands, JSON output,
exit codes, error formatting, doctor command) are all live and functional.
However, the implementation grew well beyond the original plan: the CLI
evolved from flat subcommands to nested command groups, slash commands
consolidated from five per-operation files to ten domain-grouped files, a
skills system was added for auto-activating agent behaviors, MeiliSearch
dual-indexing expanded the scope considerably, and the execution model
shifted from `python -m arcaneum.cli.main` to a pip-installed `arc` CLI
entry point.

---

## Implementation vs. Plan

### What Was Implemented as Planned

- **Plugin directory structure**: `.claude-plugin/plugin.json` and
  `.claude-plugin/marketplace.json` exist with the required fields (name,
  version, description, author, repository, keywords, commands array).
- **CLI-first architecture**: No MCP server was implemented. All commands
  execute via direct Bash, exactly as the RDR recommended over the MCP
  alternative.
- **JSON output mode**: The `--json` flag is implemented across all CLI
  commands, producing the planned `{status, message, data, errors}` structure
  via `src/arcaneum/cli/output.py`.
- **Structured error handling**: Exit codes 0 (success), 1 (error),
  2 (invalid args), 3 (not found) are implemented in
  `src/arcaneum/cli/errors.py` with custom exception classes
  (`ArcaneumError`, `InvalidArgumentError`, `ResourceNotFoundError`).
- **`[ERROR]`/`[INFO]` prefix conventions**: All progress and error output
  follows the Beads-inspired format documented in the RDR.
- **Python version checking**: `MIN_PYTHON = (3, 12)` guard at top of
  `src/arcaneum/cli/main.py`, exactly as specified.
- **Doctor command**: `arc doctor` exists with checks for Python version,
  dependencies, Qdrant connectivity, embedding models, temp directory, and
  environment variables (`src/arcaneum/cli/doctor.py`).
- **Output formatting utilities**: `src/arcaneum/cli/output.py` provides
  `print_json`, `print_error`, `print_info`, `print_progress`,
  `print_complete`, and `print_warning` -- matching the RDR's specification.
- **Validation scripts**: `scripts/validate-plugin.sh` and
  `scripts/test-plugin-commands.sh` exist and test plugin structure,
  frontmatter, JSON output, and version consistency.
- **Documentation**: `docs/reference/cli-output-format.md` documents exit
  codes, JSON schemas, and progress format. `docs/reference/plugin-compliance.md`
  tracks compliance with Claude Code best practices.
  `docs/guides/claude-code-plugin.md` provides a comprehensive local testing
  guide.

### What Diverged from the Plan

- **CLI invocation pattern**: The RDR planned
  `cd ${CLAUDE_PLUGIN_ROOT} && python -m arcaneum.cli.main <command> $ARGUMENTS`.
  The implementation uses `arc <group> $ARGUMENTS` with the `arc` CLI
  installed via pip. No `${CLAUDE_PLUGIN_ROOT}` is used in any slash command.
  This happened because installing `arc` into the PATH via pip is simpler
  and more robust than depending on the plugin root variable and Python
  module invocation.

- **CLI command structure**: The RDR planned flat top-level subcommands
  (`index-pdfs`, `index-source`, `create-collection`, `list-collections`,
  `search`). The implementation uses nested Click groups
  (`arc collection create`, `arc index pdf`, `arc search semantic`,
  `arc corpus sync`). This emerged as the project added more commands
  (MeiliSearch indexes, corpora, container management, models, config) and
  a flat namespace became unmanageable.

- **Slash command organization**: The RDR planned five individual files, one
  per operation (`index-pdfs.md`, `index-source.md`, `create-collection.md`,
  `list-collections.md`, `search.md`). The implementation has ten files
  organized by domain (`collection.md`, `index.md`, `search.md`,
  `corpus.md`, `container.md`, `models.md`, `config.md`, `doctor.md`,
  `indexes.md`, `store.md`). Each file documents an entire command group
  with all its subcommands, rather than one operation per file.

- **Slash command execution blocks**: The RDR specified execution blocks
  like `python -m arcaneum.cli.main index-pdfs $ARGUMENTS`. The actual
  execution blocks use `arc search $ARGUMENTS`, `arc collection $ARGUMENTS`,
  etc. -- calling the installed CLI directly without `python -m` or
  `CLAUDE_PLUGIN_ROOT`.

- **Plugin name**: The RDR planned the plugin name as `"arcaneum"`. The
  implementation uses `"arc"` for brevity, though the marketplace name
  remains `"arcaneum-marketplace"`.

- **`commands` array in plugin.json**: The RDR specified an explicit
  `commands` array listing all command files. The actual `plugin.json` does
  not contain a `commands` array -- Claude Code discovers commands from the
  `commands/` directory automatically.

### What Was Added Beyond the Plan

- **Skills system**: Five auto-activating skills were added in
  `.claude-plugin/skills/` (arc-search, arc-collection, arc-container,
  arc-indexes, arc-corpus). These use SKILL.md files with
  `allowed-tools: Bash(arc:*), Read` frontmatter. The RDR mentioned agent
  definitions as a "not adopted" future consideration, but skills (a
  distinct Claude Code feature) were implemented for automatic tool
  invocation when users mention searching or collection management.

- **MeiliSearch full-text search integration**: The RDR was scoped to
  Qdrant-only semantic search. The implementation added full MeiliSearch
  support including: `arc indexes` command group, `arc search text`
  subcommand, `arc index text` subgroup, dual-indexing corpora
  (`arc corpus`), and parity checking between systems.

- **Corpus management**: The entire `arc corpus` command group (create,
  delete, sync, info, items, parity, verify) was not anticipated by
  RDR-006. This became a major feature for managing content across both
  Qdrant and MeiliSearch simultaneously.

- **Container management**: `arc container` (start, stop, status, logs,
  restart, reset) for Docker service management was added, replacing
  manual shell scripts.

- **Markdown indexing**: `arc index markdown` was added as a third content
  type alongside PDF and source code.

- **Agent memory store**: `arc store` was added for agents to persist
  research and analysis content with rich metadata.

- **Models command**: `arc models list` was added to enumerate available
  embedding models.

- **Config command**: `arc config show-cache-dir` and `arc config clear-cache`
  were added for cache management.

- **HelpfulGroup**: A custom Click group class (`HelpfulGroup`) was
  implemented that provides contextual error messages and usage examples
  when users invoke a command group without a subcommand.

- **Sandbox and permissions documentation**: `.claude-plugin/README.md` and
  `.claude-plugin/settings.example.json` document sandbox configuration,
  `excludedCommands`, and permission patterns (`Bash(arc:*)`).

- **Integration test script**: `scripts/test-claude-integration.sh` was
  added as a comprehensive integration test (not in the original plan).

### What Was Planned but Not Implemented

- **`${CLAUDE_PLUGIN_ROOT}` in slash commands**: All eight commands in the
  RDR's implementation checklist were supposed to use `${CLAUDE_PLUGIN_ROOT}`.
  None of the actual commands use it because the `arc` CLI is installed via
  pip and available in the PATH directly.

- **`python -m arcaneum.cli.main` invocation**: No slash command uses this
  pattern. The `arc` entry point (`bin/arc`) replaced it.

- **`commands` array in plugin.json**: The RDR specified explicit command
  paths in plugin.json. The actual plugin.json has no `commands` array.

- **Python test files**: `tests/plugin/test_slash_commands.py` and
  `tests/cli/test_main.py` were planned but not created. Shell-based test
  scripts (`scripts/validate-plugin.sh`, `scripts/test-plugin-commands.sh`,
  `scripts/test-claude-integration.sh`) serve this purpose instead.

- **`docs/plugin-usage.md`**: Replaced by `docs/guides/claude-code-plugin.md`
  (same content, different path).

- **`docs/mcp-wrapper-design.md`**: Marked optional in the RDR and not
  created. MCP remains deferred.

- **Hooks directory**: `docs/guides/claude-code-plugin.md` references
  `.claude-plugin/hooks/arc-permissions.sh` for auto-approving arc commands,
  but the hooks directory does not exist in the repository.

---

## Drift Classification

| Category | Count | Examples |
| -------- | ----- | -------- |
| **Unvalidated assumption** | 2 | `${CLAUDE_PLUGIN_ROOT}` assumed necessary for slash commands; `commands` array in plugin.json assumed required for discovery |
| **Framework API detail** | 2 | Claude Code discovers commands from `commands/` directory without explicit listing in plugin.json; skills system (SKILL.md frontmatter) not documented in the RDR research |
| **Missing failure mode** | 0 | |
| **Missing Day 2 operation** | 1 | Sandbox configuration and permission handling for `arc` commands (excludedCommands, Bash(arc:*) patterns) not addressed in the RDR |
| **Deferred critical constraint** | 0 | |
| **Over-specified code** | 2 | Detailed `python -m arcaneum.cli.main` dispatcher code with Click decorators was substantially rewritten as nested groups; five individual slash command markdown files were fully specified but never used in that form |
| **Under-specified architecture** | 1 | CLI command namespace organization (flat vs nested groups) was not analyzed despite being the primary interface contract |
| **Scope underestimation** | 3 | MeiliSearch dual-indexing expanded from zero to a major feature; slash commands grew from 5 to 10; additional content types (markdown, agent memory) added substantial surface area |
| **Internal contradiction** | 0 | |
| **Missing cross-cutting concern** | 1 | Skills / auto-activating agent integration was a distinct Claude Code feature not captured in the plugin research despite analyzing Beads |

### Drift Category Definitions

- **Unvalidated assumption** -- a claim presented as fact
  but never verified by spike/POC
- **Framework API detail** -- method signatures, interface
  contracts, or config syntax wrong
- **Missing failure mode** -- what breaks, what fails
  silently, recovery path not considered
- **Missing Day 2 operation** -- bootstrap, CI/CD,
  removal, rollback, migration not planned
- **Deferred critical constraint** -- downstream use case
  that validates the approach was out of scope
- **Over-specified code** -- implementation code that was
  substantially rewritten
- **Under-specified architecture** -- architectural
  decision that should have been made but wasn't
- **Scope underestimation** -- sub-feature that grew into
  its own major effort
- **Internal contradiction** -- research findings or stated
  principles conflicting with the proposal
- **Missing cross-cutting concern** -- versioning,
  licensing, config cache, deployment model, etc.

---

## RDR Quality Assessment

### What the RDR Got Right

- **CLI-first over MCP decision**: The core architectural decision to use
  direct CLI execution instead of an MCP server was correct. No MCP server
  was ever needed. The analysis of Beads' MCP-first approach and the
  explicit comparison table provided strong justification that held up
  through implementation.

- **Beads best practices adoption**: The JSON output format
  (`{status, message, data, errors}`), exit code conventions (0/1/2/3),
  `[ERROR]`/`[INFO]` prefix patterns, and version compatibility checking
  were all adopted directly from the Beads analysis and implemented as
  specified. These patterns proved genuinely useful.

- **Doctor command**: The setup verification command was a valuable addition
  that the RDR correctly identified and specified. It was implemented with
  the exact check categories (Python version, dependencies, Qdrant
  connection, models, temp directory, environment variables).

- **Migration path to MCP**: The RDR's four-phase migration plan (CLI-first
  -> add MCP -> hybrid -> optional deprecation) correctly identified that
  CLI-first was non-committal and preserved future options. This foresight
  was validated by the fact that MCP has not been needed.

- **Trade-off documentation**: The detailed comparison tables (CLI-first vs
  MCP-first, 14 dimensions scored) and the explicit "what we adopted vs
  didn't adopt from Beads" sections provided clear decision rationale that
  remained useful during implementation.

### What the RDR Missed

- **CLI namespace design**: The RDR assumed flat top-level subcommands
  (`index-pdfs`, `create-collection`) without analyzing how the namespace
  would scale. As the project grew to cover MeiliSearch, corpora, container
  management, models, and config, nested groups became necessary. This
  architectural decision should have been analyzed in the RDR since it
  affects every slash command and all user-facing documentation.

- **Skills / auto-activating behaviors**: The RDR analyzed Beads' agent
  definitions and dismissed them as "future consideration," but skills
  (auto-activating SKILL.md files) are a distinct and simpler Claude Code
  feature that was missed in the research. Skills turned out to be important
  for the user experience -- they allow Claude to automatically invoke
  search when users mention relevant topics.

- **Sandbox and permission model**: The RDR did not address how Claude Code's
  sandbox interacts with CLI tools that need network access (Qdrant,
  MeiliSearch). The implementation required `excludedCommands: ["arc"]` or
  `allowLocalBinding: true` configuration, plus explicit `Bash(arc:*)`
  permission patterns. This is a Day 2 operational concern that every user
  encounters.

- **Plugin-installed CLI vs module invocation**: The RDR assumed
  `python -m arcaneum.cli.main` as the invocation pattern and built
  `${CLAUDE_PLUGIN_ROOT}` into every command. In practice, `pip install -e .`
  puts `arc` in the PATH, making the plugin root variable unnecessary. A
  spike testing the actual Claude Code execution environment would have
  caught this.

- **MeiliSearch as a scope driver**: While RDR-006 was explicitly scoped to
  Qdrant, the subsequent addition of MeiliSearch full-text search (RDRs 008,
  010, 011, 012) roughly doubled the command surface area. The RDR could
  have anticipated this by noting that any additional backend would
  necessitate a command namespace redesign.

### What the RDR Over-specified

- **Detailed slash command markdown content**: The RDR included full
  markdown content for five slash commands (index-pdfs.md, index-source.md,
  create-collection.md, list-collections.md, search.md) with complete
  frontmatter, argument lists, examples, and execution blocks. None of
  these were used as-is because the command structure, invocation pattern,
  and argument names all changed.

- **CLI dispatcher code**: The RDR included a complete `main.py` with Click
  decorators for every command. This was substantially rewritten as nested
  groups with different argument structures, making the sample code
  misleading rather than helpful.

- **Future MCP server code**: The RDR included approximately 100 lines of
  sample FastMCP server code (`src/arcaneum/mcp/server.py`) for a feature
  explicitly deferred to future work. This code was never used and added
  significant length to the document.

- **Beads comparison depth**: While the Beads analysis was valuable for
  adopting best practices, the repeated comparison tables (three separate
  CLI-vs-MCP tables with overlapping content) and the extended
  "why we didn't adopt" analysis occupied substantial document space for
  a decision that was already clear from the user's stated preference.

---

## Key Takeaways for RDR Process Improvement

1. **Spike the execution environment before specifying invocation patterns**:
   The RDR assumed `${CLAUDE_PLUGIN_ROOT}` and `python -m` invocation
   without testing how Claude Code actually executes slash commands. A
   30-minute spike creating a minimal plugin with one test command would
   have revealed that pip-installed CLIs work directly and that
   `CLAUDE_PLUGIN_ROOT` is unnecessary, saving significant rework of the
   command specifications.

2. **Analyze namespace scalability when specifying CLI interfaces**: The RDR
   specified flat subcommands without considering how the namespace would
   grow. When a CLI is the primary interface contract for a plugin, the
   command organization (flat vs nested, grouping strategy) should be an
   explicit architectural decision with its own section, not an implicit
   consequence of listing five commands.

3. **Limit implementation code samples to validated patterns**: The RDR
   included full Click command definitions, complete slash command
   markdown, and a deferred MCP server implementation -- all of which
   were substantially rewritten or unused. Implementation code in RDRs
   should be limited to patterns that have been validated by a spike or
   POC. Detailed code for deferred features should be excluded entirely.

4. **Research the full Claude Code plugin feature set, not just the target
   features**: The RDR researched slash commands and MCP servers but
   missed skills (auto-activating SKILL.md files) and sandbox/permission
   configuration. A systematic enumeration of all Claude Code plugin
   capabilities (skills, hooks, permissions, sandbox rules, settings files)
   would have identified these as relevant concerns during planning rather
   than during implementation.

5. **Flag scope-expanding dependencies explicitly**: RDR-006 was scoped to
   Qdrant-only operations, but the implementation needed to accommodate
   MeiliSearch (from subsequent RDRs). When an RDR's interface design
   (command namespace, plugin structure) will be affected by known future
   work, the RDR should explicitly note the dependency and design for
   extensibility rather than specifying a structure that only fits the
   current scope.

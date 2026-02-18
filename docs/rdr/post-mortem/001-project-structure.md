# Post-Mortem: RDR-001 Project Structure

## RDR Summary

RDR-001 proposed the foundational repository structure for Arcaneum, a CLI tool for
semantic and full-text search. The plan specified a single repository serving as both a
Python CLI development workspace and a Claude Code plugin marketplace, with `.claude-plugin/`
metadata, `commands/` slash commands as markdown files, and `src/arcaneum/` for the Python
package. The RDR explicitly deferred CLI implementation details, indexing algorithms, search
strategies, Docker configuration, and packaging to subsequent RDRs.

## Implementation Status

Partially Implemented

The core architectural vision (single repo, CLI-first, Claude Code plugin, no MCP servers,
local Docker, dual Qdrant + MeiliSearch) was implemented successfully. However, nearly every
structural detail diverged from the plan. The CLI grew from 9 flat subcommands to a hierarchical
command tree with 30+ subcommands across 7 groups. The Claude Code plugin integration shifted
from individual slash command files to group-based commands and a skills system that did not exist
when the RDR was written. The project reached 19 RDRs and a working production-grade tool, far
exceeding the "minimal structure" scope of RDR-001.

---

## Implementation vs. Plan

### What Was Implemented as Planned

- **Single repository for CLI + plugin**: The monorepo approach succeeded. The repository
  at `/Users/chris.wensel/sandbox/arcaneum/` serves as both a development workspace and a
  Claude Code plugin marketplace, exactly as planned.

- **`.claude-plugin/` directory at repo root**: Contains `plugin.json` and `marketplace.json`
  as specified. Both files follow the planned JSON schema structure with the same fields
  (name, description, version, author, repository, license, keywords, etc.).

- **`src/arcaneum/` package layout**: The Python package lives at `src/arcaneum/` with
  `__init__.py` containing `__version__`. The `src/` layout using setuptools is as planned.

- **`commands/` directory for slash commands**: The directory exists at the repo root and
  contains markdown files with YAML frontmatter, as the RDR specified.

- **No MCP servers**: The RDR explicitly chose CLI-first over MCP. No `src/arcaneum/mcp/`
  directory exists. The `plugin.json` has no `mcpServers` section. This decision held.

- **`docs/rdr/` directory**: RDR documentation directory exists and has grown to 19 RDRs,
  validating the planning infrastructure.

- **Core module directories**: `embeddings/`, `collections/`, `indexing/`, `search/`,
  `fulltext/`, and `schema/` all exist under `src/arcaneum/`, matching the planned layout.

- **`.gitignore` coverage**: The implemented `.gitignore` covers all planned entries
  (Python, Docker volumes for Qdrant and MeiliSearch, config files, OS files) plus
  additional entries for IDE files and workspaces.

- **Docker Compose with both Qdrant and MeiliSearch**: Both services are configured in
  `deploy/docker-compose.yml` with volume mounts, resource limits, and environment variables,
  matching the planned dual-service architecture.

- **Click-based CLI framework**: The RDR specified Click for CLI, and `main.py` uses Click
  groups and commands throughout.

- **MIT License**: Implemented as planned.

### What Diverged from the Plan

- **Plugin name changed from "arcaneum" to "arc"**: The RDR planned `"name": "arcaneum"` in
  `plugin.json`. The implementation uses `"name": "arc"` in both `plugin.json` and
  `marketplace.json`. This shorter name is consistent with the `arc` CLI entry point but
  diverges from the RDR's naming. The marketplace plugin entry also uses `"name": "arc"`
  instead of `"name": "arcaneum"`.

- **CLI entry point**: The RDR planned `python -m arcaneum.cli.main` as the invocation
  method, deferring packaging. The implementation provides three entry points: (1)
  `arc` via `pyproject.toml` `[project.scripts]`, (2) `bin/arc` as a development wrapper,
  and (3) `python -m arcaneum.cli.main` still works. Packaging was not deferred; `pyproject.toml`
  with setuptools was implemented immediately.

- **CLI architecture changed from flat to hierarchical**: The RDR planned 9 flat commands:
  `create-collection`, `list-collections`, `index-pdfs`, `index-source`, `search`,
  `search-text`, `create-corpus`, `sync-directory`. The implementation uses nested Click
  groups: `arc collection create`, `arc collection list`, `arc index pdf`, `arc index code`,
  `arc search semantic`, `arc search text`, `arc corpus create`, `arc corpus sync`. This is
  a fundamentally different UX pattern.

- **Slash command file naming and organization**: The RDR planned individual operation files:
  `docker-start.md`, `docker-stop.md`, `index-pdfs.md`, `index-source.md`, `search.md`,
  `search-text.md`, `create-collection.md`, `list-collections.md`, `create-corpus.md`,
  `sync-directory.md`. The actual `commands/` directory contains group-level files:
  `collection.md`, `config.md`, `container.md`, `corpus.md`, `doctor.md`, `index.md`,
  `indexes.md`, `models.md`, `search.md`, `store.md`. Each file covers an entire command
  group rather than a single operation.

- **`plugin.json` has no `commands` array**: The RDR specified a `commands` array listing
  each `.md` file path. The actual `plugin.json` contains no `commands` field at all.
  Claude Code plugin discovery apparently does not require this explicit listing.

- **Skills replaced agents**: The RDR planned an optional `.claude-plugin/agents/`
  subdirectory. The implementation has `.claude-plugin/skills/` with 5 skill directories
  (`arc-collection/`, `arc-container/`, `arc-corpus/`, `arc-indexes/`, `arc-search/`),
  each containing a `SKILL.md` file. The "skills" concept (with auto-activation based on
  intent matching) appears to have replaced or evolved from the "agents" concept after the
  RDR was written.

- **Docker Compose moved to `deploy/`**: The RDR planned `docker-compose.yml` at the repo
  root. It actually lives at `deploy/docker-compose.yml`, a cleaner separation.

- **Docker command renamed to "container"**: The RDR planned `arc docker start/stop`
  commands and `docker-start.md`/`docker-stop.md` slash commands. The implementation uses
  `arc container start/stop` (registered in `main.py` as `container_group`), with the
  slash command at `commands/container.md`.

- **Indexing submodules restructured into subdirectories**: The RDR planned flat files in
  `indexing/` (`pdf_pipeline.py`, `source_code_pipeline.py`, `git_operations.py`, etc.).
  The implementation organizes these into subdirectories: `indexing/pdf/` (chunker.py,
  extractor.py, ocr.py), `indexing/common/` (multiprocessing.py, sync.py),
  `indexing/fulltext/` (ast_extractor.py, code_indexer.py, pdf_indexer.py, sync.py),
  `indexing/markdown/` (chunker.py, discovery.py, injection.py, pipeline.py).

- **Search fulltext files not where planned**: The RDR planned `search/fulltext_searcher.py`
  and `search/fulltext_formatter.py`. These files do not exist. Full-text search is
  implemented in `cli/fulltext.py` (46KB) and `fulltext/client.py` (14KB) instead.

- **Slash command naming convention**: The RDR planned `/arc-*` prefix (e.g.,
  `/arc-docker-start`, `/arc-search`). The actual convention uses `/arc:*` with colon
  separators (e.g., `/arc:search`, `/arc:corpus`), reflecting the Claude Code plugin
  command namespacing evolution.

- **Version mismatch**: `pyproject.toml` declares version `0.2.0` while
  `src/arcaneum/__init__.py` still declares `__version__ = "0.1.0"`. The RDR planned
  version `0.1.0` everywhere.

### What Was Added Beyond the Plan

- **`arc store` command** (`cli/index_markdown.py`): Agent-generated content storage for
  long-term memory, not anticipated by RDR-001.

- **`arc doctor` diagnostic command** (`cli/doctor.py`): System prerequisite verification
  tool, attributed to RDR-006 enhancement.

- **`arc config` command group** (`cli/config.py`): Configuration and cache management,
  not in the plan.

- **`arc models` command group** (`cli/main.py`, `cli/models.py`): Embedding model
  management, not in the plan.

- **`arc indexes` command group** (`cli/fulltext.py`): MeiliSearch index management
  commands mirroring `arc collection` for Qdrant.

- **`arc corpus parity/verify/info/items`**: Extensive corpus management commands for
  checking parity between Qdrant and MeiliSearch, well beyond the planned `create-corpus`
  and `sync-directory`.

- **`arc collection export/import/verify/items`**: Collection portability and integrity
  checking, attributed to RDR-017.

- **`arc index markdown` and `arc index text` subgroups**: Markdown indexing (RDR-014) and
  full-text indexing for code/markdown/PDF (RDR-010/011) were not anticipated.

- **Monitoring module** (`src/arcaneum/monitoring/`): CPU stats and pipeline profiling,
  attributed to RDR-013.

- **Utilities module** (`src/arcaneum/utils/`): Memory management and formatting helpers.

- **SSL configuration** (`src/arcaneum/ssl_config.py`): Corporate proxy support with
  `ARC_SSL_VERIFY` environment variable.

- **Migration system** (`src/arcaneum/migrations.py`): Legacy path migration from
  `~/.arcaneum/` to XDG-compliant structure.

- **Paths module** (`src/arcaneum/paths.py`): Centralized path management with XDG
  compliance.

- **Error handling framework** (`cli/errors.py`): Structured exit codes and custom
  exception hierarchy.

- **Sync module** (`cli/sync.py` at 140KB): The largest file in the project, handling
  corpus synchronization far beyond what `sync-directory` implied.

- **Export/import module** (`cli/export_import.py`): Binary `.arcexp` format and JSONL
  export for collection portability.

- **Additional top-level files**: `CONTRIBUTING.md`, `AGENTS.md`, `CLAUDE.md`,
  `.markdownlint.json`, `.gitattributes`, `.envrc`, `.env.example`,
  `test-install-mac.sh`, `test-install-ubuntu.sh`.

- **Scripts directory** (`scripts/`): 20 utility scripts for benchmarking, profiling,
  database management, and testing.

- **Tests directory** (`tests/`): Organized into `cli/`, `docker/`, `fulltext/`,
  `indexing/`, `integration/`, `schema/`, `unit/` subdirectories.

- **Deploy directory** (`deploy/`): Docker Compose and deployment README.

- **Docs expansion** (`docs/`): Grew to include `guides/`, `reference/`, `testing/`,
  `BENCHMARKING.md`, `DEPENDENCY_NOTES.md`, `qdrant-optimization.md`.

- **Plugin README and settings.example.json**: `.claude-plugin/README.md` with sandbox
  configuration guidance and `settings.example.json` with permission declarations, neither
  planned in the RDR.

- **Additional dependencies**: The actual dependency list in `pyproject.toml` is
  substantially larger: `sentence-transformers`, `torch`, `transformers`, `pymupdf`,
  `pymupdf4llm`, `pymupdf-layout`, `pdfplumber`, `opencv-python-headless`,
  `markdown-it-py`, `python-frontmatter`, `pygments`, `socksio`, `msgpack`, `numpy`,
  `psutil`, `xxhash` -- none of which were in the RDR.

### What Was Planned but Not Implemented

- **`requirements.txt`**: The RDR planned this file (even if deferred). It was never
  created; `pyproject.toml` `[project.dependencies]` was used instead, which is the modern
  Python standard.

- **`.claude-plugin/agents/` directory**: Planned as an optional subdirectory for
  specialized agents. The concept was superseded by `.claude-plugin/skills/`.

- **`commands` array in `plugin.json`**: The RDR specified an explicit list of command
  file paths. This field was never added to the actual `plugin.json`.

- **Flat CLI subcommand pattern**: The planned `create-collection`, `list-collections`,
  `index-pdfs`, `index-source` pattern was abandoned in favor of hierarchical groups.

- **`${CLAUDE_PLUGIN_ROOT}` usage in slash commands**: The RDR showed slash commands
  using `cd ${CLAUDE_PLUGIN_ROOT}` followed by `python -m arcaneum.cli.*`. The actual
  commands use `arc` directly, which is in PATH via pip install.

---

## Drift Classification

| Category | Count | Examples |
| -------- | ----- | -------- |
| **Unvalidated assumption** | 2 | `commands` array in `plugin.json` assumed required but never used; `${CLAUDE_PLUGIN_ROOT}` workflow assumed but superseded by pip install |
| **Framework API detail** | 3 | Plugin naming convention (`arcaneum` vs `arc`); slash command namespacing (`/arc-*` vs `/arc:*`); `commands` field in `plugin.json` not required |
| **Missing failure mode** | 0 | |
| **Missing Day 2 operation** | 1 | Version synchronization between `pyproject.toml` (0.2.0) and `__init__.py` (0.1.0) not planned |
| **Deferred critical constraint** | 1 | Packaging was "deferred to future RDR" but was actually needed immediately for `arc` entry point |
| **Over-specified code** | 3 | CLI `main.py` code sample was substantially rewritten; `plugin.json` commands array never used; `docker-start.md` example pattern abandoned |
| **Under-specified architecture** | 3 | Hierarchical CLI group pattern not decided; skills system not anticipated; indexing subdirectory organization not specified |
| **Scope underestimation** | 2 | CLI grew from 9 commands to 30+; indexing module grew from 7 files to 4 subdirectories with 15+ files |
| **Internal contradiction** | 1 | "No Packaging Yet" principle conflicted with the need for a clean `arc` entry point |
| **Missing cross-cutting concern** | 2 | SSL/proxy support, XDG path compliance, and migration system not anticipated |

### Drift Category Definitions

- **Unvalidated assumption** -- a claim presented as fact but never verified by spike/POC
- **Framework API detail** -- method signatures, interface contracts, or config syntax wrong
- **Missing failure mode** -- what breaks, what fails silently, recovery path not considered
- **Missing Day 2 operation** -- bootstrap, CI/CD, removal, rollback, migration not planned
- **Deferred critical constraint** -- downstream use case that validates the approach was
  out of scope
- **Over-specified code** -- implementation code that was substantially rewritten
- **Under-specified architecture** -- architectural decision that should have been made
  but was not
- **Scope underestimation** -- sub-feature that grew into its own major effort
- **Internal contradiction** -- research findings or stated principles conflicting with
  the proposal
- **Missing cross-cutting concern** -- versioning, licensing, config cache, deployment
  model, etc.

---

## RDR Quality Assessment

### What the RDR Got Right

- **CLI-first, no MCP architecture**: This core decision has held through 19 RDRs and
  a full implementation. The `arc` CLI is the sole interface, slash commands are thin
  wrappers, and no MCP server was ever needed. This was the most consequential decision
  in the RDR.

- **Single repository colocation**: Having CLI code, plugin metadata, slash commands,
  documentation, and deployment configuration in one repo has proven practical through
  significant growth.

- **Dual-database vision (Qdrant + MeiliSearch)**: The plan for both semantic and
  full-text search engines, documented in the Context section, correctly anticipated the
  project's direction across 19 RDRs.

- **Click framework selection**: Click's group/command pattern enabled the hierarchical
  CLI that emerged, even though the RDR only showed flat usage.

- **Research on `.claude-plugin/` conventions**: The investigation of beads plugin structure
  and Claude Code plugin documentation provided a valid foundation, even though specific
  details (skills vs agents, command discovery) evolved.

- **Deferred scope boundaries**: The RDR correctly excluded CLI implementation details,
  indexing algorithms, search strategies, and Docker configuration, allowing subsequent
  RDRs to address each independently.

### What the RDR Missed

- **Claude Code plugin API evolution**: The RDR researched the plugin system as of
  2025-10-18. The skills system (`.claude-plugin/skills/` with `SKILL.md` files),
  colon-separated command namespacing (`/arc:search`), and auto-activation based on
  intent matching were not part of the researched API. A foundational architecture RDR
  should have flagged plugin API stability as a risk.

- **Need for packaging from day one**: Declaring packaging as "deferred to future RDR"
  was wrong. The `arc` CLI entry point, pip installability, and `pyproject.toml` were
  needed immediately for the slash commands to work without `cd ${CLAUDE_PLUGIN_ROOT}`
  and `python -m` gymnastics.

- **CLI organization pattern**: The flat command pattern (`create-collection`,
  `list-collections`) was never viable for a tool that grew to 30+ commands. The
  hierarchical group pattern (`collection create`, `collection list`) should have been
  the specified pattern from the start, especially since Click natively supports it.

- **Sandbox and permissions model**: The actual plugin requires sandbox configuration
  (`settings.example.json`, `excludedCommands: ["arc"]`) and explicit permissions
  (`Bash(arc:*)`). These operational requirements were not considered.

- **Cross-cutting infrastructure needs**: SSL proxy support, XDG-compliant path management,
  legacy migration, structured error handling, and logging configuration were all needed
  but not anticipated. These affect every module.

### What the RDR Over-specified

- **Detailed CLI `main.py` code**: The 70-line code sample in the RDR was entirely
  rewritten. The actual `main.py` is 787 lines with hierarchical groups, option decorators,
  error handling, SSL configuration, and migration hooks. The code sample created a false
  sense of implementation readiness.

- **`plugin.json` commands array**: The RDR specified a 10-element commands array that was
  never implemented. The actual `plugin.json` has no `commands` field, suggesting Claude
  Code discovers commands through the `commands/` directory convention.

- **`docker-start.md` slash command example**: The detailed slash command example with
  `${CLAUDE_PLUGIN_ROOT}` and `python -m arcaneum.cli.docker` was never used. The actual
  commands simply call `arc` with arguments.

- **`requirements.txt` placeholder**: Specifying a requirements file format that was
  superseded by `pyproject.toml` before implementation began.

- **`.claude-plugin/agents/` directory**: Specified as "optional: specialized agents
  (future)" but the concept was replaced by skills before it was ever used.

- **Exhaustive directory tree for files from future RDRs**: The directory structure listed
  files like `search/fulltext_searcher.py`, `search/fulltext_formatter.py`,
  `indexing/pdf_pipeline.py`, and `schema/document.py` with specific RDR attributions.
  While `schema/document.py` exists, the others were organized differently. Specifying
  file-level structure for unimplemented features is speculative waste.

---

## Key Takeaways for RDR Process Improvement

1. **Specify architectural patterns, not file-level structure for unimplemented features**:
   The RDR listed 30+ files across `indexing/`, `search/`, and `fulltext/` with specific
   names and RDR attributions. Nearly all were reorganized during implementation. The RDR
   should have specified the module boundaries (`indexing/`, `search/`, `fulltext/`,
   `schema/`) and their responsibilities, leaving internal file organization to the
   implementing RDR.

2. **Build a spike before specifying plugin integration details**: The RDR assumed
   `commands` array in `plugin.json`, `${CLAUDE_PLUGIN_ROOT}` workflow, `/arc-*` naming,
   and `agents/` directories based on documentation research. A 30-minute spike with
   an actual Claude Code plugin install would have validated or invalidated all of these.
   Future RDRs involving third-party integration APIs should require a working proof of
   concept before locking specifications.

3. **Never defer packaging in a CLI-first project**: The RDR stated "No Packaging Yet"
   and "Defer pip/PyPI to future RDR," but then designed slash commands that invoke
   `python -m arcaneum.cli.main` -- which requires the package to be importable. The
   packaging question is foundational to how every other component works. If the CLI is
   the primary interface, its installation method is a prerequisite, not a deferral.

4. **Design CLI hierarchies up front when using Click groups**: The RDR specified flat
   commands like `create-collection` and `list-collections` despite choosing Click, which
   has native support for nested groups. With 9 planned commands already spanning 4 domains
   (docker, collection, index, search), the hierarchical pattern was foreseeable. Future
   CLI RDRs should specify the group hierarchy and naming convention as a first-class
   architectural decision.

5. **Flag third-party API stability as an explicit risk**: The Claude Code plugin API
   changed between the RDR research date (2025-10-18) and implementation. The transition
   from `agents/` to `skills/`, from `/arc-*` to `/arc:*` namespacing, and the removal
   of the `commands` array were all external API changes. RDRs that depend on evolving
   external APIs should include an explicit "API stability risk" section with a plan for
   re-validation before implementation.

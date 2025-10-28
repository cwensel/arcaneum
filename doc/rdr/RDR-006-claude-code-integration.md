# Recommendation 006: Claude Code Marketplace Plugin and CLI Integration

## Metadata

- **Date**: 2025-10-20
- **Status**: Recommendation
- **Type**: Feature
- **Priority**: High
- **Related Issues**: arcaneum-6, arcaneum-46, arcaneum-47, arcaneum-48, arcaneum-49, arcaneum-50, arcaneum-51
- **Related Tests**: Plugin installation tests, slash command tests, CLI integration tests

## Problem Statement

Create a **Claude Code marketplace plugin** that exposes all Arcaneum operations (indexing, collection management, search) from RDR-003, RDR-004, and RDR-005 to Claude Code users through discoverable slash commands. The integration must:

1. **Provide discoverable interface** via Claude Code plugin marketplace (`.claude-plugin/` structure)
2. **Expose slash commands** for all operations (`/index-pdfs`, `/index-source`, `/create-collection`, `/list-collections`, `/search`)
3. **Integrate with centralized CLI** defined in RDR-001, RDR-003, RDR-004, and RDR-005 (via `arcaneum.cli.main`)
4. **Enable tool discovery** via `/help` command so users can find available operations
5. **Use direct CLI execution** via Bash (no MCP server overhead)

This addresses the **critical requirement** that Claude Code users can discover and perform all Arcaneum operations from within Claude Code through a unified plugin interface, not just from terminal.

## Context

### Background

Arcaneum provides bulk indexing tools via:

- **RDR-004**: PDF indexing with OCR support
- **RDR-005**: Source code indexing with git awareness

**The Missing Piece**: Claude Code users need a **discoverable interface** to these tools.

**Two Integration Approaches**:

1. **Slash Commands → Direct CLI** (via Bash tool)
   - Simple markdown files in `commands/` directory
   - Execute CLI tools directly: `arcaneum index pdf ...`
   - Users discover via `/help` command
   - No MCP server needed

2. **MCP Server Wrapper**
   - Structured tools registered via `.claude-plugin/plugin.json`
   - MCP server wraps CLI functionality
   - Tools appear in Claude UI with type hints
   - More structure, more complexity

**Key Design Questions** (from arcaneum-6):

- How to expose tool to Claude Code? (slash commands vs MCP vs both)
- What's the best way for tool discovery?
- CLI interface design for batch operations?
- Progress reporting to Claude UI?
- Is MCP required or is direct CLI sufficient?

### Technical Environment

**Claude Code Plugin System**:

- Plugin discovery via `.claude-plugin/plugin.json`
- Slash commands from `commands/*.md` files
- Optional MCP servers via `mcpServers` section in plugin.json
- Installation: `/plugin marketplace add user/repo`

**Existing Tools** (from prior RDRs):

- RDR-004: PDF indexing pipeline
- RDR-005: Source code indexing pipeline
- RDR-003: Collection management
- RDR-002: Qdrant Docker setup

**Integration Requirements**:

- Users install via `/plugin marketplace add yourorg/arcaneum`
- Users discover commands via `/help`
- Users execute: `/index-pdfs /Documents/papers --collection Research`
- Claude monitors progress and reports results

## Research Findings

### Investigation Process

**Six parallel research tracks** completed via Beads issues (arcaneum-46 to arcaneum-51):

1. **Claude Code CLI Integration** (arcaneum-46): Direct Bash tool execution patterns
2. **MCP Server Architecture** (arcaneum-47): When MCP is needed vs direct CLI
3. **Marketplace Examples** (arcaneum-48): Reviewed existing plugin patterns
4. **Dual-Use CLI Design** (arcaneum-49): Best practices for human + AI tools
5. **Progress Reporting** (arcaneum-50): Patterns for long-running operations
6. **Concurrent Workflows** (arcaneum-51): Qdrant Docker concurrent access

### Key Discoveries

#### 1. Claude Code Plugin Structure (from Web Research + RDR-001)

**Plugin Directory Structure**:

```
arcaneum/                        # GitHub repo
├── .claude-plugin/
│   ├── plugin.json              # Plugin metadata
│   ├── marketplace.json         # Marketplace catalog
│   └── agents/                  # Optional: custom agents
├── commands/                    # Slash commands (*.md files)
│   ├── index-pdfs.md           # /index-pdfs
│   ├── index-source.md         # /index-source
│   ├── create-collection.md    # /create-collection
│   └── search.md               # /search
├── src/arcaneum/               # Python implementation
└── README.md
```

**Installation Flow**:

1. User: `/plugin marketplace add yourorg/arcaneum`
2. Claude clones repo to `~/.claude/plugins/marketplaces/arcaneum/`
3. Claude reads `.claude-plugin/plugin.json`
4. User: `/plugin install arcaneum`
5. Slash commands available via `/help`

#### 2. Slash Commands vs MCP Servers

**CRITICAL FINDING**: Slash commands can **directly execute CLI tools** via Bash - NO MCP server required!

**Slash Command Example** (`commands/index-pdfs.md`):

```markdown
---
description: Index PDF files to Qdrant collection
argument-hint: <path> --collection <name>
---

Index PDF files for semantic search.

**Execution:**
cd ${CLAUDE_PLUGIN_ROOT}
python -m arcaneum.indexing.pdf $ARGUMENTS
```

**Comparison**:

| Aspect | Slash Commands (Direct Bash) | MCP Server |
|--------|------------------------------|------------|
| **Complexity** | ✅ Simple (just markdown files) | ❌ Complex (server implementation) |
| **Discovery** | ✅ Via `/help` command | ✅ In Claude UI tool list |
| **Type Hints** | ❌ Not structured | ✅ Full type information |
| **Progress** | ✅ Via stdout monitoring | ✅ Via tool responses |
| **Maintenance** | ✅ Low (markdown files) | ❌ High (server code) |
| **User Preference** | ✅ Per user requirement | ❌ Avoid if not needed |

**Decision Factors**:

- User explicitly prefers **NOT using MCP** if avoidable
- Slash commands are **sufficient for tool discovery** (via `/help`)
- Direct CLI execution is **simpler and more maintainable**
- MCP can be added later if structured type hints become critical

#### 3. Tool Discovery Mechanisms

**Option 1: Slash Commands Alone** (Recommended)

- Users discover via `/help` command
- Shows all available commands with descriptions
- Arguments shown via `argument-hint` in frontmatter
- Example:

  ```

    /index-pdfs <path> --collection <name> - Index PDF files to Qdrant
    /index-source <path> --collection <name> - Index source code to Qdrant
    /create-collection <name> --model <model> - Create Qdrant collection

  ```

**Option 2: MCP Tools**

- Tools appear in Claude UI automatically
- Structured interface with type information
- Better for complex parameter combinations
- Requires MCP server implementation

**Option 3: Hybrid**

- Both slash commands AND MCP tools
- Maximum discoverability
- Most complex to maintain

**Recommendation**: **Start with Option 1** (slash commands only), add MCP later if needed

#### 3a. CLI-First vs MCP-First: Detailed Comparison

**Architectural Decision Table**:

| Dimension | CLI-First (Arcaneum) | MCP-First (Beads) | Winner for Arcaneum |
|-----------|---------------------|-------------------|---------------------|
| **Implementation Complexity** | Low - Just markdown files | High - Server code + lifecycle | ✅ CLI-First |
| **Maintenance Burden** | Low - No server to maintain | High - Server updates needed | ✅ CLI-First |
| **Startup Latency** | Fast - Direct execution | Slower - MCP layer overhead | ✅ CLI-First |
| **Type Safety** | None - Text arguments | Full - Typed parameters | ❌ MCP-First |
| **Tool Discovery** | Via `/help` command | In Claude UI automatically | ⚠️ Trade-off (MCP better) |
| **Parameter Validation** | CLI parser errors | Pre-validation by Claude | ⚠️ Trade-off (MCP better) |
| **Error Handling** | Exit codes + stderr | Structured error protocol | ⚠️ Trade-off (MCP better) |
| **Progress Reporting** | stdout monitoring | Structured responses | ⚠️ Similar (both work) |
| **Future Flexibility** | Easy to add MCP later | Hard to remove MCP | ✅ CLI-First |
| **User Preference** | User explicitly prefers | User wants to avoid | ✅ CLI-First |
| **Batch Workloads** | Excellent - Direct CLI | Good - MCP overhead small vs indexing time | ✅ CLI-First |
| **API Documentation** | Slash command markdown | Auto-generated from types | ❌ MCP-First |
| **JSON Output** | With `--json` flag | Native MCP protocol | ⚠️ Similar (both support) |
| **Migration Path** | Add MCP without breaking | Remove MCP breaks tools | ✅ CLI-First |

**Decision Matrix Scoring**:

- **CLI-First Wins**: 8 dimensions (complexity, maintenance, latency, flexibility, user pref, batch, migration, implementation)
- **MCP-First Wins**: 3 dimensions (type safety, UI discovery, API docs)
- **Trade-offs**: 3 dimensions (validation, error handling, progress reporting)

**Conclusion**: CLI-first is the right choice for Arcaneum v1 given:

1. User's explicit preference for simplicity
2. Batch indexing workload characteristics
3. Clear migration path to add MCP later
4. Lower complexity and maintenance burden

#### 3b. Beads Analysis: MCP-First Architecture Deep Dive

**Research Finding**: Analyzed Beads (<https://github.com/steveyegge/beads>), a production-quality Claude Code plugin, to understand professional integration patterns.

**Beads' Architecture (MCP-First)**:

- **MCP server wraps CLI**: FastMCP server (`beads-mcp`) wraps the Go CLI (`bd`)
- **Slash commands call MCP tools**: Commands like `/bd-ready` invoke MCP tools, not direct CLI
- **Structured JSON everywhere**: CLI outputs JSON (`bd ready --json`), MCP parses it
- **Tool discovery in UI**: MCP tools appear in Claude's tool list with type hints
- **Daemon architecture**: Optional daemon for performance with CLI fallback

**Example from Beads**:

```markdown
<!-- commands/ready.md -->
Use the MCP `ready` tool to find tasks with no blockers.
```

**Why Arcaneum Chooses CLI-First Instead**:

| Factor | MCP-First (Beads) | CLI-First (Arcaneum) | Decision |
|--------|-------------------|----------------------|----------|
| **Complexity** | High - MCP server code, lifecycle mgmt | Low - Just markdown files | ✅ Keep simple |
| **Performance** | Overhead - MCP layer adds latency | Fast - Direct execution | ✅ Avoid overhead |
| **Maintenance** | High - Server code to maintain | Low - Markdown only | ✅ Easier maintenance |
| **Discovery** | In Claude UI tool list | Via `/help` command | ⚠️ Trade-off accepted |
| **Type Safety** | Full type hints and validation | Text arguments only | ⚠️ Trade-off accepted |
| **Migration** | Hard to simplify later | Easy to add MCP later | ✅ Flexible path |

**Decision Rationale**:

- User preference for **simplicity** over structure
- User preference for **performance** over type safety
- Indexing operations are **long-running** (MCP overhead negligible vs indexing time, but startup matters)
- `/help` discovery is **sufficient** for initial version
- Can add MCP wrapper later **without breaking** slash commands

**Best Practices Adopted from Beads**:

- ✅ **`${CLAUDE_PLUGIN_ROOT}` usage** - For portable plugin paths
- ✅ **Clear slash command frontmatter** - `description` and `argument-hint`
- ✅ **JSON output mode** - Will add `--json` flag to all CLI commands
- ✅ **Structured error messages** - Consistent `[ERROR]` format
- ✅ **Version compatibility checking** - Validate Python/dependency versions
- ✅ **Comprehensive documentation** - Clear examples in slash commands

❌ **Not Adopted** (and why):

- **MCP server** - Complexity/performance concerns (deferred to future)
- **Daemon mode** - Not needed for batch indexing workloads
- **Agent definitions** - Future consideration
- **Resource URIs** (like `beads://quickstart`) - Simpler markdown docs sufficient for now

**Future Migration Path** (if MCP becomes needed):

1. ✅ Phase 1: CLI-first with `--json` support (this RDR)
2. ⏱️ Phase 2: Implement FastMCP wrapper around CLI (like Beads)
3. ⏱️ Phase 3: Update slash commands to call MCP tools
4. ⏱️ Phase 4: Deprecate direct CLI execution in slash commands

#### 4. CLI Interface Requirements

**For slash commands to work, need CLI interface**:

```bash
# PDF indexing (via centralized CLI from RDR-004)
python -m arcaneum.cli.main index-pdfs /path --collection MyDocs

# Source code indexing (via centralized CLI from RDR-005)
python -m arcaneum.cli.main index-source /path --collection MyCode

# Collection management (via centralized CLI from RDR-003)
python -m arcaneum.cli.main create-collection MyCollection --model stella
python -m arcaneum.cli.main list-collections

# Search (future)
python -m arcaneum.cli.main search "authentication patterns" --collection MyCode
```

**Note**: All commands route through `arcaneum.cli.main` as defined in RDR-001, RDR-003, RDR-004, and RDR-005.

**Key Requirements**:

- All commands route through `python -m arcaneum.cli.main` (centralized CLI dispatcher)
- Accepts CLI arguments (parsed by Click framework from RDR-001)
- Outputs progress to stdout (for Claude monitoring)
- Returns exit codes (0=success, non-zero=error)
- **JSON output mode**: `--json` flag for structured responses (best practice from Beads)
  - Machine-readable format for future MCP integration
  - Example: `{"status": "complete", "files_processed": 47, "chunks_created": 1247}`
  - Human-readable text mode remains default for Claude readability

**Note**: RDR-004 and RDR-005 already implement CLI commands via `src/arcaneum/cli/`, this RDR just exposes them via slash commands

#### 5. Progress Reporting from Slash Commands

**How Claude Monitors Progress**:

1. **Slash command executes CLI tool** via Bash
2. **CLI outputs progress** to stdout:

   ```
   [INFO] Discovering PDF files...
   [INFO] Found 47 PDF files
   [INFO] Processing 1/47 (2.1%): paper1.pdf
   [INFO] Processing 2/47 (4.3%): paper2.pdf
   ...
   [INFO] Complete: 47 files indexed, 1,247 chunks
   ```

3. **Claude reads output** in real-time
4. **Claude reports to user**: "I've indexed 47 PDF files..."

**Requirements** (from arcaneum-49, arcaneum-50):

- TTY detection (plain text when piped)
- Incremental progress updates
- Structured error messages
- Exit codes for success/failure

#### 6. Parameter Passing Patterns

**Slash Command Arguments**:

```markdown
/index-pdfs /Documents/papers --collection Research --model stella
```

**Variable Substitution**:

```markdown
---
argument-hint: <path> --collection <name> [--model <model>]
---

Execute: python -m arcaneum.indexing.pdf $ARGUMENTS
```

**$ARGUMENTS expands to**: `/Documents/papers --collection Research --model stella`

**Special Variables**:
- `${CLAUDE_PLUGIN_ROOT}`: Plugin installation directory
- `${1}`, `${2}`, etc.: Positional arguments
- `$ARGUMENTS`: All arguments as string

## Proposed Solution

### Approach

**Three-Layer Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│           Layer 1: Claude Code Plugin Interface              │
│                                                              │
│  Discovery:                                                  │
│  • .claude-plugin/plugin.json (metadata)                     │
│  • commands/*.md (slash commands)                            │
│                                                              │
│  User Experience:                                            │
│  • /plugin marketplace add yourorg/arcaneum                  │
│  • /plugin install arcaneum                                  │
│  • /help (see available commands)                            │
│  • /index-pdfs /path --collection MyDocs                     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Layer 2: Slash Command Execution (Bash)              │
│                                                              │
│  Slash command markdown:                                     │
│  • Parses arguments                                          │
│  • Executes CLI tool via Bash                                │
│  • Monitors stdout for progress                              │
│  • Reports errors via stderr                                 │
│                                                              │
│  Example:                                                    │
│  ```bash                                                     │
│  python -m arcaneum.indexing.pdf $ARGUMENTS                  │
│```                                                         │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Layer 3: CLI Entry Points (Python)                   │
│                                                              │
│  Modules:                                                    │
│  • arcaneum.indexing.pdf (from RDR-004)                      │
│  • arcaneum.indexing.source_code (from RDR-005)              │
│  • arcaneum.collections (from RDR-003)                       │
│  • arcaneum.search (future)                                  │
│                                                              │
│  Each module:                                                │
│  • Accepts CLI arguments                                     │
│  • Outputs progress to stdout                                │
│  • Returns exit code                                         │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
               Qdrant Docker Service

```

**User Workflow**:

1. **Install Plugin**:
   ```

   User: "Install the arcaneum plugin"
   Claude: Executes /plugin marketplace add yourorg/arcaneum
   Claude: Executes /plugin install arcaneum
   Claude: "Arcaneum plugin installed. Use /help to see available commands."

   ```

2. **Discover Commands**:
   ```

   User: "What indexing commands are available?"
   Claude: Shows output from /help filtered to arcaneum:
     /index-pdfs <path> --collection <name> - Index PDF files
     /index-source <path> --collection <name> - Index source code
     /create-collection <name> --model <model> - Create collection

   ```

3. **Execute Indexing**:
   ```

   User: "Index PDFs in /Documents/papers to Research collection"
   Claude: Executes /index-pdfs /Documents/papers --collection Research
   Claude: Monitors output, reports: "Indexed 47 PDFs, created 1,247 chunks"

   ```

### Technical Design

#### Plugin Configuration Files

**`.claude-plugin/plugin.json`**:

```json
{
  "name": "arcaneum",
  "description": "Semantic search indexing for Qdrant. Index PDFs and source code, create collections, and search with metadata filtering.",
  "version": "0.1.0",
  "author": {
    "name": "Arcaneum Contributors",
    "url": "https://github.com/yourorg/arcaneum"
  },
  "repository": "https://github.com/yourorg/arcaneum",
  "license": "MIT",
  "homepage": "https://github.com/yourorg/arcaneum",
  "keywords": [
    "semantic-search",
    "qdrant",
    "pdf-indexing",
    "code-indexing",
    "vector-database"
  ],
  "commands": [
    "./commands/index-pdfs.md",
    "./commands/index-source.md",
    "./commands/create-collection.md",
    "./commands/list-collections.md",
    "./commands/search.md"
  ]
}
```

**Key**: NO `mcpServers` section - using direct CLI execution

**`.claude-plugin/marketplace.json`**:

```json
{
  "name": "arcaneum-marketplace",
  "description": "Arcaneum semantic search tools",
  "owner": {
    "name": "Arcaneum Contributors",
    "url": "https://github.com/yourorg"
  },
  "plugins": [
    {
      "name": "arcaneum",
      "source": "./",
      "description": "Semantic search indexing for Qdrant",
      "version": "0.1.0"
    }
  ]
}
```

#### Slash Command Examples

**`commands/index-pdfs.md`**:

```markdown
---
description: Index PDF files to Qdrant collection
argument-hint: <path> --collection <name> [options]
---

Index PDF files from a directory to a Qdrant collection for semantic search.

**Arguments:**
- <path>: Directory containing PDF files
- --collection <name>: Target Qdrant collection name (required)
- --model <model>: Embedding model (default: stella)
- --workers <n>: Parallel workers (default: 4)
- --no-ocr: Disable OCR (enabled by default for scanned documents)
- --ocr-language <lang>: OCR language code (default: eng)
- --force: Force reindex all files
- --verbose: Detailed progress output

**Examples:**
/index-pdfs /Documents/papers --collection Research
/index-pdfs /Scans --collection Archive --ocr-language fra
/index-pdfs /Books --collection Library --workers 8 --force

**Execution:**
cd ${CLAUDE_PLUGIN_ROOT}
python -m arcaneum.cli.main index-pdfs $ARGUMENTS

**Note:** This command may take several minutes for large document collections.
Progress will be reported in real-time.
```

**`commands/index-source.md`**:

```markdown
---
description: Index source code to Qdrant collection
argument-hint: <path> --collection <name> [options]
---

Index source code from git repositories to a Qdrant collection with AST-aware chunking.

**Arguments:**
- <path>: Directory containing git repositories
- --collection <name>: Target Qdrant collection name (required)
- --model <model>: Embedding model (default: jina-code)
- --workers <n>: Parallel workers (default: 4)
- --depth <n>: Git discovery depth (default: unlimited)
- --force: Force reindex all projects
- --verbose: Detailed progress output

**Examples:**
/index-source /code/projects --collection MyCode
/index-source ~/repos --collection OpenSource --model jina-code
/index-source . --collection CurrentProject --depth 0

**Execution:**
cd ${CLAUDE_PLUGIN_ROOT}
python -m arcaneum.cli.main index-source $ARGUMENTS

**Note:** Git repositories are indexed from their current branch.
Multi-branch support available.
```

**`commands/create-collection.md`**:

```markdown
---
description: Create a new Qdrant collection
argument-hint: <name> --model <model> [options]
---

Create a new Qdrant collection with specified embedding model and configuration.

**Arguments:**
- <name>: Collection name (required)
- --model <model>: Embedding model (stella, modernbert, bge, jina-code) (required)
- --hnsw-m <n>: HNSW index parameter m (default: 16)
- --hnsw-ef <n>: HNSW index parameter ef_construct (default: 100)
- --on-disk: Store vectors on disk (reduces RAM usage)

**Examples:**
/create-collection Research --model stella
/create-collection LargeArchive --model modernbert --on-disk
/create-collection CodeLibrary --model jina-code --hnsw-m 32

**Execution:**
cd ${CLAUDE_PLUGIN_ROOT}
python -m arcaneum.cli.main create-collection $ARGUMENTS
```

**`commands/list-collections.md`**:

```markdown
---
description: List all Qdrant collections
argument-hint: [--verbose]
---

List all Qdrant collections with their configurations.

**Arguments:**
- --verbose: Show detailed collection information

**Examples:**
/list-collections
/list-collections --verbose

**Execution:**
cd ${CLAUDE_PLUGIN_ROOT}
python -m arcaneum.cli.main list-collections $ARGUMENTS
```

**`commands/search.md`**:

```markdown
---
description: Search Qdrant collection semantically
argument-hint: "<query>" --collection <name> [options]
---

Perform semantic search across a Qdrant collection.

**Arguments:**
- "<query>": Search query (required, use quotes for multi-word queries)
- --collection <name>: Collection to search (required)
- --limit <n>: Number of results (default: 10)
- --filter <json>: Metadata filter as JSON
- --verbose: Show detailed match information

**Examples:**
/arc:find "authentication patterns" --collection MyCode --limit 5
/arc:find "machine learning" --collection Research --filter '{"author": "Smith"}'
/arc:find "error handling" --collection Documentation --verbose

**Execution:**
cd ${CLAUDE_PLUGIN_ROOT}
python -m arcaneum.cli.main search $ARGUMENTS
```

#### CLI Integration via Centralized Dispatcher

**Slash commands route through centralized CLI defined in RDR-001, RDR-003, RDR-004, RDR-005**:

**`src/arcaneum/cli/main.py`** (from RDR-001, extended by RDR-003/004/005):

```python
"""Centralized CLI dispatcher for arcaneum commands"""
import click

@click.group()
def cli():
    """Arcaneum: Semantic search tools for Qdrant"""
    pass

# From RDR-004: PDF indexing command
@cli.command('index-pdfs')
@click.argument('path')
@click.option('--collection', required=True)
@click.option('--model', default='stella')
@click.option('--workers', type=int, default=4)
@click.option('--no-ocr', is_flag=True)
@click.option('--ocr-language', default='eng')
@click.option('--force', is_flag=True)
@click.option('--verbose', '-v', is_flag=True)
def index_pdfs(path, collection, model, workers, ocr_enabled, ocr_language, force, verbose):
    """Index PDF files to Qdrant collection (from RDR-004)"""
    from arcaneum.cli.index_pdfs import index_pdfs_command
    index_pdfs_command(path, collection, model, workers, ocr_enabled, ocr_language, force, verbose)

# From RDR-005: Source code indexing command
@cli.command('index-source')
@click.argument('path')
@click.option('--collection', required=True)
@click.option('--model', default='jina-code')
@click.option('--workers', type=int, default=4)
@click.option('--depth', type=int)
@click.option('--force', is_flag=True)
@click.option('--verbose', '-v', is_flag=True)
def index_source(path, collection, model, workers, depth, force, verbose):
    """Index source code to Qdrant collection (from RDR-005)"""
    from arcaneum.cli.index_source import index_source_command
    index_source_command(path, collection, model, workers, depth, force, verbose)

# From RDR-003: Collection management commands
@cli.command('create-collection')
@click.argument('name')
@click.option('--model', required=True)
@click.option('--hnsw-m', type=int, default=16)
@click.option('--hnsw-ef', type=int, default=100)
@click.option('--on-disk', is_flag=True)
def create_collection(name, model, hnsw_m, hnsw_ef, on_disk):
    """Create Qdrant collection (from RDR-003)"""
    from arcaneum.cli.collections import create_collection_command
    create_collection_command(name, model, hnsw_m, hnsw_ef, on_disk)

@cli.command('list-collections')
@click.option('--verbose', '-v', is_flag=True)
def list_collections(verbose):
    """List all Qdrant collections (from RDR-003)"""
    from arcaneum.cli.collections import list_collections_command
    list_collections_command(verbose)

# Future: Search command (RDR-007)
@cli.command('search')
@click.argument('query')
@click.option('--collection', required=True)
@click.option('--limit', type=int, default=10)
@click.option('--filter', type=str)
@click.option('--verbose', '-v', is_flag=True)
def search(query, collection, limit, filter, verbose):
    """Search Qdrant collection (future RDR-007)"""
    from arcaneum.cli.search import search_command
    search_command(query, collection, limit, filter, verbose)

if __name__ == '__main__':
    cli()
```

**Implementation files already defined in prior RDRs**:

- `src/arcaneum/cli/index_pdfs.py` - PDF indexing logic (RDR-004, line 1566)
- `src/arcaneum/cli/index_source.py` - Source code indexing logic (RDR-005, inferred)
- `src/arcaneum/cli/collections.py` - Collection management logic (RDR-003)
- `src/arcaneum/cli/search.py` - Search logic (future)

#### Installation and Setup

**Prerequisites**:

- Python >= 3.12
- Git
- Dependencies installed: `pip install -r requirements.txt`
- Qdrant Docker running (from RDR-002)

**Plugin Installation**:

```bash
# In Claude Code
/plugin marketplace add yourorg/arcaneum
/plugin install arcaneum

# Verify installation
/help
# Should show /index-pdfs, /index-source, etc.
```

**Workflow Example**:

```
User: "Install arcaneum and index my research papers"

Claude:
1. /plugin marketplace add yourorg/arcaneum
2. /plugin install arcaneum
3. /create-collection Research --model stella
4. /index-pdfs /Documents/papers --collection Research

[Monitors output]
[INFO] Found 47 PDF files
[INFO] Processing 47/47 (100%)
[INFO] Complete: 47 files, 1,247 chunks

Claude: "I've installed arcaneum and indexed 47 research papers to the Research collection."
```

## Alternatives Considered

### Alternative 1: MCP Server as Primary Interface

**Description**: Implement MCP server with structured tools, use slash commands as secondary

**Example MCP Tool**:

```python
@server.tool()
async def index_pdf_files(
    input_path: str,
    collection_name: str,
    model: Literal["stella", "modernbert", "bge", "jina"] = "stella",
    workers: int = 4,
    ocr_enabled: bool = True
) -> dict:
    """Index PDF files to Qdrant collection"""
    # Call CLI internally or implement directly
```

**Plugin JSON with MCP**:

```json
{
  "mcpServers": {
    "arcaneum": {
      "command": "python",
      "args": ["-m", "arcaneum.mcp.server"]
    }
  }
}
```

**Pros**:

- ✅ Structured tool interface with type hints
- ✅ Tools appear in Claude UI automatically
- ✅ Better parameter validation
- ✅ Structured responses

**Cons**:

- ❌ MCP server implementation complexity
- ❌ Server lifecycle management
- ❌ User explicitly prefers NOT using MCP
- ❌ More code to maintain

**Reason for rejection**: User explicitly stated "strong want to not use MCP" unless required

### Alternative 2: Hybrid (Slash Commands + MCP)

**Description**: Provide BOTH slash commands AND MCP tools

**Pros**:

- ✅ Maximum flexibility
- ✅ Users choose preferred interface
- ✅ Best discoverability

**Cons**:

- ❌ Double the maintenance (two interfaces)
- ❌ Complexity without clear benefit
- ❌ User confusion (which to use?)

**Reason for rejection**: Adds complexity without sufficient benefit over slash commands alone

### Alternative 3: No Plugin, Just Documentation

**Description**: Document CLI commands in README, users execute manually via Bash tool

**Pros**:

- ✅ Simplest implementation (no plugin structure)
- ✅ No plugin installation needed

**Cons**:

- ❌ No discoverability (users must read docs)
- ❌ No `/help` integration
- ❌ Not a "plugin" (defeats marketplace purpose)
- ❌ Poor user experience

**Reason for rejection**: Fails core requirement of "Claude Code marketplace plugin"

## Trade-offs and Consequences

### Positive Consequences

1. **Simple Architecture**: Slash commands → Bash → CLI is straightforward (vs Beads' MCP layer)
2. **No MCP Overhead**: Per user preference, avoid MCP complexity and startup latency
3. **Discoverability**: Commands show in `/help`, sufficient for v1 (vs Beads' UI tool list)
4. **Direct Execution**: No intermediate layers, CLI runs directly
5. **Maintainability**: Markdown files easier to maintain than MCP server code
6. **Progress Monitoring**: Claude monitors stdout in real-time (same as Beads' daemon mode)
7. **Reusability**: CLI tools work both in and out of Claude Code
8. **Familiar Patterns**: Standard CLI argument patterns users already know
9. **Performance**: Zero MCP layer overhead for long-running indexing operations
10. **Migration Path**: Easy to add MCP later without breaking existing workflows (learned from Beads)

### Negative Consequences

1. **No Type Hints**: Parameters not validated by Claude (relies on CLI parser)
   - *Mitigation*: Add `--json` flag for structured validation in future MCP wrapper
2. **No UI Tool List**: Tools don't appear in Claude's tool dropdown (like Beads does)
   - *Mitigation*: `/help` discovery adequate for v1, MCP adds this later
3. **Text Parsing**: Claude parses text output (not structured responses like Beads)
   - *Mitigation*: Add `--json` flag for future structured output
4. **Error Handling**: Less structured than MCP error protocol
   - *Mitigation*: Consistent `[ERROR]` format, exit codes (adopted from Beads)
5. **Discovery Limitation**: Requires `/help` command (not automatic UI)
   - *Mitigation*: Clear documentation, onboarding message

### Lessons from Beads

**What We Learned from Beads' MCP-First Approach**:

1. **JSON Output Mode is Essential**: Beads' `--json` flag enables both human and machine readability
   - ✅ **Adopted**: Adding `--json` flag to all CLI commands for future MCP wrapper

2. **Structured Error Messages**: Beads uses consistent error format throughout
   - ✅ **Adopted**: Implement `[ERROR]` prefix and exit codes

3. **Version Compatibility Checking**: Beads validates CLI/server version compatibility
   - ✅ **Adopted**: Add version checking to plugin setup

4. **Clear Migration Path**: Beads shows MCP can coexist with CLI
   - ✅ **Adopted**: Design allows adding MCP wrapper without breaking slash commands

5. **Comprehensive Documentation**: Beads includes resource URIs (like `beads://quickstart`)
   - ⚠️ **Deferred**: Simpler markdown docs sufficient for v1, consider for v2

6. **MCP Overhead is Real**: Beads added daemon mode to mitigate MCP startup cost
   - ✅ **Validated**: CLI-first avoids this complexity for batch workloads

**What We Intentionally Didn't Adopt**:

1. **MCP-First Architecture**: Too complex for initial version
   - *Rationale*: User preference for simplicity, can add later

2. **Daemon Mode**: Not needed for batch indexing
   - *Rationale*: Long-running indexing operations dwarf MCP overhead

3. **Agent Definitions**: More structure than needed initially
   - *Rationale*: Slash commands adequate for discovery, defer to v2

4. **Resource URIs**: More abstraction than needed
   - *Rationale*: Standard markdown documentation is clearer for v1

### Risks and Mitigations

**Risk**: Users don't discover slash commands (don't know about `/help`)
**Mitigation**: Clear README instructions, onboarding message after plugin install

**Risk**: CLI argument parsing errors not caught before execution
**Mitigation**: Add `--help` to all CLI modules, show usage on error

**Risk**: Progress output format changes break Claude's parsing
**Mitigation**: Structure output consistently, document format in RDR

**Risk**: MCP becomes required later (type hints critical)
**Mitigation**: Architecture supports adding MCP wrapper without breaking slash commands

**Risk**: Plugin installation fails (missing dependencies)
**Mitigation**: Include `requirements.txt`, document Python version requirement

## Best Practices Adopted from Beads

After analyzing Beads (<https://github.com/steveyegge/beads>), a production-quality Claude Code plugin, we've identified and adopted several best practices for CLI-first plugin architecture:

### 1. Portable Plugin Paths with `${CLAUDE_PLUGIN_ROOT}`

**Beads Pattern**: All slash commands use `cd ${CLAUDE_PLUGIN_ROOT}` before executing CLI
**Why**: Ensures commands work regardless of user's current directory
**Arcaneum Implementation**: All slash commands include:

```bash
cd ${CLAUDE_PLUGIN_ROOT}
python -m arcaneum.cli.main <command> $ARGUMENTS
```

### 2. JSON Output Mode for Future Compatibility

**Beads Pattern**: CLI supports `--json` flag for structured output
**Example**: `bd ready --json` returns machine-parseable JSON
**Why**: Enables future MCP wrapper without changing CLI implementation
**Arcaneum Implementation**: Add `--json` flag to all CLI commands:

```python
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
def index_pdfs(..., output_json):
    if output_json:
        print(json.dumps({"status": "complete", "files": 47}))
    else:
        print("[INFO] Complete: 47 files indexed")
```

### 3. Structured Error Messages

**Beads Pattern**: Consistent error prefixes and exit codes
**Why**: Makes errors easy to parse and display to users
**Arcaneum Implementation**:

- Prefix format: `[ERROR] <message>`
- Exit codes: 0=success, 1=general error, 2=invalid args, 3=not found
- Example: `[ERROR] Collection 'Research' not found in Qdrant`

### 4. Clear Slash Command Frontmatter

**Beads Pattern**: Comprehensive YAML frontmatter in slash commands
**Why**: Claude uses this to generate help text and argument hints
**Arcaneum Implementation**:

```markdown
---
description: Index PDF files to Qdrant collection
argument-hint: <path> --collection <name> [options]
---
```

### 5. Version Compatibility Checking

**Beads Pattern**: CLI validates version compatibility on startup
**Why**: Prevents subtle bugs from version mismatches
**Arcaneum Implementation**: Add version check to CLI:

```python
# src/arcaneum/cli/main.py
import sys

MIN_PYTHON = (3, 12)
if sys.version_info < MIN_PYTHON:
    print(f"[ERROR] Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required")
    sys.exit(1)
```

### 6. Comprehensive Command Documentation

**Beads Pattern**: Each slash command includes detailed examples and explanations
**Why**: Users can understand usage without leaving Claude Code
**Arcaneum Implementation**: Every slash command markdown includes:

- Purpose description
- Full argument list with explanations
- Multiple usage examples
- Expected output description

### 7. Argument Expansion with `$ARGUMENTS`

**Beads Pattern**: Use `$ARGUMENTS` for flexible parameter passing
**Why**: Allows users to pass arbitrary CLI flags without updating slash commands
**Arcaneum Implementation**: All commands use `$ARGUMENTS`:

```bash
python -m arcaneum.cli.main index-pdfs $ARGUMENTS
# Expands to all user-provided arguments
```

### Practices Intentionally NOT Adopted

These Beads patterns were considered but not adopted for v1:

1. **MCP Server Layer**
   - **Beads**: FastMCP server wraps CLI
   - **Arcaneum**: Direct CLI execution for simplicity
   - **Future**: Can add MCP in v2 without breaking v1

2. **Daemon Mode**
   - **Beads**: Optional daemon for performance
   - **Arcaneum**: Not needed for batch indexing workloads
   - **Rationale**: Long operations make startup overhead negligible

3. **Resource URIs** (like `beads://quickstart`)
   - **Beads**: Custom resource protocol for docs
   - **Arcaneum**: Standard markdown documentation
   - **Rationale**: Simpler approach sufficient for v1

4. **Agent Definitions**
   - **Beads**: Defines custom agents for workflows
   - **Arcaneum**: Standard slash commands sufficient
   - **Future**: Consider for complex multi-step workflows

### Implementation Checklist

Based on Beads best practices, ensure:

- [x] All slash commands use `${CLAUDE_PLUGIN_ROOT}`
- [ ] Add `--json` flag to all CLI commands
- [ ] Implement consistent `[ERROR]` prefix format
- [ ] Define exit code conventions (0, 1, 2, 3)
- [ ] Add Python version checking to CLI main
- [ ] Verify all slash commands have complete frontmatter
- [ ] Add usage examples to all slash command docs
- [ ] Test `$ARGUMENTS` expansion with various flag combinations
- [ ] Document error messages and exit codes in RDR

## Implementation Plan

### Prerequisites

- [x] RDR-001: Project structure (completed)
- [x] RDR-002: Qdrant server setup (completed)
- [x] RDR-003: Collection management (completed)
- [x] RDR-004: PDF bulk indexing (completed)
- [x] RDR-005: Source code indexing (completed)
- [ ] Python >= 3.12 installed
- [ ] Git repository set up

### Step-by-Step Implementation

#### Step 1: Create Plugin Configuration Files

Create `.claude-plugin/` directory structure:

- Create `.claude-plugin/plugin.json` (metadata)
- Create `.claude-plugin/marketplace.json` (marketplace config)
- Verify JSON syntax is valid
- Test: Plugin structure recognized by Claude Code

**Estimated effort**: 1 day

#### Step 2: Create Slash Command Files

Create `commands/` directory with markdown files:

- Create `commands/index-pdfs.md`
- Create `commands/index-source.md`
- Create `commands/create-collection.md`
- Create `commands/list-collections.md`
- Create `commands/search.md`
- Each file includes:
  - Frontmatter (description, argument-hint)
  - Documentation
  - Bash execution block
- Test: Slash commands appear in `/help`

**Estimated effort**: 2 days

#### Step 3: Verify CLI Commands from Prior RDRs

Verify existing CLI commands work as expected:

- Test `python -m arcaneum.cli.main index-pdfs --help` (from RDR-004)
- Test `python -m arcaneum.cli.main index-source --help` (from RDR-005, if implemented)
- Test `python -m arcaneum.cli.main create-collection --help` (from RDR-003)
- Test `python -m arcaneum.cli.main list-collections --help` (from RDR-003)
- Verify all output progress to stdout in parseable format
- No new CLI files needed (reuse existing from RDR-003/004/005)

**Estimated effort**: 1 day (verification only, no new code)

#### Step 4: Add JSON Output Mode and Error Formatting

Implement best practices from Beads analysis:

**4a. JSON Output Mode**:

- Add `--json` flag to all CLI commands
- Implement dual output: human-readable (default) and JSON mode
- Example output structure:

  ```json
  {
    "status": "success",
    "files_processed": 47,
    "chunks_created": 1247,
    "errors": []
  }
  ```

- Test JSON parsing for future MCP integration

**4b. Structured Error Messages**:

- Implement consistent `[ERROR]` prefix format
- Define exit code conventions:
  - 0 = success
  - 1 = general error
  - 2 = invalid arguments
  - 3 = resource not found (collection, file path)
- Add error examples to documentation

**4c. Progress Output Formatting**:

- Implement consistent `[INFO]` prefixes for progress
- Add progress percentage: `[INFO] Processing 10/100 (10%)`
- Add final summary: `[INFO] Complete: X files, Y chunks`
- Handle TTY vs non-TTY (auto-detect)
- Test: Output parseable by Claude

**4d. Version Checking**:

- Add Python version check to CLI main
- Validate minimum Python 3.12
- Check required dependencies on startup

**Estimated effort**: 3 days (increased from 2 days for additional features)

#### Step 5: Integration Testing

Test complete workflow via Claude Code:

- Install plugin: `/plugin marketplace add ...`
- Verify commands: `/help`
- Execute indexing: `/index-pdfs /test-data --collection Test`
- Monitor Claude's output interpretation
- Test error handling
- Test concurrent operations

**Estimated effort**: 2 days

#### Step 6: Documentation

Create comprehensive documentation:

- Update README with plugin installation
- Create `docs/plugin-usage.md` with examples
- Document each slash command
- Add troubleshooting guide
- Create video/GIF demos if possible

**Estimated effort**: 2 days

#### Step 7: MCP Evaluation (Optional)

Document MCP wrapper design for future:

- Create `docs/mcp-wrapper-design.md`
- Document when MCP would be beneficial
- Provide example MCP server structure
- Note: Implementation deferred until needed

**Estimated effort**: 1 day

### Files to Create

**Plugin Configuration**:

- `.claude-plugin/plugin.json` - Plugin metadata
- `.claude-plugin/marketplace.json` - Marketplace config

**Slash Commands**:

- `commands/index-pdfs.md` - PDF indexing command
- `commands/index-source.md` - Source code indexing command
- `commands/create-collection.md` - Collection creation command
- `commands/list-collections.md` - List collections command
- `commands/search.md` - Search command

**CLI Integration** (extends existing files from prior RDRs):

- `src/arcaneum/cli/main.py` - Central dispatcher, add:
  - Version checking (Python >= 3.12)
  - Exit code conventions (0, 1, 2, 3)
  - Extends RDR-001/003/004/005
- `src/arcaneum/cli/index_pdfs.py` - Add `--json` flag and structured errors (extends RDR-004)
- `src/arcaneum/cli/index_source.py` - Add `--json` flag and structured errors (extends RDR-005)
- `src/arcaneum/cli/collections.py` - Add `--json` flag and structured errors (extends RDR-003)
- No new files needed - enhances existing CLI from prior RDRs

**Documentation**:

- `docs/plugin-usage.md` - Plugin usage guide
- `docs/cli-output-format.md` - JSON output schema and error codes
- `docs/mcp-wrapper-design.md` - Future MCP design (optional)

**Tests**:

- `tests/plugin/test_slash_commands.py` - Slash command tests
- `tests/cli/test_main.py` - CLI entry point tests

### Dependencies

No new dependencies beyond RDR-004 and RDR-005 requirements.

## Validation

### Testing Approach

**Plugin Installation Tests**:

- Verify `.claude-plugin/plugin.json` valid JSON
- Test plugin installation via `/plugin marketplace add`
- Verify slash commands appear in `/help`

**Slash Command Tests**:

- Execute each slash command with valid arguments
- Test argument parsing and validation
- Verify CLI execution occurs
- Monitor progress output

**Integration Tests**:

- Complete workflow: install → create collection → index → search
- Test concurrent indexing operations
- Test error handling (invalid paths, missing collections)
- Verify Claude interprets output correctly

### Test Scenarios

**Scenario 1: First-Time Plugin Installation**

- **Setup**: Fresh Claude Code installation
- **Action**: User executes `/plugin marketplace add yourorg/arcaneum`
- **Expected**: Plugin cloned, slash commands available in `/help`

**Scenario 2: PDF Indexing via Slash Command**

- **Setup**: Plugin installed, Qdrant running, collection exists
- **Action**: User executes `/index-pdfs /Documents/papers --collection Research`
- **Expected**: CLI runs, progress shown, files indexed, Claude reports results

**Scenario 3: Error Handling**

- **Setup**: Invalid collection name provided
- **Action**: Execute `/index-pdfs /path --collection NonExistent`
- **Expected**: CLI exits with error code, stderr message shown, Claude reports error

**Scenario 4: Concurrent Indexing**

- **Setup**: Two separate directories to index
- **Action**: Execute `/index-pdfs /docs1 --collection C1` and `/index-source /code --collection C2` simultaneously
- **Expected**: Both operations complete successfully without conflicts

**Scenario 5: Help and Discovery**

- **Setup**: Plugin installed
- **Action**: User asks "What can I do with arcaneum?"
- **Expected**: Claude shows `/help` output filtered to arcaneum commands

### Performance Validation

**Metrics**:

- Plugin installation time (< 30 seconds)
- Slash command invocation overhead (< 1 second)
- Progress update frequency (every 1-2 seconds minimum)

## Future Enhancements

### Optional MCP Server Wrapper (Future RDR)

**When to Add MCP**:

- If type hints become critical for complex parameter validation
- If tool UI discovery is needed (beyond `/help`)
- If structured responses are required for programmatic use
- If user requests MCP integration for better Claude UI integration

**Implementation Pattern (Based on Beads)**:

Following the production pattern from Beads, implement a **FastMCP wrapper** that calls the CLI:

```python
# src/arcaneum/mcp/server.py
"""FastMCP server wrapping Arcaneum CLI tools"""
import json
import subprocess
from typing import Literal
from fastmcp import FastMCP

mcp = FastMCP("arcaneum")

@mcp.tool()
async def index_pdf_files(
    path: str,
    collection: str,
    model: Literal["stella", "modernbert", "bge", "jina-code"] = "stella",
    workers: int = 4,
    ocr_enabled: bool = True,
    ocr_language: str = "eng",
    force: bool = False
) -> dict:
    """Index PDF files to Qdrant collection.

    Args:
        path: Directory containing PDF files
        collection: Target Qdrant collection name
        model: Embedding model to use
        workers: Number of parallel workers
        ocr_enabled: Enable OCR for scanned documents
        ocr_language: OCR language code (default: eng)
        force: Force reindex all files

    Returns:
        dict with keys:
            - status: "success" or "error"
            - files_processed: Number of files indexed
            - chunks_created: Number of chunks created
            - errors: List of error messages (if any)
    """
    # Build CLI command
    cmd = [
        "python", "-m", "arcaneum.cli.main", "index-pdfs",
        path,
        "--collection", collection,
        "--model", model,
        "--workers", str(workers),
        "--json"  # Request JSON output
    ]

    if not ocr_enabled:
        cmd.append("--no-ocr")
    if ocr_language != 'eng':
        cmd.extend(["--ocr-language", ocr_language])
    if force:
        cmd.append("--force")

    # Execute CLI, capture JSON output
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        return json.loads(result.stdout)
    else:
        return {
            "status": "error",
            "message": result.stderr
        }

@mcp.tool()
async def index_source_code(
    path: str,
    collection: str,
    model: Literal["jina-code", "stella", "modernbert"] = "jina-code",
    workers: int = 4,
    depth: int | None = None,
    force: bool = False
) -> dict:
    """Index source code to Qdrant collection with AST-aware chunking."""
    # Similar pattern to index_pdf_files
    ...

@mcp.tool()
async def create_collection(
    name: str,
    model: Literal["stella", "modernbert", "bge", "jina-code"],
    hnsw_m: int = 16,
    hnsw_ef: int = 100,
    on_disk: bool = False
) -> dict:
    """Create new Qdrant collection with specified embedding model."""
    ...

@mcp.tool()
async def list_collections(verbose: bool = False) -> dict:
    """List all Qdrant collections with their configurations."""
    ...

@mcp.tool()
async def search(
    query: str,
    collection: str,
    limit: int = 10,
    filter: str | None = None
) -> dict:
    """Perform semantic search across a Qdrant collection."""
    ...

if __name__ == "__main__":
    mcp.run()
```

**Plugin Configuration Update**:

```json
{
  "name": "arcaneum",
  "mcpServers": {
    "arcaneum": {
      "command": "python",
      "args": ["-m", "arcaneum.mcp.server"]
    }
  },
  "commands": [
    "./commands/index-pdfs.md",
    "./commands/index-source.md",
    ...
  ]
}
```

**Slash Command Update** (hybrid approach):

```markdown
---
description: Index PDF files to Qdrant collection
argument-hint: <path> --collection <name> [options]
---

Index PDF files for semantic search.

**Option 1: Via MCP Tool (Recommended)**
Use the MCP `index_pdf_files` tool for structured parameters and validation.

**Option 2: Via CLI (Direct)**
cd ${CLAUDE_PLUGIN_ROOT}
python -m arcaneum.cli.main index-pdfs $ARGUMENTS
```

**Migration Path**:
1. ✅ **Phase 1** (Current): CLI-first with `--json` support
   - Slash commands call CLI directly
   - Add `--json` flag to all commands
   - Maintain backward compatibility

2. ⏱️ **Phase 2**: Implement FastMCP wrapper
   - Create `src/arcaneum/mcp/server.py`
   - MCP tools wrap CLI with `--json` flag
   - Add `mcpServers` to plugin.json
   - Test tool discovery in Claude UI

3. ⏱️ **Phase 3**: Update slash commands to hybrid mode
   - Recommend MCP tools as primary
   - Keep CLI execution as fallback
   - Update documentation

4. ⏱️ **Phase 4**: Optional CLI deprecation
   - Monitor usage patterns
   - If MCP preferred, simplify slash commands
   - Or maintain both indefinitely (user choice)

**Benefits of Hybrid Approach**:
- ✅ Users choose their preferred interface
- ✅ MCP provides type hints and UI discovery
- ✅ CLI remains available for scripting/automation
- ✅ No breaking changes to existing workflows
- ✅ Gradual migration path based on user feedback

### Advanced Features

1. **Web UI**: Browser-based collection management
2. **Real-time Updates**: WebSocket progress streaming
3. **Hybrid Search**: Integration with full-text engines
4. **Multi-Collection Search**: Search across multiple collections
5. **Scheduled Indexing**: Cron-like scheduled reindexing

## References

- [Beads Issues arcaneum-6, arcaneum-46 to arcaneum-51](../../.beads/arcaneum.db) - Research findings
- [RDR-001: Project Structure](RDR-001-project-structure.md) - Plugin structure foundation
- [RDR-002: Qdrant Server Setup](RDR-002-qdrant-server-setup.md) - Docker setup
- [RDR-003: Collection Creation](RDR-003-collection-creation.md) - Collection management
- [RDR-004: PDF Bulk Indexing](RDR-004-pdf-bulk-indexing.md) - PDF indexing pipeline
- [RDR-005: Source Code Indexing](RDR-005-source-code-indexing.md) - Source code pipeline
- [Claude Code Plugin Documentation](https://docs.claude.com/en/docs/claude-code/plugins) - Official plugin guide
- [Claude Code Slash Commands](https://docs.claude.com/en/docs/claude-code/slash-commands) - Slash command reference
- [Beads Plugin Example](https://github.com/steveyegge/beads) - Real-world plugin reference

## Notes

**Key Design Decisions**:
1. **CLI-First over MCP-First**: Per user preference, use direct CLI execution (vs Beads' MCP-first)
   - *Rationale*: Simplicity and performance for batch operations
   - *Trade-off*: Defer type hints and UI tool discovery to future MCP wrapper
2. **Discoverability via `/help`**: Slash commands show in help, sufficient for v1
   - *Future*: MCP adds UI tool list when needed
3. **Centralized CLI Dispatcher**: All commands route through `arcaneum.cli.main` (from RDR-001)
   - *Consistency*: Single entry point for all operations
4. **JSON Output Mode**: Add `--json` flag (best practice from Beads)
   - *Purpose*: Enables future MCP wrapper without changing CLI implementation
   - *Dual Mode*: Human-readable default, JSON for machines
5. **Structured Error Messages**: Consistent `[ERROR]` prefix and exit codes (adopted from Beads)
   - *Format*: `[ERROR] <message>` with meaningful exit codes
6. **MCP Deferred**: Can add FastMCP wrapper later without breaking slash commands
   - *Migration Path*: Phase 1 (CLI-first) → Phase 2 (Add MCP) → Phase 3 (Hybrid) → Phase 4 (Optional deprecation)

**Implementation Priority**:
1. Plugin configuration files (foundation)
2. Slash command markdown files (user interface)
3. CLI entry points (execution layer)
4. Progress output formatting (monitoring)
5. Integration testing (validation)
6. Documentation (user onboarding)

**Claude Code Integration Notes**:
- Slash commands execute via Bash tool (transparent to user)
- Claude monitors stdout for progress
- Exit codes signal success/failure
- `${CLAUDE_PLUGIN_ROOT}` resolves to plugin directory
- `$ARGUMENTS` expands to all command arguments

**Development Timeline**:
- Total estimated effort: 14 days (increased from 13 days for JSON output and error formatting)
- Critical path: Config → Commands → JSON/Error formatting → Testing
- Parallel work possible: Documentation can start early
- Beads best practices: Add 1 day for implementing JSON output, version checking, error formatting

**User Experience Goals**:
- One-command plugin installation
- Discoverable slash commands via `/help`
- Natural language interaction ("Index my PDFs to Research")
- Real-time progress feedback
- Clear error messages

**Future Considerations**:
- MCP wrapper can be added without breaking slash commands
- Hybrid approach (slash commands + MCP) possible
- Web UI for advanced users
- Marketplace expansion with additional tools

# Recommendation 001: Claude Code Marketplace Base Structure

## Metadata
- **Date**: 2025-10-18
- **Status**: Recommendation
- **Type**: Architecture
- **Priority**: High
- **Related Issues**: arcaneum-1
- **Related Tests**: Initial project structure tests

## Problem Statement

Establish the foundational structure that supports:
1. **Python CLI tools** for Docker management, indexing, and search
2. **Claude Code plugin marketplace** with slash commands
3. **Colocation** of development and distribution in a single repository

This RDR focuses on the minimal structure needed to support both CLI tool development and Claude Code plugin distribution.

This RDR deliberately does NOT define:
- Specific CLI tool implementations
- Indexing algorithms
- Search strategies
- Docker configuration details

Those decisions will be made in subsequent RDRs.

## Context

### Background

Arcaneum will provide tools for semantic search with this architecture:

**Core Architecture**:
- **CLI Tools**: Fast Python CLI tools for all operations
- **Slash Commands**: Markdown files that invoke CLI tools
- **No MCP Servers**: Avoid MCP overhead, use direct CLI calls
- **Local Docker**: Qdrant and full-text engines run in Docker with mounted volumes
- **Dual Indexing**: Eventually index to both Qdrant (semantic) and full-text engines

**Installation Method**:
- Users clone the repository: `git clone https://github.com/yourorg/arcaneum`
- Run Docker scripts and CLI tools directly from the repo
- Claude Code plugin installation: `/plugin marketplace add yourorg/arcaneum`

**Note**: PyPI packaging and pip installation deferred to future RDR.

### Technical Environment

- **Python**: >= 3.12
- **Distribution**: GitHub (git clone + Claude plugin marketplace)
- **Docker**: Local instances with volume mounts
- **Packaging**: Deferred to future RDR

## Research Findings

### Investigation Process

1. **Claude Code Plugin Documentation**
   - URL: https://docs.claude.com/en/docs/claude-code/plugins
   - Focus: Plugin structure, marketplace, slash commands

2. **Beads Plugin Analysis**
   - URL: https://github.com/steveyegge/beads
   - Focus: Real-world example of `.claude-plugin/` structure

3. **Python Packaging Standards (2025)**
   - URL: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
   - Focus: Entry points and CLI tool registration

### Key Discoveries

**`.claude-plugin/` IS a Standard Convention**:
- Used by Claude Code to discover plugin metadata
- Located at repository root
- Contains `plugin.json` (metadata) and `marketplace.json` (catalog)
- Optional `agents/` subdirectory for specialized agents

**Plugin Installation Flow**:
1. User: `/plugin marketplace add yourorg/arcaneum`
2. Claude Code clones repo to `~/.claude/plugins/marketplaces/arcaneum/`
3. Claude Code reads `.claude-plugin/plugin.json`
4. User: `/plugin install arcaneum`
5. Claude Code registers slash commands from `commands/*.md` files

**Slash Commands Call CLI Tools**:
From beads example:
```markdown
---
description: Initialize beads in the current project
argument-hint: [prefix]
---

Initialize beads issue tracking in the current directory.
Use the beads MCP `init` tool...
```

For arcaneum, slash commands will execute CLI tools instead:
```markdown
---
description: Start Qdrant Docker container
argument-hint: [--port PORT]
---

Start a Qdrant instance in Docker.
Execute: `arcaneum docker start qdrant --port ${1:-6333}`
```

**${CLAUDE_PLUGIN_ROOT}** Variable:
- Expands to `~/.claude/plugins/marketplaces/<marketplace-name>/`
- Useful for referencing plugin-local resources
- Can be used in slash commands to reference scripts in the cloned repo

**Python Module Execution**:
Scripts run via Python module syntax:
```bash
python -m arcaneum.cli.main
```

Entry point setup (pip/PyPI) deferred to future RDR.

## Proposed Solution

### Approach

Create a **single repository** that serves as:
1. **Development workspace** (users git clone and run directly)
2. **Claude Code plugin marketplace** (installable via `/plugin marketplace add`)

**Directory Structure** supports:
- Python CLI scripts (`src/arcaneum/`)
- Docker management scripts (future)
- Claude Code slash commands (`commands/`)
- Plugin marketplace metadata (`.claude-plugin/`)
- Documentation (`doc/rdr/`)

**Architecture Principles**:
- **Git Clone Workflow**: Users clone repo, run scripts directly
- **CLI First**: All functionality as CLI tools/scripts
- **Slash Commands**: Thin wrappers that call CLI tools
- **No MCP Servers**: Avoid overhead, use direct process execution
- **Local Docker**: Databases run locally with volume persistence
- **No Packaging Yet**: Defer pip/PyPI to future RDR

### Technical Design

#### Directory Structure

```
arcaneum/                        # GitHub repo (yourorg/arcaneum)
├── .claude-plugin/              # ✅ Claude Code plugin metadata
│   ├── plugin.json              # Plugin info (NO mcpServers section)
│   ├── marketplace.json         # Marketplace catalog
│   └── agents/                  # Optional: specialized agents (future)
│
├── commands/                    # ✅ Slash commands (*.md files)
│   ├── docker-start.md          # /arc-docker-start
│   ├── docker-stop.md           # /arc-docker-stop
│   ├── index-pdfs.md            # /arc-index-pdfs
│   ├── index-code.md            # /arc-index-code
│   └── search.md                # /arc-search
│
├── src/                         # ✅ Python CLI scripts
│   └── arcaneum/
│       ├── __init__.py          # Version constant
│       ├── cli/                 # CLI scripts
│       │   ├── __init__.py
│       │   ├── main.py          # Main CLI dispatcher
│       │   ├── docker.py        # Docker subcommands (future)
│       │   ├── index.py         # Indexing subcommands (future)
│       │   └── search.py        # Search subcommands (future)
│       ├── docker/              # Docker management logic (future)
│       ├── indexing/            # Indexing logic (future)
│       └── search/              # Search logic (future)
│
├── README.md                    # Setup and usage instructions
├── LICENSE                      # MIT
├── .gitignore
│
└── doc/
    └── rdr/                     # RDR planning documents
        ├── README.md
        └── RDR-001-project-structure.md
```

**Note**: Scripts are run directly from the repository (e.g., `python -m arcaneum.cli.main`). Packaging deferred to future RDR.

#### Minimal .claude-plugin/plugin.json

```json
{
  "name": "arcaneum",
  "description": "CLI tools for semantic search across Qdrant and full-text engines. Manage Docker instances, index PDFs and source code, and search with metadata filtering.",
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
    "full-text-search",
    "docker",
    "indexing",
    "cli-tools"
  ]
}
```

**Key: NO `mcpServers` section** - we're not providing MCP servers.

#### Minimal .claude-plugin/marketplace.json

```json
{
  "name": "arcaneum-marketplace",
  "description": "Arcaneum semantic search tools marketplace",
  "owner": {
    "name": "Arcaneum Contributors",
    "url": "https://github.com/yourorg"
  },
  "plugins": [
    {
      "name": "arcaneum",
      "source": "./",
      "description": "CLI tools for semantic search with Qdrant and full-text engines",
      "version": "0.1.0"
    }
  ]
}
```

**Note**: `"source": "./"` means the plugin is at the repo root.

#### Example Slash Command: commands/docker-start.md

```markdown
---
description: Start Qdrant Docker container
argument-hint: [--port PORT] [--storage PATH]
---

Start a Qdrant instance in Docker with optional port and storage path.

Arguments:
- `--port`: Port to expose (default: 6333)
- `--storage`: Local directory for volume mount (default: ./qdrant_storage)

Execute the CLI script:
```bash
cd ${CLAUDE_PLUGIN_ROOT}
python -m arcaneum.cli.docker start qdrant --port $PORT --storage $STORAGE_PATH
```

After starting:
1. Show the container status
2. Show the connection URL (http://localhost:<port>)
3. Suggest creating a collection with `/arc-collection-create`
```

**Key**: Slash command invokes Python module from cloned repo via `${CLAUDE_PLUGIN_ROOT}`.

#### requirements.txt (Future)

```
# Minimal dependencies - add as features are implemented
# docker
# qdrant-client
# etc.
```

**Note**: Dependency management deferred to future RDR. Start with no external dependencies.

#### Minimal CLI Implementation

`src/arcaneum/__init__.py`:
```python
"""Arcaneum: CLI tools for semantic search."""

__version__ = "0.1.0"
```

`src/arcaneum/cli/__init__.py`:
```python
"""CLI module."""
```

`src/arcaneum/cli/main.py`:
```python
"""Main CLI entry point for Arcaneum."""

import sys
from arcaneum import __version__

def main():
    """Main CLI entry point."""
    print(f"Arcaneum v{__version__}")
    print("CLI tools for semantic search")
    print()
    print("Subcommands:")
    print("  docker   - Manage Docker containers (Qdrant, MeiliSearch)")
    print("  index    - Index PDFs and source code")
    print("  search   - Semantic search across collections")
    print()
    print("Use 'arcaneum <subcommand> --help' for more information")
    print()
    print("For Claude Code integration:")
    print("  /plugin marketplace add yourorg/arcaneum")
    print("  /plugin install arcaneum")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

#### README.md Structure

```markdown
# Arcaneum

CLI tools for semantic search across Qdrant and MeiliSearch.

## Features

- **Docker Management**: Start/stop Qdrant and MeiliSearch containers
- **Dual Indexing**: Index to both Qdrant (semantic) and MeiliSearch (full-text)
- **PDF Indexing**: OCR support for image PDFs
- **Code Indexing**: Git-aware with AST chunking
- **Semantic Search**: Query with metadata filtering

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourorg/arcaneum
cd arcaneum
```

### 2. Prerequisites

- Python 3.12+
- Docker installed and running

### 3. Run CLI Tools

```bash
python -m arcaneum.cli.main
```

### 4. Claude Code Plugin (Optional)

Inside Claude Code:
```
/plugin marketplace add yourorg/arcaneum
/plugin install arcaneum
```

Available slash commands:
- `/arc-docker-start` - Start Docker containers
- `/arc-docker-stop` - Stop Docker containers
- `/arc-index-pdfs` - Index PDF files
- `/arc-index-code` - Index source code
- `/arc-search` - Semantic search

## Quick Start

### 1. Start Qdrant

```bash
python -m arcaneum.cli.docker start qdrant
```

Or in Claude Code:
```
/arc-docker-start
```

### 2. Index Documents (Future)

```bash
python -m arcaneum.cli.index pdfs /path/to/pdfs --collection my-docs
```

### 3. Search (Future)

```bash
python -m arcaneum.cli.search "authentication patterns" --collection my-docs
```

## Architecture

- **Git Clone Workflow**: Clone repo, run scripts directly
- **CLI First**: All operations as Python scripts
- **Slash Commands**: Thin wrappers calling scripts via ${CLAUDE_PLUGIN_ROOT}
- **Local Docker**: Databases run locally with volume persistence
- **No MCP Servers**: Direct CLI execution for performance
- **Packaging Later**: PyPI/pip deferred to future RDR

## Development

See [doc/rdr/README.md](doc/rdr/README.md) for RDR-based planning process.

## License

MIT
```

### Implementation Example

The minimal implementation provides:
- Python scripts runnable via `python -m arcaneum.cli.main`
- Claude Code plugin installable via marketplace
- Placeholder CLI showing architecture
- Empty slash commands directory (add as features develop)
- Plugin metadata for discovery
- No packaging complexity (deferred)

## Alternatives Considered

### Alternative 1: MCP Servers Instead of CLI

**Description**: Provide MCP servers for Docker, indexing, search

**Pros**:
- Direct integration with Claude Code
- Richer tool schemas

**Cons**:
- **Slower**: MCP adds overhead
- More complex to develop and debug
- User explicitly wants CLI-first approach

**Reason for rejection**: Performance penalty not acceptable, CLI preferred.

### Alternative 2: Separate Repos for CLI and Plugin

**Description**: CLI in one repo, plugin marketplace in another

**Pros**:
- Clear separation of concerns
- Independent versioning

**Cons**:
- Duplication of documentation
- Users must install from two places
- Harder to keep in sync
- More repos to maintain

**Reason for rejection**: Colocation simpler, single source of truth.

### Alternative 3: Long Command Names (/arcaneum-*)

**Description**: Use full name prefix instead of `arc-`

**Pros**:
- More explicit
- No ambiguity

**Cons**:
- Verbose to type
- User prefers shorter names

**Reason for rejection**: `arc-` is shorter, memorable, unambiguous.

## Trade-offs and Consequences

### Positive Consequences

- **Fast Execution**: CLI tools avoid MCP overhead
- **Flexible**: Users can call CLI directly or via slash commands
- **Simple Development**: Standard Python CLI patterns
- **Colocation**: Single repo for everything
- **Local Docker**: No remote dependencies, reproducible environments

### Negative Consequences

- **Slash Commands Limited**: Can't provide rich schemas like MCP tools
- **Process Overhead**: Each slash command spawns a Python process
- **No MCP Benefits**: Can't use MCP features (streaming, cancellation)

### Risks and Mitigations

- **Risk**: CLI tools not in PATH when slash commands run
  **Mitigation**: README emphasizes pip install before plugin install

- **Risk**: Slash commands become complex bash scripts
  **Mitigation**: Keep commands simple, push logic to CLI tools

- **Risk**: Docker volumes permissions issues
  **Mitigation**: Document volume mount patterns, provide troubleshooting guide

## Implementation Plan

### Prerequisites

- [ ] Python 3.12+ installed
- [ ] uv >= 0.4.10 installed (optional but recommended)
- [ ] Git repository initialized
- [ ] Docker installed (for Docker features)

### Step-by-Step Implementation

#### Step 1: Create Directory Structure

```bash
mkdir -p .claude-plugin
mkdir -p commands
mkdir -p src/arcaneum/cli
mkdir -p doc/rdr
```

**Note**: Working in existing `arcaneum/` directory.

#### Step 2: Create .claude-plugin Files

1. `.claude-plugin/plugin.json` - Copy from Technical Design
2. `.claude-plugin/marketplace.json` - Copy from Technical Design

#### Step 3: Create Placeholder Slash Commands

Create empty files (content defined in future RDRs):
```bash
touch commands/docker-start.md
touch commands/docker-stop.md
touch commands/index-pdfs.md
touch commands/index-code.md
touch commands/search.md
```

Or create one example:

`commands/docker-start.md`:
```markdown
---
description: Start Qdrant Docker container
argument-hint: [--port PORT]
---

Start a Qdrant instance in Docker.

Execute: `arcaneum docker start qdrant --port ${1:-6333}`

(Full implementation defined in RDR-002)
```

#### Step 4: Create Python Package Files

1. `src/arcaneum/__init__.py` - Copy from Technical Design
2. `src/arcaneum/cli/__init__.py` - Empty or docstring
3. `src/arcaneum/cli/main.py` - Copy from Technical Design

#### Step 5: Create README.md

Copy from Technical Design section, adjust URLs as needed.

#### Step 6: Create .gitignore

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.DS_Store

# Docker volumes
qdrant_storage/
meilisearch_data/
```

#### Step 7: Create LICENSE

```
MIT License

Copyright (c) 2025 Arcaneum Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

#### Step 8: Test Python Scripts

```bash
# Test CLI script
python -m arcaneum.cli.main
```

Expected output:
```
Arcaneum v0.1.0
CLI tools for semantic search
...
```

#### Step 9: Test Claude Code Plugin (Local)

```bash
# Push to GitHub (or use local path for testing)
git init
git add .
git commit -m "Initial arcaneum structure"
# git remote add origin <your-repo>
# git push -u origin main
```

In Claude Code:
```
/plugin marketplace add yourorg/arcaneum
/plugin install arcaneum
/help
```

Verify `/arc-*` commands appear.

Test a command:
```
/arc-docker-start
```

Should execute slash command (even if placeholder).

### Files to Create

**Plugin Metadata**:
- `.claude-plugin/plugin.json`
- `.claude-plugin/marketplace.json`

**Slash Commands** (placeholders):
- `commands/docker-start.md`
- `commands/docker-stop.md`
- `commands/index-pdfs.md`
- `commands/index-code.md`
- `commands/search.md`

**Python Package**:
- `src/arcaneum/__init__.py`
- `src/arcaneum/cli/__init__.py`
- `src/arcaneum/cli/main.py`

**Project Files**:
- `README.md`
- `LICENSE`
- `.gitignore`

**Documentation** (already exists):
- `doc/rdr/README.md`
- `doc/rdr/RDR-001-project-structure.md`

### Dependencies

**Required** (initially):
- None (add as features are implemented)

**Future** (to be added in feature RDRs):
- docker (Python Docker SDK)
- qdrant-client
- Additional search engines (MeiliSearch, etc.)

**Development**:
- pytest >= 8.3.5
- black >= 24.0.0
- ruff >= 0.8.0

## Validation

### Testing Approach

Manual testing for initial structure.

### Test Scenarios

1. **Scenario**: Run `python -m arcaneum.cli.main`
   **Expected Result**: Shows help message, no errors

2. **Scenario**: Run `python -c "import arcaneum; print(arcaneum.__version__)"`
   **Expected Result**: Prints "0.1.0"

3. **Scenario**: Check `.claude-plugin/` files exist
   **Expected Result**: `plugin.json` and `marketplace.json` are valid JSON

4. **Scenario**: Add marketplace to Claude Code: `/plugin marketplace add yourorg/arcaneum`
   **Expected Result**: Marketplace added successfully

5. **Scenario**: Install plugin: `/plugin install arcaneum`
   **Expected Result**: Plugin installed, `/help` shows `/arc-*` commands

6. **Scenario**: Run `/arc-docker-start`
   **Expected Result**: Command executes (even if placeholder)

## References

- **Claude Code Plugins**: https://docs.claude.com/en/docs/claude-code/plugins
  - Plugin structure and marketplace
  - Creating slash commands
  - Plugin installation and discovery

- **Beads Plugin Example**: https://github.com/steveyegge/beads
  - Real-world `.claude-plugin/` structure
  - Slash command patterns (markdown with YAML frontmatter)
  - CLI tool integration

- **Python Packaging Guide**: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
  - pyproject.toml format
  - [project.scripts] entry points
  - Package metadata standards

- **uv Documentation**: https://github.com/astral-sh/uv
  - Modern Python project management
  - uvx for executable installation

## Notes

### What This RDR Does NOT Define

- **CLI Tool Implementation**: Subcommand structure, argument parsing
- **Docker Configuration**: Container specs, volume mounts, networking
- **Indexing Logic**: PDF extraction, AST chunking, embedding models
- **Search Implementation**: Query processing, ranking, filtering
- **Full-Text Engine Choice**: Which full-text engine to use alongside Qdrant

### Next Steps

After implementing this minimal structure:

1. **RDR-002: Docker Management**
   - Define `arcaneum docker` subcommands
   - Qdrant container configuration
   - Full-text engine container configuration
   - Volume mount patterns

2. **RDR-003: Collection Management**
   - Create/delete/list collections
   - Metadata schema
   - Embedding model configuration

3. **RDR-004: PDF Indexing**
   - OCR integration
   - Chunking strategy
   - Metadata extraction

4. **RDR-005: Code Indexing**
   - Git awareness
   - AST chunking
   - Language detection

5. **RDR-006: Search Implementation**
   - Query processing
   - Metadata filtering
   - Result ranking

### Key Insights

**CLI-First Architecture**:
- All functionality implemented as CLI tools
- Slash commands are thin wrappers calling CLI
- No MCP overhead, direct process execution
- Easy to test and debug

**Colocation Benefits**:
- Single repository serves both audiences
- Plugin metadata and CLI tools stay in sync
- Unified documentation and issues
- Simpler contribution workflow

**Local Docker Philosophy**:
- No remote dependencies
- Reproducible environments
- Volume mounts for persistence
- Easy to reset/rebuild

**Slash Command Naming**:
- `/arc-*` prefix (short, memorable)
- Not `/arcaneum-*` (too long)
- Not `/qs-*` (ambiguous - "Qdrant search" or "quick search"?)
- Generic enough for multi-engine support (Qdrant + full-text)

**Future MCP Consideration**:
If MCP performance improves or specific features are needed, we can add:
- Optional MCP server in `src/arcaneum/mcp/`
- Entry in `.claude-plugin/plugin.json` → `mcpServers` section
- Keep CLI tools as primary interface
- MCP as alternative integration path

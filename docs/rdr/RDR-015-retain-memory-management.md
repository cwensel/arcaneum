# Recommendation 015: Retain Memory Management System

## Metadata

- **Date**: 2025-10-30
- **Status**: Recommendation
- **Type**: Feature
- **Priority**: High
- **Related Issues**: arcaneum-220 (JSON API consistency), arcaneum-221 (thin-wrapper architecture)
- **Related Tests**: tests/unit/retain/test_*.py, tests/integration/test_retain.py

## Problem Statement

Create a lightweight memory management layer (`retain` plugin) that provides AI agents (like Claude) with persistent,
searchable long-term memory by wrapping arc's existing infrastructure with memory-specific conventions. This system should:

1. **Wrapper Architecture**: Thin layer over arc's indexing and search capabilities, reusing 90% of arc's infrastructure
2. **Memory Conventions**: Provide memory-specific naming, metadata schema, and ID-based organization over arc operations
3. **Separate Plugin**: Independent `retain:*` plugin (not `arc:*`) allowing enable/disable for context management
4. **Global and Project-Local Storage**: Support both `~/.arcaneum/retain` (global) and project-specific retention
   via `.arcaneum` config
5. **Context-Optimized Output**: `get` command returns formatted string for AI memory injection (not just JSON),
   includes content + selective metadata
6. **Git-Integrated Versioning**: Leverages git for local retains automatically, optional custom versioning for global
7. **Code Reuse**: Reuses arc's indexing pipeline, search APIs, corpus system, and embedding models

The goal is to enable AI agents to:

- **Store** research findings, analysis results, learned patterns
- **Retrieve** relevant memories based on semantic or full-text search
- **Update** existing memories as knowledge evolves
- **Organize** memories globally or per-project
- **Inject** memories directly into conversations as context

## Context

### Background

**Current State** (RDR-014):

- `arc store` provides basic markdown injection to collections
- Storage location: `~/.arcaneum/agent-memory/{collection}/`
- Limited API: only "put" operation
- No retrieval, update, or delete capabilities
- No project-local storage support

**Limitations**:

1. **No retrieval**: Agent stores memories but can't retrieve them programmatically
2. **No CRUD**: Can't update or delete existing memories
3. **No search**: Must use generic `arc search` which isn't optimized for memory retrieval
4. **Organization**: Only supports global storage, no per-project memories
5. **Memory context**: `get` needs to return data optimized for AI consumption

**Use Cases**:

**UC1: Research Session Memory** (primary interface: slash commands)

```bash
# In Claude Code - Agent researches GPU acceleration patterns
/retain:put analysis.md --collection knowledge --tags "gpu,performance" \
  --context "Research for arcaneum-183"

# Later: retrieve for context
/retain:get <id> --format context

# Under the hood:
# Calls: arc retain put ... → RetainManager.put() → arc index markdown
```

**UC2: Project-Specific Memory** (slash commands with local detection)

```bash
# In Claude Code - Project directory with .arcaneum config
/retain:put analysis.md --tags "security,myapp" \
  --context "Security audit findings"

# Retrieve project memories (auto-detects local config)
/retain:search "authentication"

# Under the hood:
# Detects .arcaneum config → uses local storage → calls arc wrappers
```

**UC3: Memory Updates** (slash commands for lifecycle)

```bash
# In Claude Code - Initial storage
/retain:put initial-research.md --id gpu-patterns --tags "gpu"

# Update as knowledge evolves
/retain:update gpu-patterns --content updated-research.md --tags "gpu,metal"

# View history (future)
/retain:get gpu-patterns --show-versions

# Under the hood:
# Each command calls arc retain CLI → RetainManager → arc wrappers
```

### Technical Environment

**Dependencies** (from RDR-014):

- Python >= 3.12
- Qdrant >= 1.15.4 (semantic search)
- MeiliSearch >= 1.24.0 (full-text search)
- markdown-it-py >= 4.0.0 (parsing)
- python-frontmatter >= 1.1.0 (metadata)
- FastEmbed >= 0.7.3 / sentence-transformers >= 3.3.1 (embeddings)

**Existing Infrastructure**:

- Corpus system (RDR-009): Dual-index (Qdrant + MeiliSearch)
- Markdown indexing (RDR-014): Semantic chunking, frontmatter extraction
- Collection typing (RDR-003): Type validation
- Path management (src/arcaneum/paths.py): Directory helpers

**New Requirements**:

- `.arcaneum` config file format (YAML or JSON)
- Retain-specific metadata schema
- ID-based retrieval and updates
- Context-optimized output format

## Research Findings

### Investigation Process

**Research completed**:

1. **Memory System Patterns**: Analysis of agent memory architectures (LangChain, Semantic Kernel, MemGPT)
2. **CRUD API Design**: Research on idempotent operations, ID generation, conflict resolution
3. **Config File Formats**: Investigation of .arcaneum structure (local vs global distinction)
4. **Memory Retrieval Optimization**: What metadata should be included in agent context
5. **Git Integration**: How to handle versioning when memories are git-tracked

### Key Discoveries

#### 1. Memory Organization: Global vs Local

**Global Retain** (`~/.arcaneum/retain/`):

```text
~/.arcaneum/retain/
  {collection}/
    {id}.md          # Individual memory files
    .index.json      # Optional fast lookup index
```

**Local Retain** (project-specific):

```text
/path/to/project/
  .arcaneum              # Config file
  .arcaneum-retain/      # Local memory storage
    {collection}/
      {id}.md
```

**Config File** (`.arcaneum`):

```yaml
version: 1
retain:
  enabled: true
  path: .arcaneum-retain  # Relative to project root
  history: false          # Disable versioning (git handles it)
  collections:
    - memory
    - notes
```

**Detection Logic**:

```python
def detect_retain_location(explicit_local: bool = None) -> Path:
    """Detect global vs local retain location.

    Args:
        explicit_local: Force local (True) or global (False), None for auto-detect

    Returns:
        Path to retain directory
    """
    if explicit_local is False:
        return Path.home() / ".arcaneum" / "retain"

    # Auto-detect: walk up from cwd to find .arcaneum
    if explicit_local is None or explicit_local is True:
        config_path = find_arcaneum_config()  # Walk up directory tree
        if config_path:
            config = load_arcaneum_config(config_path)
            if config.get('retain', {}).get('enabled'):
                project_root = config_path.parent
                retain_path = project_root / config['retain']['path']
                return retain_path

    # Fallback to global
    return Path.home() / ".arcaneum" / "retain"
```

**Benefits**:

- **Global**: Cross-project knowledge, persistent across all work
- **Local**: Project-specific memories, git-trackable, team-shareable

#### 2. ID-Based Memory Management

**ID Generation**:

```python
def generate_memory_id(title: str = None, content: str = None) -> str:
    """Generate unique memory ID.

    Priority:
    1. Use provided title (slugified)
    2. Extract from H1 in content
    3. Generate from content hash (first 12 chars)

    Returns:
        Unique ID suitable for filename
    """
    if title:
        return slugify(title)

    # Extract H1
    h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if h1_match:
        return slugify(h1_match.group(1))

    # Hash-based fallback
    return hashlib.sha256(content.encode()).hexdigest()[:12]

def slugify(text: str) -> str:
    """Convert text to filename-safe slug."""
    slug = text.lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    return slug.strip('-')[:50]
```

**Example IDs**:

- `gpu-acceleration-patterns`
- `security-audit-findings-2025-10`
- `5f3a2c8b12ef` (hash fallback)

**Benefits**:

- Human-readable IDs
- Stable references (same ID = same memory)
- Easy file management

#### 3. Memory Retrieval: Context-Optimized Output

**Challenge**: What should `arc retain get` return for optimal AI memory usage?

**Analysis**:

AI agents need:

1. **Full content**: The actual memory text
2. **Contextual metadata**: When/why it was created, what it relates to
3. **Search hints**: Tags, categories for related lookups
4. **NOT needed**: Low-level details (file paths, hashes, chunk counts)

**Context-Optimized Output Format** (for AI injection):

The `context` format returns a human-readable string (NOT JSON) specifically designed for injecting into AI
conversations. It includes relevant metadata formatted as readable text, followed by the full content.

**Note**: All commands will also support `--json` for programmatic use and plugin integration (see arcaneum-220).

```python
@dataclass
class MemoryContext:
    """Optimized memory data for AI agent context."""

    # Identity
    id: str                        # Memory ID
    title: str                     # Document title

    # Content
    content: str                   # Full markdown content

    # Context metadata (helps AI understand when/why memory exists)
    created_at: str                # ISO timestamp
    updated_at: Optional[str]      # Last update
    created_by: str                # Agent name
    context: Optional[str]         # Why this was created

    # Organization metadata (helps AI find related memories)
    collection: str                # Which collection
    tags: List[str]                # Searchable keywords
    category: Optional[str]        # Primary category
    related_to: List[str]          # Related IDs or issue refs

    # Optional extended metadata
    project: Optional[str]         # Project name
    priority: Optional[str]        # Importance level
    status: Optional[str]          # Lifecycle state

    def to_context_string(self) -> str:
        """Format as string for AI context injection."""
        parts = [
            f"# {self.title}",
            f"ID: {self.id}",
            f"Created: {self.created_at} by {self.created_by}",
        ]

        if self.context:
            parts.append(f"Context: {self.context}")

        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")

        if self.related_to:
            parts.append(f"Related: {', '.join(self.related_to)}")

        parts.append("")  # Blank line
        parts.append(self.content)

        return "\n".join(parts)

    def to_json(self) -> dict:
        """Format as JSON for programmatic use."""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'metadata': {
                'created_at': self.created_at,
                'updated_at': self.updated_at,
                'created_by': self.created_by,
                'context': self.context,
                'collection': self.collection,
                'tags': self.tags,
                'category': self.category,
                'related_to': self.related_to,
                'project': self.project,
                'priority': self.priority,
                'status': self.status,
            }
        }
```

**Output Formats**:

```bash
# Context format (AI-optimized string) - DEFAULT
arc retain get gpu-patterns --format context
# Outputs: Formatted string ready for AI context injection
# Example:
# # GPU Acceleration Patterns
# ID: gpu-patterns
# Created: 2025-10-30T14:23:45Z by Claude
# Context: Research for arcaneum-183
# Tags: gpu, performance, metal
#
# [full markdown content...]

# JSON format (programmatic use, plugin integration)
arc retain get gpu-patterns --format json
# Outputs: Structured JSON with all metadata
# {"id": "gpu-patterns", "title": "...", "content": "...", "metadata": {...}}

# Raw markdown (file content only)
arc retain get gpu-patterns --format raw
# Outputs: Just the markdown content without any metadata

# Note: All commands also support --json flag for programmatic output (arcaneum-220)
arc retain get gpu-patterns --json
# Shorthand for --format json
```

#### 4. CRUD Operations Design

**Primary Interface: Slash Commands** (for Claude Code)

**Put** (Create or Replace):

```bash
# Slash command (primary)
/retain:put <file> --id <id> --collection <name> [--tags <tags>] [options]
/retain:put - --id <id> --collection <name>  # stdin

# Examples in Claude Code
/retain:put notes.md --collection memory --tags "security,api"
/retain:put research.md --id gpu-patterns --collection knowledge

# CLI implementation (what slash command calls)
# arc retain put <file> --id <id> --collection <name> [options]
```

**Get** (Retrieve):

```bash
# Slash command (primary)
/retain:get <id> [--collection <name>] [--format context|json|raw]

# Examples in Claude Code
/retain:get research-gpu --format context
/retain:get research-gpu --collection knowledge --format json

# CLI implementation
# arc retain get <id> [--collection <name>] [--format ...]
```

**Update** (Modify):

```bash
# Slash command (primary)
/retain:update <id> --content <file> [--tags <tags>] [--merge-tags] [options]

# Examples in Claude Code
/retain:update research-gpu --content updated.md
/retain:update research-gpu --tags "gpu,metal,performance" --merge-tags

# CLI implementation
# arc retain update <id> [options]
```

**Delete** (Remove):

```bash
# Slash command (primary)
/retain:delete <id> [--collection <name>]

# Examples in Claude Code
/retain:delete old-research
/retain:delete research-gpu --collection knowledge

# CLI implementation
# arc retain delete <id> [--collection <name>]
```

**Search** (Query):

```bash
# Slash command (primary)
/retain:search <query> [--collection <name>] [--filter <expr>] [--semantic|--text]

# Examples in Claude Code
/retain:search "GPU acceleration"
/retain:search "security" --filter "tags=api,auth"
/retain:search "performance" --collection knowledge --semantic

# CLI implementation
# arc retain search <query> [options]
# Calls: arc search semantic "query" --collection retain-{name} --json
```

**List** (Enumerate):

```bash
# Slash command (primary)
/retain:list [--collection <name>] [--filter <expr>] [--format table|json]

# Examples in Claude Code
/retain:list
/retain:list --collection knowledge --filter "priority=high"
/retain:list --format json

# CLI implementation
# arc retain list [options]
```

#### 5. Code Reuse Strategy: Wrapping Arc Infrastructure

**Design Principle**: Retain is a thin wrapper (~10% new code) over arc's existing infrastructure (~90% reuse).

**What Arc Provides (Infrastructure Reused)**:

Arc already has everything needed for memory operations:

1. **Markdown Indexing** (RDR-014)
   - `arc index markdown` - Frontmatter extraction, semantic chunking
   - Incremental sync, directory watching
   - Already handles `.md` files with metadata

2. **Search Infrastructure**
   - `arc search semantic` - Qdrant vector search
   - `arc search text` - MeiliSearch full-text (planned)
   - Score ranking, filtering, JSON output

3. **Corpus System** (RDR-009)
   - Dual-index: Qdrant + MeiliSearch
   - Collection management
   - Batch operations

4. **Embedding Models**
   - stella, jina-code, modernbert, bge-large
   - GPU acceleration support
   - Model caching

**What Retain Adds (New Code)**:

Retain provides memory-specific conventions and formatting:

1. **Memory Conventions**
   - Collection naming: `retain-{collection}` (namespace)
   - File organization: `.arcaneum-retain/{collection}/{id}.md`
   - Metadata schema: Memory-specific frontmatter fields
   - ID management: Slugification, uniqueness

2. **Context Formatting**
   - `MemoryContext` dataclass
   - `to_context_string()` - AI-optimized output
   - `to_json()` - Structured data
   - `to_raw()` - Just content

3. **Git Integration** (thin wrapper)
   - Version history: `git log --follow`
   - Version retrieval: `git show <ref>:<path>`
   - Diff operations: `git diff`
   - No custom versioning system

4. **Wrapper CLI**
   - `arc retain put` → `arc index markdown`
   - `arc retain search` → `arc search semantic`
   - `arc retain list` → filesystem scan + arc collection info

**Wrapper Implementation Examples**:

**Approach 1: Subprocess (simple, isolated)**

```python
import subprocess

def put(content: str, collection: str, id: str, **metadata):
    """Store memory - wraps arc index markdown"""
    # 1. Write to .arcaneum-retain/{collection}/{id}.md
    file_path = get_retain_path(collection, id)
    write_memory_file(file_path, content, metadata)

    # 2. Call arc to index
    subprocess.run([
        'arc', 'index', 'markdown',
        str(file_path),
        '--collection', f'retain-{collection}'
    ], check=True)

    return {'id': id, 'file_path': str(file_path)}

def search(query: str, collection: str, limit: int = 10):
    """Search memories - wraps arc search"""
    result = subprocess.run([
        'arc', 'search', 'semantic',
        query,
        '--collection', f'retain-{collection}',
        '--limit', str(limit),
        '--json'
    ], capture_output=True, text=True, check=True)

    return json.loads(result.stdout)
```

**Approach 2: Direct Import (alternative - faster but tighter coupling)**

```python
from arcaneum.indexing.markdown.pipeline import MarkdownIndexingPipeline
from arcaneum.search.semantic import search_semantic

class RetainManager:
    def __init__(self):
        # Reuse arc's pipeline (requires arc as Python dependency)
        self.pipeline = MarkdownIndexingPipeline()

    def put(self, content: str, collection: str, id: str, **metadata):
        """Store memory - reuses arc indexing"""
        # 1. Write file
        file_path = get_retain_path(collection, id)
        write_memory_file(file_path, content, metadata)

        # 2. Index using arc's pipeline directly
        collection_name = f'retain-{collection}'
        self.pipeline.index_file(file_path, collection_name)

        return {'id': id, 'file_path': str(file_path)}

    def search(self, query: str, collection: str, limit: int = 10):
        """Search memories - reuses arc search"""
        # Call arc's search function directly
        return search_semantic(
            query=query,
            collection=f'retain-{collection}',
            limit=limit
        )
```

**Recommendation**: Use Approach 1 (subprocess) for simplicity and isolation. Approach 2 is viable if
performance becomes critical, but adds Python-level coupling between retain and arc.

**Benefits of Wrapper Approach**:

- **90% code reuse**: All indexing, search, corpus logic stays in arc
- **No duplication**: Bug fixes in arc automatically benefit retain
- **Consistent behavior**: Same embedding models, same chunking, same search
- **Faster implementation**: ~30 hours vs ~67 hours
- **Easier maintenance**: Simpler codebase, fewer tests
- **Shared infrastructure**: Both plugins use same Qdrant/MeiliSearch

#### 6. Versioning Strategy: Git Integration for Local, Optional Custom for Global

**Design Principle**: Leverage git for local retains, implement custom versioning only for global storage.

**Local Retain Versioning** (Git-based):

When memories are stored locally (`.arcaneum-retain/`), git automatically tracks all changes:

```bash
# View version history for a memory
arc retain versions gpu-patterns
# Under the hood: git log --follow -- .arcaneum-retain/knowledge/gpu-patterns.md

# Get specific version (git-based)
arc retain get gpu-patterns --version HEAD~2
# Under the hood: git show HEAD~2:.arcaneum-retain/knowledge/gpu-patterns.md

# Restore previous version
arc retain restore gpu-patterns --version HEAD~1
# Under the hood: git show HEAD~1:.arcaneum-retain/knowledge/gpu-patterns.md > .arcaneum-retain/knowledge/gpu-patterns.md

# View diff between versions
arc retain diff gpu-patterns --from HEAD~2 --to HEAD
# Under the hood: git diff HEAD~2 HEAD -- .arcaneum-retain/knowledge/gpu-patterns.md
```

**Git Integration Implementation**:

```python
def get_memory_versions_local(id: str, collection: str) -> List[MemoryVersion]:
    """Get version history from git for local retain.

    Args:
        id: Memory ID
        collection: Collection name

    Returns:
        List of versions with git commit info
    """
    file_path = f".arcaneum-retain/{collection}/{id}.md"

    # Get git log for file
    result = subprocess.run(
        ["git", "log", "--follow", "--format=%H|%aI|%an|%s", "--", file_path],
        capture_output=True,
        text=True,
        cwd=self.location.parent  # Project root
    )

    versions = []
    for line in result.stdout.strip().split('\n'):
        commit_hash, timestamp, author, subject = line.split('|', 3)
        versions.append(MemoryVersion(
            version=commit_hash[:7],  # Short hash
            timestamp=timestamp,
            author=author,
            message=subject,
            git_ref=commit_hash
        ))

    return versions

def get_memory_at_version(id: str, collection: str, version: str) -> str:
    """Retrieve memory content at specific git version.

    Args:
        id: Memory ID
        collection: Collection name
        version: Git ref (commit hash, HEAD~N, tag, etc.)

    Returns:
        Memory content at that version
    """
    file_path = f".arcaneum-retain/{collection}/{id}.md"

    result = subprocess.run(
        ["git", "show", f"{version}:{file_path}"],
        capture_output=True,
        text=True,
        cwd=self.location.parent
    )

    if result.returncode != 0:
        raise ValueError(f"Version not found: {version}")

    return result.stdout
```

**Global Retain Versioning** (Custom, Optional):

For global storage (`~/.arcaneum/retain/`), implement custom versioning:

```text
~/.arcaneum/retain/
  knowledge/
    gpu-patterns.md           # Current version
    .versions/
      gpu-patterns/
        v1_2025-10-30T14-23-45Z.md
        v2_2025-10-31T09-12-10Z.md
```

**Version Metadata** (global only):

```yaml
---
id: gpu-patterns
version: 3
version_history:
  - version: 1
    created_at: 2025-10-30T14:23:45Z
    created_by: Claude
  - version: 2
    updated_at: 2025-10-31T09:12:10Z
    updated_by: Claude
  - version: 3
    updated_at: 2025-11-01T16:45:00Z
    updated_by: Claude
---
```

**Unified API** (works for both local and global):

```bash
# List versions (git log for local, custom for global)
arc retain versions gpu-patterns

# Get specific version
arc retain get gpu-patterns --version HEAD~2  # Git ref for local
arc retain get gpu-patterns --version 2       # Version number for global

# Restore version (creates new commit for local)
arc retain restore gpu-patterns --version HEAD~1

# Diff between versions
arc retain diff gpu-patterns --from v1 --to v2
```

**Config Options**:

```yaml
# .arcaneum (local)
version: 1
retain:
  enabled: true
  path: .arcaneum-retain
  versioning: git  # Use git for version history (default for local)

# Global config (optional)
version: 1
retain:
  versioning: custom  # Custom version tracking
  max_versions: 10    # Keep last N versions
  auto_version: true  # Auto-version on update
```

**Benefits**:

- **Local**: Free versioning via git, team-shareable history, standard git workflows
- **Global**: Simple custom versioning for personal knowledge base
- **Unified API**: Same commands work for both storage types
- **Zero Config**: Local automatically uses git if available

## Proposed Solution

### Approach

Implement a **thin wrapper plugin** providing memory-specific conventions over arc's infrastructure:

1. **Primary Interface**: `retain:*` slash commands for Claude Code (e.g., `/retain:put`, `/retain:search`)
2. **Separate Plugin**: Independent `retain:*` plugin (not `arc:*`) for enable/disable control
3. **CLI Implementation**: Slash commands call `arc retain` CLI as implementation layer
4. **Wrapper Architecture**: `arc retain` calls arc commands (subprocess) for indexing/search
5. **Code Reuse**: Reuses arc's indexing pipeline, search APIs, corpus system (~90% reuse)
6. **Memory Conventions**: ID-based organization, memory-specific metadata schema
7. **Dual Storage**: Global (`~/.arcaneum/retain`) and local (`.arcaneum` config-driven)
8. **Context-Optimized Output**: `get` returns AI-friendly formatted string
9. **Git-Integrated Versioning**: Thin wrapper over git commands for local retains
10. **Minimal New Code**: ~500 lines vs ~5000 lines if implemented standalone

**User Experience**: Agents use `/retain:put` in Claude Code → CLI escape hatch: `arc retain put`

### Technical Design

#### Architecture Overview

**Thin Wrapper Over Arc Infrastructure**:

```text
┌────────────────────────────────────────────────────┐
│         retain:* Slash Commands (Separate Plugin)  │
│  retain:put | retain:get | retain:search | ...     │
└──────────────────────┬─────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────┐
│          arc retain CLI (Thin Wrapper)             │
│  - ID generation & file management                 │
│  - Memory metadata conventions                     │
│  - Context formatting (MemoryContext)              │
│  - Git wrapper (versions, diff)                    │
└──────────────────────┬─────────────────────────────┘
                       ↓
         ┌─────────────┴─────────────┐
         │ Subprocess Calls          │
         │ OR Direct Imports         │
         └─────────────┬─────────────┘
                       ↓
┌────────────────────────────────────────────────────┐
│         Arc Infrastructure (Reused 90%)            │
│                                                    │
│  arc index markdown  →  Indexing Pipeline          │
│  arc search semantic →  Qdrant Search              │
│  arc search text     →  MeiliSearch Search         │
│                                                    │
│  → Corpus System (Dual-Index)                      │
│  → Embedding Models (stella, jina-code, etc)      │
│  → Collection Management                           │
└────────────────────────────────────────────────────┘
         ↓                          ↓
┌──────────────────┐    ┌──────────────────────┐
│  File Storage    │    │  Vector Databases    │
│                  │    │                      │
│  .arcaneum-      │    │  → Qdrant            │
│   retain/        │    │     (semantic)       │
│  {collection}/   │    │  → MeiliSearch       │
│   {id}.md        │    │     (full-text)      │
└──────────────────┘    └──────────────────────┘
```

**Key Principle**: Retain adds ~10% new code (conventions, formatting) and wraps arc's ~90% existing infrastructure.

#### Core Components

**1. Retain Manager**

```python
# src/arcaneum/retain/manager.py

from pathlib import Path
from typing import Optional, List, Dict
import yaml
import frontmatter
from datetime import datetime

class RetainManager:
    """Core memory management system."""

    def __init__(self, location: Optional[Path] = None):
        """Initialize retain manager.

        Args:
            location: Explicit retain location (None for auto-detect)
        """
        self.location = location or self._detect_location()
        self.config = self._load_config()

    def _detect_location(self) -> Path:
        """Detect global vs local retain location."""
        # Walk up from cwd to find .arcaneum
        config_path = self._find_arcaneum_config()

        if config_path:
            config = self._load_arcaneum_config(config_path)
            if config.get('retain', {}).get('enabled'):
                project_root = config_path.parent
                retain_path = project_root / config['retain']['path']
                if retain_path.exists():
                    return retain_path

        # Fallback to global
        from arcaneum.paths import get_arcaneum_dir
        return get_arcaneum_dir() / "retain"

    def _find_arcaneum_config(self) -> Optional[Path]:
        """Walk up directory tree to find .arcaneum config."""
        current = Path.cwd()

        while current != current.parent:
            config_path = current / ".arcaneum"
            if config_path.exists():
                return config_path
            current = current.parent

        return None

    def _load_arcaneum_config(self, path: Path) -> dict:
        """Load .arcaneum config file (YAML or JSON)."""
        content = path.read_text()

        # Try YAML first
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError:
            # Try JSON
            import json
            return json.loads(content)

    def _load_config(self) -> dict:
        """Load retain-specific configuration."""
        if self._is_local():
            config_path = self._find_arcaneum_config()
            if config_path:
                full_config = self._load_arcaneum_config(config_path)
                return full_config.get('retain', {})

        return {
            'history': True,  # Global default: history enabled
            'collections': []
        }

    def _is_local(self) -> bool:
        """Check if using local (project-specific) retain."""
        return '.arcaneum-retain' in str(self.location) or \
               self.location != (Path.home() / ".arcaneum" / "retain")

    def put(
        self,
        content: str,
        collection: str,
        id: Optional[str] = None,
        title: Optional[str] = None,
        tags: List[str] = None,
        category: Optional[str] = None,
        context: Optional[str] = None,
        metadata: Dict = None,
        created_by: str = "Claude"
    ) -> Dict:
        """Store a memory.

        Args:
            content: Markdown content
            collection: Collection name
            id: Memory ID (generated if not provided)
            title: Document title
            tags: List of tags
            category: Category label
            context: Context description
            metadata: Additional metadata
            created_by: Agent name

        Returns:
            Dict with id, file_path, collection
        """
        # Generate ID if not provided
        if not id:
            id = self._generate_id(title, content)

        # Parse existing frontmatter if present
        post = frontmatter.loads(content)

        # Build complete metadata
        now = datetime.now().isoformat()
        meta = {
            'id': id,
            'title': title or self._extract_title(post.content),
            'collection': collection,
            'created_at': now,
            'created_by': created_by,
            'tags': tags or [],
            'category': category,
            'context': context,
            **(metadata or {}),
        }

        # Create frontmatter document
        post.metadata = meta

        # Ensure collection directory exists
        collection_dir = self.location / collection
        collection_dir.mkdir(parents=True, exist_ok=True)

        # Write file
        file_path = collection_dir / f"{id}.md"
        file_path.write_text(frontmatter.dumps(post))

        # Index via arc (wrapper approach)
        collection_name = f"retain-{collection}"
        self._call_arc_index(file_path, collection_name)

        return {
            'id': id,
            'file_path': str(file_path),
            'collection': collection
        }

    def get(
        self,
        id: str,
        collection: Optional[str] = None,
        format: str = 'context'
    ) -> Optional[MemoryContext]:
        """Retrieve a memory.

        Args:
            id: Memory ID
            collection: Collection name (searches all if None)
            format: Output format (context, json, raw)

        Returns:
            MemoryContext or None if not found
        """
        file_path = self._find_memory_file(id, collection)

        if not file_path:
            return None

        # Load file
        post = frontmatter.load(file_path)
        meta = post.metadata

        # Build MemoryContext
        memory = MemoryContext(
            id=meta.get('id', id),
            title=meta.get('title', 'Untitled'),
            content=post.content,
            created_at=meta.get('created_at'),
            updated_at=meta.get('updated_at'),
            created_by=meta.get('created_by', 'Unknown'),
            context=meta.get('context'),
            collection=meta.get('collection', collection),
            tags=meta.get('tags', []),
            category=meta.get('category'),
            related_to=meta.get('related_to', []),
            project=meta.get('project'),
            priority=meta.get('priority'),
            status=meta.get('status')
        )

        return memory

    def update(
        self,
        id: str,
        content: Optional[str] = None,
        collection: Optional[str] = None,
        tags: List[str] = None,
        merge_tags: bool = False,
        metadata: Dict = None
    ) -> Dict:
        """Update an existing memory.

        Args:
            id: Memory ID
            content: New content (None to keep existing)
            collection: Collection hint for faster lookup
            tags: New tags (replaces or merges based on merge_tags)
            merge_tags: If True, merge with existing tags
            metadata: Additional metadata updates

        Returns:
            Dict with id, file_path, updated_at
        """
        file_path = self._find_memory_file(id, collection)

        if not file_path:
            raise ValueError(f"Memory not found: {id}")

        # Load existing
        post = frontmatter.load(file_path)

        # Update content if provided
        if content:
            post.content = content

        # Update metadata
        now = datetime.now().isoformat()
        post.metadata['updated_at'] = now

        if tags:
            if merge_tags:
                existing = set(post.metadata.get('tags', []))
                post.metadata['tags'] = list(existing | set(tags))
            else:
                post.metadata['tags'] = tags

        if metadata:
            post.metadata.update(metadata)

        # Write back
        file_path.write_text(frontmatter.dumps(post))

        # Re-index via arc (wrapper approach)
        corpus_name = f"retain-{post.metadata['collection']}"
        self._call_arc_index(file_path, corpus_name)

        return {
            'id': id,
            'file_path': str(file_path),
            'updated_at': now
        }

    def delete(
        self,
        id: str,
        collection: Optional[str] = None
    ) -> bool:
        """Delete a memory.

        Args:
            id: Memory ID
            collection: Collection hint

        Returns:
            True if deleted, False if not found

        Note:
            File is removed from disk immediately. To remove from search index,
            run `arc index markdown` with --force on the collection directory,
            or the next incremental sync will detect the deletion.
        """
        file_path = self._find_memory_file(id, collection)

        if not file_path:
            return False

        # Delete file (index cleanup happens on next sync)
        file_path.unlink()

        return True

    def search(
        self,
        query: str,
        collection: Optional[str] = None,
        filter_expr: Optional[str] = None,
        semantic: bool = True,
        limit: int = 10
    ) -> List[Dict]:
        """Search memories.

        Args:
            query: Search query
            collection: Limit to collection
            filter_expr: Metadata filter (e.g., "tags=gpu,performance")
            semantic: Use semantic (True) or full-text (False) search
            limit: Max results

        Returns:
            List of search results with scores
        """
        corpus_name = f"retain-{collection}" if collection else "retain-*"

        if semantic:
            # Semantic search via arc wrapper
            results = self._call_arc_search(
                query=query,
                collection=corpus_name,
                limit=limit,
                filter_expr=filter_expr
            )
        else:
            # Full-text search via arc wrapper
            results = self._call_arc_search_text(
                query=query,
                collection=corpus_name,
                limit=limit,
                filter_expr=filter_expr
            )

        return results

    def list(
        self,
        collection: Optional[str] = None,
        filter_expr: Optional[str] = None
    ) -> List[Dict]:
        """List all memories.

        Args:
            collection: Limit to collection
            filter_expr: Metadata filter

        Returns:
            List of memory summaries
        """
        memories = []

        if collection:
            collections = [collection]
        else:
            # List all collection directories
            collections = [d.name for d in self.location.iterdir() if d.is_dir()]

        for coll in collections:
            coll_dir = self.location / coll
            if not coll_dir.exists():
                continue

            for file_path in coll_dir.glob("*.md"):
                post = frontmatter.load(file_path)
                meta = post.metadata

                # Apply filter if provided
                if filter_expr and not self._matches_filter(meta, filter_expr):
                    continue

                memories.append({
                    'id': meta.get('id', file_path.stem),
                    'title': meta.get('title', 'Untitled'),
                    'collection': coll,
                    'created_at': meta.get('created_at'),
                    'tags': meta.get('tags', []),
                    'category': meta.get('category')
                })

        return memories

    def _generate_id(self, title: Optional[str], content: str) -> str:
        """Generate memory ID from title or content."""
        import re
        import hashlib

        if title:
            return self._slugify(title)

        # Try to extract H1
        h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if h1_match:
            return self._slugify(h1_match.group(1))

        # Hash fallback
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _slugify(self, text: str) -> str:
        """Convert text to filename-safe slug."""
        import re
        slug = text.lower()
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        return slug.strip('-')[:50]

    def _extract_title(self, content: str) -> str:
        """Extract title from first H1 or first line."""
        import re
        h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1)

        lines = content.strip().split('\n')
        return lines[0][:50] if lines else "Untitled"

    def _find_memory_file(self, id: str, collection: Optional[str]) -> Optional[Path]:
        """Find memory file by ID."""
        if collection:
            file_path = self.location / collection / f"{id}.md"
            if file_path.exists():
                return file_path
        else:
            # Search all collections
            for coll_dir in self.location.iterdir():
                if not coll_dir.is_dir():
                    continue
                file_path = coll_dir / f"{id}.md"
                if file_path.exists():
                    return file_path

        return None

    def _call_arc_index(self, file_path: Path, collection: str):
        """Index via arc (wrapper approach).

        Two implementation options:

        Option 1 - Subprocess (simple, isolated):
            subprocess.run(['arc', 'index', 'markdown', str(file_path), '--collection', collection])

        Option 2 - Direct import (faster, shared state):
            from arcaneum.indexing.markdown.pipeline import MarkdownIndexingPipeline
            pipeline = MarkdownIndexingPipeline()
            pipeline.index_file(file_path, collection)
        """
        import subprocess
        subprocess.run([
            'arc', 'index', 'markdown',
            str(file_path),
            '--collection', collection
        ], check=True)

    def _call_arc_search(
        self,
        query: str,
        collection: str,
        limit: int = 10,
        filter_expr: Optional[str] = None
    ):
        """Search via arc semantic search (wrapper approach)."""
        import subprocess
        import json

        cmd = [
            'arc', 'search', 'semantic',
            query,
            '--collection', collection,
            '--limit', str(limit),
            '--json'
        ]

        if filter_expr:
            cmd.extend(['--filter', filter_expr])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        return json.loads(result.stdout)

    def _call_arc_search_text(
        self,
        query: str,
        collection: str,
        limit: int = 10,
        filter_expr: Optional[str] = None
    ):
        """Search via arc full-text search (wrapper approach)."""
        import subprocess
        import json

        cmd = [
            'arc', 'search', 'text',
            query,
            '--collection', collection,
            '--limit', str(limit),
            '--json'
        ]

        if filter_expr:
            cmd.extend(['--filter', filter_expr])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        return json.loads(result.stdout)

    def _matches_filter(self, metadata: dict, filter_expr: str) -> bool:
        """Check if metadata matches filter expression."""
        # Simple filter: "key=value" or "key=value1,value2"
        key, value = filter_expr.split('=', 1)
        key = key.strip()

        if key not in metadata:
            return False

        values = [v.strip() for v in value.split(',')]
        meta_value = metadata[key]

        if isinstance(meta_value, list):
            return any(v in meta_value for v in values)
        else:
            return str(meta_value) in values
```

**2. Memory Context Data Class**

```python
# src/arcaneum/retain/types.py

from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class MemoryContext:
    """Optimized memory data for AI agent context."""

    # Identity
    id: str
    title: str

    # Content
    content: str

    # Context metadata
    created_at: str
    updated_at: Optional[str]
    created_by: str
    context: Optional[str]

    # Organization metadata
    collection: str
    tags: List[str]
    category: Optional[str]
    related_to: List[str]

    # Optional metadata
    project: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None

    def to_context_string(self) -> str:
        """Format as string for AI context injection."""
        parts = [
            f"# {self.title}",
            f"ID: {self.id}",
            f"Created: {self.created_at} by {self.created_by}",
        ]

        if self.context:
            parts.append(f"Context: {self.context}")

        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")

        if self.category:
            parts.append(f"Category: {self.category}")

        if self.related_to:
            parts.append(f"Related: {', '.join(self.related_to)}")

        parts.append("")  # Blank line
        parts.append(self.content)

        return "\n".join(parts)

    def to_json(self) -> dict:
        """Format as JSON for programmatic use."""
        return asdict(self)

    def to_raw(self) -> str:
        """Return raw content only."""
        return self.content
```

**3. CLI Commands** (Implementation Layer)

**Note**: The primary user interface is `/retain:*` slash commands in Claude Code. The CLI (`arc retain`)
is the implementation layer that slash commands call, and also serves as an escape hatch for direct use.

CLI commands delegate to RetainManager, which wraps arc commands via subprocess or direct imports.

Architecture flow: `/retain:<cmd>` → `arc retain <cmd>` → `RetainManager.<method>()` → `arc <subcommand>` (subprocess)

```python
# src/arcaneum/cli/retain.py

import click
import sys
import json
from rich.console import Console
from rich.table import Table

console = Console()

@click.group('retain')
def retain():
    """Memory management system for AI agents.

    All commands use RetainManager which wraps arc's indexing and search
    infrastructure via subprocess calls (see RetainManager implementation).
    """
    pass

@retain.command('put')
@click.argument('file', type=click.Path())
@click.option('--id', help='Memory ID (auto-generated if not provided)')
@click.option('--collection', required=True, help='Collection name')
@click.option('--title', help='Document title')
@click.option('--tags', help='Comma-separated tags')
@click.option('--category', help='Category label')
@click.option('--context', help='Context description')
@click.option('--metadata', help='Additional metadata as JSON')
@click.option('--created-by', default='Claude', help='Agent name')
@click.option('--local', is_flag=True, help='Force local retain')
@click.option('--global', 'use_global', is_flag=True, help='Force global retain')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON')
def put_command(file, id, collection, title, tags, category, context, metadata, created_by, local, use_global, output_json):
    """Store a memory.

    Examples:
      # From stdin
      echo "# Research\\n..." | arc retain put - --collection knowledge

      # From file with metadata
      arc retain put notes.md --collection memory --tags "security,api"

      # With explicit ID
      arc retain put research.md --id gpu-patterns --collection knowledge
    """
    from arcaneum.retain.manager import RetainManager

    # Read content
    if file == '-':
        content = sys.stdin.read()
    else:
        from pathlib import Path
        content = Path(file).read_text()

    # Parse metadata
    meta = {}
    if metadata:
        meta = json.loads(metadata)

    # Determine location
    location = None
    if local:
        # Force local detection
        pass  # Let manager detect
    elif use_global:
        from arcaneum.paths import get_arcaneum_dir
        location = get_arcaneum_dir() / "retain"

    # Initialize manager
    manager = RetainManager(location=location)

    # Put memory (RetainManager writes file, then calls arc wrapper)
    # Flow: manager.put() -> writes .md file -> calls `arc index markdown`
    result = manager.put(
        content=content,
        collection=collection,
        id=id,
        title=title,
        tags=[t.strip() for t in tags.split(',')] if tags else None,
        category=category,
        context=context,
        metadata=meta,
        created_by=created_by
    )

    if output_json:
        console.print_json(data=result)
    else:
        console.print(f"[green]✅ Stored memory: {result['id']}[/green]")
        console.print(f"Collection: {result['collection']}")
        console.print(f"File: {result['file_path']}")

@retain.command('get')
@click.argument('id')
@click.option('--collection', help='Collection name (faster lookup)')
@click.option('--format', type=click.Choice(['context', 'json', 'raw']), default='context', help='Output format')
@click.option('--local', is_flag=True, help='Force local retain')
@click.option('--global', 'use_global', is_flag=True, help='Force global retain')
def get_command(id, collection, format, local, use_global):
    """Retrieve a memory.

    Examples:
      # Get for AI context
      arc retain get research-gpu --format context

      # Get as JSON
      arc retain get research-gpu --format json

      # Get raw content
      arc retain get research-gpu --format raw
    """
    from arcaneum.retain.manager import RetainManager

    # Determine location
    location = None
    if use_global:
        from arcaneum.paths import get_arcaneum_dir
        location = get_arcaneum_dir() / "retain"

    manager = RetainManager(location=location)

    memory = manager.get(id, collection=collection, format=format)

    if not memory:
        console.print(f"[red]❌ Memory not found: {id}[/red]")
        sys.exit(1)

    if format == 'context':
        console.print(memory.to_context_string())
    elif format == 'json':
        console.print_json(data=memory.to_json())
    elif format == 'raw':
        console.print(memory.to_raw())

@retain.command('update')
@click.argument('id')
@click.option('--content', type=click.Path(exists=True), help='New content file')
@click.option('--collection', help='Collection hint')
@click.option('--tags', help='New tags (comma-separated)')
@click.option('--merge-tags', is_flag=True, help='Merge with existing tags')
@click.option('--metadata', help='Additional metadata as JSON')
@click.option('--local', is_flag=True, help='Force local retain')
@click.option('--global', 'use_global', is_flag=True, help='Force global retain')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON')
def update_command(id, content, collection, tags, merge_tags, metadata, local, use_global, output_json):
    """Update an existing memory.

    Examples:
      # Update content
      arc retain update research-gpu --content updated.md

      # Update tags
      arc retain update research-gpu --tags "gpu,metal" --merge-tags

      # Update metadata
      arc retain update research-gpu --metadata '{"priority": "high"}'
    """
    from arcaneum.retain.manager import RetainManager
    from pathlib import Path

    location = None
    if use_global:
        from arcaneum.paths import get_arcaneum_dir
        location = get_arcaneum_dir() / "retain"

    manager = RetainManager(location=location)

    # Read new content if provided
    new_content = None
    if content:
        new_content = Path(content).read_text()

    # Parse metadata
    meta = None
    if metadata:
        meta = json.loads(metadata)

    # Update memory (RetainManager updates file, then re-indexes via arc)
    # Flow: manager.update() -> updates .md file -> calls `arc index markdown`
    result = manager.update(
        id=id,
        content=new_content,
        collection=collection,
        tags=[t.strip() for t in tags.split(',')] if tags else None,
        merge_tags=merge_tags,
        metadata=meta
    )

    if output_json:
        console.print_json(data=result)
    else:
        console.print(f"[green]✅ Updated memory: {result['id']}[/green]")
        console.print(f"File: {result['file_path']}")

@retain.command('delete')
@click.argument('id')
@click.option('--collection', help='Collection hint')
@click.option('--local', is_flag=True, help='Force local retain')
@click.option('--global', 'use_global', is_flag=True, help='Force global retain')
def delete_command(id, collection, local, use_global):
    """Delete a memory.

    File is removed immediately. Index cleanup happens on next sync.

    Examples:
      arc retain delete old-research
      arc retain delete research-gpu --collection knowledge
    """
    from arcaneum.retain.manager import RetainManager

    location = None
    if use_global:
        from arcaneum.paths import get_arcaneum_dir
        location = get_arcaneum_dir() / "retain"

    manager = RetainManager(location=location)

    if manager.delete(id, collection=collection):
        console.print(f"[green]✅ Deleted memory: {id}[/green]")
    else:
        console.print(f"[red]❌ Memory not found: {id}[/red]")
        sys.exit(1)

@retain.command('search')
@click.argument('query')
@click.option('--collection', help='Limit to collection')
@click.option('--filter', 'filter_expr', help='Metadata filter (e.g., tags=gpu,performance)')
@click.option('--semantic', is_flag=True, default=True, help='Semantic search (default)')
@click.option('--text', is_flag=True, help='Full-text search')
@click.option('--limit', type=int, default=10, help='Max results')
@click.option('--local', is_flag=True, help='Force local retain')
@click.option('--global', 'use_global', is_flag=True, help='Force global retain')
@click.option('--json', 'output_json', is_flag=True, help='Output JSON')
def search_command(query, collection, filter_expr, semantic, text, limit, local, use_global, output_json):
    """Search memories.

    Examples:
      arc retain search "GPU acceleration"
      arc retain search "security" --filter "tags=api,auth"
      arc retain search "performance" --collection knowledge
    """
    from arcaneum.retain.manager import RetainManager

    location = None
    if use_global:
        from arcaneum.paths import get_arcaneum_dir
        location = get_arcaneum_dir() / "retain"

    manager = RetainManager(location=location)

    use_semantic = not text  # Default to semantic unless --text specified

    # Search memories (RetainManager wraps arc search)
    # Flow: manager.search() -> calls `arc search semantic` or `arc search text`
    results = manager.search(
        query=query,
        collection=collection,
        filter_expr=filter_expr,
        semantic=use_semantic,
        limit=limit
    )

    if output_json:
        console.print_json(data=results)
    else:
        if not results:
            console.print("[yellow]No results found[/yellow]")
            return

        table = Table(title=f"Search Results ({len(results)})")
        table.add_column("ID", style="cyan")
        table.add_column("Title")
        table.add_column("Score", justify="right")
        table.add_column("Tags")

        for result in results:
            table.add_row(
                result['id'],
                result['title'],
                f"{result['score']:.3f}",
                ', '.join(result.get('tags', []))
            )

        console.print(table)

@retain.command('list')
@click.option('--collection', help='Limit to collection')
@click.option('--filter', 'filter_expr', help='Metadata filter')
@click.option('--local', is_flag=True, help='Force local retain')
@click.option('--global', 'use_global', is_flag=True, help='Force global retain')
@click.option('--format', type=click.Choice(['table', 'json']), default='table', help='Output format')
def list_command(collection, filter_expr, local, use_global, format):
    """List all memories.

    Examples:
      arc retain list
      arc retain list --collection knowledge
      arc retain list --filter "priority=high"
    """
    from arcaneum.retain.manager import RetainManager

    location = None
    if use_global:
        from arcaneum.paths import get_arcaneum_dir
        location = get_arcaneum_dir() / "retain"

    manager = RetainManager(location=location)

    memories = manager.list(collection=collection, filter_expr=filter_expr)

    if format == 'json':
        console.print_json(data=memories)
    else:
        if not memories:
            console.print("[yellow]No memories found[/yellow]")
            return

        table = Table(title=f"Memories ({len(memories)})")
        table.add_column("ID", style="cyan")
        table.add_column("Title")
        table.add_column("Collection")
        table.add_column("Tags")
        table.add_column("Created")

        for mem in memories:
            table.add_row(
                mem['id'],
                mem['title'],
                mem['collection'],
                ', '.join(mem.get('tags', [])),
                mem.get('created_at', 'Unknown')[:10]  # Just date
            )

        console.print(table)
```

### Implementation Example

**Basic Workflow**:

```bash
# Store a memory
echo "# GPU Research\n\nMetal vs CUDA..." | arc retain put - \
  --collection knowledge \
  --id gpu-patterns \
  --tags "gpu,performance,metal" \
  --context "Research for arcaneum-183"

# Retrieve for AI context
arc retain get gpu-patterns --format context

# Update as knowledge evolves
arc retain update gpu-patterns --content updated-research.md --merge-tags

# Search memories
arc retain search "GPU acceleration" --collection knowledge

# List all memories
arc retain list --filter "tags=gpu"

# Delete old memory
arc retain delete outdated-research
```

**Project-Local Workflow**:

```bash
# In project directory
cd ~/projects/myapp

# Create .arcaneum config
cat > .arcaneum <<EOF
version: 1
retain:
  enabled: true
  path: .arcaneum-retain
  history: false  # Git handles versioning
  collections:
    - memory
    - notes
EOF

# Store project-specific memory (auto-detects local)
arc retain put security-findings.md --collection memory --tags "security,myapp"

# Retrieve (auto-detects local)
arc retain get security-findings --format context

# Explicitly use global
arc retain get some-global-id --global
```

## Alternatives Considered

### Alternative 1: Extend `arc store` Instead of New `retain` Command

**Description**: Add subcommands to existing `store` command

**Pros**:

- No new top-level command
- Backward compatible
- Simpler CLI structure

**Cons**:

- `store` implies one-way operation (put only)
- Confusing to have `arc store get` (contradiction)
- Harder to deprecate later
- Less intuitive for CRUD operations

**Reason for rejection**: `retain` better communicates memory management concept. `store` can remain for simple
injection, `retain` for full CRUD.

### Alternative 2: Collection Per Operation (No Explicit Collection Flag)

**Description**: Use collection in ID: `knowledge:gpu-patterns`

**Pros**:

- More compact CLI
- ID uniquely identifies memory
- No need for `--collection` flag

**Cons**:

- Breaks filename conventions (colons problematic)
- Harder to organize files
- Complex ID parsing
- Not intuitive for users

**Reason for rejection**: Explicit `--collection` flag is clearer and more flexible.

### Alternative 3: Database Storage (SQLite) Instead of Files

**Description**: Store memories in SQLite database

**Pros**:

- Faster queries
- Better indexing
- Transactional updates
- Easier versioning

**Cons**:

- Not human-readable
- Not git-friendly
- Requires backup strategy
- Binary format (less portable)
- Contradicts markdown-native philosophy

**Reason for rejection**: Markdown files align with Arcaneum's philosophy: human-readable, git-trackable, portable.

### Alternative 4: Automatic Versioning Always Enabled

**Description**: Always track versions for all updates

**Pros**:

- Complete history
- Easy to restore
- Audit trail

**Cons**:

- Storage overhead (especially for local/git)
- Complexity in local git repos (double history)
- Performance impact
- Not needed for all use cases

**Reason for rejection**: Make versioning optional/future. Git handles local history, global can enable later.

## Trade-offs and Consequences

### Positive Consequences

1. **Complete CRUD API**: Full memory management (put, get, update, delete, search)
2. **Flexible Storage**: Global and project-local options
3. **Agent-Optimized**: Context-friendly output format for AI consumption
4. **Reuses Arc Infrastructure**: Leverages arc's dual-index (semantic + full-text)
5. **Git-Friendly**: Markdown files, git-aware history handling
6. **ID-Based Management**: Stable references, human-readable
7. **Backward Compatible**: Keep `store` command initially
8. **Extensible**: Designed for future versioning

### Negative Consequences

1. **Additional Complexity**: More code paths (global vs local storage)
   - *Mitigation*: Clear separation of concerns, wrapper architecture keeps retain simple
   - *Benefit*: Handles diverse use cases

2. **Corpus Naming Convention**: `retain-{collection}` may conflict
   - *Mitigation*: Document naming convention, validate on creation
   - *Benefit*: Clear namespace separation

3. **Two Storage Commands**: `store` and `retain` may confuse users
   - *Mitigation*: Document use cases, consider deprecation path
   - *Benefit*: Gradual migration, backward compatibility

4. **File-Based Limitations**: No transactions, slower queries
   - *Mitigation*: Optional in-memory index (`.index.json`)
   - *Benefit*: Human-readable, git-trackable

### Risks and Mitigations

**Risk**: ID collisions across collections

**Mitigation**:

- IDs are unique per collection (not globally)
- File structure: `{collection}/{id}.md` prevents collisions
- Validation on put: warn if ID exists in different collection

**Risk**: Large memory corpus impacts search performance

**Mitigation**:

- Corpus optimization (RDR-013): batch indexing, efficient chunking
- Collection-scoped searches reduce search space
- MeiliSearch handles full-text efficiently

**Risk**: Config file format confusion (YAML vs JSON)

**Mitigation**:

- Try YAML first, fallback to JSON
- Document recommended format (YAML)
- Validate on load, provide helpful errors

**Risk**: Local retain not found due to config path issues

**Mitigation**:

- Walk up directory tree (find nearest .arcaneum)
- Log detection logic in verbose mode
- Explicit `--local` and `--global` flags for override

## Implementation Plan

### Prerequisites

- [x] Markdown indexing implemented (RDR-014) - arc wraps this
- [x] Semantic search implemented (RDR-007) - arc wraps this
- [x] Collection typing (RDR-003) - arc uses this
- [ ] **arc CLI must be installed** (retain wraps arc commands)
- [ ] Path management helpers (extend paths.py)
- [ ] Python >= 3.12
- [ ] pyyaml >= 6.0 (for .arcaneum config)

### Step-by-Step Implementation

#### Step 1: Retain Manager Core (4 hours)

Create `src/arcaneum/retain/manager.py`:

- Implement `RetainManager` class (thin wrapper)
- Location detection (global vs local)
- Config loading (`.arcaneum`)
- ID generation and slugification
- File operations (read/write markdown with frontmatter)
- Basic CRUD methods calling arc wrappers
- Wrapper methods: `_call_arc_index()`, `_call_arc_search()`

#### Step 2: Memory Context Types (2 hours)

Create `src/arcaneum/retain/types.py`:

- Implement `MemoryContext` dataclass
- Output formatters (context, json, raw)
- Metadata schema definition

#### Step 3: Search and List Operations (2 hours)

Extend `RetainManager`:

- Search method wrapping `arc search` (semantic + full-text)
- List method (local file enumeration)
- Filter expression parsing
- Result formatting from arc JSON output

#### Step 4: CLI Commands (4 hours)

Create `src/arcaneum/cli/retain.py`:

- `put` command with all options (delegates to RetainManager)
- `get` command with format options
- `update` command
- `delete` command (no purge-index option)
- `search` command
- `list` command
- Rich output formatting
- JSON output support

#### Step 5: CLI Integration (2 hours)

Modify `src/arcaneum/cli/main.py`:

- Register `retain` command group
- Add help text
- Update main CLI documentation

#### Step 6: Config File Support (2 hours)

Create `src/arcaneum/config.py`:

- `.arcaneum` file format definition (simplified for wrapper)
- YAML/JSON parsing
- Config validation
- Directory tree walking for config discovery

#### Step 7: Path Management (2 hours)

Extend `src/arcaneum/paths.py`:

- `get_retain_dir()` helper
- Local retain path resolution
- Config-aware path detection

#### Step 8: Store Command Compatibility (3 hours)

Decide on `store` command fate:

- Keep as-is with deprecation notice
- Redirect to `retain put` internally
- Document migration path
- Update help text

#### Step 9: Testing (8 hours)

Create comprehensive tests:

- Unit tests for `RetainManager` (wrapper methods)
- Unit tests for location detection
- Unit tests for ID generation
- Unit tests for file operations
- Unit tests for config loading
- Integration test: global retain workflow
- Integration test: local retain workflow
- Integration test: arc wrapper integration (mock arc commands)
- Integration test: search operations (via arc)
- Integration test: git versioning (local)
- Edge cases: ID collisions, missing config, arc command failures

#### Step 10: Documentation (4 hours)

Create/update documentation:

- `docs/guides/retain-memory-management.md` - Complete guide (wrapper approach)
- Update README with retain examples
- CLI reference for `arc retain`
- `.arcaneum` config file format
- Migration guide from `store` to `retain`
- Agent integration examples
- Note: arc dependency for indexing/search

#### Step 11: Git Integration for Versioning (4 hours)

Implement git-based versioning for local retains:

- Git command wrappers (log, show, diff) via subprocess
- Version detection (is git repo?)
- Simplified version API (git-only for local, optional for global)
- Error handling (git not available, not a repo)

### Total Estimated Effort

**31 hours** (~4 days of focused work)

**Breakdown** (Thin wrapper approach):

- Core manager: 4h (wrapper implementation)
- Types: 2h (unchanged)
- Search/list: 2h (arc wrapper calls)
- CLI commands: 4h (delegation to RetainManager)
- CLI integration: 2h (unchanged)
- Config support: 2h (simpler config)
- Path management: 2h (unchanged)
- Store compatibility: 3h (unchanged)
- Git versioning integration: 4h (subprocess wrappers)
- Testing: 8h (fewer integration tests)
- Documentation: 4h (simpler architecture)

### Files to Create

**Note on Repository Structure**:

This RDR shows retain as part of the arc repository (`src/arcaneum/retain/`). Alternatively,
retain could be a separate plugin repository with its own `pyproject.toml`. Benefits:

- **Same repo** (recommended for v1): Simpler development, shared testing, single release
- **Separate repo**: Independent versioning, smaller footprint, clearer separation

For this implementation, retain lives in the arc repository but is architecturally independent
(wraps arc via CLI, not Python imports).

**Core Modules**:

- `src/arcaneum/retain/__init__.py`
- `src/arcaneum/retain/manager.py` - Core memory management (wrapper class)
- `src/arcaneum/retain/types.py` - Memory context types
- `src/arcaneum/config.py` - Config file handling (shared with arc)

**CLI**:

- `src/arcaneum/cli/retain.py` - CLI commands

**Tests**:

- `tests/unit/retain/test_manager.py` (wrapper methods)
- `tests/unit/retain/test_types.py`
- `tests/unit/retain/test_config.py`
- `tests/unit/retain/test_git_integration.py`
- `tests/integration/test_retain_global.py`
- `tests/integration/test_retain_local.py`
- `tests/integration/test_retain_arc_integration.py` (mock arc commands)
- `tests/integration/test_retain_versioning.py`

**Documentation**:

- `docs/guides/retain-memory-management.md`
- `docs/reference/arcaneum-config-format.md`

### Files to Modify

- `src/arcaneum/cli/main.py` - Register retain command
- `src/arcaneum/paths.py` - Add retain path helpers
- `docs/rdr/README.md` - Update RDR index
- `README.md` - Add retain examples
- `pyproject.toml` - Add pyyaml dependency

### Dependencies

**System Requirements**:

- **arc CLI** (required) - retain wraps arc commands via subprocess
- Python >= 3.12
- Git (optional, for local retain versioning)

**Python Dependencies** - Add to `pyproject.toml`:

```toml
[project.dependencies]
# Retain-specific dependencies
pyyaml = ">=6.0"                # .arcaneum config parsing
python-frontmatter = ">=1.1.0"  # Markdown frontmatter handling
rich = ">=13.0"                 # CLI output formatting
click = ">=8.1"                 # CLI framework

# Note: Heavy dependencies (fastembed, sentence-transformers, qdrant-client)
# are provided by arc, not duplicated here.
```

## Validation

### Testing Approach

**Unit Tests**:

- Retain manager CRUD operations (file I/O)
- Arc wrapper methods (`_call_arc_index`, `_call_arc_search`)
- ID generation (title, H1, hash)
- Location detection (global vs local)
- Config file parsing (YAML, JSON)
- Filter expression parsing
- Memory context formatting

**Integration Tests** (wrapper-focused):

- End-to-end global retain workflow
- End-to-end local retain workflow
- Arc wrapper integration (mock subprocess calls to arc)
- Search operations via arc (mock `arc search` responses)
- Update and reindex via arc (verify arc called)
- Delete flow (file removal, no corpus test)

**Edge Case Tests**:

- ID collisions
- Missing .arcaneum config
- Invalid config format
- Malformed markdown
- Empty content
- Arc command failures (subprocess errors)

### Test Scenarios

#### Scenario 1: Global Retain Basic Workflow

**Setup**: Clean global retain directory
**Actions** (in Claude Code):

1. `/retain:put research.md --collection knowledge --id gpu-patterns`
   - Calls: `arc retain put ...` → writes file → `arc index markdown ...`
2. `/retain:get gpu-patterns --format context`
   - Calls: `arc retain get ...` → reads file → formats as MemoryContext string
3. `/retain:update gpu-patterns --tags "gpu,metal"`
   - Calls: `arc retain update ...` → updates file → `arc index markdown ...`
4. `/retain:search "GPU" --collection knowledge`
   - Calls: `arc retain search ...` → `arc search semantic "GPU" --collection retain-knowledge --json`
5. `/retain:delete gpu-patterns`
   - Calls: `arc retain delete ...` → deletes file (index cleanup on next sync)

**Expected**: All slash commands succeed, arc handles all indexing/search

#### Scenario 2: Project-Local Retain

**Setup**: Project with `.arcaneum` config
**Actions** (in Claude Code, within project directory):

1. Create `.arcaneum` with retain config
2. `/retain:put notes.md --collection memory`
3. `/retain:get <id>` (auto-detects local)
4. `/retain:list --local`

**Expected**: Memories stored in `.arcaneum-retain/`, not global

#### Scenario 3: Search Across Collections

**Setup**: Multiple collections with memories
**Actions** (in Claude Code):

1. Put memories in `knowledge`, `memory`, `research` collections
2. `/retain:search "security"` (all collections)
   - Calls: `arc retain search ...` → `arc search semantic "security" --collection "retain-*" --json`
3. `/retain:search "security" --collection knowledge` (scoped)
   - Calls: `arc search semantic "security" --collection retain-knowledge --json`

**Expected**: Relevant results returned, scoped search faster (arc optimizes collection queries)

#### Scenario 4: Context-Optimized Get

**Setup**: Memory with full metadata
**Actions** (in Claude Code):

1. Put memory with tags, category, context, related_to
2. `/retain:get <id> --format context`

**Expected**: Output includes content + contextual metadata, excludes low-level details

#### Scenario 5: Update Without Content Change

**Setup**: Existing memory
**Actions** (in Claude Code):

1. `/retain:update <id> --tags "new,tags" --merge-tags`
2. `/retain:get <id>`

**Expected**: Tags updated, content unchanged, updated_at timestamp set

#### Scenario 6: Git-Based Versioning (Local)

**Setup**: Project with `.arcaneum` config and git repo
**Actions** (in Claude Code):

1. `/retain:put research.md --collection knowledge --id gpu-patterns`
2. Git commits the change
3. Update memory: `/retain:update gpu-patterns --content updated.md`
4. Git commits the update
5. `/retain:versions gpu-patterns` (shows git log)
6. `/retain:get gpu-patterns --version HEAD~1` (get previous version)
7. `/retain:diff gpu-patterns --from HEAD~1 --to HEAD`

**Expected**: All operations leverage git, version history from git log, seamless integration

### Performance Validation

**Metrics**:

- Put operation latency (file write + index)
- Get operation latency (file read + parse)
- Search latency (corpus query)
- List operation latency (file scan)

**Targets**:

- Put: < 500ms (including indexing)
- Get: < 50ms
- Search: < 200ms (10 results)
- List: < 100ms (100 memories)

### Security Validation

**Config Security**:

- Validate `.arcaneum` path doesn't escape project root
- Sanitize collection names (no path traversal)
- Validate ID format (no special characters)

**Storage Security**:

- Ensure retain directory permissions (700)
- Validate file paths before write
- Sanitize slugified IDs

## References

### Related RDRs

- [RDR-014: Markdown Indexing](RDR-014-markdown-indexing.md) - Markdown pipeline, injection handler
- [RDR-009: Dual Indexing Strategy](RDR-009-dual-indexing-strategy.md) - Corpus architecture
- [RDR-003: Collection Creation](RDR-003-collection-creation.md) - Collection typing
- [RDR-002: Qdrant Setup](RDR-002-qdrant-server-setup.md) - Vector database

### External Resources

- **LangChain Memory**: <https://python.langchain.com/docs/modules/memory/>
- **Semantic Kernel Memory**: <https://github.com/microsoft/semantic-kernel>
- **MemGPT**: <https://github.com/cpacker/MemGPT>
- **python-frontmatter**: <https://python-frontmatter.readthedocs.io/>
- **PyYAML**: <https://pyyaml.org/wiki/PyYAMLDocumentation>

## Notes

### Implementation Priority

1. **Core retain manager** (CRUD operations, wrapper methods)
2. **CLI commands** (user-facing API)
3. **Arc integration** (wrapper for indexing/search via subprocess)
4. **Local retain** (project-specific storage)
5. **Context-optimized get** (AI-friendly output)
6. **Search operations** (memory retrieval via arc)

### Future Enhancements

**Enhanced Versioning**:

- Semantic version tagging (v1.0.0, v1.1.0)
- Version comparison with visual diff
- Automatic versioning policies (time-based, change-based)
- Version pruning for global storage

**Advanced Search**:

- Fuzzy search
- Date range filters
- Similarity threshold tuning
- Multi-collection search with ranking

**Memory Consolidation**:

- Merge similar memories
- Summarize old memories
- Archival policies (auto-archive old memories)

**Collaboration**:

- Share memories across team
- Remote retain storage
- Sync protocol

### Success Criteria

- ✅ Full CRUD API (put, get, update, delete, search, list)
- ✅ Global and local retain support
- ✅ Context-optimized get output
- ✅ Arc wrapper integration (> 90% code reuse)
- ✅ ID-based memory management
- ✅ Config file support (`.arcaneum`)
- ✅ Git-integrated versioning (local retains)
- ✅ Agent-friendly (JSON output)
- ✅ < 35 hours implementation (wrapper approach)
- ✅ Comprehensive test coverage (wrapper integration tests)
- ✅ Markdownlint compliant documentation
- ✅ Separate plugin architecture (can toggle on/off)

This RDR provides a complete specification for a lightweight memory management system that enables AI agents to store,
retrieve, and organize long-term memories by wrapping arc's indexing and search infrastructure with memory-specific
conventions. Supports both global and project-local use cases.

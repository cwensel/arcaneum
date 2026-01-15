# Recommendation 011: Git-Aware Source Code Full-Text Indexing to MeiliSearch

## Metadata

- **Date**: 2025-10-27
- **Updated**: 2026-01-15
- **Status**: Recommendation
- **Type**: Feature
- **Priority**: High
- **Related Issues**: arcaneum-70, arcaneum-86 through arcaneum-92, arcaneum-vau8,
  arcaneum-03wm, arcaneum-40m1, arcaneum-y53c, arcaneum-efmz, arcaneum-xoiq,
  arcaneum-e2gn
- **Related Tests**: Source code full-text indexing tests, dual indexing integration tests

## Problem Statement

Create a production-ready source code indexing system for MeiliSearch that enables
exact phrase search and keyword matching across source code with git awareness and
function/class-level precision. The system must:

1. **Index at function/class granularity** - Enable precise location results like "found in calculate_total() lines 42-67"
2. **Git awareness** - Multi-branch support with same composite identifier pattern as RDR-005
3. **Dual indexing** - Integrate with RDR-005 vector indexing for efficient parallel indexing
4. **Change detection** - Metadata-based sync using MeiliSearch as single source of truth
5. **Exact search** - Support phrase matching, regex, and keyword searches
   complementary to RDR-005 semantic search

This addresses the need for exact string matching in code ("find def authenticate")
complementary to semantic search ("find authentication patterns") from RDR-005.

## Context

### Background

Arcaneum provides semantic search via Qdrant (RDR-005) but requires complementary full-text search for:

- **Exact string matching**: Find literal code patterns like `"class UserAuth"`
- **Regex searches**: Complex patterns for API names
- **Function/class lookup**: Find exact identifiers without semantic ambiguity
- **Line-level precision**: Return file.py lines 42-67 for exact locations

**Expected Workflow**:

1. User: "Find authentication patterns" → Claude uses **semantic search** (RDR-005)
2. User: "Find exact string 'def authenticate'" → Claude uses **full-text search** (this RDR)

**Parallel to RDR-010**: Just as RDR-010 indexes PDFs to MeiliSearch (complementary
to RDR-004 vector), this RDR indexes source code to MeiliSearch (complementary to
RDR-005 vector).

### Technical Environment

- **Python**: >= 3.12
- **MeiliSearch**: v1.32.x (from RDR-008, updated 2026-01-14)
- **Git**: >= 2.30 (for metadata extraction)
- **AST Parsing**:
  - tree-sitter-language-pack >= 0.10.0 (130+ languages, from RDR-005)
  - **Note**: LlamaIndex CodeSplitter is NOT used for function extraction (it's
    for chunking). We use tree-sitter directly via `get_parser()`.
- **MeiliSearch Client**:
  - meilisearch-python >= 0.31.0 (from RDR-008)
- **Supporting Libraries**:
  - GitPython >= 3.1.45 (from RDR-005)
  - tenacity (retry logic)
  - tqdm/rich (progress tracking)

### CLI Naming Convention (Updated 2026-01-15)

This RDR follows the symmetric CLI naming convention established for semantic/text
operations (see arcaneum-h6bo in RDR-010):

```bash
# Search commands (existing)
arc search semantic "query" --collection X    # Vector search (Qdrant)
arc search text "query" --index X             # Full-text search (MeiliSearch)

# Index commands (symmetric naming)
arc index source /path --collection X         # Semantic indexing (Qdrant) - RDR-005
arc index text pdf /path --index X            # Full-text PDF (MeiliSearch) - RDR-010
arc index text code /path --index X           # Full-text code (MeiliSearch) - THIS RDR

# Dual indexing (RDR-009 pattern)
arc corpus create NAME --type code --model MODEL
arc corpus sync /path --corpus NAME           # Indexes to BOTH Qdrant and MeiliSearch

# Management commands
arc collection list/create/delete             # Qdrant collection management
arc indexes list/create/delete                # MeiliSearch index management
```

**Note**: The `arc fulltext` command group was renamed to `arc indexes` per
arcaneum-h6bo. This RDR uses the updated naming.

## Research Findings

### Investigation Process

**Research tracks completed via Beads issues** (arcaneum-86 through arcaneum-92):

1. **Indexing Granularity** (arcaneum-86): Analyzed whole-file vs line-based vs function/class-level
2. **Metadata Schema** (arcaneum-87): Designed MeiliSearch document structure aligned with RDR-005
3. **Change Detection** (arcaneum-88): Verified MeiliSearch filter-based deletion for git-aware sync
4. **Function/Class Extraction** (arcaneum-89): Confirmed tree-sitter AST approach from RDR-005
5. **Dual Indexing Workflow** (arcaneum-90): Evaluated integration patterns with RDR-005
6. **Batch Upload Optimization** (arcaneum-91): Determined optimal batch sizes for MeiliSearch
7. **CLI Command Structure** (arcaneum-92): Designed command extensions for consistency

### Key Discoveries

#### 1. Function/Class-Level Granularity (arcaneum-86)

**Decision**: Index at function/class level with line ranges

**Rationale**:

- Provides precise location info: "found in calculate_total() lines 42-67"
- Manageable document count: ~5-50 documents per file (not thousands)
- Natural code boundaries: functions/classes are semantic units
- Leverages existing tree-sitter from RDR-005
- Mirrors RDR-010's page-level pattern adapted for code structure

**Edge Cases Handled**:

- Module-level code (not in functions): Create single "module" document covering all non-function code
- Nested functions: Index as separate documents with qualified names
- Large functions (>200 lines): Index as-is (rare in practice)

#### 2. Git-Aware Metadata Schema (arcaneum-87)

**MeiliSearch Document Structure**:

```python
{
  # Primary key (unique across branches, files, functions)
  "id": "{git_project_identifier}:{file_path}:{qualified_name}:{start_line}",

  # Searchable content
  "content": "def calculate_total(items):\n    return sum(...)",
  "function_name": "calculate_total",
  "class_name": None,  # or class name if applicable
  "qualified_name": "utils.calculate_total",
  "filename": "utils.py",

  # Git metadata (from RDR-005 pattern)
  "git_project_identifier": "arcaneum#main",
  "git_project_name": "arcaneum",
  "git_branch": "main",
  "git_commit_hash": "abc123...",  # Full 40-char SHA
  "git_remote_url": "https://github.com/user/arcaneum",
  "file_path": "/path/to/src/utils.py",

  # Location metadata (function/class-level)
  "start_line": 42,
  "end_line": 67,
  "line_count": 26,
  "code_type": "function",  # function, class, method, module

  # Language
  "programming_language": "python"
}
```

**MeiliSearch Index Settings** (Updated 2026-01-15):

```python
# src/arcaneum/fulltext/indexes.py
# NOTE: This is a NEW settings dict for function-level indexing.
# The existing SOURCE_CODE_SETTINGS is for chunk-level (RDR-005 style).

SOURCE_CODE_FULLTEXT_SETTINGS = {
    "searchableAttributes": [
        "content",           # Primary search field (function/class code)
        "function_name",     # For identifier search (single name, not array)
        "class_name",        # For class search (single name, not array)
        "qualified_name",    # For fully-qualified searches (e.g., "MyClass.method")
        "filename",          # For file-specific searches
    ],
    "filterableAttributes": [
        "programming_language",     # Language filter (aligned with RDR-005)
        "git_project_name",         # Project name only
        "git_branch",               # Branch name
        "git_project_identifier",   # Composite "project#branch" for branch-specific queries
        "git_commit_hash",          # For change detection (NEW)
        "file_path",                # File location
        "code_type",                # "function", "class", "method", "module" (NEW)
        "file_extension",           # For file type filtering
    ],
    "sortableAttributes": [
        "start_line",  # CRITICAL: For sorting by location in file
    ],
    "typoTolerance": {
        "enabled": True,
        "minWordSizeForTypos": {
            "oneTypo": 7,   # Higher threshold for code identifiers
            "twoTypos": 12
        }
    },
    "stopWords": [],  # Preserve all code keywords (def, class, function, etc.)
    "pagination": {
        "maxTotalHits": 10000  # Higher than default: function-level = more docs per file
    }
}
```

**Note**: This differs from the existing `SOURCE_CODE_SETTINGS` in `fulltext/indexes.py`
which is designed for chunk-level indexing. This RDR proposes function-level granularity
requiring different fields (single function_name vs array, code_type for definition type,
start_line for precise location).

#### 3. Git Change Detection Strategy (arcaneum-88)

**Pattern**: Exact mirror of RDR-005 metadata-based sync

**Implementation**:

```python
class GitMetadataSync:
    """Query MeiliSearch for indexed projects (source of truth)."""

    def get_indexed_projects(self, index_name: str) -> Dict[str, str]:
        """Get all (git_project_identifier, git_commit_hash) from MeiliSearch."""
        # Query MeiliSearch for distinct projects
        documents = self.meili_client.get_documents(
            index_name=index_name,
            attributes_to_retrieve=[
                'git_project_identifier',
                'git_commit_hash'
            ],
            limit=10000
        )

        indexed = {}
        for doc in documents:
            identifier = doc['git_project_identifier']
            commit = doc['git_commit_hash']
            if identifier not in indexed:
                indexed[identifier] = commit

        return indexed

    def should_reindex_project(
        self,
        index_name: str,
        project_identifier: str,
        current_commit: str
    ) -> bool:
        """Check if (project, branch) needs re-indexing."""
        indexed = self.get_indexed_projects(index_name)

        # Not indexed yet
        if project_identifier not in indexed:
            return True

        # Commit changed
        if indexed[project_identifier] != current_commit:
            return True

        # Unchanged
        return False
```

**Branch-Specific Deletion**:

```python
def delete_branch_documents(
    self,
    index_name: str,
    project_identifier: str
):
    """Delete all documents for specific (project, branch)."""
    # MeiliSearch filter-based deletion
    index = self.meili_client.get_index(index_name)
    index.delete_documents(
        filter=f'git_project_identifier = "{project_identifier}"'
    )
    logger.info(f"Deleted documents for {project_identifier}")
```

**Benefits**:

- MeiliSearch is single source of truth (can't get out of sync)
- Idempotent reindexing (crash recovery automatic)
- Branch-level deletion (other branches unaffected)
- Same pattern as RDR-005 for consistency

#### 4. Tree-Sitter Function/Class Extraction (arcaneum-89, Updated 2026-01-15)

**IMPORTANT CLARIFICATION**: This RDR uses tree-sitter DIRECTLY for function/class
extraction, NOT LlamaIndex CodeSplitter. CodeSplitter is designed for chunking
(creating overlapping text chunks for embeddings) and does NOT provide function
names, qualified names, or line ranges.

**Difference from RDR-005's ast_chunker.py**:

| Component  | ast_chunker.py (RDR-005)                       | ast_extractor.py (THIS RDR)                  |
| ---------- | ---------------------------------------------- | -------------------------------------------- |
| Purpose    | Create overlapping chunks for embeddings       | Extract discrete function/class definitions  |
| Output     | `List[Chunk]` (content + method)               | `List[CodeDefinition]` (name, lines, type)   |
| Boundaries | May split mid-function for optimal embed size  | Exact function/class boundaries              |
| Used by    | Semantic search (Qdrant)                       | Full-text search (MeiliSearch)               |

**Approach**: Use tree-sitter-language-pack directly via `get_parser()` and traverse
the AST to find function/class definition nodes.

**Key tree-sitter Node Properties**:

```python
# Node properties for extraction (from py-tree-sitter)
node.type          # "function_definition", "class_definition", etc.
node.start_point   # Point(row, column) - 0-indexed line number
node.end_point     # Point(row, column) - 0-indexed line number
node.text          # bytes - actual code content
node.children      # list[Node] - child nodes
node.child_by_field_name("name")  # Get identifier node
```

**Function/Class Extraction Implementation**:

```python
from tree_sitter_language_pack import get_parser
from tree_sitter import Node
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CodeDefinition:
    """Function/class definition with location."""
    name: str
    qualified_name: str
    code_type: str  # "function", "class", "method", "module"
    start_line: int  # 1-indexed for user display
    end_line: int
    content: str
    file_path: str

# Language -> node types that represent definitions
DEFINITION_TYPES = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
    },
    "javascript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
    },
    "typescript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "interface_declaration": "interface",
    },
    "java": {
        "method_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "class",  # structs
    },
    "rust": {
        "function_item": "function",
        "impl_item": "class",
        "struct_item": "class",
        "trait_item": "interface",
    },
    # ... 130+ languages supported
}

class ASTFunctionExtractor:
    """Extract function/class definitions using tree-sitter directly."""

    # Language mapping: file extension -> tree-sitter language name
    LANGUAGE_MAP = {
        ".py": "python",
        ".java": "java",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".cpp": "cpp",
        ".cs": "c_sharp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        # ... (reuse LANGUAGE_MAP from ast_chunker.py)
    }

    def extract_definitions(
        self,
        file_path: str,
        code: str
    ) -> List[CodeDefinition]:
        """
        Extract function/class definitions with line ranges.

        Uses tree-sitter directly (NOT LlamaIndex CodeSplitter).

        Returns list of CodeDefinition objects.
        Fallback to module-level if AST parsing fails.
        """
        import os
        file_ext = os.path.splitext(file_path)[1].lower()
        language = self.LANGUAGE_MAP.get(file_ext)

        if not language or language not in DEFINITION_TYPES:
            # Fallback: entire file as "module"
            return [self._create_module_definition(file_path, code)]

        try:
            parser = get_parser(language)
            tree = parser.parse(bytes(code, "utf8"))

            definitions = []
            self._extract_from_node(
                tree.root_node,
                code,
                language,
                file_path,
                definitions,
                parent_name=None
            )

            # Add module-level code if any
            module_code = self._extract_module_code(code, definitions, file_path)
            if module_code:
                definitions.append(module_code)

            return definitions if definitions else [self._create_module_definition(file_path, code)]

        except Exception as e:
            logger.warning(f"AST extraction failed for {file_path}: {e}")
            return [self._create_module_definition(file_path, code)]

    def _extract_from_node(
        self,
        node: Node,
        code: str,
        language: str,
        file_path: str,
        definitions: List[CodeDefinition],
        parent_name: Optional[str]
    ):
        """Recursively extract definitions from AST nodes."""
        def_types = DEFINITION_TYPES.get(language, {})

        if node.type in def_types:
            # Get the name from the identifier child
            name_node = node.child_by_field_name("name")
            name = name_node.text.decode("utf8") if name_node else "anonymous"

            # Build qualified name (e.g., "MyClass.method")
            qualified = f"{parent_name}.{name}" if parent_name else name

            # Line numbers: tree-sitter uses 0-indexed, convert to 1-indexed
            start_line = node.start_point.row + 1
            end_line = node.end_point.row + 1

            definitions.append(CodeDefinition(
                name=name,
                qualified_name=qualified,
                code_type=def_types[node.type],
                start_line=start_line,
                end_line=end_line,
                content=node.text.decode("utf8"),
                file_path=file_path
            ))

            # Recurse into nested definitions (e.g., methods in classes)
            for child in node.children:
                self._extract_from_node(
                    child, code, language, file_path,
                    definitions, parent_name=qualified
                )
        else:
            # Not a definition node, but might contain definitions
            for child in node.children:
                self._extract_from_node(
                    child, code, language, file_path,
                    definitions, parent_name=parent_name
                )

    def _create_module_definition(
        self,
        file_path: str,
        code: str
    ) -> CodeDefinition:
        """Create a module-level definition for the entire file."""
        lines = code.splitlines()
        return CodeDefinition(
            name="module",
            qualified_name="module",
            code_type="module",
            start_line=1,
            end_line=len(lines) if lines else 1,
            content=code,
            file_path=file_path
        )
```

**Fallback Strategy**:

- If AST parsing fails: Index entire file as single "module" document
- If language not in DEFINITION_TYPES: Index as "module"
- Ensures all code is indexed even if tree-sitter fails
- Graceful degradation with warning logs

#### 5. Dual Indexing Workflow (arcaneum-90, Updated 2026-01-15)

**IMPORTANT**: RDR-009 establishes `arc corpus sync` as THE dual indexing command.
This RDR does NOT propose a new dual-indexing pathway. Instead:

- **PRIMARY**: Standalone `arc index text code` command (MeiliSearch only)
- **SECONDARY**: Ensure compatibility with `arc corpus sync` (RDR-009)

**Pattern**: Use RDR-009's established dual indexing via `arc corpus sync`

**Workflow**:

```bash
# Dual indexing via RDR-009 (RECOMMENDED)
arc corpus create MyCode --type code --model stella
arc corpus sync ./src --corpus MyCode
# Creates: Qdrant "MyCode" collection + MeiliSearch "MyCode" index

# Standalone full-text only (this RDR)
arc index text code ./src --index MyCode-fulltext
```

**Implementation** (integrated with RDR-009's DualIndexer):

```text
┌─────────────────────────────────────────────────────────────┐
│     arc corpus sync (RDR-009 Dual Indexing)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 │   SHARED STAGES         │
                 │  (Single Pass)          │
                 └────────────┬────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    Git Discovery      Read Files          Parse AST
    (RDR-005)         (shared)            (tree-sitter)
          │                   │                   │
          └───────────────────┴───────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
    RDR-005: Generate                    RDR-011: Extract
    embedding chunks                     function/class definitions
    (via ast_chunker.py)                 (via ast_extractor.py)
          │                                       │
    Upload to Qdrant                      Upload to MeiliSearch
    (via DualIndexer)                     (via DualIndexer)
```

**Benefits**:

- Single pass over codebase (efficient)
- Consistent git metadata across both systems
- Uses established RDR-009 DualIndexer infrastructure
- No new dual-indexing command to maintain

**Comparison to RDR-010 (PDF)**:

- RDR-010: `arc index text pdf` is standalone (no embedding generation)
- RDR-011: `arc index text code` is also standalone (this RDR)
- Dual indexing: Both use `arc corpus sync` (RDR-009)

**Scope Clarification**:

- This RDR focuses on the **standalone** `arc index text code` command
- Dual indexing is handled by RDR-009's `arc corpus sync`
- Ensure `DualIndexDocument` schema supports function-level fields (already does)

#### 6. Batch Upload Optimization (arcaneum-91)

**Decision**: batch_size=1000 with per-batch task waiting

**Rationale**:

- Function/class documents: ~2-5KB average
- 1000 docs × 5KB = ~5MB per batch (well under MEILI_MAX_INDEXING_MEMORY of 2.5GiB)
- Same as RDR-010 (proven for PDFs)
- Sweet spot for throughput vs memory

**Implementation**:

```python
def upload_documents_batch(
    self,
    index_name: str,
    documents: List[Dict[str, Any]],
    batch_size: int = 1000
):
    """Upload documents in batches with progress tracking."""
    index = self.meili_client.get_index(index_name)

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]

        # Upload batch
        task_info = index.add_documents(batch)

        # Wait for this batch to complete
        self.meili_client.wait_for_task(task_info['taskUid'])

        # Update progress
        logger.info(f"Indexed batch {i//batch_size + 1}: {len(batch)} documents")
```

**Benefits**:

- Early error detection (fail fast)
- Progress tracking per batch
- Prevents memory buildup
- User sees incremental progress

#### 7. CLI Command Structure (arcaneum-92, Updated 2026-01-15)

**IMPORTANT**: CLI command naming updated per arcaneum-h6bo. The `arc fulltext`
command group was renamed to `arc indexes`.

**Dual Indexing** (use RDR-009's `arc corpus sync`):

```bash
# Create corpus (both Qdrant collection + MeiliSearch index)
arc corpus create MyCode --type code --model stella

# Sync directory to both systems
arc corpus sync ./src --corpus MyCode
# Result: Qdrant "MyCode" collection + MeiliSearch "MyCode" index
```

**Full-Text Only** (new command, this RDR):

```bash
# Standalone MeiliSearch indexing (no embeddings)
arc index text code ./src --index MyCode-fulltext
```

**Search** (existing from RDR-008):

```bash
# Full-text search
arc search text '"def authenticate"' \
  --index MyCode-fulltext \
  --filter 'programming_language = python AND git_branch = main'
```

**Index Management** (uses `arc indexes`, not `arc fulltext`):

```bash
# List MeiliSearch indexes
arc indexes list

# Create index with source code settings
arc indexes create MyCode-fulltext --type source-code

# Delete index
arc indexes delete MyCode-fulltext

# Show index info
arc indexes info MyCode-fulltext
```

**Project Management within Index** (new commands for this RDR):

```bash
# List indexed git projects with commit hashes
arc indexes list-projects --index MyCode-fulltext

# Delete specific project/branch from index
arc indexes delete-project arcaneum#main --index MyCode-fulltext
```

## Proposed Solution

### Approach

Implement a **git-aware source code full-text indexing pipeline** that:

1. **Discovers git projects** using RDR-005's discovery logic
2. **Extracts function/class definitions** using tree-sitter AST parsing
3. **Indexes at function/class granularity** to MeiliSearch with line ranges
4. **Supports multi-branch** via same composite identifier pattern as RDR-005
5. **Detects changes** via metadata-based sync (MeiliSearch as source of truth)
6. **Deletes efficiently** using filter-based branch-specific deletion
7. **Integrates with RDR-005** for parallel dual indexing in single pass

### Technical Design

#### Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│       Source Code Full-Text Indexing Pipeline               │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
       Git Discovery              Query MeiliSearch Metadata
       (RDR-005 reuse)            (source of truth)
            │                               │
    find .git dirs              get indexed (project#branch,
    + extract metadata          commit) tuples
            │                               │
            └───────────────┬───────────────┘
                            │
                   ┌────────▼─────────┐
                   │  Compare Commits │
                   │  git HEAD vs     │
                   │  MeiliSearch     │
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │ Changed?         │
                   │ Yes: Delete &    │
                   │      Re-index    │
                   │ No: Skip         │
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────────────┐
                   │ Get Files (git ls-files) │
                   └────────┬─────────────────┘
                            │
                   ┌────────▼────────────────────┐
                   │  AST Function/Class Extract │
                   │  (tree-sitter, RDR-005)     │
                   └────────┬────────────────────┘
                            │
                   ┌────────▼────────────────────┐
                   │  Build MeiliSearch Docs     │
                   │  (function/class-level)     │
                   └────────┬────────────────────┘
                            │
                   ┌────────▼────────────────────┐
                   │  Batch Upload (1000 docs)   │
                   │  (wait per batch)           │
                   └─────────────────────────────┘
```

#### Core Components

**1. Git Operations** (reuse from RDR-005):

```python
# Reuse RDR-005's GitProjectDiscovery
from ..indexing.git_operations import GitProjectDiscovery

# Same git metadata extraction
git_metadata = GitProjectDiscovery().extract_metadata(project_root)
# Returns: project_name, branch, commit_hash, remote_url
```

**2. AST Function/Class Extractor**:

```python
# src/arcaneum/indexing/fulltext/ast_extractor.py

from tree_sitter_language_pack import get_parser
from typing import List
from dataclasses import dataclass

@dataclass
class CodeDefinition:
    """Function/class definition with location."""
    name: str
    qualified_name: str
    code_type: str  # "function", "class", "method", "module"
    start_line: int
    end_line: int
    content: str
    file_path: str

class ASTFunctionExtractor:
    """Extract function/class definitions using tree-sitter."""

    def extract_definitions(
        self,
        file_path: str,
        code: str
    ) -> List[CodeDefinition]:
        """
        Extract function/class definitions with line ranges.

        Returns list of CodeDefinition objects.
        Fallback to module-level if AST parsing fails.
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        language = self.LANGUAGE_MAP.get(file_ext)

        if not language:
            return [self._create_module_definition(file_path, code)]

        try:
            parser = get_parser(language)
            tree = parser.parse(bytes(code, "utf8"))

            # Language-specific extraction
            definitions = self._extract_by_language(
                language, tree, code, file_path
            )

            # Add module-level code
            module_code = self._extract_module_code(
                code, definitions, file_path
            )
            if module_code:
                definitions.append(module_code)

            return definitions

        except Exception as e:
            logger.warning(f"AST failed for {file_path}: {e}, using module-level")
            return [self._create_module_definition(file_path, code)]
```

**3. Metadata-Based Sync** (mirrors RDR-005):

```python
# src/arcaneum/indexing/fulltext/sync.py

class GitMetadataSync:
    """Query MeiliSearch for indexed projects (source of truth)."""

    def __init__(self, meili_client: FullTextClient):
        self.meili_client = meili_client
        self._cache = {}

    def get_indexed_projects(self, index_name: str) -> Dict[str, str]:
        """Get all (git_project_identifier, git_commit_hash) pairs."""
        if index_name in self._cache:
            return self._cache[index_name]

        # Query MeiliSearch
        index = self.meili_client.get_index(index_name)
        documents = index.get_documents(
            attributes_to_retrieve=[
                'git_project_identifier',
                'git_commit_hash'
            ],
            limit=10000
        )

        indexed = {}
        for doc in documents['results']:
            identifier = doc['git_project_identifier']
            commit = doc['git_commit_hash']
            if identifier not in indexed:
                indexed[identifier] = commit

        self._cache[index_name] = indexed
        return indexed
```

**4. MeiliSearch Indexer**:

```python
# src/arcaneum/indexing/fulltext/code_indexer.py

class SourceCodeFullTextIndexer:
    """Index source code to MeiliSearch at function/class level."""

    def __init__(
        self,
        meili_client: FullTextClient,
        index_name: str,
        batch_size: int = 1000
    ):
        self.meili_client = meili_client
        self.index_name = index_name
        self.batch_size = batch_size

        # Reuse RDR-005 components
        self.git_discovery = GitProjectDiscovery()
        self.ast_extractor = ASTFunctionExtractor()
        self.sync = GitMetadataSync(meili_client)

    def index_directory(
        self,
        input_path: str,
        depth: Optional[int] = None,
        force: bool = False
    ):
        """Index all repositories in directory."""

        # Query MeiliSearch for indexed projects
        if not force:
            indexed_projects = self.sync.get_indexed_projects(self.index_name)
            logger.info(f"Found {len(indexed_projects)} indexed project branches")
        else:
            indexed_projects = {}

        # Discover git projects (RDR-005)
        git_projects = self.git_discovery.find_git_projects(input_path, depth)

        # Process each project
        for project_root in git_projects:
            git_metadata = self.git_discovery.extract_metadata(project_root)
            identifier = f"{git_metadata.project_name}#{git_metadata.branch}"

            # Check if needs reindexing
            needs_indexing = self.sync.should_reindex_project(
                self.index_name, identifier, git_metadata.commit_hash
            )

            if not needs_indexing and not force:
                logger.info(f"Skipping {identifier} (up to date)")
                continue

            # Delete old documents if reindexing
            if identifier in indexed_projects:
                self._delete_branch_documents(identifier)

            # Index files
            self._index_project(project_root, git_metadata, identifier)

    def _index_project(
        self,
        project_root: str,
        git_metadata: GitMetadata,
        identifier: str
    ):
        """Index all files in a project."""
        # Get tracked files (RDR-005 pattern)
        code_files = self._get_git_files(project_root)

        documents = []
        for file_path in code_files:
            try:
                code = open(file_path).read()

                # Extract function/class definitions
                definitions = self.ast_extractor.extract_definitions(
                    file_path, code
                )

                # Build MeiliSearch documents
                for defn in definitions:
                    doc = self._build_document(
                        defn, git_metadata, identifier
                    )
                    documents.append(doc)

                # Upload in batches
                if len(documents) >= self.batch_size:
                    self._upload_batch(documents)
                    documents = []

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        # Upload remaining
        if documents:
            self._upload_batch(documents)

    def _build_document(
        self,
        defn: CodeDefinition,
        git_metadata: GitMetadata,
        identifier: str
    ) -> Dict[str, Any]:
        """Build MeiliSearch document from CodeDefinition."""
        return {
            "id": f"{identifier}:{defn.file_path}:{defn.qualified_name}:{defn.start_line}",
            "content": defn.content,
            "function_name": defn.name if defn.code_type in ["function", "method"] else None,
            "class_name": defn.name if defn.code_type == "class" else None,
            "qualified_name": defn.qualified_name,
            "filename": os.path.basename(defn.file_path),
            "git_project_identifier": identifier,
            "git_project_name": git_metadata.project_name,
            "git_branch": git_metadata.branch,
            "git_commit_hash": git_metadata.commit_hash,
            "git_remote_url": git_metadata.remote_url,
            "file_path": defn.file_path,
            "start_line": defn.start_line,
            "end_line": defn.end_line,
            "line_count": defn.end_line - defn.start_line + 1,
            "code_type": defn.code_type,
            "programming_language": self._detect_language(defn.file_path),
        }
```

### Implementation Example

**Complete Workflow** (Updated 2026-01-15):

```bash
# Option A: Dual indexing via RDR-009 (RECOMMENDED for most users)
# ---------------------------------------------------------------

# 1. Create corpus (both Qdrant collection + MeiliSearch index)
arc corpus create MyCode --type code --model stella

# 2. Sync directory to both systems
arc corpus sync ./src --corpus MyCode
# Result: Qdrant "MyCode" + MeiliSearch "MyCode" index

# 3. Search both ways
arc search semantic "authentication patterns" --collection MyCode
arc search text '"def authenticate"' --index MyCode


# Option B: Standalone full-text only (this RDR)
# -----------------------------------------------

# 1. Create MeiliSearch index with source code settings
arc indexes create MyCode-fulltext --type source-code

# 2. Index source code to MeiliSearch only (no embeddings)
arc index text code ./src --index MyCode-fulltext

# 3. Search for exact phrase
arc search text '"def authenticate"' --index MyCode-fulltext

# 4. Search with filters
arc search text 'calculate_total' \
  --index MyCode-fulltext \
  --filter 'programming_language = python AND git_branch = main'

# 5. List indexed projects
arc indexes list-projects --index MyCode-fulltext

# 6. Delete specific branch
arc indexes delete-project arcaneum#feature-x --index MyCode-fulltext
```

## Alternatives Considered

### Alternative 1: Whole-File Indexing (No Function/Class Granularity)

**Description**: Index entire files as single MeiliSearch documents

**Pros**:

- Simpler implementation
- Fewer documents (1 per file vs 5-50 per file)
- Easier metadata management

**Cons**:

- **No line-number precision** (can't return "lines 42-67")
- **Less useful results** (just "found in file.py")
- **Doesn't match RDR-010 pattern** (page-level for PDFs)
- **Poor UX** (users need line numbers for code search)

**Reason for rejection**: Line-level precision is critical for code search. Users need
exact locations for citations and navigation. Function/class-level provides perfect
balance.

### Alternative 2: Line-Based Indexing (One Document Per Line)

**Description**: Index each line of code as separate MeiliSearch document

**Pros**:

- Exact line-number precision (file.py:123)
- Simple logic (no AST parsing)

**Cons**:

- **Massive document count** (10K lines = 10K documents per file)
- **Loss of context** (single line has no semantic meaning)
- **Poor search quality** (matches spread across thousands of docs)
- **MeiliSearch performance** (unnecessary overhead)

**Reason for rejection**: Too granular, creates performance issues, loses semantic context of functions/classes.

### Alternative 3: Sequential Dual Indexing (Not Parallel)

**Description**: Separate commands for vector and full-text indexing

```bash
arc index-code ./src --collection MyCode  # Qdrant only
arc fulltext index-code ./src --index MyCode-fulltext  # MeiliSearch only
```

**Pros**:

- Simpler implementation (independent pipelines)
- Clear separation of concerns

**Cons**:

- **Reads files twice** (inefficient)
- **Two commands** (worse UX)
- **Inconsistent metadata** (slight timing differences)
- **Against RDR-009 pattern** (dual indexing recommended)

**Reason for rejection**: Source code indexing shares too much processing (git, AST)
to justify separate commands. Parallel dual indexing is more efficient.

### Alternative 4: Regex Function Extraction (No Tree-Sitter)

**Description**: Use regex patterns to extract function/class names

**Pros**:

- Faster than AST parsing
- Simpler implementation

**Cons**:

- **Fragile** (breaks on complex syntax)
- **Language-specific** (need regex per language)
- **Misses edge cases** (decorators, async, nested)
- **No line numbers** (hard to extract accurately)
- **Against RDR-005** (already have tree-sitter)

**Reason for rejection**: Tree-sitter already integrated from RDR-005, more accurate, handles 130+ languages.

## Trade-offs and Consequences

### Positive Consequences

1. **Function/Class-Level Precision**: Returns "calculate_total() lines 42-67" for exact navigation
2. **Git-Aware Multi-Branch**: Same composite identifier pattern as RDR-005 for consistency
3. **Metadata-Based Sync**: MeiliSearch as single source of truth, idempotent reindexing
4. **Parallel Dual Indexing**: Single pass over codebase, shared git/AST processing
5. **Filter-Based Deletion**: Branch-specific updates don't affect other branches
6. **130+ Language Support**: Tree-sitter covers all major programming languages
7. **Proven Patterns**: Reuses RDR-005 (git), RDR-008 (MeiliSearch), RDR-010 (page-level)
8. **Simple UX**: One command for dual indexing, clear CLI structure

### Negative Consequences

1. **More Documents**: ~5-50 documents per file vs 1 (function/class-level)
   - *Mitigation*: MeiliSearch handles large document counts efficiently
   - *Benefit*: Precise search results with line ranges
2. **AST Parsing Required**: Adds processing time vs whole-file indexing
   - *Mitigation*: Already integrated from RDR-005, fallback to module-level
   - *Benefit*: Accurate extraction, semantic boundaries
3. **Module-Level Fallback**: Top-level code may not have precise line ranges
   - *Mitigation*: Single "module" document covers all non-function code
   - *Benefit*: All code indexed even if not in functions

### Risks and Mitigations

**Risk**: Tree-sitter fails on malformed code

**Mitigation**: Automatic fallback to module-level indexing (entire file), error logging for debugging

**Risk**: Git operations timeout on large repositories

**Mitigation**: 5-second timeout per git command (from RDR-005), skip problematic repos with warning

**Risk**: MeiliSearch index size grows large

**Mitigation**: Monitor disk usage, document retention policies, implement index cleanup commands

**Risk**: Filter-based deletion doesn't work as expected

**Mitigation**: Validate filter syntax, dry-run mode, test with sample data before production

**Risk**: Memory exhaustion with large files

**Mitigation**: File size limits (10MB), batch size optimization (1000 docs), progress tracking

## Implementation Plan

### Prerequisites

- [x] RDR-005: Git-aware vector indexing (reuse git discovery) - **Implemented**
- [x] RDR-008: MeiliSearch server setup (deployment, client) - **Implemented**
- [x] RDR-009: Dual indexing strategy (DualIndexer, DualIndexDocument) - **Implemented**
- [x] RDR-010: PDF full-text indexing (patterns reference) - **Implemented**
- [x] Python >= 3.12 installed
- [x] tree-sitter-language-pack >= 0.10.0 installed
- [x] meilisearch-python >= 0.31.0 installed
- [x] GitPython >= 3.1.45 installed

### Step-by-Step Implementation (Updated 2026-01-15)

#### Step 1: Create AST Function/Class Extractor (NEW)

Create `src/arcaneum/indexing/fulltext/ast_extractor.py`:

- **NOTE**: This is a NEW component, distinct from `ast_chunker.py` (RDR-005)
- Implement `ASTFunctionExtractor` class using tree-sitter DIRECTLY
- Do NOT use LlamaIndex CodeSplitter (it's for chunking, not extraction)
- Use `get_parser()` from tree-sitter-language-pack
- Traverse AST via `node.start_point.row`, `node.end_point.row`
- Extract function/class definitions with `node.child_by_field_name("name")`
- Language detection from file extensions (reuse LANGUAGE_MAP from ast_chunker.py)
- Handle nested functions with qualified names
- Fallback to module-level if parsing fails

**Key difference from ast_chunker.py**:

- ast_chunker.py: Creates overlapping chunks for embeddings (Qdrant)
- ast_extractor.py: Extracts discrete definitions with line ranges (MeiliSearch)

#### Step 2: Extend Metadata-Based Sync for MeiliSearch

Extend existing `src/arcaneum/indexing/fulltext/sync.py` (from RDR-010):

- Add `GitCodeMetadataSync` class (mirrors RDR-005's `GitMetadataSync`)
- Query MeiliSearch for indexed (project#branch, commit) tuples
- Reuse patterns from RDR-010's PDF sync
- Cache results to avoid repeated queries
- Provide `should_reindex_project()` method

**NOTE**: fulltext/sync.py already exists from RDR-010; extend it for code.

#### Step 3: Create Source Code MeiliSearch Indexer

Create `src/arcaneum/indexing/fulltext/code_indexer.py`:

- Implement `SourceCodeFullTextIndexer` orchestrator
- Pattern after `PDFFullTextIndexer` from RDR-010
- Reuse RDR-005 git discovery (`GitProjectDiscovery`)
- Integrate AST extractor for function/class extraction
- Build MeiliSearch documents with full metadata
- Filter-based branch deletion on commit change
- Batch upload with progress tracking (1000 docs)
- Integrate with metadata-based sync

#### Step 4: Verify Compatibility with RDR-009 Dual Indexing

Update integration with RDR-009's `arc corpus sync`:

- **DO NOT create new dual-indexing command**
- Verify `DualIndexDocument` schema supports function-level fields (it does)
- Verify `DualIndexer` can handle source code documents
- Ensure `arc corpus sync` correctly routes code to both systems
- Test: `arc corpus create X --type code && arc corpus sync ./src --corpus X`

#### Step 5: Create CLI Commands

Extend `src/arcaneum/cli/index_text.py` (pattern from RDR-010):

- Add `index_text_code_command()` function
- Register as `arc index text code` subcommand
- Mirror `index_text_pdf_command()` parameters
- Options: --index, --recursive, --depth, --force, --batch-size

Add to `src/arcaneum/cli/indexes.py`:

- Add `list-projects` subcommand to `arc indexes`
- Add `delete-project` subcommand to `arc indexes`

Update `src/arcaneum/fulltext/indexes.py`:

- Add `SOURCE_CODE_FULLTEXT_SETTINGS` constant

**NOTE**: Use `arc indexes` not `arc fulltext` (renamed per arcaneum-h6bo)

#### Step 6: Testing

Create comprehensive tests:

- Unit tests for AST extractor (per language)
- Unit tests for metadata sync (MeiliSearch queries)
- Integration tests: end-to-end indexing workflow
- Multi-branch tests: verify branch isolation
- Change detection tests: commit hash comparison
- Dual indexing tests: verify both systems populated
- Performance tests: batch upload throughput

**Estimated effort**: 3-4 days

#### Step 7: Documentation

Update documentation:

- README with full-text code search examples
- CLI reference for new commands
- Cooperative workflow guide (semantic → exact)
- Troubleshooting guide (AST failures, index management)
- Update RDR index with RDR-011 reference

**Estimated effort**: 2 days

### Files to Create (Updated 2026-01-15)

**New Modules**:

- `src/arcaneum/indexing/fulltext/ast_extractor.py` - Function/class extraction (NEW)
  - **NOTE**: This is distinct from ast_chunker.py - different purpose
- `src/arcaneum/indexing/fulltext/code_indexer.py` - MeiliSearch code indexer

**Tests**:

- `tests/indexing/fulltext/test_ast_extractor.py` - Extractor tests
- `tests/indexing/fulltext/test_code_indexer.py` - Indexer tests
- `tests/integration/test_fulltext_code_indexing.py` - End-to-end tests

### Files to Modify (Updated 2026-01-15)

**Existing Modules**:

- `src/arcaneum/indexing/fulltext/sync.py` - Add git project sync for code
  - Already exists from RDR-010; extend for git metadata
- `src/arcaneum/cli/index_text.py` - Add `arc index text code` subcommand
  - Pattern after `index_text_pdf_command()`
- `src/arcaneum/cli/indexes.py` - Add list-projects, delete-project subcommands
- `src/arcaneum/fulltext/indexes.py` - Add SOURCE_CODE_FULLTEXT_SETTINGS
- `docs/rdr/README.md` - Update RDR index

**NOT Modified** (per design clarification):

- `src/arcaneum/indexing/source_code_pipeline.py` - NO changes needed
  - Dual indexing handled by RDR-009's `arc corpus sync`
- `src/arcaneum/cli/index.py` - NO new `--corpus` flag
  - Use `arc corpus sync` instead (RDR-009)

### Dependencies

Already satisfied by pyproject.toml:

- tree-sitter-language-pack >= 0.10.0 (from RDR-005)
- llama-index-core >= 0.14.6 (for AST chunking in RDR-005, NOT used here)
- GitPython >= 3.1.45 (from RDR-005)
- meilisearch >= 0.31.0 (from RDR-008)
- tenacity >= 9.1.2
- tqdm >= 4.67.1
- rich >= 14.2.0

**NOTE**: LlamaIndex is NOT used for function extraction in this RDR.
We use tree-sitter directly via `get_parser()`.

## Validation

### Testing Approach

1. **Unit Tests**: Test AST extraction per language, metadata sync logic
2. **Integration Tests**: Test complete indexing workflow, change detection
3. **Multi-Branch Tests**: Verify branch isolation, composite identifiers
4. **Dual Indexing Tests**: Verify both Qdrant and MeiliSearch populated
5. **Performance Tests**: Measure indexing throughput, query latency

### Test Scenarios

#### Scenario 1: Initial Code Repository Indexing

- **Setup**: Fresh git repository with Python files containing functions/classes
- **Action**: `arcaneum index-code --corpus MyCode ./src`
- **Expected**:
  - Functions/classes extracted with line ranges
  - Documents created in MeiliSearch (function/class-level)
  - Git metadata stored (project#branch, commit hash)
  - Dual indexing: Both Qdrant and MeiliSearch populated

#### Scenario 2: Incremental Sync After Commit

- **Setup**: Repository from Scenario 1
- **Action**: Edit files, commit, re-run `arcaneum index-code --corpus MyCode ./src`
- **Expected**:
  - Metadata query detects commit change
  - Filter-based deletion removes old branch documents
  - Branch re-indexed with new commit hash
  - Other branches unaffected

#### Scenario 3: Multi-Branch Coexistence

- **Setup**: Same repo on two branches (main, feature-x)
- **Action**: Index main, checkout feature-x, index again
- **Expected**:
  - Both branches exist in MeiliSearch: "arcaneum#main", "arcaneum#feature-x"
  - Can query either branch independently
  - Updates to one branch don't affect other

#### Scenario 4: Function/Class Search with Line Ranges

- **Setup**: Indexed Python code with function `calculate_total`
- **Action**: `arcaneum fulltext search '"def calculate_total"' --index MyCode-fulltext`
- **Expected**:
  - Returns document with function_name="calculate_total"
  - Includes start_line and end_line
  - Content shows actual function code
  - Result: "Found in utils.py, calculate_total() lines 42-67"

#### Scenario 5: AST Fallback for Module-Level Code

- **Setup**: Python file with top-level code (no functions)
- **Action**: Index file
- **Expected**:
  - Single "module" document created
  - Covers entire file (lines 1-N)
  - code_type="module"
  - All code searchable

#### Scenario 6: Crash Recovery via Re-indexing

- **Setup**: Large repository indexing (simulate interruption after 50%)
- **Action**: Kill process, restart indexing
- **Expected**:
  - Metadata-based sync detects partial indexing
  - Idempotent re-indexing (no duplicates)
  - Completes indexing from where it left off

### Performance Validation

**Metrics to Track**:

- Files indexed per second
- Functions/classes extracted per second
- Documents uploaded per second
- AST parsing success rate per language
- Filter-based deletion speed
- Metadata query performance
- Memory usage during large file processing

**Targets**:

- 50-100 files/sec indexing throughput
- > 95% AST parsing success (rest fallback to module)
- < 500ms for branch-specific deletion
- < 5s for metadata query on 1000 projects
- < 2.5GiB memory usage (within MEILI_MAX_INDEXING_MEMORY)

### Security Validation

- Remote URLs stripped of credentials (from RDR-005)
- File paths don't expose sensitive directories
- No secrets stored in MeiliSearch metadata
- MeiliSearch API key required for indexing
- Volume permissions: User-only write (0755)

## References

### Related RDRs (Updated 2026-01-15)

- [RDR-005: Git-Aware Source Code Indexing](RDR-005-source-code-indexing.md) - **Implemented**
  - Reuse: git discovery (`GitProjectDiscovery`), LANGUAGE_MAP
  - Note: ast_chunker.py is for chunking, NOT function extraction
- [RDR-008: Full-Text Search Server Setup](RDR-008-fulltext-search-server-setup.md) - **Implemented**
  - MeiliSearch v1.32.x deployment, client
- [RDR-009: Dual Indexing Strategy](RDR-009-dual-indexing-strategy.md) - **Implemented**
  - Use `arc corpus sync` for dual indexing (NOT new command)
  - DualIndexer, DualIndexDocument schema
- [RDR-010: PDF Full-Text Indexing](RDR-010-pdf-fulltext-indexing.md) - **Implemented**
  - Pattern reference for `arc index text code`
  - CLI renamed: `arc fulltext` → `arc indexes` (arcaneum-h6bo)

### Beads Issues

**Original Research**:

- [arcaneum-70](../../.beads/arcaneum.db) - Original RDR request
- [arcaneum-86](../../.beads/arcaneum.db) - Indexing granularity research
- [arcaneum-87](../../.beads/arcaneum.db) - Metadata schema design
- [arcaneum-88](../../.beads/arcaneum.db) - Git change detection strategy
- [arcaneum-89](../../.beads/arcaneum.db) - Function/class extraction
- [arcaneum-90](../../.beads/arcaneum.db) - Dual indexing workflow
- [arcaneum-91](../../.beads/arcaneum.db) - Batch upload optimization
- [arcaneum-92](../../.beads/arcaneum.db) - CLI command structure

**2026-01-15 Review Issues**:

- arcaneum-vau8 - Update technical environment section
- arcaneum-03wm - CRITICAL: Rewrite AST extraction to use tree-sitter directly
- arcaneum-40m1 - Update CLI command naming
- arcaneum-y53c - Update SOURCE_CODE_FULLTEXT_SETTINGS
- arcaneum-efmz - Clarify dual indexing with RDR-009
- arcaneum-xoiq - Clarify ast_extractor vs ast_chunker
- arcaneum-e2gn - Update Related RDRs status

### Official Documentation

- **MeiliSearch Documentation**: <https://www.meilisearch.com/docs>
- **meilisearch-python Client**: <https://github.com/meilisearch/meilisearch-python>
- **tree-sitter-language-pack**: <https://github.com/Goldziher/tree-sitter-language-pack>
  - Note: Maintained replacement for grantjenks/py-tree-sitter-language-pack
- **py-tree-sitter**: <https://tree-sitter.github.io/py-tree-sitter/>
- **GitPython Documentation**: <https://gitpython.readthedocs.io/>

## Notes

### Key Design Decisions (Updated 2026-01-15)

1. **Function/Class-Level Granularity**: Provides precise line ranges while maintaining semantic boundaries
2. **Reuse RDR-005 Git Components**: Git discovery (`GitProjectDiscovery`), LANGUAGE_MAP
3. **NEW AST Extractor (NOT reusing ast_chunker.py)**:
   - ast_chunker.py = overlapping chunks for embeddings (Qdrant)
   - ast_extractor.py = discrete definitions with line ranges (MeiliSearch)
4. **Use tree-sitter DIRECTLY**: NOT LlamaIndex CodeSplitter (which is for chunking)
5. **Metadata-Based Sync**: MeiliSearch as single source of truth (mirrors RDR-005 pattern)
6. **Use RDR-009 for Dual Indexing**: `arc corpus sync` instead of new command
7. **Filter-Based Branch Deletion**: Branch-specific updates, other branches unaffected
8. **CLI Naming Convention**: `arc index text code` (parallel to `arc index text pdf`)

### Future Enhancements

**Hybrid Search**:

- Combine semantic (Qdrant) + exact (MeiliSearch) results
- Reciprocal Rank Fusion (RRF) for result merging
- `arc find MyCode "query" --hybrid`

**Advanced AST Analysis**:

- Extract import statements for dependency tracking
- Extract docstrings for enhanced search
- Track function calls for usage analysis

**Real-Time Indexing**:

- File watchers (inotify, FSEvents) for automatic reindexing
- Incremental updates on file changes
- `arcaneum index-code --watch`

**Multi-Language Support**:

- Extend typo tolerance per language
- Language-specific stop words
- Custom tokenization rules

### Known Limitations

- **AST parsing dependency**: Requires tree-sitter for accurate extraction
  - *Mitigation*: Fallback to module-level if parsing fails
- **Module-level code**: Top-level code has less precise boundaries
  - *Mitigation*: Single "module" document with full file range
- **Large functions**: Functions > 200 lines indexed as-is
  - *Mitigation*: Rare in practice, could split in future
- **Nested functions**: Indexed separately, may lose some context
  - *Mitigation*: Use qualified names to maintain hierarchy

### Success Criteria (Updated 2026-01-15)

- ✅ Function/class-level indexing with line ranges
- ✅ Git-aware multi-branch support (composite identifiers)
- ✅ Metadata-based sync (MeiliSearch as source of truth)
- ✅ Dual indexing via `arc corpus sync` (RDR-009)
- ✅ Filter-based branch-specific deletion
- ✅ 130+ language support via tree-sitter
- ✅ CLI commands: `arc index text code`, `arc indexes list-projects`
- ✅ Batch upload optimization (1000 docs)
- ✅ Markdownlint compliant

This RDR provides the complete specification for indexing source code to MeiliSearch
for full-text exact phrase and keyword search at function/class granularity,
complementary to Qdrant's semantic search (RDR-005), reusing git discovery
infrastructure, and enabling parallel dual indexing via RDR-009 for efficiency.

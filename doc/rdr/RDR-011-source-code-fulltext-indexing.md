# Recommendation 011: Git-Aware Source Code Full-Text Indexing to MeiliSearch

## Metadata

- **Date**: 2025-10-27
- **Status**: Recommendation
- **Type**: Feature
- **Priority**: High
- **Related Issues**: arcaneum-70, arcaneum-86, arcaneum-87, arcaneum-88, arcaneum-89, arcaneum-90, arcaneum-91, arcaneum-92
- **Related Tests**: Source code full-text indexing tests, dual indexing integration tests

## Problem Statement

Create a production-ready source code indexing system for MeiliSearch that enables exact phrase search and keyword matching across source code with git awareness and function/class-level precision. The system must:

1. **Index at function/class granularity** - Enable precise location results like "found in calculate_total() lines 42-67"
2. **Git awareness** - Multi-branch support with same composite identifier pattern as RDR-005
3. **Dual indexing** - Integrate with RDR-005 vector indexing for efficient parallel indexing
4. **Change detection** - Metadata-based sync using MeiliSearch as single source of truth
5. **Exact search** - Support phrase matching, regex, and keyword searches complementary to RDR-005 semantic search

This addresses the need for exact string matching in code ("find def authenticate") complementary to semantic search ("find authentication patterns") from RDR-005.

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

**Parallel to RDR-010**: Just as RDR-010 indexes PDFs to MeiliSearch (complementary to RDR-004 vector), this RDR indexes source code to MeiliSearch (complementary to RDR-005 vector).

### Technical Environment

- **Python**: >= 3.12
- **MeiliSearch**: v1.24.0 (from RDR-008)
- **Git**: >= 2.30 (for metadata extraction)
- **AST Parsing**:
  - tree-sitter-language-pack >= 0.5.0 (165+ languages, from RDR-005)
  - LlamaIndex >= 0.9.0 (CodeSplitter integration, from RDR-005)
- **MeiliSearch Client**:
  - meilisearch-python >= 0.31.0 (from RDR-008)
- **Supporting Libraries**:
  - GitPython >= 3.1.40 (from RDR-005)
  - tenacity (retry logic)
  - tqdm/rich (progress tracking)

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

**MeiliSearch Index Settings**:

```python
SOURCE_CODE_FULLTEXT_SETTINGS = {
    "searchableAttributes": [
        "content",           # Primary search field
        "function_name",     # For identifier search
        "class_name",        # For class search
        "qualified_name",    # For fully-qualified searches
        "filename",          # For file-specific searches
    ],
    "filterableAttributes": [
        "programming_language",
        "git_project_name",
        "git_branch",
        "git_project_identifier",  # Composite for branch-specific queries
        "file_path",
        "code_type",
    ],
    "sortableAttributes": [
        "start_line",  # For sorting by location
    ],
    "typoTolerance": {
        "enabled": True,
        "minWordSizeForTypos": {
            "oneTypo": 7,   # Higher threshold for code
            "twoTypos": 12
        }
    },
    "stopWords": [],  # Preserve all code keywords
    "pagination": {
        "maxTotalHits": 10000
    }
}
```

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

#### 4. Tree-Sitter Function/Class Extraction (arcaneum-89)

**Approach**: Reuse RDR-005's tree-sitter integration with different queries

**Function/Class Extraction**:

```python
from tree_sitter_language_pack import get_parser

class ASTFunctionExtractor:
    """Extract function/class definitions with line ranges."""

    LANGUAGE_MAP = {
        ".py": "python",
        ".java": "java",
        ".js": "javascript",
        ".ts": "typescript",
        # ... 165+ languages supported
    }

    def extract_definitions(
        self,
        file_path: str,
        code: str
    ) -> List[CodeDefinition]:
        """
        Extract function/class definitions with line numbers.

        Returns list of CodeDefinition with:
        - name: identifier (function/class name)
        - qualified_name: full path (e.g., "MyClass.method")
        - code_type: "function", "class", "method", "module"
        - start_line: beginning line number
        - end_line: ending line number
        - content: actual code text
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        language = self.LANGUAGE_MAP.get(file_ext)

        if not language:
            # Fallback: entire file as "module"
            return [self._create_module_definition(code)]

        try:
            parser = get_parser(language)
            tree = parser.parse(bytes(code, "utf8"))

            definitions = []

            # Language-specific queries
            if language == "python":
                definitions = self._extract_python_definitions(tree, code)
            elif language == "javascript" or language == "typescript":
                definitions = self._extract_js_definitions(tree, code)
            # ... other languages

            # Add module-level code if any
            module_code = self._extract_module_code(code, definitions)
            if module_code:
                definitions.append(module_code)

            return definitions

        except Exception as e:
            logger.warning(f"AST extraction failed for {file_path}: {e}")
            return [self._create_module_definition(code)]

    def _extract_python_definitions(
        self,
        tree,
        code: str
    ) -> List[CodeDefinition]:
        """Extract Python function/class definitions."""
        query = """
        (function_definition
          name: (identifier) @function.name
        ) @function.definition

        (class_definition
          name: (identifier) @class.name
        ) @class.definition
        """

        # Execute tree-sitter query
        # Extract line numbers from nodes
        # Return CodeDefinition objects
        pass
```

**Fallback Strategy**:

- If AST parsing fails: Index entire file as single "module" document
- Ensures all code is indexed even if tree-sitter fails
- Graceful degradation

#### 5. Dual Indexing Workflow (arcaneum-90)

**Pattern**: Parallel dual indexing with shared processing (RDR-009 pattern)

**Workflow**:

```bash
# Single command indexes to both systems
arc index-code ./src --corpus MyCode
```

**Implementation**:

```text
┌─────────────────────────────────────────────────────────────┐
│           Dual Indexing Pipeline (Single Pass)              │
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
    embedding chunks                     function/class metadata
          │                                       │
    Upload to Qdrant                      Upload to MeiliSearch
    (collection: MyCode)                  (index: MyCode-fulltext)
```

**Benefits**:

- Single pass over codebase (efficient)
- Consistent git metadata across both systems
- Shared AST parsing (performance optimization)
- Simple UX: one command for dual indexing

**Comparison to RDR-010 (PDF)**:

- RDR-010: Separate commands (different processing: OCR vs text-only)
- RDR-011: Unified command (shared processing: git + AST)
- Tighter integration justified for source code

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

#### 7. CLI Command Structure (arcaneum-92)

**Dual Indexing** (extend RDR-005's existing command):

```bash
arc index-code ./src --corpus MyCode
# Creates: Qdrant "MyCode" + MeiliSearch "MyCode-fulltext"
```

**Full-Text Only** (new command):

```bash
arc fulltext index-code ./src --index MyCode-fulltext
```

**Search** (existing from RDR-008):

```bash
arc fulltext search '"def authenticate"' \
  --index MyCode-fulltext \
  --filter 'language = python AND git_branch = main'
```

**Project Management** (new commands):

```bash
# List indexed projects with commit hashes
arc fulltext list-projects --index MyCode-fulltext

# Delete specific project/branch
arc fulltext delete-project arcaneum#main --index MyCode-fulltext
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

**Complete Workflow**:

```bash
# 1. Create MeiliSearch index with source code settings
export MEILI_MASTER_KEY=your_master_key
arc fulltext create-index MyCode-fulltext --type source-code

# 2. Dual index source code (RDR-005 + this RDR)
arc index-code ./src --corpus MyCode
# Creates: Qdrant "MyCode" + MeiliSearch "MyCode-fulltext"

# 3. Search for exact phrase
arc fulltext search '"def authenticate"' --index MyCode-fulltext

# 4. Search with filters
arc fulltext search 'calculate_total' \
  --index MyCode-fulltext \
  --filter 'language = python AND git_branch = main'

# 5. List indexed projects
arc fulltext list-projects --index MyCode-fulltext

# 6. Delete specific branch
arc fulltext delete-project arcaneum#feature-x --index MyCode-fulltext
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

**Reason for rejection**: Line-level precision is critical for code search. Users need exact locations for citations and navigation. Function/class-level provides perfect balance.

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

**Reason for rejection**: Source code indexing shares too much processing (git, AST) to justify separate commands. Parallel dual indexing is more efficient.

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

**Reason for rejection**: Tree-sitter already integrated from RDR-005, more accurate, handles 165+ languages.

## Trade-offs and Consequences

### Positive Consequences

1. **Function/Class-Level Precision**: Returns "calculate_total() lines 42-67" for exact navigation
2. **Git-Aware Multi-Branch**: Same composite identifier pattern as RDR-005 for consistency
3. **Metadata-Based Sync**: MeiliSearch as single source of truth, idempotent reindexing
4. **Parallel Dual Indexing**: Single pass over codebase, shared git/AST processing
5. **Filter-Based Deletion**: Branch-specific updates don't affect other branches
6. **165+ Language Support**: Tree-sitter covers all major programming languages
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

- [x] RDR-005: Git-aware vector indexing (reuse git discovery, AST parsing)
- [x] RDR-008: MeiliSearch server setup (deployment, client)
- [x] RDR-010: PDF full-text indexing (patterns reference)
- [ ] Python >= 3.12 installed
- [ ] tree-sitter-language-pack installed
- [ ] meilisearch-python >= 0.31.0 installed
- [ ] GitPython >= 3.1.40 installed

### Step-by-Step Implementation

#### Step 1: Create AST Function/Class Extractor

Create `src/arcaneum/indexing/fulltext/ast_extractor.py`:

- Implement `ASTFunctionExtractor` class
- Language detection from file extensions
- Tree-sitter query execution for function/class definitions
- Extract line ranges (start_line, end_line)
- Handle nested functions with qualified names
- Fallback to module-level if parsing fails
- Extract module-level code (not in functions)

**Estimated effort**: 2-3 days

#### Step 2: Create Metadata-Based Sync Module

Create `src/arcaneum/indexing/fulltext/sync.py`:

- Implement `GitMetadataSync` class (mirrors RDR-005 pattern)
- Query MeiliSearch for indexed (project#branch, commit) tuples
- Cache results to avoid repeated queries
- Provide `should_reindex_project()` method
- Single source of truth (MeiliSearch metadata)

**Estimated effort**: 1 day

#### Step 3: Create MeiliSearch Indexer

Create `src/arcaneum/indexing/fulltext/code_indexer.py`:

- Implement `SourceCodeFullTextIndexer` orchestrator
- Reuse RDR-005 git discovery (`GitProjectDiscovery`)
- Integrate AST extractor for function/class extraction
- Build MeiliSearch documents with full metadata
- Filter-based branch deletion on commit change
- Batch upload with progress tracking (1000 docs)
- Integrate with metadata-based sync

**Estimated effort**: 3-4 days

#### Step 4: Extend RDR-005 for Dual Indexing

Update `src/arcaneum/indexing/source_code_pipeline.py`:

- Add optional full-text indexing flag to existing `index-code` command
- When `--corpus` flag used: index to BOTH Qdrant and MeiliSearch
- Share git discovery, file reading, AST parsing
- Parallel output: embedding chunks → Qdrant, function metadata → MeiliSearch
- Progress tracking shows both operations

**Estimated effort**: 2-3 days

#### Step 5: Create CLI Commands

Create `src/arcaneum/cli/fulltext_code.py`:

- Implement `fulltext index-code` subcommand (standalone full-text only)
- Implement `fulltext list-projects` subcommand
- Implement `fulltext delete-project` subcommand
- Register with main CLI
- Add SOURCE_CODE_FULLTEXT_SETTINGS to `src/arcaneum/fulltext/indexes.py`

Update `src/arcaneum/cli/index.py`:

- Extend existing `index-code` command with dual indexing support
- When `--corpus` specified: trigger both RDR-005 and RDR-011 indexing

**Estimated effort**: 2 days

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

### Files to Create

**New Modules**:

- `src/arcaneum/indexing/fulltext/__init__.py` - Module init
- `src/arcaneum/indexing/fulltext/ast_extractor.py` - Function/class extraction
- `src/arcaneum/indexing/fulltext/sync.py` - Metadata-based sync
- `src/arcaneum/indexing/fulltext/code_indexer.py` - MeiliSearch indexer
- `src/arcaneum/cli/fulltext_code.py` - CLI commands

**Tests**:

- `tests/indexing/fulltext/test_ast_extractor.py` - Extractor tests
- `tests/indexing/fulltext/test_sync.py` - Sync tests
- `tests/indexing/fulltext/test_code_indexer.py` - Indexer tests
- `tests/integration/test_fulltext_code_indexing.py` - End-to-end tests
- `tests/integration/test_dual_indexing.py` - Dual indexing tests

### Files to Modify

**Existing Modules**:

- `src/arcaneum/indexing/source_code_pipeline.py` - Add dual indexing support
- `src/arcaneum/cli/index.py` - Extend `index-code` command
- `src/arcaneum/fulltext/indexes.py` - Add SOURCE_CODE_FULLTEXT_SETTINGS
- `README.md` - Add full-text code search examples
- `doc/rdr/README.md` - Update RDR index

### Dependencies

Already satisfied by RDR-005 and RDR-008:

- tree-sitter-language-pack >= 0.5.0 (from RDR-005)
- llama-index >= 0.9.0 (from RDR-005)
- GitPython >= 3.1.40 (from RDR-005)
- meilisearch-python >= 0.31.0 (from RDR-008)
- tenacity >= 8.2.0
- tqdm >= 4.65.0
- rich >= 13.0.0

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

### Related RDRs

- [RDR-005: Git-Aware Source Code Indexing](RDR-005-source-code-indexing.md) - **PRIMARY DEPENDENCY** (git discovery, AST parsing)
- [RDR-008: Full-Text Search Server Setup](RDR-008-fulltext-search-server-setup.md) - MeiliSearch deployment, client
- [RDR-009: Dual Indexing Strategy](RDR-009-dual-indexing-strategy.md) - Shared metadata patterns
- [RDR-010: PDF Full-Text Indexing](RDR-010-pdf-fulltext-indexing.md) - Parallel pattern reference

### Beads Issues

- [arcaneum-70](../../.beads/arcaneum.db) - Original RDR request
- [arcaneum-86](../../.beads/arcaneum.db) - Indexing granularity research
- [arcaneum-87](../../.beads/arcaneum.db) - Metadata schema design
- [arcaneum-88](../../.beads/arcaneum.db) - Git change detection strategy
- [arcaneum-89](../../.beads/arcaneum.db) - Function/class extraction
- [arcaneum-90](../../.beads/arcaneum.db) - Dual indexing workflow
- [arcaneum-91](../../.beads/arcaneum.db) - Batch upload optimization
- [arcaneum-92](../../.beads/arcaneum.db) - CLI command structure

### Official Documentation

- **MeiliSearch Documentation**: <https://www.meilisearch.com/docs>
- **meilisearch-python Client**: <https://github.com/meilisearch/meilisearch-python>
- **tree-sitter-language-pack**: <https://github.com/grantjenks/py-tree-sitter-language-pack>
- **GitPython Documentation**: <https://gitpython.readthedocs.io/>

## Notes

### Key Design Decisions

1. **Function/Class-Level Granularity**: Provides precise line ranges while maintaining semantic boundaries
2. **Reuse RDR-005 Components 100%**: Git discovery, AST parsing, change detection patterns
3. **Metadata-Based Sync**: MeiliSearch as single source of truth (consistent with RDR-005)
4. **Parallel Dual Indexing**: Single pass over codebase for efficiency
5. **Filter-Based Branch Deletion**: Branch-specific updates, other branches unaffected

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

### Success Criteria

- ✅ Function/class-level indexing with line ranges
- ✅ Git-aware multi-branch support (composite identifiers)
- ✅ Metadata-based sync (MeiliSearch as source of truth)
- ✅ Parallel dual indexing with RDR-005
- ✅ Filter-based branch-specific deletion
- ✅ 165+ language support via tree-sitter
- ✅ CLI command extensions for consistency
- ✅ Batch upload optimization (1000 docs)
- ✅ Implementation < 20 days
- ✅ Markdownlint compliant

This RDR provides the complete specification for indexing source code to MeiliSearch for full-text exact phrase and keyword search at function/class granularity, complementary to Qdrant's semantic search (RDR-005), reusing git discovery and AST parsing infrastructure, and enabling parallel dual indexing for efficiency.

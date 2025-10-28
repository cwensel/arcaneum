# Recommendation 005: Git-Aware Source Code Indexing with AST Chunking

## Metadata
- **Date**: 2025-10-19
- **Status**: Implemented
- **Type**: Feature
- **Priority**: High
- **Implementation Date**: 2025-10-27
- **Related Issues**: arcaneum-5, arcaneum-24, arcaneum-25, arcaneum-26, arcaneum-27, arcaneum-28, arcaneum-29, arcaneum-30, arcaneum-41, arcaneum-42, arcaneum-44
- **Related Tests**: AST chunking tests, git metadata extraction tests, change detection integration tests

## Problem Statement

Create a production-ready git-aware source code indexing system for Qdrant that handles 15+ programming languages with AST-aware chunking, respects .gitignore patterns, tracks git project changes with multi-branch support, and optimizes for code-specific embeddings. The system must:

1. **Discover git projects** with configurable depth control and extract metadata
2. **Index source code** with AST-aware chunking preserving function/class boundaries
3. **Track changes** via git commit hashes with branch-aware intelligent re-indexing
4. **Support multiple branches** of the same repository simultaneously for comparison
5. **Respect .gitignore** using native git ls-files integration
6. **Optimize for code embeddings** with appropriate chunking strategies
7. **Integrate seamlessly** with RDR-002 (Qdrant server) and RDR-003 (collection management)
8. **Read-only operations** - never mutate git state (no pull, fetch, checkout)

This addresses the need to index large source code repositories (monorepos, libraries, frameworks) for semantic code search across the Arcaneum ecosystem.

## Context

### Background

Arcaneum requires a robust source code indexing pipeline adapted from the proven `chroma-embedded/upload.sh` implementation. The system must handle diverse git-based coding scenarios:

- **Git-tracked projects**: Repositories with .git directories (100% of use cases)
- **Monorepos**: Large repositories with multiple sub-projects
- **Multi-branch indexing**: Same repository on different branches coexist in collection
- **Branch comparison**: Query and compare implementations across branches
- **Directory of repositories**: User specifies directory containing multiple git repos
- **15+ languages**: Python, Java, JavaScript, TypeScript, C#, Go, Rust, C/C++, PHP, Ruby, Kotlin, Scala, Swift, and more

**Design Questions** (from arcaneum-5):
- Git project discovery strategy (--depth control)?
- How to integrate ASTChunk for 15+ languages?
- Commit hash change detection - bulk delete or incremental?
- How to support multiple branches of same repo?
- How to identify (project, branch) combinations uniquely?
- Fallback when ASTChunk fails?
- Metadata schema for git info with branch awareness?

**Reference Implementation**: The `chroma-embedded/upload.sh` script (lines 373-433 for git discovery, lines 846-976 for change detection, lines 1743-1788 for AST chunking) provides battle-tested patterns for ChromaDB that we adapt for Qdrant with significant performance optimizations.

### Technical Environment

- **Python**: >= 3.12
- **Qdrant**: v1.15.4+ (from RDR-002)
- **Git**: >= 2.30 (for metadata extraction)
- **AST Parsing**:
  - tree-sitter-language-pack >= 0.5.0 (165+ languages)
  - LlamaIndex >= 0.9.0 (CodeSplitter integration)
- **Embedding**:
  - FastEmbed >= 0.3.0 (from RDR-003)
  - jina-code-embeddings-1.5b OR jina-embeddings-v2-base-code
  - qdrant-client[fastembed] >= 1.15.0
- **Supporting Libraries**:
  - GitPython >= 3.1.40 (git operations)
  - tenacity (retry logic)
  - tqdm/rich (progress tracking)

**Target Embedding Models**:
- jina-code-embeddings-1.5b: 1536D, 32K token context (recommended)
- jina-embeddings-v2-base-code: 768D, 8K token context (fallback)
- stella_en_1.5B_v5: 1024D, 512-1024 token chunks (general purpose)

## Research Findings

### Investigation Process

**Research tracks completed** to inform this RDR (tracked via Beads issues):

1. **Git Handling Patterns** (arcaneum-24): Deep analysis of chroma-embedded git discovery, metadata extraction, change detection, and .gitignore integration
2. **AST Chunking Libraries** (arcaneum-25): Comprehensive research on ASTChunk limitations and tree-sitter-language-pack alternatives
3. **Qdrant vs ChromaDB** (arcaneum-26): Comparative analysis identifying 40-100x performance gains with Qdrant's filter-based deletion
4. **Code Embeddings** (arcaneum-27): Investigation of jina-code models revealing 32K context window advantage
5. **Open Source Tools** (arcaneum-28): Analysis of 20+ production code indexing systems for best practices
6. **Git Metadata Best Practices** (arcaneum-29): Study of robust metadata extraction with edge case handling
7. **Change Detection Strategies** (arcaneum-30): Trade-off analysis of bulk delete vs incremental update approaches
8. **Multi-Branch Support Design** (arcaneum-32 to arcaneum-40): Composite identifier pattern enabling multiple branches per repository to coexist
9. **Metadata-Based Sync** (arcaneum-41): Architectural alignment with RDR-04 using Qdrant as single source of truth
10. **SQLite Checkpoint Removal** (arcaneum-42, arcaneum-44): Simplified design removing SQLite checkpoint database (idempotent re-indexing sufficient for source code)

### Key Discoveries

#### 1. Git Project Discovery and Metadata

**Discovery Pattern** (from arcaneum-24):
```bash
# Find .git directories with optional depth control
find "$input_path" -maxdepth $((depth + 1)) -name ".git" -type d | while read git_dir; do
    project_root=$(dirname "$git_dir")
    echo "$project_root"
done | sort -u
```

**Critical Insight**: Depth parameter adds 1 to maxdepth because we're searching for `.git` directories, not project roots.

**Metadata Extraction** (from arcaneum-29):
- **Commit hash**: Store full 40-char SHA (enables cryptographic verification)
- **Remote URL**: Priority order - origin > upstream > first remote, with credential sanitization
- **Branch detection**: Robust fallback chain handles detached HEAD states
- **Project name**: Derive from remote URL first, fallback to directory basename
- **Composite identifier**: `git_project_identifier = f"{project_name}#{branch}"` enables multi-branch support

**Edge Cases Handled**:
- Detached HEAD states (use git describe --tags or git name-rev)
- Shallow clones (detect via .git/shallow file)
- Missing remotes (fallback to local path)
- Submodules (optional tracking via .gitmodules)
- Corrupt repositories (git fsck validation)

#### 2. AST-Aware Chunking Strategy

**CRITICAL FINDING** (from arcaneum-25): **ASTChunk library is insufficient**
- Supports ONLY 4 languages (Python, Java, C#, TypeScript)
- Missing 11 required languages (JavaScript, Go, Rust, C/C++, PHP, Ruby, Kotlin, Scala, Swift, etc.)
- No built-in fallback mechanisms

**Recommended Solution**: **tree-sitter-language-pack** + LlamaIndex CodeSplitter
- Supports 165+ languages (covers all requirements)
- Mature implementation with error handling
- Built-in fallback to line-based chunking when AST parsing fails

**cAST Algorithm** (from arcaneum-28):
```
1. Parse code into AST using tree-sitter
2. Recursively split nodes that exceed max_chunk_size
3. Greedily merge adjacent small nodes to optimize chunk count
4. Preserve syntactic boundaries (functions, classes, modules)
5. Add optional overlap for context preservation
```

**Proven Results**: 4.3 point gain in Recall@5 on RepoEval benchmark

**Language Support Matrix**:
| Language | tree-sitter Support | Fallback |
|----------|-------------------|----------|
| Python | ✅ Full | Line-based |
| Java | ✅ Full | Line-based |
| JavaScript | ✅ Full | Line-based |
| TypeScript | ✅ Full | Line-based |
| C# | ✅ Full | Line-based |
| Go | ✅ Full | Line-based |
| Rust | ✅ Full | Line-based |
| C/C++ | ✅ Full | Line-based |
| PHP | ✅ Full | Line-based |
| Ruby | ✅ Full | Line-based |
| Kotlin | ✅ Full | Line-based |
| Scala | ✅ Full | Line-based |
| Swift | ✅ Full | Line-based |
| + 152 more | ✅ Full | Line-based |

#### 3. Code Embedding Models

**THREE Jina Options** (from arcaneum-27):

**Option 1: jina-code-embeddings-1.5b** (Recommended)
- **Performance**: 79.04% average, 92.37% on StackOverflow, 86.45% on CodeSearchNet
- **Context**: 32K tokens (4x larger than v2/v3, can embed entire files)
- **Dimensions**: 1536D with Matryoshka support
- **Foundation**: Built on Qwen2.5-Coder
- **Chunking**: 2K-4K token chunks for large files, entire files for most code
- **Status**: May require manual FastEmbed integration

**Option 2: jina-embeddings-v2-base-code** (Fallback)
- **Performance**: 0.7753 accuracy (proven for code)
- **Context**: 8K tokens
- **Dimensions**: 768D
- **License**: Apache 2.0 (open source)
- **Chunking**: 400-512 token chunks
- **FastEmbed**: Fully supported

**Option 3: jina-embeddings-v3** (NOT Recommended for Code)
- **Performance**: 0.7564 accuracy (worse than v2 for code)
- **License**: CC BY-NC 4.0 (commercial restrictions)
- **Note**: Generic model, inferior to code-specific alternatives

**Character-to-Token Ratio**: Conservative estimate of 3.5 chars/token for code
- 400 tokens ≈ 1,200-1,400 characters
- 2K tokens ≈ 7,000-8,000 characters

#### 4. Qdrant vs ChromaDB Performance

**Key Advantages** (from arcaneum-26):

| Aspect | Qdrant | ChromaDB | Performance Gain |
|--------|--------|----------|-----------------|
| **Deletion** | Filter-based | ID retrieval + batch delete | 40-100x faster |
| **Batch Size** | 100-200 chunks | 25-50 chunks (HTTP limit) | 2-4x throughput |
| **Protocol** | REST + gRPC | HTTP only | 2-3x with gRPC |
| **Embedding** | Client-side (FastEmbed) | Server-side | Better scalability |

**Filter-Based Deletion Example**:
```python
# Delete all chunks for a specific (project, branch) combination (50-500ms)
client.delete(
    collection_name="code",
    points_selector=Filter(
        must=[FieldCondition(
            key="git_project_identifier",
            match=MatchValue(value="my-project#main")
        )]
    )
)
```

vs ChromaDB's approach (20-50s for same operation):
```python
# Retrieve all IDs first (slow)
all_ids = []
for batch in paginated_query():
    all_ids.extend(batch['ids'])

# Delete in batches
for batch in chunked(all_ids, 100):
    collection.delete(ids=batch)
```

#### 5. Change Detection Strategy

**Metadata-Based Sync with Git Optimization** (follows RDR-04 pattern):

**Primary: Qdrant Metadata Queries** (Source of truth)
- Query Qdrant for indexed `(git_project_identifier, git_commit_hash)` pairs on startup
- Compare with current git state to determine what needs indexing
- **Benefit**: Single source of truth, can't get out of sync with Qdrant
- **Pattern**: Same as RDR-04 PDF indexing (architectural consistency)
- Cost: One scroll query per collection on startup (O(n) projects, but cached)

**Level 1: Git Commit Hash Comparison** (Branch-level, O(1))
- Compare Qdrant's stored commit hash with current git HEAD
- If different or missing: Trigger bulk delete of branch chunks (filter-based, 50-500ms)
- Granularity: Per-branch deletion (other branches unaffected)

**Level 2: File Modification Time** (File-level optimization)
- After Level 1 detects branch change, check individual file mtimes
- Skip files with mtime older than last index time
- Avoids processing unchanged files after bulk deletion
- Risk: mtime can be unreliable (copied files, timezone issues) - use conservatively

**Metadata-Based Sync Pattern** (from RDR-04):
```python
class GitMetadataSync:
    """Query Qdrant metadata for indexed projects (source of truth)."""

    def get_indexed_projects(self, collection_name: str) -> Dict[str, str]:
        """Get all (git_project_identifier, git_commit_hash) from Qdrant."""
        indexed = {}
        offset = None

        while True:
            points, offset = self.qdrant.scroll(
                collection_name=collection_name,
                with_payload=["git_project_identifier", "git_commit_hash"],
                with_vectors=False,
                limit=100,
                offset=offset
            )

            if not points:
                break

            for point in points:
                identifier = point.payload.get("git_project_identifier")
                commit = point.payload.get("git_commit_hash")
                if identifier and commit:
                    indexed[identifier] = commit

            if offset is None:
                break

        return indexed
```

**Design Rationale**:
- **Metadata-based sync**: Qdrant is single source of truth, can't get out of sync
- **Git commit comparison**: O(1) detection of branch changes
- **Idempotent re-indexing**: If crash occurs, re-run indexing - metadata sync will skip already-indexed content
- **No SQLite checkpoint**: Source code processing is fast (no OCR like PDFs), crash recovery not needed

#### 6. .gitignore Integration

**Pattern** (from arcaneum-24):
```bash
# Use git ls-files to respect .gitignore
cd "$project_root"
for ext in $extensions; do
    git ls-files "*$ext" 2>/dev/null >> "$output_file"
done

# Convert relative to absolute paths
while read relative_path; do
    echo "$project_root/$relative_path"
done < "$output_file"
```

**Benefits**:
- Native .gitignore awareness (no custom parser needed)
- Respects .git/info/exclude and global excludes
- Handles nested .gitignore files correctly
- Efficient (git's internal implementation)

#### 7. Multi-Branch Support Pattern

**Composite Identifier Design**:

The `git_project_identifier` combines project name and branch: `f"{project_name}#{branch}"`

**Examples**:
- `"arcaneum#main"` - Main branch of arcaneum project
- `"arcaneum#feature-x"` - Feature branch of arcaneum project
- `"project-a#develop"` - Develop branch of project-a

**Benefits**:
- Multiple branches of same repo coexist in collection
- Branch-specific deletion on updates
- Branch-aware querying and comparison
- Other branches unaffected by updates to one branch

**Usage Pattern**:
```python
# Index directory with repos on different branches
$ arcaneum index ~/code/ --collection MyCode

# Result in Qdrant:
# - "project-a#main" indexed (commit abc123)
# - "project-b#feature-x" indexed (commit def456)
# - "project-c#develop" indexed (commit ghi789)

# User switches project-a to feature-y branch
$ cd ~/code/project-a && git checkout feature-y

# Re-index
$ arcaneum index ~/code/ --collection MyCode

# Result:
# - "project-a#main" still exists (untouched)
# - "project-a#feature-y" now indexed (NEW)
# - "project-b#feature-x" unchanged (skipped)
# - "project-c#develop" unchanged (skipped)
```

#### 8. Production Code Indexing Patterns

**Insights from 20+ Open Source Projects** (from arcaneum-28):

**ChunkHound** (MCP Integration):
- Implements Model Context Protocol for Claude integration
- Real-time indexing with file watching
- Multi-hop semantic search (follow code relationships)

**SCIP (Sourcegraph)**:
- Language Server Index Format
- 10x faster than LSIF in CI environments
- 4x smaller file sizes (compressed)
- Human-readable symbol IDs

**CodeSearchNet** (GitHub Dataset):
- 6M methods from open source repositories
- Tree-sitter tokenization
- NDCG-based evaluation metrics
- Supports Go, Java, JavaScript, Python, PHP, Ruby

**Key Techniques**:
1. **cAST Algorithm**: Recursive splitting + greedy merging (4.3 point gain)
2. **Tree-sitter**: Dominant parser (165+ languages)
3. **Hybrid Search**: Combine semantic + full-text for better recall
4. **Real-time Indexing**: File watchers for automatic updates
5. **Multi-hop Search**: Follow imports, inheritance, references

## Proposed Solution

### Approach

Implement a **git-aware source code indexing pipeline with multi-branch support** that:

1. **Discovers git projects** using recursive .git directory search with depth control
2. **Extracts git metadata** (commit hash, remote URL, branch, project name) with robust error handling
3. **Supports multiple branches** via composite identifier pattern (`project#branch`)
4. **Respects .gitignore** using git ls-files for tracked files only
5. **Chunks code intelligently** using tree-sitter AST parsing with fallback to line-based chunking
6. **Detects changes** via metadata-based sync (query Qdrant for indexed commits, compare with git HEAD)
7. **Deletes efficiently** using Qdrant's filter-based branch-specific deletion (40-100x faster than ChromaDB)
8. **Embeds with code-specific models** (jina-code-embeddings or v2-base-code)
9. **Read-only operations** - never mutates git state (no pull, fetch, checkout)

### Technical Design

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              Source Code Indexing Pipeline                  │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
       Git Discovery              Query Qdrant Metadata
       ─────────────              ─────────────────────
            │                               │
    find .git dirs              get indexed (project, branch,
    + extract metadata          commit) tuples - SOURCE OF TRUTH
            │                               │
            └───────────────┬───────────────┘
                            │
                   ┌────────▼─────────┐
                   │  Compare Commits │
                   │  git HEAD vs     │
                   │  Qdrant stored   │
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
                   │  AST-Aware Chunking         │
                   │  (tree-sitter + fallback)   │
                   └────────┬────────────────────┘
                            │
                   ┌────────▼────────────────────┐
                   │  Embedding Generation       │
                   │  (FastEmbed + jina-code)    │
                   └────────┬────────────────────┘
                            │
                   ┌────────▼────────────────────┐
                   │  Batch Upload to Qdrant     │
                   │  (150 chunks, retry logic)  │
                   └─────────────────────────────┘
```

#### Core Components

**1. Git Project Discovery**
```python
class GitProjectDiscovery:
    def find_git_projects(self, input_path: str, depth: Optional[int] = None) -> List[str]:
        """Find all .git directories with optional depth control."""
        find_cmd = ["find", input_path]
        if depth is not None:
            find_cmd.extend(["-maxdepth", str(depth + 1)])
        find_cmd.extend(["-name", ".git", "-type", "d"])

        # Execute and extract project roots
        result = subprocess.run(find_cmd, capture_output=True, text=True)
        git_dirs = result.stdout.strip().split('\n')
        project_roots = [os.path.dirname(d) for d in git_dirs if d]
        return sorted(set(project_roots))

    def extract_metadata(self, project_root: str) -> GitMetadata:
        """Extract git metadata with robust error handling."""
        try:
            repo = git.Repo(project_root)
            commit_hash = repo.head.commit.hexsha

            # Handle detached HEAD
            try:
                branch = repo.active_branch.name
            except TypeError:  # Detached HEAD
                try:
                    branch = f"(tag){repo.git.describe('--tags', '--exact-match')}"
                except git.GitCommandError:
                    branch = f"(detached-{commit_hash[:12]})"

            # Get remote URL with credential sanitization
            try:
                remote_url = repo.remote('origin').url
                remote_url = self._sanitize_url(remote_url)
            except ValueError:
                remote_url = "unknown"

            project_name = self._derive_project_name(remote_url, project_root)

            return GitMetadata(
                project_root=project_root,
                commit_hash=commit_hash,
                branch=branch,
                remote_url=remote_url,
                project_name=project_name
            )
        except git.InvalidGitRepositoryError:
            return None
```

**2. AST-Aware Code Chunker**
```python
from tree_sitter_language_pack import get_parser
from llama_index.text_splitter import CodeSplitter

class ASTCodeChunker:
    LANGUAGE_MAP = {
        ".py": "python",
        ".java": "java",
        ".js": "javascript",
        ".ts": "typescript",
        ".go": "go",
        ".rs": "rust",
        # ... 165+ languages supported
    }

    def chunk_code(self, file_path: str, code: str, max_chunk_size: int = 400) -> List[Chunk]:
        """Chunk code using AST with fallback to line-based."""
        file_ext = os.path.splitext(file_path)[1].lower()
        language = self.LANGUAGE_MAP.get(file_ext, "text")

        try:
            # Attempt AST-based chunking
            splitter = CodeSplitter(
                language=language,
                chunk_lines=max_chunk_size,  # Token equivalent
                chunk_overlap=20,  # 5% overlap
                max_chars=max_chunk_size * 3.5  # Conservative char estimate
            )
            chunks = splitter.split_text(code)
            extraction_method = f"ast_{language}"
        except Exception as e:
            # Fallback to line-based chunking
            logger.warning(f"AST parsing failed for {file_path}: {e}, falling back")
            chunks = self._line_based_chunking(code, max_chunk_size)
            extraction_method = "line_based"

        return [Chunk(content=c, method=extraction_method) for c in chunks]
```

**3. Metadata-Based Sync and Change Detection**
```python
class GitMetadataSync:
    """Query Qdrant for indexed projects (source of truth, follows RDR-04 pattern)."""

    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant = qdrant_client
        self._cache = {}  # Cache indexed projects per collection

    def get_indexed_projects(self, collection_name: str) -> Dict[str, str]:
        """Get all (git_project_identifier, git_commit_hash) from Qdrant."""
        if collection_name in self._cache:
            return self._cache[collection_name]

        indexed = {}
        offset = None

        while True:
            points, offset = self.qdrant.scroll(
                collection_name=collection_name,
                with_payload=["git_project_identifier", "git_commit_hash"],
                with_vectors=False,
                limit=100,
                offset=offset
            )

            if not points:
                break

            for point in points:
                identifier = point.payload.get("git_project_identifier")
                commit = point.payload.get("git_commit_hash")
                if identifier and commit:
                    indexed[identifier] = commit

            if offset is None:
                break

        self._cache[collection_name] = indexed
        return indexed

    def should_reindex_project(self, collection_name: str,
                               project_identifier: str,
                               current_commit: str) -> bool:
        """Check if (project, branch) needs re-indexing by comparing commits."""
        indexed = self.get_indexed_projects(collection_name)

        # Not indexed yet
        if project_identifier not in indexed:
            return True

        # Commit changed
        if indexed[project_identifier] != current_commit:
            return True

        # Unchanged
        return False
```

**4. Qdrant Bulk Deletion (Branch-Specific)**
```python
class QdrantIndexer:
    def delete_branch_chunks(self, collection_name: str, project_identifier: str):
        """Delete chunks for specific (project, branch) combination."""
        # Filter by composite identifier
        self.client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="git_project_identifier",
                        match=MatchValue(value=project_identifier)
                    )
                ]
            )
        )
        logger.info(f"Deleted chunks for {project_identifier} (50-500ms)")

    def upload_chunks_batch(self, collection_name: str, chunks: List[CodeChunk], batch_size: int = 150):
        """Upload chunks in optimized batches for Qdrant."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            points = [
                PointStruct(
                    id=chunk.id,
                    vector=chunk.embedding,
                    payload={
                        "git_project_identifier": chunk.git_project_identifier,  # PRIMARY
                        "file_path": chunk.file_path,
                        "git_project_name": chunk.git_project_name,
                        "git_branch": chunk.git_branch,
                        "git_commit_hash": chunk.git_commit_hash,
                        "programming_language": chunk.language,
                        "ast_chunked": chunk.ast_chunked,
                        # ... other metadata
                    }
                )
                for chunk in batch
            ]
            self.client.upsert(collection_name=collection_name, points=points)
```


#### Metadata Schema

```python
@dataclass
class CodeChunkMetadata:
    # PRIMARY identifier for multi-branch support
    git_project_identifier: str  # "project-name#branch" (e.g., "arcaneum#main")

    # Common file fields
    file_path: str
    filename: str
    file_extension: str
    programming_language: str
    file_size: int
    line_count: int
    chunk_index: int
    chunk_count: int
    text_extraction_method: str  # "ast_python", "ast_java", "line_based"

    # Git metadata fields (all required for git-only mode)
    git_project_root: str
    git_project_name: str  # Component of identifier, kept for filtering
    git_branch: str         # Component of identifier, kept for filtering
    git_commit_hash: str    # Full 40-char SHA
    git_remote_url: Optional[str] = None

    # Code analysis fields
    ast_chunked: bool = False
    has_functions: bool = False
    has_classes: bool = False
    has_imports: bool = False

    # Embedding metadata
    embedding_model: str = "jina-code-embeddings-1.5b"
    store_type: str = "source-code"
```

### Implementation Example

**Main Indexing Pipeline (Metadata-Based Sync)**:
```python
class SourceCodeIndexer:
    def __init__(self, qdrant_client, embedding_model="jina-code-embeddings-1.5b"):
        self.qdrant = QdrantIndexer(qdrant_client)
        self.git_discovery = GitProjectDiscovery()
        self.chunker = ASTCodeChunker()
        self.sync = GitMetadataSync(qdrant_client)  # Source of truth
        self.embedder = FastEmbedModel(embedding_model)

    def index_directory(self, input_path: str, collection_name: str,
                       depth: Optional[int] = None, force: bool = False):
        """Index all repositories in directory, respecting current branches.

        Args:
            force: If True, bypass incremental sync and reindex all projects
        """

        # Step 1: Query Qdrant for already-indexed projects (source of truth)
        if not force:
            indexed_projects = self.sync.get_indexed_projects(collection_name)
            logger.info(f"Found {len(indexed_projects)} already indexed (project, branch) combinations")
        else:
            indexed_projects = {}
            logger.info("Force mode: bypassing incremental sync")

        # Step 2: Discover git projects
        git_projects = self.git_discovery.find_git_projects(input_path, depth)
        logger.info(f"Found {len(git_projects)} git projects")

        # Step 3: Process each project on its CURRENT branch
        projects_to_index = []
        for project_root in git_projects:
            # Extract metadata (includes current branch)
            git_metadata = self.git_discovery.extract_metadata(project_root)

            # Create composite identifier
            identifier = f"{git_metadata.project_name}#{git_metadata.branch}"

            # Check if THIS branch needs indexing (query Qdrant metadata)
            needs_indexing = self.sync.should_reindex_project(
                collection_name, identifier, git_metadata.commit_hash
            )

            if not needs_indexing and not force:
                logger.info(f"Skipping {identifier} (commit {git_metadata.commit_hash[:12]} already indexed)")
                continue

            if identifier in indexed_projects:
                # Delete ONLY this branch's chunks (commit changed)
                logger.info(f"Re-indexing {identifier} (commit changed: "
                          f"{indexed_projects[identifier][:12]} → {git_metadata.commit_hash[:12]})")
                self.qdrant.delete_branch_chunks(collection_name, identifier)
            else:
                logger.info(f"Indexing new branch: {identifier}")

            projects_to_index.append((project_root, git_metadata, identifier))

        # Step 4: Index files for each project
        for project_root, git_metadata, identifier in projects_to_index:
            code_files = self._get_git_files(project_root)
            self._process_files(code_files, collection_name, git_metadata, identifier)

    def _process_files(self, files: List[str], collection_name: str,
                       git_metadata: GitMetadata, identifier: str):
        """Process files and upload chunks in batches."""
        batch = []

        for file_path in files:
            try:
                # Read and chunk
                code = open(file_path).read()
                chunks = self.chunker.chunk_code(file_path, code)

                # Generate embeddings
                embeddings = self.embedder.embed([c.content for c in chunks])

                # Create metadata with composite identifier
                for chunk, embedding in zip(chunks, embeddings):
                    batch.append(CodeChunk(
                        content=chunk.content,
                        embedding=embedding,
                        file_path=file_path,
                        git_project_identifier=identifier,  # PRIMARY
                        git_project_name=git_metadata.project_name,
                        git_branch=git_metadata.branch,
                        git_commit_hash=git_metadata.commit_hash,
                        # ... other fields
                    ))

                    # Upload when batch is full
                    if len(batch) >= 150:
                        self.qdrant.upload_chunks_batch(collection_name, batch)
                        batch = []

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        # Upload remaining batch
        if batch:
            self.qdrant.upload_chunks_batch(collection_name, batch)

    def _get_git_files(self, project_root: str) -> List[str]:
        """Get tracked files using git ls-files."""
        os.chdir(project_root)
        result = subprocess.run(
            ["git", "ls-files", "*.py", "*.java", "*.js", "*.go"],
            capture_output=True,
            text=True
        )
        relative_paths = result.stdout.strip().split('\n')
        return [os.path.join(project_root, p) for p in relative_paths if p]
```

## Alternatives Considered

### Alternative 1: Use ASTChunk Library Directly

**Description**: Use ASTChunk as the primary AST chunking library without alternatives

**Pros**:
- Proven cAST algorithm with 4.3 point gain on RepoEval
- Simple integration
- Active development

**Cons**:
- Only supports 4 languages (Python, Java, C#, TypeScript)
- Missing 11 required languages
- No built-in fallback mechanisms
- Would require implementing tree-sitter parsing separately

**Reason for rejection**: Insufficient language coverage (4 vs 15+ requirement)

### Alternative 2: Incremental Update Only (No Bulk Delete)

**Description**: Use file-level change detection exclusively, never bulk delete projects

**Pros**:
- Optimal for small changes (single file edits)
- More granular control
- Potentially faster for continuous indexing

**Cons**:
- Complex to implement correctly
- Higher risk of partial failures
- Difficult to resume after interruption
- May miss moved/renamed files
- More complex recovery logic

**Reason for rejection**: Trade-off complexity vs benefit not justified; hybrid approach provides best of both worlds

### Alternative 3: Use jina-embeddings-v3 for Code

**Description**: Use generic jina-v3 model with code LoRA adapter

**Pros**:
- Latest Jina model
- Supports multiple task adapters
- Matryoshka dimensionality

**Cons**:
- Inferior code performance (0.7564 vs 0.7753 for v2-base-code)
- License restrictions (CC BY-NC 4.0 requires commercial licensing)
- Code adapter unpublished/unavailable
- No advantage over code-specific models

**Reason for rejection**: Worse performance and licensing issues make it unsuitable for production code indexing

### Alternative 4: Branch as Separate Metadata Field (No Composite Key)

**Description**: Keep `git_project_name` as primary key, use `git_branch` for filtering only

**Pros**:
- Simpler metadata schema (no composite identifier)
- Slightly less storage (no duplicate identifier field)

**Cons**:
- Deletion still affects all branches (no isolation)
- Change detection ambiguous (which branch changed?)
- Cannot distinguish between "project changed" vs "branch changed"
- Requires complex filtering to isolate branches
- Risk of accidental deletion across all branches

**Reason for rejection**: Composite identifier provides clear branch isolation and safer deletion semantics

## Trade-offs and Consequences

### Positive Consequences

1. **40-100x Faster Deletion**: Qdrant's filter-based branch-specific deletion dramatically outperforms ChromaDB's ID-based approach
2. **165+ Language Support**: tree-sitter-language-pack provides comprehensive coverage beyond original requirements
3. **Multi-Branch Support**: Enables branch comparison workflows and safe branch-specific updates
4. **Metadata-Based Sync**: Qdrant is single source of truth, can't get out of sync (follows RDR-04 pattern)
5. **Branch Isolation**: Updates to one branch don't affect others (safer deletion semantics)
6. **Manual Deletion Recovery**: Automatically detects missing chunks and re-indexes (Qdrant metadata query)
7. **Code-Specific Embeddings**: jina-code models optimized for code retrieval (79% accuracy, 32K context)
8. **Proven Algorithm**: cAST algorithm validated with 4.3 point gain on academic benchmarks
9. **Read-Only Operations**: No git mutations provide safety and predictability
10. **Architectural Consistency**: Same metadata-based sync pattern as RDR-04 PDF indexing

### Negative Consequences

1. **Metadata Query Overhead**: Must scroll Qdrant on startup to get indexed projects (O(n) projects, cached per run)
2. **Storage Overhead**: Multiple branches of same repo increase storage (intentional trade-off for comparison capability)
3. **Branch Complexity**: Multi-branch support adds composite identifier logic vs simple project names
4. **Model Size**: jina-code-embeddings-1.5b (1.54B params) is larger than alternatives (may require GPU)
5. **Git-Only Requirement**: No support for non-git directories (simplification trade-off)
6. **First-Time Cost**: Initial indexing processes all files (no incremental benefit until changes occur)

### Risks and Mitigations

**Risk**: tree-sitter parser crashes on malformed code
**Mitigation**: Automatic fallback to line-based chunking, error logging for debugging

**Risk**: Git operations timeout on very large repositories
**Mitigation**: 5-second timeout per git command, skip problematic repos with warning

**Risk**: FastEmbed doesn't support jina-code-embeddings models yet
**Mitigation**: Use jina-embeddings-v2-base-code (proven, Apache 2.0) as fallback

**Risk**: Filter-based deletion deletes wrong chunks if metadata incorrect
**Mitigation**: Dry-run mode, careful metadata validation, audit trail in transactions table

**Risk**: Memory exhaustion with large files (32K context embeddings)
**Mitigation**: File size limits (10MB), chunk large files even with 32K context

## Implementation Plan

### Prerequisites

- [ ] Qdrant server running (RDR-002)
- [ ] Collection created with appropriate embedding dimensions (RDR-003)
- [ ] Python >= 3.12 installed
- [ ] Git >= 2.30 installed
- [ ] tree-sitter-language-pack installed
- [ ] FastEmbed with jina model support

### Step-by-Step Implementation

#### Step 1: Core Git Operations Module

Create `src/arcaneum/indexing/git_operations.py`:
- Implement `GitProjectDiscovery` class
- Implement `extract_metadata()` with robust error handling
- Handle edge cases: detached HEAD, shallow clones, missing remotes, submodules
- Add credential sanitization for remote URLs
- Include timeout protection (5s per git command)

**Estimated effort**: 2-3 days

#### Step 2: AST Chunking Module

Create `src/arcaneum/indexing/ast_chunker.py`:
- Integrate tree-sitter-language-pack
- Implement `ASTCodeChunker` class with language detection
- Add fallback to line-based chunking
- Configure chunk sizes per embedding model (32K vs 8K context)
- Track chunking method in metadata

**Estimated effort**: 3-4 days

#### Step 3: Metadata-Based Sync Module

Create `src/arcaneum/indexing/git_metadata_sync.py`:
- Implement `GitMetadataSync` class (follows RDR-04 pattern)
- Query Qdrant for indexed (project, branch, commit) tuples
- Cache results to avoid repeated queries
- Provide `should_reindex_project()` method
- Single source of truth (Qdrant metadata)

**Estimated effort**: 1-2 days

#### Step 4: Qdrant Integration Module

Create `src/arcaneum/indexing/qdrant_indexer.py`:
- Implement filter-based bulk deletion
- Create batch upload with 100-200 chunk batching
- Add gRPC support for faster uploads
- Implement retry logic with exponential backoff

**Estimated effort**: 2-3 days

#### Step 5: Main Orchestration Pipeline

Create `src/arcaneum/indexing/source_code_pipeline.py`:
- Implement `SourceCodeIndexer` orchestrator class
- Integrate all modules (git, AST, change detection, Qdrant)
- Add progress reporting (tqdm/rich)
- Add CLI interface with argument parsing

**Estimated effort**: 3-4 days

#### Step 6: Branch Comparison Query Examples

Add to documentation/examples:
- Query specific branches using `git_project_identifier`
- Compare implementations across branches
- List all branches of a project
- Branch-specific deletion examples

**Estimated effort**: 1 day

#### Step 7: MCP Plugin Wrapper

Create `plugins/qdrant-indexer/mcp_server.py`:
- Expose `index_source_code()` MCP tool
- Add `check_indexing_status()` tool
- Implement `delete_project()` tool
- Return structured progress updates
- Handle errors gracefully for Claude UI

**Estimated effort**: 2 days

### Files to Create

- `src/arcaneum/indexing/git_operations.py` - Git discovery and metadata extraction with branch support
- `src/arcaneum/indexing/ast_chunker.py` - AST-aware code chunking
- `src/arcaneum/indexing/git_metadata_sync.py` - Metadata-based sync (queries Qdrant, follows RDR-04)
- `src/arcaneum/indexing/qdrant_indexer.py` - Qdrant upload/branch-specific deletion
- `src/arcaneum/indexing/source_code_pipeline.py` - Main orchestrator with metadata-based sync
- `src/arcaneum/indexing/types.py` - Metadata schema with git_project_identifier
- `plugins/qdrant-indexer/mcp_server.py` - MCP plugin interface
- `tests/test_git_operations.py` - Git operations and branch detection tests
- `tests/test_ast_chunker.py` - AST chunking tests
- `tests/test_metadata_sync.py` - Metadata-based sync tests (Qdrant queries)
- `tests/test_qdrant_indexer.py` - Qdrant integration and branch deletion tests
- `tests/test_multi_branch.py` - Multi-branch workflow tests
- `tests/test_incremental_sync.py` - Incremental indexing tests (like RDR-04)

### Dependencies

Add to `pyproject.toml`:
```toml
[project.dependencies]
qdrant-client = "^1.15.0"
fastembed = "^0.3.0"
tree-sitter-language-pack = "^0.5.0"
llama-index = "^0.9.0"
GitPython = "^3.1.40"
tenacity = "^8.2.0"
rich = "^13.7.0"
```

## Validation

### Testing Approach

**Unit Tests**:
- Git metadata extraction with mocked git repos
- AST chunking for each supported language
- Metadata-based sync (Qdrant queries)
- Qdrant upload/deletion with test collection

**Integration Tests**:
- End-to-end indexing of sample git repository
- Incremental sync with commit changes (query Qdrant metadata)
- Manual deletion recovery (Qdrant as source of truth)
- Force reindex bypass

**Performance Tests**:
- Benchmark filter-based deletion vs ID-based
- Measure AST chunking speed per language
- Test batch upload throughput (chunks/sec)
- Validate metadata query overhead (Qdrant scroll)
- Measure incremental sync performance (commit detection)

### Test Scenarios

**Scenario 1: Initial Git Repository Indexing**
- **Setup**: Fresh git repository with 100 Python files
- **Action**: Run `arcaneum index --input /path/to/repo --collection CodeLibrary`
- **Expected**: All files indexed with git metadata

**Scenario 2: Incremental Sync After Commit**
- **Setup**: Repository from Scenario 1
- **Action**: Edit files, commit, re-run indexing
- **Expected**: Qdrant metadata query detects commit change, bulk deletion triggered, branch re-indexed

**Scenario 3: Manual Deletion Recovery**
- **Setup**: Indexed repository (chunks in Qdrant)
- **Action**: Manually delete some chunks from Qdrant via admin tool, re-run indexing
- **Expected**: Metadata query detects missing chunks (no match for commit hash), re-indexes automatically (source of truth = Qdrant)

**Scenario 4: Branch Comparison Query**
- **Setup**: Same file on two branches with different content
- **Action**: Search for same term in both branches using `git_project_identifier` filter
- **Expected**: Different results reflecting branch-specific content

**Scenario 5: Multiple Branches Coexist**
- **Setup**: Index repo on main, then checkout and index feature-x branch
- **Action**: Query for chunks from each branch
- **Expected**: Both branches exist in collection, can query either independently

**Scenario 6: Crash Recovery via Re-indexing**
- **Setup**: Large repository indexing (simulate interruption after 50%)
- **Action**: Kill process, restart indexing
- **Expected**: Metadata-based sync skips already-indexed commits, idempotent re-indexing (no duplicates)

### Performance Validation

**Metrics to Track**:
- Files indexed per second
- Chunks uploaded per second
- Git operations latency (metadata extraction)
- AST parsing success rate per language
- Filter-based deletion speed
- Metadata query performance (Qdrant scroll)
- Memory usage during large file processing

**Targets**:
- 100-200 files/sec indexing throughput
- < 1s per git metadata extraction
- > 95% AST parsing success (rest fallback to line-based)
- < 500ms for branch-specific bulk deletion
- < 5s for metadata query on collection with 1000 projects

### Security Validation

**Credential Sanitization**:
- Verify remote URLs stripped of credentials
- Test with https://user:pass@github.com/repo patterns
- Validate SSH key URLs don't expose keys

**Metadata Security**:
- Ensure file paths don't expose sensitive directories
- Validate no secrets stored in Qdrant metadata

## Branch Comparison Query Examples

### Query Specific Branch

```python
# Search in main branch
results_main = client.search(
    collection_name="MyCode",
    query_vector=embed("authentication logic"),
    query_filter=Filter(must=[
        FieldCondition(
            key="git_project_identifier",
            match=MatchValue("arcaneum#main")
        )
    ]),
    limit=10
)
```

### Compare Implementations Across Branches

```python
# Search in feature branch
results_feature = client.search(
    collection_name="MyCode",
    query_vector=embed("authentication logic"),
    query_filter=Filter(must=[
        FieldCondition(
            key="git_project_identifier",
            match=MatchValue("arcaneum#feature-auth")
        )
    ]),
    limit=10
)

# Compare results side-by-side
for main_chunk, feature_chunk in zip(results_main, results_feature):
    print(f"Main: {main_chunk.payload['file_path']}")
    print(f"Feature: {feature_chunk.payload['file_path']}")
    print(f"Similarity: {main_chunk.score} vs {feature_chunk.score}")
```

### List All Branches of a Project

```python
# Find all branches of arcaneum project
all_branches = client.scroll(
    collection_name="MyCode",
    scroll_filter=Filter(must=[
        FieldCondition(
            key="git_project_name",
            match=MatchValue("arcaneum")
        )
    ]),
    limit=1000,
    with_payload=["git_project_identifier", "git_branch", "git_commit_hash"]
)

# Extract unique branches
branches = set(point.payload["git_branch"] for point in all_branches[0])
print(f"Available branches: {branches}")
# Output: {'main', 'feature-auth', 'develop'}
```

### Delete Specific Branch

```python
# Delete only the feature branch, keep main and develop
client.delete(
    collection_name="MyCode",
    points_selector=Filter(must=[
        FieldCondition(
            key="git_project_identifier",
            match=MatchValue("arcaneum#feature-auth")
        )
    ])
)
```

## References

- [Beads Issues arcaneum-24 to arcaneum-30, arcaneum-32 to arcaneum-42](../../.beads/arcaneum.db) - Detailed research findings
- [chroma-embedded/upload.sh](../../../research/chroma-embedded/upload.sh) - Reference implementation patterns
- [RDR-002: Qdrant Server Setup](RDR-002-qdrant-server-setup.md) - Server configuration
- [RDR-003: Collection Creation](RDR-003-collection-creation.md) - Collection management
- [RDR-004: PDF Bulk Indexing](RDR-004-pdf-bulk-indexing.md) - Parallel research workflow
- [tree-sitter-language-pack Documentation](https://github.com/grantjenks/py-tree-sitter-language-pack)
- [LlamaIndex CodeSplitter](https://docs.llamaindex.ai/en/stable/api_reference/text_splitter/)
- [Jina Embeddings v2 Model Card](https://huggingface.co/jinaai/jina-embeddings-v2-base-code)
- [jina-code-embeddings Blog Post](https://jina.ai/news/jina-code-embeddings-v1)
- [cAST Paper: Enhancing Code Retrieval-Augmented Generation](https://arxiv.org/abs/XXXX.XXXXX)
- [Qdrant Filter Documentation](https://qdrant.tech/documentation/concepts/filtering/)
- [GitPython Documentation](https://gitpython.readthedocs.io/)

## Notes

**Implementation Priority**:
1. Git operations with branch awareness (core functionality)
2. Branch-aware change detection with composite keys (critical)
3. AST chunking with tree-sitter (quality improvements)
4. Qdrant integration with branch-specific deletion (performance)
5. MCP plugin wrapper (user experience)

**Future Enhancements**:
- Real-time indexing with file watchers (inotify, FSEvents)
- Multi-hop semantic search (follow imports, inheritance)
- Hybrid search combining semantic + full-text (MeiliSearch integration)
- SCIP index generation for code intelligence
- Support for Jupyter notebooks (.ipynb) with code cell extraction
- Parallel worker pool for multi-core indexing
- Automatic branch cleanup (delete stale branches)
- Branch-to-branch diff queries

**Compatibility Notes**:
- Requires Qdrant >= 1.15.4 for filter-based deletion
- Git >= 2.30 for git ls-files patterns
- Python >= 3.12 for modern type hints
- tree-sitter binaries for all target platforms

**Migration from ChromaDB**:
- Collection schema compatible (same metadata fields)
- Embedding dimensions match (jina-v2-base-code: 768D, jina-code: 1536D)
- Can run migration script to copy existing chunks
- Filter-based queries map to Qdrant's FieldCondition syntax

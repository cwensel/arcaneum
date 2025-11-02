# Testing Source Code Indexing (RDR-005)

## Prerequisites

1. **Qdrant Server Running**
   ```bash
   # Start Qdrant
   arc container start
   ```

2. **Verify Installation**
   ```bash
   pip install -e .
   arc --help
   ```

## Quick Start

### 1. Create a Typed Collection

```bash
# Create a collection for source code (typed)
arc collection create code --model stella --type code

# Or for PDFs (separate collection)
arc collection create docs --model stella --type pdf
```

**Important:** Collections are now typed (pdf or code). You cannot mix PDFs and source code in the same collection.

**Collection vs Corpus:**
- **Collection**: Qdrant-only (semantic search)
- **Corpus**: Qdrant + MeiliSearch (semantic + full-text search, see RDR-009)
- Both use the same type system

### 2. Index Source Code

**Index a single repository:**
```bash
arc index code /path/to/your/repo --collection code
```

**Index directory with multiple repositories:**
```bash
# Index all repos in ~/projects
arc index code ~/projects --collection code

# With depth limit (only immediate subdirectories)
arc index code ~/projects --collection code --depth 1
```

**Force re-index (bypass incremental sync):**
```bash
arc index code /path/to/repo --collection code --force
```

## Features

### Metadata-Based Incremental Sync

The indexer automatically detects which projects need re-indexing by querying Qdrant:

- **First run**: Indexes all projects
- **Subsequent runs**: Only re-indexes changed projects (commit hash comparison)
- **Unchanged projects**: Skipped automatically

### Multi-Branch Support

Multiple branches of the same repository can coexist in a collection:

```bash
# Index main branch
cd ~/myproject
git checkout main
arc index code ~ --collection code

# Index feature branch (both will exist in collection)
git checkout feature-x
arc index code ~ --collection code
```

Result in Qdrant:
- `myproject#main` - chunks from main branch
- `myproject#feature-x` - chunks from feature branch

### Supported Languages

15+ primary languages with AST-aware chunking:
- Python, Java, JavaScript, TypeScript
- C#, Go, Rust, C/C++
- PHP, Ruby, Kotlin, Scala, Swift
- Plus 40+ more via tree-sitter fallback

## Workflow Example

```bash
# 1. Create typed collection
arc collection create code --model stella --type code

# 2. Initial index
arc index code ~/code --collection code

# 3. Make some commits in your repos
cd ~/code/project-a
# ... make changes, commit ...

# 4. Re-index (only changed projects will be re-indexed)
arc index code ~/code --collection code

# Output shows:
# ✓ project-a#main (commit abc123 already indexed) - skipped
# ↻ project-b#main (commit changed: def456 → ghi789) - re-indexed
```

## Verifying Results

### Check Collection Info

```bash
arc collection info code

# Shows:
# Collection: code
# Type: code  ← Collection type
# Points: 1234
# Status: green
# ...
```

### Query Metadata (Python)

```python
from qdrant_client import QdrantClient
from arcaneum.indexing.collection_metadata import get_collection_type

client = QdrantClient("localhost", port=6333)

# Check collection type
collection_type = get_collection_type(client, "code")
print(f"Collection type: {collection_type}")  # "code"

# Get all indexed projects
result = client.scroll(
    collection_name="code",
    with_payload=["git_project_identifier", "git_commit_hash", "store_type"],
    with_vectors=False,
    limit=100
)

for point in result[0]:
    print(f"{point.payload['git_project_identifier']}: {point.payload['git_commit_hash'][:12]}")
```

### Search Code (future RDR-007)

```bash
# Coming in RDR-007 semantic search
arc search "authentication logic" --collection code
```

## Performance

**Target metrics:**
- 100-200 files/sec indexing throughput
- <500ms branch-specific deletion
- <5s metadata query for 1000 projects
- >95% AST parsing success rate

## Troubleshooting

### Issue: No projects found

```bash
# Check if directory contains .git repositories
find /path/to/search -name ".git" -type d

# Try increasing depth
arc index code /path --collection test --depth 3
```

### Issue: Slow indexing

```bash
# Use verbose mode to see bottlenecks
arc index code /path --collection test --verbose
```

### Issue: Collection already exists

```bash
# List collections
arc collection list

# Check collection type
arc collection info my-collection

# Use existing collection or delete old one
arc collection delete old-collection --confirm
```

### Issue: Type mismatch

```bash
# Error: Collection 'docs' is type 'pdf', cannot index code code
arc index code ~/projects --collection docs

# Solution: Create separate typed collections
arc collection create code --model stella --type code
arc index code ~/projects --collection code
```

## Advanced Usage

### Custom Embedding Models

```bash
# Use different model (when supported)
arc index code /path --collection code --model jina-code
```

### Branch-Specific Deletion

```python
from arcaneum.indexing.qdrant_indexer import QdrantIndexer, create_qdrant_client

client = create_qdrant_client("localhost")
indexer = QdrantIndexer(client)

# Delete specific branch
indexer.delete_branch_chunks("my-collection", "project#old-branch")
```

### Statistics

```python
from arcaneum.indexing.git_metadata_sync import GitMetadataSync

sync = GitMetadataSync(client)

# Get all branches of a project
branches = sync.get_all_branches("my-collection", "myproject")
print(f"Branches: {branches}")

# Get project stats
stats = sync.get_project_stats("my-collection", "myproject#main")
print(f"Chunks: {stats.point_count}, Commit: {stats.commit_hash[:12]}")
```

## Testing Components

### Unit Tests

```bash
# Run all indexing tests
pytest tests/indexing/ -v

# Specific modules
pytest tests/indexing/test_git_operations.py -v
pytest tests/indexing/test_ast_chunker.py -v
pytest tests/indexing/test_git_metadata_sync.py -v
pytest tests/indexing/test_qdrant_indexer.py -v
```

**Test Coverage:**
- Git operations: 15 tests ✓
- AST chunking: 27 tests ✓
- Metadata sync: 26 tests ✓
- Qdrant integration: 26 tests ✓
- Type definitions: 13 tests ✓
- **Total: 107 passing unit tests**

## Known Limitations

1. **Read-only git operations**: No pull, fetch, or checkout
2. **Current branch only**: Indexes whatever branch is checked out
3. **No file watching**: Re-run manually to pick up changes
4. **FastEmbed models**: Limited code-specific models (expanding)

## Next Steps

- **RDR-007**: Semantic search for code
- **RDR-012**: Claude Code MCP integration
- **Performance benchmarking**: Validate 100-200 files/sec target
- **Multi-hop search**: Follow imports and references

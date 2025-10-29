# CLI Output Format Documentation

This document describes the output formats, error codes, and conventions used by the Arcaneum CLI following RDR-006 best practices.

## Exit Codes

All Arcaneum CLI commands follow a consistent exit code convention:

| Exit Code | Name | Meaning | Examples |
|-----------|------|---------|----------|
| **0** | SUCCESS | Command completed successfully | Collection created, files indexed, search returned results |
| **1** | ERROR | General error (network, server, unexpected) | Qdrant connection failed, embedding model error, unexpected exception |
| **2** | INVALID_ARGS | Invalid arguments or configuration | Unknown model name, invalid filter syntax, missing required field |
| **3** | NOT_FOUND | Resource not found | Collection doesn't exist, file path not found, git repository not found |

### Usage Examples

```bash
# Check exit code in bash
arc create-collection test --model stella
echo $?  # 0 = success

arc create-collection test --model invalid-model
echo $?  # 2 = invalid args

arc collection-info nonexistent
echo $?  # 3 = not found
```

## Output Modes

### Text Output (Default)

Human-readable formatted output using Rich library with colors and formatting.

**Features:**
- Color-coded messages (green for success, red for errors, yellow for warnings)
- Tables for list commands
- Progress indicators for long-running operations
- Emoji/symbols for visual clarity (✓, ✗, ➕, ↻)

**Example:**
```
✅ Created collection 'Research' with 2 models
  • stella: 1024D
  • bge: 1024D
```

### JSON Output (`--json` flag)

Machine-readable structured JSON output for scripting and automation.

**Standard Format:**
```json
{
  "status": "success|error",
  "message": "Human-readable summary",
  "data": {
    // Command-specific data
  },
  "errors": []
}
```

## Progress Output Format

Long-running commands (indexing, bulk operations) output progress messages with structured prefixes for Claude Code UI parsing.

### Progress Message Format

**Initialization:**
```
[INFO] Found 47 projects
[INFO] Indexing 5 projects...
```

**Progress Updates:**
```
[INFO] Processing 1/5 (20%) project-name#branch
[INFO] Processing 2/5 (40%) another-project#main
[INFO] Processing 3/5 (60%) third-project#develop
```

**Completion:**
```
[INFO] Complete: 5 projects, 142 files, 3847 chunks
```

### Error Messages

All errors use the `[ERROR]` prefix:
```
[ERROR] Collection 'Research' not found
[ERROR] Unknown model: invalid-model. Available: stella, bge, jina-code, modernbert
[ERROR] Invalid filter syntax: expected key=value
```

## JSON Schemas by Command

### Collection Management

#### `create-collection --json`

```json
{
  "status": "success",
  "message": "Created collection 'Research' with 2 models",
  "data": {
    "collection": "Research",
    "type": "pdf",
    "models": ["stella", "bge"],
    "vectors": {
      "stella": 1024,
      "bge": 1024
    },
    "hnsw": {
      "m": 16,
      "ef_construct": 100
    },
    "on_disk_payload": false
  },
  "errors": []
}
```

#### `list-collections --json`

```json
{
  "status": "success",
  "message": "Found 3 collections",
  "data": {
    "collections": [
      {
        "name": "Research",
        "points_count": 1247,
        "vectors": {
          "stella": {
            "size": 1024,
            "distance": "Cosine"
          }
        }
      },
      {
        "name": "Code",
        "points_count": 3842,
        "vectors": {
          "jina-code": {
            "size": 768,
            "distance": "Cosine"
          }
        }
      }
    ]
  },
  "errors": []
}
```

#### `collection-info <name> --json`

```json
{
  "status": "success",
  "message": "Collection 'Research' information",
  "data": {
    "name": "Research",
    "type": "pdf",
    "points_count": 1247,
    "status": "green",
    "vectors": {
      "stella": {
        "size": 1024,
        "distance": "Cosine"
      }
    },
    "hnsw_config": {
      "m": 16,
      "ef_construct": 100
    }
  },
  "errors": []
}
```

#### `delete-collection <name> --json --confirm`

```json
{
  "status": "success",
  "message": "Deleted collection 'OldCollection'",
  "data": {
    "deleted": "OldCollection"
  },
  "errors": []
}
```

### Model Management

#### `list-models --json`

```json
{
  "status": "success",
  "message": "Found 8 embedding models",
  "data": {
    "models": [
      {
        "alias": "jina-code",
        "model": "jinaai/jina-embeddings-v2-base-code",
        "dimensions": 768,
        "description": "Code-specific (768D, 8K context, best for source code)"
      },
      {
        "alias": "stella",
        "model": "dunzhang/stella_en_1.5B_v5",
        "dimensions": 1024,
        "description": "General purpose (1024D, high quality for docs/PDFs)"
      }
    ]
  },
  "errors": []
}
```

### Indexing Commands

#### `index-pdfs --json`

```json
{
  "status": "success",
  "message": "Indexed 47 PDF files",
  "data": {
    "files_discovered": 47,
    "files_processed": 47,
    "files_skipped": 0,
    "chunks_created": 1247,
    "chunks_uploaded": 1247,
    "ocr_used": 3
  },
  "errors": []
}
```

#### `index-source --json`

```json
{
  "status": "success",
  "message": "Indexed 5 projects",
  "data": {
    "projects_discovered": 10,
    "projects_indexed": 5,
    "projects_skipped": 5,
    "files_processed": 142,
    "chunks_created": 3847,
    "chunks_uploaded": 3847
  },
  "errors": []
}
```

### Search Commands

#### `search --json`

```json
{
  "status": "success",
  "message": "Found 5 results",
  "data": {
    "query": "authentication patterns",
    "collection": "Code",
    "vector_name": "jina-code",
    "total_results": 5,
    "execution_time_ms": 234.5,
    "results": [
      {
        "score": 0.95,
        "id": 12345,
        "file_path": "/code/auth.py",
        "programming_language": "python",
        "git_project_identifier": "myproject#main",
        "chunk_index": 0,
        "content": "def authenticate_user(username, password):..."
      }
    ]
  },
  "errors": []
}
```

## Error Format

### Error Response Structure

```json
{
  "status": "error",
  "message": "[ERROR] Collection 'Nonexistent' not found",
  "data": {},
  "errors": [
    "Collection 'Nonexistent' not found"
  ]
}
```

### Common Error Messages

#### Invalid Arguments (Exit Code 2)

```
[ERROR] Unknown model: invalid-model. Available: stella, bge, jina-code, modernbert
[ERROR] Invalid filter syntax: expected key=value or JSON object
[ERROR] --confirm flag required for non-interactive deletion
[ERROR] Python 3.12+ required
```

#### Resource Not Found (Exit Code 3)

```
[ERROR] Collection 'Research' not found
[ERROR] Path does not exist: /nonexistent/path
[ERROR] No git repository found at /path/to/non-repo
[ERROR] Model 'jina-code' not downloaded. Use 'arc download-model jina-code' first
```

#### General Errors (Exit Code 1)

```
[ERROR] Failed to connect to Qdrant at http://localhost:6333
[ERROR] Failed to create collection: Timeout waiting for server
[ERROR] Embedding generation failed: Out of memory
[ERROR] Search failed: Connection refused
```

## Progress Output Parsing

Claude Code can parse progress messages to track long-running operations.

### Pattern Matching

**Discovery Messages:**
```regex
^\[INFO\] Found (\d+) (projects|files|PDFs)$
```

**Progress Updates:**
```regex
^\[INFO\] Processing (\d+)/(\d+) \((\d+)%\) (.*)$
```
- Capture groups: current, total, percentage, item_name

**Completion Messages:**
```regex
^\[INFO\] Complete: (\d+) (projects|files), (\d+) (files|chunks), (\d+) chunks$
```
- Capture groups: item_count1, item_type1, item_count2, item_type2, chunk_count

### Example Progress Sequence

```
[INFO] Found 3 projects
[INFO] Indexing 3 projects...
[INFO] Processing 1/3 (33%) myproject#main
[INFO] Processing 2/3 (67%) another-project#develop
[INFO] Processing 3/3 (100%) third-project#feature-x
[INFO] Complete: 3 projects, 42 files, 1234 chunks
```

## Troubleshooting

### Common Issues

#### Exit Code 1: Qdrant Connection Failed

**Problem:**
```
[ERROR] Failed to connect to Qdrant at http://localhost:6333
```

**Solutions:**
1. Check if Qdrant Docker container is running: `docker ps | grep qdrant`
2. Start Qdrant: `docker compose up -d`
3. Verify port: `curl http://localhost:6333/healthz`

#### Exit Code 2: Unknown Model

**Problem:**
```
[ERROR] Unknown model: invalid. Available: stella, bge, jina-code, modernbert
```

**Solutions:**
1. Use one of the available models from the list
2. Run `arc list-models` to see all available models
3. Check for typos in model name

#### Exit Code 3: Collection Not Found

**Problem:**
```
[ERROR] Collection 'Research' not found
```

**Solutions:**
1. List existing collections: `arc list-collections`
2. Create the collection first: `arc create-collection Research --model stella`
3. Check for typos in collection name

### Verbose Mode

Add `--verbose` or `-v` flag to any command for detailed logging:

```bash
arc index-source /code --collection Code --verbose
```

This shows:
- Detailed step-by-step progress
- Timing information
- Debug output from libraries
- Full stack traces on errors

### JSON Mode for Debugging

Use `--json` flag to get machine-readable output for scripting:

```bash
arc list-collections --json | jq '.data.collections[] | .name'
```

## Best Practices

### For Scripting

1. **Always check exit codes:**
   ```bash
   if arc create-collection test --model stella --json; then
       echo "Success!"
   else
       echo "Failed with exit code: $?"
   fi
   ```

2. **Parse JSON output with jq:**
   ```bash
   arc list-models --json | jq -r '.data.models[] | select(.dimensions == 1024) | .alias'
   ```

3. **Use --confirm for non-interactive deletion:**
   ```bash
   arc delete-collection old-data --confirm --json
   ```

### For Claude Code Integration

1. **Monitor progress messages:**
   - Claude watches stdout for `[INFO]` prefixed lines
   - Progress updates show percentage completion
   - Completion summary provides final stats

2. **Handle errors gracefully:**
   - Check exit codes to determine error type
   - Parse `[ERROR]` messages for user feedback
   - Use `--verbose` for debugging

3. **Use JSON for structured data:**
   - Search results with metadata
   - Collection listings for automation
   - Model information for validation

## Future Enhancements

As noted in RDR-006, future versions may add:

1. **MCP Server Wrapper** - Structured tools with type hints
2. **Streaming Progress** - WebSocket-based real-time updates
3. **Background Jobs** - Daemon mode for long-running operations
4. **Enhanced Metrics** - Detailed performance and accuracy metrics

For now, the CLI-first approach with JSON support provides the foundation for these enhancements without breaking backward compatibility.

## References

- **RDR-006**: Claude Code Integration spec (lines 1140-1186)
- **Beads Plugin**: Reference implementation for best practices
- **Error Classes**: `src/arcaneum/cli/errors.py`
- **Output Utilities**: `src/arcaneum/cli/output.py`

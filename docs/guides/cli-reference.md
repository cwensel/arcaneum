# Arc CLI Reference

The `arc` command-line tool provides all operations for semantic and full-text search with Qdrant and MeiliSearch.

## Installation & Setup

### Development Mode

```bash
# From repository root
bin/arc <command> [options]
```

### Installed Mode

```bash
# After pip install -e .
arc <command> [options]
```

## Command Overview

### Which Commands to Use

**For general use, always prefer `arc corpus` commands.** They manage a paired
Qdrant collection and MeiliSearch index together, which is the intended
workflow for almost all cases.

- **Use Corpus** (`arc corpus ...`): Both semantic (conceptual) and full-text
  (exact phrase) search. Recommended for all new work.
- **Use Collection** (`arc collection ...` / `arc index ...`): Qdrant-only,
  semantic search only. Advanced — use only if you explicitly don't want
  MeiliSearch.
- **Use Indexes** (`arc indexes ...`): MeiliSearch-only, full-text search only.
  Advanced — use only if you explicitly don't want Qdrant.

### Corpus Management (Recommended)

A "corpus" is a paired Qdrant collection and MeiliSearch index with the same name,
providing both semantic and full-text search capabilities.

```bash
arc corpus create <name> --type <type>        # Create both collection and index
arc corpus list                               # List corpora with cached item counts and descriptions
arc corpus list --details                     # Add model, last sync, chunk counts, and exact item counts
arc corpus update <name>                      # Update description metadata
arc corpus sync <name> <path> [<path>...]     # Index to both systems
arc corpus repair <name>                      # Re-index incomplete/garbled files
arc corpus items <name>                       # List items with parity status
arc corpus verify <name>                      # Verify corpus health
arc corpus parity <name>                      # Check/restore parity
arc corpus delete <name>                      # Delete both collection and index
```

### Search Commands

```bash
arc search semantic <query> --corpus <name>                 # Semantic search (Qdrant)
arc search semantic <query> --corpus <n1> --corpus <n2>     # Multi-corpus search
arc search text <query> --corpus <name>                     # Full-text search (MeiliSearch)
```

### Log Commands

```bash
arc log tail               # Tail the current interaction log
arc log tail --lines 50    # Print recent log lines, then follow new entries
```

### Collection Management (Qdrant Only)

Use collections when you only need semantic search:

```bash
arc collection create <name> --type <type>    # Create Qdrant collection
arc collection list                           # List all collections
arc collection info <name>                    # Show collection details
arc collection items <name>                   # List indexed files/repos
arc collection verify <name>                  # Verify collection integrity
arc collection export <name> -o <file>        # Export to portable format
arc collection import <file>                  # Import from export file
arc collection delete <name>                  # Delete collection
```

### Index Management (MeiliSearch Only)

Use indexes when you only need full-text search:

```bash
arc indexes create <name> --type <type>              # Create MeiliSearch index
arc indexes list                                     # List all indexes
arc indexes items <name>                             # List indexed documents
arc indexes info <name>                              # Show index details
arc indexes verify <name>                            # Verify index health
arc indexes export <name> -o <file>                  # Export documents to JSONL
arc indexes import <file> --into <name>              # Import documents from JSONL
arc indexes list-projects <name>                     # List git projects in an index
arc indexes delete-project <name> <project>          # Delete one git project/branch
arc indexes delete <name>                            # Delete index
arc indexes update-settings <name> --type <type>    # Update index settings
```

### Indexing Commands (Single-System, Advanced)

For direct indexing to a single system. **Prefer `arc corpus sync` for normal
use** — these commands exist for workflows that deliberately need only one
system.

```bash
arc index pdf <path> --collection <name>      # Index PDFs to Qdrant (semantic)
arc index code <path> --collection <name>     # Index source code to Qdrant
arc index markdown <path> --collection <name> # Index Markdown to Qdrant
arc index text pdf <path> --index <name>      # Index PDFs to MeiliSearch (full-text)
arc index text code <path> --index <name>     # Index code to MeiliSearch
arc index text markdown <path> --index <name> # Index Markdown to MeiliSearch
```

### Corpus Commands (Detailed)

A "corpus" is a paired Qdrant collection and MeiliSearch index with the same name.

```bash
arc corpus create <name> --type <type> --models <model>  # Create both
arc corpus update <name> --description <text>            # Update metadata
arc corpus delete <name>                                 # Delete both
arc corpus sync <name> <path> [<path>...]                # Index to both (multiple paths supported)
arc corpus repair <name>                                 # Re-index incomplete/garbled files
arc corpus info <name>                                   # Show corpus details
arc corpus items <name>                                  # List indexed items with parity status
arc corpus verify <name>                                 # Verify corpus health across both systems
arc corpus parity <name>                                 # Restore parity between indexes
```

**Note:** If you already have a collection and index with the same name, you can
use `corpus sync` directly - no need to run `corpus create` first.

### Corpus Sync Options

```bash
arc corpus sync MyCorpus /path/to/files                    # Basic sync (file-level change detection)
arc corpus sync MyCorpus /path/one /path/two               # Multiple paths
arc corpus sync MyCorpus /path --parity                    # Detect renames, remove files no longer on disk
arc corpus sync MyCorpus /path --parity --dry-run          # Preview rename/remove changes
arc corpus sync MyCorpus /path --force                     # Force reindex all
arc corpus sync MyCorpus /path --verify                    # Verify after sync
arc corpus sync MyCorpus /path --gpu                       # Opt into accelerator embedding
arc corpus sync MyCorpus /path --models arctic-m           # Use specific model
arc corpus sync MyCorpus /path --max-embedding-batch 8     # Limit batch size (OOM recovery)
arc corpus sync MyCorpus /path --verbose                   # Detailed progress
arc corpus sync MyCorpus --from-file paths.txt             # Read paths from file (or "-" for stdin)
```

**Git-Aware Sync Options (for code corpora):**

```bash
arc corpus sync MyCorpus /path/to/repo --git-update        # Skip repos whose HEAD commit is unchanged
arc corpus sync MyCorpus /path/to/repo --git-version       # Keep multiple versions indexed
```

**Options:**

- `--parity`: Check cross-system parity between Qdrant and MeiliSearch, detect
  renamed/moved files by content hash, and **remove indexed entries for files
  that no longer exist on disk** (scoped to the directories being synced).
  Slower — scrolls the full index. For standalone cross-system repair without
  indexing a directory, use [`arc corpus parity`](#corpus-parity) instead.
- `--dry-run`: Show what would be synced, renamed, or removed without making
  changes. Pairs well with `--parity` to preview cleanup.
- `--force`: Force reindex all files (deletes existing chunks first)
- `--verify`: Verify collection integrity after indexing
- `--gpu`: Opt into accelerator embedding. CPU is the stable default.
- `--cpu-workers`: Batch parallelization workers for CPU mode (default: 1)
- `--models`: Embedding models to use (comma-separated; default comes from corpus metadata)
- `--file-types`: Comma-separated extensions to index (e.g., `.py,.md`)
- `--from-file`: Read paths from a file (one per line), or `-` for stdin
- `--max-embedding-batch`: Cap embedding batch size (use 8-16 for OOM recovery)
- `--text-workers`: Parallel workers for code AST chunking (default: auto)
- `--skip-dir-prefix`: Skip directories starting with PREFIX (default: `_`, repeatable)
- `--no-skip-dir-prefix`: Disable all directory prefix skipping
- `--timeout`: Qdrant timeout in seconds (default: 120, increase for very large files)
- `--verbose`: Show detailed progress (files, chunks, indexing)
- `--json`: Output JSON format for scripting
- `--git-update`: Commit-hash fast path; skip an entire git repo when its HEAD commit
  matches the last indexed commit
- `--git-version`: Keep multiple versions indexed (different commits coexist)
- `--mem-probe-interval`: Seconds between background memory snapshots (default: 0=off).
  Survives encode hangs that the per-file probe can't observe.
- `--mem-probe-log`: JSONL output path for memory snapshots (default:
  `~/.arcaneum/logs/arc-mem-<utc>-<pid>.jsonl`; pass `-` for stderr).

**Git Sync Modes:**

| Mode           | Behavior                                      | Use Case                     |
| -------------- | --------------------------------------------- | ---------------------------- |
| Default        | Discover git repos with `git ls-files`; check changed files by path, mtime, and size | General use, local worktrees |
| `--git-update` | Before file checks, skip whole repos whose HEAD commit is unchanged | CI/batch re-indexing committed snapshots |
| `--git-version`| Index each commit as separate version         | Compare code across versions |

**Note:** Default code sync is already git-aware. `--git-update` is not needed to
recognize repos; it is an opt-in commit-hash skip and can skip uncommitted
working-tree changes when HEAD has not changed. `--git-update` and `--git-version`
are mutually exclusive. Use `--force` to override either mode.

### Corpus Repair

Detect and re-index files with incomplete chunks or garbled text extraction:

```bash
arc corpus repair MyCorpus                            # Detect and fix quality issues
arc corpus repair MyCorpus --dry-run                  # Preview what would be repaired
arc corpus repair MyCorpus --quality-threshold 0.5    # More aggressive detection
arc corpus repair MyCorpus --verbose                  # Show per-file quality scores
```

**Options:**

- `--quality-threshold`: Text quality score threshold (0.0-1.0, default: 0.9)
- `--dry-run`: Show what would be repaired without making changes
- `--gpu`: Opt into accelerator embedding. CPU is the stable default.
- `--max-embedding-batch`: Cap embedding batch size
- `--verbose`: Show per-file old → new quality scores
- `--json`: Output JSON format

**How it works:**

1. Scans all indexed chunks and scores text quality (stop word frequency, replacement characters, ASCII ratio)
2. Identifies files with incomplete chunk sets or average quality below threshold
3. Re-extracts affected files (with auto-OCR for garbled text from corrupt fonts)
4. Compares new quality to old — only replaces if improved
5. Reports per-file results (improved, skipped, incomplete)

**GPU and Apple Silicon:**

Large models like `stella` (1.5B params) may cause system instability on Macs with
limited unified memory when GPU is enabled. CPU is the stable default; use `--gpu`
only when accelerator throughput is worth the memory-risk tradeoff, or switch to a
smaller model like `bge` (0.3B params).

**Memory Diagnostics:**

The verbose output includes a `mem:` line per file with rss, MPS driver, and
system pressure deltas. That probe only fires *after* a file completes — if the
GPU hangs mid-encode (common on machines with high baseline memory pressure),
no telemetry is captured. The `--mem-probe-interval` flag starts a background
thread that writes JSONL snapshots on a fixed cadence and survives hostile
encode loops:

```bash
# Snapshot every second to ~/.arcaneum/logs/arc-mem-<utc>-<pid>.jsonl
arc corpus sync MyCorpus /path --mem-probe-interval 1 --verbose

# Or via env var (no command change)
ARC_MEM_PROBE_INTERVAL=1 arc corpus sync MyCorpus /path --verbose

# Custom path / stderr
arc corpus sync MyCorpus /path --mem-probe-interval 1 --mem-probe-log /tmp/mem.jsonl
arc corpus sync MyCorpus /path --mem-probe-interval 1 --mem-probe-log -
```

Each line is a JSON object with `ts`, `phase` (e.g. `encoding:foo.pdf:stella`,
`indexing:foo.pdf`), `rss`, `mps_current`, `mps_driver`, `sys_used_pct`, and
related fields. The output file is line-buffered so partial data survives a
SIGKILL from macOS jetsam.

After a hang or crash, inspect the log to find when pressure ramped:

```bash
jq -r '[.ts, .phase, .rss, .sys_used_pct] | @tsv' \
  ~/.arcaneum/logs/arc-mem-*.jsonl | tail -50
```

Mid-run, you can also send `kill -USR1 <pid>` to dump full memory state plus
thread stacks to stderr (useful when a single encode call has hung).

### Corpus Parity

Check and restore parity between Qdrant and MeiliSearch indexes without scanning
a directory. This is useful when one index gets out of sync with the other.

```bash
# Check parity status (dry-run, no changes)
arc corpus parity MyCorpus --dry-run

# Restore parity with verbose output
arc corpus parity MyCorpus --verbose

# JSON output for scripting
arc corpus parity MyCorpus --json
```

**How it works:**

- Compares indexed file paths in both Qdrant and MeiliSearch
- Backfills missing entries in each direction:
  - **Qdrant -> MeiliSearch**: Copies metadata from Qdrant (no file access needed)
  - **MeiliSearch -> Qdrant**: Re-chunks and embeds files (requires file access)
- Files that don't exist on disk are skipped with a warning

**Options:**

- `--dry-run`: Show what would be backfilled without making changes
- `--verify`: Verify chunk counts match between systems (detects partial uploads)
- `--repair-metadata`: Repair git metadata in MeiliSearch (code corpora only):
  - Backfills missing `git_project_identifier` from Qdrant
  - Computes and repairs `git_version_identifier` for version-aware search
- `--create-missing`: Create missing MeiliSearch indexes for qdrant_only corpora
- `--confirm`: Skip confirmation prompt when processing all corpora
- `--verbose`: Show detailed progress for each file
- `--json`: Output JSON format for scripting

**Example output:**

```text
Checking parity for corpus 'Papers'...
Corpus type: pdf, Models: stella

Index Status:
  Files in both systems:     150
  Files in Qdrant only:      5
  Files in MeiliSearch only: 3

Backfilling 5 files to MeiliSearch...
  ✓ document1.pdf: 12 chunks
  ✓ document2.pdf: 8 chunks

Backfilling 3 files to Qdrant...
  ⚠ report.pdf: File not found, skipping
  ✓ notes.pdf: 15 chunks

✅ Parity restored for corpus 'Papers'
   Backfilled to MeiliSearch: 5 files (47 chunks)
   Backfilled to Qdrant: 2 files (23 chunks)
   Skipped (not found): 1 file
```

**All-Corpora Mode:**

When run without a corpus name, parity discovers all corpora and processes them:

```bash
# Discover and process all corpora
arc corpus parity

# Preview all corpora status
arc corpus parity --dry-run

# Process all without confirmation prompt
arc corpus parity --confirm
```

**Creating Missing Indexes:**

The `--create-missing` flag promotes single-sided Qdrant collections into full corpora
by automatically creating missing MeiliSearch indexes:

```bash
# Preview what indexes would be created
arc corpus parity --create-missing --dry-run

# Create missing indexes and sync
arc corpus parity --create-missing

# Single corpus: create missing index if needed
arc corpus parity MyCorpus --create-missing
```

This is useful when you have Qdrant collections that were indexed before MeiliSearch
was set up, or when migrating to dual-index workflows.

**Note:** `meili_only` corpora cannot be auto-created because creating a Qdrant
collection requires specifying `--type` and `--model`. These must be created manually
with `arc corpus create`.

### Corpus Items

List all indexed items in a corpus with parity status between Qdrant and MeiliSearch:

```bash
# Human-readable table output
arc corpus items MyCorpus

# JSON output for automation
arc corpus items MyCorpus --json
```

**Features:**

- **Type-aware output**: Different displays for code vs PDF/markdown corpora
- **Parity status**: Shows sync status for each item (synced, mismatch, qdrant_only, meili_only)
- **Chunk counts**: Shows chunks in each system (Q and M columns)
- **Deduplication**: Groups chunks by file/repository

**Code Corpus Output:**

| Project | Branch  | Commit       | Q     | M     | Status   |
| ------- | ------- | ------------ | ----- | ----- | -------- |
| my-app  | main    | a1b2c3d4e5f6 | 1,532 | 1,532 | synced   |
| my-lib  | develop | f6e5d4c3b2a1 | 847   | 800   | mismatch |

**PDF/Markdown Corpus Output:**

| File               | Size  | Q  | M  | Status      |
| ------------------ | ----- | -- | -- | ----------- |
| research-paper.pdf | 2.3MB | 42 | 42 | synced      |
| documentation.md   | 15KB  | 8  | 0  | qdrant_only |

**Status Values:**

- `synced`: Same chunk count in both systems
- `mismatch`: Different chunk counts (may indicate partial upload)
- `qdrant_only`: Item exists only in Qdrant
- `meili_only`: Item exists only in MeiliSearch

### Corpus Verify

Verify corpus health across both Qdrant and MeiliSearch systems:

```bash
# Basic health check
arc corpus verify MyCorpus

# Detailed output with file-level results
arc corpus verify MyCorpus --verbose

# Filter by project (code corpora only)
arc corpus verify MyCode --project my-app

# JSON output for automation
arc corpus verify MyCorpus --json
```

**Features:**

- **Qdrant health check**: Verifies chunk completeness for all indexed items
- **MeiliSearch health check**: Verifies index accessibility and document retrieval
- **Parity status**: Reports whether both systems have the corpus
- **Detailed diagnostics**: Shows warnings for configuration issues

**Example Output:**

```text
Verifying Qdrant collection 'MyCorpus'...
Verifying MeiliSearch index 'MyCorpus'...

Corpus: MyCorpus
Overall Status: Healthy

Qdrant Collection:
  Status: Healthy
  Items: 150 (150 complete)

MeiliSearch Index:
  Status: Healthy
  Documents: 2,847
  Sample retrieval: OK

All checks passed
```

**Options:**

- `--verbose` / `-v`: Show detailed file-level verification results
- `--project`: Filter by project identifier (code corpora only)
- `--json`: Output JSON format for scripting

**JSON Output Format:**

```json
{
  "status": "success",
  "message": "Corpus 'MyCorpus' is healthy",
  "data": {
    "corpus": "MyCorpus",
    "overall_healthy": true,
    "parity_status": "needs_review",
    "qdrant": {
      "collection": "MyCorpus",
      "type": "pdf",
      "is_healthy": true,
      "total_points": 2847,
      "total_items": 150,
      "complete_items": 150,
      "incomplete_items": 0
    },
    "meilisearch": {
      "is_healthy": true,
      "document_count": 2847,
      "is_indexing": false,
      "sample_accessible": true,
      "issues": [],
      "warnings": []
    }
  }
}
```

### Corpus Delete

Delete both the Qdrant collection and MeiliSearch index for a corpus:

```bash
# With confirmation prompt
arc corpus delete MyCorpus

# Skip confirmation (for scripts)
arc corpus delete MyCorpus --confirm

# JSON output
arc corpus delete MyCorpus --confirm --json
```

**Behavior:**

- Checks if either Qdrant collection or MeiliSearch index exists
- Prompts for confirmation before deleting (unless `--confirm` is passed)
- Deletes both systems; if one fails, continues with the other
- Reports partial deletion if only one system was deleted

## Collection Management Examples

### Create Collection

```bash
# Create collection with type (model inferred automatically)
arc collection create pdf-docs --type pdf

# Create with specific model (optional override)
arc collection create pdf-docs --type pdf --model stella

# With custom HNSW parameters
arc collection create pdf-docs --type pdf --hnsw-m 16 --hnsw-ef 100

# Store vectors on disk (for large collections)
arc collection create pdf-docs --type pdf --on-disk
```

**Model Inference:** If `--model` is not specified, the model is automatically inferred from `--type`:

- `--type pdf` → `arctic-m` (stable FastEmbed document retrieval)
- `--type code` → `jina-code` (stable FastEmbed code retrieval)
- `--type markdown` → `arctic-m` (stable FastEmbed document retrieval)

### List Collections

```bash
# Simple list (shows name, model, and point count)
arc collection list

# Verbose output (adds collection type and vector details)
arc collection list --verbose

# JSON output for scripting
arc collection list --json
```

**Output includes:**

- **Name**: Collection name
- **Model**: Embedding model used (e.g., stella, jina-code)
- **Points**: Number of indexed documents
- **Type** (verbose): Collection type (pdf, code, markdown)
- **Vectors** (verbose): Vector dimensions and distance metrics

### Collection Info

```bash
# Show collection details (including model and type)
arc collection info pdf-docs

# JSON output
arc collection info pdf-docs --json
```

Shows detailed information including collection type, model, point count, vector configuration, and HNSW index parameters.

### Delete Collection

```bash
# With confirmation prompt
arc collection delete pdf-docs

# Skip confirmation
arc collection delete pdf-docs --confirm
```

### List Collection Items

List all indexed files or repositories in a collection:

```bash
# Human-readable table output
arc collection items MyCode

# JSON output for automation
arc collection items MyCode --json
```

**Features:**

- **Type-aware output**: Different displays for code vs PDF/markdown collections
- **Metadata included**: File sizes, chunk counts, git information
- **Deduplication**: Shows unique files/repos (not individual chunks)
- **Efficient**: Batched retrieval for large collections

**Code Collections Output:**

Shows repositories with git metadata:

| Project | Branch  | Commit       | Chunks |
| ------- | ------- | ------------ | ------ |
| my-app  | main    | a1b2c3d4e5f6 | 1,532  |
| my-lib  | develop | f6e5d4c3b2a1 | 847    |

**PDF/Markdown Collections Output:**

Shows files with size information:

| File               | Size   | Chunks |
| ------------------ | ------ | ------ |
| research-paper.pdf | 2.3MB  | 42     |
| documentation.md   | 15.2KB | 8      |

**JSON Output Format:**

```json
{
  "status": "success",
  "message": "Found 2 items in collection 'MyCode'",
  "data": {
    "collection": "MyCode",
    "type": "code",
    "item_count": 2,
    "items": [
      {
        "git_project_name": "my-app",
        "git_project_identifier": "/path/to/my-app",
        "git_branch": "main",
        "git_commit_hash": "a1b2c3d4e5f6",
        "git_remote_url": "https://github.com/user/my-app",
        "chunk_count": 1532
      }
    ]
  }
}
```

**Use Cases:**

- Verify indexing completed successfully
- Audit collection contents
- Check which branches are indexed
- Count total chunks per file/repo
- Export collection metadata for reporting

### Export Collection

Export a collection to a portable format for migration or backup:

```bash
# Default: Compressed binary format (.arcexp)
arc collection export MyPDFs -o backup.arcexp

# Human-readable JSONL format (for debugging)
arc collection export MyPDFs -o backup.jsonl --format jsonl

# Filter by file path patterns
arc collection export MyPDFs -o reports.arcexp --include "*/reports/*.pdf"
arc collection export MyPDFs -o subset.arcexp --exclude "*/drafts/*"

# Filter code collections by repo
arc collection export MyCode -o arcaneum.arcexp --repo arcaneum
arc collection export MyCode -o main-only.arcexp --repo arcaneum#main

# Combined filters
arc collection export MyCode -o subset.arcexp \
    --include "*/src/*" \
    --repo arcaneum#main \
    --exclude "*/test/*"

# Detached export (strips root prefix for shareable archives)
arc collection export MyCode -o shareable.arcexp --detach

# JSON output for scripting
arc collection export MyPDFs -o backup.arcexp --json
```

**Filter Options:**

| Option               | Description                                  | Example                |
| -------------------- | -------------------------------------------- | ---------------------- |
| `--include`          | Include files matching glob (multiple = OR)  | `--include "*.pdf"`    |
| `--exclude`          | Exclude files matching glob (multiple = AND) | `--exclude "*/temp/*"` |
| `--repo`             | Filter by repo name (code collections)       | `--repo arcaneum`      |
| `--repo name#branch` | Filter by repo and branch                    | `--repo arcaneum#main` |
| `--detach`           | Strip root prefix, store relative paths      | `--detach`             |

**Export Formats:**

| Format | Extension | Size         | Use Case                    |
| ------ | --------- | ------------ | --------------------------- |
| Binary | `.arcexp` | ~10x smaller | Migration, backup (default) |
| JSONL  | `.jsonl`  | Larger       | Debugging, inspection       |

### Import Collection

Import a collection from an export file:

```bash
# Import to original collection name
arc collection import backup.arcexp

# Import to different collection name
arc collection import backup.arcexp --into MyPDFs-restored

# Import detached export with new root path
arc collection import shareable.arcexp --attach /home/bob/projects

# Remap paths for cross-machine migration
arc collection import backup.arcexp --remap /Users/alice/docs:/home/bob/docs

# Multiple path remappings
arc collection import backup.arcexp \
    --remap /Users/alice/repos:/home/bob/repos \
    --remap /Users/alice/docs:/home/bob/documents

# JSON output for scripting
arc collection import backup.arcexp --json
```

**Path Handling Options:**

| Option            | Description                    | Use Case                 |
| ----------------- | ------------------------------ | ------------------------ |
| `--into`          | Target collection name         | Import to different name |
| `--attach`        | Prepend root to relative paths | For detached exports     |
| `--remap old:new` | Substitute path prefixes       | Cross-machine migration  |

**Format Auto-Detection:**

Import automatically detects file format (binary or JSONL) from file content.

**Cross-Machine Migration Workflow:**

```bash
# On source machine: Export with detach
arc collection export MyCode -o shareable.arcexp --detach
# Transfer shareable.arcexp to new machine...

# On target machine: Import with attach
arc collection import shareable.arcexp --attach /home/newuser/projects
```

**Path Remapping Workflow (non-detached):**

```bash
# On source machine: Export normally
arc collection export MyDocs -o backup.arcexp
# Transfer backup.arcexp to new machine...

# On target machine: Import with path remapping
arc collection import backup.arcexp \
    --remap /Users/alice:/home/bob \
    --into MyDocs-migrated
```

## PDF Indexing Examples

### Basic Usage

```bash
# CPU-first stable default
# Model is automatically retrieved from collection metadata
arc index pdf /path/to/pdfs --collection pdf-docs
```

### With OCR

```bash
arc index pdf /path/to/scanned-pdfs \
  --collection pdf-docs \
  --ocr-language eng
```

### Force Reindex

```bash
arc index pdf /path/to/pdfs \
  --collection pdf-docs \
  --force
```

### Performance Tuning

**Maximum throughput:**

```bash
arc index pdf /path/to/pdfs \
  --collection pdf-docs \
  --embedding-batch-size 500 \
  --process-priority low
```

Note: Larger embedding batches (300-500) improve throughput 10-20%. Process priority is for background indexing.

### GPU Control

```bash
# Default: CPU embedding for stable unattended indexing
arc index pdf /path/to/pdfs --collection pdf-docs

# Opt into accelerator embedding (MPS on Apple Silicon, CUDA on NVIDIA)
arc index pdf /path/to/pdfs --collection pdf-docs --gpu
```

**Note:** The `--model` flag is deprecated. Models are now set at collection
creation time with `arc collection create --type pdf`.

### Debug Mode

```bash
# Show all library warnings (including HuggingFace transformers)
arc index pdf /path/to/pdfs --collection pdf-docs --debug
```

## PDF Full-Text Indexing (MeiliSearch)

Index PDFs to MeiliSearch for exact phrase and keyword search, complementing
semantic search in Qdrant. This mirrors the `arc collection` commands for Qdrant
with `arc indexes` commands for MeiliSearch.

### Basic Usage

```bash
# Create MeiliSearch index first (mirrors arc collection create)
arc indexes create pdf-docs --type pdf

# Index PDFs to MeiliSearch
arc index text pdf /path/to/pdfs --index pdf-docs
```

### Command Options

```bash
arc index text pdf <directory> --index <name> [options]
```

**Options:**

- `--index`: Target MeiliSearch index name (required)
- `--recursive / --no-recursive`: Search subdirectories (default: recursive)
- `--force`: Force reindex all files (skip change detection)
- `--ocr / --no-ocr`: Enable/disable OCR for scanned PDFs (default: enabled)
- `--ocr-language`: OCR language code (default: eng)
- `--batch-size`: Documents per batch upload (default: 1000)
- `--verbose`: Show detailed progress
- `--json`: JSON output for scripting

### Examples

```bash
# Index with OCR for scanned documents
arc index text pdf ./scanned-docs --index pdf-docs --ocr-language eng

# Force reindex all
arc index text pdf ./pdfs --index pdf-docs --force

# Disable OCR (text-only PDFs)
arc index text pdf ./text-pdfs --index pdf-docs --no-ocr

# JSON output
arc index text pdf ./pdfs --index pdf-docs --json
```

### Dual Indexing Workflow (Recommended: Corpus)

For comprehensive search, use corpus commands to index to both Qdrant and MeiliSearch:

```bash
# Create corpus (creates both collection and index)
arc corpus create pdf-docs --type pdf

# Sync to both systems
arc corpus sync pdf-docs /path/to/pdfs

# Semantic search (conceptual matches)
arc search semantic "machine learning" --corpus pdf-docs

# Full-text search (exact phrases)
arc search text '"neural network"' --corpus pdf-docs
```

### Dual Indexing (Manual - Advanced)

Alternatively, manage Qdrant and MeiliSearch separately:

```bash
# Create both collection and index (mirrored commands)
arc collection create pdf-docs --type pdf      # Qdrant
arc indexes create pdf-docs --type pdf         # MeiliSearch

# Index to Qdrant (semantic search)
arc index pdf /path/to/pdfs --collection pdf-docs

# Index to MeiliSearch (full-text search)
arc index text pdf /path/to/pdfs --index pdf-docs

# Search (both use --corpus flag)
arc search semantic "machine learning" --corpus pdf-docs
arc search text '"neural network"' --corpus pdf-docs
```

## Model Selection

### General Purpose Models

| Model        | Dimensions | Best For                        | Late Chunking |
| ------------ | ---------- | ------------------------------- | ------------- |
| `arctic-m`   | 768D       | Stable docs/PDFs/markdown       | No            |
| `stella`     | 1024D      | High-quality documents          | Yes           |
| `mxbai-large`| 1024D      | High-quality FastEmbed docs     | No            |
| `bge`        | 1024D      | Legacy BGE documents            | No            |

### Code-Specific Models

**Stable default: `jina-code`**

For source code indexing, use specialized code models optimized for programming languages:

| Model            | Dimensions | Context | Best For              | Notes                                    |
| ---------------- | ---------- | ------- | --------------------- | ---------------------------------------- |
| `jina-code`      | 768D       | 8K      | Stable default        | Lower memory, good code retrieval        |
| `jina-code-0.5b` | 896D       | 32K     | Higher quality        | SOTA Sept 2025, opt-in                   |
| `jina-code-1.5b` | 1536D      | 32K     | Highest quality       | SOTA Sept 2025, slower but best accuracy |
| `codesage-large` | 1024D      | -       | 9 languages           | CodeSage V2, Dec 2024                    |
| `nomic-code`     | 3584D      | -       | 6 languages           | 7B params, highest quality, slowest      |

**Recommendation:** Use `jina-code` for the most stable default. Use `jina-code-0.5b`
or `jina-code-1.5b` when quality is more important than memory footprint.

**Usage:**

```bash
# Create code collection (uses jina-code by default)
arc collection create MyCode --type code

# Or explicitly specify a higher-quality model
arc collection create MyCode --type code --model jina-code-0.5b
```

## Common Workflows

### Setup New Project (Recommended: Corpus)

```bash
# 1. Start services
arc container start

# 2. Create corpus (indexes to both Qdrant and MeiliSearch)
arc corpus create my-docs --type pdf

# 3. Sync documents
arc corpus sync my-docs ./documents

# 4. Search with semantic or full-text
arc search semantic "query" --corpus my-docs
arc search text "exact phrase" --corpus my-docs
```

### Setup New Project (Single System)

```bash
# 1. Start services
arc container start

# 2. Create collection (Qdrant only)
arc collection create my-docs --type pdf

# 3. Index documents
arc index pdf ./documents --collection my-docs
```

### Incremental Updates

```bash
# First run: indexes all PDFs
arc corpus sync my-docs ./docs

# Add new files to ./docs/...

# Second run: only indexes new/modified files
arc corpus sync my-docs ./docs
```

### JSON Output for Automation

```bash
# List collections
arc collection list --json | jq '.collections[].name'

# Index with JSON output
arc index pdf ./docs --collection my-docs --json > results.json

# Check results
jq '.stats.chunks' results.json
```

## Global Options

Most commands support:

- `--json`: Output JSON format (for scripting)
- `--verbose` / `-v`: Verbose output (show progress and stats, suppress library warnings)
- `--debug`: Debug mode (show all library warnings including transformers)
- `--help`: Show command help

### Indexing Options

Additional options for `arc index` commands:

**Basic Options:**

- `--gpu`: Opt into accelerator embedding (CPU is the stable default)
- `--workers N`: Number of parallel upload workers (default: 4)
- `--force`: Force reindex all files (skip incremental sync)
- `--offline`: Use cached models only (no network calls)
- `--streaming`: Stream embeddings to Qdrant immediately (lower memory usage)

**Performance Tuning:**

- `--embedding-batch-size N`: Batch size for embedding generation [default: 200]
  - Larger batches (300-500) improve throughput 10-20%
  - Limited benefit from thread parallelism due to embedding lock (see Architecture Notes below)
- `--process-priority low|normal|high`: Process scheduling priority [default: normal]
  - Use `low` for background indexing to avoid blocking foreground tasks

**Memory Optimization:**

- `--streaming`: Upload embeddings immediately after each batch instead of accumulating all in memory
  - Reduces memory from O(total_chunks × vector_dim) to O(batch_size × vector_dim)
  - Recommended for large files or collections
  - Partial uploads can be recovered with `--verify` flag

**Note on Parallelism:** File and embedding worker flags were removed because they provided minimal benefit
due to the embedding lock (required for GPU thread-safety). The single-threaded embedding approach with
larger batches is actually more efficient. Use `--embedding-batch-size` for throughput tuning.

### GPU Acceleration

CPU embedding is the default for stable unattended indexing. GPU acceleration is
available with `--gpu`:

- **Apple Silicon**: Uses MPS (Metal Performance Shaders) backend
- **NVIDIA GPUs**: Uses CUDA backend
- **FastEmbed/CoreML**: Experimental on Apple Silicon; set `ARC_EXPERIMENTAL_COREML=1`

**Performance**: 1.5-3x speedup with GPU for supported embedding models.

**Compatible Models** (verified with GPU):

- `stella` - MPS support for high-quality PDFs/markdown
- `jina-code` - MPS support for source code
- `bge-small` - experimental CoreML support
- `bge-base` - experimental CoreML support

**Enable GPU** when speed is more important than maximum stability:

```bash
arc index pdf /path/to/pdfs --collection docs --gpu
```

## Exit Codes

- `0`: Success
- `1`: Error (with error message to stderr)

## Environment Variables

Configure via environment or `.env` file:

```bash
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-qdrant-api-key  # Optional: for secured/hosted Qdrant
MEILISEARCH_URL=http://localhost:7700
MEILISEARCH_API_KEY=your-api-key  # Optional: auto-generated if not set
```

Arcaneum also accepts `ARC_QDRANT_URL` and `ARC_QDRANT_API_KEY`; these take
precedence over the unprefixed Qdrant variables when both are set. Qdrant
credentials can also be stored in `~/.config/arcaneum/config.yaml` under
`qdrant.api_key`. On first default Qdrant config load, a legacy
`~/.arcaneum/config.yaml` is copied to the XDG config path if the XDG file does
not already exist.

Note: `MEILISEARCH_API_KEY` is auto-generated on first `arc container start` and stored
in `~/.config/arcaneum/meilisearch.key`. You only need to set it manually if you want
to use a custom key or share the same key across machines.

## For Claude Code Agents

The `arc` CLI is the entrypoint for all Claude Code plugins and slash commands:

```bash
# These commands are available in Claude Code via slash commands
/arc:corpus create my-docs --type pdf
/arc:corpus sync my-docs ./documents
/arc:search semantic "machine learning" --corpus my-docs
/arc:search text "exact phrase" --corpus my-docs
```

See individual slash command files in `/commands/` directory for detailed usage.

## Full-Text Search (MeiliSearch)

Full-text search provides exact phrase matching, typo-tolerant keyword search, and filtered queries.

### Index Management Commands

MeiliSearch index commands mirror Qdrant collection commands:

| Qdrant (arc collection)   | MeiliSearch (arc indexes) |
| ------------------------- | ------------------------- |
| `arc collection create`   | `arc indexes create`      |
| `arc collection list`     | `arc indexes list`        |
| `arc collection info`     | `arc indexes info`        |
| `arc collection items`    | `arc indexes items`       |
| `arc collection verify`   | `arc indexes verify`      |
| `arc collection export`   | `arc indexes export`      |
| `arc collection import`   | `arc indexes import`      |
| `arc collection delete`   | `arc indexes delete`      |

### Create Index

```bash
# Create index with type-specific settings
arc indexes create source-code --type source-code
arc indexes create pdf-docs --type pdf
arc indexes create my-docs --type markdown

# JSON output
arc indexes create my-docs --type pdf --json
```

**Index Types:**

| Type            | Aliases    | Optimized For                    |
| --------------- | ---------- | -------------------------------- |
| `source-code`   | `code`     | Code with higher typo thresholds |
| `pdf-docs`      | `pdf`      | PDF documents with stop words    |
| `markdown-docs` | `markdown` | Markdown with headings search    |

### List Indexes

```bash
# List all indexes with document counts
arc indexes list

# JSON output for scripting
arc indexes list --json
```

**Output includes:**

- **Name**: Index name (uid)
- **Primary Key**: Document primary key field
- **Documents**: Number of indexed documents
- **Created**: Creation date

### Index Info

```bash
# Show detailed index information
arc indexes info source-code

# JSON output
arc indexes info source-code --json
```

Shows detailed information including:

- Document count and indexing status
- Searchable attributes configuration
- Filterable attributes for query filters
- Sortable attributes
- Typo tolerance settings

### Index Items

List all indexed files/documents in an index with chunk counts:

```bash
# Human-readable table output
arc indexes items MyIndex

# Limit number of items shown
arc indexes items MyIndex --limit 50

# Paginate through results
arc indexes items MyIndex --limit 50 --offset 100

# JSON output for automation
arc indexes items MyIndex --json
```

**Output includes:**

| File         | Language | Project | Chunks |
| ------------ | -------- | ------- | ------ |
| auth.py      | python   | myapp   | 12     |
| handlers.go  | go       | myapp   | 8      |
| document.pdf | -        | -       | 45     |

### Verify Index

Verify index health and integrity:

```bash
# Basic health check
arc indexes verify MyIndex

# JSON output for automation
arc indexes verify MyIndex --json
```

**Checks performed:**

- Index accessibility and document retrieval
- Searchable attributes configuration
- Filterable attributes availability
- Current indexing status

**Example output:**

```text
Index: MyIndex

Status: Healthy
Documents: 2,847
Searchable: 5 attributes
Filterable: 8 attributes
Sample retrieval: OK

All checks passed
```

### Export Index

Export index documents to JSONL file for backup or migration:

```bash
# Export to JSONL file
arc indexes export MyIndex -o backup.jsonl

# JSON output showing export stats
arc indexes export MyIndex -o backup.jsonl --json
```

**Export file format:**

- First line: Index metadata (name, primary key, settings)
- Subsequent lines: One document per line as JSON

### Import Index

Import documents from a previously exported JSONL file:

```bash
# Import to original index name
arc indexes import backup.jsonl

# Import to different index name
arc indexes import backup.jsonl --into MyIndex-restored

# JSON output
arc indexes import backup.jsonl --json
```

**Behavior:**

- Creates the index if it doesn't exist (using exported settings)
- Adds documents in batches for efficiency
- Preserves original index configuration

### Update Settings

Update index settings from a preset type:

```bash
# Apply source-code settings to existing index
arc indexes update-settings MyIndex --type source-code

# Apply PDF settings
arc indexes update-settings MyIndex --type pdf

# JSON output
arc indexes update-settings MyIndex --type code --json
```

Use this to reconfigure an existing index with optimized settings for a different content type.

### Delete Index

```bash
# With confirmation prompt
arc indexes delete source-code

# Skip confirmation (for scripts)
arc indexes delete source-code --confirm

# JSON output
arc indexes delete source-code --confirm --json
```

### Git Project Management (Code Indexes)

For indexes containing git-aware source code, additional commands are available:

#### List Projects

```bash
# Show all indexed git projects
arc indexes list-projects MyCode

# JSON output
arc indexes list-projects MyCode --json
```

**Example output:**

| Project Identifier | Commit Hash      |
| ------------------ | ---------------- |
| arcaneum#main      | a1b2c3d4e5f6...  |
| mylib#develop      | f6e5d4c3b2a1...  |

#### Delete Project

Remove all documents for a specific git project/branch:

```bash
# Delete with confirmation
arc indexes delete-project arcaneum#main --index MyCode

# Skip confirmation
arc indexes delete-project myrepo#feature-x --index MyCode --confirm

# JSON output
arc indexes delete-project arcaneum#main --index MyCode --json
```

Other projects/branches in the same index are unaffected.

### Search

```bash
# Basic search
arc search text "authentication" --corpus source-code

# Exact phrase search (use quotes)
arc search text '"def authenticate"' --corpus source-code

# Multi-corpus search
arc search text "authentication" --corpus code1 --corpus code2

# With filter
arc search text "authentication" --corpus source-code --filter "language = python"

# With pagination
arc search text "query" --corpus source-code --limit 20 --offset 10

# JSON output
arc search text "query" --corpus source-code --json

# Verbose output (shows language, project metadata)
arc search text "query" --corpus source-code --verbose
```

**Filter Syntax:**

```bash
# Single filter
--filter "language = python"

# Multiple conditions
--filter "language = python AND project = myapp"

# Numeric comparison
--filter "page_number > 10"
```

### Configuration

MeiliSearch API key is auto-generated on first `arc container start` and stored in:

```text
~/.config/arcaneum/meilisearch.key
```

To override, set the environment variable:

```bash
export MEILISEARCH_API_KEY=your-custom-key
```

## Service Management

### Container Commands

Manage Qdrant and MeiliSearch container services:

```bash
# Start services
arc container start

# Stop services
arc container stop

# Check status
arc container status

# Back up indexed data
arc container backup

# Restore indexed data
arc container restore BACKUP_DIRECTORY

# View logs
arc container logs

# Follow logs in real-time
arc container logs --follow

# Restart services
arc container restart

# Reset all data (WARNING: deletes all collections)
arc container reset --confirm
```

**Examples:**

```bash
# Start Qdrant before indexing
arc container start

# Check if healthy
arc container status

# View recent logs
arc container logs --tail 50

# Follow logs for debugging
arc container logs --follow

# Restart if having issues
arc container restart

# Back up Qdrant snapshots and MeiliSearch indexes
arc container backup -o ./arcaneum-backup

# Restore from a backup directory
arc container restore ./arcaneum-backup

# Nuclear option: delete everything and start fresh
arc container reset --confirm
```

**Data Location:**

- Qdrant and MeiliSearch use Docker named volumes for persistence
- Survives container restarts
- `arc container backup` protects Qdrant collection snapshots and MeiliSearch
  index settings plus JSONL document exports, including Arcaneum corpus metadata
  stored there
- Run backups when indexing is idle; the command checks MeiliSearch for active
  tasks before and after export and aborts if any MeiliSearch task appears
  during the backup window
- `arc container restore` recreates same-named MeiliSearch indexes from the
  backup
- Backups do not include source files, cached embedding models, Docker images, or
  local configuration secrets

## Configuration & Cache Management

### Cache Commands

Manage embedding model cache:

```bash
# Show cache location and sizes
arc config show-cache-dir

# Clear model cache to free space
arc config clear-cache --confirm
```

**Examples:**

```bash
# Check where models are stored
arc config show-cache-dir
# Output:
#   Arcaneum directories:
#     Cache:  /Users/you/.cache/arcaneum
#     Data:   /Users/you/.local/share/arcaneum
#     Config: /Users/you/.config/arcaneum
#     Models size: 2.5 GB
#     Data size: 266.8 MB

# Clear cache if running low on disk space
arc config clear-cache --confirm
```

**Cache Location:**

- Models stored in `~/.cache/arcaneum/models/`
- Auto-downloaded on first use
- Shared across all arc commands
- ~1-2GB per model

## Troubleshooting

### Command Not Found

```bash
# Development mode
bin/arc --help

# After install
pip install -e .
arc --help
```

### Qdrant Connection Error

```bash
# Check if running
arc container status

# Start if needed
arc container start
```

### Permission Denied

```bash
chmod +x bin/arc
```

### GPU Memory Errors (MPS/CUDA)

Large embedding models can exceed GPU memory, especially on Apple Silicon (MPS):

```text
RuntimeError: MPS backend out of memory (MPS allocated: 12.25 GiB...)
```

**Solutions:**

1. The system uses adaptive batch sizes based on model size (automatic)
2. Omit `--gpu` to stay on the stable CPU default
3. When using `--gpu`, use `--embedding-batch-size 100` to reduce GPU memory usage
4. Try a smaller model (e.g., `minilm` instead of `stella`)

See [PDF Indexing Guide](pdf-indexing.md#gpu-memory-errors-mpscuda) for model memory requirements.

## More Documentation

- [PDF Indexing Guide](pdf-indexing.md) - Detailed PDF indexing documentation
- [RDR Directory](../docs/rdr/) - Technical specifications for all features

# Recommendation 017: Collection Export and Import for Cross-Machine Migration

## Metadata

- **Date**: 2026-01-07
- **Status**: Recommendation
- **Type**: Feature
- **Priority**: High
- **Related Issues**: None
- **Related Tests**: TBD

## Problem Statement

Users need to migrate Qdrant collections between Arcaneum installations on different
machines. The current snapshot-based backup/restore mechanism
(`scripts/qdrant-backup.sh`, `scripts/qdrant-restore.sh`) works but has limitations:

1. **Docker-dependent**: Scripts rely on `docker cp` to extract/inject snapshots
2. **All-or-nothing**: Cannot selectively export subsets (e.g., specific projects)
3. **No CLI integration**: Requires running shell scripts outside `arc` command hierarchy
4. **Qdrant-version sensitive**: Snapshots may not restore across major versions

A portable export format with CLI integration would enable easier cross-machine
transfer and selective export capabilities.

## Context

### Background

Arcaneum indexes content (PDFs, source code, markdown) into Qdrant vector
collections for semantic search. Each collection contains:

- **Vectors**: 768-1024 dimensional embeddings from models like stella, bge, jina-code
- **Payloads**: Metadata including file paths, content hashes, git info, chunk indices
- **Collection metadata**: Reserved point storing collection type, model, creation time

Users with indexed collections on one machine may want to:

- Transfer collections to a new workstation
- Share collections with team members
- Back up collections in a portable format
- Debug or inspect collection contents (rare)

### Technical Environment

- **Qdrant Client**: qdrant-client Python library
- **Vector Storage**: Named vectors (e.g., `{"stella": [0.041, 0.030, ...]}`)
- **Point IDs**: Mix of UUIDs and integers depending on collection type
- **Collection Types**: pdf, code, markdown (enforced via metadata point)
- **Typical Sizes**: 10,000 - 500,000+ points per collection

### Current Capabilities

| Operation  | Method                       | Limitations                        |
|------------|------------------------------|------------------------------------|
| Backup     | `scripts/qdrant-backup.sh`   | Binary snapshots, Docker-dependent |
| Restore    | `scripts/qdrant-restore.sh`  | Requires matching Qdrant version   |
| List items | `arc collection items`       | Read-only, no export               |
| Verify     | `arc collection verify`      | Integrity check only               |

## Research Findings

### Investigation Process

1. Tested Qdrant API `scroll()` with `with_vectors=True` - confirmed vector retrieval
2. Analyzed existing codebase scroll patterns in `verify.py`, `sync.py`, `collections.py`
3. Reviewed `upsert()` patterns for import feasibility
4. Examined collection metadata storage in `collection_metadata.py`
5. Analyzed size implications of JSON vs binary formats for vectors

### Key Discoveries

1. **Vector Retrieval Works**: `scroll(with_vectors=True, with_payload=True)` returns
   full embedding vectors

   ```python
   points, offset = client.scroll(
       collection_name="MyCollection",
       with_vectors=True,
       with_payload=True,
       limit=100
   )
   # Returns: point.vector = {"stella": [0.041, 0.030, ...]}
   ```

2. **Named Vectors**: Collections use named vectors, not flat arrays. Export format
   must preserve vector names.

3. **Collection Configuration**: Must export vector params for recreation:
   - Vector name (e.g., "stella")
   - Dimensions (e.g., 1024)
   - Distance metric (COSINE)
   - HNSW parameters (m, ef_construct)

4. **Reserved Metadata Point**: UUID `00000000-0000-0000-0000-000000000001` stores
   collection metadata - must be preserved during export/import

5. **Size Analysis**: JSON format is ~10x larger than compressed binary for vectors due to
   float-to-string conversion overhead. Binary format with gzip compression provides the
   best balance of size and speed.

6. **Collection Type Payload Differences**: Each type has different metadata fields:

   | Field                    | PDF | Code | Markdown |
   |--------------------------|-----|------|----------|
   | `file_path`              | Yes | Yes  | Yes      |
   | `git_project_root`       | -   | Yes  | -        |
   | `git_project_identifier` | -   | Yes  | -        |
   | `git_branch`             | -   | Yes  | -        |
   | `git_commit_hash`        | -   | Yes  | -        |
   | `file_hash`              | Yes | -    | Yes      |
   | `page_count`             | Yes | -    | -        |
   | `line_count`             | -   | Yes  | -        |
   | `programming_language`   | -   | Yes  | -        |
   | `header_path`            | -   | -    | Yes      |

7. **Absolute Paths Problem**: All collections store absolute file paths that will be
   invalid after migration to another machine:
   - PDF: `file_path` (e.g., `/Users/alice/Documents/report.pdf`)
   - Code: `file_path` + `git_project_root` (e.g., `/home/alice/repos/myproject`)
   - Markdown: `file_path` (e.g., `/Users/alice/notes/readme.md`)

## Proposed Solution

### Approach

Add two new CLI commands to the `arc collection` subgroup:

```bash
# Default: Binary format (compact, fast)
arc collection export <name> -o <file.arcexp>
arc collection import <file.arcexp> [--into <name>]

# Path-based filtering (works on file_path - all collection types)
arc collection export MyPDFs -o reports.arcexp --include "*/reports/*.pdf"
arc collection export MyPDFs -o single.arcexp --include "/path/to/specific.pdf"
arc collection export Docs -o subset.arcexp --exclude "*/drafts/*"

# Multiple patterns (OR for includes, AND for excludes)
arc collection export MyPDFs -o subset.arcexp \
    --include "*/reports/*.pdf" --include "*/docs/*.pdf" \
    --exclude "*/temp/*"

# Code: path-based filtering (if collection indexes multiple root paths)
arc collection export Code -o repos.arcexp --include "~/repos/*"

# Code: repo-based filtering (on git_project metadata)
arc collection export Code -o arcaneum.arcexp --repo arcaneum        # all branches
arc collection export Code -o main.arcexp --repo arcaneum#main       # specific branch
arc collection export Code -o multi.arcexp --repo arcaneum --repo mylib  # multiple repos

# Code: combined path + repo filtering
arc collection export Code -o subset.arcexp \
    --include "~/repos/*" \
    --repo arcaneum#main \
    --exclude "*/test/*"

# Detached export: strip root prefix, store relative paths (shareable)
arc collection export MyCode -o shareable.arcexp --detach

# Import detached export with new root (--attach is symmetric with --detach)
arc collection import shareable.arcexp --attach /home/bob/projects

# Path substitution for non-detached exports (explicit old:new mapping)
arc collection import backup.arcexp --remap /Users/alice/docs:/home/bob/documents

# Optional: JSONL format for debugging
arc collection export <name> -o <file.jsonl> --format jsonl
```

**Default format**: Compressed binary (`.arcexp` extension) using:

- **msgpack** for metadata and payloads (compact, schema-less, fast)
- **numpy** for vectors (native float32 arrays)
- **gzip** compression wrapper

**Debug format**: JSONL (opt-in via `--format jsonl`) for inspection/debugging.

### Technical Design

#### Binary Export Format (`.arcexp`)

The `.arcexp` file is a gzip-compressed stream with the following structure:

```text
[GZIP COMPRESSED STREAM]
├── Magic bytes: "ARCE" (4 bytes) - Arcaneum Export
├── Version: uint8 (1 byte) - format version (currently 1)
├── Header length: uint32 (4 bytes, little-endian)
├── Header: msgpack-encoded dict
│   ├── collection_name: str
│   ├── collection_type: str (pdf/code/markdown)
│   ├── model: str (stella/bge/jina-code)
│   ├── vector_config: dict {name: {size, distance}}
│   ├── point_count: int
│   ├── root_prefix: str (common path prefix, auto-detected)
│   ├── detached: bool (true if paths are relative, root stripped)
│   └── exported_at: str (ISO timestamp)
├── Points: sequence of msgpack-encoded records
│   ├── Point 1: {id, vectors: {name: bytes(float32[])}, payload}
│   ├── Point 2: {id, vectors: {name: bytes(float32[])}, payload}
│   └── ...
└── EOF marker: msgpack nil (0xc0)
```

Vectors are stored as raw `float32` bytes within msgpack `bin` type, avoiding
float-to-string conversion overhead.

#### Filter Implementation Notes

**Filter flags:**

| Flag | Target Field | Matching | Use Case |
| --- | --- | --- | --- |
| `--include <glob>` | `file_path` | fnmatch glob | PDF, Markdown, Code path filtering |
| `--exclude <glob>` | `file_path` | fnmatch glob | Exclude paths from export |
| `--repo <name>` | `git_project_name` | exact match | Code: all branches of repo |
| `--repo <name>#<branch>` | `git_project_identifier` | exact match | Code: specific branch |

**Filter logic (all must pass):**

1. **Include patterns**: If any `--include` specified, `file_path` must match at least one
   pattern (multiple includes = OR).
2. **Exclude patterns**: `file_path` must NOT match any exclude pattern (multiple excludes
   = AND).
3. **Repo filter** (code only): If `--repo` specified, must match at least one repo
   (multiple repos = OR).
4. **No filters**: Export entire collection.

**Implementation:**

- `--include`/`--exclude`: Uses Python's `fnmatch` module, matched against full `file_path`.
- `--repo`: Uses Qdrant's native filtering for efficiency. `--repo arcaneum` filters on
  `git_project_name == "arcaneum"`, `--repo arcaneum#main` filters on
  `git_project_identifier == "arcaneum#main"`.
- **Metadata point preservation**: Filtered exports always include the reserved metadata
  point (UUID `00000000-0000-0000-0000-000000000001`) regardless of filter criteria, as
  it is required for collection recreation on import.

#### JSONL Debug Format (`.jsonl`)

Human-readable format for debugging and inspection:

```text
{"_header": true, "_version": 1, "collection_name": "MyPDFs", ...}
{"id": "abc-123", "vector": {"stella": [0.041, ...]}, "payload": {...}}
{"id": "def-456", "vector": {"stella": [0.022, ...]}, "payload": {...}}
```

#### Export Architecture

```text
arc collection export MyPDFs -o backup.arcexp
         │
         ▼
┌─────────────────────────────────────┐
│  CollectionExporter                 │
│  ├─ get_collection_config()         │  → Vector params, HNSW config
│  ├─ get_collection_metadata()       │  → Type, model from reserved point
│  ├─ write_header()                  │  → Magic + version + msgpack header
│  └─ stream_points()                 │
│      └─ scroll(with_vectors=True)   │  → Paginated retrieval
│          └─ serialize_point()       │  → msgpack with binary vectors
└─────────────────────────────────────┘
         │
         ▼
    backup.arcexp (gzip-compressed stream)
```

#### Import Architecture

```text
arc collection import backup.arcexp --into MyPDFs-restored
         │
         ▼
┌─────────────────────────────────────┐
│  CollectionImporter                 │
│  ├─ read_header()                   │  → Validate magic, parse header
│  ├─ validate_header()               │  → Check version, required fields
│  ├─ create_collection()             │  → With vector config from header
│  └─ stream_import()                 │
│      └─ batch_points()              │  → Group into batches of 100-500
│          └─ upsert()                │  → Upload batch
└─────────────────────────────────────┘
         │
         ▼
    Qdrant collection (progressively populated)
```

### Implementation Example

```python
# src/arcaneum/cli/export_import.py (illustrative)

import gzip
import struct
import msgpack
import numpy as np

MAGIC = b"ARCE"
VERSION = 1

def export_collection(client, name: str, output_path: str):
    """Export collection to .arcexp format (illustrative)."""
    info = client.get_collection(name)

    with gzip.open(output_path, 'wb') as f:
        # Write format header
        f.write(MAGIC)
        f.write(struct.pack('B', VERSION))

        # Write collection metadata
        header = msgpack.packb({
            "collection_name": name,
            "vector_config": {"stella": {"size": 1024, "distance": "Cosine"}},
            "point_count": info.points_count
        })
        f.write(struct.pack('<I', len(header)))
        f.write(header)

        # Stream points with binary vectors
        offset = None
        while True:
            points, offset = client.scroll(name, with_vectors=True, limit=100, offset=offset)
            for p in points:
                # Convert vector to binary (avoids float→string overhead)
                vec_bytes = np.array(p.vector["stella"], dtype=np.float32).tobytes()
                f.write(msgpack.packb({"id": str(p.id), "vectors": {"stella": vec_bytes}, "payload": p.payload}))
            if offset is None:
                break
        f.write(msgpack.packb(None))  # EOF marker


def import_collection(client, input_path: str, path_remaps: list[tuple[str, str]] = None):
    """Import collection with optional path remapping (illustrative)."""
    with gzip.open(input_path, 'rb') as f:
        # Validate and read header
        assert f.read(4) == MAGIC
        f.read(1)  # version
        header_len = struct.unpack('<I', f.read(4))[0]
        header = msgpack.unpackb(f.read(header_len))

        # Create collection, stream points...
        for point_data in msgpack.Unpacker(f):
            if point_data is None:
                break
            # Remap paths for cross-machine migration
            payload = point_data["payload"]
            if path_remaps and "file_path" in payload:
                for old, new in path_remaps:
                    if payload["file_path"].startswith(old):
                        payload["file_path"] = new + payload["file_path"][len(old):]
                        break
            # Reconstruct vector from binary
            vec = np.frombuffer(point_data["vectors"]["stella"], dtype=np.float32).tolist()
            # ... upsert point
```

### CLI Integration

```python
# Add to src/arcaneum/cli/collections.py

@collection.command()
@click.argument("name")
@click.option("-o", "--output", required=True, type=click.Path(),
              help="Output file path (.arcexp or .jsonl)")
@click.option("--format", "fmt", type=click.Choice(["binary", "jsonl"]),
              default="binary", help="Export format (default: binary)")
@click.option("--include", "includes", multiple=True,
              help="Include files matching glob pattern (file_path)")
@click.option("--exclude", "excludes", multiple=True,
              help="Exclude files matching glob pattern (file_path)")
@click.option("--repo", "repos", multiple=True,
              help="Filter by repo name or repo#branch (code collections)")
@click.option("--detach", is_flag=True,
              help="Strip root prefix, store relative paths (shareable)")
@click.option("--json", "output_json", is_flag=True, help="Output stats as JSON")
def export(name: str, output: str, fmt: str, includes: tuple,
           excludes: tuple, repos: tuple, detach: bool, output_json: bool):
    """Export collection to portable format.

    Default format is compressed binary (.arcexp) for efficiency.
    Use --format jsonl for human-readable debug output.

    Filter options (all filters combined with AND):
      --include   Include files matching glob pattern on file_path (multiple = OR)
      --exclude   Exclude files matching glob pattern on file_path (multiple = AND)
      --repo      Filter by repo name (all branches) or repo#branch (code only)

    Path options:
      --detach    Strip common root prefix from paths, storing relative paths.
                  Use --attach on import to prepend new root. Enables sharing
                  collections without exposing your directory structure.

    Examples:
        arc collection export MyPDFs -o backup.arcexp
        arc collection export MyPDFs -o reports.arcexp --include "*/reports/*.pdf"
        arc collection export Code -o arcaneum.arcexp --repo arcaneum#main
        arc collection export Code -o subset.arcexp --include "~/repos/*" --repo arcaneum#main
        arc collection export MyCode -o shareable.arcexp --detach
    """
    from arcaneum.cli.export_import import BinaryExporter, JsonlExporter

    client = get_qdrant_client()

    # Build filter based on options
    scroll_filter, path_filter = build_export_filter(
        includes=includes,
        excludes=excludes,
        repos=repos
    )

    def progress(current, total):
        if not output_json:
            pct = (current / total * 100) if total > 0 else 0
            click.echo(f"\rExporting: {current}/{total} ({pct:.1f}%)", nl=False)

    if fmt == "jsonl":
        exporter = JsonlExporter(client)
    else:
        exporter = BinaryExporter(client)

    result = exporter.export(
        collection_name=name,
        output_path=Path(output),
        scroll_filter=scroll_filter,  # Qdrant filter (for --repo)
        path_filter=path_filter,       # Post-scroll filter (for --include/--exclude)
        detach=detach,                 # Strip root prefix, store relative paths
        progress_callback=progress if not output_json else None
    )

    if not output_json:
        click.echo()
        click.echo(f"Exported {result['exported_count']} points to {result['output_path']}")
        size_mb = result['file_size_bytes'] / (1024 * 1024)
        click.echo(f"File size: {size_mb:.2f} MB")
    else:
        click.echo(json.dumps(result))


@collection.command("import")
@click.argument("file", type=click.Path(exists=True))
@click.option("--into", "target_name", help="Target collection name")
@click.option("--attach", "attach_root",
              help="Attach root path to relative paths (for detached exports)")
@click.option("--remap", "remaps", multiple=True,
              help="Path substitution: old:new prefix mapping (for non-detached exports)")
@click.option("--json", "output_json", is_flag=True, help="Output stats as JSON")
def import_collection(file: str, target_name: str, attach_root: str,
                      remaps: tuple, output_json: bool):
    """Import collection from export file.

    Automatically detects format from file content (binary .arcexp or JSONL).

    Path handling options:
      --attach     Prepend root path to relative paths. Use with detached exports.
                   Symmetric with --detach on export.
      --remap      Explicit path substitution (old:new format). Use with non-detached
                   exports to update absolute paths for new machine.

    Examples:
        arc collection import backup.arcexp
        arc collection import backup.arcexp --into MyPDFs-restored
        arc collection import shareable.arcexp --attach /home/bob/projects
        arc collection import backup.arcexp --remap /Users/alice/docs:/home/bob/docs
    """
    from arcaneum.cli.export_import import BinaryImporter, JsonlImporter

    client = get_qdrant_client()
    input_path = Path(file)

    # Parse path remappings (old:new explicit substitution)
    path_remaps = []
    for remap in remaps:
        if ':' not in remap:
            raise click.UsageError("--remap requires old:new format (e.g., /old/path:/new/path)")
        old, new = remap.split(':', 1)
        path_remaps.append((old, new))

    # Auto-detect format by reading first bytes
    with open(input_path, 'rb') as f:
        magic = f.read(4)

    if magic[:2] == b'\x1f\x8b':  # gzip magic
        importer = BinaryImporter(client)
    else:
        importer = JsonlImporter(client)

    def progress(current, total):
        if not output_json:
            pct = (current / total * 100) if total > 0 else 0
            click.echo(f"\rImporting: {current}/{total} ({pct:.1f}%)", nl=False)

    result = importer.import_collection(
        input_path=input_path,
        target_name=target_name,
        attach_root=attach_root,       # For detached exports: prepend this root
        path_remaps=path_remaps if path_remaps else None,  # For non-detached: old:new substitution
        progress_callback=progress if not output_json else None
    )

    if not output_json:
        click.echo()
        click.echo(f"Imported {result['imported_count']} points into "
                   f"'{result['collection_name']}'")
    else:
        click.echo(json.dumps(result))
```

## Alternatives Considered

### Alternative 1: JSONL as Default Format

**Description**: Use JSON Lines as the primary export format

**Pros**:

- Human-readable, inspectable with `head`, `jq`
- No binary parsing required
- Standard format

**Cons**:

- ~10x larger than compressed binary
- Slower serialization (float to string conversion)
- Impractical for large collections (500k points = 7.5 GB)

**Decision**: Keep as opt-in debug format (`--format jsonl`), not default

### Alternative 2: Qdrant Native Snapshots

**Description**: Enhance existing snapshot scripts instead of new format

**Pros**:

- Native Qdrant format (fastest, most reliable)
- Includes internal optimizations
- Already implemented

**Cons**:

- Docker-dependent extraction
- Cannot filter/subset exports
- Version-sensitive
- No CLI integration

**Reason for rejection**: Does not address selective export or CLI integration needs

### Alternative 3: Parquet/Arrow Format

**Description**: Use columnar format for efficient storage

**Pros**:

- Excellent compression
- Industry standard
- Fast analytics queries

**Cons**:

- Adds pyarrow dependency (~100MB)
- Overkill for sequential point export
- Complex schema for nested vector data

**Reason for rejection**: msgpack + numpy achieves similar compression without heavy deps

### Alternative 4: SQLite Export

**Description**: Export to SQLite database file

**Pros**:

- Single portable file
- SQL query capability

**Cons**:

- Vector storage awkward (BLOB columns)
- Slower than streaming format
- Not designed for this use case

**Reason for rejection**: Binary stream format is simpler and faster

## Trade-offs and Consequences

### Positive Consequences

- **Compact files**: ~10x smaller than JSON, transfers faster
- **Fast I/O**: Binary serialization avoids string conversion overhead
- **Selective export**: Filter by path patterns (--include/--exclude) or repo (--repo)
- **CLI integration**: Consistent with existing `arc collection` commands
- **Streaming**: Memory-efficient for large collections
- **Debug option**: JSONL available when inspection needed

### Negative Consequences

- **Not human-readable by default**: Requires `--format jsonl` for inspection
- **Dependencies**: Adds msgpack and numpy (both lightweight, commonly installed)
- **Custom format**: Not a standard interchange format

### Risks and Mitigations

- **Risk**: Format version incompatibility in future
  **Mitigation**: Version byte in header; maintain backward compatibility

- **Risk**: Import into existing collection could corrupt data
  **Mitigation**: Refuse to import into existing collection; require `--into`

- **Risk**: Corrupted export file undetectable
  **Mitigation**: Consider adding checksum in future version

## Implementation Plan

### Prerequisites

- [ ] Verify msgpack and numpy are in project dependencies
- [ ] Review existing collection metadata implementation
- [ ] Test vector serialization roundtrip (export then import)

### Step-by-Step Implementation

#### Step 1: Create Export/Import Module

Create `src/arcaneum/cli/export_import.py` with:

- `BinaryExporter` class for `.arcexp` format
- `BinaryImporter` class for `.arcexp` format
- `JsonlExporter` class for debug format
- `JsonlImporter` class for debug format

#### Step 2: Add CLI Commands

Add `export` and `import` commands to `src/arcaneum/cli/collections.py`.

#### Step 3: Add Tests

Create `tests/test_export_import.py` with:

- Unit tests for binary serialization/deserialization
- Unit tests for JSONL serialization/deserialization
- Integration tests with small test collection
- Roundtrip test (export then import, verify identical)
- Format auto-detection test

#### Step 4: Update Documentation

- Add usage examples to CLI reference
- Document file format specification
- Add to migration guide as alternative to snapshots

### Files to Modify

- `src/arcaneum/cli/export_import.py` - New file: export/import logic
- `src/arcaneum/cli/collections.py` - Add export/import commands
- `tests/test_export_import.py` - New file: tests
- `docs/guides/cli-reference.md` - Document new commands
- `docs/guides/qdrant-migration.md` - Add export as option
- `pyproject.toml` - Add msgpack and numpy dependencies

### Dependencies

- `msgpack` - Compact binary serialization (must be added to pyproject.toml)
- `numpy` - Vector array handling (must be added to pyproject.toml)
- `gzip` - Compression (stdlib)

## Validation

### Testing Approach

1. Create small test collection with known content
2. Export to binary format
3. Verify file structure (magic, version, header)
4. Import into new collection with different name
5. Compare point counts and sample content
6. Repeat with JSONL format

### Test Scenarios

1. **Scenario**: Export small collection (100 points) to binary
   **Expected Result**: Compressed file ~400 KB (vs ~5 MB uncompressed JSON)

2. **Scenario**: Export with project filter
   **Expected Result**: Only matching points exported

3. **Scenario**: Import into existing collection name
   **Expected Result**: Error message, no changes made

4. **Scenario**: Export then import roundtrip
   **Expected Result**: New collection has identical point count and content

5. **Scenario**: Auto-detect format on import
   **Expected Result**: Correctly identifies binary vs JSONL

6. **Scenario**: Import with path remapping
   **Expected Result**: `file_path` updated from `/Users/alice/docs` to `/home/bob/docs`

7. **Scenario**: Import code collection with path remapping
   **Expected Result**: Both `file_path` and `git_project_root` remapped correctly

8. **Scenario**: Verify chunk integrity after import
   **Expected Result**: `arc collection verify` passes (all chunks present)

9. **Scenario**: Export code collection with `--repo arcaneum` (all branches)
   **Expected Result**: Only chunks from arcaneum repo exported (all branches)

10. **Scenario**: Export code collection with `--repo arcaneum#main` (specific branch)
    **Expected Result**: Only chunks from arcaneum#main exported

11. **Scenario**: Export with `--include "*.py"` and `--exclude "*/test/*"`
    **Expected Result**: Only Python files outside test directories exported

12. **Scenario**: Export code with `--include "~/repos/*"` (path-based filtering)
    **Expected Result**: Only files under ~/repos/ exported

13. **Scenario**: Combined `--include "~/repos/*" --repo arcaneum#main`
    **Expected Result**: Only arcaneum#main files that are under ~/repos/ exported

14. **Scenario**: Combined `--repo arcaneum#main --exclude "*/test/*"`
    **Expected Result**: arcaneum#main files excluding test directories

15. **Scenario**: Export with --detach flag
    **Expected Result**: Paths stored as relative (root prefix stripped), header has
    `detached: true` and `root_prefix` set to stripped prefix

16. **Scenario**: Import detached export with --attach
    **Expected Result**: New root prepended to relative paths, resulting in valid
    absolute paths on new machine

17. **Scenario**: Import detached export without --attach
    **Expected Result**: Paths remain relative; search works but file lookup would fail

18. **Scenario**: Import non-detached export with --remap (explicit old:new substitution)
    **Expected Result**: Old path prefix replaced with new path prefix in all
    `file_path` fields

### Performance Validation

| Operation            | Target (10k points) | Target (100k points) |
|----------------------|---------------------|----------------------|
| Binary export        | < 30 seconds        | < 5 minutes          |
| Binary import        | < 60 seconds        | < 10 minutes         |
| JSONL export         | < 60 seconds        | < 10 minutes         |
| File size (binary)   | ~4 MB               | ~40 MB               |
| File size (JSONL)    | ~50 MB              | ~500 MB              |

### Security Validation

- File paths in payloads preserved as-is (user responsibility)
- No execution of payload content
- Validate msgpack parsing (no injection risks)

## References

- [Qdrant Scroll API](https://qdrant.tech/documentation/concepts/points/#scroll-points)
- [msgpack Python](https://github.com/msgpack/msgpack-python)
- [NumPy Binary Format](https://numpy.org/doc/stable/reference/routines.io.html)
- Existing implementation: `src/arcaneum/indexing/verify.py` (scroll patterns)
- Existing implementation: `src/arcaneum/indexing/collection_metadata.py`

## Notes

### Post-Import Workflow

After importing with path adjustments:

```bash
# Import detached export with --attach (symmetric with --detach)
arc collection import shareable.arcexp --attach /home/bob/projects

# Import non-detached export with --remap (explicit path substitution)
arc collection import backup.arcexp --remap /Users/alice:/home/bob

# Verify chunk integrity (existing command - checks for missing chunks)
arc collection verify MyCollection

# List indexed files to inspect paths (existing command)
arc collection items MyCollection
```

**Note**: The existing `arc collection verify` checks chunk completeness (are all chunks
present for each file), NOT file existence on disk. Verifying files exist at new
paths would require a new feature.

### Future Enhancements

- **Checksum**: Add CRC32 or SHA256 for integrity verification
- **Incremental export**: Export only points modified since timestamp
- **Multi-collection archive**: Export multiple collections to single file
- **Direct transfer**: `arc collection transfer <name> --to <remote-host>`
- **Parallel import**: Use multiple threads for upsert batches
- **File existence check**: `arc collection verify --check-files` to verify source files exist
- **Import dry-run**: `--dry-run` to preview path remappings before importing

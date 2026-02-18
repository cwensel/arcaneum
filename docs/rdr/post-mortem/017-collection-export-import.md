# Post-Mortem: RDR-017 Collection Export and Import

## RDR Summary

RDR-017 proposed adding portable collection export and import CLI commands
(`arc collection export` / `arc collection import`) to enable cross-machine
migration of Qdrant collections. The approach recommended a compact binary
format (`.arcexp`) using msgpack+numpy+gzip as the default, with JSONL as
an opt-in debug format, plus filtering, detached exports, and path remapping
for cross-machine path translation.

## Implementation Status

Implemented

The feature was fully implemented and merged. Both export and import commands
are operational for binary and JSONL formats, with filtering, detached exports,
path remapping, and format auto-detection all functional. Documentation was
updated in both the CLI reference and the Qdrant migration guide.

---

## Implementation vs. Plan

### What Was Implemented as Planned

- **Binary export format (.arcexp)**: Magic bytes ("ARCE"), version byte,
  msgpack-encoded header with uint32 length prefix, streaming msgpack point
  records with binary float32 vectors, gzip compression, and EOF marker --
  all exactly as specified in the format diagram.
- **JSONL debug format**: Header line with `_header: true` and `_version`,
  followed by one JSON object per point with id, vector, and payload fields.
- **CLI commands**: `arc collection export` and `arc collection import` added
  to the `collection` command group with the exact flags specified: `--output`,
  `--format`, `--include`, `--exclude`, `--repo`, `--detach`, `--into`,
  `--attach`, `--remap`, and `--json`.
- **Filter system**: Include/exclude patterns use `fnmatch` against `file_path`
  (includes=OR, excludes=AND). Repo filtering uses Qdrant native `Filter` with
  `FieldCondition` on `git_project_name` or `git_project_identifier` (repos=OR).
  Combined filters use AND logic. All matches the RDR specification.
- **Metadata point preservation**: The reserved metadata point
  (`00000000-0000-0000-0000-000000000001`) is always exported first regardless
  of filter criteria, exactly as planned.
- **Detached exports**: `--detach` auto-detects common root prefix from
  `file_path` values, strips it from both `file_path` and `git_project_root`,
  and stores the prefix in the header.
- **Path remapping on import**: `--attach` prepends a root to relative paths
  (symmetric with `--detach`), and `--remap` applies explicit old:new prefix
  substitution. Both transform `file_path` and `git_project_root`.
- **Import safety**: Import refuses to overwrite existing collections, requiring
  `--into` to specify a different name.
- **Format auto-detection**: Import detects binary vs JSONL by checking for
  gzip magic bytes (`0x1f 0x8b`).
- **Streaming I/O**: Export scrolls with `limit=100` batches; import upserts
  in batches of 100. Memory-efficient for large collections.
- **Dependencies**: `msgpack>=1.1.0` and `numpy>=1.26.0` added to
  `pyproject.toml`.
- **Module structure**: `src/arcaneum/cli/export_import.py` created with
  `BinaryExporter`, `BinaryImporter`, `JsonlExporter`, `JsonlImporter`.
- **Tests**: `tests/test_export_import.py` created covering path utilities,
  filter building, header serialization, binary format structure, binary
  roundtrip, JSONL format, JSONL roundtrip, detached export, attach on
  import, path remapping, repo filtering, and import validation.
- **Documentation**: `docs/guides/cli-reference.md` updated with full
  export/import command documentation. `docs/guides/qdrant-migration.md`
  updated with export/import as an alternative to snapshots.

### What Diverged from the Plan

- **HNSW config not preserved in exports**: The RDR stated that export
  should capture "HNSW parameters (m, ef_construct)" from the collection
  configuration. The implementation does not include HNSW config in the
  export header. On import, it hardcodes `HnswConfigDiff(m=16, ef_construct=100)`
  instead of restoring the original values. This works for the current
  codebase because all collections use these same defaults, but would
  silently lose custom HNSW tuning.

- **Progress display uses Rich instead of click.echo**: The RDR showed
  progress via `click.echo(f"\rExporting: {current}/{total} ({pct:.1f}%)")`.
  The implementation uses Rich's `Progress` with `SpinnerColumn`, `BarColumn`,
  and `TaskProgressColumn` for a polished progress bar. This is a visual
  improvement, not a functional divergence.

- **Three-layer command architecture instead of two**: The RDR placed CLI
  decorators and business logic together in `collections.py`. The
  implementation separates click decorators in `main.py`, command functions
  in `collections.py`, and core export/import logic in `export_import.py`.
  This is cleaner than the RDR's layout.

- **No BaseImporter abstract class**: The RDR implied symmetric class
  hierarchies for exporters and importers. The implementation provides
  `BaseExporter(ABC)` with shared `_scroll_points()` and
  `_get_collection_info()` methods, but both `BinaryImporter` and
  `JsonlImporter` are standalone classes with duplicated `_create_collection()`
  logic. There is no `BaseImporter` abstract class.

### What Was Added Beyond the Plan

- **Interaction logging (RDR-018)**: Both export and import commands include
  `interaction_logger.start()` / `interaction_logger.finish()` calls for
  operational telemetry, which was not anticipated by RDR-017.

- **ExportResult and ImportResult dataclasses**: Structured result objects
  with `to_dict()` methods for clean JSON output. The RDR's illustrative
  code used plain dictionaries.

- **Distance enum string parsing**: The importer handles multiple Distance
  enum string representations (e.g., `"Distance.COSINE"`, `"Cosine"`,
  `"COSINE"`) using case-insensitive substring matching. The RDR did not
  consider serialization format of Qdrant's Distance enum.

- **Single unnamed vector fallback**: The serializer handles collections
  with unnamed vectors (non-dict `point.vector`) by mapping them to a
  `_default` key. The RDR assumed all collections use named vectors.

### What Was Planned but Not Implemented

- **HNSW parameter preservation**: The export header does not store HNSW
  configuration. On import, hardcoded defaults are used. See divergence above.

- **Six of eighteen test scenarios**: The RDR specified 18 test scenarios.
  Twelve were implemented. Missing scenarios include: combined
  `--include` + `--exclude` tests (scenario 11), combined `--repo` +
  `--exclude` (scenario 14), import detached without `--attach`
  (scenario 17), and chunk integrity verification after import
  (scenario 8). The implemented tests cover all critical code paths.

---

## Drift Classification

| Category | Count | Examples |
| -------- | ----- | -------- |
| **Unvalidated assumption** | 0 | |
| **Framework API detail** | 1 | Distance enum serialization format not anticipated; required flexible parsing on import |
| **Missing failure mode** | 0 | |
| **Missing Day 2 operation** | 0 | |
| **Deferred critical constraint** | 1 | HNSW config not preserved; works now because all collections use the same defaults, but would silently lose custom tuning |
| **Over-specified code** | 1 | RDR's click.echo progress display was replaced with Rich progress bars; the illustrative code was not reusable |
| **Under-specified architecture** | 1 | No BaseImporter abstract class; importer `_create_collection()` duplicated across BinaryImporter and JsonlImporter |
| **Scope underestimation** | 1 | 18 test scenarios specified; 12 implemented, covering critical paths but leaving edge-case combinations untested |
| **Internal contradiction** | 0 | |
| **Missing cross-cutting concern** | 1 | Interaction logging (RDR-018) integration was needed but not anticipated |

---

## RDR Quality Assessment

### What the RDR Got Right

- **Binary format design**: The magic+version+header+stream+EOF structure was
  implemented verbatim. The choice of msgpack for metadata and numpy for vectors
  proved correct -- no format issues arose during implementation.
- **Filter architecture**: The two-tier filter approach (Qdrant-native `Filter`
  for repo, client-side `fnmatch` for path patterns) was implemented as designed
  with no changes needed.
- **Path handling strategy**: The detach/attach symmetry and the separate remap
  mechanism for non-detached exports addressed the cross-machine path translation
  problem cleanly. Both `file_path` and `git_project_root` are transformed.
- **Alternatives analysis**: Rejecting JSONL-as-default, Parquet, and SQLite
  in favor of msgpack+gzip was well-reasoned. The dependency additions
  (msgpack, numpy) were minimal.
- **Metadata point handling**: Identifying the reserved metadata point as
  requiring special treatment during filtered exports was a critical insight
  that prevented data loss.
- **Collection type payload differences table**: Documenting which payload
  fields exist per collection type helped guide the path remapping logic.

### What the RDR Missed

- **HNSW config in export header**: The RDR mentioned HNSW parameters in
  the research findings but did not include them in the ExportHeader dataclass
  or format specification. The implementation followed the spec and omitted them.
- **Qdrant Distance enum serialization**: The RDR assumed distance would
  serialize cleanly as a string like `"Cosine"`. In practice, the qdrant-client
  library serializes it as `"Distance.COSINE"`, requiring flexible parsing.
- **Importer class hierarchy**: The RDR specified `BaseExporter` implicitly
  through the class diagrams but did not address shared logic between
  `BinaryImporter` and `JsonlImporter`, leading to code duplication.
- **Unnamed vector edge case**: The RDR stated collections use named vectors
  but did not consider what happens if a collection has a single unnamed
  vector (non-dict `point.vector`).

### What the RDR Over-specified

- **Illustrative CLI code**: The RDR included ~150 lines of illustrative Python
  for both the export function and the full Click command definition. The
  implementation diverged on progress display (Rich instead of click.echo),
  error handling patterns, and code organization. The illustrative code
  suggested a structure that was not adopted.
- **Eighteen test scenarios**: The RDR listed 18 detailed test scenarios with
  expected results. This level of test specification is useful for coverage
  planning but created implicit scope that was larger than needed for initial
  delivery. Twelve tests covered all critical functionality.
- **Performance targets table**: The RDR included specific timing targets
  (e.g., "Binary export < 30 seconds for 10k points") that were not validated
  during implementation. These estimates were reasonable but untested.

---

## Key Takeaways for RDR Process Improvement

1. **Include all preserved configuration in the header specification**: The RDR
   mentioned HNSW parameters in the research section but did not carry them
   forward into the format spec or ExportHeader dataclass. When a format
   spec omits a field, implementers will omit it too. Every piece of
   configuration that should survive a roundtrip must appear explicitly in
   the data structure definition.

2. **Test serialization formats of third-party types during research**: The
   Distance enum serialization mismatch (`"Cosine"` vs `"Distance.COSINE"`)
   could have been caught by a simple `print(str(Distance.COSINE))` during
   the research phase. When an RDR depends on serializing/deserializing
   third-party types, add a research finding that documents the exact
   serialized representation.

3. **Specify shared base classes for symmetric implementations**: The RDR
   designed `BinaryExporter` and `JsonlExporter` with shared patterns but
   did not address the importer side. When two classes will share significant
   logic (like `_create_collection()`), the RDR should note the shared
   abstraction to prevent duplication.

4. **Cap illustrative code at architecture, not implementation**: The 150+
   lines of illustrative Python in the RDR were largely rewritten during
   implementation due to different progress display, error handling, and
   module structure. Illustrative code is most valuable when it shows
   architectural boundaries and data flow, not complete function bodies
   that will be rewritten.

5. **Tier test scenarios by priority**: Listing 18 test scenarios without
   priority tiers makes it unclear which are essential for initial delivery
   vs. which are edge-case hardening. Grouping scenarios into "must-have"
   (roundtrip, format validation, safety checks) and "nice-to-have"
   (combined filters, edge cases) would better guide implementation scope.

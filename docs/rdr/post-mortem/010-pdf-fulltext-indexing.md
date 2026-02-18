# Post-Mortem: RDR-010 Bulk PDF Indexing to Full-Text Search (MeiliSearch)

## RDR Summary

RDR-010 proposed a three-phase pipeline to index PDF documents into MeiliSearch
for exact phrase and keyword search: (1) reuse the RDR-004 PyMuPDF + OCR extraction
pipeline, (2) build page-level MeiliSearch documents with shared metadata, and
(3) batch upload with change detection. The approach was to complement Qdrant
semantic search (RDR-004) with MeiliSearch full-text search via a symmetric CLI
command `arc index text pdf`.

## Implementation Status

Implemented

The core feature is fully implemented and operational. The `arc index text pdf`
command indexes PDFs to MeiliSearch with page-level granularity, SHA-256 change
detection, OCR support, and shared metadata for cooperative workflows. The scope
expanded significantly beyond the original plan, with the `arc index text` group
also covering code (RDR-011) and markdown indexing, and the `arc indexes`
management commands growing to include verify, items, export/import, and
project-level operations. Two planned items remain unimplemented: the
`arc index semantic pdf` rename (deferred arcaneum-h6bo) and the `--text-only`
flag on `arc corpus sync`.

---

## Implementation vs. Plan

### What Was Implemented as Planned

- **PDFFullTextIndexer class** (`src/arcaneum/indexing/fulltext/pdf_indexer.py`):
  Created with the planned methods `index_pdf()`, `_build_meilisearch_documents()`,
  `_compute_file_hash()`, `_is_already_indexed()`, and `index_directory()`. The
  three-phase pipeline (extract, prepare, upload) matches the RDR architecture.

- **100% reuse of RDR-004 extraction pipeline**: The indexer imports and uses
  `PDFExtractor` and `OCREngine` from `indexing.pdf.extractor` and
  `indexing.pdf.ocr` exactly as planned.

- **Page-level indexing**: Each PDF page becomes a separate MeiliSearch document
  with `page_number`, `file_path`, `file_hash`, and all other shared metadata
  fields from the RDR-009 schema.

- **SHA-256 file hash change detection**: Implemented with 8KB chunked streaming
  as specified, used to skip already-indexed files.

- **PDF_DOCS_SETTINGS**: The MeiliSearch index settings in
  `src/arcaneum/fulltext/indexes.py` match the RDR specification exactly,
  including searchable attributes (content, title, author, filename), filterable
  attributes (file_path, page_number, file_hash, extraction_method, is_image_pdf),
  sortable (page_number), typo tolerance settings, stop words, and
  `maxTotalHits: 10000`.

- **CLI command `arc index text pdf`**: Implemented under `arc index text` group
  in `src/arcaneum/cli/main.py` with options for `--index`, `--recursive`,
  `--no-ocr`, `--batch-size`, `--force`, and `--json` output.

- **Batch size 1000**: Default batch size matches the RDR recommendation.

- **`arc indexes` management commands**: list, create, delete, and info
  implemented in `src/arcaneum/cli/fulltext.py`.

- **Tests**: Unit tests (`tests/indexing/fulltext/test_pdf_indexer.py`) and
  integration tests (`tests/indexing/fulltext/test_pdf_fulltext_integration.py`)
  covering hash computation, change detection (new/modified/unchanged/force),
  orphan detection, document building, page splitting, OCR metadata, exact phrase
  search, filtered search, and document deletion.

### What Diverged from the Plan

- **Document ID generation**: The RDR planned simple IDs like
  `{pdf_path.stem}_page{page_num}`. The implementation uses
  `{sanitized_stem}_{path_hash}_p{page_num}` where `path_hash` is an MD5 hash
  of the absolute path. This was necessary because two PDFs with the same filename
  in different directories would produce colliding IDs under the original scheme.
  The stem is also sanitized (non-alphanumeric characters replaced) and truncated
  to 200 characters to comply with MeiliSearch's 511-byte primary key limit.

- **Page splitting strategy**: The RDR planned a simple form-feed split with
  padding. The implementation uses a three-tier fallback: (1) precise
  `page_boundaries` metadata from PDFExtractor (preferred), (2) form-feed
  character splitting, (3) regex-based page marker detection for PyMuPDF4LLM
  markdown output, (4) single-page fallback. The `page_boundaries` approach is
  more reliable than form-feed splitting because it uses character offsets tracked
  during extraction.

- **PDFExtractor return type**: The RDR planned modifying `PDFExtractor.extract()`
  to return `List[PageText]` instead of a concatenated string (Step 2). This
  refactoring was not done. Instead, the extractor was enhanced to include
  `page_boundaries` in its metadata dict (with `start_char` and
  `page_text_length` per page), allowing the indexer to split the concatenated
  string accurately without changing the extractor's interface. This preserved
  backward compatibility with all existing callers.

- **MeiliSearch upload method**: The RDR specified using
  `add_documents_in_batches()` from `meilisearch-python`. The implementation uses
  a custom `add_documents_sync()` method on `FullTextClient` that calls
  `add_documents()` then `wait_for_task()`. This provides explicit error handling
  and status checking (the built-in batch method returns task info without
  waiting).

- **Change detection filter expression**: The RDR showed unquoted filter values
  (`file_path = {pdf_path}`). The implementation correctly quotes string values
  (`file_path = "{file_path_str}" AND file_hash = "{file_hash}"`) as required by
  MeiliSearch filter syntax. The RDR also used `estimatedTotalHits` for detection;
  the implementation uses `len(hits) > 0` which is more reliable for filtered
  queries.

- **CLI module structure**: The RDR planned the CLI command in a Click group class
  pattern (`index_text.command('pdf')`). The implementation uses function-based
  commands registered in `main.py`, with the logic in `cli/index_text.py` as plain
  functions rather than Click group methods. This matches the pattern used by all
  other CLI commands in the project.

- **`cli/fulltext.py` rename**: The RDR planned renaming `cli/fulltext.py` to
  `cli/indexes.py` as part of arcaneum-h6bo. The file was not renamed. Instead,
  the group is aliased: `cli.add_command(indexes_group, name='indexes')` in
  `main.py`, exposing it as `arc indexes` while keeping the module named
  `fulltext.py`.

### What Was Added Beyond the Plan

- **Markdown conversion mode**: The `PDFFullTextIndexer` accepts a
  `markdown_conversion` parameter (default True) and `normalize_only` CLI option,
  integrating RDR-016 markdown extraction (PyMuPDF4LLM) which was developed after
  RDR-010 was written.

- **`delete_pdf_documents()` method**: Not in the RDR but implemented for
  re-indexing workflows where old documents must be removed before new ones are
  added.

- **Empty page skipping**: Pages with only whitespace are skipped during document
  building, reducing unnecessary documents in the index.

- **`--from-file` input mode**: Accepts a file containing PDF paths (one per
  line) or stdin, enabling pipeline workflows like
  `find . -name "*.pdf" | arc index text pdf --from-file -`.

- **Additional CLI options**: `--process-priority`, `--debug`, `--ocr-workers`,
  `--ocr-language`, `--normalize-only` were added for production use but not
  anticipated by the RDR.

- **Interaction logging (RDR-018)**: All CLI commands log operation metadata for
  telemetry and debugging.

- **Health check and settings verification**: The CLI verifies MeiliSearch is
  available and that existing indexes have the required filterable attributes
  before proceeding.

- **Signal handling**: Ctrl-C handler for graceful interruption of long-running
  indexing operations.

- **Sync module expansion**: `indexing/fulltext/sync.py` grew to include
  `find_files_to_index()` (batch change detection with new/modified/unchanged
  classification), `get_orphaned_files()` (detect indexed files that no longer
  exist), and `GitCodeMetadataSync` (RDR-011 git-aware sync).

- **`arc indexes` additional commands**: verify, items, update-settings,
  export/import, list-projects, delete-project were all added beyond the original
  list/create/delete/info plan.

- **`arc index text code` and `arc index text markdown`**: The `arc index text`
  group expanded to cover source code (RDR-011) and markdown indexing, not just
  PDFs.

- **FullTextClient enhancements**: `add_documents_sync()`,
  `add_documents_batch_parallel()`, `delete_documents_by_file_paths()`,
  `get_all_file_paths()`, `get_chunk_counts_by_file()`, `get_index_settings()`,
  and `update_index_settings()` were all added to the client beyond what was
  needed for this RDR alone.

### What Was Planned but Not Implemented

- **`arc index semantic pdf` rename (arcaneum-h6bo)**: The RDR expected
  `arc index pdf` would be renamed to `arc index semantic pdf` for symmetric
  naming. This rename has not happened. The commands are `arc index pdf`
  (Qdrant) and `arc index text pdf` (MeiliSearch).

- **`--text-only` flag on `arc corpus sync`** (Step 5): The RDR planned adding
  a flag to the dual-indexing sync command to index only to MeiliSearch. This
  was not implemented. Users use `arc index text pdf` for MeiliSearch-only
  indexing as a standalone pathway instead.

- **`indexing/fulltext/__init__.py`**: The planned `__init__.py` for the
  `indexing/fulltext/` package was never created. Python still discovers the
  modules because the parent `indexing/` package has its own `__init__.py`.

- **`PDFExtractor` return type refactoring**: Returning `List[PageText]` objects
  instead of concatenated text was planned but replaced by the
  `page_boundaries` metadata approach.

---

## Drift Classification

| Category | Count | Examples |
| -------- | ----- | -------- |
| **Unvalidated assumption** | 2 | MeiliSearch filter syntax (unquoted values); `estimatedTotalHits` reliability for filtered queries |
| **Framework API detail** | 2 | `add_documents_in_batches()` not used (custom sync method needed); Document ID constraints (511 bytes, alphanumeric only) |
| **Missing failure mode** | 1 | Empty pages not considered (would create searchless documents) |
| **Missing Day 2 operation** | 2 | No delete operation planned; No index settings update/verify commands planned |
| **Deferred critical constraint** | 1 | `arc index semantic pdf` rename deferred indefinitely (arcaneum-h6bo) |
| **Over-specified code** | 1 | RDR code samples for `_split_into_pages()` were substantially rewritten (three-tier fallback replaced simple form-feed split) |
| **Under-specified architecture** | 1 | Sync module scope: RDR planned a small `sync.py` for change detection but it grew to cover batch detection, orphan detection, and git-aware sync |
| **Scope underestimation** | 2 | `arc indexes` management commands (verify, items, export/import, project operations); `arc index text` group expanding to code and markdown |
| **Internal contradiction** | 0 | |
| **Missing cross-cutting concern** | 2 | Interaction logging (RDR-018); Process priority and signal handling for production indexing |

---

## RDR Quality Assessment

### What the RDR Got Right

- **Technology selection**: Reusing the RDR-004 extraction pipeline was the
  correct call. The implementation imports `PDFExtractor` and `OCREngine`
  directly with zero duplication, validating the "100% code reuse" principle.

- **Page-level indexing decision**: The analysis of page-level vs document-level
  vs token-aware chunking correctly identified page-level as the right
  granularity for full-text search. This carried through to implementation
  unchanged.

- **Shared metadata schema**: The field mapping between Qdrant and MeiliSearch
  (file_path, filename, page_number, file_hash, extraction_method, is_image_pdf)
  was implemented exactly as specified and enables cooperative workflows.

- **PDF_DOCS_SETTINGS specification**: The MeiliSearch index configuration
  (searchable attributes, filterable attributes, typo tolerance, pagination) was
  implemented verbatim. This is a rare case of implementation code matching RDR
  code exactly.

- **Three-phase pipeline architecture**: The extract-prepare-upload pipeline
  structure mapped cleanly to the implementation, providing clear separation
  of concerns.

- **Alternatives analysis**: Rejecting document-level indexing, separate
  extraction pipelines, token-aware chunking, and Elasticsearch were all
  well-reasoned decisions that held up during implementation.

### What the RDR Missed

- **Document ID uniqueness across directories**: The simple
  `{stem}_page{N}` scheme would cause collisions for same-named PDFs in
  different directories. The implementation needed path hashing and character
  sanitization for MeiliSearch's primary key constraints.

- **Page splitting reliability**: Depending on form-feed characters as the
  primary splitting mechanism is fragile. The RDR acknowledged this as a known
  limitation but did not propose the `page_boundaries` metadata approach that
  the implementation adopted.

- **Delete/update workflows**: The RDR did not plan for removing previously
  indexed documents when PDFs are updated or deleted. The implementation added
  `delete_pdf_documents()` and orphan detection to handle these cases.

- **Production CLI concerns**: Signal handling, process priority, health checks,
  settings verification, and interaction logging are all necessary for production
  use but absent from the RDR.

- **MeiliSearch filter syntax**: The RDR code samples used unquoted string
  values in filter expressions, which would fail at runtime. MeiliSearch requires
  quoted strings in filter expressions.

### What the RDR Over-specified

- **Implementation code samples**: The RDR included 200+ lines of Python code
  for `PDFFullTextIndexer` and `index_text_pdf` CLI command. While these
  provided useful direction, approximately 60% of the code was substantially
  rewritten. The page splitting, document ID generation, error handling, and
  MeiliSearch client interaction all diverged from the samples.

- **Step 2 (Modify PDFExtractor)**: The planned refactoring to return
  `List[PageText]` was unnecessary. The `page_boundaries` metadata approach
  solved the problem without touching the extractor's public API, saving the
  estimated 2 hours and avoiding downstream breakage.

- **Performance benchmarks**: The RDR specified precise throughput metrics
  (100-200 PDFs/minute, 10MB per 100 pages) without measurements. These
  numbers were aspirational and not validated.

- **Effort estimate for Step 5 (RDR-009 integration)**: 4 hours were estimated
  for adding `--text-only` to corpus sync. This was skipped entirely because
  the standalone `arc index text pdf` pathway was sufficient.

---

## Key Takeaways for RDR Process Improvement

1. **Validate API syntax in code samples against official documentation**: The
   RDR included MeiliSearch filter expressions with unquoted string values and
   referenced `add_documents_in_batches()` without verifying the actual client
   API behavior. Code samples should be tested against the real API or marked
   as pseudocode. This would have caught the filter syntax and batch upload
   method issues before implementation.

2. **Plan for document identity edge cases when specifying primary key schemes**:
   The `{stem}_page{N}` ID scheme failed for the common case of same-named files
   in different directories. RDRs specifying primary keys or unique identifiers
   should enumerate collision scenarios (same name different path, special
   characters, length limits) and validate against the target system's
   constraints.

3. **Separate interface-preserving enhancements from interface-breaking
   refactors**: Step 2 planned a breaking change to `PDFExtractor.extract()`
   return type. The implementation found a non-breaking alternative
   (page_boundaries metadata). RDRs should evaluate whether a change can be
   achieved by adding metadata to existing interfaces before proposing return
   type changes that affect all callers.

4. **Include Day 2 operations (delete, update, verify) in the initial plan**:
   The RDR focused entirely on the indexing path and did not address deletion,
   re-indexing of changed files, or index health verification. These were all
   needed and implemented. RDRs for indexing pipelines should include a section
   on data lifecycle (create, update, delete, verify) even if implementation
   is deferred.

5. **Scope cross-cutting CLI features (signal handling, logging, health checks)
   as a shared pattern rather than per-RDR**: Three separate additions
   (interaction logging, signal handling, health checks) were needed but not
   anticipated. These patterns are identical across `arc index text pdf`,
   `arc index text code`, and `arc index text markdown`. A single RDR
   establishing CLI command patterns would prevent rediscovering these needs
   per feature.

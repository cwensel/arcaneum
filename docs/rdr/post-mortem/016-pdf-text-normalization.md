# Post-Mortem: RDR-016 PDF Text Normalization

## RDR Summary

RDR-016 proposed replacing raw PDF text extraction with a quality-first approach
using PyMuPDF4LLM for markdown conversion with built-in whitespace normalization.
The plan recommended markdown as the default strategy (semantic structure for
better retrieval) with a normalization-only opt-in mode (maximum 47-48% token
savings). CLI flags `--normalize-only` and `--preserve-images` would control
extraction behavior.

## Implementation Status

Implemented

The core recommendation was fully implemented: PyMuPDF4LLM markdown conversion
is the default extraction strategy, normalization-only mode is available via
`--normalize-only`, and `--preserve-images` supports future multimodal use.
The implementation also includes several failure-mode mitigations and
architectural additions that the RDR did not anticipate.

---

## Implementation vs. Plan

### What Was Implemented as Planned

- **PyMuPDF4LLM as the markdown conversion engine**: The library was added as a
  dependency and is used via `pymupdf4llm.to_markdown()` for default extraction.
- **`_normalize_whitespace_edge_cases()` function**: Implemented exactly as
  specified in the RDR, handling tabs, Unicode whitespace characters
  (U+00A0, U+1680, U+2000-U+200A, U+202F, U+205F, U+3000), and 4+ consecutive
  newlines.
- **`_extract_with_markdown()` as the default extraction path**: Markdown
  conversion is the default when `markdown_conversion=True` (constructor
  default).
- **`_extract_with_pymupdf_normalized()` for normalization-only mode**: Raw
  PyMuPDF extraction with comprehensive whitespace normalization (multiple
  spaces collapsed, excessive newlines reduced, trailing whitespace removed,
  plus edge case handling).
- **`--normalize-only` CLI flag**: Implemented on both `arc index pdf` and
  `arc index text-pdf` commands.
- **`--preserve-images` CLI flag**: Implemented on `arc index pdf`, overrides
  `ignore_images` when set.
- **`ignore_images=True` default**: Images are skipped by default for
  performance, as planned.
- **`table_strategy="lines_strict"`**: Used in the `to_markdown()` call,
  matching the RDR recommendation.
- **`force_text=True`**: Used in the `to_markdown()` call, matching the RDR
  recommendation.
- **Unit tests for normalization**: Created at
  `tests/unit/indexing/pdf/test_normalization.py` covering tab conversion,
  Unicode whitespace, excessive newlines, leading/trailing whitespace, empty
  text, combined edge cases, mode configuration, and image flag behavior.
- **PyMuPDF >= 1.26.6 version requirement**: Matches the RDR specification.

### What Diverged from the Plan

- **PyMuPDF4LLM version bumped from 0.1.7 to 0.2.2**: The RDR specified
  `>=0.1.7`. The implementation uses `>=0.2.2` with a comment noting the
  update was needed "to fix table extraction bugs." This is a typical
  framework API detail where the specified version had issues discovered
  during implementation.

- **Strategy configured at construction time, not per-call**: The RDR planned
  `extract(pdf_path, normalize_only=False)` with the strategy selected per
  invocation. The implementation uses `markdown_conversion` as a constructor
  parameter on `PDFExtractor`, making the strategy a configuration of the
  extractor instance rather than a per-file decision. This simplifies the
  calling code in the uploader, which passes `markdown_conversion=not
  normalize_only` once at initialization.

- **Page-by-page extraction instead of whole-document**: The RDR showed
  `pymupdf4llm.to_markdown(str(pdf_path))` as a single whole-document call.
  The implementation extracts page-by-page using `pages=[page_num]` in a loop
  to track page boundaries (character offsets for each page). This enables
  page-number-aware chunking in the downstream `PDFChunker`, which calculates
  which page a chunk belongs to and stores `page_number` in chunk metadata.

- **No YAML configuration section**: The RDR planned a `pdf_extraction` section
  in `models.yaml` with keys like `markdown_conversion`, `ignore_images`,
  `table_strategy`, etc. This was never implemented. Configuration is handled
  entirely through CLI flags and constructor parameters, which proved
  sufficient.

- **Type3 font detection and fallback**: The RDR did not anticipate that
  PyMuPDF4LLM hangs indefinitely on PDFs with Type3 fonts (user-defined fonts
  drawn with PDF graphics commands). The implementation adds
  `_has_type3_fonts()` which scans all pages for Type3 font entries and falls
  back to normalized extraction when detected.

- **Font digest error handling**: The RDR did not anticipate RuntimeError
  exceptions from PyMuPDF4LLM's style analysis when encountering fonts without
  embedded data (Base-14 fonts, system fonts). The implementation catches
  `RuntimeError` with "font" or "code=4" in the message and falls back to
  normalized extraction.

### What Was Added Beyond the Plan

- **`pymupdf-layout` integration**: A new dependency (`pymupdf-layout>=0.1.0`)
  was added for enhanced page layout detection. The extractor optionally uses
  `pymupdf_layout.Layout` to analyze text blocks, headers, footers, sections,
  and column layouts. This feeds layout metadata into extraction results. The
  RDR did not mention this library.

- **Page boundary tracking**: Every extraction method tracks `page_boundaries`
  (a list of dicts with `page_number`, `start_char`, and `page_text_length`).
  The `PDFChunker._calculate_page_number()` method uses these boundaries to
  assign a `page_number` to each chunk based on its character position. This
  was not discussed in the RDR but is valuable for search result provenance.

- **`use_layout_analysis` constructor parameter**: Controls whether the
  optional `pymupdf-layout` analysis is used. Defaults to True when the
  library is available.

- **Memory cleanup**: `gc.collect()` is called after PDF extraction in the
  uploader pipeline, with a comment that "pymupdf4llm can hold large buffers."
  GPU cache clearing also occurs between PDFs. The RDR did not address memory
  management.

- **Normalize-only support in `arc index text-pdf`**: The `--normalize-only`
  flag was added to the full-text PDF indexing command in addition to the
  semantic PDF indexing command.

### What Was Planned but Not Implemented

- **`pdf_extraction` YAML config section**: The RDR specified a YAML
  configuration block with keys for `markdown_conversion`, `ignore_images`,
  `ignore_graphics`, `force_text`, `table_strategy`, and
  `detect_bg_color`. None of this was implemented; CLI flags and constructor
  defaults suffice.

- **Integration tests for retrieval quality**: The RDR specified a
  `test_retrieval_quality_with_markdown()` test that would create raw and
  markdown collections, run test queries, and compare MRR. No retrieval
  quality comparison tests exist.

- **Performance benchmark tests**: The RDR specified a
  `benchmark_extraction_strategies()` test comparing extraction time across
  raw, normalized, and markdown strategies. No performance benchmarks exist.

- **A/B quality validation with test corpus**: Phase 3 of the implementation
  plan called for creating a test corpus of 10-20 representative PDFs and
  running precision/recall comparisons. This was not done.

- **Migration guide for existing collections**: Phase 4 planned a migration
  runbook for re-indexing existing collections with the new extraction
  strategy. No migration documentation was created.

- **`marker-pdf` optional dependency**: The RDR proposed adding `marker-pdf`
  as an optional dependency under `[tool.poetry.group.ml.dependencies]` for
  advanced use cases. This was not added to `pyproject.toml`.

---

## Drift Classification

| Category | Count | Examples |
| -------- | ----- | -------- |
| **Unvalidated assumption** | 0 | |
| **Framework API detail** | 1 | PyMuPDF4LLM version bumped from 0.1.7 to 0.2.2 for table extraction bugs |
| **Missing failure mode** | 2 | Type3 font hangs; font digest RuntimeError (code=4) |
| **Missing Day 2 operation** | 1 | No migration guide for existing collections |
| **Deferred critical constraint** | 1 | Quality validation (A/B retrieval comparison) never executed |
| **Over-specified code** | 3 | `extract()` method signature rewritten; whole-doc extraction became page-by-page; YAML config section never needed |
| **Under-specified architecture** | 1 | Page boundary tracking needed for page-number-aware chunking |
| **Scope underestimation** | 1 | `pymupdf-layout` integration added as unanticipated enhancement |
| **Internal contradiction** | 0 | |
| **Missing cross-cutting concern** | 1 | Memory management after extraction (gc.collect, GPU cache clearing) |

### Drift Category Definitions

- **Unvalidated assumption** -- a claim presented as fact
  but never verified by spike/POC
- **Framework API detail** -- method signatures, interface
  contracts, or config syntax wrong
- **Missing failure mode** -- what breaks, what fails
  silently, recovery path not considered
- **Missing Day 2 operation** -- bootstrap, CI/CD,
  removal, rollback, migration not planned
- **Deferred critical constraint** -- downstream use case
  that validates the approach was out of scope
- **Over-specified code** -- implementation code that was
  substantially rewritten
- **Under-specified architecture** -- architectural
  decision that should have been made but wasn't
- **Scope underestimation** -- sub-feature that grew into
  its own major effort
- **Internal contradiction** -- research findings or stated
  principles conflicting with the proposal
- **Missing cross-cutting concern** -- versioning,
  licensing, config cache, deployment model, etc.

---

## RDR Quality Assessment

### What the RDR Got Right

- **Technology selection was accurate**: PyMuPDF4LLM was the correct choice
  over Marker, Markdrop, and pdf2markdown4llm. The trade-off analysis
  correctly identified that Marker and Markdrop were too heavy for the use
  case, and that PyMuPDF4LLM provided the best balance of quality, performance,
  and dependency weight. The implementation validated this choice.

- **Default strategy decision was sound**: Markdown-first with normalization-only
  as opt-in proved to be the right approach. The implementation follows this
  exactly, and the CLI ergonomics (markdown by default, `--normalize-only` flag
  for cost optimization) work well in practice.

- **Normalization function for edge cases**: The `_normalize_whitespace_edge_cases()`
  function was implemented almost verbatim from the RDR. The identification of
  tabs, Unicode whitespace, and 4+ newlines as the remaining edge cases after
  PyMuPDF4LLM's built-in normalization was correct.

- **CLI flag design**: The `--normalize-only` and `--preserve-images` flags
  were implemented as specified and provide clean user-facing controls for the
  two alternative modes.

- **Whitespace waste analysis**: The research finding of 47-48% whitespace
  waste in the Standards collection was cited throughout the implementation
  comments and informed the design rationale.

- **PyMuPDF4LLM parameter selection**: `ignore_images=True`,
  `force_text=True`, and `table_strategy="lines_strict"` were all used
  exactly as recommended.

### What the RDR Missed

- **PyMuPDF4LLM failure modes**: The RDR's "Known Limitations" section
  discussed multi-column layouts, table detection, graphics-heavy documents,
  header detection, and scanned documents -- but missed the two failure modes
  that actually required code: Type3 fonts causing indefinite hangs and font
  digest RuntimeErrors (code=4). These required defensive checks and graceful
  fallback paths that were not anticipated.

- **Page boundary tracking**: The RDR focused entirely on extraction quality
  and token efficiency but did not consider that page-number metadata is
  valuable for search results. The implementation needed page-by-page
  extraction to track character offsets and map chunks to source pages.

- **Memory management**: The RDR did not discuss memory implications of
  PyMuPDF4LLM extraction. In practice, the library can hold large buffers
  that need explicit cleanup via `gc.collect()`, especially when processing
  many PDFs sequentially.

- **Layout analysis as a useful enhancement**: The RDR evaluated five
  PDF-to-markdown libraries but did not consider `pymupdf-layout` as a
  complementary tool for structure detection.

### What the RDR Over-specified

- **Implementation code samples that were rewritten**: The `extract()` method
  signature used `normalize_only` as a call-time parameter, but the
  implementation moved strategy selection to the constructor. The whole-document
  extraction code was replaced with page-by-page extraction. These code samples
  created false confidence in the implementation approach.

- **YAML configuration section**: The `pdf_extraction` config block with six
  keys was detailed but never needed. CLI flags and constructor defaults proved
  sufficient for all use cases.

- **Extensive alternative analysis**: The RDR devoted significant space to
  Marker, Markdrop, pdf2markdown4llm, and textacy with feature tables,
  trade-off matrices, and code examples. None of these alternatives were
  implemented or even added as optional dependencies. While the analysis
  justified the PyMuPDF4LLM selection, the depth was disproportionate to its
  impact.

- **Integration and performance test code**: The RDR included detailed test
  code for retrieval quality comparison (MRR calculation) and performance
  benchmarks. Neither was implemented, and the test code was not directly
  usable.

- **Future enhancement code examples**: Adaptive strategy selection,
  per-collection defaults, and quality monitoring dashboard included code
  examples for features explicitly deferred to future quarters.

---

## Key Takeaways for RDR Process Improvement

1. **Test the library against real documents before specifying failure modes**:
   The RDR listed five known limitations from documentation and changelogs but
   missed the two failure modes (Type3 fonts, font digest errors) that actually
   required fallback code. A brief spike running PyMuPDF4LLM against 5-10
   representative PDFs from the target corpus would have surfaced these issues
   and produced a more accurate risk assessment.

2. **Specify configuration boundaries, not configuration mechanisms**: The RDR
   specified a YAML config schema that was never needed. Instead of prescribing
   where settings live (YAML file, CLI flags, constructor params), specify
   which settings are configurable and their valid ranges. Let the
   implementation decide the most natural configuration mechanism.

3. **Design extraction interfaces around downstream data needs**: The RDR
   designed the extraction API around input parameters (`normalize_only`) but
   did not consider what the downstream chunker needed (page boundaries).
   This forced an architectural change (page-by-page extraction) that
   propagated through the chunker. When specifying an extraction interface,
   enumerate what consumers of the output need, not just what the extractor
   produces.

4. **Limit code samples to architectural patterns, not method-level
   implementations**: Three of the RDR's code samples were substantially
   rewritten -- the method signature changed, the extraction loop changed,
   and the configuration mechanism changed. Code samples in RDRs should
   illustrate architectural decisions (e.g., "strategy pattern with fallback
   chain") rather than production-ready implementations that create false
   specificity.

5. **Allocate RDR space proportionally to implementation risk**: The RDR spent
   roughly 40% of its content on alternatives that were never implemented
   (Marker, Markdrop, pdf2markdown4llm, textacy) and future enhancements. The
   failure modes that required actual fallback code received zero coverage.
   Weight research sections toward risks in the chosen approach rather than
   exhaustive analysis of rejected alternatives.

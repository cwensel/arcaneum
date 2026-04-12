# Recommendation 022: Advanced PDF Extraction for Complex Technical Documents

> Revise during planning; lock at implementation.
> If wrong, abandon code and iterate RDR.

## Metadata

- **Date**: 2026-04-12
- **Status**: Recommendation
- **Type**: Enhancement
- **Priority**: High
- **Related Issues**: None
- **Related RDRs**: RDR-004 (PDF Bulk Indexing), RDR-016 (PDF Text Normalization)

## Problem Statement

The current PDF extraction pipeline (PyMuPDF4LLM) fails on dual-column academic and
technical documents. Specifically:

1. **Cross-column interleaving**: The extractor alternates between left and right columns
   instead of reading each column fully, producing sentences that splice content from
   unrelated columns
2. **Cross-article contamination**: Article headers and page numbers from adjacent
   columns/pages are injected mid-sentence (e.g., "study the dynamic **A** Algorithm
   DC-Tree for _k_ Servers on Trees 9 model's equilibrium")
3. **Bracket-fragmented text**: Individual words near column boundaries or mathematical
   formulas are captured as separate bracketed tokens
   (`[whose][property][is][that]`)
4. **Poor mathematical notation**: PyMuPDF4LLM scores 6.67/10 on formula extraction
   benchmarks, well below leaders

These issues degrade both search quality (garbled chunks produce poor embeddings and
misleading full-text matches) and downstream LLM consumption of retrieved context.

### Observed Evidence

Tested against Springer-Verlag Encyclopedia of Algorithms (dual-column, 1000+ pages,
heavy math/pseudocode). Specific violations documented:

- **Chunk 63, page 57 (1-indexed)**: Pseudocode steps 8-9 (right column) interleaved
  with Definitions 4-12 (left column), then steps 10-12 (right column)
- **Chunk 64, page 57 (1-indexed)**: Left-column sentence "study the dynamic"
  concatenated with right-column next-article header "Algorithm DC-Tree for k
  Servers on Trees"
- **Chunk page 146 (1-indexed)**: Bracket-fragmented text
  `[whose][property][is][that]` from column-boundary extraction failure

### Scope

This RDR addresses two concerns:

1. **Detection and routing**: Automatically detect when a PDF requires advanced
   extraction (column layout, math density) during indexing, and route to the
   appropriate extractor
2. **Selective re-indexing**: Re-index an existing corpus using the new extractor,
   but only for documents that were parsed with the old extractor and would benefit
   from the upgrade

## Context

### Background

RDR-016 evaluated Marker as an alternative and deferred it as "overkill for most
technical documentation" with a note: "Add as opt-in for specialized corpora
(scientific papers with equations)." RDR-016 also documented the known limitation:
"Complex multi-column layouts: Reading order detection can fail on overly complex
page layouts."

The Springer Encyclopedia test confirms this limitation is not theoretical -- it
produces unusable chunks for a significant class of technical reference material.

The existing pipeline already tracks `extraction_method` in MeiliSearch as a
filterable attribute. The `corpus sync` command supports `--force` (bypass change
detection) and the separate `corpus repair` command handles re-indexing of
garbled or incomplete files. This infrastructure can be extended for selective
re-indexing by extraction method.

### Technical Environment

- **Python**: >= 3.12
- **Current PDF Stack**: PyMuPDF + pymupdf4llm (default), pdfplumber (fallback)
- **Indexing**: Dual-index to Qdrant (semantic) + MeiliSearch (full-text)
- **Metadata**: `extraction_method` stored in MeiliSearch (filterable),
  Qdrant payload
- **Sync**: Metadata-based change detection via quick_hash (mtime+size)
  and file_hash (content hash)
- **Quality**: `quality.py` scores text readability (replacement chars,
  stop words, ASCII ratio, word length)

## Research Findings

### Investigation

Comprehensive web research evaluated 10+ PDF extraction tools against the specific
requirements of dual-column academic documents with mathematical content.

#### Formula Extraction Benchmark (arXiv 2512.09874, Dec 2025)

| Tool | Score (0-10) | Notes |
| --- | --- | --- |
| Gemini 3 Pro | 9.75 | Commercial API |
| Mathpix | 9.64 | Commercial API, $0.005/page |
| MinerU | ~9.0 (est.) | Open source, AGPL-3.0 |
| Marker | ~8.5 (est.) | Open source, GPL-3.0 |
| Mistral OCR | ~8.5-9.0 | Commercial API |
| olmOCR | 82.3% accuracy | Open source, Apache 2.0 |
| PyMuPDF4LLM | 6.67 | Current tool |
| GROBID | 5.70 | Java-based, weak on math |

#### Dependency Source Verification

| Dependency | Source Searched? | Key Findings |
| --- | --- | --- |
| MinerU | Yes (Spike) | AGPL-3.0; PaddlePaddle removed in 3.x; uses PyTorch port of PaddleOCR internally; Python 3.10-3.13 |
| Marker | Yes (Spike) | GPL-3.0; `pip install marker-pdf`; Python ^3.10 (no upper bound); surya-ocr dependency |
| Nougat | No | GitHub README reviewed; MIT license, trained on arXiv, can hallucinate |
| Docling | No | GitHub README reviewed; MIT license, weaker formula support |
| pymupdf4llm | Yes (RDR-016) | Column detection known limitation documented in source |

### Key Discoveries

#### Candidate Tools

**MinerU (PDF-Extract-Kit) v3.0.9** — [Verified via spike]

- LayoutLMv3 for column detection (77.6% mAP on academic papers)
- YOLOv8 + UniMERNet for formula detection and LaTeX conversion
- 0.21 sec/page on Nvidia L4 GPU (fastest ML-based);
  ~142s/page on CPU/Apple Silicon (spike-measured, includes model load)
- 86.2 on OmniDocBench (CVPR 2025)
- Python SDK: `pip install "mineru[pipeline]"` (Python >=3.10, <3.14)
- Output: Markdown and JSON
- License: AGPL-3.0 (viral copyleft -- potential concern)
- Dependencies: torch, transformers, onnxruntime, opencv-python.
  PaddlePaddle removed in 3.x; uses PyTorch port of PaddleOCR internally

**Marker v1.10.2** — [Verified via spike]

- Four-stage pipeline: text detection, recognition, layout analysis (Surya),
  reading order prediction
- Texify model for equation-to-LaTeX conversion
- ~25 pages/sec on H100 batch; ~242s/page on CPU/Apple Silicon
  (spike-measured, includes model load)
- `pip install marker-pdf` (Python ^3.10, no upper bound)
- Output: Markdown, JSON, HTML, chunks
- License: GPL-3.0
- Downloads ~2GB models on first run

**Nougat (Meta AI)** — [Documented]

- Vision Transformer (Swin + mBART decoder), processes pages as images
- Trained on 8M+ arXiv and PubMed Central pages
- Outputs LaTeX directly, excellent for math
- License: MIT
- Can hallucinate on unfamiliar layouts; repetition artifacts
- GPU required; ~4x slower than Marker

**Docling (IBM)** — [Documented]

- DocLayNet for layout analysis, good reading order
- MIT license, 20K+ GitHub stars
- Formula support less mature than competitors
- `pip install docling` (Python 3.10+)

#### Column Detection Approaches

- **Rule-based** (PyMuPDF4LLM): Uses text block coordinates to infer columns.
  Fails on complex layouts with mixed-width content, figures spanning columns,
  and algorithm pseudocode boxes
- **ML-based** (MinerU, Marker, Docling): Neural layout detection models trained
  on academic papers. Significantly better at identifying column boundaries,
  reading order, and content type classification
- **Vision-based** (Nougat): Bypasses column detection entirely by treating pages
  as images and reconstructing text via OCR. Highest potential quality but
  slowest and most resource-intensive

#### Detection Heuristics for Routing — [Verified via spike]

Auto-detection spiked on 17 PDFs (1 dual-column encyclopedia, 1 newspaper,
15 single-column books). **94% accuracy** (16/17 correct).

**Primary heuristic — page geometry analysis** (spiked, verified):

1. Open PDF with PyMuPDF, sample 6 pages spread through the document
2. Extract text block bounding boxes via `page.get_text("dict")`
3. Filter to significant blocks (>15% page width, >15px height)
4. Group blocks by page center into left/right groups
5. Classify as dual-column when:
   - Both groups have 2+ blocks
   - Gap between groups is 1-15% of page width
   - Each column width is 25-50% of page width
   - Column width ratio > 0.7 (similar widths, not sidebar + main)
6. Majority vote across sampled pages

**Results**:

- Springer Encyclopedia: correctly detected (5/6 pages, gap ~2%, widths ~40%)
- All 15 single-column books: correctly rejected (zero false positives)
- Head First Design Patterns (mixed sidebar layout): correctly rejected
  (gap 10-30% and width <25% filter catches sidebar layouts)
- Newspaper: missed (variable-width columns, not target use case)

**Supplementary heuristics** (not yet spiked, lower priority):

- Content type signals: math symbol density, bracket-fragmented text
- Post-extraction quality scoring for column-interleaving artifacts
- PDF producer metadata patterns (Springer, IEEE, ACM templates)

### Critical Assumptions

- [x] MinerU or Marker can be installed alongside existing PyMuPDF without
  dependency conflicts — **Status**: Verified — **Method**: Spike
  (both installed cleanly in isolated venvs with Python 3.12; MinerU 3.0.9
  removed PaddlePaddle framework dep, uses PyTorch port of PaddleOCR
  internally)
- [x] Advanced extractor produces correct column reading order for the
  Springer Encyclopedia test case — **Status**: Verified — **Method**: Spike
  (both MinerU and Marker pass all 5 validation checks; see Spike Results)
- [x] MeiliSearch filter on `extraction_method` returns all documents
  extracted with a specific method for selective re-indexing —
  **Status**: Verified — **Method**: Source Search (filterable attribute
  confirmed in `fulltext/indexes.py`)
- [x] Performance is acceptable for batch re-indexing (target: < 2 sec/page)
  — **Status**: Verified — **Method**: Spike (MinerU on MPS: 1.1 s/page
  steady state with formulas at 20-page batches. 1000 pages in ~18 min.
  CPU-only is 142 s/page — not viable for large corpora; MPS/CUDA required.)
- [x] AGPL-3.0 (MinerU) license is acceptable for this project as an
  optional dependency — **Status**: Verified — **Method**: Decision
  (confirmed by project owner 2026-04-12)

### Spike Results (2026-04-12)

Tested both MinerU 3.0.9 and Marker 1.10.2 on Springer-Verlag Encyclopedia
of Algorithms (dual-column, dense math, pseudocode, theorems).
Test environment: macOS, Python 3.12, Apple M2 Pro (19-core GPU, Metal 4).

#### Test Protocol

Extracted pages 55-56 (0-indexed) and validated against 5 checks:

1. Column separation (Definitions not interleaved with pseudocode)
2. No cross-article contamination ("dynamic model" sentence intact)
3. No bracket fragmentation
4. Forward-looking Nash equilibrium sentence intact
5. Theorem 13 text intact

#### MinerU Results

- **Validation**: 5/5 PASS
- **Quality**: Excellent column separation. Left column content
  (Definitions 4-12) fully contiguous. Right column (pseudocode steps 1-12)
  separate. "Algorithm DC-Tree" article starts cleanly without contamination.
- **Math quality**: LaTeX output for formulas (`$\forall i \in \mathcal{N}$`).
  Equations properly formatted with `$$...$$` blocks and `\tag{N}` numbering.
- **Minor issues**: Some inline text uses "8e 2 E" instead of proper
  `$\forall \mathbf{e} \in \mathcal{E}$` (mixed plain text and LaTeX)
- **Dependencies**: torch, transformers, onnxruntime, opencv-python.
  PaddlePaddle framework removed in 3.x; OCR uses a PyTorch port of
  PaddleOCR models internally (no paddle pip packages). AGPL-3.0 license.
- **MPS/GPU support**: Yes, auto-detects MPS on Apple Silicon.
  Also supports `MINERU_DEVICE_MODE` env var override.
- **Speed tuning**: `formula_enable=False` and `table_enable=False`
  can be set per-document for faster extraction when not needed.

#### MinerU MPS Performance (Apple M2 Pro, 3 runs per config)

| Config | Run 1 | Run 2 | Run 3 | Avg s/page |
| --- | --- | --- | --- | --- |
| 2 pages, no formula | 9.9 s/pg | 1.1 s/pg | 1.1 s/pg | 4.0 |
| 2 pages, formula ON | 4.2 s/pg | 3.3 s/pg | 3.4 s/pg | 3.6 |
| 10 pages, formula ON | 2.8 s/pg | 1.9 s/pg | 1.9 s/pg | 2.2 |
| 20 pages, formula ON | 1.8 s/pg | 1.1 s/pg | 1.1 s/pg | 1.3 |

Run 1 includes model warm-up overhead. Steady-state (runs 2-3):

- **No formula**: ~1.1 s/page
- **With formula, 10+ page batch**: ~1.1-1.9 s/page
- **1000 pages at 1.1 s/pg**: ~18 minutes total

#### Marker Results

- **Validation**: 5/5 PASS (one false-negative in test script due to
  case-sensitive string matching; manual verification confirmed pass)
- **Quality**: Excellent column separation. Definitions 4-12
  contiguous. Pseudocode rendered in code block. Clean article boundaries.
- **Math quality**: Superior LaTeX — proper calligraphic letters
  (`$\mathcal{E}$`), bold vectors (`$\hat{\mathbf{b}}$`),
  piecewise functions with `\begin{cases}`.
- **Minor issues**: Page number + running header leaked into text
  ("8 Adwords Pricing" between Definition 2 and its continuation, line 55)
- **Dependencies**: torch, transformers, surya-ocr, pdftext. GPL-3.0 license.
- **MPS/GPU support**: Partial — layout uses MPS, but text detection
  falls back to CPU ("MPS device does not work for text detection").
  Table recognition also falls back to CPU.

#### Marker MPS Performance (Apple M2 Pro)

| Config | Run 1 |
| --- | --- |
| 2 pages | 95.0 s/pg |

Marker test was terminated after first run — 95 s/page on MPS is
~86x slower than MinerU steady-state (1.1 s/page). The bottleneck
is text recognition, which falls back to CPU even when MPS is
available. Full variability testing was not warranted given this gap.

At 95 s/page, a 1000-page document would take ~26 hours.

#### Comparison Summary

| Aspect | MinerU 3.0.9 | Marker 1.10.2 |
| --- | --- | --- |
| Column separation | Excellent | Excellent |
| Math/LaTeX quality | Good (mixed plain/LaTeX) | Superior (consistent LaTeX) |
| Pseudocode | Plain text, readable | Code block formatting |
| Running header leaks | None observed | Minor (1 instance) |
| MPS speed (steady state) | **1.1 s/page** | 95 s/page |
| MPS utilization | Full (layout + OCR) | Partial (layout only) |
| 1000 pages estimate | **~18 min** | ~26 hours |
| License | AGPL-3.0 | GPL-3.0 |
| Python version | >=3.10, <3.14 | ^3.10 (no upper bound) |
| Install size | ~2.5GB (models) | ~2GB (models) |

#### Spike Conclusion

MinerU is the clear winner. Both solve the cross-column problem with
equal quality, but MinerU is **~86x faster** on Apple Silicon MPS due
to full GPU acceleration. At 1.1 s/page steady state, a 1000-page
document completes in ~18 minutes — practical for batch re-indexing.

Marker's superior LaTeX quality does not justify the 86x speed
penalty, especially since the extracted text is used for search
indexing (not LaTeX rendering). MinerU's "good enough" LaTeX is
fully adequate for embedding and full-text search.

**Recommendation**: MinerU as the sole advanced extractor. Marker
is not competitive on performance and should not be integrated.
MinerU's `formula_enable` and `table_enable` flags provide a
further speed/quality tradeoff per document when needed.

## Proposed Solution

### Approach

Add an advanced PDF extraction backend as an optional dependency, with automatic
detection of documents that need it, and a selective re-indexing mechanism for
existing corpora.

The solution has three components:

1. **Extractor backend**: Integrate MinerU as an alternative to PyMuPDF4LLM
   for complex documents
2. **Auto-detection during indexing**: Detect multi-column layouts and math-heavy
   content before or after initial extraction, routing to the appropriate backend
3. **Selective re-indexing**: Query existing corpus metadata to identify documents
   extracted with the old method, and re-index only those with the new extractor

### Technical Design

#### Component 1: Advanced Extractor Backend

Extend `PDFExtractor` with a new extraction method that delegates to the chosen
ML-based tool. The new method should:

- Accept a PDF path and return `(text, metadata)` in the same format as existing
  methods
- Set `extraction_method` to identify the backend used (e.g., `marker_markdown`
  or `mineru_markdown`)
- Produce Markdown output compatible with the existing chunking pipeline
- Handle GPU availability gracefully (fall back to CPU)

```text
// Illustrative — verify API signatures during implementation
PDFExtractor.extract(path, method='auto')
  → if method == 'mineru': always use MinerU
  → if method == 'auto': run detection → pick backend
  → if method == 'default': always use pymupdf4llm (current behavior)
```

The `--advanced-pdf` flag on `corpus sync` controls which mode is used:

```text
arc corpus sync DevRef /path --advanced-pdf auto   # detect per-file (default when MinerU installed)
arc corpus sync DevRef /path --advanced-pdf on      # force MinerU for all PDFs
arc corpus sync DevRef /path --advanced-pdf off     # disable, use pymupdf4llm only
arc corpus sync DevRef /path                        # 'off' when MinerU not installed, 'auto' when installed
```

#### Component 2: Auto-Detection

Two-phase detection strategy:

**Phase A — Pre-extraction geometry check** (fast, no ML):

Uses the page geometry heuristic validated in the spike (see Research Findings >
Detection Heuristics). If multi-column detected → use advanced extractor.

**Phase B — Post-extraction quality check** (fallback):

- Extract with PyMuPDF4LLM as normal
- Run quality scoring on extracted text
- Check for column-interleaving artifacts:
  - Orphaned single-letter section headers mid-paragraph (e.g., `**A**` between
    sentences)
  - Bracket-fragmented text patterns (`[word][word][word]`)
  - Abrupt page-number insertions mid-sentence
- If artifacts detected → re-extract with advanced extractor

Detection results should be cached per-file (via file_hash) to avoid repeated
geometry analysis on re-runs.

#### Component 3: Selective Re-Indexing via `corpus repair`

Extend the existing `corpus repair` command with an additional detection
criterion. `repair` already scans indexed chunks, identifies files needing
re-extraction (via quality scoring), and re-indexes just those files. The
upgrade-extractor case is the same pattern with a different detection
signal: `extraction_method` metadata instead of quality score.

When MinerU is installed and `corpus repair` runs:

1. In addition to quality scoring, query MeiliSearch for documents with
   `extraction_method = "pymupdf4llm_markdown"`
2. For each such file, run auto-detection (Phase A geometry check)
3. If the file would benefit from advanced extraction (multi-column,
   math-heavy), flag it for re-extraction alongside garbled files
4. Re-extract with MinerU, delete old chunks, index new ones
5. Files that pass detection as single-column/non-math are skipped

This piggybacks on existing infrastructure:

- `repair` already has `--dry-run`, `--verbose`, `--no-gpu` flags
- `repair` already handles atomic file-level re-indexing
- `repair` already uses `CollectionVerifier` for scanning
- `extraction_method` is already filterable in MeiliSearch

```text
// Illustrative CLI usage — same command, new detection criterion
arc corpus repair DevRef --dry-run
arc corpus repair DevRef --verbose
```

No new CLI flag needed. When MinerU is installed, `repair` automatically
detects documents that would benefit from re-extraction. When MinerU is
not installed, `repair` behaves exactly as before (quality scoring only).

### Existing Infrastructure Audit

| Proposed Component | Existing Module | Decision |
| --- | --- | --- |
| Advanced extractor backend | `indexing/pdf/extractor.py` | Extend: add new extraction method alongside existing ones |
| Auto-detection | `indexing/pdf/quality.py` | Extend: add column-detection and artifact-detection functions |
| Selective re-indexing | `cli/sync.py` (`corpus repair`), `indexing/verify.py` | Extend: add extraction-method detection to existing repair flow |
| Chunk deletion by file | `indexing/common/sync.py` | Reuse: `delete_chunks_by_file_hash()` already exists |
| MeiliSearch method filter | `fulltext/indexes.py` | Reuse: `extraction_method` already filterable |

### Decision Rationale

Extending the existing extractor with an optional ML backend is preferred over
replacing PyMuPDF4LLM because:

1. PyMuPDF4LLM is adequate for single-column documents (the majority of the corpus)
   and is much faster with no GPU requirement
2. ML-based extractors have heavy dependencies that not all users need
3. The auto-detection approach means users don't need to manually classify documents
4. Selective re-indexing respects the existing metadata infrastructure and avoids
   wasteful full-corpus rebuilds

## Alternatives Considered

### Alternative 1: Replace PyMuPDF4LLM Entirely with Marker/MinerU

**Description**: Use the ML-based extractor for all PDFs

**Pros**:

- Simpler code path (one extractor)
- Consistently high quality

**Cons**:

- Heavy dependencies for all users (PyTorch + ML models, ~2GB)
- 10-50x slower for simple documents that don't need ML
- GPU strongly recommended
- License restrictions (both GPL/AGPL) applied to all usage
- ~142-242s/page on CPU vs <1s/page for PyMuPDF4LLM (spike-measured)

**Reason for rejection**: Disproportionate resource cost for the majority of
documents that extract correctly with PyMuPDF4LLM

### Alternative 2: Commercial API (Mathpix / Mistral OCR)

**Description**: Use a cloud API for complex documents

**Pros**:

- Highest quality (9.64/10 for Mathpix)
- No local GPU or heavy dependencies
- Simple integration (REST API)

**Cons**:

- Per-page cost ($0.005/page = $5 per 1000 pages)
- Requires internet connectivity
- Data leaves local machine (privacy concern for some corpora)
- Vendor dependency

**Reason for rejection**: Cost and privacy concerns for large corpora.
Could be offered as an additional option but not the default.

### Alternative 3: Vision-Based (Nougat)

**Description**: Use Meta's Nougat to treat pages as images

**Pros**:

- MIT license
- Excellent math quality (trained on arXiv)
- Bypasses column detection entirely

**Cons**:

- Can hallucinate on unfamiliar layouts
- ~4x slower than Marker
- GPU required
- Trained primarily on arXiv format

**Reason for rejection**: Hallucination risk on non-arXiv documents
(like the Springer Encyclopedia) makes it unreliable for general use

### Briefly Rejected

- **GROBID**: Java-based, 5.70/10 on math — worse than current tool
- **olmOCR**: Requires 7B VLM, GPU required, mid-range math accuracy
- **OpenDataLoader PDF**: Heuristic column detection, unproven on academic math

## Trade-offs

### Consequences

- Positive: Complex technical documents produce correct, usable chunks
- Positive: Existing simple documents unaffected (no regression risk)
- Positive: Selective re-indexing avoids full corpus rebuild cost
- Negative: Optional heavy dependency (PyTorch + ML models, ~2GB)
- Negative: Longer extraction time for complex documents (seconds vs milliseconds)
- Negative: AGPL-3.0 license for the advanced extractor (accepted as optional
  dependency; see Critical Assumptions)

### Risks and Mitigations

- **Risk**: ML extractor dependency conflicts with existing packages
  **Mitigation**: Install as optional dependency (`pip install arcaneum[advanced-pdf]`);
  spike dependency resolution before implementation

- **Risk**: Auto-detection false positives (simple docs routed to slow extractor)
  **Mitigation**: Conservative detection thresholds; manual override via
  `--advanced-pdf off`

- **Risk**: Auto-detection false negatives (complex docs not detected)
  **Mitigation**: Post-extraction quality check as fallback; `--advanced-pdf on`
  to force MinerU for all PDFs in a corpus

- **Risk**: Advanced extractor produces different chunk boundaries than PyMuPDF4LLM,
  breaking existing search patterns
  **Mitigation**: Both produce Markdown; chunking pipeline is shared and
  format-agnostic

### Failure Modes

- **Advanced extractor not installed**: Falls back to PyMuPDF4LLM with a warning
  log. No crash. Detection phase skipped.
- **GPU not available**: MinerU falls back to CPU (~142 s/page vs 1.1 s/page
  on MPS). Log warning. CPU-only is not viable for large corpora but works
  for small re-indexing jobs.
- **Selective re-indexing interrupted**: Each file is atomic (delete old + index new).
  Interrupted files can be detected by missing chunks and re-processed on next run.
- **Detection disagrees with reality**: Manual override available via
  `--advanced-pdf on` (force MinerU) or `--advanced-pdf off` (force default)

## Implementation Plan

### Prerequisites

- [x] All Critical Assumptions verified (see Critical Assumptions section)
- [x] License decision: MinerU (AGPL-3.0) accepted as optional dependency
  (confirmed by project owner 2026-04-12)
- [x] Test extraction of Springer Encyclopedia with MinerU and Marker
  (see Spike Results)

### Minimum Viable Validation

Extract the Springer-Verlag Encyclopedia of Algorithms pages 56-57 (1-indexed)
with the chosen advanced extractor and verify:

1. Definitions 4-12 appear contiguously (left column not interleaved with
   right column pseudocode)
2. The sentence "Cary et al. also study the dynamic model's equilibrium" is
   intact without injected article headers
3. Mathematical notation is preserved (LaTeX or clean Unicode, no bracket
   fragmentation)

Then index the extracted pages and verify full-text search returns exact
sentence matches.

### Phase 1: Spike — Tool Selection and Validation (Complete)

Completed 2026-04-12. MinerU selected as sole advanced extractor.
See Spike Results and Critical Assumptions sections for details.

### Phase 2: Code Implementation

#### Step 1: Advanced Extractor Backend

Add new extraction method to `PDFExtractor` in `extractor.py`.
Wire up as optional import with graceful fallback.

#### Step 2: Auto-Detection

Add column detection and math density analysis to `quality.py` or
a new `detection.py` module. Integrate into extraction flow.

#### Step 3: Selective Re-Indexing

Extend `CollectionVerifier` to detect documents with old extraction
methods when MinerU is available. Add extraction-method criterion
to `corpus repair` flow alongside existing quality scoring.

#### Step 4: CLI and Configuration

Add `--advanced-pdf` option to `corpus sync` with values `auto`, `on`,
`off`. Default: `auto` when MinerU is installed, `off` otherwise.
If `on` or `auto` is passed without MinerU installed, error with:
`Error: --advanced-pdf requires mineru. Install with: pip install arcaneum[advanced-pdf]`

Add optional dependency group to `pyproject.toml`:

```toml
[project.optional-dependencies]
advanced-pdf = ["mineru[pipeline]>=3.0.9"]
```

Uses the same conditional-import pattern as `HAS_PYMUPDF_LAYOUT` in
`extractor.py` and `[ocr]` extras for easyocr.

#### Step 5: Install Documentation

Update README.md and quickstart guide to document all optional extras.
Currently undocumented extras: `[ocr]` (easyocr), `[dev]` (test tools).
This step adds `[advanced-pdf]` and documents all three:

```bash
# Base install
pipx install arcaneum

# With advanced PDF extraction (multi-column, math-heavy documents)
pipx install "arcaneum[advanced-pdf]"

# With OCR support (scanned documents)
pipx install "arcaneum[ocr]"

# All optional features
pipx install "arcaneum[advanced-pdf,ocr]"
```

### Phase 3: Operational Activation

#### Activation Step 1: Re-Index Test Corpus

Run `arc corpus repair DevRef --dry-run` to identify documents that would
benefit from advanced extraction. Then run without `--dry-run`.

### Day 2 Operations

| Resource | List | Info | Delete | Verify | Backup |
| --- | --- | --- | --- | --- | --- |
| Advanced extractor config | In scope | In scope | N/A | In scope (detection test) | N/A |
| Re-indexed chunks | Reuse existing | Reuse existing | Reuse existing | In scope (search validation) | Reuse existing |

### New Dependencies

| Dependency | Version | License | Purpose |
| --- | --- | --- | --- |
| mineru[pipeline] | >= 3.0.9 | AGPL-3.0 | ML-based PDF extraction |

MinerU is the sole advanced extractor (see Spike Conclusion for selection
rationale).

## Validation

### Testing Strategy

1. **Scenario**: Dual-column PDF extraction (Springer Encyclopedia pp. 56-57, 1-indexed)
   **Expected**: Left and right columns extracted separately with correct
   reading order; no interleaved content

2. **Scenario**: Mathematical formula preservation
   **Expected**: Formulas rendered as LaTeX or clean Unicode, no bracket
   fragmentation

3. **Scenario**: Auto-detection of multi-column layout
   **Expected**: Springer Encyclopedia detected as needing advanced extractor;
   single-column PDFs detected as not needing it

4. **Scenario**: Selective re-indexing via `corpus repair`
   **Expected**: When MinerU is installed, `repair` detects documents with
   old `extraction_method` that need advanced extraction; already-upgraded
   documents are skipped; behavior unchanged when MinerU is not installed

5. **Scenario**: Advanced extractor not installed
   **Expected**: Graceful fallback to PyMuPDF4LLM with warning log

### Performance Expectations

See MinerU MPS Performance table in Spike Results for detailed benchmarks.
Acceptable for production use because:

- Only a subset of documents will be routed to the advanced extractor
- Indexing is a batch operation, not latency-sensitive
- MPS/CUDA acceleration makes large corpora practical
- CPU-only fallback works for small jobs but is much slower
- `formula_enable=False` available when math extraction not needed

## Finalization Gate

> Complete each item with a written response before
> marking this RDR as **Final**.

### Contradiction Check

No contradictions found between spike results and proposed design.
MinerU spike validates all claims about column separation quality
and MPS performance. Marker rejected on performance, not quality.

### Assumption Verification

All five Critical Assumptions verified via spike. See Critical
Assumptions section for individual verification status and methods.

#### API Verification

| API Call | Library | Verification |
| --- | --- | --- |
| mineru extract or equivalent | mineru | Spike (verified 2026-04-12) |
| MeiliSearch filter by extraction_method | meilisearch | Source Search (verified) |
| delete_chunks_by_file_hash | arcaneum sync | Source Search (verified) |

### Scope Verification

MVV confirmed in scope. The spike validated extraction of Springer
Encyclopedia pages with MinerU, which directly tests the MVV criteria
(column separation, sentence integrity, math preservation).

### Cross-Cutting Concerns

- **Versioning**: extraction_method metadata tracks which version/tool extracted
  each document
- **Build tool compatibility**: Optional dependency group (`[advanced-pdf]`)
- **Licensing**: See Critical Assumptions for AGPL-3.0 acceptance decision
- **Deployment model**: Local CLI tool; no server-side implications
- **IDE compatibility**: N/A
- **Incremental adoption**: Auto-detection means zero config change for existing
  users; advanced extractor only activates when installed
- **Secret/credential lifecycle**: N/A
- **Memory management**: ML models require ~2GB RAM/VRAM; extraction is
  per-document so memory is bounded

### Proportionality

This RDR is appropriately sized for the scope. The spike phase (Phase 1) is
complete and confirmed MinerU as the sole advanced extractor. The implementation
sections are ready for Phase 2.

## References

- [RDR-004: Bulk PDF Indexing with OCR Support](RDR-004-pdf-bulk-indexing.md)
- [RDR-016: PDF Text Normalization and Markdown Conversion](RDR-016-pdf-text-normalization.md)
- [Benchmarking Document Parsers on Mathematical Formula Extraction (arXiv 2512.09874)](https://arxiv.org/html/2512.09874v1)
- [MinerU GitHub](https://github.com/opendatalab/MinerU)
- [Marker GitHub](https://github.com/datalab-to/marker)
- [Nougat GitHub (Meta AI)](https://github.com/facebookresearch/nougat)
- [Docling GitHub](https://github.com/docling-project/docling)
- [OmniDocBench (CVPR 2025)](https://github.com/opendatalab/OmniDocBench)
- [PyMuPDF4LLM column detection limitation](RDR-016-pdf-text-normalization.md#known-limitations)

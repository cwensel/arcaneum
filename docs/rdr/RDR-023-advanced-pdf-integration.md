# Recommendation 023: Advanced PDF Extraction Integration Strategy

> Revise during planning; lock at implementation.
> If wrong, abandon code and iterate RDR.

## Metadata

- **Date**: 2026-04-12
- **Status**: Recommendation
- **Type**: Enhancement
- **Priority**: High
- **Related Issues**: None
- **Related RDRs**: RDR-022 (Advanced PDF Extraction — spike results and tool
  selection), RDR-016 (PDF Text Normalization)

## Problem Statement

RDR-022 validated MinerU as the advanced PDF extractor for dual-column and
math-heavy documents (5/5 validation checks, 1.1 s/page on MPS). Implementation
was abandoned because MinerU 3.0.9 cannot be installed alongside arcaneum's
existing dependencies:

- **MinerU 3.0.9** (`[pipeline]` extra) requires `transformers>=4.57.3`
- **Arcaneum** pins `transformers<4.54.0` because the Stella embedding model
  (`dunzhang/stella_en_1.5B_v5`) breaks with transformers 4.54+

The Stella model's custom `modeling_qwen.py` (hosted on HuggingFace) calls
`DynamicCache.get_usable_length()` in 4 places (lines 279, 383, 681, 1003 of
the upstream file), a method removed in transformers 4.54.0 (PR #39106). The
upstream issue (HuggingFace discussion #47) remains open and unfixed since
2025-08-30.

This RDR addresses how to integrate MinerU given this constraint. All extraction
quality findings, spike results, and detection heuristics from RDR-022 remain
valid and are not repeated here.

## Context

### Background

RDR-022 completed Phase 1 (spike) and confirmed:

- MinerU 3.0.9 correctly extracts dual-column Springer Encyclopedia (5/5 checks)
- MinerU is ~86x faster than Marker on Apple Silicon MPS (1.1 vs 95 s/page)
- Column detection heuristic achieves 94% accuracy on 17-PDF test set
- `extraction_method` is already filterable in MeiliSearch
- `corpus repair` infrastructure supports selective re-indexing

Phase 2 (implementation) was abandoned when `pip install -e ".[advanced-pdf]"`
failed due to the transformers version conflict. The original spike tested MinerU
in an isolated venv, not co-installed with arcaneum's dependencies.

### Technical Environment

- **Python**: >= 3.12
- **Arcaneum transformers pin**: `>=4.40.0,<4.54.0` (see `docs/DEPENDENCY_NOTES.md`)
- **Stella model**: `dunzhang/stella_en_1.5B_v5` loaded with `trust_remote_code=True`
- **MinerU 3.0.9**: requires `transformers>=4.57.3` for `[pipeline]` extra
- **MinerU CLI**: `mineru -p <pdf> -o <outdir> --method auto --backend pipeline`
- **MinerU API**: Auto-starts local FastAPI server when `--api-url` not provided

## Research Findings

### Investigation

Three integration strategies were identified during the abandoned implementation.

#### Dependency Source Verification

| Dependency | Source Searched? | Key Findings |
| --- | --- | --- |
| mineru 3.0.9 | Yes (PyPI metadata) | `[pipeline]` extra requires `transformers>=4.57.3`, confirmed from published wheel METADATA |
| stella_en_1.5B_v5 | Yes (HuggingFace) | `modeling_qwen.py` has 4 calls to `DynamicCache.get_usable_length()`. Last modified 2025-07-28. Discussion #47 open since 2025-08-30, unfixed |
| transformers | Yes (Source) | `get_usable_length()` removed in 4.54.0 via PR #39106 |
| sentence-transformers 5.2.0 | Yes (Source) | No compatibility shim for DynamicCache changes |

### Key Discoveries

#### Option A: Subprocess/CLI Isolation — [Spike Completed, Fallback]

Install MinerU in a separate Python environment. Detect via
`shutil.which("mineru")`. Invoke as subprocess:

```text
mineru -p <pdf> -o <tmpdir> --method auto --backend pipeline --formula --table
```

Read markdown from output directory (`<outdir>/<stem>/auto/<stem>.md`).

- **Pros**: Zero dependency coupling; arcaneum and MinerU each use their own
  transformers version; no changes to arcaneum's dependency pins
- **Cons**: User manages a second Python environment; MinerU CLI auto-starts
  a local FastAPI server per invocation (startup overhead); model download
  (~2.5GB) happens in the separate environment
- **Spike result**: 5.4s model init overhead per invocation; API server
  auto-starts and stops with each command. 1.31 s/page warm (1.2x vs
  in-process). See Critical Assumptions for full results.

#### Option B: Lift Transformers Cap — [Spike Completed, Failed]

The Stella model's custom `modeling_qwen.py` is loaded because `config.json`
has an `auto_map` entry and sentence-transformers passes `trust_remote_code=True`.
If loaded *without* `trust_remote_code`, transformers uses its built-in
`Qwen2Model` implementation which does not call the removed API.

- **Pros**: If embedding quality is preserved, the `<4.54.0` cap can be removed
  entirely; MinerU becomes a normal `[advanced-pdf]` optional dependency;
  simplest user experience
- **Cons**: Needs empirical validation that embedding quality is unchanged;
  may lose model-specific optimizations in the custom code; risk of subtle
  quality regression in search results
- **Spike result**: FAILED. Mean cosine similarity = 0.73, min = 0.51.
  All 50 pairs below 0.95. The custom `modeling_qwen.py` produces
  fundamentally different embeddings than the built-in `Qwen2Model`.
  Option B is not viable.

#### Option C: Wait for Upstream Fix — [Documented]

- Stella maintainer (`infgrad`) pinned discussion #47 but has not applied a fix
- `transformers` v5 stable released (5.5.3 as of April 2026); `get_usable_length`
  remains removed — no backward compatibility shim was added
- Community fork `it-just-works/stella_en_1.5B_v5_bf16` is **alive** (last updated
  2025-09-16, 1994 downloads). The fork replaces all 4 `get_usable_length()` calls
  with a helper that uses `get_seq_length()` + `get_max_cache_shape()`. This is a
  clean, targeted fix.
- **Assessment**: No clear timeline for official Stella fix; not actionable

#### Option D: Monkey-Patch DynamicCache Shim — [Spike Completed, Verified]

Apply a compatibility shim at arcaneum import time that restores
`Cache.get_usable_length()` for transformers >= 4.54.0. The shim is 6 lines
of code that replicates the removed method using `get_seq_length()` and
`get_max_cache_shape()`, which remain available through transformers 5.5.3.
See Technical Design section for the shim implementation.

- **Pros**: Eliminates the dependency conflict entirely; MinerU becomes a normal
  `[advanced-pdf]` optional dependency; Stella embeddings are **bit-identical**
  (verified); simplest user experience; no separate venv; low maintenance burden
  (6 lines of code with clear deprecation path)
- **Cons**: Monkey-patching is inherently fragile; if transformers changes the
  `get_seq_length` or `get_max_cache_shape` APIs, the shim breaks; requires a
  test that runs on each transformers upgrade
- **Spike result**: Mean cosine similarity = 1.0000000000, bit-identical
  (`np.allclose(atol=1e-7) = True`) across 10 representative chunks.
  `get_seq_length` and `get_max_cache_shape` confirmed present in transformers
  4.57.6. Shim is safe.
- **Deprecation path**: Remove shim when either (a) Stella upstream fixes
  `modeling_qwen.py`, (b) arcaneum switches to the community fork, or
  (c) arcaneum switches embedding models

#### Option E: Use Community Fork Directly — [Documented]

Switch from `dunzhang/stella_en_1.5B_v5` to `it-just-works/stella_en_1.5B_v5_bf16`
which already has the `get_usable_length` fix applied.

- **Pros**: No monkey-patching; official-quality fix by community contributor;
  already handles the DynamicCache migration
- **Cons**: Different model weights (bf16 quantized); would require re-indexing
  all existing corpora; less widely used than the official model (1994 vs
  official downloads); risk of fork going unmaintained
- **Assessment**: Viable fallback if Option D proves unstable, but re-indexing
  cost is significant

### Critical Assumptions

- [x] Stella embedding quality is preserved without `trust_remote_code=True`
  — **Status**: **FAILED** — **Method**: Spike completed (2026-04-12)
  Mean cosine similarity = 0.73, min = 0.51. All 50 pairs below 0.95.
  The custom `modeling_qwen.py` produces fundamentally different embeddings
  than the built-in `Qwen2Model`. `trust_remote_code=True` is required.
- [x] MinerU CLI subprocess overhead is acceptable for batch processing
  — **Status**: **Verified** — **Method**: Spike completed (2026-04-12)
  42-page PDF: cold start 1.65 s/page, warm start 1.31 s/page (vs 1.1 s/page
  in-process from RDR-022). Model init overhead: 5.4s per invocation.
  Only 1.2x slower than in-process. CLI auto-starts/stops API server.
- [x] DynamicCache monkey-patch produces identical Stella embeddings
  — **Status**: **Verified** — **Method**: Spike completed (2026-04-12)
  Shim applied to `Cache` base class. Mean cosine similarity = 1.0,
  bit-identical (`np.allclose(atol=1e-7)`). `get_seq_length` and
  `get_max_cache_shape` confirmed present in transformers 4.57.6.
- [x] MinerU extraction quality and column detection are validated
  — **Status**: Verified — **Method**: Spike (RDR-022, 2026-04-12)
- [x] Column detection heuristic achieves acceptable accuracy
  — **Status**: Verified — **Method**: Spike (RDR-022, 94% on 17 PDFs)

## Proposed Solution

### Approach

**Recommended: Option D (Monkey-Patch DynamicCache Shim)**

Apply a 6-line compatibility shim at arcaneum import time that restores
`Cache.get_usable_length()` for transformers >= 4.54.0. This allows:

1. Remove the `transformers<4.54.0` cap from `pyproject.toml`
2. Add `[advanced-pdf]` extras with `mineru[pipeline]>=3.0.9`
3. Use the in-process integration design from RDR-022 Phase 2

The shim produces **bit-identical** Stella embeddings (verified by spike).
No re-indexing is required for existing corpora.

**Fallback: Option A (Subprocess Isolation)**

If the monkey-patch approach proves fragile across transformers releases,
fall back to CLI subprocess invocation. The spike confirmed only 1.2x
overhead vs in-process (1.31 vs 1.1 s/page on a 42-page paper).

### Technical Design

#### DynamicCache Shim

Add a compatibility module (e.g., `src/arcaneum/embeddings/_compat.py`)
that applies the shim at import time, guarded by a version check:

```python
from transformers.cache_utils import Cache

if not hasattr(Cache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length, layer_idx=0):
        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length
    Cache.get_usable_length = _get_usable_length
```

Import this module before any `SentenceTransformer` loading occurs.
Add a test that verifies embedding equivalence across transformers versions.

#### Dependency Changes

```toml
# Remove upper bound:
transformers>=4.40.0   # was: >=4.40.0,<4.54.0

# Add optional extra:
[project.optional-dependencies]
advanced-pdf = ["mineru[pipeline]>=3.0.9"]
```

The RDR-022 Phase 2 design (extractor backend, auto-detection, selective
re-indexing, CLI flags) remains valid — the shim only changes the dependency
resolution, not the integration architecture.

### Decision Rationale

Option D is preferred because:

1. **Option B failed**: Stella without `trust_remote_code` produces fundamentally
   different embeddings (mean cosine similarity = 0.73). Not viable.
2. **Option D is verified**: Monkey-patch produces bit-identical embeddings.
   The shim is 6 lines with clear semantics and a deprecation path.
3. **Option A is viable but worse UX**: Requires users to manage a separate
   Python environment. Reserved as fallback.
4. **Option E (community fork)** requires re-indexing all existing corpora.
   Only worthwhile if Option D proves unstable over time.

## Alternatives Considered

See RDR-022 Alternatives Considered for full tool selection analysis
(MinerU vs Marker vs Nougat vs Docling vs commercial APIs).

### Briefly Rejected

Options B, C, and E were investigated and rejected. See Research Findings
for full analysis and spike results.

## Trade-offs

See RDR-022 Trade-offs section. Additional considerations for this RDR:

### Consequences

- Positive: Resolves the dependency conflict that blocked RDR-022
- Positive: No re-indexing required — existing Stella embeddings unchanged
- Positive: Simplest user experience (`pip install "arcaneum[advanced-pdf]"`)
- Negative: Monkey-patch is inherently fragile; requires maintenance test
- Negative: If `get_seq_length`/`get_max_cache_shape` APIs change in future
  transformers releases, the shim must be updated

### Risks and Mitigations

- **Risk**: Transformers removes `get_seq_length` or `get_max_cache_shape`
  in a future release, breaking the shim
  **Mitigation**: Add a CI test that loads Stella and verifies embeddings
  with the shim. Pin transformers upper bound if breakage detected.
  Fallback to Option A (subprocess) or Option E (community fork).

- **Risk**: MinerU `[pipeline]` extra pulls in additional transitive
  dependencies that conflict with arcaneum
  **Mitigation**: Test full dependency resolution before release. Option A
  (subprocess isolation) remains available as fallback.

## Implementation Plan

### Prerequisites

- [x] Spike 1: Stella `trust_remote_code` test — **FAILED** (2026-04-12)
- [x] Spike 2: MinerU CLI overhead — **PASSED** (2026-04-12, 1.2x overhead)
- [x] Spike 3: DynamicCache monkey-patch — **PASSED** (2026-04-12, bit-identical)
- [x] Integration strategy selected: **Option D (monkey-patch shim)**

### Minimum Viable Validation

Same as RDR-022: Extract Springer-Verlag Encyclopedia pages 56-57 with MinerU
through the full `arc corpus sync` pipeline and verify column separation,
sentence integrity, and math preservation.

### Phase 1: Spike — Integration Strategy (COMPLETED)

All three spikes completed 2026-04-12. See Critical Assumptions for detailed
results. Summary:

- **Spike 1 (Option B)**: FAILED — Stella without `trust_remote_code` produces
  different embeddings (mean cosine = 0.73). Not viable.
- **Spike 2 (Option A)**: PASSED — MinerU CLI is 1.2x slower than in-process
  (1.31 vs 1.1 s/page). Acceptable as fallback.
- **Spike 3 (Option D)**: PASSED — DynamicCache shim produces bit-identical
  embeddings. Selected as recommended approach.

### Phase 2: Code Implementation

1. Add `src/arcaneum/embeddings/_compat.py` with DynamicCache shim
2. Import shim before any `SentenceTransformer` loading in `client.py`
3. Update `pyproject.toml`: remove `<4.54.0` cap, add `[advanced-pdf]` extras
4. Implement MinerU extractor backend per RDR-022 Phase 2 design
5. Add CI test: verify Stella embedding equivalence with shim active
6. Update `docs/DEPENDENCY_NOTES.md` with new constraint rationale

### Day 2 Operations

See RDR-022 Day 2 Operations. Additional:

- Monitor `get_seq_length` and `get_max_cache_shape` availability across
  transformers releases. CI test will catch breakage.
- If shim breaks: fall back to Option A (subprocess) or Option E (fork)

### New Dependencies

| Dependency | Version | License | Condition |
| --- | --- | --- | --- |
| mineru[pipeline] | >= 3.0.9 | AGPL-3.0 | `[advanced-pdf]` extras |

## Validation

### Testing Strategy

See RDR-022 Testing Strategy. Additional tests for this RDR:

Scenarios 1-3 (shim equivalence, trust_remote_code, CLI overhead) were validated
as spikes. See Critical Assumptions for results.

- **CI test (to implement)**: Load Stella with shim on current transformers,
  generate embeddings for fixed test vectors, compare against golden reference.
  Catches breakage if transformers changes `get_seq_length`/`get_max_cache_shape`.

### Performance Expectations

See RDR-022 Performance Expectations.

## Finalization Gate

> Complete each item with a written response before
> marking this RDR as **Final**.

### Contradiction Check

No contradictions with RDR-022. The monkey-patch approach (Option D) was not
considered in the original RDR because the investigation focused on whether
`trust_remote_code=False` would work. The spike result (FAIL at 0.73 similarity)
proves that Stella's custom code is essential, which makes the shim approach
the correct resolution.

### Assumption Verification

All critical assumptions verified. See Critical Assumptions section above.

#### API Verification

| API Call | Library | Verification |
| --- | --- | --- |
| SentenceTransformer(trust_remote_code=True) | sentence-transformers | Verified — required, produces correct embeddings |
| SentenceTransformer(trust_remote_code=False) | sentence-transformers | Verified — FAILS, embeddings differ fundamentally |
| Cache.get_usable_length (shim) | transformers | Verified — bit-identical embeddings with shim |
| Cache.get_seq_length | transformers | Verified — present in 4.53.3 and 4.57.6 |
| Cache.get_max_cache_shape | transformers | Verified — present in 4.53.3 and 4.57.6 |
| mineru CLI invocation | mineru | Verified — 1.31 s/page warm, clean output |

### Scope Verification

MVV is in scope (same as RDR-022).

### Cross-Cutting Concerns

- **Versioning**: Remove `transformers<4.54.0` cap; add `[advanced-pdf]` extras
- **Build tool compatibility**: `[advanced-pdf]` extras for `pip install`
- **Licensing**: AGPL-3.0 (same as RDR-022, accepted)
- **Deployment model**: Local CLI tool
- **IDE compatibility**: N/A
- **Incremental adoption**: Auto-detection means zero config for existing users
- **Secret/credential lifecycle**: N/A
- **Memory management**: Same as RDR-022
- **Backward compatibility**: Shim is no-op on transformers < 4.54.0;
  existing installations unaffected

### Proportionality

This RDR is intentionally thin — it references RDR-022 for all extraction
quality research, spike results, detection heuristics, and implementation
design. It focuses solely on resolving the dependency conflict.

## References

- [RDR-022: Advanced PDF Extraction for Complex Technical Documents](RDR-022-advanced-pdf-extraction.md)
- [Arcaneum Dependency Notes](../DEPENDENCY_NOTES.md)
- [Stella DynamicCache Discussion #47](https://huggingface.co/dunzhang/stella_en_1.5B_v5/discussions/47)
- [Community Fork with DynamicCache Fix](https://huggingface.co/it-just-works/stella_en_1.5B_v5_bf16)
- [transformers PR #39106: Remove get_usable_length](https://github.com/huggingface/transformers/pull/39106)
- [MinerU CLI Documentation](https://github.com/opendatalab/MinerU)

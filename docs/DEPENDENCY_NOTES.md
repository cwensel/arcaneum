# Dependency Constraints & Upgrade Path

This document explains the reasoning behind key dependency constraints in `pyproject.toml`.

Last reviewed: August 2026

## DynamicCache Breaking Change (transformers v4.54+)

**Status: mitigated.** The cap was relaxed from `<4.54.0` to `<5.0` in August 2026.
A runtime shim in `src/arcaneum/embeddings/client.py` (`_ensure_dynamic_cache_compat`)
restores the removed `DynamicCache.get_usable_length` by delegating to its direct
replacement `get_seq_length()`. A local spike verified embeddings are **bit-identical**
(cosine 1.0, max element-wise diff 0.0) between transformers 4.53.3 and 4.57.6+shim
on the pinned Stella revision. transformers 5.x remains excluded — see below.

### Issue

**Error:** `'DynamicCache' object has no attribute 'get_usable_length'`

**Root Cause:** transformers v4.54.0 (June 2024) introduced a **major breaking refactor** of the
caching system, moving from a monolithic `DynamicCache` to a per-layer cache architecture.
This removed critical methods:

- `get_usable_length()` - **removed, breaks embedding models**
- `get_max_length()` - replaced with `get_max_cache_shape()`
- `is_updated` - removed

### Impact

Embedding models that depend on these methods (including **Stella** and **NV-Embed-v2**) fail when:

- sentence-transformers calls the embedding model
- The model tries to use removed cache methods
- Batches fail with `RuntimeError: 'DynamicCache' object has no attribute 'get_usable_length'`

### Upstream Status (August 2026)

- **stella model** (`dunzhang/stella_en_1.5B_v5`) - effectively unmaintained;
  README-only commits for over a year and
  [not patched by maintainers](https://huggingface.co/NovaSearch/stella_en_1.5B_v5/discussions/47).
  Its `modeling_qwen.py` still calls `get_usable_length()` plus other removed APIs.
- **transformers 4.54–4.x** - works with the arcaneum shim (verified bit-identical
  embeddings on 4.57.6)
- **transformers v5.x** - hard failure at model init:
  `'Qwen2Config' object has no attribute 'rope_theta'` (v5 nests rope parameters,
  the remote code reads them flat). Not fixable with a cache shim; would require
  the community fork below.

**Community workaround exists:**
[`it-just-works/stella_en_1.5B_v5_bf16`][stella-fix] reimplements the deprecated method,
but we use the official model with constrained transformers versions for stability.

[stella-fix]: https://huggingface.co/it-just-works/stella_en_1.5B_v5_bf16/commit/03aedd040580357ec688f3467f1109af5e053249

### Solution: Runtime Shim + `<5.0` Cap (Tested & Verified, August 2026)

**Current constraints in pyproject.toml:**

```toml
sentence-transformers>=5.5.1        # no forcing function; supports transformers >=4.41,<6
transformers>=4.40.0,<5.0           # 4.54+ works via shim; 5.x breaks Stella remote code
torch>=2.12.0
```

**The shim:** `_ensure_dynamic_cache_compat()` in `src/arcaneum/embeddings/client.py`,
applied immediately after the lazy `sentence_transformers` import at both load sites.
It re-adds `get_usable_length(new_seq_length, layer_idx=0)` as a delegate to
`get_seq_length(layer_idx)` only when missing (no-op on transformers <4.54).
Covered by `tests/unit/embeddings/test_transformers_compat.py`.

**Spike results (August 2026), all on CPU/fp32 at pinned revision `7817065`:**

- ✅ **transformers 4.57.6 + shim** - bit-identical to the 4.53.3 baseline
  (cosine 1.0 on all probe texts, max element-wise diff 0.0); no re-indexing
  needed for existing Stella corpora
- ❌ **transformers 4.57.6 unpatched** - reproduces
  `'DynamicCache' object has no attribute 'get_usable_length'`
- ❌ **transformers 5.14.1** - fails at init on `rope_theta` before the cache
  path is even reached
- ❌ **`trust_remote_code=False` (native Qwen2 path)** - loads and runs but is
  **silently wrong**: cosine vs. baseline drops to 0.42–0.85 and retrieval
  breaks. Root cause: Stella's custom code runs the backbone with
  **bidirectional attention** (`is_causal=False`), while native `Qwen2Model`
  is strictly causal. Never load Stella without its remote code.

**Path to transformers 5.x (future work):**

- Requires abandoning the official model for the maintained community fork
  [`it-just-works/stella_en_1.5B_v5_bf16`][stella-fork] (patched through
  transformers 5.x as of July 2026). Costs: bf16 weights produce slightly
  different embeddings (existing Stella corpora would need re-indexing), and
  the fork's README documents 5.x load caveats requiring post-load sanity checks.
- Watch [stella model discussions](https://huggingface.co/NovaSearch/stella_en_1.5B_v5/discussions)
  for an official fix, though the repo appears unmaintained.

[stella-fork]: https://huggingface.co/it-just-works/stella_en_1.5B_v5_bf16

### Testing & Troubleshooting

**During initial setup:**

1. After `pip install -e .`, test with a small PDF sample:

   ```bash
   arc index pdf [single-pdf] --collection TestPapers
   ```

2. If you see `'DynamicCache' object has no attribute 'get_usable_length'`:
   - You are likely loading the model outside arcaneum's client (the shim in
     `embeddings/client.py` is applied at arcaneum's load sites only), or
     running transformers 5.x
   - Reinstall with correct constraint: `pip install "transformers>=4.40.0,<5.0"`
   - Then reinstall arcaneum: `pip install -e .`

3. Check your versions:

   ```bash
   pip show transformers sentence-transformers torch | grep Version
   ```

   Expected: transformers 4.x (any, 4.40+), sentence-transformers 5.5.x+, torch 2.12.x+

**If you have persistent embedding errors:**

1. Try cache-disabling workaround (slower but may work):

   ```bash
   TRANSFORMERS_NO_CACHE=1 arc index pdf [path] --collection [name]
   ```

2. Verify model downloads:

   ```bash
   arc config show-cache-dir
   ```

3. Reinstall models cleanly:

   ```bash
   pip cache purge
   pip install --force-reinstall -e .
   ```

### Related Issues

- [transformers#36071][tf-36071] - Cache refactor tracking (Phi-3 specific, closed with workaround)
- [stella_en_1.5B_v5#47][stella-47] - Stella DynamicCache issue
- [chronos#310][chronos-310] - Impact on other projects
- [sentence-transformers issues][st-issues] - Follow for updates

[tf-36071]: https://github.com/huggingface/transformers/issues/36071
[stella-47]: https://huggingface.co/NovaSearch/stella_en_1.5B_v5/discussions/47
[chronos-310]: https://github.com/amazon-science/chronos/issues/310
[st-issues]: https://github.com/UKPLab/sentence-transformers/issues

---

## Other Dependencies

All other dependencies use flexible version constraints and should not cause similar issues:

- **qdrant-client:** >=1.18.0
- **sentence-transformers:** >=5.5.1 (compatible with the transformers <5.0 cap)
- **torch:** >=2.12.0
- **fastembed:** >=0.8.0 (uses ONNX, no transformers dependency)
- **meilisearch:** >=0.41.0
- **PyMuPDF/PyMuPDF4LLM/pymupdf-layout:** >=1.27.2.3
- **tree-sitter-language-pack:** >=1.8.1; the AST extractor supports both the
  older bytes-based binding and the 1.8+ method-based binding
- **llama-index-core:** >=0.14.6 (abstracts transformers, no direct dependency)

### pymupdf-layout Integration

**Status:** ✅ Integrated and automatically used

The `pymupdf-layout` package is now **automatically integrated** into the PDF extraction pipeline:

- **Used for:** Enhanced layout detection and structure analysis
- **When:** Automatically applied during markdown extraction (RDR-016)
- **Benefit:** Better semantic structure preservation, improved chunking for search
- **Fallback:** If pymupdf-layout unavailable, gracefully uses standard extraction
- **Performance:** Minimal overhead - runs once per PDF, not per chunk

**How it works:**

1. PDFExtractor detects if pymupdf-layout is installed
2. During markdown extraction, analyzes PDF layout structure
3. Uses layout information to enhance semantic understanding
4. Metadata includes layout analysis results (text blocks, pages analyzed)
5. Improves semantic chunking for better search results

**Transparent to users:** No CLI flag needed, just works automatically when installed via `pip install -e .`

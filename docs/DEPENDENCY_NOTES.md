# Dependency Constraints & Upgrade Path

This document explains the reasoning behind key dependency constraints in `pyproject.toml`.

Last reviewed: April 2026

## DynamicCache Compatibility Shim (RDR-023)

### Issue

**Error:** `'DynamicCache' object has no attribute 'get_usable_length'`

**Root Cause:** transformers v4.54.0 (June 2024) removed `Cache.get_usable_length()` as part
of a caching system refactor. Stella's custom `modeling_qwen.py` calls this method in 4 places
(lines 279, 383, 681, 1003). The upstream issue remains unfixed (HuggingFace discussion #47).

### Solution: Monkey-Patch Shim (RDR-023)

As of April 2026, the `<4.54.0` upper bound on transformers has been **removed**. Instead,
a compatibility shim in `src/arcaneum/embeddings/_compat.py` restores `Cache.get_usable_length()`
at import time using `get_seq_length()` and `get_max_cache_shape()`.

**Current constraints in pyproject.toml:**

```toml
sentence-transformers>=5.2.0        # 5.2.0 fixes torch 2.9+ compatibility
transformers>=4.40.0                # DynamicCache shim handles 4.54+ (RDR-023)
torch>=2.8.0                        # 2.9.x now works with sentence-transformers 5.2.0+
```

**The shim:**

- Is 6 lines of code in `src/arcaneum/embeddings/_compat.py`
- Produces **bit-identical** Stella embeddings (verified via spike: `np.allclose(atol=1e-7)`)
- Is imported before any `SentenceTransformer` loading in `client.py`
- Is guarded by `hasattr` — no-op on transformers < 4.54.0

**Why the shim instead of an upper bound:**

- The upper bound blocked MinerU integration (requires `transformers>=4.57.3`)
- Option B (`trust_remote_code=False`) was tested and failed — embeddings differ
  fundamentally (mean cosine similarity = 0.73)
- The shim is the least disruptive path to unblocking MinerU as a dependency

**Deprecation path — remove the shim when any of:**

- Stella upstream fixes `modeling_qwen.py` (discussion #47)
- Arcaneum switches to the community fork (`it-just-works/stella_en_1.5B_v5_bf16`)
- Arcaneum switches embedding models

**Monitoring:**

- `get_seq_length` and `get_max_cache_shape` confirmed present through transformers 5.5.3
- CI test should verify embedding equivalence with shim active on each transformers upgrade
- If either API is removed in a future release, the shim will fail and tests will catch it

### Upstream Status (April 2026)

- **transformers 5.5.3** - `get_usable_length` still absent, `get_seq_length`/`get_max_cache_shape` present
- **stella model** (`dunzhang/stella_en_1.5B_v5`) - [Not patched](https://huggingface.co/NovaSearch/stella_en_1.5B_v5/discussions/47)
- **Community fork** (`it-just-works/stella_en_1.5B_v5_bf16`) - Has fix, last updated 2025-09-16

### Testing & Troubleshooting

**During initial setup:**

1. After `pip install -e .`, test with a small PDF sample:

   ```bash
   arc corpus sync TestCorpus /path/to/test.pdf
   ```

2. If you see `'DynamicCache' object has no attribute 'get_usable_length'`:
   - The shim may not be loading — check that `_compat.py` is imported in `client.py`
   - Verify: `python -c "import arcaneum.embeddings._compat; print('shim loaded')"`

3. Check your versions:

   ```bash
   pip show transformers sentence-transformers torch | grep Version
   ```

### Related Issues

- [transformers#36071][tf-36071] - Cache refactor tracking
- [stella_en_1.5B_v5#47][stella-47] - Stella DynamicCache issue
- [chronos#310][chronos-310] - Impact on other projects
- [RDR-023](rdr/RDR-023-advanced-pdf-integration.md) - Full spike results and decision rationale

[tf-36071]: https://github.com/huggingface/transformers/issues/36071
[stella-47]: https://huggingface.co/NovaSearch/stella_en_1.5B_v5/discussions/47
[chronos-310]: https://github.com/amazon-science/chronos/issues/310

---

## Other Dependencies

All other dependencies use flexible version constraints and should not cause similar issues:

- **sentence-transformers:** >=5.2.0 (compatible with transformers 4.34+, torch 2.9+)
- **torch:** >=2.8.0 (no upper bound needed with sentence-transformers 5.2.0+)
- **fastembed:** >=0.7.3 (uses ONNX, no transformers dependency)
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

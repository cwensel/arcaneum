# Dependency Constraints & Upgrade Path

This document explains the reasoning behind key dependency constraints in `pyproject.toml`.

Last reviewed: January 2026

## DynamicCache Breaking Change (transformers v4.54+)

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

### Upstream Status (January 2026)

**The issue remains unfixed upstream:**

- **transformers 4.54.0 through 4.57.6** - All versions have the breaking change
- **transformers v5.0** - Still in release candidate stage (RC2 as of Jan 8, 2026), not production ready
- **stella model** (`dunzhang/stella_en_1.5B_v5`) - [Not patched by maintainers](https://huggingface.co/NovaSearch/stella_en_1.5B_v5/discussions/47)

**Community workaround exists:**
[`it-just-works/stella_en_1.5B_v5_bf16`][stella-fix] reimplements the deprecated method,
but we use the official model with constrained transformers versions for stability.

[stella-fix]: https://huggingface.co/it-just-works/stella_en_1.5B_v5_bf16/commit/03aedd040580357ec688f3467f1109af5e053249

### Solution: Stable Version Matrix (Tested & Verified)

**Current constraints in pyproject.toml:**

```toml
sentence-transformers>=5.2.0        # 5.2.0 fixes torch 2.9+ compatibility
transformers>=4.40.0,<4.54.0        # 4.54+ has cache breaking changes
torch>=2.8.0                        # 2.9.x now works with sentence-transformers 5.2.0+
pymupdf-layout>=0.1.0               # Better PDF layout detection
```

**Why this version matrix:**

1. ✅ **sentence-transformers 5.2.0** - Adds transformers v5 support, fixes torch 2.9+ compatibility
2. ✅ **transformers 4.40-4.53** - Avoids cache breaking change in 4.54+
3. ✅ **torch 2.8.0+** - Compatible with sentence-transformers 5.2.0+
4. ✅ **pymupdf-layout** - Improves PDF semantic chunking quality

**What we tested & rejected:**

- ❌ **transformers 4.57.5** - DynamicCache breaking change still present
- ❌ **transformers 4.56.x, 4.55.x, 4.54.x** - All have the cache breaking change
- ❌ **transformers v5.0-RC2** - Release candidate, not production ready

**Rationale:**

- The DynamicCache breaking change has not been resolved in any transformers 4.54+ release
- The stella model maintainers have not updated their code to use the new Cache API
- This matrix prioritizes **stability & reliability** over latest features
- sentence-transformers 5.2.0 requires transformers>=4.34.0, compatible with our <4.54.0 constraint

**Future improvements:**

- Monitor [transformers v5.0 release](https://github.com/huggingface/transformers/releases) for final stable release
- Watch [stella model discussions](https://huggingface.co/NovaSearch/stella_en_1.5B_v5/discussions) for upstream fix
- Consider switching to patched stella model if upstream remains unfixed after v5.0 releases

### Testing & Troubleshooting

**During initial setup:**

1. After `pip install -e .`, test with a small PDF sample:

   ```bash
   arc index pdf [single-pdf] --collection TestPapers
   ```

2. If you see `'DynamicCache' object has no attribute 'get_usable_length'`:
   - The upper bound constraint may have been removed or overridden
   - Reinstall with correct constraint: `pip install "transformers>=4.40.0,<4.54.0"`
   - Then reinstall arcaneum: `pip install -e .`

3. Check your versions:

   ```bash
   pip show transformers sentence-transformers torch | grep Version
   ```

   Expected: transformers 4.53.x or lower, sentence-transformers 5.2.x, torch 2.8.x+

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

# Dependency Constraints & Upgrade Path

This document explains the reasoning behind key dependency constraints in `pyproject.toml`.

## DynamicCache Breaking Change (transformers v4.54+)

### Issue

**Error:** `'DynamicCache' object has no attribute 'get_usable_length'`

**Root Cause:** transformers v4.54.0 (June 2024) introduced a **major breaking refactor** of the caching system, moving from a monolithic `DynamicCache` to a per-layer cache architecture. This removed critical methods:

- `get_usable_length()` - **removed, breaks embedding models**
- `get_max_length()` - replaced with `get_max_cache_shape()`
- `is_updated` - removed

### Impact

Embedding models that depend on these methods (including **Stella** and **NV-Embed-v2** used by jina-code) fail when:
- sentence-transformers calls the embedding model
- The model tries to use removed cache methods
- Batches fail with `RuntimeError: 'DynamicCache' object has no attribute 'get_usable_length'`

### Solution: Stable Version Matrix (Tested & Verified)

**Current (Tested working versions):**
```toml
sentence-transformers>=5.0.0,<5.1.0  # Safe range (5.1.2 incompatible with torch 2.9)
transformers>=4.40.0,<4.54.0         # Safe range (4.54+ has cache breaking changes)
torch>=2.8.0,<2.9.0                  # Constrain to 2.8.x (2.9.1 causes segfault)
pymupdf-layout>=0.1.0                # Better PDF layout detection (new)
```

**Why this version matrix:**

1. ✅ **sentence-transformers 5.0.x** - Stable, tested working (5.1.2 segfaults with torch 2.9)
2. ✅ **transformers 4.40-4.53** - Avoids cache breaking change in 4.54+
3. ✅ **torch 2.8.x** - Stable, no segfault issues (2.9.1 incompatible with sentence-transformers)
4. ✅ **pymupdf-layout** - Improves PDF semantic chunking quality

**What we tested & rejected:**

- ❌ **torch 2.9.1** - Causes segmentation fault on import of sentence-transformers
- ❌ **sentence-transformers 5.1.2** - Incompatible with torch 2.9.x, causes segfault
- ❌ **transformers 4.57.1** - Still has cache breaking changes, risk not worth it

**Rationale:**

After aggressive testing to upgrade to latest versions:
- torch 2.9.x performance gains are NOT worth the segfault risk
- sentence-transformers 5.0.x is stable and proven working
- transformers 4.54+ still has unfixed cache issues (upstream hasn't resolved it)
- This matrix prioritizes **stability & reliability** over latest features

**Future improvements:**

- Monitor [transformers releases](https://github.com/huggingface/transformers/releases) for cache fixes
- Watch for [transformers 5.0 release](https://github.com/huggingface/transformers)
- Update constraints as upstream fixes mature

### Testing & Troubleshooting

**During initial setup:**

1. After `pip install -e .`, test with a small PDF sample:
   ```bash
   arc index pdf [single-pdf] --collection TestPapers
   ```

2. If you see `'DynamicCache' object has no attribute 'get_usable_length'`:
   - Current constraint allows transformers 4.57.1
   - This error means cache refactor still hasn't been fixed upstream
   - Immediately downgrade: `pip install "transformers>=4.40.0,<4.54.0"`
   - Reinstall: `pip install -e .`

3. Check your versions:
   ```bash
   pip show transformers sentence-transformers torch | grep Version
   ```

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

- [huggingface/transformers#36071](https://github.com/huggingface/transformers/issues/36071) - Cache refactor tracking
- [amazon-science/chronos#310](https://github.com/amazon-science/chronos/issues/310) - Impact on other projects
- [sentence-transformers compatibility](https://github.com/UKPLab/sentence-transformers/issues) - Follow for updates

---

## Other Dependencies

All other dependencies use flexible version constraints and should not cause similar issues:

- **sentence-transformers:** >=5.0.0,<5.1.0 (stable, proven working)
- **torch:** >=2.8.0,<2.9.0 (2.9.1 causes segfault, stays stable)
- **fastembed:** >=0.7.3 (compatible with all tested transformers versions)
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

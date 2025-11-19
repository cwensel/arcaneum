---
description: Index content into collections
argument-hint: <pdf|code|markdown> <path> [options]
---

Index PDFs, markdown, or source code into Qdrant collections for semantic search.

**Subcommands:**

- pdf: Index PDF documents (with OCR support)
- markdown: Index markdown files (with frontmatter extraction)
- code: Index source code repositories (git-aware)

**Common Options:**

- --collection: Target collection (required)
- --model: Embedding model (auto-selected by content type)
- --workers: Parallel workers (default: 4)
- --force: Force reindex all files
- --no-gpu: Disable GPU acceleration (GPU enabled by default)
- --verbose: Show detailed progress (suppress library warnings)
- --debug: Show all library warnings including transformers
- --json: Output in JSON format

**PDF Indexing Options:**

- --no-ocr: Disable OCR (enabled by default for scanned PDFs)
- --ocr-language: OCR language code (default: eng)
- --offline: Use cached models only (no network)

**Markdown Indexing Options:**

- --chunk-size: Target chunk size in tokens (overrides model default)
- --chunk-overlap: Overlap between chunks in tokens
- --offline: Use cached models only (no network)

**Source Code Indexing Options:**

- --depth: Git discovery depth (traverse subdirectories)

**Examples:**

```text
# Basic indexing (GPU enabled by default)
/index pdf ~/Documents/Research --collection PDFs --model stella
/index markdown ~/notes --collection Notes --model stella
/index code ~/projects/myapp --collection MyCode --model jina-code

# With options
/index markdown ~/docs --collection Docs --chunk-size 512 --verbose
/index pdf ~/scanned-docs --collection Scans --no-ocr --offline

# Force CPU-only mode (disable GPU)
/index pdf ~/Documents/Research --collection PDFs --model stella --no-gpu

# Debug mode (show all warnings)
/index pdf ~/Documents/Research --collection PDFs --model stella --debug
```

**Execution:**

```bash
cd ${CLAUDE_PLUGIN_ROOT}
arc index $ARGUMENTS
```

**How It Works:**

**PDF Indexing:**
1. Extract text from PDFs (PyMuPDF + pdfplumber fallback)
2. Auto-trigger OCR for scanned PDFs (< 100 chars extracted)
3. Chunk text with 15% overlap for context
4. Generate embeddings (stella default: 1024D for documents)
5. Upload to Qdrant with metadata (file path, page numbers)
6. Incremental: Skips unchanged files (file hash metadata check)

**Markdown Indexing:**
1. Discover markdown files (.md, .markdown extensions)
2. Extract YAML frontmatter (title, author, tags, category, etc.)
3. Semantic chunking preserving document structure (headers, code blocks)
4. Generate embeddings (stella default: 1024D for documents)
5. Upload to Qdrant with metadata (file path, frontmatter fields, header context)
6. Incremental: Skips unchanged files (SHA256 content hash check)

**Source Code Indexing:**
1. Discover git repositories in directory tree
2. Extract git metadata (project, branch, commit)
3. Parse code with tree-sitter (AST-aware chunking, 15+ languages)
4. Generate embeddings (jina-code default: 768D for code)
5. Upload to Qdrant with metadata (git info, file path, language)
6. Multi-branch support: project#branch identifier
7. Incremental: Skips unchanged commits (metadata-based sync)

**Default Models:**

- PDFs: stella (1024D, document-optimized)
- Markdown: stella (1024D, document-optimized)
- Source: jina-code (768D, code-optimized)

**Performance:**

- PDF: ~10-30 PDFs/minute (depends on OCR workload)
- Markdown: ~50-100 files/minute (depends on file size)
- Source: 100-200 files/second (depends on file size)
- Batch upload: 100-200 chunks per batch
- Parallel workers: 4 (adjustable with --workers for PDF/source)
- **GPU acceleration**: 1.5-3x speedup (enabled by default, use --no-gpu to disable)

**GPU Acceleration:**

GPU acceleration is **enabled by default** for faster embedding generation:
- **Apple Silicon**: MPS (Metal Performance Shaders) backend
- **NVIDIA GPUs**: CUDA backend
- **CPU fallback**: Automatic if GPU unavailable
- **Disable GPU**: Use --no-gpu flag (for thermal/battery concerns)

**Compatible models** (verified with GPU support):
- stella (recommended for PDFs/markdown) - Full MPS support
- jina-code (recommended for source code) - Full MPS support
- bge-small, bge-base - CoreML support

**Offline Mode:**

Use --offline for corporate proxies or SSL issues:
- Requires models pre-downloaded: `arc models download`
- No network calls during indexing
- Fails if model not cached

**Related Commands:**

- /collection create - Create collection before indexing
- /search semantic - Search indexed content
- /corpus create - Create both vector + full-text indexes

**Debug Mode:**

Use --debug to troubleshoot indexing issues:
- Shows all library warnings (including HuggingFace transformers)
- Displays detailed stack traces
- Helps diagnose model loading or GPU issues
- Use --verbose for user-facing progress without library warnings

**Implementation:**

- RDR-004: PDF bulk indexing
- RDR-005: Git-aware source code indexing
- RDR-013: Performance optimization with GPU acceleration
- RDR-014: Markdown content indexing
- RDR-006: Claude Code integration

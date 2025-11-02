# Recommendation 013: Indexing Pipeline Performance Optimization

## Metadata
- **Date**: 2025-10-29
- **Implementation Date**: 2025-11-02
- **Status**: Implemented (Phases 1-2 Complete, Phase 3 Deferred)
- **Type**: Technical Debt
- **Priority**: High
- **Related Issues**: arcaneum-198, arcaneum-108, arcaneum-ce28, arcaneum-e4e7, arcaneum-f13e
- **Related Tests**: Manual testing completed
- **Implementation Notes**:
  - ✅ Phase 1 Complete: CPU monitoring, batch size configuration, process priority
  - ✅ Phase 2 Complete: GPU acceleration (default), parallel embedding generation, file-level parallelism (PDF/source/markdown), parallel OCR
  - ❌ Phase 2.5 Deferred: Progress bars/ETAs (basic progress implemented, tqdm not added)
  - ❌ Phase 3 Deferred: Embedding cache, Qdrant bulk mode, metadata sync optimization (future enhancements)

## Problem Statement

The Arcaneum indexing pipeline suffers from significant performance bottlenecks that result in slow throughput when indexing large PDF collections and source code repositories. Current implementation processes files sequentially, generates embeddings without parallelization, and uses suboptimal Qdrant upload configurations. For large-scale indexing operations (thousands of PDFs or large codebases), indexing can take hours or days when it should take minutes to hours.

## Context

### Background

This performance analysis was triggered by arcaneum-198, a critical priority task to conduct comprehensive performance profiling of the entire indexing pipeline. The goal is to identify and eliminate CPU waste, optimize throughput, and reduce processing time for both PDF and source code indexing operations.

The indexing pipeline is the core of Arcaneum's functionality, responsible for:
- Extracting text from PDFs (with OCR support)
- Chunking source code using AST parsing
- Generating embeddings via FastEmbed/SentenceTransformers
- Uploading vectors to Qdrant for semantic search

### Technical Environment

**Current Architecture:**
- Python 3.10+
- FastEmbed (ONNX Runtime) for embeddings
- SentenceTransformers (PyTorch) for some models
- Qdrant vector database (v1.7+)
- PyMuPDF / pdfplumber for PDF extraction
- Tesseract / EasyOCR for OCR
- tree-sitter / LlamaIndex for AST chunking

**Hardware Context:**
- Development: Mac M1/M2 (8-16 cores)
- Production: Varies (typically 4-16 CPU cores)

## Research Findings

### Investigation Process

1. **External Research**: Conducted web research on:
   - Embedding pipeline optimization techniques (Ray Data distributed processing)
   - Qdrant bulk upload best practices (indexing deferral, gRPC, batching)
   - ONNX Runtime performance tuning (thread management, execution providers)

2. **Codebase Analysis**: Deep exploration of:
   - Complete pipeline flow (PDF and source code paths)
   - Embedding generation code (FastEmbed/SentenceTransformers usage)
   - Chunking operations (PDF chunker, AST chunker)
   - Qdrant client configuration and upload logic
   - Current batch sizes, thread counts, and configuration values

3. **Bottleneck Identification**: Traced execution paths to identify:
   - Sequential operations that could be parallelized
   - Suboptimal configurations (batch sizes, API usage)
   - Redundant queries and processing
   - Memory-intensive operations

### Key Discoveries

#### CPU Utilization Discrepancy

**Observation**: When running `arc index` on macOS, `htop` shows CPUs are pegged (high utilization) but the `arc` process itself only shows ~14% CPU usage.

**Root Cause**: FastEmbed uses ONNX Runtime which spawns internal threads for parallel inference. These threads perform the actual computation but aren't attributed to the parent Python process in standard process monitors like `htop` or `top`.

**Impact on Analysis**:
- The low CPU percentage for the main process is **misleading**
- Actual CPU utilization is much higher (spread across ONNX Runtime threads)
- This confirms embedding generation IS CPU-bound and parallelization will help
- Need better monitoring to track true CPU usage (see Phase 1, Step 1.4)

**Technical Details**:
- ONNX Runtime uses thread pools for matrix operations
- Default behavior: uses all available CPU cores for inference
- Thread count controlled by `OMP_NUM_THREADS` and `inter_op_num_threads` / `intra_op_num_threads`
- Current implementation doesn't configure these, so ONNX uses defaults

**Monitoring Solution**: Use `psutil.Process.cpu_percent()` which aggregates all thread CPU usage (see implementation in Phase 1, Step 1.4)

#### GPU Acceleration Compatibility (Verified)

**Testing performed on macOS M1/M2 with actual models:**

**✅ Fully GPU-Compatible (MPS via SentenceTransformers):**
- stella (dunzhang/stella_en_1.5B_v5): Default for PDFs - ✅ Tested on MPS
- jina-code (jinaai/jina-embeddings-v2-base-code): Default for source code - ✅ Tested on MPS

**⚠️ Partially GPU-Compatible (CoreML via FastEmbed - Hybrid execution):**
- bge-large (BAAI/bge-large-en-v1.5, 1024D): ⚠️ Works but with warnings - Hybrid CoreML/CPU
- bge-base (BAAI/bge-base-en-v1.5, 768D): ✅ Works with CoreML
- bge-small (BAAI/bge-small-en-v1.5, 384D): ✅ Works with CoreML
- **Note**: CoreML warnings "input dim > 16384" are non-fatal. ONNX Runtime runs unsupported ops on CPU, rest on CoreML (879/1223 nodes for bge-large)

**❌ GPU-Incompatible (CPU only):**
- jina-v3 (jinaai/jina-embeddings-v3): ❌ True failure - Model execution error -2
  - CoreML cannot build execution plan for this model architecture
  - Must use CPU only

**Impact**: The **default models** (stella for PDFs, jina-code for code) both have full MPS support. This means GPU acceleration will work out-of-the-box for standard indexing workflows. Users of bge-large or jina-v3 will fall back to CPU automatically.

#### Critical Bottlenecks (Priority 0)

**1. Sequential Embedding Generation** (Estimated Speedup: 2-4x)
- **Location**: `src/arcaneum/indexing/source_code_pipeline.py:376-398`, `src/arcaneum/indexing/uploader.py:245-260`
- **Issue**: Each 100-chunk batch processed sequentially with no parallelization
- **Impact**: Embedding generation is CPU/GPU bound and consumes 99% of indexing time
- **Evidence**:
  - Code shows single-threaded loop over batches
  - Time analysis: 10,000 chunks × 50ms/chunk = ~500 seconds
  - Upload time: ~67 requests × 10ms = ~0.7 seconds (<1% of total)
- **Speedup Justification**:
  - Anyscale Ray Data case study: 20x speedup for 2,000 PDFs using 20 parallel workers (from 75 min → <4 min) [Ref 1]
  - With 4-core CPU and ThreadPoolExecutor, theoretical maximum 4x speedup
  - Practical speedup: 2-4x depending on batch overhead and GIL contention
  - **This is the REAL bottleneck** - 99% of time is spent here
  - Reference: [Turbocharge LangChain Guide](https://www.anyscale.com/blog/turbocharge-langchain-now-guide-to-20x-faster-embedding)

**2. Sequential File Processing** (Estimated Speedup: 2-8x)
- **Location**: `src/arcaneum/indexing/source_code_pipeline.py:300-367`, `src/arcaneum/indexing/uploader.py:166-338`
- **Issue**: Each file processed one at a time, leaving CPU cores idle
- **Impact**: Poor CPU utilization, especially on multi-core systems
- **Opportunity**: Process N files in parallel (N = CPU cores)
- **Speedup Justification**:
  - Amdahl's Law: With N cores, theoretical max speedup = N (if fully parallelizable)
  - File I/O and parsing are CPU-bound operations suitable for multiprocessing
  - 4-core system: ~3-4x speedup (accounting for overhead)
  - 8-core system: ~6-8x speedup
  - ProcessPoolExecutor avoids Python GIL limitations for CPU-bound work
  - **Combines with parallel embedding** for multiplicative speedup

#### High Priority Bottlenecks (Priority 1)

**3. No Qdrant Indexing Optimization** (Estimated Speedup: 1.3-1.5x)
- **Location**: Collection creation and bulk upload operations
- **Issue**: HNSW index built incrementally during uploads (default behavior)
- **Impact**: Indexing overhead on every batch upload
- **Best Practice**: Set `indexing_threshold=0` and `m=0` during bulk, rebuild after
- **Speedup Justification**:
  - Qdrant documentation: "Setting m=0 prevents dense vector index construction during ingestion" [Ref 3]
  - Incremental HNSW index updates are O(log n) per insert; bulk rebuild is more efficient
  - Measured improvement: 30-50% faster bulk uploads with deferred indexing
  - Trade-off: One-time index rebuild cost at end (amortized across all uploads)
  - Reference: [Qdrant Indexing Optimization](https://qdrant.tech/articles/indexing-optimization/)

**4. OCR Sequential Page Processing** (Estimated Speedup: 4-8x for OCR-heavy workloads)
- **Speedup Justification**:
  - OCR is CPU-intensive (Tesseract/EasyOCR) and highly parallelizable per page
  - Similar to file parallelization: N cores = ~N speedup for independent pages
  - 4-core system: ~3-4x speedup; 8-core system: ~6-8x speedup
  - Each page OCR is independent with no shared state
- **Location**: `src/arcaneum/indexing/pdf/ocr.py:69-85`
- **Issue**: Each PDF page OCR'd sequentially
- **Impact**: OCR is extremely slow (seconds per page), no parallelization
- **Opportunity**: Process pages in parallel with ProcessPoolExecutor

#### Medium Priority Bottlenecks (Priority 2)

**5. Synchronous Qdrant Uploads with wait=True** (Estimated Speedup: Minor)
- **Location**: `src/arcaneum/indexing/qdrant_indexer.py:172`, `src/arcaneum/indexing/uploader.py:403`
- **Issue**: `wait=True` forces each batch to wait for Qdrant indexing before proceeding
- **Impact**: Minimal - upload time is <1% of total indexing time
- **Analysis**:
  - 10,000 chunks ÷ 150 per batch = ~67 upload requests
  - Upload time with wait=True: ~10ms × 67 = 0.7 seconds
  - Embedding time: ~500 seconds (99% of total)
  - **Setting wait=False saves <1 second out of 500 seconds**
- **Recommendation**: Low priority unless doing many small uploads or remote Qdrant

**6. Small Embedding Batch Size** (Estimated Speedup: 1.1-1.2x)
- **Location**: `EMBEDDING_BATCH_SIZE = 100` in multiple files
- **Issue**: 100-chunk batches may be suboptimal for FastEmbed internal processing
- **Opportunity**: Test larger batches (200-500) for better throughput
- **Speedup Justification**:
  - Anyscale case study used 100 chunks/GPU batch as optimal [Ref 1]
  - Larger batches reduce Python overhead and improve ONNX Runtime efficiency
  - Trade-off: Batch size inversely correlates with chunk size (memory constraints)
  - Typical improvement: 10-20% from reduced batch boundary overhead

**7. No Embedding Cache** (Estimated Speedup: 1.1-1.3x depending on duplication)
- **Location**: EmbeddingClient has no caching layer
- **Issue**: Identical chunks (common imports, license headers) re-embedded every time
- **Opportunity**: Add LRU or persistent cache keyed by content hash
- **Speedup Justification**:
  - Cache hit avoids embedding generation entirely (instant vs 10-100ms per chunk)
  - Typical code duplication: 10-30% (imports, boilerplate, generated code)
  - Best case (30% hit rate): 1.3x speedup
  - Worst case (10% hit rate): 1.1x speedup
  - Most benefit on re-indexing or similar codebases

**8. Redundant Metadata Queries** (Estimated Speedup: 2-5x for sync phase)
- **Location**: `src/arcaneum/indexing/git_metadata_sync.py:89-97`
- **Issue**: Scrolls through ALL points (`batch_size=100`) to get indexed projects
- **Impact**: Slow startup for large collections (thousands of projects)
- **Opportunity**: Use aggregation queries, more aggressive caching
- **Speedup Justification**:
  - Current: O(N) scroll through all points with small batch size (100)
  - Improvement 1: Larger batch size (100→1000) = 10x fewer network round trips
  - Improvement 2: Persistent cache eliminates query on subsequent runs
  - For 10K points: ~10 API calls (1000/batch) vs ~100 calls (100/batch)
  - Cache hit eliminates query entirely: infinite speedup for cached case

#### Low Priority Optimizations (Priority 3)

**9. PDF Indexing Uses HTTP Instead of gRPC** (Estimated Speedup: <0.1%)
- **Location**: `src/arcaneum/cli/index_pdfs.py:112`
- **Issue**: Uses HTTP instead of gRPC for Qdrant communication
- **Impact**: Negligible for bulk indexing operations
- **Analysis**:
  - gRPC saves ~7ms per request vs HTTP (~10ms → ~3ms)
  - For 67 batches: 7ms × 67 = ~0.5 seconds saved
  - Out of 500+ seconds total = **0.1% improvement**
- **When it matters**: Many small queries (search ops), remote Qdrant server, or frequent small uploads
- **Recommendation**: Very low priority for bulk indexing; may help for search workloads
- **Justification**:
  - Qdrant docs show gRPC is faster [Ref 2]
  - But the time spent on uploads is <1% of total indexing time
  - Reference: [Qdrant Bulk Upload Best Practices](https://qdrant.tech/documentation/database-tutorials/bulk-upload/)

#### External Research Findings

**Ray Data Distributed Processing:**
- Achieved 20x speedup (75 minutes → <4 minutes) for 2,000 PDFs using 20 GPUs
- Key techniques: Lazy evaluation, stateful actors, batch processing (100 chunks/GPU)
- Inverse correlation: larger chunk size → smaller batch size to prevent OOM

**Qdrant Bulk Upload Best Practices:**
- Defer HNSW index construction: `m=0` during upload, restore to 16-32 after
- Disable indexing threshold: `indexing_threshold=0`, restore to 20000 after
- Use gRPC (Rust client fastest, but Python gRPC significantly faster than HTTP)
- Parallelize across shards: 2-4 shards per machine for distributed writes
- Enable memmap storage: `on_disk=True` for large datasets to bypass RAM constraints

**ONNX Runtime Performance Tuning:**
- Thread management: Configure inter-op and intra-op parallelism
- Graph optimizations: Use ORT format models for pre-optimized inference
- Memory consumption: Monitor and tune allocation patterns
- Profiling tools: Built-in tools for bottleneck identification
- **Important**: ONNX Runtime spawns worker threads that show as separate CPU usage in system monitors
  - This explains why `arc` process shows low CPU (~14%) while system CPU is pegged
  - True CPU usage is sum of main process + ONNX worker threads
  - Use `psutil` to aggregate thread-level CPU statistics

### Current Configuration Summary

| Component | Batch Size | Workers | Location |
|-----------|------------|---------|----------|
| Embedding (PDF) | 100 chunks | 1 (sequential) | `uploader.py:245` |
| Embedding (Source) | 100 chunks | 1 (sequential) | `source_code_pipeline.py:377` |
| Qdrant Upload (Source) | 150 chunks | 1 | `qdrant_indexer.py:44` |
| Qdrant Upload (PDF) | 100 chunks | 4 (internal) | `uploader.py:400-401` |
| File Processing | 1 file | 1 (sequential) | All indexing code |
| OCR Pages | 1 page | 1 (sequential) | `ocr.py:69-85` |

**Chunk Sizes:**
- PDF (stella): 768 tokens, 15% overlap
- PDF (modernbert): 1536 tokens, 15% overlap
- Source Code: 400 tokens, 5% overlap (20 tokens)

## Proposed Solution

### Approach

Implement a **phased optimization strategy** that addresses bottlenecks in order of impact, starting with quick wins (high impact, low effort) and progressing to more complex architectural changes. Each phase is independent and can be validated before proceeding to the next.

### Performance Profiles

To balance user-friendliness with maximum performance, implement **three performance profiles**:

**1. Default Profile (User-Friendly)**
- **Goal**: Keep laptop responsive for multitasking during indexing
- Workers: CPU cores / 2 (e.g., 4 on 8-core)
- GPU: Disabled by default
- Batch size: 100-200 (moderate)
- Priority: Normal process priority
- **Use case**: Indexing while working on other tasks

**2. Balanced Profile (`--fast`)**
- **Goal**: Faster indexing with acceptable responsiveness
- Workers: CPU cores - 1 (e.g., 7 on 8-core, leaves 1 for UI)
- GPU: Enabled
- Batch size: 200-500
- Priority: Normal
- **Use case**: Focused indexing session, occasional UI interaction

**3. Maximum Profile (`--turbo` or `--max-performance`)**
- **Goal**: Maximum throughput, laptop unattended
- Workers: CPU cores (100% utilization)
- GPU: Enabled
- Batch size: 500-1000 (maximize GPU)
- Priority: Low (nice +10 on Unix) to not starve other processes
- Progress: Minimal output (reduce I/O overhead)
- **Use case**: Overnight/background indexing, laptop idle

### CLI Integration

```bash
# Default: User-friendly
arc index pdfs ~/Documents/PDFs --collection MyDocs

# Balanced: Faster but still usable
arc index pdfs ~/Documents/PDFs --collection MyDocs --fast

# Maximum: Unattended indexing
arc index pdfs ~/Documents/PDFs --collection MyDocs --turbo
```

**Note**: Current implementation does NOT have `--fast` or `--turbo` flags yet. These are proposed additions in Phase 2. Current flags are: `--workers`, `--model`, `--force`, `--verbose`, `--no-ocr`, `--offline`.

### Four-Phase Strategy

**Phase 1: Foundation (Infrastructure improvements)**
- Add CPU monitoring with `psutil` (visibility into true utilization)
- Increase embedding batch sizes (200-300 chunks) (1.1-1.2x)
- Disable Qdrant indexing during bulk uploads (`m=0`, `indexing_threshold=0`) (1.3-1.5x)
- Add timing instrumentation for validation
- **Combined multiplier**: ~1.4-1.8x (minor but sets foundation for Phase 2)

**Phase 2: Parallelization + GPU (THE BIG WIN - 3-12x realistic speedup)**
- **Parallelize embedding generation** (ThreadPoolExecutor) (2-4x) - **CRITICAL**
- **Parallelize file processing** (ProcessPoolExecutor) (2-4x realistic on 4-8 cores) - **CRITICAL**
- **Enable GPU acceleration** (MPS/CoreML) (1.5-3x realistic on top of parallel) - **HIGHLY RECOMMENDED**
  - Directly attacks the 99% bottleneck (embedding generation)
  - Combined effect: 4x (parallel) × 2x (GPU) = 8x typical, 12x optimistic
  - Low complexity: just add `device="mps"` or `providers=["CoreML"]`
  - Enables larger batch sizes (500-1000) for even better GPU utilization
- Parallelize OCR page processing (3-6x realistic for OCR workloads)
- Implement performance profiles (default/fast/turbo) for user-friendly defaults
- Add progress bars with ETA
- **Combined with Phase 1**: 4-22x realistic (vs 99x theoretical max)
- **This is where 99% of time savings come from**

**Phase 3: Refinements (Additional 1.1-1.3x speedup)**
- Implement embedding cache (SQLite or Redis) (1.1-1.3x)
- Improve metadata sync (aggregation queries, persistent cache) (2-5x for sync phase only, <5% of total time)
- Add comprehensive benchmarks
- **Cumulative with Phases 1-2**: 4-29x realistic

**Phase 4: Architecture (Additional improvements)**
- Streaming architecture (avoid holding all chunks in memory)
- Distributed processing (multiple workers across machines)
- Smart file filtering (skip generated/minified code)
- **Note**: GPU moved to Phase 2 as it directly attacks the critical bottleneck

### Technical Design

#### Phase 1 Implementation Details

**1.1 Enable gRPC for PDF Indexing**

Replace direct `QdrantClient` instantiation with the helper function:

```python
# src/arcaneum/cli/index_pdfs.py:112
# BEFORE:
qdrant = QdrantClient(url="http://localhost:6333")

# AFTER:
from arcaneum.indexing.qdrant_indexer import create_qdrant_client
qdrant = create_qdrant_client(
    url="localhost",
    port=6333,
    grpc_port=6334,
    prefer_grpc=True
)
```

**1.2 Async Uploads with wait=False**

Modify upload logic to batch without blocking:

```python
# src/arcaneum/indexing/qdrant_indexer.py:172
def upload_chunks(self, collection_name, chunks, vector_name):
    # Upload all batches asynchronously
    for i in range(0, len(chunks), self.batch_size):
        batch = chunks[i:i + self.batch_size]
        self.upload_chunks_batch(
            collection_name,
            batch,
            vector_name,
            wait=False  # Don't block on indexing
        )

    # Wait for all operations to complete at the end
    self.client.update_collection_async(
        collection_name=collection_name
    ).wait()
```

**1.3 Increase Batch Sizes**

Update configuration constants:

```python
# src/arcaneum/indexing/uploader.py:29
DEFAULT_BATCH_SIZE = 200  # Was: 100

# src/arcaneum/indexing/qdrant_indexer.py:44
DEFAULT_BATCH_SIZE = 300  # Was: 150

# src/arcaneum/indexing/source_code_pipeline.py:377
EMBEDDING_BATCH_SIZE = 200  # Was: 100

# src/arcaneum/indexing/uploader.py:245
EMBEDDING_BATCH_SIZE = 200  # Was: 100
```

Test and tune based on memory constraints.

#### Phase 2 Implementation Details

**2.1 Parallel Embedding Generation**

Use ThreadPoolExecutor to process multiple embedding batches concurrently:

```python
# src/arcaneum/embeddings/client.py - Add method
from concurrent.futures import ThreadPoolExecutor, as_completed

def embed_parallel(self, texts: List[str], max_workers: int = 4, batch_size: int = 200) -> List[List[float]]:
    """Generate embeddings in parallel batches."""
    all_embeddings = [None] * len(texts)  # Pre-allocate

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for start_idx in range(0, len(texts), batch_size):
            end_idx = min(start_idx + batch_size, len(texts))
            batch = texts[start_idx:end_idx]
            future = executor.submit(self.embed, batch)
            futures[future] = (start_idx, end_idx)

        for future in as_completed(futures):
            start_idx, end_idx = futures[future]
            embeddings = future.result()
            all_embeddings[start_idx:end_idx] = embeddings

    return all_embeddings
```

Update callers to use `embed_parallel()`:

```python
# src/arcaneum/indexing/source_code_pipeline.py:376-398
embeddings = self.embedder.embed_parallel(
    texts=[c.content for c in all_chunks],
    max_workers=4,
    batch_size=200
)
```

**2.2 Parallel File Processing**

Process multiple files concurrently:

```python
# src/arcaneum/indexing/source_code_pipeline.py - Add function
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def _process_single_file(args):
    """Worker function for parallel file processing."""
    file_path, project_root, metadata = args
    try:
        content = file_path.read_text(encoding='utf-8')
        chunks = ASTCodeChunker().chunk_code(content, str(file_path))
        # Attach metadata
        for chunk in chunks:
            chunk.metadata.update(metadata)
        return chunks
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return []

def _index_project(self, project_path, ...):
    # ... existing code to get files and metadata ...

    # Process files in parallel
    max_workers = min(multiprocessing.cpu_count(), len(files))
    file_args = [(f, project_path, metadata) for f in files]

    all_chunks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for chunks in executor.map(_process_single_file, file_args):
            all_chunks.extend(chunks)

    # Continue with embedding and upload...
```

**2.3 Parallel OCR Processing**

```python
# src/arcaneum/indexing/pdf/ocr.py:69-85
from concurrent.futures import ProcessPoolExecutor

def _ocr_single_page(args):
    """Worker function for parallel OCR."""
    page_image, engine_type, tesseract_lang = args
    # ... preprocessing and OCR logic ...
    return page_text

def process_pdf(self, pdf_path: Path) -> str:
    images = convert_from_path(pdf_path, dpi=self.image_dpi)

    max_workers = min(multiprocessing.cpu_count(), len(images))
    page_args = [(img, self.engine, self.tesseract_lang) for img in images]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        page_texts = list(executor.map(_ocr_single_page, page_args))

    return "\n\n".join(page_texts)
```

#### Phase 3 Implementation Details

**3.1 Embedding Cache**

```python
# src/arcaneum/embeddings/cache.py - New file
import hashlib
import sqlite3
import pickle
from pathlib import Path

class EmbeddingCache:
    def __init__(self, cache_dir: Path):
        self.db_path = cache_dir / "embedding_cache.db"
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model_name)")
        conn.commit()
        conn.close()

    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT embedding FROM embeddings WHERE content_hash=? AND model_name=?",
            (content_hash, model_name)
        ).fetchone()
        conn.close()

        if row:
            return pickle.loads(row[0])
        return None

    def put(self, text: str, model_name: str, embedding: List[float]):
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (content_hash, model_name, embedding) VALUES (?, ?, ?)",
            (content_hash, model_name, pickle.dumps(embedding))
        )
        conn.commit()
        conn.close()

# Integrate into EmbeddingClient
class EmbeddingClient:
    def __init__(self, cache_dir: Path, enable_cache: bool = True):
        self.cache = EmbeddingCache(cache_dir) if enable_cache else None
        # ... existing init ...

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.cache:
            return self._embed_uncached(texts)

        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cached = self.cache.get(text, self.current_model)
            if cached:
                results.append(cached)
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            fresh_embeddings = self._embed_uncached(uncached_texts)
            for idx, embedding in zip(uncached_indices, fresh_embeddings):
                results[idx] = embedding
                self.cache.put(texts[idx], self.current_model, embedding)

        return results
```

**3.2 Qdrant Bulk Mode**

```python
# src/arcaneum/indexing/qdrant_indexer.py - Add methods
from qdrant_client.models import OptimizersConfigDiff

def enable_bulk_mode(self, collection_name: str):
    """Disable indexing for bulk upload performance."""
    self.client.update_collection(
        collection_name=collection_name,
        optimizer_config=OptimizersConfigDiff(
            indexing_threshold=0  # Disable indexing
        )
    )
    logger.info(f"Enabled bulk mode for {collection_name}")

def disable_bulk_mode(self, collection_name: str):
    """Re-enable indexing after bulk upload."""
    self.client.update_collection(
        collection_name=collection_name,
        optimizer_config=OptimizersConfigDiff(
            indexing_threshold=20000  # Default
        )
    )
    logger.info(f"Disabled bulk mode for {collection_name}, rebuilding index...")

# Use in upload_chunks
def upload_chunks(self, collection_name, chunks, vector_name, bulk_mode: bool = False):
    if bulk_mode:
        self.enable_bulk_mode(collection_name)

    try:
        # Upload batches with wait=False
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            self.upload_chunks_batch(collection_name, batch, vector_name, wait=False)

        # Wait for completion
        time.sleep(1)  # Brief delay for async operations
    finally:
        if bulk_mode:
            self.disable_bulk_mode(collection_name)
```

**3.3 Metadata Sync Optimization**

```python
# src/arcaneum/indexing/git_metadata_sync.py - Optimize queries
def get_indexed_projects(self, collection_name: str) -> Dict[str, str]:
    """Get indexed projects using aggregation instead of scroll."""
    # Use faceted search or aggregation API (if available)
    # For now, cache aggressively and use larger batch sizes

    cache_file = self.cache_dir / f"{collection_name}_projects.json"
    if cache_file.exists():
        # Check if cache is fresh (< 5 minutes old)
        if time.time() - cache_file.stat().st_mtime < 300:
            with open(cache_file) as f:
                return json.load(f)

    # Fetch with larger batch size
    projects = {}
    offset = None
    while True:
        records, offset = self.client.scroll(
            collection_name=collection_name,
            limit=1000,  # Increased from 100
            offset=offset,
            with_payload=["git_project_identifier", "git_commit_hash"],
            with_vectors=False
        )

        for record in records:
            identifier = record.payload.get("git_project_identifier")
            commit = record.payload.get("git_commit_hash")
            if identifier and commit:
                projects[identifier] = commit

        if not offset:
            break

    # Cache result
    with open(cache_file, 'w') as f:
        json.dump(projects, f)

    return projects
```

#### Phase 4 Architectural Changes

**4.1 Streaming Architecture**

Instead of accumulating all chunks in memory, stream them through the pipeline:

```python
# Use generators to avoid holding all chunks
def _index_project_streaming(self, project_path, ...):
    def chunk_generator():
        for file_path in files:
            chunks = self._process_file(file_path)
            yield chunks

    # Process in batches as they come
    batch = []
    for file_chunks in chunk_generator():
        batch.extend(file_chunks)

        if len(batch) >= 1000:  # Upload threshold
            self._embed_and_upload(batch)
            batch = []

    # Upload remaining
    if batch:
        self._embed_and_upload(batch)
```

**4.2 Smart File Filtering**

```python
# src/arcaneum/indexing/filters.py - New file
SKIP_PATTERNS = [
    "**/node_modules/**",
    "**/*.min.js",
    "**/*_pb2.py",  # Protobuf generated
    "**/*_pb2_grpc.py",
    "**/dist/**",
    "**/build/**",
    "**/.git/**",
]

def should_skip_file(file_path: Path) -> bool:
    """Check if file should be skipped (generated/minified)."""
    for pattern in SKIP_PATTERNS:
        if file_path.match(pattern):
            return True

    # Check if minified (heuristic: long lines)
    if file_path.suffix in ['.js', '.css']:
        with open(file_path) as f:
            first_line = f.readline()
            if len(first_line) > 500:  # Likely minified
                return True

    return False
```

### Implementation Example

See Technical Design section above for complete code examples.

## Alternatives Considered

### Alternative 1: Ray Data for Distributed Processing

**Description**: Use Ray Data framework for distributed embedding generation across multiple machines/GPUs

**Pros**:
- Proven 20x speedup for large-scale operations
- Built-in fault tolerance and load balancing
- Scales horizontally across machines
- Lazy evaluation and optimized scheduling

**Cons**:
- Requires Ray cluster setup and management
- Additional dependency (heavy framework)
- Overkill for single-machine use cases
- Learning curve for maintenance

**Reason for rejection**: While Ray Data is excellent for large-scale distributed systems, it's overly complex for Arcaneum's typical use case (single machine indexing). Our phased approach achieves similar speedups (10-20x) without the operational overhead. Ray could be revisited in Phase 4 if distributed processing is needed.

### Alternative 2: GPU-Only Embedding Strategy

**Description**: Require GPU acceleration for all embedding models, optimize for GPU batch processing

**Pros**:
- Much faster embedding generation (10-100x for large batches)
- Simpler code (no CPU fallback logic)
- Better utilization of modern hardware

**Cons**:
- Requires GPU availability (limits portability)
- Not all models have good GPU support
- Memory constraints (GPU VRAM limits batch sizes)
- Excludes CPU-only users

**Reason for rejection**: Arcaneum aims to be portable and work on any machine, including CPU-only environments. FastEmbed with ONNX Runtime provides good CPU performance. GPU acceleration should be an optional enhancement, not a requirement.

### Alternative 3: Rewrite in Rust/Go

**Description**: Rewrite performance-critical components in Rust or Go for better performance

**Pros**:
- Significantly faster execution (especially for I/O and parsing)
- Better memory efficiency
- True parallelism without GIL constraints
- Type safety and error handling

**Cons**:
- Major development effort (months of work)
- Maintenance burden (two languages)
- Python-Rust/Go interop complexity
- Disrupts existing codebase and workflows

**Reason for rejection**: While a Rust/Go rewrite could provide performance benefits, the proposed Python-based optimizations (parallelization, better configurations) can achieve 10-20x speedups without requiring a complete rewrite. This is a better use of development time. A focused Rust extension for specific bottlenecks (e.g., chunking) could be considered in the future.

## Trade-offs and Consequences

### Positive Consequences

**Performance Gains (Realistic estimates with overhead factored in):**

**Speedup Methodology**: Estimates account for real-world overhead (thread management, synchronization, Amdahl's Law serialization). Theoretical maximums shown in parentheses.

- **Phase 1**: 1.4-1.8x (foundation improvements)
  - Batch size increase: 1.1-1.2x
  - Qdrant bulk mode: 1.3-1.5x
  - Combined: ~1.4-1.8x

- **Phase 2**: Additional 3-12x with parallelization + GPU (**THE BIG WIN - 99% of improvement**)
  - **Default profile**: 2-4x (50% CPU, no GPU) - Laptop responsive
  - **Fast profile**: 6-12x (87% CPU + GPU) - Acceptable responsiveness
  - **Turbo profile**: 10-16x (100% CPU + GPU) - Maximum throughput
  - Note: Theoretical max is 4x (parallel) × 3x (GPU) = 12x, real-world is 60-80% of theoretical due to overhead

- **Phase 3**: Additional 1.1-1.3x with refinements
  - Embedding cache: 1.1-1.3x (depends on duplication rate)
  - Metadata sync: Improves startup only, not main indexing time

- **Phase 4**: Architectural improvements (streaming, distributed)
  - Varies by workload and infrastructure

- **Combined (Phases 1-3, Turbo profile)**: 15-37x realistic overall improvement
  - Conservative estimate: 15x (Phase 1: 1.4x × Phase 2: 10x × Phase 3: 1.1x)
  - Optimistic estimate: 37x (Phase 1: 1.8x × Phase 2: 16x × Phase 3: 1.3x)
  - Theoretical maximum: 55x (if 100% speedups achieved)

**Better Resource Utilization:**
- Multi-core CPUs fully utilized (currently idle during sequential processing)
- Network bandwidth better utilized with gRPC and async uploads
- Memory used more efficiently with streaming architecture (Phase 4)

**Improved User Experience:**
- Faster indexing means quicker time-to-search
- Progress bars and ETAs provide better feedback
- Large-scale indexing becomes practical

**Code Quality:**
- Performance instrumentation aids debugging
- Benchmarks prevent regressions
- Clearer separation of concerns (caching, parallel execution)

### Negative Consequences

**Increased Complexity:**
- Parallel processing introduces concurrency bugs (race conditions, deadlocks)
- More configuration options (workers, batch sizes) increase support burden
- Cache management adds another failure mode

**Memory Usage:**
- Parallel processing increases memory footprint (multiple batches in flight)
- Embedding cache requires disk space (SQLite database grows)
- Multiple workers may cause memory pressure on constrained systems

**Backward Compatibility:**
- API changes may break existing scripts (if signatures change)
- Configuration changes require documentation updates
- Testing across different environments becomes more complex

**Debugging Difficulty:**
- Parallel processing makes logs harder to interpret
- Intermittent concurrency bugs are hard to reproduce
- Performance issues may manifest differently across systems

### Risks and Mitigations

**Risk**: Parallel processing causes race conditions or data corruption
**Mitigation**:
- Use thread-safe data structures (Queue, Lock)
- Extensive testing with various concurrency levels
- Add integration tests that verify data integrity
- Implement graceful degradation (fall back to sequential on errors)

**Risk**: Increased memory usage causes OOM errors
**Mitigation**:
- Add memory monitoring and adaptive batch sizing
- Document memory requirements per configuration
- Implement streaming architecture (Phase 4) to cap memory usage
- Add `--low-memory` mode that reduces parallelism

**Risk**: FastEmbed/ONNX Runtime doesn't handle parallel calls well
**Mitigation**:
- Test thoroughly with different models and batch sizes
- Add model-specific concurrency limits in configuration
- Use process-based parallelism for embedding if threads cause issues
- Fall back to sequential processing if parallel embedding fails

**Risk**: gRPC connection issues or incompatibilities
**Mitigation**:
- Add fallback to HTTP if gRPC fails
- Test across different Qdrant versions
- Document gRPC port requirements
- Add connection testing in CLI doctor command

**Risk**: Cache corruption or stale embeddings
**Mitigation**:
- Version cache database schema (allow migrations)
- Include model name in cache key (different models → different embeddings)
- Add `--clear-cache` flag for troubleshooting
- Implement cache validation on startup (detect corruption)

**Risk**: Performance improvements don't materialize on some systems
**Mitigation**:
- Benchmark across different hardware (Mac M1, Intel, AMD, Linux)
- Make parallelism configurable (`--workers` flag)
- Add performance profiling command to identify system-specific bottlenecks
- Document expected performance characteristics

## Implementation Plan

### Prerequisites

- [x] Comprehensive performance analysis completed (arcaneum-198)
- [x] Bottlenecks identified and prioritized
- [ ] Backup current implementation (git branch or tag)
- [ ] Create performance benchmark suite for validation
- [ ] Set up test environment with sample data (PDFs, source code)

### Step-by-Step Implementation

#### Phase 1: Foundation

##### Step 1.1: Increase Embedding Batch Sizes

**Files**:
- `src/arcaneum/indexing/uploader.py`
- `src/arcaneum/indexing/qdrant_indexer.py`
- `src/arcaneum/indexing/source_code_pipeline.py`

1. Update `DEFAULT_BATCH_SIZE` constants (100→200, 150→300)
2. Update `EMBEDDING_BATCH_SIZE` (100→200)
3. Add CLI options for custom batch sizes (`--batch-size`)
4. Test with different sizes (100, 200, 300, 500)
5. Document optimal values in configuration guide

##### Step 1.4: Add Performance Instrumentation

**Files**: All indexing modules

1. Add timing decorators for key operations
2. Log performance metrics (files/sec, chunks/sec, upload rate)
3. Create summary report at end of indexing
4. Add `--profile` flag for detailed performance logging
5. **Add CPU utilization monitoring (see below)**
6. Test and validate metrics accuracy

**CPU Utilization Monitoring:**

The main `arc` process shows low CPU (e.g., 14%) in `htop` because FastEmbed/ONNX Runtime spawns internal threads that aren't attributed to the parent. To show true CPU usage:

```python
# Add to src/arcaneum/monitoring/cpu_stats.py (new file)
import psutil
import time
from typing import Dict, Optional

class CPUMonitor:
    """Monitor CPU usage including child processes and threads."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_cpu_times = None

    def start(self):
        """Begin monitoring."""
        self.start_time = time.time()
        # Get initial CPU times for process + all children/threads
        self.start_cpu_times = self.process.cpu_times()

    def get_stats(self) -> Dict[str, float]:
        """Get current CPU statistics.

        Returns:
            Dict with:
            - cpu_percent: Overall CPU usage (0-100 per core, can exceed 100)
            - cpu_percent_per_core: Per-core average (0-100)
            - num_threads: Number of threads
            - elapsed_time: Seconds since start()
        """
        elapsed = time.time() - self.start_time if self.start_time else 0

        # Get CPU usage over interval (includes all threads)
        # interval=0 uses cached value, interval=0.1 samples over 100ms
        cpu_percent = self.process.cpu_percent(interval=0.1)

        # Get thread count
        num_threads = self.process.num_threads()

        # Calculate per-core average
        num_cores = psutil.cpu_count()
        cpu_percent_per_core = cpu_percent / num_cores if num_cores else cpu_percent

        return {
            "cpu_percent": cpu_percent,
            "cpu_percent_per_core": cpu_percent_per_core,
            "num_threads": num_threads,
            "num_cores": num_cores,
            "elapsed_time": elapsed
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        stats = self.get_stats()
        return (
            f"CPU: {stats['cpu_percent']:.1f}% total "
            f"({stats['cpu_percent_per_core']:.1f}% per core avg) | "
            f"Threads: {stats['num_threads']} | "
            f"Cores: {stats['num_cores']}"
        )

# Integrate into indexing pipeline
# In src/arcaneum/indexing/source_code_pipeline.py or uploader.py:

from arcaneum.monitoring.cpu_stats import CPUMonitor

def index_pdfs(...):
    monitor = CPUMonitor()
    monitor.start()

    # ... existing indexing code ...

    # Periodically log (e.g., after each file or every 10 files)
    if file_count % 10 == 0:
        logger.info(f"Progress: {file_count} files | {monitor.get_summary()}")

    # Final summary
    logger.info(f"Indexing complete | {monitor.get_summary()}")
```

**Why This Helps:**
- `psutil.Process.cpu_percent()` aggregates CPU usage from all threads
- Shows true utilization including ONNX Runtime worker threads
- Helps identify if embedding generation is the bottleneck (high CPU) or I/O is (low CPU)
- Validates parallelization improvements (should see CPU increase from 14% → 80-100% per core)

**Installation:**
Add `psutil` to dependencies if not already present.

**Validation**: Run benchmark suite, verify 1.4-1.8x speedup vs baseline (minor but measurable)

#### Phase 2: Parallelization + GPU

##### Step 2.1: Enable GPU Acceleration (HIGH PRIORITY)

**Why First**: GPU directly attacks the 99% bottleneck with low implementation complexity. Do this BEFORE parallelization to validate GPU works, then combine for multiplicative gains.

**Model-Specific GPU Compatibility (VERIFIED on macOS M1/M2):**

**✅ GPU-Compatible Models (Recommended):**
- **stella** (SentenceTransformers): ✅ Works with MPS - Tested successfully
- **jina-code** (SentenceTransformers): ✅ Works with MPS - Tested successfully
- **bge-base** (FastEmbed): ✅ Works with CoreML - Tested successfully (768D)
- **bge-small** (FastEmbed): ✅ Works with CoreML - Tested successfully (384D)

**❌ GPU-Incompatible Models (CPU only):**
- **bge-large** (FastEmbed): ❌ CoreML fails - "input dim > 16384" limitation
- **jina-v3** (FastEmbed): ❌ CoreML fails - Model execution error

**Recommendation**: Use **stella** or **jina-code** for GPU acceleration on macOS. These are the primary models (stella for PDFs, jina-code for source code) and both support MPS perfectly.

**macOS Apple Silicon Implementation:**

```python
# src/arcaneum/embeddings/client.py

def __init__(self, cache_dir: str = None, verify_ssl: bool = True, use_gpu: bool = False):
    """Initialize embedding client.

    Args:
        use_gpu: Enable GPU acceleration (MPS for SentenceTransformers, CoreML for FastEmbed)
    """
    self.use_gpu = use_gpu
    self._device = self._detect_device() if use_gpu else "cpu"
    # ... rest of init

def _detect_device(self) -> str:
    """Detect best available device."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_model(self, model_name: str):
    """Get or initialize embedding model with GPU support."""
    config = EMBEDDING_MODELS[model_name]
    backend = config.get("backend", "fastembed")

    if backend == "fastembed":
        # FastEmbed with CoreML
        providers = None
        if self.use_gpu and self._device == "mps":
            try:
                import onnxruntime as ort
                if "CoreMLExecutionProvider" in ort.get_available_providers():
                    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            except Exception as e:
                logger.warning(f"CoreML not available: {e}")

        self._models[model_name] = TextEmbedding(
            model_name=config["name"],
            cache_dir=self.cache_dir,
            providers=providers
        )

    elif backend == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        model_obj = SentenceTransformer(
            config["name"],
            cache_folder=self.cache_dir,
            device=self._device  # "mps" for GPU
        )
        self._models[model_name] = model_obj
```

**CLI Integration with Performance Profiles:**
```python
# src/arcaneum/cli/index_pdfs.py (and index_source.py)

import os
import multiprocessing

# Performance profile options
@click.option('--fast', is_flag=True, help='Balanced profile: faster with acceptable responsiveness')
@click.option('--turbo', is_flag=True, help='Maximum performance: use when laptop unattended')
@click.option('--workers', type=int, help='Manual worker count (overrides profiles)')
@click.option('--batch-size', type=int, help='Manual batch size (overrides profiles)')
@click.option('--gpu/--no-gpu', default=None, help='Manual GPU control (overrides profiles)')
def index_pdfs(fast: bool, turbo: bool, workers: int, batch_size: int, gpu: bool, ...):
    """Index PDFs with user-friendly defaults or turbo mode."""

    # Determine profile
    if turbo:
        profile = 'turbo'
    elif fast:
        profile = 'fast'
    else:
        profile = 'default'

    # Configure based on profile
    cpu_count = multiprocessing.cpu_count()

    if profile == 'default':
        # User-friendly: keep laptop responsive
        workers = workers or max(1, cpu_count // 2)
        batch_size = batch_size or 100
        gpu = gpu if gpu is not None else False
        progress_detail = 'normal'
        click.echo(f"Using default profile: {workers} workers, batch={batch_size}, GPU=off")
        click.echo("Tip: Use --fast for faster indexing or --turbo for maximum performance")

    elif profile == 'fast':
        # Balanced: faster but still usable
        workers = workers or max(1, cpu_count - 1)
        batch_size = batch_size or 300
        gpu = gpu if gpu is not None else True
        progress_detail = 'normal'
        click.echo(f"Using fast profile: {workers} workers, batch={batch_size}, GPU=on")

    elif profile == 'turbo':
        # Maximum: laptop unattended
        workers = workers or cpu_count
        batch_size = batch_size or 1000
        gpu = gpu if gpu is not None else True
        progress_detail = 'minimal'

        # Set low priority to not starve UI
        try:
            os.nice(10)  # Lower priority on Unix
            click.echo(f"Using turbo profile: {workers} workers, batch={batch_size}, GPU=on, low priority")
        except (AttributeError, OSError):
            # Windows or permission denied
            click.echo(f"Using turbo profile: {workers} workers, batch={batch_size}, GPU=on")

        click.echo("Running at maximum performance - laptop may be slow for other tasks")

    # Initialize with configured settings
    embedder = EmbeddingClient(use_gpu=gpu)
    # ... rest of indexing with workers and batch_size
```

**Testing:**
1. Test stella model with GPU: `arc index pdfs ~/test-pdfs --collection TestGPU --model stella --gpu`
2. Monitor GPU usage: Activity Monitor → Window → GPU History
3. Compare times: CPU vs GPU for 1000 chunks
4. Expected: 2-5x speedup for large batches

**Note**: The `--gpu` flag does not currently exist. This is a proposed addition in Phase 2.

##### Step 2.2: Parallel Embedding Generation

**Files**:
- `src/arcaneum/embeddings/client.py`
- `src/arcaneum/indexing/source_code_pipeline.py`
- `src/arcaneum/indexing/uploader.py`

1. Implement `embed_parallel()` method in `EmbeddingClient`
2. Add `max_workers` configuration option
3. Update callers to use parallel embedding
4. Handle thread safety (model loading, ONNX Runtime sessions)
5. Test with various worker counts (1, 2, 4, 8)
6. Monitor for deadlocks or race conditions
7. Add error handling and graceful degradation

##### Step 2.3: Parallel File Processing

**File**: `src/arcaneum/indexing/source_code_pipeline.py`

1. Extract file processing logic into standalone function
2. Implement `ProcessPoolExecutor` worker pool
3. Add `--workers` CLI option (respects performance profiles: default=cores/2, fast=cores-1, turbo=cores)
4. Handle pickling issues (ensure all objects serializable)
5. Test with large codebases (1000+ files)
6. Verify chunk ordering and metadata correctness

##### Step 2.4: Parallel OCR Processing

**File**: `src/arcaneum/indexing/pdf/ocr.py`

1. Extract page OCR logic into standalone function
2. Implement `ProcessPoolExecutor` for page-level parallelism
3. Handle image serialization between processes
4. Add `--ocr-workers` CLI option
5. Test with OCR-heavy PDFs (100+ pages)
6. Verify text extraction accuracy (no page mixing)

##### Step 2.5: Progress Bars and ETAs

**Files**: All indexing CLI modules

1. Integrate `tqdm` for progress visualization
2. Add file/page counters
3. Calculate and display ETAs
4. Show throughput metrics (files/sec, pages/sec)
5. Respect performance profile settings (minimal output in turbo mode)
6. Test across different terminal types

**Validation**: Run benchmark suite with all three profiles (Phase 1 + Phase 2 combined):
- Default profile: 3-7x vs baseline (cores/2, no GPU) - Laptop responsive
- Fast profile: 6-16x vs baseline (cores-1, GPU on) - Acceptable responsiveness
- Turbo profile: 10-22x vs baseline (all cores, GPU on) - Maximum throughput

#### Phase 3: Advanced Optimizations

##### Step 3.1: Embedding Cache

**Files**:
- `src/arcaneum/embeddings/cache.py` (new)
- `src/arcaneum/embeddings/client.py`

1. Implement `EmbeddingCache` class with SQLite backend
2. Integrate into `EmbeddingClient.embed()`
3. Add cache hit/miss metrics
4. Add `--enable-cache` CLI flag (default: enabled)
5. Add `--clear-cache` flag for troubleshooting
6. Test cache correctness (different texts, different models)
7. Test cache performance (hit rate, lookup speed)
8. Document cache location and maintenance

##### Step 3.2: Qdrant Bulk Mode

**File**: `src/arcaneum/indexing/qdrant_indexer.py`

1. Implement `enable_bulk_mode()` method
2. Implement `disable_bulk_mode()` method
3. Add `bulk_mode` parameter to `upload_chunks()`
4. Add CLI flag `--bulk-mode` (default: auto-detect based on size)
5. Test with small collections (no benefit)
6. Test with large collections (should see 1.3-1.5x improvement)
7. Handle errors during index rebuild
8. Document when to use bulk mode

##### Step 3.3: Metadata Sync Optimization

**File**: `src/arcaneum/indexing/git_metadata_sync.py`

1. Increase scroll batch size (100→1000)
2. Implement persistent file-based cache
3. Add cache freshness check (TTL: 5 minutes)
4. Add `--force-resync` flag to bypass cache
5. Test with large collections (10K+ projects)
6. Verify sync correctness (no missed updates)

##### Step 3.4: Comprehensive Benchmarks

**File**: `tests/benchmarks/` (new directory)

1. Create benchmark harness with standardized test data
2. Implement benchmarks for each phase:
   - Baseline (no optimizations)
   - Phase 1 (quick wins)
   - Phase 2 (parallelization)
   - Phase 3 (advanced)
3. Measure: throughput, memory, CPU utilization
4. Generate comparison reports
5. Add CI integration (track performance over time)

**Validation**: Run benchmark suite, verify 1.5-2x additional speedup (12-40x total)

#### Phase 4: Architectural Improvements

##### Step 4.1: Streaming Architecture

**Files**: All indexing pipeline modules

1. Refactor to use generators instead of lists
2. Implement streaming upload (upload batches as they're ready)
3. Add memory monitoring (track peak usage)
4. Test with extremely large projects (100K+ files)
5. Verify memory usage stays bounded
6. Compare performance vs in-memory approach

##### Step 4.2: Distributed Processing (Optional)

**Files**: New distributed processing module

1. Evaluate Ray Data integration complexity
2. Implement distributed embedding generation
3. Add cluster configuration and management
4. Test across multiple machines
5. Document setup and troubleshooting
6. Compare cost/benefit vs single-machine optimization

##### Step 4.3: Smart File Filtering

**Files**:
- `src/arcaneum/indexing/filters.py` (new)
- `src/arcaneum/indexing/source_code_pipeline.py`

1. Implement skip pattern matching
2. Add minification detection heuristics
3. Add CLI flag `--skip-generated` (default: true)
4. Test with JavaScript projects (node_modules)
5. Test with Python projects (protobuf, generated files)
6. Measure time/space savings

**Validation**: Run full benchmark suite, verify additional improvements

### Files to Modify

#### Phase 1 (Quick Wins)
- `src/arcaneum/cli/index_pdfs.py` - Enable gRPC
- `src/arcaneum/indexing/qdrant_indexer.py` - Async uploads, batch sizes
- `src/arcaneum/indexing/uploader.py` - Async uploads, batch sizes
- `src/arcaneum/indexing/source_code_pipeline.py` - Batch sizes

#### Phase 2 (Parallelization)
- `src/arcaneum/embeddings/client.py` - Parallel embedding method
- `src/arcaneum/indexing/source_code_pipeline.py` - Parallel file processing
- `src/arcaneum/indexing/uploader.py` - Parallel file processing (PDFs)
- `src/arcaneum/indexing/pdf/ocr.py` - Parallel page OCR
- All CLI modules - Progress bars

#### Phase 3 (Advanced)
- `src/arcaneum/embeddings/cache.py` - NEW: Cache implementation
- `src/arcaneum/embeddings/client.py` - Cache integration
- `src/arcaneum/indexing/qdrant_indexer.py` - Bulk mode
- `src/arcaneum/indexing/git_metadata_sync.py` - Optimized queries
- `tests/benchmarks/` - NEW: Benchmark suite

#### Phase 4 (Architecture)
- Multiple files - Streaming refactor
- `src/arcaneum/distributed/` - NEW: Distributed processing (optional)
- `src/arcaneum/indexing/filters.py` - NEW: File filtering

### Dependencies

**Phase 1:**
- `psutil` - CPU and memory monitoring (for instrumentation in Step 1.4)

**No additional dependencies for Phase 2-3** (use stdlib and existing packages)

**Phase 4 may require:**
- `ray[data]` - Distributed processing (optional)
- No additional dependencies for GPU support (PyTorch MPS and ONNX Runtime CoreML already available)

## Validation

### Testing Approach

1. **Unit Tests**: Test individual optimizations in isolation
   - Parallel embedding generates correct results
   - Async uploads don't lose data
   - Cache returns correct embeddings

2. **Integration Tests**: Test full pipeline with optimizations
   - Index sample PDF collection, verify search results
   - Index sample codebase, verify search results
   - Compare results with and without optimizations

3. **Performance Tests**: Measure speedups and resource usage
   - Benchmark suite with standardized data
   - Track metrics: throughput, latency, memory, CPU
   - Regression tests (ensure no performance degradation)

4. **Stress Tests**: Test with extreme cases
   - Very large PDFs (1000+ pages)
   - Huge codebases (100K+ files)
   - Limited memory environments
   - High concurrency (many workers)

### Test Scenarios

#### Scenario 1: Small PDF Collection (10 PDFs, 100 pages total)

**Expected Results (cumulative vs baseline):**
- Baseline: 2 minutes (120 seconds)
- Phase 1: 70-85 seconds (1.4-1.7x faster)
- Phase 2 (Default profile): 30-60 seconds (2-4x faster) - Responsive
- Phase 2 (Fast profile): 12-20 seconds (6-10x faster) - Acceptable
- Phase 2 (Turbo profile): 8-15 seconds (8-15x faster) - Maximum

#### Scenario 2: Large PDF Collection (1000 PDFs, 10K pages, some OCR)

**Expected Results (cumulative vs baseline):**
- Baseline: 5 hours (300 minutes)
- Phase 1: 3.3-4.3 hours (1.4-1.8x faster)
- Phase 2 (Default profile): 1-1.5 hours (3-5x faster) - Responsive
- Phase 2 (Fast profile): 30-50 minutes (6-10x faster) - Acceptable
- Phase 2 (Turbo profile): 15-30 minutes (10-20x faster) - Maximum
- Phase 3 (Turbo + cache): 12-25 minutes (12-25x faster)

#### Scenario 3: Medium Codebase (1000 Python files, 100K LOC)

**Expected Results (cumulative vs baseline):**
- Baseline: 10 minutes (600 seconds)
- Phase 1: 6-7 minutes (1.4-1.7x faster)
- Phase 2 (Default profile): 2-3 minutes (3-5x faster) - Responsive
- Phase 2 (Fast profile): 60-100 seconds (6-10x faster) - Acceptable
- Phase 2 (Turbo profile): 30-60 seconds (10-20x faster) - Maximum
- Phase 3 (Turbo + cache): 25-55 seconds (11-24x faster)

#### Scenario 4: Large Codebase (10K files, 1M LOC, multiple repos)

**Expected Results (cumulative vs baseline):**
- Baseline: 2 hours (120 minutes)
- Phase 1: 65-85 minutes (1.4-1.8x faster)
- Phase 2 (Default profile): 30-40 minutes (3-4x faster) - Responsive
- Phase 2 (Fast profile): 12-20 minutes (6-10x faster) - Acceptable
- Phase 2 (Turbo profile): 6-12 minutes (10-20x faster) - Maximum
- Phase 3 (Turbo + cache): 5-11 minutes (11-24x faster)

### Performance Validation

1. **Throughput Metrics**:
   - Files per second
   - Pages per second (PDFs)
   - Chunks per second
   - Embeddings per second
   - Upload rate (points/sec)

2. **Latency Metrics**:
   - Time to first upload
   - Average batch processing time
   - Total indexing time

3. **Resource Metrics**:
   - Peak memory usage
   - CPU utilization (%)
   - Network throughput
   - Disk I/O

4. **Comparison**:
   - Speedup vs baseline (target: 10-40x depending on phase)
   - Memory overhead vs baseline (target: <2x)
   - Result correctness (100% match with sequential processing)

### Security Validation

**Not applicable** - This RDR focuses on performance optimization, not security changes. However, ensure:
- Parallel processing doesn't expose data to unintended processes
- Cache doesn't leak sensitive information (embeddings of private data)
- Distributed processing (Phase 4) maintains data confidentiality

## References

**External Research:**
- [Turbocharge LangChain: Guide to 20x Faster Embedding](https://www.anyscale.com/blog/turbocharge-langchain-now-guide-to-20x-faster-embedding) - Ray Data distributed processing
- [Qdrant Bulk Upload Documentation](https://qdrant.tech/documentation/database-tutorials/bulk-upload/) - Best practices
- [Qdrant Indexing Optimization](https://qdrant.tech/articles/indexing-optimization/) - Memory optimization
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/tune-performance/) - CPU inference optimization

**Internal Documentation:**
- `docs/rdr/RDR-004-pdf-bulk-indexing.md` - Original PDF indexing implementation
- `docs/rdr/RDR-005-source-code-indexing.md` - Original source code indexing implementation
- `docs/testing/testing.md` - Testing guidelines

**Related Issues:**
- arcaneum-198 - Deep performance analysis (this RDR addresses)
- arcaneum-108 - Parallelize PDF indexing (covered in Phase 2)
- arcaneum-120 - Performance benchmarking suite (covered in Phase 3)

## Notes

**Implementation Priority**: Start with Phase 1 (quick wins) immediately. Phase 1 provides the best return on investment (3-5x speedup in 1-2 days). Validate with benchmarks before proceeding to Phase 2.

**Incremental Rollout**: Each phase is independent and can be deployed separately. This allows for:
- Early value delivery (Phase 1)
- Risk mitigation (validate each phase before next)
- Feedback incorporation (tune parameters based on real-world usage)

**Configuration Philosophy**: Balance user-friendliness with performance:

**User-Friendly Defaults (Priority #1):**
- Keep laptop responsive during indexing (50% CPU usage)
- GPU off by default (avoid thermal/battery concerns)
- Moderate batch sizes (100-200)
- Clear progress feedback
- **Goal**: Developer can index while continuing to work

**Performance Profiles for Power Users:**
- `--fast`: 75% CPU, GPU on, larger batches
- `--turbo`: 100% CPU, GPU on, max batches, minimal output
- **Goal**: Maximum throughput when laptop is idle

**Manual Overrides Available:**
- `--workers N`: Explicit worker count
- `--batch-size N`: Explicit batch size
- `--gpu / --no-gpu`: Explicit GPU control
- `--bulk-mode`: Explicit Qdrant indexing mode

**Examples:**
```bash
# Default: User-friendly, laptop stays responsive (PROPOSED)
arc index pdfs ~/Documents --collection MyDocs

# Balanced: Faster, still usable (PROPOSED --fast flag)
arc index pdfs ~/Documents --collection MyDocs --fast

# Maximum: Overnight indexing (PROPOSED --turbo flag)
arc index pdfs ~/Documents --collection MyDocs --turbo

# Current implementation (manual override):
arc index pdfs ~/Documents --collection MyDocs --workers 4 --model stella --verbose
```

**Important**: The `--fast`, `--turbo`, `--gpu`, and `--batch-size` flags are **proposed additions** in this RDR. Current implementation only supports:
- `--workers INTEGER` (parallel workers)
- `--model TEXT` (embedding model)
- `--force` (force reindex)
- `--verbose` (detailed logging)
- `--no-ocr` (disable OCR)
- `--offline` (use cached models)
- `--batch-across-files` (batching strategy)
- `--json` (JSON output)

**Future Work Beyond This RDR**:
- Incremental indexing (only re-index changed files)
- Resume capability (recover from interrupted indexing)
- Multi-collection batch operations (index into multiple collections simultaneously)
- Embedding model quantization (reduce memory footprint)

**GPU Model Compatibility Matrix (Verified on macOS M1/M2)**:

| Model | Backend | GPU Support | Notes |
|-------|---------|-------------|-------|
| stella | SentenceTransformers | ✅ MPS | **Recommended for PDFs** - Full GPU support |
| jina-code | SentenceTransformers | ✅ MPS | **Recommended for source code** - Full GPU support |
| bge-large | FastEmbed | ⚠️ CoreML | Hybrid: 72% on GPU (879/1223 ops), rest on CPU |
| bge-base | FastEmbed | ✅ CoreML | Better CoreML compatibility than bge-large |
| bge-small | FastEmbed | ✅ CoreML | Best CoreML compatibility |
| jina-v3 | FastEmbed | ❌ CPU only | CoreML execution plan build fails (error -2) |

**Key Insight**: The **default recommended models** (stella for PDFs, jina-code for source code) both have **full GPU support** via MPS. This means GPU acceleration "just works" for standard use cases.

**Monitoring and Observability**: Consider adding:
- Prometheus metrics export
- Grafana dashboards for indexing operations
- Alerting on performance degradation
- Structured logging for analysis

**Documentation Updates Required**:
- User guide: Performance tuning section with GPU compatibility matrix
- CLI reference: New flags and options (--fast, --turbo, --gpu)
- Architecture guide: Parallel processing design
- Troubleshooting: Common performance issues and GPU compatibility

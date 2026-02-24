# Recommendation 021: Contextual Embedding Enrichment

> Revise during planning; lock at implementation.
> If wrong, abandon code and iterate RDR.

## Metadata

- **Date**: 2026-02-22
- **Status**: Final
- **Type**: Feature
- **Priority**: Medium
- **Related Issues**: RDR-005 (Source Code Indexing), RDR-014 (Markdown Indexing),
  RDR-004 (PDF Bulk Indexing)

## Problem Statement

Arcaneum's indexing pipelines compute chunk embeddings on raw chunk text only. Structural
context captured during chunking (markdown header paths, source code file paths, PDF
filenames) is stored as Qdrant payload metadata but is invisible to vector similarity
search. When a user searches for "how does Arcaneum connect to Qdrant", a chunk containing
only "The pool maintains 4 connections by default with exponential backoff retry" will rank
poorly despite having `header_path: "Arcaneum Architecture > Qdrant Integration > Connection Pooling"`
in its payload — because cosine similarity operates on the embedding vector alone.

This is a known limitation in RAG systems called "context loss during chunking." The chunk
loses its association with the document's broader topic, entity names, and structural position
once it is embedded as an isolated string.

## Context

### Background

Investigation of Voyage AI's `voyage-context-3` model and its "contextualized chunk embeddings"
approach revealed that Arcaneum already captures the right structural context at chunking time
but discards it at embedding time. The cheapest path to improved retrieval quality is prepending
existing context to chunk text before embedding — no new dependencies, no API costs, no model
changes.

Voyage AI's benchmarks across 93 retrieval datasets show that prepending document metadata to
chunks before embedding with a standard model captures a meaningful portion of the retrieval
improvement. Their `voyage-3-large` with prepended metadata improved retrieval quality vs
without metadata, though `voyage-context-3` (which uses cross-chunk attention internally) still
outperformed by ~5.5%. The metadata prepending approach captures the document-identity signal
(which document is this chunk from, what section, what topic) without requiring cross-chunk
neural attention.

### Technical Environment

- **Markdown pipeline**: `src/arcaneum/indexing/markdown/pipeline.py` — line 216:
  `texts = [chunk.text for chunk in chunks]`
- **Source code pipeline**: `src/arcaneum/indexing/source_code_pipeline.py` — line 712:
  `texts = [chunk.content for chunk in all_chunks]`
- **PDF pipeline**: Uses `chunk.text` from `PDFChunker` output
- **Embedding client**: `src/arcaneum/embeddings/client.py` — `embed()` and `embed_parallel()`
  accept `List[str]`, model-agnostic
- **Chunkers**: All chunkers already produce context metadata:
  - `MarkdownChunk.header_path`: `List[str]` — parent headers on the dataclass; stored in
    `chunk.metadata['header_path']` as a joined string (e.g., `"Architecture > Qdrant > Pooling"`)
  - `MarkdownChunk.metadata`: `Dict` — includes `filename`, `title` (from frontmatter)
  - `CodeChunk.metadata`: `CodeChunkMetadata` dataclass — includes `file_path` (absolute path),
    `programming_language`, `git_project_name`, `git_project_root`
  - PDF `Chunk.metadata`: includes `filename`

## Research Findings

### Investigation

Verified all three pipelines listed in Technical Environment above. In every case, the
`texts` list passed to embedding contains only raw chunk content. Structural context
(`header_path`, `file_path`, `filename`) is computed and stored in chunk metadata/payloads
but never included in the embedding input.

Reviewed Voyage AI documentation and blog post benchmarks:

- Voyage tested `voyage-3-large` with and without prepended metadata on proprietary
  technical documentation and document header evaluation datasets.
- Prepending metadata to a standard model improved retrieval quality measurably.
- `voyage-context-3` without metadata still outperformed `voyage-3-large` with metadata
  by up to 5.53%, demonstrating that cross-chunk neural attention provides additional
  value beyond text prepending.
- The gap between standard and contextualized models is largest with small chunks (6.63%
  at 64 tokens) and narrows with larger chunks (~1-2% at 512 tokens).

### Key Discoveries

- **Documented** — Arcaneum already computes and stores the structural context needed
  for enrichment (`header_path`, `file_path`, `filename`). The change is limited to the
  `texts` list construction, not the chunkers or the embedding client.
- **Documented** — The embedding client (`embed()`, `embed_parallel()`) accepts plain
  `List[str]`. No API changes needed — the enrichment happens before the list is
  constructed.
- **Documented** — Qdrant payload `text` field stores the chunk content for display in
  search results. This field should continue to store the original chunk text (without
  prepended context) so search results remain clean.
- **Assumed** — Prepending a short context prefix (header path or file path) will not
  cause chunks to exceed embedding model `max_seq_length` in typical cases. Needs
  validation for edge cases where chunks are already near the token limit.

### Critical Assumptions

- [x] Context metadata is already available on chunk objects at the point where `texts`
  is constructed — **Status**: Verified — **Method**: Source Search
- [ ] Prepending context prefix does not push typical chunks over model `max_seq_length` —
  **Status**: Unverified — **Method**: Spike (measure prefix lengths vs remaining token
  budget)
- [x] Embedding client is agnostic to input content (accepts any string) —
  **Status**: Verified — **Method**: Source Search
- [ ] Prepending context improves retrieval quality for Arcaneum's typical content —
  **Status**: Unverified — **Method**: Spike (before/after comparison on existing corpus)

## Proposed Solution

### Approach

Prepend structural context to chunk text before embedding, while preserving the original
chunk text in the Qdrant payload for display. The enrichment is a simple string
concatenation at the point where `texts` is constructed in each pipeline. No changes to
chunkers, embedding client, or Qdrant schema.

The context prefix format uses a consistent separator to distinguish context from content.
This follows the pattern used by embedding model providers (Voyage, Jina, BGE) that
prepend task-specific prompts to inputs.

### Technical Design

#### Context Prefix Format

Each pipeline constructs a context prefix from available metadata:

**Markdown**: Use `chunk.metadata['header_path']` (already a joined string) and optional
`title` from frontmatter.

```text
// Illustrative — verify token budget during implementation
chunk.metadata['header_path'] + "\n" + chunk.text
// Example: "Architecture > Qdrant Integration > Connection Pooling\nThe pool maintains 4 connections..."
```

**Source code**: Use file path relative to project root, derived from
`chunk.metadata.file_path` (absolute) minus `chunk.metadata.git_project_root`.

```text
// Illustrative
relative_path + "\n" + chunk.content
// Example: "src/arcaneum/embeddings/client.py\ndef embed(self, texts: List[str]..."
// Note: chunk.metadata.file_path stores absolute paths; compute relative path at enrichment time
```

**PDF**: Use `filename` (and `title` if extracted from PDF metadata).

```text
// Illustrative
filename + "\n" + chunk.text
// Example: "NIST-SP-800-53.pdf\nAccess control policies must be reviewed annually..."
```

#### Per-Corpus Opt-Out

Enrichment is enabled by default (on by default). A `--no-context-enrichment` flag on
`arc corpus sync` disables it for that invocation. The flag is also persisted in
collection metadata so subsequent syncs respect the setting without requiring the flag
each time.

Collection metadata (`collection_metadata.py`) already stores arbitrary key-value pairs
via `set_collection_metadata(**extra_metadata)`. Add a `context_enrichment` boolean
(default `True`) to the metadata point. The sync pipeline reads this value and skips
enrichment when `False`.

```text
// Illustrative — CLI flag
arc corpus sync MyCorpus /path --no-context-enrichment   # disable for this corpus
arc corpus sync MyCorpus /path                            # reads stored setting
arc corpus sync MyCorpus /path --context-enrichment       # re-enable explicitly
```

#### Reindexing With Enrichment

Two paths to upgrade an existing corpus with enriched embeddings:

**Path A: Full reindex (existing)** — `arc corpus sync MyCorpus /path --force`

Re-reads source files, re-chunks, re-embeds, re-uploads. Requires source files on disk.
Bypasses change detection (`sync.py:950`) and deletes existing chunks before re-uploading
(`sync.py:1132`).

**Path B: In-place re-embed (new)** — `arc corpus update --re-embed MyCorpus`

Scrolls through all existing points in Qdrant, reads `payload['text']` and the context
metadata (`header_path`, `file_path`, or `filename`) already stored in payloads, constructs
enriched embedding input via `build_embedding_text()`, re-embeds in batches, and updates
vectors in place via `qdrant.update_vectors()`. Payloads, point IDs, and MeiliSearch index
are untouched.

Advantages of Path B over Path A:

- No source files needed — works entirely from Qdrant payload data
- No re-chunking — preserves existing chunk boundaries exactly
- No MeiliSearch sync — only vectors change
- Faster — skips file I/O, parsing, and chunking stages

Path B is the preferred upgrade path for existing corpora. Path A remains available
for cases where re-chunking is also desired (e.g., after changing chunk size config).

#### Embedding Input vs Display Text

Two distinct values at the point of embedding:

- `embedding_text`: Context prefix + chunk text — passed to `embed_parallel()`
- `payload['text']`: Original chunk text only — stored in Qdrant for search result display

This separation already exists implicitly (the `texts` list is constructed independently
of the `payload` dict). The change only modifies the `texts` list construction.

#### Token Budget Safety

Context prefixes are short relative to chunk sizes:

- Markdown `header_path`: typically 30-80 characters (~10-25 tokens)
- Source code `file_path`: typically 40-120 characters (~12-36 tokens)
- PDF `filename`: typically 20-60 characters (~6-18 tokens)

Against chunk sizes of 460-768 tokens with model `max_seq_length` of 8K-32K tokens, this
is negligible. However, implementation should include a safety check: if `embedding_text`
exceeds the model's `max_seq_length`, truncate the context prefix rather than the chunk
content.

#### Change Locations

Three `texts =` lines change across two pipeline files:

1. `src/arcaneum/indexing/markdown/pipeline.py` — line 216 (markdown sync) and line 658 (PDF inject)
2. `src/arcaneum/indexing/source_code_pipeline.py` — line 712 (source code sync)

Additionally: one new helper module (`context.py`), CLI flag additions, collection metadata
extension, and a new `arc corpus update` command (see Implementation Plan for full scope).

Each `texts =` line changes from:

```python
texts = [chunk.text for chunk in chunks]
```

to a context-enriched variant. Extract a shared helper function to construct the enriched
text to avoid duplicating prefix logic.

#### Helper Function Signature

```text
// Illustrative — place in src/arcaneum/indexing/context.py
def build_embedding_text(content: str, context_prefix: str | None, max_tokens: int | None) -> str
    # Returns context_prefix + "\n" + content, with prefix truncation if needed
```

### Existing Infrastructure Audit

| Proposed Component | Existing Module | Decision |
| --- | --- | --- |
| `build_embedding_text()` helper | No existing equivalent | New function in `indexing/context.py` |
| Per-corpus opt-out flag | `indexing/collection_metadata.py` | Extend: add `context_enrichment` to metadata point |
| Full reindex | `cli/sync.py` (`--force` flag) | Reuse: existing `--force` reindexes all files |
| In-place re-embed | `qdrant.scroll()` + `qdrant.update_vectors()` | New: `arc corpus update --re-embed` command |
| Markdown enrichment | `markdown/pipeline.py:216` | Extend: modify `texts` construction |
| Source code enrichment | `source_code_pipeline.py:712` | Extend: modify `texts` construction |
| PDF enrichment | `markdown/pipeline.py:658` (shared pipeline) | Extend: modify `texts` construction |

### Decision Rationale

Text prepending is chosen over alternatives because:

1. Zero new dependencies — uses existing metadata already computed by chunkers
2. Zero cost — runs on existing local models, no API calls
3. Minimal code change — three `texts =` lines plus one helper function
4. Reversible — if it degrades quality for some content type, disable per-pipeline
5. Compatible with future Voyage integration — if Voyage support is added later,
   the prepending can be disabled for Voyage (which handles context internally)

## Alternatives Considered

### Alternative 1: Voyage AI `voyage-context-3` Integration

**Description**: Add Voyage as a remote embedding provider with `List[List[str]]` grouped
input for cross-chunk neural attention.

**Pros**:

- Best retrieval quality per benchmarks (+7.96% over `voyage-3-large`)
- Model handles context internally — no text prepending needed

**Cons**:

- API-only ($0.18/M tokens) — no offline/local capability
- Requires new embedding client, API key management, rate limiting
- No code-specific model available
- Significant implementation scope

**Reason for rejection**: Not rejected permanently — deferred. Text prepending captures
the document-identity signal now, for free. Voyage integration can be evaluated later if
the remaining ~5% quality gap matters for specific use cases.

### Alternative 2: LLM-Generated Context Summaries (Anthropic Contextual Retrieval)

**Description**: Use an LLM to generate a context summary for each chunk, prepend the
summary before embedding.

**Pros**:

- Richer context than structural metadata alone

**Cons**:

- Requires LLM API call per chunk — expensive at scale
- Adds latency to indexing pipeline
- Voyage benchmarks show it underperforms `voyage-context-3` by 20.54%

**Reason for rejection**: Disproportionate cost and complexity for marginal benefit over
simple metadata prepending.

### Briefly Rejected

- **Embedding metadata in a separate named vector**: Increases storage and query complexity
  without clear retrieval benefit over prepending.

## Trade-offs

### Consequences

- Positive: Improved retrieval for queries that reference document identity (titles,
  filenames, section topics) rather than chunk-local content
- Positive: Zero-cost improvement — no new dependencies or API calls
- Negative: Slightly larger embedding inputs (10-36 extra tokens) — negligible impact on
  throughput
- Negative: Existing indexed collections must be re-indexed to benefit — enrichment only
  affects new embeddings

### Risks and Mitigations

- **Risk**: Context prefix degrades retrieval for some content types (e.g., code chunks
  where file path adds noise rather than signal).
  **Mitigation**: Measure before/after on existing corpora. Per-corpus opt-out via
  `--no-context-enrichment` allows disabling enrichment for specific corpora. Reindex
  with `--force` to revert to non-enriched embeddings.

- **Risk**: Edge case where prepended context pushes chunk over model `max_seq_length`,
  causing silent truncation by the model.
  **Mitigation**: `build_embedding_text()` includes a `max_tokens` parameter that
  truncates the prefix (not the content) if the combined length exceeds the budget.

### Failure Modes

- **Silent quality degradation**: If prepending hurts retrieval for certain content types,
  the failure is not visible — search results simply rank differently. Detection requires
  manual comparison of search quality before and after re-indexing.
  Recovery: Re-index without enrichment (revert the `texts` construction).

- **Truncated prefix**: If the prefix is too long and gets truncated, the enrichment
  provides partial context. This is a graceful degradation, not a failure.

## Implementation Plan

### Prerequisites

- [ ] All Critical Assumptions verified
- [ ] Baseline search quality measurements on at least one markdown and one source code
  corpus (for before/after comparison)

### Minimum Viable Validation

Index a markdown corpus with and without enrichment. Run the same 5-10 representative
queries against both and compare ranking of expected results. Specifically test queries
that reference document-level context (filenames, section topics) vs chunk-local content.

### Phase 1: Code Implementation

#### Step 1: Add Context Helper

Create `src/arcaneum/indexing/context.py` with `build_embedding_text()` function.
Accepts content string, optional context prefix, and optional max token budget.
Returns concatenated string with prefix truncation safety.

#### Step 2: Add Per-Corpus Opt-Out Flag

Add `--no-context-enrichment` / `--context-enrichment` flags to `arc corpus sync` in
`src/arcaneum/cli/main.py`. Store the setting in collection metadata via
`set_collection_metadata()`. Read the setting in the sync pipeline to gate enrichment.
Default is `True` (enrichment enabled).

#### Step 3: Enrich Markdown Pipeline

Modify `src/arcaneum/indexing/markdown/pipeline.py` lines 216 and 658 to construct
`texts` using `build_embedding_text()` with `header_path` (and `title` from frontmatter
when available) as the context prefix. Keep `payload['text']` as original chunk text.
Gate behind the `context_enrichment` setting from collection metadata.

#### Step 4: Enrich Source Code Pipeline

Modify `src/arcaneum/indexing/source_code_pipeline.py` line 712 to construct `texts`
using `build_embedding_text()` with the file's relative path as the context prefix.
Keep `payload['text']` as original chunk content. Gate behind the `context_enrichment`
setting from collection metadata.

#### Step 5: Add `arc corpus update` Command

New CLI command for in-place mutations on an existing corpus without re-reading source
files. Initial implementation supports `--re-embed` flag. Future extensions can add
other mutation flags (e.g., metadata backfills, payload migrations).

`--re-embed` scrolls all points in a Qdrant collection, reads `payload['text']` and
context metadata (`header_path`, `file_path`, or `filename`), constructs enriched
embedding input, re-embeds in batches via `embed_parallel()`, and calls
`qdrant.update_vectors()` to replace vectors in place. Respects the `context_enrichment`
collection metadata setting. Supports `--verbose` and `--batch-size` flags. Skips the
metadata point (`METADATA_POINT_ID`).

Accepts an optional `--model` flag to re-embed with a different model than the one
originally used. This enables model migration (e.g., switching from `bge` to `stella`)
without re-reading source files. When `--model` is provided, the new model must already
be configured in the collection's named vectors. If not, the command fails with a
clear error.

```text
// Illustrative — CLI usage
arc corpus update MyCorpus --re-embed                           # re-embed current model + enrichment
arc corpus update MyCorpus --re-embed --model stella            # re-embed with different model
arc corpus update MyCorpus --re-embed --no-context-enrichment   # re-embed without enrichment
```

Scroll uses `qdrant.scroll()` with `limit=256, with_payload=True, with_vectors=False`
(note: `scroll()` defaults to `limit=10`, so an explicit limit is required for efficient
iteration). Pagination uses the returned offset to fetch subsequent pages until `None`.
Context field selection is derived from collection type:

- `markdown` / `pdf`: `payload.get('header_path') or payload.get('filename')`
- `code`: relative path derived from `payload.get('file_path')` minus
  `payload.get('git_project_root')` (both are stored in the Qdrant payload by
  `CodeChunkMetadata.to_payload()`)

#### Step 6: Validate

Run minimum viable validation: index a corpus with enrichment, compare search quality
against the same corpus indexed without enrichment.

### Day 2 Operations

| Resource | List | Info | Delete | Verify | Backup |
| --- | --- | --- | --- | --- | --- |
| Enriched embeddings in Qdrant | N/A (same collections) | N/A | N/A | N/A | N/A |
| `context_enrichment` metadata | `arc corpus info` | In scope | `--context-enrichment` to reset | N/A | N/A |

No new persistent resources created. Existing collections are re-indexed in place.
The `context_enrichment` boolean is stored in the existing collection metadata point.

### New Dependencies

None. Uses only existing metadata from chunkers and existing embedding models.

## Validation

### Testing Strategy

1. **Scenario**: Markdown chunk with `header_path` "Architecture > Qdrant > Pooling"
   and text "The pool maintains 4 connections" — search for "Qdrant connection pooling"
   **Expected**: Enriched embedding ranks this chunk higher than non-enriched

2. **Scenario**: Source code chunk from `src/arcaneum/embeddings/client.py` containing
   a generic helper function — search for "arcaneum embedding client"
   **Expected**: Enriched embedding ranks this chunk higher due to file path in vector

3. **Scenario**: Chunk already at `max_seq_length` boundary with long file path prefix
   **Expected**: Prefix is truncated gracefully, chunk content is fully preserved

4. **Scenario**: Unit test for `build_embedding_text()` — verify concatenation, prefix
   truncation, and None/empty prefix handling

### Performance Expectations

Embedding throughput should be unchanged — the additional 10-36 tokens per chunk is
within noise for batch processing. No measurable impact on indexing time expected.

## Finalization Gate

> Complete each item with a written response before
> marking this RDR as **Final**.

### Contradiction Check

No contradictions found between research findings and proposed solution. The approach
is consistent with Voyage AI's own benchmarks showing metadata prepending improves
retrieval quality for standard embedding models.

### Assumption Verification

Two assumptions remain unverified and require spikes before implementation:

1. Token budget safety — measure actual prefix lengths against model limits
2. Retrieval quality improvement — before/after comparison on existing corpus

### API Verification

| API Call | Library | Verification |
| --- | --- | --- |
| `embed_parallel(texts, model_name)` | `EmbeddingClient` | Source Search — accepts `List[str]`, content-agnostic |
| `PointStruct(payload={'text': ...})` | `qdrant-client` | Source Search — `Payload = Dict[str, Any]`, arbitrary dict |
| `QdrantClient.scroll(limit, with_payload, with_vectors)` | `qdrant-client` | Source Search — returns `(list[Record], Optional[PointId])` for pagination |
| `QdrantClient.update_vectors(collection_name, points)` | `qdrant-client` | Source Search — takes `Sequence[PointVectors]`, preserves payloads |

### Scope Verification

Minimum viable validation (before/after search quality comparison) is in scope as
Phase 1 Step 6. It will be executed during implementation using an existing corpus.

### Cross-Cutting Concerns

- **Versioning**: N/A — no schema changes
- **Build tool compatibility**: N/A
- **Licensing**: N/A — no new dependencies
- **Deployment model**: N/A
- **IDE compatibility**: N/A
- **Incremental adoption**: Enrichment is per-pipeline; can be enabled one pipeline
  at a time. Existing non-enriched collections continue to work.
- **Secret/credential lifecycle**: N/A
- **Memory management**: Negligible — context prefix adds ~100 bytes per chunk string.
  No change to streaming upload pattern.

### Proportionality

This RDR is right-sized. The change is small (three lines + one helper) but the
rationale and context (Voyage AI research, gap analysis) warrant documentation for
future reference when evaluating deeper contextual embedding approaches.

## References

- [Voyage AI Contextualized Chunk Embeddings](https://docs.voyageai.com/docs/contextualized-chunk-embeddings)
- [Voyage AI Blog: voyage-context-3](https://blog.voyageai.com/2025/07/23/voyage-context-3/)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- `src/arcaneum/indexing/markdown/pipeline.py` — markdown embedding pipeline
- `src/arcaneum/indexing/source_code_pipeline.py` — source code embedding pipeline
- `src/arcaneum/indexing/markdown/chunker.py` — markdown chunker with `header_path`
- `src/arcaneum/indexing/ast_chunker.py` — AST code chunker
- `src/arcaneum/embeddings/client.py` — embedding client

# Recommendation 027: Qdrant as a Per-Corpus Full-Text Backend Beside MeiliSearch

> Revise during planning; lock at implementation.
> If wrong, abandon code and iterate RDR.

## Metadata

- **Date**: 2026-11-20
- **Status**: Draft
- **Type**: Architecture
- **Priority**: High
- **Related Issues**: this RDR is `arcaneum#161`; decides `arcaneum#122`; part of the remote-arc program
  `arcaneum#150`; rests on the S1 spike `arcaneum#154` and the load spike `arcaneum#159`; carries the request-size fix
  of `arcaneum#160`; extends RDR-026 Q6 (filter dialect) and Q7 (text ranking); RDR-024 and `arcaneum#118`-`#120`
  tuned the MeiliSearch path this RDR makes optional

## Problem Statement

MeiliSearch is arc's heaviest local service and the one thing that keeps search from living in a single store. On the
laptop it holds 5.2 GiB at idle under an 8 GiB limit, and a single file re-sync costs it 6-30 s of
delete-by-filter on the full production index. Qdrant 1.20 can serve full-text search from a payload text index over
the chunk text it already stores, and two spikes found it better and lighter (`arcaneum#154`, `arcaneum#159`).

The owner asked for both backends, not a replacement (2026-11-20): "let's make a plan to allow arc to support qdrant
for both full text and vector. arc should retain support for meili. we should also figure out a way to extract
snippets from meili and re-index them into qdrant while retaining the existing vector search. so we can extract the
full text, stop meili, then index in qdrant. once per corpus."

This RDR settles: the backend seam, where a corpus records its backend, the Qdrant text index, query translation,
hits and totals, the filter dialect across backends, the write paths, status and parity, the one-time migration and
its rollback, when MeiliSearch becomes optional, and the Qdrant upgrade the feature needs.

## Context

### Background

- **S1 (`arcaneum#154`, Qdrant v1.20.2 vs MeiliSearch v1.57, five corpora, 1,249 queries):** Qdrant's text index,
  requiring every term, beat MeiliSearch on oracle P@10/R@10 in every query class (phrase 95.9/94.8 vs 88.9/90.7). It
  built in 64 s vs 257 s on a sixth of the disk, and re-added 50 chunks in 0.11 s vs 1.13 s. Without typo tolerance
  0.3% of queries lose every hit. Phrase p95 was 25 ms vs 4 ms.
- **Load spike (`arcaneum#159`, laptop container limits):** Qdrant-only cut everyday sync CPU to 0.18 of tuned
  MeiliSearch and search p95 during syncs to 16 ms from 469 ms. It freed 6.45 GiB at idle and 7.42 GiB at peak. A
  `cold` text index removed the small memory stalls a `pinned` one showed. Rebuild CPU was 0.79 of MeiliSearch's bulk
  load.
- **Owner direction:** retain MeiliSearch; migrate once per corpus; cleanup belongs to `arc corpus repair` ("also we
  have arc repair to clean things up in the future"); release main before any of this lands.

### Technical Environment

- arc 0.11.0 (0.12.0 is main as released before this work); Python >= 3.12.
- `qdrant-client>=1.18.0`, locked at 1.18.0 (`pyproject.toml:38`, `uv.lock`); Qdrant `v1.18.2`
  (`deploy/docker-compose.yml:5`).
- `meilisearch>=0.43.0`, locked at 0.43.0; MeiliSearch `v1.54.0` (`deploy/docker-compose.yml:40`).
- Support window before 1.0: Qdrant 1.18.x, MeiliSearch 1.54.0. Changing either needs a release note and an upgrade
  test from the previous supported image (`docs/reference/release-compatibility-policy.md`).
- 16 corpora on the laptop, 731,313 chunks; PapersFast holds 421,078 of them.

## Research Findings

### Investigation

- **Store parity, read-only, all 16 corpora** (evidence `remote-arc/store-text-parity.md`): every MeiliSearch document
  against every Qdrant point, by id and text.
- **Code surveys** of every MeiliSearch touchpoint and of the Qdrant schema, metadata and search seams; cited inline
  below.
- **Interaction logs, 2026-01-15 to 2026-11-20** (evidence `remote-arc/search-usage-filters.md`): 19,389 text and
  14,605 semantic searches, every filter classified by arc's own parser.
- **Qdrant documentation:** upgrades (no skipped minors), the full-text index (`scoring`, `phrase_matching`), the
  `TextQuery`, and payload index memory tiers.
- **Spikes S1 and the load spike**, as above.

#### Dependency Source Verification

| Dependency | Source Searched? | Key Findings |
| --- | --- | --- |
| qdrant-client 1.18.0 | Yes (`.venv/.../qdrant_client/http/models/models.py`) | `TextIndexParams` has `tokenizer`, `lowercase`, `ascii_folding`, `phrase_matching`, `stopwords`, `on_disk`; `MatchPhrase`, `MatchText`, `MatchTextAny` exist as filters; no text-scored query in the `Query` union |
| qdrant-client 1.20.x | No (docs and release notes only) | `models.TextQuery(text=...)` with `using=<field>`; `TextIndexParams(scoring=...)`; release notes: "BM25 is now supported through `TextQuery`" |
| Qdrant server 1.20 | Spike (S1, load spike, REST) | text index with `scoring`, `phrase_matching`, custom stop words and `memory: cold` or `pinned` accepted over REST; BM25 `TextQuery` with `using: "text"` |
| MeiliSearch 1.54 | Yes (arc's use of it) | arc sends only `q`, `limit`, `offset`, `filter`, `attributesToHighlight`, `showRankingScore`, `attributesToRetrieve` (`fulltext/client.py:362-404`) |

### Key Discoveries

- **Verified: Qdrant already holds the text, under the same ids.** All 731,313 chunks are in both stores with
  byte-identical text (MeiliSearch `content`, Qdrant payload `text`) under the same id. The only exceptions are 2
  MeiliSearch stubs in PapersFast that have an id, filename and path but no content. Sync gives each chunk one uuid4
  (`cli/sync.py:4231-4232`), which both `to_qdrant_point` and `to_meilisearch_doc` use (`schema/document.py:276`,
  `:295`). A migration therefore builds an index; it does not re-extract or re-embed.
- **Documented: two paths break the shared id.** The MeiliSearch→Qdrant backfill re-chunks from disk and mints new
  ids with a thinner payload (`cli/sync.py:6001-6052`). The `arc index text *` indexers write MeiliSearch-only
  documents with their own ids and shapes (page, function or whole-file documents).
- **Documented: arc creates one payload index, `metadata_type`** (`indexing/common/sync.py:475-496`). Every
  `file_path` and `chunk_index` filter (sync deletes, renames, passage windows) runs unindexed. One `file_path` count
  on PapersFast took 18.9 s (Verified).
- **Documented: reads must not write.** The retrieval core is read-only, and tests fail if a read creates a payload
  index (`tests/unit/retrieval/test_passage.py:134-135`). Indexes are created on create and sync.
- **Documented: the hit total is cosmetic.** `estimated_total_hits` reaches only the CLI message "Found N results"
  (`cli/fulltext.py:148-152`). The JSON `data` and the MCP output carry no total (`retrieval/hits.py:281-289`).
- **Documented: highlighting is cosmetic.** `_formatted` is read only by the human text formatter
  (`cli/fulltext.py:169-184`). JSON, compact and MCP output take the start of the chunk.
- **Documented: the multi-corpus text merge is round-robin** (`retrieval/text.py:215-229`), so BM25's unbounded scores
  and MeiliSearch's 0-1 scores never need to be compared.
- **Verified: filters are rare, and every native MeiliSearch filter observed is a conjunction.**
  - 272 of 33,995 searches (0.8%) used a filter.
  - Of 46 native MeiliSearch text filters, 40 are only `key="quoted value"` (arc's parser treats any double quote as
    MeiliSearch syntax), 5 join comparisons with `AND`, and 1 is a plain `AND`.
  - None uses OR, NOT, `!=`, IN, TO, EXISTS or IS NULL/EMPTY.
- **Verified: text search is an agent's tool.** 99% of text searches come from Claude. Since August, 53.8% are quoted
  phrases and 10.5% return nothing even with typo tolerance on.
- **Documented: Qdrant forbids skipping minor versions on upgrade**, so the path is 1.18.2 → 1.19.2 → 1.20.x.
- **Verified (box): a `cold` text index over 646,025 chunks built in 71 s within 2 CPUs** (`arcaneum#159`,
  `reindex-cold`).

### Critical Assumptions

- [x] qdrant-client 1.20 exposes `TextQuery`, and its `TextIndexParams` carries `scoring`, `phrase_matching`, custom
  `stopwords` and a memory placement for text indexes. — **Status**: Verified (client 1.20.0 against Qdrant 1.20.2;
  `using` is a `query_points` argument) — **Method**: Spike (C1 of the assumptions spike).
- [x] An in-place upgrade 1.18.2 → 1.19.2 → 1.20.x keeps every collection, point and vector intact. — **Status**:
  Verified on StateMachineRes copied into fresh 1.18.2 storage: identical id, vector and payload digests and top-10
  results after each step — **Method**: Spike (C2).
- [x] The text index covers every chunk whose text yields a token. — **Status**: Verified per chunk on DevRef and
  StateMachineRes. S1's DevRef gap is exactly the 1,090 chunks that yield no token (underscores, backticks, `###` and
  `**` left by extraction), so coverage is measured against tokenful chunks (Q8) — **Method**: Spike (C3).
- [x] A chunk is findable by a distinctive span of its own text. — **Status**: Verified as membership (the chunk's own
  id AND `MatchPhrase`, all 1,218 sampled chunks). As a top-10 search it fails 1.2% of chunks, whose spans other
  chunks share, so Q9's gate counts membership — **Method**: Spike (C4).
- [x] Both stores hold the same text under the same ids. — **Status**: Verified — **Method**: Spike (store parity).
- [x] Qdrant's text search meets the latency bar during syncs within the laptop's 2-CPU, 4 GiB envelope. —
  **Status**: Verified on the box (p95 23.2 ms cold) — **Method**: Spike (`arcaneum#159` B5).

## Proposed Solution

### Approach

Put a text-backend seam in front of every MeiliSearch use, add a Qdrant implementation over a text index on the
existing `text` payload, and let each corpus choose its backend. A one-time, per-corpus command checks the stores
agree, copies MeiliSearch's text into any Qdrant point that lacks it, builds the index, proves it, and switches the
corpus. MeiliSearch stays supported, and becomes optional once no corpus needs it.

### Technical Design

#### Q1. The backend seam

A `TextBackend` protocol in the retrieval core, with two implementations, `MeiliTextBackend` and
`QdrantTextBackend`. Callers hold a backend chosen per corpus, never a MeiliSearch client.

```text
// Illustrative — verify API signatures during implementation
search(corpus, query, filter: ParsedFilter | None, limit, offset) -> list[RawHit]   // id, text, score, payload
add(corpus, docs: list[DualIndexDocument]) -> None
delete_files(corpus, file_paths: list[str]) -> None
rename(corpus, moves: list[tuple[old_path, new_path]]) -> None
ensure(corpus, corpus_type) -> None          // index and settings; called on create and sync, never on read
drop(corpus) -> None
coverage(corpus) -> Coverage                 // chunks, searchable chunks, per-file counts for parity
health() -> None                             // raises ServiceDownError
```

`MeiliTextBackend` wraps `FullTextClient`. The call sites that reach past it to the SDK move behind it:

- renames (`cli/sync.py:1958-2017`);
- the Qdrant→MeiliSearch backfill (`cli/sync.py:5509-5704`);
- metadata repair (`cli/sync.py:5306-5506`);
- corpus info and items (`cli/corpus.py:581-995`);
- PDF promotion (`cli/sync.py:404-621`).

`search_text`, `get_passage` and `list_corpora` take backends instead of a MeiliSearch client. Their duck-typed seams
(`retrieval/text.py:124-213`, `retrieval/passage.py:237-337`, `retrieval/corpora.py:495-577`) become the protocol.

#### Q2. Where a corpus records its backend

`text_backend: "meili" | "qdrant"` on the collection metadata point, where the type, models and `source_roots` live
(`indexing/collection_metadata.py`). An absent key means `meili`.

- `arc corpus create --text-backend` sets it (`cli/corpus.py:379-385`).
- Only the migration changes it later.
- Search reads it in the same metadata read as `source_roots` (`retrieval/hits.py:177-214`).
- Sync takes it from the metadata it already loads (`cli/sync.py:2729-2738`).
- `arc corpus list|info` show it.

#### Q3. The Qdrant text index and payload indexes

On `text`, as measured in S1 and `arcaneum#159`:

- word tokenizer, lowercase, ASCII folding, `phrase_matching: true`;
- BM25 `scoring`;
- MeiliSearch's 11 English stop words (`fulltext/indexes.py:74-117`);
- no stemming;
- memory `cold`.

Every corpus also gets a keyword index on `file_path` and an integer index on `chunk_index`, for sync deletes, renames
and passage windows. Indexes are created idempotently on corpus create and at the start of sync, following
`ensure_file_manifest_payload_index` (`indexing/common/sync.py:475-496`), and never by a read. The `file_path` and
`chunk_index` indexes go on every corpus, whatever its backend.

#### Q4. Query translation

S1's `all` variant, the one that beat MeiliSearch:

- Each quoted phrase becomes a `MatchPhrase` filter on `text`.
- The remaining words become one `MatchText` filter (every term required).
- Results are ordered by a BM25 `TextQuery` over the whole query (`using: "text"`).
- If nothing matches, it falls back to the BM25 query alone (any term), as S1 did; S1 showed `any` alone regresses
  (A4), so it is only a fallback.
- Metadata points are excluded by `metadata_exclusion_filter`.

The query string keeps its documented syntax: words, plus double quotes for exact phrases (`mcp_server.py:101-107`).

#### Q5. Hits, scores and totals

- **Hits:** a Qdrant hit goes through `build_hit` with `content_keys=("text",)`, and its BM25 score is `score`. The
  round-robin merge stays (RDR-026 Q7), so BM25 and MeiliSearch scores are never compared.
- **One field set:** a Qdrant text hit's payload is projected onto the fields a MeiliSearch document carries
  (`schema/document.py::to_meilisearch_doc`) before `build_hit`, so `--verbose` `metadata` is the same on both
  backends. Without the projection, a Qdrant hit's `metadata` would add every other payload field (the whole
  `quality_manifest` among them, `arcaneum#160`) and keep payload names where MeiliSearch renames them
  (`programming_language`, not `language`). Snippets and cites need no change: both come from `text`, `file_path`,
  `page_number`, `start_line`/`end_line` and `section`/`header_path`, which every chunk point already carries.
- **No totals, for either backend:**
  - Text search reports the number of hits returned, as semantic search does.
  - `estimated_total_hits` and `processing_time_ms` leave `TextSearchResult`.
  - An exact Qdrant total would cost a second filtered query per corpus on every search, for one line of text.
- **Highlighting:** the human formatter highlights query terms client-side for Qdrant hits, in place of
  `_formatted`.
- **Zero hits:** a text search that returns nothing says so and suggests `arc search semantic`, the step agents
  already take.
- **Typo tolerance:** none on Qdrant corpora, and no fuzzy fallback (see Alternatives).

#### Q6. One filter dialect across three backends

RDR-026 Q6's neutral dialect grows to cover what agents actually type:

- quoted values, single or double (`filename="a, b.pdf"`), which also stops a value's comma from splitting terms;
- `AND` joins;
- the symbolic comparisons `=`, `>`, `>=`, `<`, `<=`.

`filename="x.pdf" AND page_number >= 39` then parses as neutral and runs on MeiliSearch, Qdrant text and semantic
search alike; semantic refuses it today.

- **MeiliSearch's remaining syntax** (OR, NOT, `!=`, IN, TO, EXISTS, IS NULL/EMPTY) stays native and passes through
  to MeiliSearch corpora only.
- **On a Qdrant corpus**, that syntax is `bad-filter`, with a message giving Qdrant's JSON form
  (`must`/`should`/`must_not`), which Qdrant text search accepts as semantic search does.
- **The filterable-field check** uses MeiliSearch's `filterableAttributes` for MeiliSearch corpora. For Qdrant
  corpora it uses the corpus type's payload fields, so `chunk_index` and `section_type` filter there, while a field
  that no chunk carries (`page`) is still refused by name.
- **`contains`** works on Qdrant text, as it does in semantic search.
- **Mixed corpora:** when one search spans both backends, a native form one backend refuses fails the whole call as
  `bad-filter`, naming the corpora it would not run on.

#### Q7. Writes per backend

- A Qdrant corpus writes nothing extra: the text is already in the point.
- `DualIndexer.index_batch` skips MeiliSearch when the corpus's backend is `qdrant`, and sync deletes and renames go to
  the corpus's backend only.
- Sync's MeiliSearch gate (`cli/sync.py:2717-2727`) applies only to MeiliSearch corpora.
- Upserts split by serialized size as well as by count, so no file's batch exceeds Qdrant's 32 MiB request limit
  (`arcaneum#160`).

#### Q8. Status, parity and doctor

For a Qdrant corpus, "parity" becomes text coverage: chunks that yield a token against chunks the text index holds.
A chunk whose text yields no token under Q3's rules (only punctuation and markup, 1,090 of DevRef's chunks) is never in
the index and matches no query on either backend. Sync and the migration's reconcile tokenize each chunk with Q3's
rules, the same tokenizer Q5's highlighting needs, and store `text_tokens: false` (an indexed bool) on a chunk that
yields none. Coverage is complete when the index's points equal the chunk points minus those flagged. Flagged chunks
are reported as a chunking finding (`arcaneum#128`), not a coverage gap; the full reindex is where the chunker drops
them. These commands follow the corpus's backend:

- `arc corpus list|info|items|verify|doctor|parity|repair`;
- `arc doctor`.

JSON output gains backend-neutral keys (`text_backend`, `text_chunks`) beside the existing `meili_*` keys, which stay
for MeiliSearch corpora. `research doctor` reads only `.data.corpora[].name` (RDR-026), which is unchanged.

#### Q9. The one-time migration

`arc corpus migrate-text <corpus> --to qdrant|meili [--dry-run] [--json]`, under the corpus write lock.

`--to qdrant`:

1. **Preflight:**
   - Qdrant ≥ 1.20; the corpus is on `meili`; MeiliSearch answers; the collection is green.
   - `arc corpus verify` reports parity. Otherwise the command stops and names the `arc corpus repair` run that fixes
     it; the migration repairs nothing itself.
2. **Reconcile** by streaming MeiliSearch's documents (`id`, `content`) against Qdrant's chunk points (`id`, `text`):
   - **Equal:** nothing to do.
   - **A Qdrant point with no `text`, an older `content` key, or different text:** `set_payload` copies MeiliSearch's
     content in. The vector and id are untouched. This is the "extract from MeiliSearch" step, and it finds nothing
     today.
   - **MeiliSearch stubs without content:** reported and skipped (2 today).
   - **Anything else** (MeiliSearch documents with content but no Qdrant point): reported; the corpus is left on
     `meili` with the repair to run.
3. **Index:** ensure Q3's indexes, wait for green, and require the coverage of Q8 to be complete.
4. **Prove:**
   - **Gate:** a sample of tokenful chunks (1%, at least 200), each with its most distinctive span of its own text
     (highest summed IDF): an exact count of the chunk's own id AND `MatchPhrase` of the span must be 1 for every one.
     Ranking is not gated, since a span is often shared by other chunks.
   - **Report, not gate:** the share of sampled chunks in the top 10 of their own span's search, and a query sample
     through both backends with its top-10 overlap.
5. **Switch:** set `text_backend: qdrant` with a migration stamp (time, counts, arc and Qdrant versions).
6. **Keep MeiliSearch's index:** it is left in place and goes stale. `--finalize` deletes it.

`--to meili` sets the corpus back to `meili`, then refills MeiliSearch from Qdrant through repair's existing
Qdrant→MeiliSearch backfill (`cli/sync.py:5509-5704`, whose `git_*` field names are corrected to MeiliSearch's).
Rollback is therefore the same command. `--dry-run` runs steps 1, 2 and the coverage count, and writes nothing.

#### Q10. MeiliSearch becomes optional

In 0.14.0, after the corpora have migrated and soaked:

- New corpora default to `qdrant`.
- The MeiliSearch compose service moves behind a profile, so `arc container start` leaves it stopped unless some
  corpus uses it.
- `arc doctor` and `arc corpus doctor` require MeiliSearch only when some corpus is on `meili`.
- The MeiliSearch volume is kept until `--finalize` has run on every corpus.

#### Q11. Out of scope

`arc index text *` and `arc indexes *` stay MeiliSearch-only. They are the single-system commands CLAUDE.md marks as
advanced, with their own document shapes and ids. The full reindex that picks up new embedding models
(`arcaneum#150`) is separate. It will write Qdrant only, and carries `arcaneum#160`'s payload cleanup.

Lexical and semantic fusion (`arcaneum#84`) is a follow-on. A Qdrant-text corpus holds both retrievals in one
collection under one id per chunk, so fusion there could be one Query API request (a BM25 prefetch and a dense
prefetch fused by rank) with no cross-system deduplication. Whether a text-index `TextQuery` can serve as a prefetch
in 1.20 is unverified, and this RDR does not depend on it.

#### Q12. Qdrant upgrade

The server moves 1.18.2 → 1.19.2 → 1.20.x in place, with every collection snapshotted first. The `qdrant-client`
floor and lock move to 1.20. The pins move in:

- `deploy/docker-compose.yml`;
- CI (`.github/workflows/test.yml`);
- `deploy/README.md`;
- the compatibility policy, with its release note and an upgrade test from 1.18.2.

`arc container upgrade` steps through each minor version rather than jumping.

### Existing Infrastructure Audit

| Proposed Component | Existing Module | Decision |
| --- | --- | --- |
| `TextBackend`, `MeiliTextBackend` | `fulltext/client.py` (`FullTextClient`) | Extend: wrap it, and move the SDK bypasses behind it |
| `QdrantTextBackend.search` | `search/searcher.py:79-169` (`search_collection`) | Reuse its metadata exclusion and `query_points` call shape |
| Hit building | `retrieval/hits.py` (`build_hit`, `search_data`) | Reuse unchanged |
| Multi-corpus fetch | `cli/search_merge.py` (`fetch_from_corpora`) | Reuse unchanged |
| Filter dialect | `search/filters.py` | Extend: quoted values, `AND`, symbolic comparisons; Qdrant text path |
| `text_backend` setting | `indexing/collection_metadata.py` | Extend: one key |
| Payload index creation | `indexing/common/sync.py:475-496` | Reuse the pattern for `text`, `file_path`, `chunk_index` |
| Migration reconcile | `cli/sync.py:5509-5704` (Qdrant→MeiliSearch backfill) | Reuse for `--to meili`; fix field names |
| MeiliSearch→Qdrant backfill | `cli/sync.py:6001-6052` | Leave for MeiliSearch corpora; not used by the migration (it re-chunks and mints ids) |
| Passage windows | `retrieval/passage.py:257-289` | Reuse; it already resolves Qdrant ids first and filters on `file_path` and `chunk_index` |

### Decision Rationale

- **Lighter and better at once:** the spikes show Qdrant's text index beats MeiliSearch on quality, CPU, memory and
  latency during syncs.
- **The migration is cheap and reversible:**
  - The text is already in Qdrant under the same ids, so switching a corpus costs an index build and a check.
  - Every vector stays.
  - MeiliSearch's index stays until the owner finalizes, so a rollback loses nothing.
- **The contracts don't change for callers:**
  - Hits keep their schema and cites.
  - Filters keep one dialect, grown to cover what agents type.
  - The only lost behaviours are typo tolerance and a cosmetic total.

## Alternatives Considered

### Alternative 1: Replace MeiliSearch outright

**Description**: drop MeiliSearch and serve text search from Qdrant for every corpus.

**Pros**:

- One store, one code path.

**Cons**:

- Contradicts the owner's direction to retain MeiliSearch.
- Removes the `arc index text *` and `arc indexes *` workflows.
- Leaves no rollback.

**Reason for rejection**: the owner asked for both backends; a per-corpus switch gets the same end state for every
migrated corpus.

### Alternative 2: Migrate by re-extracting or re-embedding

**Description**: rebuild each corpus into Qdrant from its sources (or through the MeiliSearch→Qdrant backfill).

**Pros**:

- Would also refresh extraction and models.

**Cons**:

- Re-embedding is the costliest work arc does (76% of sync time), and a rebuild needs every source on disk.
- The existing backfill mints new ids, which breaks the shared-id property cites and passages rely on.
- Unneeded: the text is already in Qdrant.

**Reason for rejection**: store parity shows Qdrant holds the text. The full reindex for new models is a separate,
later step (`arcaneum#150`).

### Alternative 3: Sparse BM25 vectors instead of a text index

**Description**: store a sparse BM25 vector per chunk (Qdrant's IDF modifier) and query it.

**Pros**:

- Available before 1.20.

**Cons**:

- Writes a new vector for every chunk, which is a reindex.
- Needs a tokenizer of arc's own at index and query time.
- No phrase matching: phrases would still need `MatchPhrase` filters over a text index.

**Reason for rejection**: the 1.20 text index does BM25 and phrases over the payload already there, and S1 measured
it.

### Alternative 4: A fuzzy fallback for typo tolerance

**Description**: when a Qdrant text query finds nothing, retry with fuzzy or prefix matching.

**Pros**:

- Recovers the 0.3% of queries that typo tolerance rescues.

**Cons**:

- Qdrant's text index has no fuzzy matching, so this means a second index (prefix tokenizer) or client-side edits.
- In S1, 2 of the 818 hits typo tolerance added contained the query's terms.
- The callers are agents copying exact identifiers and phrases, and they already fall back to semantic search.

**Reason for rejection**: it adds a second mechanism to recover hits that are mostly noise.

### Alternative 5: A pinned text index

**Description**: keep the text index in memory (`pinned`, Qdrant's default).

**Pros**:

- p95 15.8 ms against 23.2 ms cold.

**Cons**:

- 3.94 GiB at idle against 1.59 GiB.
- 1.67 GiB anonymous against 0.10 GiB.
- Memory stalls at the laptop's 4 GiB limit (0.11-0.23% of phase time).

**Reason for rejection**: memory pressure is the problem this work exists to relieve, and cold already meets the
latency bar by a wide margin.

### Briefly Rejected

- **Exact totals by a Qdrant count:** a second filtered query per corpus per search, for a line only humans see.
- **Refusing all MeiliSearch syntax on Qdrant corpora:** would break 43 of the 46 native text filters that worked,
  which are quoted values and `AND` joins the grown neutral dialect covers.
- **Translating all of MeiliSearch's syntax to Qdrant:** OR, NOT and EXISTS never appear in 10 months of logs, and
  Qdrant JSON already expresses them.
- **Writing both stores during a transition:** doubles the load the change removes; the retained MeiliSearch index
  plus `--to meili` covers rollback.
- **SQLite FTS5:** measured in S1 (fast and small), but a third store, outside Qdrant, with its own sync path.

## Trade-offs

### Consequences

- **Positive:**
  - A Qdrant corpus needs no MeiliSearch at search or sync time.
  - Sync deletes and passage windows become indexed (`file_path`, `chunk_index`) for every corpus.
  - Filters reach every payload field on Qdrant corpora.
- **Negative:**
  - Qdrant corpora lose typo tolerance.
  - Text search loses its "Found N" estimate for every corpus.
  - Two backends mean two code paths behind one protocol until MeiliSearch corpora are gone.
  - Status and parity commands must branch on the backend.

### Risks and Mitigations

- **Risk**: the text index misses some chunks.
  **Mitigation**: the migration requires complete coverage before it switches (Q9 step 3), and the membership gate
  proves findability. The spike found no tokenful chunk missing; S1's DevRef gap was zero-token chunks.
- **Risk**: the in-place Qdrant upgrade damages a collection.
  **Mitigation**: snapshots before each step and the policy's upgrade test. The in-place steps kept a real collection
  byte-identical (spike C2), and S1 verified snapshot recovery into 1.20 as the fallback.
- **Risk**: phrase queries are slower than MeiliSearch's (S1: 25 ms vs 4 ms p95).
  **Mitigation**: still an order of magnitude under the 200 ms bar. During syncs MeiliSearch is the slow one (469 ms).
- **Risk**: an agent's habitual MeiliSearch filter fails on a Qdrant corpus.
  **Mitigation**: the grown dialect covers every form seen in the logs, and the error names the JSON equivalent.

### Failure Modes

- **Qdrant below 1.20:** `migrate-text`, and any search on a Qdrant corpus, fail with a version error naming the
  upgrade.
- **The text index is missing on a Qdrant corpus** (restored from an old snapshot, or dropped by hand): the search
  fails with a coded error naming `arc corpus repair`. Sync recreates the index, and repair reports coverage.
- **A corpus records `qdrant` but MeiliSearch still has its index:** harmless. The index is stale and unused until
  `--finalize`.
- **A migration is interrupted:** the switch is the last write, so the corpus stays on `meili`. A rerun is
  idempotent: the indexes exist, and the reconcile finds nothing.

## Implementation Plan

### Prerequisites

- [ ] Release 0.12.0 from main as it stands, before any of this lands (owner, 2026-11-20).
- [ ] All Critical Assumptions verified.

### Minimum Viable Validation

On the laptop after the Qdrant upgrade:

1. `arc corpus migrate-text StateMachineLit --to qdrant`.
2. The MCP `search_text` returns cite-ready hits for a quoted phrase with a `filename="..." AND page_number >= N`
   filter.
3. `arc corpus sync` re-indexes a changed file without MeiliSearch running.
4. `arc corpus migrate-text StateMachineLit --to meili` restores MeiliSearch search with the same results as before.

### Phase 1: Code Implementation (0.13.0)

#### Step 1: Qdrant 1.20 and payload indexes

The upgrade path of Q12, the client lock, the pins, the policy note and upgrade test, and the `file_path` and
`chunk_index` indexes ensured on create and sync.

#### Step 2: The backend seam

`TextBackend` and `MeiliTextBackend`, with the SDK bypasses moved behind it. No behaviour change; existing tests stay
green.

#### Step 3: The filter dialect

Quoted values, `AND` joins and symbolic comparisons in the neutral parser. The Qdrant-corpus field check and refusal
message. Contract tests over every filter form in `remote-arc/search-usage-filters.md`.

#### Step 4: The Qdrant backend and the switch

`QdrantTextBackend` (Q3-Q5), the `text_backend` setting, `arc corpus create --text-backend`, the write paths of Q7,
the `text_tokens` flag of Q8, and the size-split upserts.

#### Step 5: Status surfaces

Q8 across list, info, items, verify, doctor, parity and repair, and `arc doctor`.

#### Step 6: The migration

`arc corpus migrate-text` (Q9) with `--dry-run`, `--to meili` and `--finalize`.

### Phase 2: Operational Activation

#### Activation Step 1: Migrate the laptop's corpora

Smallest first, PapersFast last, each with `--dry-run` first. Stop if a gate fails.

#### Activation Step 2: Soak, then 0.14.0

Normal use with every corpus on `qdrant` and MeiliSearch running but unused. Then the default flip and the optional
MeiliSearch service of Q10.

### Day 2 Operations

| Resource | List | Info | Delete | Verify | Backup |
| --- | --- | --- | --- | --- | --- |
| Text index on `text` | In scope (`arc corpus info`) | In scope (coverage) | With the collection | In scope (`arc corpus verify`) | In the collection snapshot |
| `file_path`, `chunk_index` indexes | In scope | In scope | With the collection | In scope | In the collection snapshot |
| `text_backend` setting and migration stamp | In scope (`arc corpus list`) | In scope | `--to meili` | In scope | In the metadata point |
| Retained MeiliSearch index | In scope (`arc indexes list`) | In scope | `--finalize` | N/A (stale by design) | Existing MeiliSearch backup |

### New Dependencies

- `qdrant-client>=1.20` (Apache-2.0, already a dependency; floor raised). Qdrant image `v1.20.x` (Apache-2.0).

## Validation

### Testing Strategy

1. **Scenario**: a Qdrant corpus answers the S1 query set.
   **Expected**: P@10 and R@10 at or above MeiliSearch's for each query class on the same corpus.
2. **Scenario**: every filter form in the usage evidence, on a MeiliSearch corpus, a Qdrant corpus, and semantic
   search.
   **Expected**: the neutral and grown forms run on all three; OR, NOT and EXISTS are `bad-filter` on Qdrant with the
   JSON hint.
3. **Scenario**: a file changes, is renamed, and is removed on a Qdrant corpus.
   **Expected**: chunks are replaced, moved and deleted with no MeiliSearch call. Coverage stays complete.
4. **Scenario**: `migrate-text --to qdrant`, then `--to meili`, on a fixture corpus.
   **Expected**: identical hit ids for a phrase query before and after the round trip.
5. **Scenario**: a migration interrupted after indexing and before the switch.
   **Expected**: the corpus is still on `meili`, and a rerun completes.
6. **Scenario**: the MCP tools on a mixed set of corpora.
   **Expected**: one hit schema, round-robin merged, cites unchanged.
7. **Scenario**: the same chunk returned by `arc search text --verbose --json` before and after its corpus migrates.
   **Expected**: identical `cite`, `snippet` and `metadata` keys and values; only `score` differs.
8. **Scenario**: a fixture corpus with a chunk of only punctuation and markup, migrated and then synced.
   **Expected**: the chunk carries `text_tokens: false`, coverage reports complete, and doctor lists the chunk as a
   chunking finding.

### Performance Expectations

From the spikes, under the laptop's container limits:

- text search p95 of tens of milliseconds during syncs, against MeiliSearch's 469 ms;
- a text index over all corpora in about a minute and a half of build;
- 6-7 GiB less memory than today's pair once MeiliSearch stops.

The laptop itself is unmeasured; the Minimum Viable Validation records its numbers.

## Finalization Gate

> Complete each item with a written response before
> marking this RDR as **Final**. Written responses
> prevent rubber-stamping and produce a review record.

### Contradiction Check

No contradictions found between research findings, design principles, and proposed solution. RDR-026 Q6 rejected a
neutral-only dialect to keep native MeiliSearch filters working. This RDR keeps that passthrough for MeiliSearch
corpora, and covers the observed native forms in the neutral dialect for Qdrant corpora.

### Assumption Verification

Every Critical Assumption is verified. The 1.20 client API, the in-place upgrade, text index coverage and phrase
findability rest on the assumptions spike
(`arcaneum-evidence:notes/remote-arc/spike-2026-11-21-qdrant-120-assumptions`); text parity and latency on the earlier
spikes.

#### API Verification

| API Call | Library | Verification |
| --- | --- | --- |
| `query_points(query=TextQuery(...), using="text")` | qdrant-client 1.20.0 | Spike (C1) |
| `create_payload_index(field_schema=TextIndexParams(scoring, phrase_matching, stopwords, memory))` | qdrant-client 1.20.0 | Spike (C1) |
| `MatchPhrase`, `MatchText` filters | qdrant-client 1.20.0 | Spike (C1) |
| `set_payload(points=[id], payload={"text": ...})` | qdrant-client 1.18.0 | Source Search (used at `indexing/common/sync.py:1551-1565`) |

### Scope Verification

The Minimum Viable Validation is in scope as the last step of Phase 1 and runs on the laptop before any other corpus
migrates.

### Cross-Cutting Concerns

- **Versioning**:
  - 0.12.0 releases main first.
  - 0.13.0 carries the upgrade, the backend and the migration, with the policy's release note.
  - 0.14.0 flips the default.
- **Build tool compatibility**: N/A
- **Licensing**: N/A (dependencies already present)
- **Deployment model**: one Qdrant serves both search modes, which simplifies the remote box of `arcaneum#150`.
- **IDE compatibility**: N/A
- **Incremental adoption**: per corpus, with rollback; MeiliSearch stays supported.
- **Secret/credential lifecycle**: N/A (no new secrets; the MeiliSearch key is unchanged while MeiliSearch runs)
- **Memory management**: the cold text index keeps anonymous memory near 0.1 GiB. The reconcile streams both stores
  page by page.

### Proportionality

Right-sized for a backend change touching search, sync, status and operations. Q10 and Q11 could shrink to
paragraphs if the owner prefers.

## References

- Evidence repo `arcaneum-evidence/notes/remote-arc/`:
  - `store-text-parity.md`;
  - `search-usage-filters.md`;
  - `spike-2026-11-19-qdrant-text-vs-meili/README.md` (S1);
  - `spike-2026-11-19-index-load-qdrant-vs-meili/README.md` (load spike);
  - `spike-2026-11-21-qdrant-120-assumptions/README.md` (assumptions spike).
- RDR-024 (adaptive cross-file index batching); RDR-026 (MCP server, Q6 filter dialect, Q7 text ranking).
- Qdrant documentation:
  - Upgrades: <https://qdrant.tech/documentation/upgrades/>;
  - Full-text search: <https://qdrant.tech/documentation/search/text-search/full-text-search/>;
  - Indexing: <https://qdrant.tech/documentation/manage-data/indexing/>;
  - qdrant-client releases: <https://github.com/qdrant/qdrant-client/releases>.
- Issues: `arcaneum#122`, `#150`, `#154`, `#159`, `#160`, `#161`.

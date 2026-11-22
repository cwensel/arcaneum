# Moving to Qdrant-only text search

A new install runs Qdrant alone: its corpora serve full-text search from a BM25
text index in their Qdrant collection. An install that already ran MeiliSearch
keeps it, and its corpora keep searching text on MeiliSearch, until you move
them. This guide moves every corpus to Qdrant and then removes MeiliSearch.
Every vector and chunk id stays; nothing is re-extracted or re-embedded.

## Who needs this

You need this guide when either holds:

- `~/.config/arcaneum/meilisearch.key` exists (any earlier `arc container start`
  wrote it), or `MEILISEARCH_API_KEY` is set
- `arc corpus list` shows `meili` in the `Text` column for any corpus

A corpus already on `qdrant` needs no migration. Settings are described in
[Optional MeiliSearch](cli-reference.md#optional-meilisearch). This guide is not
[Qdrant Migration](qdrant-migration.md), which moves Qdrant's storage from bind
mounts to named volumes.

## Steps

Keep `meilisearch.enabled: true` until step 7: the migration reads each
corpus's text from MeiliSearch.

### 1. Back up

```bash
arc container backup
```

The backup holds Qdrant snapshots and every MeiliSearch index. `arc container
restore <dir>` rebuilds both if anything below goes wrong.

### 2. Upgrade Qdrant to 1.20

A `qdrant` corpus needs Qdrant 1.20 or later. `arc corpus migrate-text`
refuses an older server and names this command:

```bash
arc container upgrade --dry-run   # Show the upgrade plan
arc container upgrade             # Snapshot, then step Qdrant to the pinned version
```

### 3. Dry-run every corpus

```bash
arc corpus migrate-text --all --to qdrant --dry-run
```

`--all` takes every corpus whose text is on `meili`, one at a time. For each it
checks the Qdrant version, that MeiliSearch holds the corpus's index, that the
collection is `green`, and that Qdrant and MeiliSearch agree chunk for chunk.
It then counts what the reconcile would copy into Qdrant. Nothing is written.
A corpus that fails a check is reported and skipped; the rest go on, and the
command exits non-zero when any corpus failed. The summary names each one.

### 4. Fix each corpus the dry run names

Each failure names the command that unblocks it:

- **Stores disagree** (`mismatch` or `qdrant_only` in the message): run
  `arc corpus parity <corpus> --verify`. It compares per-file chunk counts and
  copies each mismatched file's chunks from Qdrant to MeiliSearch, without
  re-extracting or re-embedding, so it is much faster than a full repair. If it
  leaves the corpus out of parity, run `arc corpus repair <corpus>`.
- **MeiliSearch holds chunks Qdrant lacks** (`meili_only`): run
  `arc corpus repair <corpus>`.
- **Collection not green**: Qdrant's optimizer is still running; wait and rerun.
- **MeiliSearch has no index for the corpus**: see
  [Troubleshooting](#troubleshooting).

Rerun step 3 until it passes for every corpus.

### 5. Migrate every corpus

```bash
arc corpus migrate-text --all --to qdrant
```

For each corpus, under its write lock: it copies MeiliSearch's text into any
chunk point that lacks it, builds the text index, requires the index to cover
every chunk with searchable text, proves a 1% sample (at least 200) of chunks
findable by their own text, and only then switches the corpus to `qdrant`. A
corpus that stops stays on `meili`. A rerun takes only the corpora still on
`meili`, so fix what the summary names and run the command again.

MeiliSearch keeps each migrated corpus's index until step 6, so you can still
roll back.

### 6. Verify search with MeiliSearch stopped

```bash
docker stop meilisearch-arcaneum
arc corpus list                                    # Text column reads qdrant for every corpus
arc search text "a term you know" --corpus <corpus>  # Repeat for each corpus
docker start meilisearch-arcaneum
```

`arc corpus list` reports MeiliSearch as unreachable while it is stopped; that is
expected here. Every `arc search text` must still return results, since a
`qdrant` corpus never contacts MeiliSearch.

### 7. Finalize

```bash
arc corpus migrate-text --all --finalize --dry-run   # Name the indexes it would delete
arc corpus migrate-text --all --finalize             # Delete each retained MeiliSearch index
```

`--all --finalize` takes every corpus on `qdrant`. A corpus created on `qdrant`
has no MeiliSearch index, and is reported as holding none.

### 8. Turn MeiliSearch off

Edit `~/.config/arcaneum/config.yaml`:

```yaml
meilisearch:
  enabled: false
corpus:
  default_text_backend: qdrant   # new corpora no longer ask for MeiliSearch
```

Then restart the services:

```bash
arc container stop
arc container start
```

`arc container stop` removes the MeiliSearch container. `arc container start`
starts Qdrant alone; it refuses to finish if any corpus is still on `meili`,
naming each one and its `arc corpus migrate-text <corpus> --to qdrant` command.

### 9. Remove MeiliSearch's data

The three MeiliSearch volumes stay until you delete them:

```bash
docker rm -v meilisearch-arcaneum 2>/dev/null   # Only if a container is left behind
docker volume rm arcaneum_meilisearch-arcaneum-data \
  arcaneum_meilisearch-arcaneum-dumps \
  arcaneum_meilisearch-arcaneum-snapshots
```

## Rollback

- **Before step 7**: `arc corpus migrate-text <corpus> --to meili` switches the
  corpus back, clears its retained MeiliSearch index and refills it from Qdrant.
  `--all` does not roll back; run it per corpus.
- **After step 7 or 9**: set `meilisearch.enabled: true`, run
  `arc container start`, then `arc corpus migrate-text <corpus> --to meili` for
  each corpus. It recreates the index and fills it from Qdrant.

## Troubleshooting

- **The gate fails**: the message counts the sampled chunks that did not match
  their own text. `--json` names them: `data.results[].report.gate.failures`
  under `--all` (`data.gate.failures` for one corpus), each with the chunk `id`
  and the `span` it was queried by. The corpus stays on `meili`.
- **Zero-token chunks**: a chunk whose text yields no index token (only
  punctuation or stop words, for example) is flagged and left out of the text
  index's coverage and the gate's sample; the reconcile's `flagged` count reports
  them. This is expected.
- **`chunk_mismatch` in `arc corpus list`, or `mismatch` in the migration's
  message**: run `arc corpus parity <corpus> --verify`, then rerun.
- **Collection not green**: rerun once Qdrant's optimizer finishes.
- **The corpus lock is held**: another process is writing the corpus, usually a
  sync, repair or hook drain, which `arc corpus jobs` lists. Rerun when it
  finishes. Under `--all` the corpus is reported and the rest go on.
- **MeiliSearch has no index for the corpus**: a collection made with
  `arc collection create` records no text backend, so it reads as `meili`.
  Create its MeiliSearch index from Qdrant with
  `arc corpus parity <corpus> --create-missing`, then rerun the migration.
- **`arc indexes` and `arc index text`**: these stay MeiliSearch-only. Their
  indexes are not corpora, so the migration leaves them alone, and they need
  `meilisearch.enabled: true`.

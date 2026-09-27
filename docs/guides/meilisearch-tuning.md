# MeiliSearch Indexing Performance Tuning

How the MeiliSearch container is sized, why those numbers were chosen, and
what to re-measure if indexing slows down again.

## Symptom this addresses

`arc corpus sync` logs repeated warnings during indexing:

```text
Timed out waiting for MeiliSearch task 38122; retrying wait (2/3)
```

These warnings are **not** a defect. `_wait_for_task_with_retries`
(`src/arcaneum/fulltext/client.py`) deliberately keeps polling while the server
reports a task as `processing`, so a slow task is waited out rather than
abandoned. The warning means indexing is slow, not that anything failed.

The underlying cost is that MeiliSearch merge cost scales with **total index
size**, not with batch size. A 50-document task against a 350,000-document
index pays a full-index merge.

## Measured impact of the 2026-08-19 tuning

Same corpus (`PapersFast`, ~348,000 documents), same workload shape, measured
from the server's own `GET /tasks` durations.

| | Before | After |
| --- | --- | --- |
| Throughput (recent tasks) | 0.62 docs/s | 4.89 docs/s |
| Mean task duration | 88.6 s | 9.2 s |

Matched single-task comparison at the same index size:

| Task | Documents | Duration |
| --- | --- | --- |
| 38134 (before) | 51 | 213.4 s |
| 38185 (after) | 50 | 9.1 s |

Roughly a 20x improvement in wall-clock indexing time.

## v1.12.8 to v1.54.0 upgrade (2026-09-27)

Migrated in place with `arc container upgrade`: 13 indexes, 1.22M documents,
23.7 GiB. The JSONL backup (4.1 GB) took about 7 minutes. The
`upgradeDatabase` task took 101 s. Every index kept its document count, and
`arc corpus verify` shows Qdrant/MeiliSearch counts matching.

Baseline from the v1.12.8 task history, last 100 successful document additions
per index. Batched tasks share their batch's duration.

| Index | Documents | Median task | p90 task | Docs/task (median) | Docs/s |
| --- | --- | --- | --- | --- | --- |
| `Claude` | 655,954 | 187.2 s | 352.4 s | 263 | 1.00 |
| `PapersFast` | 347,751 | 21.1 s | 88.0 s | 61 | 2.38 |
| `DevRef` | 101,060 | 6.9 s | 10.6 s | 282 | 26.68 |

Merge-cost probe, re-adding 50 unchanged documents (see "Measuring"). The
first run follows a restart, with a cold page cache. Host load averaged about
21 on 12 cores during the v1.54.0 runs, so treat single values as noisy.

| Index | v1.12.8 cold | v1.12.8 warm | v1.54.0 cold | v1.54.0 warm |
| --- | --- | --- | --- | --- |
| `Claude` | 57.8 s | 22.2-27.1 s | 33.5 s | 1.5-4.2 s |
| `PapersFast` | 36.2 s | 2.5-2.8 s | 7.5 s | 1.4-8.2 s |
| `DevRef` | 3.5 s | 0.5 s | 2.4 s | 0.4-0.7 s |

The largest index gained most: warm merges on `Claude` are 6-15x faster.
Smaller indexes were already cheap and did not change measurably.

v1.54.0 reports per-step timings in `GET /batches/<uid>`
(`stats.progressTrace`), which v1.12 lacked. In the cold `Claude` run,
`post processing facets > facet search` took 26.7 s of 33.5 s. Warm runs
spend about 1.2 s in `post processing words > word fst`. Facet search is
unused by `arc search text`, so disabling it (kata `3h9w`) targets the largest
remaining step.

## What was actually wrong

The binding constraint was the **Docker VM**, not any MeiliSearch setting.

The VM was allocated 8,092 MiB. The compose file requested 4 GiB for Qdrant
plus 4 GiB for MeiliSearch — 8,192 MiB, slightly more than the whole VM. Under
contention MeiliSearch could not reach even its configured 2.5 GiB indexing
budget, so merges spilled instead of running in memory.

This is why raising `MEILI_MAX_INDEXING_MEMORY` alone would not have helped:
the memory was not available to claim. **Check the VM size before tuning
container limits.**

## Current configuration

Set in `deploy/docker-compose.yml`:

| Setting | Value |
| --- | --- |
| Docker Desktop VM memory | 14 GiB |
| MeiliSearch container limit | 8 GiB / 8 CPU |
| `MEILI_MAX_INDEXING_MEMORY` | 6GiB |
| `MEILI_MAX_INDEXING_THREADS` | 6 |

Both `MEILI_*` values are env-overridable, so they can be swept without editing
the compose file:

```bash
MEILI_MAX_INDEXING_MEMORY=4GiB arc container start
```

### Why 6 threads

MeiliSearch's [documentation][ram-threads] states the indexer targets at most
half the available processing units, and warns that allowing full core usage
degrades search latency during indexing. On a 12-core host that is 6.

Thread count matters on v1.12 and later specifically: the ["Indexer edition
2024"][indexer-2024] rewrite made merging parallel by hash-partitioning
database keys. On earlier versions merging was single-threaded and this setting
had little effect.

Re-checked against v1.54.0 (2026-09-27): the upstream thread default is still
half the processing units. Since v1.27 the batched task size defaults to half
of `MEILI_MAX_INDEXING_MEMORY`. Both steady-state values stay.

### Restores, reindexes, and upgrades

The half-cores rule protects search latency. While nothing is being searched,
such as during a full restore or reindex, give the indexer every core:

```bash
MEILI_MAX_INDEXING_THREADS=12 MEILI_CPUS=12 arc container start
arc container restore <backup>
arc container start   # back to the steady-state values
```

`arc container upgrade` does this automatically for its `--upgrade-db` phase,
then restarts MeiliSearch on the steady-state values.

### Why the VM matters more than the container limit

The container limit is not what made indexing fast. LMDB memory-maps the index,
so merge speed depends on how much of the index stays in the **Docker VM's page
cache** — memory the container's own accounting never shows.

Measured on the tuned VM: MeiliSearch reports ~3 GiB resident, while the VM
holds **10+ GiB in `buff/cache`**. That cache is the index, and it is what
turned 213 s merges into sub-second ones.

The cache is cold after any restart, so the first task pays to fault the index
back in. Measured on a 14 GiB VM, re-adding 50 unchanged documents:

| Run | Server duration |
| --- | --- |
| 1 (cold cache) | 11.5 s |
| 2 (warm) | 0.8 s |
| 3 (warm) | 0.7 s |

Do not judge a configuration by the first task after a restart.

Size the VM against the index on disk:

```bash
docker exec meilisearch-arcaneum du -sh /meili_data/data.ms
```

At 14.2 GiB total (of which `PapersFast` is ~7.6 GiB, its largest single
working set), a 14 GiB VM keeps the active corpus cached with room for both
containers. Dropping the VM far below the working set is what reintroduces the
slow path — not lowering the container limit.

`MEILI_MAX_INDEXING_MEMORY` is 6 GiB inside an 8 GiB container. The gap is
deliberate: the setting budgets the *indexer only*, and the process still needs
room for search structures.

## Tuning further

The current values are known-good, not proven optimal. Two open questions:

1. **How low can the VM go?** 14 GiB is sized to keep the ~7.6 GiB
   `PapersFast` working set cached, not to the ~3 GiB the process reports.
   20 GiB and 14 GiB both hold the full speedup; below the working set the
   merge cost is expected to return. Trimming toward the resident figure will
   look harmless at idle and then degrade merges once the cache no longer holds
   the index — the original 2.5 GiB configuration also looked comfortable at
   idle. If the VM must shrink further, re-measure with the probe below rather
   than trusting `docker stats`.
2. **Where is the new knee in the curve?** Merge cost still scales with index
   size. This tuning improved the constant factor; the same wall will be hit at
   a larger corpus size.

### Measuring

Read authoritative server-side durations from the task queue rather than timing
the client, which includes embedding generation:

```bash
curl -s "http://localhost:7700/tasks?indexUids=PapersFast\
&statuses=succeeded&types=documentAdditionOrUpdate&limit=40" \
  -H "Authorization: Bearer $MEILI_KEY"
```

Each result carries `duration` (ISO-8601) and `details.indexedDocuments`.
Compare docs/second across a fixed file set before and after any change, and
note the highest task `uid` beforehand so the two runs stay separable.

Peak client-side memory during sync is available via `--mem-probe-interval`,
which writes JSONL to `~/.arcaneum/logs/arc-mem-*.jsonl`.

To measure merge cost alone, without embedding time, re-add existing documents
unchanged. Fetching from `GET /indexes/<name>/documents` and POSTing the same
records back submits identical `documentAdditionOrUpdate` work and pays the
same full-index merge, but changes no data — the document count must be
identical afterwards. Run it two or three times: the first result reflects a
cold page cache.

## Options deliberately not taken

- **Lowering `proximityPrecision` to `byAttribute`.** This would cut merge cost
  materially, but degrades phrase-proximity ranking. Search quality is the
  reason larger embedding models are used on smaller corpora; trading it for
  indexing speed is the wrong direction for this project.
- **Sharding.** MeiliSearch sharding distributes one index across multiple
  server *instances*. It is [Enterprise Edition only][sharding], unavailable in
  the open-source build, and has no single-machine mode for improving thread
  utilization. Splitting a large corpus across several ordinary indexes remains
  possible but is a much larger change (search fan-out, corpus semantics).
- **Raising the client-side wait timeout.** Cosmetic. It suppresses the warning
  without changing any cost, and would hide the real curve.

## Related

- `docs/rdr/RDR-024-adaptive-cross-file-index-batching.md` — cross-file
  batching to collapse undersized tasks. Still Draft and unimplemented; it
  attacks task *count*, which is complementary to the per-task cost addressed
  here. Its "baseline captured" prerequisite is satisfied by the before/after
  figures above.
- `docs/rdr/RDR-009-dual-indexing-strategy.md` — fail-fast contract.
- `src/arcaneum/fulltext/client.py` — task wait and retry behavior.

[ram-threads]: https://www.meilisearch.com/docs/learn/indexing/ram_multithreading_performance
[indexer-2024]: https://www.meilisearch.com/blog/introducing-indexer-2024
[sharding]: https://www.meilisearch.com/blog/horizontal-scaling-with-sharding

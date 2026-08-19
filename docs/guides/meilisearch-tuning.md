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

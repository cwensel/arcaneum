# Remote Embedding Workers

Document embedding is the largest CPU and memory load `arc` puts on the machine
it runs on. Remote embedding workers move that load to spare machines on the
local network or a Tailscale tailnet. The workstation keeps running `arc` as
before; only the embedding of document chunks for selected models runs on the
workers. The workers run arcaneum's own embedding code, so their vectors match
the workstation's and the two can share a collection.

This topology is optional. Without configured endpoints nothing changes.

The goal is lower load on the workstation, not faster indexing: older worker
machines are usually slower than the workstation, and sync embeds one file at a
time (see [Limitations](#limitations)).

Design record: [RDR-025](../rdr/RDR-025-remote-embedding-worker-pool.md).

## Topology

```text
 Workstation                                  Worker hosts (dev-0, dev-1, ...)
 ───────────                                  ────────────────────────────────
 arc corpus sync / repair / parity            Docker container arcaneum-embed-<port>
 arc index pdf / code / markdown                runs `arc models serve`
   │                                            one resident model at a time
   │ document chunks ──────── TCP 7300 ───────► named volume arcaneum-embed-models
   │ ◄─────────────────────── vectors            (model cache)
   │
 arc search ── query embedding stays local
 arc models deploy ── ssh/scp + remote docker CLI ──► builds image, runs container
 arc models workers ── status-only connections ─────► state, versions, model
```

| Runs on the workstation | Runs on the workers |
| --- | --- |
| File discovery, extraction, OCR, chunking | Document embedding for routed models |
| Query embedding (`arc search`) | Model download into the cache volume |
| Qdrant and MeiliSearch writes | |
| Embedding for models not routed remotely | |
| `arc models deploy` and `arc models workers` | |

## Requirements

**Workers**

- Linux with Docker (validated on x86_64 Ubuntu with Docker 29).
- Memory for one resident model: arctic-m peaks near 7 GB and qwen3-embed near
  6 GB at the default batch sizes, so about 12 GB free is comfortable.
- Key-based SSH from the workstation as a user in the `docker` group, so the
  remote `docker` CLI runs without sudo or prompts.
- Network access while an image builds (Docker Hub, PyPI, the PyTorch CPU
  index) and for the first download of each model (Hugging Face). A built
  worker restarts without network access.

**Workstation**

- The system `ssh` and `scp` (deploy runs them in batch mode, so
  `~/.ssh/config` applies).
- `uv` when deploying a source tree with `arc models deploy --build`.
- Optional: Tailscale with MagicDNS, for `bind: tailscale`.

## Quick Setup

1. Add the workers to `~/.config/arcaneum/config.yaml`:

    ```yaml
    remote_embedding:
      unavailable: local
      endpoints:
        - host: dev-0
          bind: tailscale
        - host: dev-1
          bind: tailscale
    ```

2. Preview what deploy will do; this changes nothing on the workers:

    ```bash
    arc models deploy --dry-run
    ```

3. Deploy. Use `--build` when this workstation runs a source checkout or any
   build that is not a release:

    ```bash
    arc models deploy            # release installs
    arc models deploy --build    # source checkouts and development builds
    ```

    The first build on each worker downloads Python and CPU torch and takes
    several minutes; verification then downloads each served model.

4. Confirm both workers are up:

    ```bash
    arc models workers
    ```

5. Sync as usual. A run that routes models prints one line at the start:

    ```text
       Remote embedding: arctic-m on dev-0:7300, dev-1:7300
    ```

## Configuration

All settings live under `remote_embedding` in
`~/.config/arcaneum/config.yaml`. Only `endpoints` is required.

```yaml
remote_embedding:
  unavailable: wait            # wait | local | fail
  authkey: shared-secret       # optional; ARC_EMBED_AUTHKEY overrides
  routing:                     # per-model routing; defaults shown
    arctic-m: remote
    qwen3-embed: on-request
  idle_grace: 60               # seconds; see Model Turns
  turn_length: 600             # seconds; see Model Turns
  probe_interval: 10           # seconds between reconnect attempts to a down worker
  deadlines:
    loading: 1800              # first download and load of a model
    encode_per_chunk:          # encode allowance per chunk in a sub-batch
      arctic-m: 10
      qwen3-embed: 160
    encode_per_chunk_default: 160
    heartbeat_interval: 5
    heartbeat_timeout: 30
    connect_timeout: 10
  endpoints:
    - host: dev-0              # name the workstation resolves; also the SSH target
      port: 7300
      bind: tailscale          # all (default) | tailscale | an IP address
      ssh: admin@dev-0         # deploy's SSH target, if not the host
      models: [arctic-m]       # default: every remote and on-request model
      batch_size:              # encode batch size per model on this worker
        arctic-m: 64
```

### Endpoint Settings

| Key | Default | Meaning |
| --- | --- | --- |
| `host` | required | Name or address the workstation connects to |
| `port` | `7300` | TCP port the worker listens on |
| `bind` | `all` | Address the worker listens on (see [Listening Address](#listening-address)) |
| `ssh` | `host` | SSH target for `arc models deploy` |
| `models` | routed models | Models this worker serves; confines a model to chosen workers |
| `batch_size` | 100 (fastembed), 8 (SentenceTransformers) | Largest sub-batch the worker encodes at once; bounds its peak memory |

### Unavailable Policy

`unavailable` decides what happens when no worker can serve a routed model
(all down, unreachable, or running different code):

| Policy | Behavior |
| --- | --- |
| `wait` (default) | Block and keep probing; print the unreachable workers once a minute |
| `local` | Embed on the workstation, as without workers; warn once per model |
| `fail` | End the run; the file being embedded gets no chunks, finished files keep theirs |

`local` suits a workstation whose `arc` runs a source checkout: every commit
changes the code fingerprint, and syncs keep working on the workstation until
the workers are redeployed.

### Environment Overrides

| Variable | Effect |
| --- | --- |
| `ARC_EMBED_ENDPOINTS=host[:port],...` | Replaces the configured endpoints for one process; empty disables the workers |
| `ARC_EMBED_AUTHKEY` | Shared key for this process (workstation) or server (worker) |

## Routing

Each model has a routing mode:

| Mode | Document embedding |
| --- | --- |
| `remote` | Goes to the workers whenever endpoints are configured |
| `on-request` | Goes to the workers only with `--remote-embed` |
| `local` | Always on the workstation (every unlisted model) |

Defaults: `arctic-m` is `remote`, `qwen3-embed` is `on-request`, everything else
is `local`. A fastembed model can be routed only if the model registry pins its
revision, so a worker cannot load different weights; `arctic-m` is pinned.

| Command | Routing |
| --- | --- |
| `arc corpus sync`, `arc corpus repair` | Configuration; `--remote-embed` adds `on-request` models, `--local-embed` routes nothing |
| `arc corpus parity`, `arc index pdf/code/markdown` | Configuration (`remote` models only) |
| `arc search` | Never routed: query embedding is always local |

`ARC_EMBED_ENDPOINTS=` (empty) turns routing off for any command.

```bash
arc corpus sync DevRef ~/papers --remote-embed      # qwen3-embed on the workers
arc corpus repair PapersFast --local-embed          # this run entirely local
ARC_EMBED_ENDPOINTS= arc corpus parity PapersFast   # parity without workers
```

## How Work Is Distributed

Sync embeds each file's chunks with one call per model. For each call:

1. The workstation sends a heartbeat to every connected worker at once and
   ranks the eligible ones: workers already holding the model (loaded or
   loading), then empty workers, then idle workers holding another model.
   When every worker is busy with another model, the work queues at one worker
   only, so two workers do not both switch models.
2. The chunks are split into sub-batches no larger than the worker's encode
   batch size (and spread evenly across the chosen workers).
3. The chosen workers pull sub-batches from a shared queue, so a faster worker
   takes more. A file with fewer sub-batches than workers uses only the
   most-preferred ones; a one-chunk file goes to one worker.
4. Results are reassembled in the original chunk order.

When a worker disconnects, stops answering heartbeats or exceeds a deadline, it
is marked down, its sub-batch returns to the queue, and it is reconnected in
the background every `probe_interval` until it answers again. A sub-batch that
fails on its own (an encode error or invalid vectors) is retried once and then
fails only that file; the worker stays up.

There is no distribution setting beyond `models`, `batch_size` and the turn
settings below.

## Model Turns

Each worker keeps one model resident; two would not fit in a 16 GB worker's
memory. A request for another model waits (`queued`) until the resident model
yields, which it does after its current encode once another model is waiting
and either:

- it has had no work for `idle_grace` (default 60 s), or
- it has been resident for `turn_length` (default 600 s).

A waiting request therefore waits at most the rest of a turn plus one encode
and one model load. In practice, a `--remote-embed` qwen3-embed run on both
workers gives one worker to an arctic-m sync that starts meanwhile and keeps the
other; after the arctic-m sync ends, the idle worker returns to qwen3-embed
after the grace period.

## Version Matching

A worker serves only a workstation running exactly the same code. Each
connection starts with a handshake comparing the protocol version, arcaneum
version, model revision, prompt policy, dimensions and a **code fingerprint**:
a hash of the `arcaneum` package's Python sources and declared requirements,
the same whether computed from an installed package or a wheel. Development
builds that report a release's version still get their own fingerprint.

A worker whose fingerprint differs is reported as `mismatch` and is not used;
the unavailable policy applies until `arc models deploy` updates it.

- **Release installs (pipx):** after upgrading arcaneum on the workstation, run
  `arc models deploy`.
- **Source checkouts (editable installs):** every commit or pull changes the
  fingerprint; run `arc models deploy --build` afterwards.

## Deploying and Updating Workers

`arc models deploy [HOSTS...]` brings each configured worker (or the named
ones) to the workstation's code, one at a time:

1. **Wheel**: the GitHub release wheel for the workstation's version (default),
   one built from the source tree (`--build`), or a given wheel (`--wheel
   PATH`). A wheel whose fingerprint differs from the workstation's is refused.
2. **Image**: built on the worker from a generated Dockerfile
   (`python:3.12-slim`, CPU torch in its own cached layer, then the wheel with
   its `sentence-transformers` extra), tagged `arcaneum-embed:<fingerprint>`.
   Images are never pushed to a registry.
3. **Container**: `arcaneum-embed-<port>`, `--restart unless-stopped`, host
   networking (no published ports), the `arcaneum-embed-models` volume as the
   model cache.
4. **Health**: deploy waits until the worker reports the new fingerprint,
   printing the container's log if it does not.
5. **Verification**: for each served model, deploy embeds a fixed probe set on
   the worker and on the workstation and requires cosine ≥ 0.9999 for every
   text. On failure, or if the worker crashes or disconnects during the check,
   deploy removes the container (keeping the image) and reports the model, the
   cosine and both sides' dependency versions.
6. **Cleanup**: deploy keeps the images for the current and previous
   fingerprint and removes older ones.

A failed build leaves the running worker untouched. A failed host is reported
and skipped, the others continue, and deploy exits non-zero.

| Situation | Command | Action |
| --- | --- | --- |
| First install | `arc models deploy` | Build image, create container |
| Workstation upgraded | `arc models deploy` | Build image, replace container |
| Workstation downgraded | `arc models deploy` | Reuse retained image, replace container |
| Development build | `arc models deploy --build` | Build image, replace container |
| Endpoint settings changed | `arc models deploy` | Replace container, reuse image |
| Base image or dependency updates | `arc models deploy --refresh` | Rebuild with `--pull --no-cache`, replace container |
| Container stopped or missing | `arc models deploy` | Start or create container |
| Container not answering | `arc models deploy` | Replace container |
| Nothing changed | `arc models deploy` | None |

`--dry-run` shows each host's current and desired version, fingerprint and
action without changing anything. `--no-verify` skips the parity check.

### Removing a Worker

There is no undeploy command yet. On the worker host:

```bash
docker rm -f -v arcaneum-embed-7300
docker image rm $(docker image ls -q --filter label=arcaneum.role=embed-worker)
docker volume rm arcaneum-embed-models    # cached models; re-downloadable
```

Then remove the endpoint from `config.yaml`.

## Listening Address

The worker binds host addresses itself (the container uses host networking):

| `bind` | Listens on |
| --- | --- |
| `all` | Every interface |
| `tailscale` | The host's Tailscale address (`100.64.0.0/10`, on `tailscale0` if present) |
| an IP address | That address |

For `tailscale` and an IP address, the worker waits for the address to exist,
polling every 2 s and logging once a minute, then binds. After a reboot the
container starts at once and binds as soon as Tailscale has configured the
address, with no changes to the host's service order.

On the workstation, a `bind: tailscale` worker whose name resolves outside the
tailnet is reported with the resolved address and a pointer to Tailscale or
MagicDNS, rather than only as unreachable.

## Monitoring

- `arc models workers` (or `--json`) shows each worker's state (`up`,
  `mismatch`, `unreachable`), version, fingerprint, resident model, whether it
  is busy, its served models and its torch, onnxruntime, fastembed,
  sentence-transformers and transformers versions. Status connections never
  load or switch a model.
- Runs that route models print `Remote embedding: <models> on <workers>` at the
  start.
- With `--mem-probe-interval`, each memory-probe JSONL sample has a
  `local_models` field: an empty list means the workstation loaded no model.
- Worker logs: `ssh <host> docker logs --tail 50 arcaneum-embed-7300`. They
  show connections, model loads and unloads, and the bind address.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `mismatch` in `arc models workers`; syncs wait, fall back or fail | Workstation code changed (upgrade, commit, pull) | `arc models deploy` (release) or `arc models deploy --build` (source) |
| `unreachable (... Connection refused)` | Container stopped or missing, or wrong port | `arc models deploy`; check `docker ps` on the host |
| `... resolves to <address> ... outside the tailnet` | Workstation not on the tailnet, or MagicDNS off | Start Tailscale and enable MagicDNS on the workstation |
| `authentication failed ... same authkey` or `undecodable message ... authkey` | `authkey` set on only one side or different | Set the same `remote_embedding.authkey` and redeploy |
| Deploy: `vector parity check failed` | Worker and workstation dependency versions embed differently (both sides' versions are printed) | Align the workstation's packages with the worker's, or serve only the passing models (`models:`), then redeploy |
| Deploy: `parity check could not complete` | The worker crashed or disconnected during the check (its log is printed) | Read the log; rerun deploy |
| Sync prints `Waiting for remote embedding endpoints ...` | `unavailable: wait` and no worker can serve the model | Fix the workers, or rerun with `--local-embed` / `ARC_EMBED_ENDPOINTS=` |
| Workers idle while a sync runs | The sync is scanning or extracting, or the corpus model is `on-request` | Normal between files; add `--remote-embed` for `on-request` models |

### Dependency Drift

Deploy checks vectors against the workstation's own environment, so a
workstation with older or newer inference packages than the workers can fail
verification even though the code matches. For example, a workstation with
transformers 4.53 and tokenizers 0.21 embedded code-like text with qwen3-embed
at cosine 0.993 to workers running transformers 4.57 and tokenizers 0.22.
Reinstall the workstation's arcaneum with its declared requirements (for an
editable install, `pip install --user -e '<checkout>[sentence-transformers]'`)
and redeploy. Upgrading also changes new local vectors for affected text
compared with vectors already stored.

### Worker Crashes

A worker process that dies is restarted by Docker; the pool retries its batch
on another worker and reconnects when it answers again. An intermittent
`invalid opcode` in `libtorch_cpu.so` (exit code 132) has been seen once on a
pre-AVX2 worker during deploy verification and did not reproduce; check
`journalctl -k` on the host if it recurs.

## Security

The protocol uses Python pickle and is meant for a trusted LAN or tailnet only.
Anyone who can connect to a worker without an authkey can make it run code.

- Prefer `bind: tailscale`, which accepts connections only over the tailnet,
  where Tailscale access controls decide who can connect. The worker logs a
  warning when it listens on all interfaces without an authkey.
- An `authkey` adds a shared-secret challenge. It is passed to the container as
  an environment variable, visible to anyone with Docker access on the worker.
- Never expose a `bind: all` port beyond the local network.

## Limitations

- **One file at a time**: sync embeds each file before preparing the next, so
  workers idle between files and a one-chunk file uses one worker. With two
  older workers, arctic-m syncs take about twice as long as on a modern
  workstation; qwen3-embed is about 14 times slower per worker on pre-AVX2
  CPUs and suits long unattended runs. Cross-file batching is deferred.
- **Queries stay local**, including for qwen3-embed, whose local model load
  dominates search time.
- **One resident model per worker**; serving both models makes workers switch
  models under contention.
- **No undeploy command**; see [Removing a Worker](#removing-a-worker).
- Only models with a pinned revision (all SentenceTransformers models, and
  fastembed models such as `arctic-m`) can be routed.

## See Also

- [CLI Reference: Remote Embedding Workers](cli-reference.md#remote-embedding-workers)
- [Embedding acceleration](accelerators.md)
- [RDR-025: Remote Embedding Worker Pool](../rdr/RDR-025-remote-embedding-worker-pool.md)

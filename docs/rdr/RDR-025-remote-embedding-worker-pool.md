# Recommendation 025: Remote Embedding Worker Pool

> Revise during planning; lock at implementation.
> If wrong, abandon code and iterate RDR.

## Metadata

- **Date**: 2026-10-14
- **Status**: Draft
- **Type**: Architecture
- **Priority**: Medium
- **Related Issues**: Extends RDR-022 (spawned accelerator worker
  protocol); intermediate step toward kata `arcaneum#tpcs` under umbrella
  `arcaneum#rrky` (`docs/cloud-bulk-embedding-feasibility.md`); deferred
  work relates to RDR-024 (cross-file batching) and `arcaneum#vby2`
  (chunk-hash embedding cache)

## Problem Statement

The workstation running `arc` is shared with other concurrent projects.
Document embedding during `arc corpus sync` and `arc corpus repair` is the
largest CPU and memory consumer `arc` itself places on it. `nice -n 20`
lowers its priority but does not remove the work, the threads, or the
resident model.

The goal is to **lower local CPU and memory use**, not to make embedding
faster. Document embedding should run on spare machines on the local
network, in Docker containers, over a simple RPC, while producing vectors
identical (within float tolerance) to those the workstation produces, so
remote and local vectors can be mixed in one collection and queried with
locally embedded queries.

Phase 1 covers `arctic-m` document embedding by default and `qwen3-embed`
document embedding on request, for long unattended runs such as a repair or
forced resync. Query embedding stays local in Phase 1.

## Context

### Background

Evidence from the workstation's own logs and a local profile
(2026-10-13/14):

- **Sync is mostly encoding.** Memory-probe samples (~1 Hz) for two
  `arc corpus sync` runs on 2026-10-14 attribute 76% and 84% of samples to
  the `encoding` phase, 12–15% to `file` (extraction and chunking), and
  4–9% to `indexing`. Both runs encoded in-process on CPU (no MPS memory,
  no accelerator worker). Samples are wall-clock, not CPU time; encoding
  uses all cores minus two while extraction is mostly single-threaded, so
  encoding's CPU share is likely higher still.
- **Search cost depends on the model.** Over the same two days, semantic
  searches against `qwen3-embed` corpora had median 9.2 s (671 calls) and
  against `arctic-m` corpora median 2.7 s (156 calls). A local profile
  attributes 10.0 s of a 15.2 s `qwen3-embed` search to model load and first
  query, but only 0.3 s of a 4.0 s `arctic-m` search. Remote query
  embedding would therefore help `qwen3-embed` search substantially and
  `arctic-m` search hardly at all; it is deferred (see Deferred Work).
- **Memory is tight.** During sync the workstation sat near 75% RAM use with
  about 18 GB of swap in use; `arc` reached 2.7 GB RSS.

Models in use (Qdrant collection vectors, 2026-10-14):

- `arctic-m` (fastembed ONNX, 768D): `Claude` (657k points), `PapersFast`
  (390k), `StateMachineRes` (19k).
- `qwen3-embed` (sentence-transformers, 1024D), ~263k points across 12
  collections: `DevRef` (101k), `RetrofitRDR` (69k), `NextLit` (29k),
  `SchemaEvo` (18k), `SchemaEvoRes` (14k), `MergeLit`, `LLM`, `SqlSafety`
  (~7k each), `SpecDrivenDev`, `CodeMaintenance`, `SchemaEvoMarket`
  (~3–4k each), `StateMachineLit` (1k).

### Technical Environment

- Workstation: Apple M2 Pro (8 performance + 4 efficiency cores), 32 GB,
  macOS, Python 3.12, fastembed 0.8.0.
- Candidate workers: `dev-0`, `dev-1` — Intel i7-3720QM (Ivy Bridge,
  4 cores / 8 threads, AVX and F16C but **no AVX2 or FMA**), 16 GB RAM
  (~12 GB available), Ubuntu with Linux 6.8, Docker 29.8.1, no GPU, local
  network.
- Embedding entry points: `EmbeddingClient.embed()`
  (`src/arcaneum/embeddings/client.py:1411`), called once per file per
  model by sync (`src/arcaneum/cli/sync.py:4081`, `4496`, `5848`) and by
  the indexing pipelines through `embed_parallel()`
  (`client.py:1766`).
- RDR-022 protocol: `src/arcaneum/embeddings/worker_protocol.py` (versioned
  envelopes, request IDs, `WorkerBackend.encode()`), currently over
  `multiprocessing` spawn queues.
- Existing command options: `arc corpus sync` and `arc corpus repair` both
  accept `--no-gpu` and `--max-embedding-batch`; neither has a remote
  option.

## Research Findings

### Investigation

**Where the remote boundary must sit.** The RDR-022 worker is only used for
the sentence-transformers accelerator path (`client.py:980-1000`,
`1729`). CPU encodes short-circuit to in-process
`_encode_on_cpu_fallback` (`client.py:1716-1717`), and fastembed models are
always in-process. A remote backend placed behind the accelerator worker
would therefore miss every encode on this workstation. The boundary has to
be "embed these document texts with model M", at
`EmbeddingClient.embed()` / `embed_parallel()`. Two details of those entry
points matter: query embedding for sentence-transformers models also goes
through `embed()` with `prompt_type="query"`
(`src/arcaneum/search/embedder.py:130`), and `embed_parallel()` constructs
the local model with `get_model()` before batching (`client.py:1835`).

**Existing worker loop.** `_worker_main` (`worker_protocol.py:138`) loads
its single backend synchronously when it receives `initialize`, then
serves one command at a time. It cannot answer heartbeats while a model
loads and cannot switch models; `encoded` carries the result as a nested
list plus `shape` and `dtype` (`worker_protocol.py:176`).

**Observability.** The memory-probe JSONL records phase, RSS, MPS, swap and
accelerator-worker fields (`memory_probe.py:306-335`); nothing in it shows
which models the process has loaded, and fastembed always runs in-process,
so `worker_rss` does not reveal it either.

**Distribution.** arcaneum is installed from the GitHub release wheel
(`README.md:9`), not PyPI, and unreleased working trees report the same
version string as the last release. Wheels are built by the setuptools
backend (`pyproject.toml`); the release workflow runs `python -m build`,
and development checkouts carry a `uv.lock`, so `uv build --wheel` builds
the same wheel from a source tree.

**Worker access (2026-10-14).** `ssh -o BatchMode=yes` reaches `dev-0`
and `dev-1` as a user in the `docker` group, so the remote `docker` CLI
runs without sudo or prompts. Both are x86_64 with Docker 29.8.1 and
110 GB / 163 GB free, with no containers or named volumes present. `dev-1`
has no ufw; Docker's published ports bypass host firewall rules in any
case (Docker documentation, "Docker and ufw").

**Tailscale (2026-10-14).** The workstation, `dev-0` and `dev-1` share a
tailnet with MagicDNS enabled (search domain `tail27d7.ts.net`, resolver
`100.100.100.100`). On the workstation, `socket.getaddrinfo("dev-0")`
already returns only `dev-0`'s Tailscale address (`100.84.12.108`; `dev-1`
likewise), so SSH and any client using the bare name go over the tailnet.
The workers run tailscaled 1.102.4 in kernel mode with a `tailscale0`
interface. A node's Tailscale address is stable; the problem at boot is
that it does not exist yet when services start: `docker.service` is not
ordered after `tailscaled`, and `net.ipv4.ip_nonlocal_bind` is 0, so
binding that address fails until tailscaled has configured it. A
Docker-published port on that address is bound when the container starts,
so the container fails to start after a reboot and is not retried
(**Assumed**, from commonly reported behavior; not reproduced). Fixing
that on the host (systemd ordering, the sysctl) or configuring
`tailscale serve` (no serve config or operator user is set) requires sudo
on each worker, which is password-protected on `dev-0`. `psutil`, already
an arcaneum dependency, lists interface addresses.

**Model revisions.** sentence-transformers registry entries pin a commit
hash and refuse to load without one (`client.py:406-408`). fastembed
registry entries such as `arctic-m` (`client.py:309-316`) carry no
revision. fastembed 0.8.0 has no revision parameter: a non-offline download
resolves the repository's current HEAD (`model_info(repo).sha`,
`fastembed/common/model_management.py:235`). It does accept
`specific_model_path`, which loads a model directory directly and skips
download (`model_management.py:402-404`; accepted by `TextEmbedding` via
`fastembed/text/onnx_embedding.py:209,251,256`). On 2026-10-14 the
workstation cache, the dev-0 cache, and Hugging Face HEAD all resolve
`Snowflake/snowflake-arctic-embed-m` to `fc74610d18462d218e312aa986ec5c8a75a98152`
(last modified 2024-12-13).

**Transport.** `multiprocessing.connection` frames each message with a
4-byte length, switching to an 8-byte length above 2 GiB
(`multiprocessing/connection.py:406-416`), so payload size is not a
practical limit; a 64-chunk `arctic-m` batch is about 196 KB of float32.
`Listener`/`Client` take an optional `authkey` and use an HMAC challenge
(`connection.py:928-949`).

**Feasibility spike (2026-10-14).** Real chunks sampled from Qdrant payloads
(96 `NextLit` chunks for `qwen3-embed`, p50 2,385 chars; 192 `PapersFast`
chunks for `arctic-m`, p50 1,443 chars) were embedded with arcaneum's own
`EmbeddingClient(use_gpu=False)` at commit `b7a8a2e`. Remote: `dev-0` in a
`python:3.12-slim` container that installed arcaneum and its
`sentence-transformers` extra at startup into a cached venv volume,
CPU-only torch, models downloaded on first use into a mounted cache.
Local: the workstation, same script, under `nice -n 20` while the machine
carried a load average of ~18 on 12 cores (its normal contended state, not
an idle baseline).

| Measure | `arctic-m` workstation | `arctic-m` dev-0 | `qwen3-embed` workstation | `qwen3-embed` dev-0 |
| --- | --- | --- | --- | --- |
| Threads (`OMP_NUM_THREADS`) | 10 | 6 | 10 | 6 |
| First load (s) | 0.6 | 7.3 | 8.4 | 30.0 |
| Encode (s) | 50.0 | 189.9 | 106.2 | 1,524.7 |
| Chunks/s | 3.84 | 1.01 | 0.90 | 0.063 |
| Peak RSS (MB) | 5,774 | 7,070 | 4,307 | 6,004 |
| Resident query latency (s) | — | 0.02–0.05 | — | 0.46–0.62 |

- **Parity**: dev-0 vectors match workstation vectors for both models:
  per-chunk cosine 1.000000 (min), max absolute difference 9.7e-8
  (`arctic-m`) and 5.7e-7 (`qwen3-embed`), identical nearest neighbor for
  every chunk within each sample.
- **Throughput**: per box, dev-0 is ~3.8× slower than the contended
  workstation for `arctic-m` (ONNX) and ~14× slower for `qwen3-embed`
  (torch). torch reports CPU capability `DEFAULT` on Ivy Bridge, i.e. no
  AVX2 kernels; onnxruntime degrades far less. Across two boxes,
  `arctic-m` runs at roughly half the workstation's rate, while
  `qwen3-embed` runs at ~450 chunks/hour, which suits unattended repairs
  and forced resyncs but not routine syncs.
- **Memory**: peak RSS is 6.0–7.1 GB per model in separate processes,
  driven largely by batch activations (`arctic-m` peaks higher than the
  larger `qwen3-embed`). Either model fits on a 16 GB box with ~12 GB
  available; both resident in one server would not fit comfortably.
- **Install at startup**: a stock `python:3.12-slim` container with a
  cached venv volume ran arcaneum's CPU path unchanged (torch 2.14.1+cpu,
  numpy 2.5.4, onnxruntime 1.31.0, sentence-transformers 5.7.0). Install
  duration was not isolated.

#### Dependency Source Verification

| Dependency | Source Searched? | Key Findings |
| --- | --- | --- |
| `multiprocessing.connection` (stdlib 3.12) | Yes | 4-byte length frame, 8-byte above 2 GiB (`connection.py:406-416`); optional `authkey` HMAC challenge (`connection.py:928-949`). |
| fastembed 0.8.0 | Yes | No revision parameter; download resolves repo HEAD (`model_management.py:235`); `specific_model_path` loads a local directory and skips download (`model_management.py:402-404`, `onnx_embedding.py:209-256`). |
| huggingface_hub `snapshot_download` | Yes (as called by fastembed) | Accepts `revision` via kwargs passthrough in fastembed's own call (`model_management.py:226-231`, `258-262`). |
| sentence-transformers 5.7.0 / torch 2.14.1+cpu | No (spike only) | Ran on Ivy Bridge; torch selects `DEFAULT` CPU capability. Registry already pins revisions. |

### Key Discoveries

- **Verified**: arcaneum's CPU embedding path runs unchanged on `dev-0` in
  a stock Python container with install-at-startup and on-demand model
  download, and its vectors match the workstation's.
- **Verified**: the candidate machines have enough memory for one resident
  model each, not two; compute, not memory, limits them, severely so for
  `qwen3-embed`.
- **Documented**: the remote boundary must wrap `EmbeddingClient`'s
  document path, not the RDR-022 accelerator worker (`client.py:1716`).
- **Documented**: routing must key on `prompt_type="document"` and run
  before `embed_parallel()`'s `get_model()` call, or queries go remote and
  remote-routed work still loads a local model.
- **Documented**: the RDR-022 serve loop cannot report `loading` or switch
  models; only its envelopes carry over to a socket server.
- **Documented**: fastembed cannot pin a revision itself, but arcaneum can
  download a pinned snapshot and pass it as `specific_model_path`.
- **Verified**: both workers can be managed from the workstation with
  non-interactive SSH and the remote `docker` CLI.
- **Verified**: bare worker names already resolve to Tailscale addresses
  on the workstation, so a server bound only to its Tailscale address is
  reachable as `dev-0` with no client-side configuration.
- **Documented**: binding a Tailscale address at boot needs the binding
  process to wait for the address; Docker port publishing cannot wait, and
  host-level fixes need sudo on every worker.
- **Verified**: `qwen3-embed` search latency is dominated by local model
  load; `arctic-m` search latency is not.

### Critical Assumptions

- [x] Remote vectors agree with local CPU vectors for every model in use
  (cosine ≥ 0.9999 per chunk) — **Status**: Verified for `arctic-m` and
  `qwen3-embed`, the only models in use (min cosine 1.000000)
  — **Method**: Spike
- [x] fastembed models can be pinned so a worker loads the same weights as
  the workstation — **Status**: Verified (pinned snapshot download plus
  `specific_model_path`; both machines currently resolve the same snapshot)
  — **Method**: Source Search
- [x] The transport carries batch payloads without size limits relevant to
  this use — **Status**: Verified — **Method**: Source Search

## Proposed Solution

### Approach

Run arcaneum itself as an embedding server inside Docker on each worker
machine, and route the workstation's **document** embedding for selected
models to a pool of those servers over the local network.

1. **Server (`arc models serve`)**: a long-lived process wrapping a CPU
   `EmbeddingClient`. It accepts RDR-022 envelopes over
   `multiprocessing.connection.Listener` on a port of the host's LAN or
   Tailscale address, loads the model a
   connection names in its `initialize` message (downloading the pinned
   revision into a mounted cache volume), and keeps it resident.
2. **Worker image**: built on each worker, never published. It is the
   stock `python:3.12-slim` base plus a standard `pip install` of an
   arcaneum wheel with its `sentence-transformers` extra (CPU-only
   torch), the same install the workstation has; the container's command
   is `arc models serve`. Model weights live in a named cache volume, not
   in the image.
3. **Client pool**: a configured list of `host:port` endpoints. One request
   in flight per endpoint per client (RDR-022's rule, applied per
   endpoint). An `embed()` call is split into sub-batches, spread evenly
   across the idle endpoints that serve the model and no larger than the
   server's encode batch size; results are reassembled in order by request
   ID.
4. **Routing**: per-model configuration decides whether a model's document
   embedding is remote. Only `prompt_type="document"` calls are routed;
   query embedding stays local regardless of configuration. Phase 1
   defaults: `arctic-m` remote whenever endpoints are configured;
   `qwen3-embed` local unless the command is run with `--remote-embed`.
5. **Deployment (`arc models deploy`)**: one idempotent command, run on the
   workstation, that installs, upgrades, updates, refreshes and
   reconfigures the worker on each configured endpoint over SSH, building
   the image there and verifying vector parity before the pool uses it.

Security is an explicit non-goal: the pool runs on a trusted local network
or tailnet, pickle over the socket is acceptable because both ends run the
same pinned arcaneum, and an optional static `authkey` (plus Tailscale
access controls when `bind: tailscale`) is the only access control.

### Technical Design

**Handshake.** The first message on a connection is `initialize`, carrying
protocol version, arcaneum version, code fingerprint, model key, pinned
model revision, prompt-policy version, and expected dimensions. The code
fingerprint is a hash of the installed `arcaneum` package's Python
sources, computed the same way on both sides, so a development build that
still reports the last release's version is distinguished from that
release. The server loads the model (downloading the pinned revision if
needed) and replies `initialized` with the same fields as resolved on its
side. Any mismatch excludes that endpoint for that model. An `initialize`
without a model key opens a status-only connection that loads nothing;
`arc models workers` and `arc models deploy` use it. Heartbeat replies on
any connection report the server's arcaneum version, code fingerprint,
resident model and state, and the installed versions of torch,
onnxruntime, fastembed, sentence-transformers and transformers.

**Server concurrency.** The server reads each connection on its own
thread, so heartbeats are answered at any time, and runs model loads and
encodes on a single executor, so encodes from several clients (for
example a git-hook sync during a repair) are serialized in arrival order.
Heartbeat replies report the connection's state: `loading` while its
model downloads or loads, `queued` while its request waits behind another
client's, `encoding` while it runs. The client's encode deadline counts
only time in `encoding`, so first use and queueing never trip a timeout.

**fastembed revision pinning.** fastembed registry entries gain a
`revision` field (`arctic-m`: `fc74610d18462d218e312aa986ec5c8a75a98152`).
When a fastembed model has a pinned revision, both the workstation and the
server obtain that snapshot themselves and construct `TextEmbedding` with
`specific_model_path` pointing at it. fastembed models without a pinned
revision are not eligible for remote routing.

**Encode.** `encode` carries document texts and model key; `encoded` keeps
the existing envelope (nested list plus `shape` and `dtype`), which the
client converts to an independently owned C-contiguous float32 array and
validates with the existing `_validate_embeddings` checks before use.

**Model placement.** A 16 GB worker holds one resident model comfortably,
not two. Each endpoint keeps at most one model resident. A request for
another model evicts the resident one only after the resident model has
had no queued or running encode for an eviction grace period (default
60 s), so two clients using different models do not reload on every
batch; the waiting client sees `queued`. Endpoint configuration may
restrict which models an endpoint serves so that, if desired, a long
`qwen3-embed` run can be confined to one box while the other keeps serving
`arctic-m`. The server's encode batch size is configurable to bound peak
RSS.

**Pool behavior.**

- An endpoint that errors, disconnects or exceeds the encode deadline is
  marked down and its in-flight batch is retried on another endpoint.
  Encoding has no side effects, so retry is safe.
- Down endpoints are probed with the existing heartbeat message and
  rejoin when healthy.
- When no endpoint can serve a remote-routed model, a configured policy
  applies: `wait` (default; block and keep probing, reporting the
  unreachable endpoints), `local` (embed locally, today's behavior), or
  `fail` (end the run; the file being embedded gets no chunks written,
  files already completed keep theirs).
- An endpoint excluded by a handshake mismatch counts as unable to serve
  that model. Under `wait` the client re-handshakes on each probe, so
  redeploying the worker with a matching wheel resumes the run; the wait
  message shows both sides' versions and fingerprints and names
  `arc models deploy` to update the workers and `--local-embed` to proceed
  without them.
- Routing is decided before any local model is constructed: a
  remote-routed `embed_parallel()` call skips its up-front `get_model()`.
  The client records which models it has loaded locally; the memory-probe
  JSONL gains a `local_models` field listing them.

```text
// Illustrative — verify API signatures during implementation
RemoteEmbeddingPool(endpoints: list[Endpoint], unavailable: "wait"|"local"|"fail")
  .embed(texts: list[str], model: str) -> np.ndarray   # (n, dim) float32
  .health() -> list[EndpointHealth]  # endpoint, state, resident model, versions
```

**Configuration.** Endpoints, per-model routing and the unavailable policy
live in arcaneum config, with an environment override for endpoints
(following the `ARC_QDRANT_URL` pattern), e.g.
`ARC_EMBED_ENDPOINTS=dev-0:7300,dev-1:7300`. Configured routing applies to
every command that embeds documents through `EmbeddingClient`: `arc corpus
sync` and `arc corpus repair`, and the `arc index` commands through the
shared model cache (`embeddings/model_cache.py`). `arc corpus sync` and
`arc corpus repair` gain `--remote-embed` (route every model in the run to
the pool) and `--local-embed` (route none), overriding configuration for
that invocation; the `arc index` commands follow configuration only in
Phase 1.

Each configured endpoint describes both how to reach the worker and how
to run it: `host` (a name such as `dev-0`, resolved normally by the
client), `port`, `ssh` (SSH target for deployment, default the host, so
`~/.ssh/config` applies), `bind` (see Listening address), the models it
serves, and its encode batch size. The `authkey` is shared by all
endpoints. Endpoints given only through `ARC_EMBED_ENDPOINTS` use the
defaults.

**Listening address.** The worker container uses host networking, so the
server binds host addresses itself and Docker publishes no ports. `bind`
selects the address:

- `all` (default): all interfaces.
- `tailscale`: the host's Tailscale IPv4 address, found at startup with
  `psutil.net_if_addrs()` as the address in `100.64.0.0/10` on
  `tailscale0` (or on any interface if `tailscale0` is absent).
- A literal IP address.

For `tailscale` or a literal address, the server waits until the address
exists on the host, polling every 2 s and logging once a minute that it
is waiting, then binds. After a reboot the container starts at once and
the server binds as soon as tailscaled has configured the address,
whatever the service start order, with no host changes. The server
re-checks every 30 s and rebinds if the Tailscale address changes. On the
client, an endpoint whose `bind` is `tailscale` and whose `host` resolves
outside `100.64.0.0/10` is reported by name with the resolved address
and a pointer to Tailscale or MagicDNS on the workstation, instead of
showing only as unreachable.

**Deployment (`arc models deploy [HOST...]`).** Brings the worker on each
named endpoint (default: all configured endpoints, one at a time) to the
desired state, using the system `ssh` and `scp` in batch mode to run the
remote `docker` CLI as the SSH user. The desired state has three parts:

- **Wheel.** Workers always run the workstation's code, so the wheel must
  have the workstation's code fingerprint, which deploy computes from the
  wheel's contents. By default deploy uses the release wheel for the
  workstation's own arcaneum version, downloaded from GitHub releases into
  arcaneum's local cache. `--build` instead builds one with
  `uv build --wheel` from the source tree the running arcaneum was
  imported from (the worktree development loop), and `--wheel PATH` uses
  a given wheel. A wheel whose fingerprint differs is refused; when the
  workstation runs an unreleased or modified build, the message names
  `--build`.
- **Image.** Built on the worker from a Dockerfile generated by deploy:
  `FROM python:3.12-slim`, CPU-only torch from the PyTorch CPU index in
  its own layer (cached across wheel changes), then
  `pip install <wheel>[sentence-transformers]`, with `arc models serve` as
  the command. Tagged `arcaneum-embed:<fingerprint prefix>` and labeled
  with arcaneum version, fingerprint and wheel SHA-256. The build is
  skipped when an image with that wheel's label exists; deploy keeps the
  current and previous image and removes older ones.
- **Container.** `arcaneum-embed-<port>`, `--restart unless-stopped`,
  `--network host`, the named volume `arcaneum-embed-models` mounted as
  the model cache, and the endpoint's port, `bind`, other settings and
  `authkey` passed as environment. Labeled with a hash of the full run
  specification.

Deploy compares that desired state with the worker's current container
and image labels, and the same command covers every lifecycle case:

| Situation | Command | Action |
| --- | --- | --- |
| First install | `arc models deploy` | Build image, create container |
| Upgrade (workstation now runs a new release) | `arc models deploy` | Build image, replace container |
| Downgrade (workstation reverted to the previous release) | `arc models deploy` | Reuse retained image, replace container |
| Update (development build) | `arc models deploy --build` or `--wheel PATH` | Build image, replace container |
| Reconfigure (endpoint settings changed) | `arc models deploy` | Replace container, reuse image |
| Refresh (base image or dependency updates, suspect worker) | `arc models deploy --refresh` | `docker build --pull --no-cache`, replace container |
| Container stopped or missing | `arc models deploy` | Start or create container |
| Nothing changed | `arc models deploy` | None; reports up to date |

Per host the order is: preflight (SSH in batch mode, remote
`docker version`); copy the wheel and Dockerfile to a build directory on
the host, skipped when the SHA-256 matches; build the image; only then
replace the container (`docker rm -f -v` of the old one, `docker run -d`),
so a failed build leaves the running worker untouched; wait until a
status-only connection reports the expected fingerprint, printing the
container's log tail on timeout or exit; then verify. A host that fails
is reported and skipped, the remaining hosts proceed, and deploy exits
non-zero. `--dry-run` shows each host's current and desired version,
fingerprint and action without changing anything.

**Deploy verification.** The fingerprint covers arcaneum's code but not
the dependency versions pip resolves when the image is built, so after
start deploy checks parity directly unless `--no-verify` is given: for
each remote-eligible model the endpoint serves, it handshakes (which also
downloads and loads the pinned revision, warming the cache), embeds a
fixed probe set shipped with arcaneum (including markdown-emphasis text,
the spike's one tokenization outlier) remotely and locally, and requires
cosine ≥ 0.9999 per text. Verification runs whenever deploy creates or
replaces a container. On failure deploy removes the new container
(keeping its image) and exits non-zero, naming the model, the minimum
cosine and both sides' dependency versions. Because the container is
removed rather than stopped, a later deploy recreates and re-verifies it
instead of restarting it; until then the worker is down, so syncs follow
the unavailable policy rather than mixing drifted vectors into a
collection.

Deploy does not drain in-flight work. Replacing a container drops its
connections, which the pool handles like any endpoint restart: the
in-flight batch is retried elsewhere and the endpoint rejoins after a
successful re-handshake.

### Existing Infrastructure Audit

| Proposed Component | Existing Module | Decision |
| --- | --- | --- |
| Message envelopes, validation | `embeddings/worker_protocol.py` (`make_message`, `_validate_message`, `MessageType`) | Reuse: same envelope and version field over a socket transport |
| Worker serve loop | `worker_protocol._worker_main` | Keep for the spawn-queue transport; the server gets its own loop (per-connection reader threads, single encode executor, model switching) because `_worker_main` loads synchronously and serves one fixed backend |
| Message transport | `AcceleratorWorkerSession` queue send/receive | Extend: factor envelope send/receive behind a connection-like interface so queues and sockets share validation |
| Load observability | `embeddings/memory_probe.py` snapshot JSONL | Extend: add `local_models` from the client's loaded-model set |
| Server-side encoding | `EmbeddingClient` CPU path | Reuse: the server embeds through the same client so preprocessing and prompt policy are identical |
| Output validation | `EmbeddingClient._validate_embeddings` | Reuse on the client side for remote results |
| fastembed model loading | `EmbeddingClient` fastembed construction and cache checks (`client.py:2377-2441`) | Extend: pinned snapshot plus `specific_model_path` when a revision is set |
| Thread sizing | `client.py:1024-1040` | Reuse on the server; the server sizes threads to its own cores |
| Command options | `--no-gpu`, `--max-embedding-batch` on sync and repair | Extend: add `--remote-embed` / `--local-embed` alongside |
| Docker invocation | `cli/docker.py` (`shutil.which` check, `subprocess.run` with timeouts and captured output) | Reuse the pattern for remote `docker` commands run through `ssh` |
| Wheel build | setuptools backend; release workflow `python -m build` | Reuse: `--build` runs `uv build --wheel` on the same backend |
| Local cache directories | `paths.py` | Reuse for downloaded release wheels |

### Decision Rationale

The spike shows the existing CPU path runs unchanged on the candidate
machines with vectors equal to within 6e-7, so the only new parts are a
transport, a pool, a server entry point, fastembed revision pinning, and
a deploy command.
Running arcaneum's own code on the server keeps preprocessing, truncation
and prompt policy identical to local embedding, which is what lets remote
and local vectors share a collection. `multiprocessing.connection` is the
smallest transport that carries the existing envelopes unchanged on a
trusted network.

Starting with `arctic-m` puts the frequent, small `Claude` corpus syncs on
the workers. Two busy boxes run at about half the workstation's rate, but
embedding is dispatched per file, so a file with fewer chunks than one
sub-batch per endpoint keeps only one box busy (about a quarter of the
workstation's rate) until cross-file batching (Deferred Work) lands.
`qwen3-embed` is 14× slower per box, so it is routed remotely only when
asked, for long unattended runs where wall-clock does not matter.

Workers must run exactly the workstation's code, so the workstation is
the natural place to drive their deployment: `arc models deploy` derives
the wheel from what the workstation runs, builds the image where it is
used (no registry, no per-release publishing), and checks vector parity
before the pool can use a new build. One idempotent command covers first
install, upgrades, development updates and refreshes.

## Alternatives Considered

### Alternative 1: Ollama endpoints for `qwen3-embed`

**Description**: Run the official `ollama/ollama` container on workers and
embed `qwen3-embed` corpora through its `/api/embed`, applying the
registry's query prompt client-side.

Spike on dev-0 (2026-10-14, Ollama 0.40.4, same 96 chunks and five queries,
batches of 8, llama.cpp on 4 threads), compared with sentence-transformers
vectors:

| Tag | Chunks/s | Query (s) | Server RSS | Doc cosine min / p50 | Query cosine min | Top-10 overlap |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3-embedding:0.6b` (Q8_0) | 0.084 | 0.43–0.78 | ~2.6 GB | 0.99621 / 0.99932 | 0.99914 | 0.98 |
| `qwen3-embedding:0.6b-fp16` | 0.077 | 0.44–0.66 | ~3.2 GB | 0.99664 / 1.00000 | 1.00000 | 1.00 |

Top-10 overlap is the mean overlap of each query's top ten chunks (Ollama
query against sentence-transformers documents) with the all-
sentence-transformers ranking. With fp16, 95 of 96 chunks match at cosine
≥ 0.9999; the one outlier (0.99664) is a chunk ending in markdown emphasis,
likely a tokenization difference (**Assumed**).

**Pros**:

- fp16 vectors are compatible with existing `qwen3-embed` collections.
- About half the server memory of the torch path (3.2 GB vs 6.0 GB).
- Off-the-shelf server; the model is pulled at startup.

**Cons**:

- Only ~1.2–1.3× faster than torch on the pre-AVX2 workers.
- Prompt policy is reimplemented outside sentence-transformers, and model
  identity is an Ollama digest rather than the registry revision.
- Covers `qwen3-embed` only; `arctic-m`, the Phase 1 model, still needs an
  arcaneum endpoint.

**Reason for rejection**: Phase 1 targets `arctic-m`, which Ollama does not
cover with verified parity, and `qwen3-embed` through `arc models serve`
already matches exactly. Revisit if `qwen3-embed` worker memory becomes the
constraint (e.g. when query routing makes it resident). Ollama Cloud is not
an option today: its model list (`https://ollama.com/api/tags`,
2026-10-14) has no embedding models, and ollama/ollama#17129 requesting
them is open.

### Alternative 2: HTTP + msgpack transport

**Description**: Serve the envelopes over HTTP with msgpack bodies.

**Pros**:

- Language-neutral, inspectable, no pickle.

**Cons**:

- More code (server framework, array encoding) with no benefit on a
  trusted LAN where both ends run identical arcaneum.

**Reason for rejection**: Revisit when the same server is exposed beyond
the LAN (e.g. the `arcaneum#tpcs` cloud path).

### Briefly Rejected

- **Remote query embedding in Phase 1**: helps only `qwen3-embed` search,
  and keeping `qwen3-embed` resident pins a 6 GB model to one of two
  workers; deferred.
- **TEI or Infinity servers**: different preprocessing and prompt handling
  from arcaneum's, so vectors could drift from locally embedded queries.
- **Run all of `arc corpus sync` on the remote machine**: indexed file paths
  are corpus identity, and git hooks trigger syncs locally.
- **Thread caps only (`OMP_NUM_THREADS`)**: lowers peak contention but keeps
  all the work and the resident model local; worth doing separately.
- **Published arcaneum image**: needs a registry and an image build per
  release, and cannot follow development builds; `arc models deploy`
  builds the same image on each worker instead.
- **Install into a venv volume at container start** (the spike's setup):
  works, but pip runs whenever the wheel changes at start, the install
  lives in a mutable volume, and a restart may need PyPI; an image built
  on the worker makes container start do nothing but serve.
- **Docker over SSH (`docker -H ssh://`, Docker contexts)**: still needs a
  way to get the wheel to the host; plain `ssh` and `scp` cover both with
  one tool the workstation already has.
- **Workers updating themselves from GitHub releases**: cannot follow
  development builds, and the workstation, not the latest release, is the
  version the workers must match.
- **Ansible or Compose for deployment**: a new dependency for two hosts
  and one container each.
- **Docker-published port on the Tailscale address**
  (`-p 100.x.y.z:7300:7300`): fails at boot when Docker starts before
  tailscaled has configured the address; fixing that needs systemd
  ordering or `net.ipv4.ip_nonlocal_bind=1` set with sudo on every worker.
- **`tailscale serve --tcp` forwarding to a loopback listener**: the
  Tailscale-native way to expose a port only on the tailnet, and immune to
  boot order, but needs sudo or an operator user on every worker, keeps
  its configuration outside what deploy can see and reconcile, and makes
  every client appear as `127.0.0.1` in server logs.
- **Tailscale sidecar container** (`tailscale/tailscale`, sharing its
  network namespace): gives each worker its own tailnet node, which needs
  an auth key and state volume and changes the name clients use from
  `dev-0` to a new node name.

## Trade-offs

### Consequences

- Positive: `arctic-m` sync embedding CPU and model memory leave the
  workstation by default once endpoints are configured.
- Positive: repairs and forced resyncs of `qwen3-embed` corpora can run
  unattended without loading the workstation.
- Positive: the server and pool are the shape the cloud path
  (`arcaneum#tpcs`, `clusterless#2cy0`) needs for fan-out.
- Positive: one idempotent command keeps workers in step with the
  workstation through upgrades, development updates and refreshes; worker
  containers restart after a reboot without network access.
- Negative: deployment needs key-based SSH to each worker as a user in the
  `docker` group (true for `dev-0` and `dev-1`), and an image build takes
  minutes (the torch layer is cached across wheel changes).
- Negative: sync wall-clock depends on worker speed: about 2× longer for
  `arctic-m` when both workers are busy, up to about 4× for corpora of
  small files that keep only one busy, and hours rather than minutes for
  `qwen3-embed` with `--remote-embed`.
- Negative: with the default `wait` policy, a sync stalls while all
  workers are down or version-mismatched (for example after upgrading the
  workstation but not the workers) instead of falling back.

### Risks and Mitigations

- **Risk**: A worker loads different weights or code than the workstation
  and silently mixes incompatible vectors into a collection.
  **Mitigation**: Handshake rejects any version, code-fingerprint,
  revision, prompt-policy or dimension mismatch; fastembed models are
  remote-eligible only with a pinned revision; deploy refuses wheels
  whose fingerprint differs from the workstation's.
- **Risk**: The image build resolves dependency versions (torch,
  onnxruntime, tokenizers) whose output differs from the workstation's,
  which the code fingerprint cannot detect.
  **Mitigation**: Deploy verification compares remote and local probe
  vectors before the worker serves and removes the container on failure;
  heartbeat health and `arc models workers` show dependency versions.
- **Risk**: First-use model download or another client's queued work
  makes a worker look hung.
  **Mitigation**: `loading` and `queued` heartbeat states; deadlines count
  only time in `encoding`.
- **Risk**: A `--remote-embed` `qwen3-embed` run evicts `arctic-m` on both
  workers and stalls routine syncs, or two clients with different models
  reload on every batch.
  **Mitigation**: Eviction waits for the resident model to go idle for the
  grace period; per-endpoint model restriction confines `qwen3-embed` to
  one worker when desired.
- **Risk**: Pickle deserialization from the network.
  **Mitigation**: Accepted for a trusted network. With host networking no
  Docker-published port bypasses the host firewall. `bind: tailscale`
  makes the server reachable only over the tailnet, where Tailscale
  access controls decide who can connect; document that a `bind: all`
  port must not be exposed beyond the LAN.
- **Risk**: After a reboot, a worker bound to its Tailscale address
  starts before tailscaled has configured that address.
  **Mitigation**: The server waits for the address and then binds;
  meanwhile the pool sees the endpoint as down and probes it.

### Failure Modes

- All endpoints down during sync: `wait` blocks visibly, naming the
  unreachable endpoints; `fail` ends the run with no chunks written for
  the file being embedded; `local` logs that it fell back.
- Endpoint dies mid-batch: batch retried elsewhere; logged with endpoint
  and request ID.
- Handshake mismatch: endpoint excluded for that model and treated as
  unable to serve it under the unavailable policy; shown by
  `arc models workers` with both sides' versions and fingerprints.
- Deploy preflight or build fails on a host: that host's running worker
  is left untouched, the remaining hosts proceed, and deploy exits
  non-zero naming the host and failing step.
- Deploy verification fails: the new container is removed, the worker is
  down until a deploy verifies, and syncs follow the unavailable policy.
- Worker's Tailscale address absent (tailscaled stopped or logged out):
  the server keeps waiting and logging; the pool sees the endpoint as
  down; deploy's health wait times out with the log tail showing the wait.
- Workstation resolves a `bind: tailscale` worker's name to a
  non-Tailscale address: the client reports the endpoint, the resolved
  address and the likely cause (Tailscale or MagicDNS off on the
  workstation).
- Diagnosis: `arc models workers` reports each endpoint's state, versions,
  fingerprint, dependency versions and resident model;
  `arc models deploy --dry-run` reports each host's deployed and desired
  build; the memory-probe log's `local_models` field shows whether the
  workstation process loaded a model.

## Implementation Plan

### Prerequisites

- [x] All Critical Assumptions verified
- [x] Docker available on each worker (Docker 29.8.1 on `dev-0`, `dev-1`)
- [x] Key-based SSH to each worker as a `docker`-group user (verified for
  `dev-0`, `dev-1`)
- [x] For `bind: tailscale`: workstation and workers on one tailnet with
  MagicDNS, worker names resolving to Tailscale addresses (verified)

### Minimum Viable Validation

With `dev-0` and `dev-1` configured as `host: dev-0` / `host: dev-1` with
`bind: tailscale`, deployed from the implementation branch with
`arc models deploy --build`, and the unavailable policy set to `fail`:

1. `arc corpus sync` of a scratch `arctic-m` corpus of a few markdown files
   completes, the workstation's memory-probe log shows an empty
   `local_models` throughout, and stored vectors match a local CPU
   re-embedding at cosine ≥ 0.9999.
2. `arc corpus sync --remote-embed --force` of a scratch `qwen3-embed`
   corpus of two small PDFs completes with the same checks.
3. `arc search semantic` against both scratch corpora (local query
   embedding) returns the same top results as when the corpora are indexed
   locally.
4. Deploy lifecycle: the first `--build` deploy creates and verifies both
   workers; an immediate rerun reports up to date; a code change followed
   by `--build` replaces both workers and `arc models workers` shows the
   new fingerprint; `--refresh` rebuilds without cache and verifies.
5. Listening address: a connection to a worker's LAN address is refused
   while `dev-0` connects; after the operator reboots one worker, it
   rejoins the pool with no intervention and its log shows the wait for
   the Tailscale address.

### Phase 1: Code Implementation

#### Step 1: Transport-neutral envelopes

Factor envelope send/receive and validation behind a connection-like
interface used by both the spawn-queue session and a socket connection;
keep the spawn-queue transport passing its existing tests. Add the code
fingerprint helper.

#### Step 2: fastembed revision pinning

Add `revision` to fastembed registry entries (`arctic-m` first); load
pinned snapshots via `specific_model_path` on both sides.

#### Step 3: `arc models serve`

Server entry point: listener, per-connection reader threads, single encode
executor, handshake, `loading` / `queued` / `encoding` heartbeat states,
model load on `initialize` with single-resident eviction after the grace
period, encode, and the listening address (`all`, `tailscale` or a
literal address; wait for it, bind, rebind on change).

#### Step 4: Remote pool client and routing

Endpoint config, handshake with re-handshake on probe, even sub-batch
dispatch, retry, health probing, unavailable policy, document-only
per-model routing in `EmbeddingClient.embed()` and `embed_parallel()`
(before `get_model()`), the client's loaded-model set and the memory
probe's `local_models` field, and `--remote-embed` / `--local-embed` on
`arc corpus sync` and `arc corpus repair`.

#### Step 5: `arc models workers`

List configured endpoints with state, versions, fingerprint, dependency
versions and resident model over status-only connections.

#### Step 6: `arc models deploy`

Wheel resolution (release download, `--build`, `--wheel`) with
fingerprint-from-wheel check; generated Dockerfile and image build on the
host; container specification and labels; desired-versus-current plan for
every lifecycle case, `--dry-run` and `--refresh`; preflight, copy,
build-then-replace order and health wait; verification with the probe set
and `--no-verify`.

### Phase 2: Operational Activation

#### Activation Step 1: Configure the workstation

Set the endpoints in arcaneum config (`host: dev-0` / `dev-1`, `port`,
`bind: tailscale`, served models, batch size, `authkey`; `ssh` only if it
differs from the host).

#### Activation Step 2: Deploy workers

Run `arc models deploy` with the released arcaneum to build, start and
verify `dev-0` and `dev-1`, then confirm with `arc models workers`. After
each later upgrade of arcaneum on the workstation, run
`arc models deploy` again.

### Deferred Work

- **Remote query embedding** for `qwen3-embed` corpora (local model load is
  ~10 s per search; a resident remote model answers in ~0.6 s). Needs a
  decision on dedicating a worker, or the Ollama alternative's lower
  memory.
- **Cross-file batching** for embedding (the RDR-024 shape), so corpora of
  small files keep more than one endpoint busy.
- **Worker removal command** (`arc models undeploy`): until then,
  `docker rm -f -v arcaneum-embed-<port>` and `docker image rm` on the
  host.

### Day 2 Operations

| Resource | List | Info | Delete | Verify | Backup |
| --- | --- | --- | --- | --- | --- |
| Endpoint config | In scope (`arc models workers`) | In scope (state, versions, resident model) | In scope (config edit) | In scope (handshake + heartbeat) | N/A |
| Worker images and containers | In scope (`arc models deploy --dry-run`) | In scope (labels: version, fingerprint, wheel SHA-256) | Deferred (`docker rm -f -v`, `docker image rm`) | In scope (deploy verification) | N/A |
| Worker model cache volume | Deferred | Deferred | Deferred (`docker volume rm arcaneum-embed-models`) | N/A | N/A |

Images are rebuilt from the wheel and the volume holds re-downloadable
models; Docker's own commands suffice for removal initially.

### New Dependencies

No Python dependencies. On the workstation, `arc models deploy` uses the
system OpenSSH client (`ssh`, `scp`), and `--build` uses `uv`. Workers
need Docker and, when an image is built, network access to Docker Hub
(base image), PyPI and the PyTorch CPU index; model downloads need
Hugging Face. A built worker restarts without network access.

## Validation

### Testing Strategy

1. **Scenario**: Socket transport round-trip with the deterministic fake
   backend.
   **Expected**: Same results and error envelopes as the queue transport.
2. **Scenario**: Handshake with mismatched protocol version, arcaneum
   version, code fingerprint, revision, prompt policy or dimension.
   **Expected**: Endpoint excluded for that model; no encode sent.
3. **Scenario**: Server still downloading or loading a model.
   **Expected**: Client sees `loading`, does not time out, and proceeds
   after `initialized`.
4. **Scenario**: Endpoint killed mid-batch with a second endpoint healthy.
   **Expected**: Batch retried; result identical and in order.
5. **Scenario**: All endpoints down under each policy.
   **Expected**: `wait` blocks and recovers, `fail` ends the run with no
   chunks written for the file being embedded, `local` embeds locally.
6. **Scenario**: Every endpoint mismatched under `wait`, then one is
   redeployed with a matching wheel.
   **Expected**: Wait message shows both sides' versions; the run resumes
   after re-handshake.
7. **Scenario**: Two clients on one endpoint, same model and then
   different models.
   **Expected**: Encodes serialized; the waiting client sees `queued` and
   does not time out; no eviction while the resident model has work
   within the grace period.
8. **Scenario**: Query embedding (`prompt_type="query"`) for a model
   configured as remote.
   **Expected**: Embedded locally; no request sent to the pool.
9. **Scenario**: Remote-routed `embed()` and `embed_parallel()`.
   **Expected**: No local model constructed; `local_models` stays empty.
10. **Scenario**: fastembed model without a pinned revision configured for
   remote routing.
   **Expected**: Refused with a message naming the missing revision.
11. **Scenario**: `--remote-embed` and `--local-embed` on sync and repair.
   **Expected**: Override per-model routing for that invocation only.
12. **Scenario**: Deploy plan for each lifecycle case (first install,
   upgrade, downgrade with retained image, update, reconfigure, refresh,
   stopped or missing container, nothing changed) against a fake remote
   runner.
   **Expected**: The table's action for each; `--dry-run` issues no
   changing remote command.
13. **Scenario**: Default deploy while the workstation runs a modified
   source tree; and a wheel built from that tree.
   **Expected**: Release wheel refused with a message naming `--build`;
   the fingerprint computed from the built wheel equals the installed
   tree's.
14. **Scenario**: Image build fails on one of two hosts.
   **Expected**: That host's running container is untouched; the other
   host deploys; exit non-zero.
15. **Scenario**: Verification probe below cosine 0.9999.
   **Expected**: New container removed, image kept, exit non-zero naming
   the model, minimum cosine and dependency versions; the next deploy
   recreates and re-verifies rather than restarting.
16. **Scenario**: Status-only connection to a running server.
   **Expected**: No model loaded; heartbeat reports version, fingerprint,
   dependency versions, resident model and state.
17. **Scenario**: `bind: tailscale` with no Tailscale address present,
   which then appears, then changes (fake interface table).
   **Expected**: Server waits and logs, binds when the address appears,
   and rebinds to the new address.
18. **Scenario**: `bind: all`, `tailscale` and a literal address.
   **Expected**: Listener on `0.0.0.0`, the Tailscale address, and the
   literal address respectively; deploy passes `--network host` and
   publishes no ports.
19. **Scenario**: Client endpoint with `bind: tailscale` whose host
   resolves outside `100.64.0.0/10`.
   **Expected**: Error names the endpoint, the resolved address and the
   Tailscale or MagicDNS cause.

### Performance Expectations

The goal is local resource reduction, measured as workstation CPU time and
peak RSS of `arc corpus sync` with and without the pool. Remote throughput
is in the feasibility spike above; no throughput target is set.

## Finalization Gate

### Contradiction Check

No contradictions found between research findings, design principles, and
proposed solution. The earlier draft proposed remote query embedding for
all models; the local profile showed it would not help `arctic-m` search,
so query routing moved to Deferred Work. A review against the code then
found that routing at `embed()` would also catch queries, that the RDR-022
serve loop cannot report `loading` or switch models, that the memory probe
cannot show model loads, that version strings do not distinguish
development builds, and that concurrent clients and per-file dispatch
were unaddressed; the design now covers each. Deployment then moved from
a manual `docker run` with install at container start to
`arc models deploy`, which builds the image on each worker; pre-staging a
different version was dropped because a worker that does not match the
workstation can neither be verified nor used by the pool. Binding to a
Tailscale address moved from Docker port publishing, which fails at boot
before tailscaled configures the address, to host networking with the
server waiting for and binding the address itself.

### Assumption Verification

All three Critical Assumptions are verified: vector parity by spike for
both models in use, fastembed pinning by source search of fastembed 0.8.0,
and transport framing by source search of the Python 3.12 stdlib.

#### API Verification

| API Call | Library | Verification |
| --- | --- | --- |
| `multiprocessing.connection.Listener` / `Client` (`authkey`) | stdlib 3.12 | Source Search |
| `Connection.send_bytes` length framing | stdlib 3.12 | Source Search |
| `TextEmbedding(..., specific_model_path=...)` | fastembed 0.8.0 | Source Search |
| `snapshot_download(repo_id, revision=..., cache_dir=...)` | huggingface_hub | Source Search (fastembed's own call site); exercised in Step 2 tests |
| `EmbeddingClient.embed()` on CPU in a Linux container | arcaneum | Spike |
| `ssh -o BatchMode=yes <host> docker version` | OpenSSH, Docker 29.8.1 | Verified on `dev-0`, `dev-1` |
| `uv build --wheel` on the setuptools backend | uv | Exercised in Step 6 tests |

### Scope Verification

The Minimum Viable Validation is in scope: the five checks run against
`dev-0` and `dev-1` as the last step of Phase 1, before activation.

### Cross-Cutting Concerns

- **Versioning**: Handshake carries protocol and arcaneum versions and a
  code fingerprint; deploy builds a new image when the wheel changes and
  refuses wheels that do not match the workstation.
- **Build tool compatibility**: `--build` uses `uv build --wheel` on the
  existing setuptools backend.
- **Licensing**: N/A (no new dependencies)
- **Deployment model**: `arc models deploy` builds an unpublished image on
  each worker from stock `python:3.12-slim` and runs it with a named model
  cache volume.
- **IDE compatibility**: N/A
- **Incremental adoption**: Off unless endpoints are configured;
  `qwen3-embed` stays local unless `--remote-embed` is given.
- **Secret/credential lifecycle**: Trusted LAN; the optional static
  `authkey` is passed to the container as environment, visible to anyone
  with Docker access on the worker. Deployment uses the operator's
  existing SSH keys.
- **Memory management**: The workstation loads no model for remote-routed
  work. Workers keep one resident model (6.0–7.1 GB peak measured) with a
  configurable batch size.

### Proportionality

Right-sized. The Ollama spike table is retained because it records why a
lower-memory option was not chosen for Phase 1. Deploy is in Phase 1
because workers must track the workstation's exact code through every
upgrade and development build, which a manual `docker run` would leave to
the operator; removal stays deferred.

## References

- RDR-022: Spawned Accelerator Worker Protocol
- RDR-024: Adaptive Cross-File Index Batching
- `docs/cloud-bulk-embedding-feasibility.md`
- Katas: `arcaneum#rrky`, `arcaneum#tpcs`, `arcaneum#45ed`, `arcaneum#vby2`
- `src/arcaneum/embeddings/worker_protocol.py`,
  `src/arcaneum/embeddings/client.py`, `src/arcaneum/search/embedder.py`,
  `src/arcaneum/cli/sync.py`
- fastembed 0.8.0: `fastembed/common/model_management.py`,
  `fastembed/text/onnx_embedding.py`
- Python 3.12 `multiprocessing/connection.py`
- Ollama Cloud docs (`https://docs.ollama.com/cloud`); ollama/ollama#17129

# RDR-022: Spawned Accelerator Worker Protocol

## Status

Accepted

## Context

Arcaneum currently invokes PyTorch accelerator work in a daemon thread to impose a
deadline. Python cannot cancel that thread or reclaim native Metal/CUDA work when the
deadline expires. Loading a CPU fallback while the timed-out thread still owns native
state can increase unified-memory pressure, and interpreter shutdown can race tensor
and allocator teardown. A daemon thread therefore reports a timeout but does not
contain the failed computation.

## Decision

Accelerator execution will move behind a persistent process created with the
`multiprocessing` **spawn** context. The protocol is backend-neutral and versioned.
It defines serializable initialize/initialized, encode/encoded,
heartbeat/health, shutdown/shutdown-complete, and structured error envelopes. Every
message carries a request identifier.

Each worker has bounded command and reply queues and permits exactly one in-flight
request. The backend model is constructed once in the child. Parent modules must not
import or initialize Torch, MPS, CUDA, or another accelerator runtime. Encoded output
crosses the boundary as plain CPU data and is reconstructed as an independently owned,
C-contiguous NumPy array.

Timeouts, malformed replies, crashes, and interrupts terminate and reap the process
before control returns to fallback code. Graceful shutdown is requested first and is
also followed by a join. Heartbeat describes process/backend health; it is distinct
from encode completion. A later implementation may use a dedicated child heartbeat
channel while an encode is active without changing the wire envelope.

This decision supplies protocol and lifecycle primitives plus a deterministic fake
backend. Migrating real accelerator execution is deliberately deferred to kata
`7yd3`.

## Consequences

- Native accelerator state has a single, killable owner.
- Model load cost is paid once per persistent worker, not once per batch.
- Spawn has startup and serialization costs, so batches must remain coarse enough to
  amortize them.
- An encode that times out loses the worker and its model state; restart/fallback
  policy belongs to the caller.
- Only one encode runs per worker. Scaling requires explicit additional workers and
  memory qualification, not hidden thread concurrency.

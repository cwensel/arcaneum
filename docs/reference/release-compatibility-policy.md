# Arcaneum 1.0 Release and Compatibility Policy

Arcaneum remains an alpha package until the gates below are met. Do not change
the `Development Status :: 3 - Alpha` classifier in `pyproject.toml` until a
release candidate satisfies this checklist and the release owner records the
evidence in the release notes.

## Supported Runtime Matrix

Arcaneum 1.0 supports Python 3.12 and 3.13. A release candidate must pass the
unit test suite on both versions before the classifier can move to Beta or
Production/Stable.

The service compatibility window for 1.0 is:

| Service | Supported range | Gate |
| ------- | --------------- | ---- |
| Qdrant | 1.16.x | `arc doctor`, corpus create/sync/search, backup, and restore pass against `qdrant/qdrant:v1.16.2` |
| MeiliSearch | 1.12.x | corpus create/sync/search, parity, backup, and restore pass against `getmeili/meilisearch:v1.12` |

Changing either range before 1.0 requires a release note and an upgrade test
from the previous supported image. After 1.0, widening a range is
backward-compatible; dropping a previously supported minor version is breaking
unless the old service is already unsupported upstream or has a published data
loss/security issue.

## Embedding Backend Support

Support tiers define the promise users can rely on:

| Tier | Backends | 1.0 promise |
| ---- | -------- | ----------- |
| Stable default | FastEmbed CPU models used by default corpus types | Must run in CI-covered unit paths and in release smoke tests without optional ML extras |
| Supported opt-in | SentenceTransformers models behind `arcaneum[sentence-transformers]` or `arcaneum[large-models]` | Must preserve documented dependency caps and produce clear model/runtime errors |
| Experimental accelerator | `--gpu`, CUDA, MPS, and CoreML/FastEmbed paths | Must remain opt-in, have smoke tests where hardware is available, and degrade to actionable errors rather than implicit data changes |

Default model changes are compatibility events because they affect embedding
dimensions, prompt policy, or retrieval behavior. A 1.0 default-model change
requires release notes, a reindex warning, and tests covering stale prompt-policy
or dimension rejection.

## Persisted Data Compatibility

Arcaneum treats stored Qdrant collection metadata, Qdrant payloads, and
MeiliSearch documents as persisted user data. The
[Persisted Schema Contract](../guides/persisted-schema.md) is part of the 1.0
compatibility promise.

The 1.0 gate requires:

- `schema_version` and `app_version` are written on new persisted records.
- `arc collection verify --json` and `arc corpus verify --json` reject missing,
  older, invalid, or newer schema versions with structured errors.
- Any breaking persisted-field change increments `schema_version` and ships a
  migration, verifier, or explicit reindex path.
- Export/import and backup/restore preserve schema metadata and fail closed when
  they cannot preserve it.

Backward-compatible additions may add optional payload fields without a schema
bump when older readers can ignore them and newer readers tolerate absence.
Renames, removals, type changes, or meaning changes are breaking.

## Migration and Deprecation Rules

Before 1.0, compatibility-breaking changes are allowed only with a release note
and a direct repair path such as reindex, export/import, parity repair, or a
targeted migration command.

After 1.0:

- Public CLI commands, JSON fields, persisted schema, and default model policies
  require one minor release of deprecation before removal or incompatible
  behavior change.
- Deprecation warnings must name the replacement and the first release where
  removal can happen.
- Security fixes may bypass the deprecation window when preserving the old
  behavior would keep users exposed.
- Best-effort migration is required for user-created data; silent deletion,
  silent re-embedding, or silent prompt-policy changes are not acceptable.

## Security Defaults

The 1.0 release must be secure by default for local development:

- Services bind to local interfaces unless the user explicitly configures
  otherwise.
- Backup and restore commands do not include local configuration secrets, source
  files, cached embedding models, or Docker images.
- Corporate-proxy and offline guidance keeps certificate overrides explicit and
  documented.
- Remote model downloads are opt-in through documented model selection and cache
  behavior; default smoke tests must run without requiring private credentials.

Any change that broadens network exposure, weakens TLS/certificate handling, or
starts including secrets in artifacts is release-blocking until reviewed.

## Backup and Restore Expectations

`arc container backup` is the supported 1.0 backup path for indexed data. A
release candidate must prove that backup and restore cover:

- Qdrant collections and collection metadata.
- MeiliSearch index settings and JSONL document exports.
- Corpus metadata needed for parity and verification commands.
- A restore into a clean service pair using the supported service matrix.

Backups are crash-consistent only when indexing is idle. The command must keep
checking MeiliSearch task activity and abort when it detects active work during
the backup window. Source files referenced by indexes remain outside the backup
contract and must be called out in release notes and docs.

## CI and Release Gates

A 1.0 release candidate is eligible only when all of these pass from a clean
checkout:

- Unit tests on Python 3.12 and 3.13.
- Integration smoke tests against supported Qdrant and MeiliSearch images.
- CLI smoke tests for `arc doctor`, `arc corpus create`, `arc corpus sync`,
  semantic search, text search, parity, backup, and restore.
- Offline/default-model smoke tests that avoid optional SentenceTransformers
  extras.
- Optional accelerator smoke tests on available CUDA or MPS hardware, with the
  result recorded as hardware-specific evidence.
- Package build, install via `pipx` or equivalent isolated environment, and
  command discovery for the Claude Code plugin.

Only after those gates pass may the release owner update the package classifier,
tag the release, and publish release notes with compatibility evidence,
supported service versions, migration notes, and known experimental limits.

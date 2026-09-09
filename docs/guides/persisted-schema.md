# Persisted Schema Contract

Arcaneum writes a `schema_version` and `app_version` to persisted Qdrant
collection metadata, Qdrant payloads, and MeiliSearch documents.

## Compatibility

`schema_version` tracks the shape and meaning of persisted fields. Adding an
optional payload field is backward-compatible and does not require a version
bump. Renaming a field, removing a field, changing a field type, or changing a
field's meaning is breaking and requires a new `schema_version`.

`app_version` records the Arcaneum release that wrote the metadata or document.
It is diagnostic only; compatibility decisions use `schema_version`.

## File Manifests and Indexing Policy

Corpus sync publishes a file manifest only after that file's chunks are durable.
The manifest records the source hash, quick metadata hash, chunk count, source
type, `indexing_policy`, and—where available—a `quality_manifest`. Manifest
records are control metadata and are excluded from ordinary search results.

`indexing_policy` independently identifies extraction and chunking behavior for
each corpus type. A source file can therefore be unchanged while its indexed
representation is stale. Ordinary sync and repair report policy-only staleness
without rewriting the file; `--include-stale-policy` opts into that migration.

PDF quality manifests use schema version 1 and record page coverage, OCR
provenance, extraction candidates, replacement-character omissions, warnings,
and the active indexing policy. Content-identical PDFs also retain their source
paths as aliases of one canonical searchable document. These are optional,
backward-compatible fields: legacy records remain readable and verification
reports the missing evidence until an explicit migration or reindex.

## Migration

Collections without `schema_version` are legacy schema v0. Collections with an
older schema version should be repaired before use by reindexing the corpus, or
by a targeted metadata backfill when the stored payload shape is known to still
match the current contract.

`arc collection verify <name> --json` and `arc corpus verify <name> --json`
surface legacy, older, invalid, or newer schema versions in the `errors` field
and mark the Qdrant side unhealthy until the metadata is repaired.

For corpus-level extraction and policy findings, use `arc corpus verify <name>
--json`. Use `arc corpus repair <name> --dry-run --json` to preview automatic
content repairs, and add `--include-stale-policy` only when intentionally
migrating otherwise healthy files.

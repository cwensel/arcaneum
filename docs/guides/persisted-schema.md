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

## Migration

Collections without `schema_version` are legacy schema v0. Collections with an
older schema version should be repaired before use by reindexing the corpus, or
by a targeted metadata backfill when the stored payload shape is known to still
match the current contract.

`arc collection verify <name> --json` and `arc corpus verify <name> --json`
surface legacy, older, invalid, or newer schema versions in the `errors` field
and mark the Qdrant side unhealthy until the metadata is repaired.

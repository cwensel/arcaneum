---
description: Manage corpora for semantic and full-text search (recommended)
argument-hint: <create|list|update|delete|sync|repair|migrate-text|hook|jobs|info|items|parity|verify|doctor> <name> [paths...] [options]
---

**Recommended for most users.** Manage corpora that combine vector search and full-text
search for the same content. Full-text search comes from a text index in the corpus's Qdrant
collection (the `qdrant` text backend, the default for a new install) or, on an install that
runs MeiliSearch, from a MeiliSearch index of the same name (`meili`).

**IMPORTANT:** You must specify a subcommand.

**Subcommands:**

- `create`: Create the Qdrant collection (and a MeiliSearch index for a `meili` corpus)
- `list`: List all corpora with text coverage (qdrant) or parity (meili) status
- `update`: Update corpus metadata without reindexing
- `delete`: Delete the Qdrant collection and any MeiliSearch index of the corpus
- `sync`: Index a directory into the corpus (Qdrant, plus MeiliSearch on a `meili` corpus)
- `repair`: Selectively repair verifier-selected unhealthy files and PDF duplicates,
  then restore text coverage (qdrant) or parity (meili)
- `migrate-text`: Move a corpus's text search between MeiliSearch and Qdrant; `--all --to qdrant`
  moves every corpus on MeiliSearch (see docs/guides/qdrant-only-text.md)
- `hook`: Install a git hook that auto-syncs a repo on every commit
- `jobs`: List running syncs, repairs, and hook drains, plus queued hook work
  (read-only; takes no locks)
- `info`: Show corpus details
- `items`: List indexed items (with parity status on a `meili` corpus)
- `parity`: Check and restore the text index; Qdrant/MeiliSearch parity applies to `meili` corpora only
- `verify`: Verify corpus health (and Qdrant/MeiliSearch parity for a `meili` corpus)
- `doctor`: Check all corpora (or the named ones) read-only and list the commands
  that fix each problem

**Common Options:**

- --json: Output in JSON format
- --details: Show extended list columns, including exact item counts

**Create Options:**

- name: Corpus name (required)
- --type: Corpus type - code, pdf, or markdown (required)
- --models: Embedding models, comma-separated (default inferred from --type: arctic-m for pdf/markdown, jina-code for code)

**Delete Options:**

- name: Corpus name (required)
- --confirm: Skip confirmation prompt
- --json: Output in JSON format

**Sync Options:**

- name: Corpus name (required, first positional argument)
- directories: One or more directory paths to index (required)
- --models: Embedding models (default: use corpus metadata)
- --file-types: File extensions to index (e.g., .py,.md)
- --gpu: Opt into accelerator embedding (CPU is the stable default)
- --remote-embed: Also send on-request models (e.g. qwen3-embed) to the
  configured remote embedding workers for this run
- --local-embed: Embed every model locally for this run, ignoring configured
  remote embedding workers
- --background: Run below interactive work (compilers, tests) for continuous
  indexing; nice 10 on macOS, SCHED_IDLE on Linux
- --changed-since REV: Sync only what a git commit or range touched (e.g. HEAD,
  ORIG_HEAD..HEAD), removing the files it deleted
- --order: Process files by path (default), newest modification first, or oldest
  modification first; this changes processing order, not search ranking
- --include-stale-policy: Also re-index otherwise unchanged files whose indexing
  policy is stale or missing
- --from-index SELECTOR: Instead of paths, re-extract the indexed files whose
  stored metadata matches SELECTOR (e.g. method=pymupdf_normalized,
  version<0.10.0); shorthand keys are method, fallback, ocr, warning, version;
  `!key` selects files that do not record key
- --no-wait: Fail instead of queueing when another sync of this corpus is running
- --lock-timeout: Seconds to wait for the corpus write lock (default: 600)

**Repair Options:**

- name: Corpus name (required)
- --quality-threshold: Text quality threshold (0.0-1.0, default: 0.9)
- --dry-run: Preview what would be repaired without making changes
- --include-stale-policy: Also re-index files whose only issue is stale or
  missing indexing policy
- --gpu: Opt into accelerator embedding (CPU is the stable default)
- --remote-embed / --local-embed: Route this repair's document embedding as for
  sync (remote embedding workers)
- --background: Run below interactive work, as for sync
- --verbose: Show per-file quality scores and details
- --json: Output in JSON format

**Hook Options:**

`hook` takes its own subcommand: `install`, `uninstall`, or `status`.

- name: Corpus name. Omit it on `install` to be walked through picking or
  creating a corpus, choosing hook points, and backfilling.
- --repo: Repository to act on (default: current directory)
- --branch: (install) Branch to follow - the remote's default branch unless
  named; other branches and worktrees are never indexed
- --hook: Hook point, repeatable - reference-transaction, post-commit,
  post-merge, post-checkout, or post-rewrite. Install defaults to
  reference-transaction, post-checkout and post-rewrite; uninstall defaults to
  all of them.
- --no-spawn: (install) Queue touched paths but start no background worker
- --yes/-y: (install) Take the defaults instead of prompting
- --service: Also register/remove an OS watcher that drains the spool after a
  reboot or a failed spawn
- --json: Output in JSON format

**Info/Items Options:**

- name: Corpus name (required)
- --json: Output in JSON format

**Parity Options** (the MeiliSearch ones apply to `meili` corpora only):

- name: Corpus name (optional - if omitted, processes all corpora)
- --dry-run: Preview what would be backfilled without making changes
- --verify: Verify chunk counts match between Qdrant and MeiliSearch (`meili` corpora)
- --repair-metadata: Update MeiliSearch docs with missing git metadata (code corpora)
- --create-missing: Create missing MeiliSearch indexes for qdrant_only corpora
- --confirm: Skip confirmation prompt when processing all corpora
- --prune-orphans: Delete MeiliSearch documents for files gone from both Qdrant
  and disk (files whose directory is missing are kept: possible unmounted volume)
- --force-prune: Like --prune-orphans, but also prune files whose directory is missing
- --verbose: Show detailed progress
- --json: Output in JSON format

**Doctor Options:**

- names: Corpus names (optional - if omitted, checks all corpora)
- --verbose: Also show informational findings and counts
- --strict: Exit 1 when any corpus needs action
- --json: Output in JSON format (one document; `data.commands` lists what to run)

`/corpus doctor` writes nothing. Required fixes are combined per corpus (for
example `arc corpus repair NAME --prune-orphans`); policy migration and pruning
files whose directory is missing are listed separately as optional. PDFs whose
only defect is extraction from before the column-order fix get a targeted
`arc corpus sync NAME --from-index 'method=pymupdf_normalized,!fallback'`.

**Repair Behavior:**

`/corpus repair NAME` re-indexes only files selected by corpus verification,
including corrupt PDF extraction, incomplete or duplicate chunks, and duplicate
PDF sources. Files whose only finding is stale or missing indexing policy are
reported but deferred by default; pass `--include-stale-policy` to migrate them.
Files that also have an integrity or quality defect remain selected without the
flag. Preview the exact set with `/corpus repair NAME --dry-run --json`.
Identical PDFs retain every physical path as an alias of one searchable canonical
document; parity cleanup promotes a live alias if the canonical path disappears.
On a `meili` corpus, repair then restores Qdrant/MeiliSearch parity (as
`/corpus parity NAME --verify` does), so it also fixes the `mismatch` that
`/corpus verify` reports. Under `--json` the parity result is nested at
`data.parity`. MeiliSearch documents for files gone from both Qdrant and disk
are only reported; pass `--prune-orphans` (or `--force-prune`) to delete them.

**Examples:**

```text
/corpus create MyDocs --type pdf
/corpus create MyDocsQuality --type pdf --models stella
/corpus sync MyDocs ~/Documents
/corpus create CodeBase --type code
/corpus sync CodeBase ~/projects --file-types .py,.js,.md
/corpus sync CodeBase ~/project1 ~/project2 ~/project3
/corpus repair PapersFast
/corpus repair PapersFast --dry-run
/corpus repair PapersFast --include-stale-policy
/corpus repair PapersFast --quality-threshold 0.5
/corpus sync CodeBase --changed-since HEAD
/corpus sync PapersFast --from-index method=pymupdf_normalized --dry-run
/corpus sync CodeBase ~/projects --order newest
/corpus hook install
/corpus hook install CodeBase
/corpus hook install CodeBase --branch develop
/corpus hook status
/corpus hook uninstall CodeBase
/corpus info MyDocs
/corpus items CodeBase
/corpus parity CodeBase --verify
/corpus parity CodeBase --repair-metadata
/corpus parity --create-missing --dry-run
/corpus parity --create-missing --confirm
/corpus delete OldCorpus
/corpus delete OldCorpus --confirm
```

**Execution:**

```bash
arc corpus $ARGUMENTS
```

**What Is a Corpus?**

A corpus combines two search modes over one Qdrant collection:

1. **Vector search** (Qdrant): Semantic similarity, concept matching
2. **Full-text search**: Keyword and phrase matching, from a BM25 text index in the
   same collection (default, `qdrant` backend) or from a MeiliSearch index (`meili`
   backend, optional; boolean operators)

To move an existing MeiliSearch install to Qdrant-only text, run
`arc corpus migrate-text --all --to qdrant` (see docs/guides/qdrant-only-text.md).

This enables hybrid search strategies:

- Broad semantic discovery (vector search)
- Precise keyword refinement (full-text search)
- Combined results for best of both worlds

**When to Use Corpus vs Collection:**

**Use Corpus (Recommended):**

- Default choice for most use cases
- Need both semantic and keyword search
- Users search different ways (concepts vs exact terms)
- Want fast keyword filtering of semantic results
- Building search UIs with multiple search modes

**Use Collection (Advanced):**

- Only need semantic search (no full-text)
- Working with embeddings/vectors directly
- Integrating with existing vector workflows
- Full-text search not needed

**How Sync Works:**

1. Discovers files in directory (respects .gitignore for code)
2. Chunks content appropriately (PDFs vs code)
3. Generates embeddings with specified models
4. Uploads to Qdrant (vector search)
5. Builds the text index in the same Qdrant collection; a `meili` corpus also indexes to
   MeiliSearch, sharing the same document IDs and metadata

**Keeping a Repo in Sync Automatically:**

`/corpus hook install NAME` installs git hooks so an indexed repo stops
drifting behind its source tree between manual syncs. The corpus follows one
branch (the remote's default unless `--branch` names another). Whenever that
branch moves while checked out (commit, merge, pull, rebase, reset, or a tool's
`update-ref`), the hook asks git which paths changed since the last commit it
queued, queues them, and indexes them in the background, so a burst of commits
pays one embedding-model load instead of one per commit. It never blocks or
fails a git operation.

Run `/corpus hook install` with no corpus name for a guided setup: it lists
existing corpora or offers to create one (inferring the type from the repo's
contents), suggests hook points, and offers the initial backfill at background
priority. A hook only sees future commits, so already-committed files need one
real sync.

`/corpus sync NAME --changed-since HEAD` does the same thing on demand, without
installing anything.

Changed paths are written to a durable spool, so a failed or interrupted worker
does not lose queued work. Git operations never wait for indexing. Sync writes
to one corpus are serialized, while different corpora can proceed concurrently.
Hook and drain outcomes are recorded in
`~/.local/state/arcaneum/hook.log`.

Install the optional `arcaneum[proctitle]` extra to make background workers
appear as readable `arc ...` commands in `ps` and `top`.

**Performance:**

A `meili` corpus sync is about 2x slower than single-system indexing (dual upload); a `qdrant`
corpus has no second upload:

- PDFs: ~5-15/minute
- Source files: 50-100 files/second

**Related Commands:**

- /arc:search semantic - Search vector index
- /arc:search text - Search full-text index
- /arc:collection create - Create vector-only collection (advanced)
- /arc:indexes create - Create MeiliSearch full-text index only (advanced; needs `meilisearch.enabled`)
- /arc:index pdf - Index PDFs to vector only (advanced)
- /arc:index code - Index code to vector only (advanced)

**Implementation:**

- RDR-009: Dual indexing strategy (the `meili` backend)
- RDR-006: Claude Code integration

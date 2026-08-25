---
description: Manage dual-index corpora (recommended)
argument-hint: <create|list|update|delete|sync|repair|hook|info|items|parity|verify> <name> [paths...] [options]
---

**Recommended for most users.** Manage corpora that combine both vector search (Qdrant) and
full-text search (MeiliSearch) for the same content.

**IMPORTANT:** You must specify a subcommand.

**Subcommands:**

- `create`: Create both Qdrant collection and MeiliSearch index
- `list`: List all corpora with parity status
- `update`: Update corpus metadata without reindexing
- `delete`: Delete both Qdrant collection and MeiliSearch index
- `sync`: Index directory to both systems simultaneously
- `repair`: Re-index incomplete or garbled files (text quality detection)
- `hook`: Install a git hook that auto-syncs a repo on every commit
- `info`: Show corpus details (both systems)
- `items`: List indexed items with parity status
- `parity`: Check and restore parity between systems
- `verify`: Verify corpus health across both Qdrant and MeiliSearch

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
- --changed-since REV: Sync only what a git commit or range touched (e.g. HEAD,
  ORIG_HEAD..HEAD), removing the files it deleted
- --no-wait: Fail instead of queueing when another sync of this corpus is running
- --lock-timeout: Seconds to wait for the corpus write lock (default: 600)

**Repair Options:**

- name: Corpus name (required)
- --quality-threshold: Text quality threshold (0.0-1.0, default: 0.9)
- --dry-run: Preview what would be repaired without making changes
- --gpu: Opt into accelerator embedding (CPU is the stable default)
- --verbose: Show per-file quality scores and details
- --json: Output in JSON format

**Hook Options:**

`hook` takes its own subcommand: `install`, `uninstall`, or `status`.

- name: Corpus name. Omit it on `install` to be walked through picking or
  creating a corpus, choosing hook points, and backfilling.
- --repo: Repository to act on (default: current directory)
- --hook: Hook point - post-commit (default), post-merge, post-checkout, or
  post-rewrite. On uninstall, defaults to all of them.
- --no-spawn: (install) Queue touched paths but start no background worker
- --yes/-y: (install) Take the defaults instead of prompting
- --service: Also register/remove an OS watcher that drains the spool after a
  reboot or a failed spawn
- --json: Output in JSON format

**Info/Items Options:**

- name: Corpus name (required)
- --json: Output in JSON format

**Parity Options:**

- name: Corpus name (optional - if omitted, processes all corpora)
- --dry-run: Preview what would be backfilled without making changes
- --verify: Verify chunk counts match between systems
- --repair-metadata: Update MeiliSearch docs with missing git metadata (code corpora)
- --create-missing: Create missing MeiliSearch indexes for qdrant_only corpora
- --confirm: Skip confirmation prompt when processing all corpora
- --verbose: Show detailed progress
- --json: Output in JSON format

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
/corpus repair PapersFast --quality-threshold 0.5
/corpus sync CodeBase --changed-since HEAD
/corpus hook install
/corpus hook install CodeBase
/corpus hook install CodeBase --hook post-merge
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

A corpus combines two search systems:

1. **Vector search** (Qdrant): Semantic similarity, concept matching
2. **Full-text search** (MeiliSearch): Keyword, phrase, boolean operators

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
- MeiliSearch not available/needed

**How Sync Works:**

1. Discovers files in directory (respects .gitignore for code)
2. Chunks content appropriately (PDFs vs code)
3. Generates embeddings with specified models
4. Uploads to Qdrant (vector search)
5. Indexes to MeiliSearch (full-text search)
6. Both indexes share same document IDs and metadata

**Keeping a Repo in Sync Automatically:**

`/corpus hook install NAME` installs a git hook so an indexed repo stops
drifting behind its source tree between manual syncs. On each commit the hook
asks git which paths changed, queues them, and indexes them in the background —
so a burst of commits pays one embedding-model load instead of one per commit.
It never blocks or fails a git operation.

Run `/corpus hook install` with no corpus name for a guided setup: it lists
existing corpora or offers to create one (inferring the type from the repo's
contents), suggests hook points, and offers the initial backfill. A hook only
sees future commits, so already-committed files need one real sync.

`/corpus sync NAME --changed-since HEAD` does the same thing on demand, without
installing anything.

**Performance:**

Corpus sync is approximately 2x slower than single-system indexing due to dual upload, but still efficient:

- PDFs: ~5-15/minute
- Source files: 50-100 files/second

**Related Commands:**

- /arc:search semantic - Search vector index
- /arc:search text - Search full-text index
- /arc:collection create - Create vector-only collection (advanced)
- /arc:indexes create - Create full-text index only (advanced)
- /arc:index pdf - Index PDFs to vector only (advanced)
- /arc:index code - Index code to vector only (advanced)

**Implementation:**

- RDR-009: Dual indexing strategy
- RDR-006: Claude Code integration

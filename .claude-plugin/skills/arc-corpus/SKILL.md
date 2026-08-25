---
name: arc-corpus
description: Dual-index corpus management for combined semantic and full-text search. Use when user mentions corpus, dual indexing, syncing content to both Qdrant and MeiliSearch, checking parity between systems, deleting corpora, installing a git hook to keep an indexed repo current automatically, or managing content that needs both search types.
allowed-tools: Bash(arc:*), Read
---

# Corpus Management (Dual-Index)

A corpus maintains both a Qdrant collection (semantic search) and a MeiliSearch index (full-text search) in sync.

**For adding or updating content, always prefer `arc corpus sync` over `arc index`, `arc collection`, or `arc indexes`.**
The `corpus` commands manage Qdrant and MeiliSearch together. Single-system commands
(`arc index`, `arc collection`, `arc indexes`) are for advanced workflows that need one system without the other.

```bash
# Create corpus (creates both collection and index)
arc corpus create MyCorpus --type pdf
arc corpus create MyCorpus --type code
arc corpus create MyCorpus --type markdown
arc corpus create MyCorpus --type pdf --models qwen3-embed,jina-v3  # Multiple models

# Delete corpus (deletes both collection and index)
arc corpus delete MyCorpus              # With confirmation prompt
arc corpus delete MyCorpus --confirm    # Skip confirmation
arc corpus delete MyCorpus --confirm --json  # JSON output

# Sync files to both systems (preferred command for adding/updating content)
arc corpus sync MyCorpus /path/to/files
arc corpus sync MyCorpus /path/one /path/two /path/three    # Multiple directories
arc corpus sync MyCorpus /path/to/files --parity            # Detect renames, remove files no longer on disk
arc corpus sync MyCorpus /path/to/files --parity --dry-run  # Preview parity changes first
arc corpus sync MyCorpus /path/to/files --force             # Force reindex
arc corpus sync MyCorpus /path/to/files --verify            # Verify after sync
arc corpus sync MyCorpus /path/to/files --verbose           # Show progress
arc corpus sync MyCorpus /path/to/files --no-gpu            # CPU-only mode (stable on Apple Silicon)

# Keep a git repo in sync automatically (installs a git hook)
arc corpus hook install                               # Guided: pick/create corpus, choose hooks
arc corpus hook install MyCorpus                      # Auto-sync on every commit
arc corpus hook install MyCorpus --hook post-merge    # Also stay current after pulls
arc corpus hook install MyCorpus --service            # Plus an OS watcher for reboot recovery
arc corpus hook status                                # What is installed in this repo
arc corpus hook uninstall MyCorpus                    # Remove just this corpus's hook

# Sync only what a commit touched, without installing a hook
arc corpus sync MyCorpus --changed-since HEAD              # Last commit
arc corpus sync MyCorpus --changed-since ORIG_HEAD..HEAD   # Since a pull

# Repair incomplete or garbled files
arc corpus repair MyCorpus                            # Detect and fix quality issues
arc corpus repair MyCorpus --dry-run                  # Preview what would be repaired
arc corpus repair MyCorpus --quality-threshold 0.5    # More aggressive detection
arc corpus repair MyCorpus --verbose                  # Show per-file quality scores

# View corpus info (both systems)
arc corpus info MyCorpus
arc corpus info MyCorpus --json

# List indexed items with parity status
arc corpus items MyCorpus               # Table output with Q/M chunk counts
arc corpus items MyCorpus --json        # JSON output for automation

# Check and restore parity between systems
arc corpus parity MyCorpus              # Check and backfill single corpus
arc corpus parity MyCorpus --dry-run    # Preview only
arc corpus parity MyCorpus --verify     # Verify chunk counts match
arc corpus parity MyCorpus --repair-metadata  # Fix missing git metadata (code corpora)
arc corpus parity MyCorpus --verbose    # Detailed progress

# All-corpora mode (no corpus name)
arc corpus parity                       # Process all corpora
arc corpus parity --dry-run             # Preview all
arc corpus parity --confirm             # Skip confirmation prompt

# Create missing MeiliSearch indexes for qdrant_only corpora
arc corpus parity --create-missing --dry-run   # Preview what would be created
arc corpus parity --create-missing --confirm   # Create and sync all
```

## When to Use Corpus vs Collection/Index

- **Use Corpus**: When you need both semantic search (conceptual queries) AND full-text search (exact phrases)
- **Use Collection alone**: When you only need semantic/conceptual search
- **Use Index alone**: When you only need exact keyword/phrase search

## Parity Behavior

The `--parity` flag on `arc corpus sync` and the standalone `arc corpus parity`
command both ensure both systems hold the same content, but operate differently:

- `arc corpus sync --parity <path>`: Runs during a sync. In addition to indexing
  new/changed files in `<path>`, it detects renamed/moved files by content hash
  and **removes indexed entries for files that no longer exist on disk** (scoped
  to the directories being synced).
- `arc corpus parity`: Standalone cross-system repair with no directory scan.
  Backfills missing entries between Qdrant and MeiliSearch but does not remove
  missing-from-disk files.

Cross-system backfill direction (both commands):

- **Qdrant -> MeiliSearch**: Copies metadata (fast, no file access needed)
- **MeiliSearch -> Qdrant**: Re-chunks and embeds files (requires file access)

**Creating Missing Indexes:**

Use `--create-missing` to promote single-sided Qdrant collections into full corpora:

- Creates MeiliSearch indexes for `qdrant_only` collections
- Reads corpus type from Qdrant metadata
- Applies appropriate index settings automatically
- Then proceeds with normal parity sync

Note: `meili_only` corpora cannot be auto-created (require `--type` and `--model`).

## Automatic Sync via Git Hook

Without a hook, an indexed repo drifts behind its source tree between manual
syncs. `arc corpus hook install <corpus>` closes that gap.

How it works:

- On each commit the hook asks git which paths changed (`git diff-tree`, not a
  tree walk), queues them, and starts a background worker.
- The worker drains the queue until empty, so a burst of commits pays one
  embedding-model load instead of one per commit.
- Deleted and renamed-away files are removed from both Qdrant and MeiliSearch.
- The hook never blocks or fails a git operation; it logs to
  `~/.local/state/arcaneum/hook.log`.

Run `arc corpus hook install` with **no corpus name** to be walked through it:
it lists existing corpora or offers to create one (inferring the type from the
repo's contents), suggests hook points, and offers the initial backfill. Prefer
this when the user has not said which corpus to use. Pass `--yes` to take the
defaults without prompting.

A hook only indexes future commits, so files already committed need one real
`arc corpus sync <corpus> <repo>` to get in.

Installing is safe and reversible: the hook block is delimited by marker
comments and appended to any existing hook, so a user's own hook keeps working,
installing twice does not duplicate it, and `uninstall` removes only its own
block. `core.hooksPath` is respected, and one repo can feed several corpora.

Use `--hook post-merge` / `post-checkout` / `post-rewrite` so pulls, checkouts,
and rebases stay in sync too. Use `--service` to also register an OS watcher
(launchd on macOS, systemd on Linux) that drains work left over after a reboot
or a failed spawn.

For a one-off equivalent with no hook installed, use
`arc corpus sync <corpus> --changed-since HEAD`.

## Concurrent Syncs

Syncs of one corpus are serialized by a per-corpus write lock, so two runs
cannot duplicate entries or drift the two indexes apart. A second sync waits by
default; `--no-wait` fails immediately instead, and `--lock-timeout <seconds>`
bounds the wait. Different corpora never block each other, and `--dry-run` takes
no lock.

## GPU Acceleration and Apple Silicon

By default, corpus sync uses GPU acceleration (MPS on Apple Silicon, CUDA on NVIDIA).

**Large models on Apple Silicon:** Models like `jina-code-1.5b` (1.5B params) may cause system
instability on Macs with limited memory. If you experience lockups:

```bash
# Use CPU-only mode (slower but stable)
arc corpus sync MyCorpus /path --no-gpu

# Or use the environment variable
ARC_NO_GPU=1 arc corpus sync MyCorpus /path

# Or use a smaller model (bge is 0.3B params)
arc corpus sync MyCorpus /path --models bge
```

**Model sizes:**

- `bge`, `bge-base`, `bge-small`: Safe for all systems
- `stella` (1.5B, deprecated - use `qwen3-embed`): May cause issues on Macs with <16GB RAM
- `nomic-code` (7B): Requires dedicated GPU with significant VRAM

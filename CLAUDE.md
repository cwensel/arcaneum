# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Arcaneum is a CLI tool for semantic and full-text search across Qdrant and MeiliSearch vector databases.
It provides Claude Code integration through slash commands and plugins.

This repository uses Recommendation Decisioning Records (RDRs) for detailed technical planning before implementation.

## Recommendation Decisioning Records (RDRs)

RDRs are specification prompts built through iterative research and refinement. They evolve during
planning and become locked specifications for implementation.

- **Location**: `docs/rdr/`
- **Template**: `docs/rdr/TEMPLATE.md`
- **Complete guide**: `docs/rdr/README.md`

### When to Use RDRs

Create an RDR for complex implementations, architectural decisions, framework workarounds,
or when you want to iterate on an approach before committing to code.

### Quick RDR Workflow

1. **Create** (Draft) — Document problem, constraints, technical environment
2. **Research** (Draft) — Investigate, add findings, label as Verified/Documented/Assumed
3. **Decide** (Draft) — Select approach, document rationale, complete Finalization Gate
4. **Lock** (Final) — All gate items answered; RDR is the spec for implementation
5. **Implement** (Final) — Use locked RDR as spec; do not edit during implementation
6. **Close** — Update status; create post-mortem in `docs/rdr/post-mortem/`

**IMPORTANT**: RDRs are **immutable** once locked (Final). They are specification documents
that show what was planned and why. If implementation reveals the RDR is wrong, abandon
implementation, iterate on the RDR with lessons learned, and start fresh.

See `docs/rdr/README.md` for complete RDR workflow, format details, and usage guidelines.

## Markdown Files

When creating or modifying markdown files in this project:

- **ALWAYS** run `markdownlint <file>` to validate markdown formatting before completing the task
- Fix any linting errors reported by markdownlint
- Follow the project's markdown style guidelines enforced by markdownlint

## Arc CLI Quick Reference

**Prefer `arc corpus` over `arc index` / `arc collection` / `arc indexes`.**
The `corpus` commands manage Qdrant and MeiliSearch together. The single-system
commands exist for advanced workflows that need only one system; do not reach
for them for general indexing or searching.

### Adding or updating content

```bash
# Create a corpus once (creates both Qdrant collection and MeiliSearch index)
arc corpus create MyCorpus --type pdf        # or: code, markdown

# Add or update files (preferred command for indexing)
arc corpus sync MyCorpus /path/to/files

# Same, but also detect renames, remove indexed entries for files that
# no longer exist on disk, and check Qdrant/MeiliSearch cross-system
# parity (all modes pick up edited files via mtime+size)
arc corpus sync MyCorpus /path/to/files --parity

# Preview parity changes without writing
arc corpus sync MyCorpus /path/to/files --parity --dry-run
```

### Searching

```bash
# Semantic search (most common)
arc search semantic "your query" --corpus CorpusName

# Multi-corpus search
arc search semantic "your query" --corpus Corp1 --corpus Corp2

# Full-text search
arc search text "your query" --corpus CorpusName

# List available corpora
arc corpus list
```

**IMPORTANT:** The subcommand (`semantic` or `text`) must come BEFORE the query.
Do NOT use `arc search --corpus` without a subcommand - that syntax is incorrect.

## Releasing

Version is tracked in four files. Use the bump script to update all of them:

```bash
./scripts/bump-version.sh 0.3.0
```

This updates: `pyproject.toml`, `src/arcaneum/__init__.py`,
`.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`, and `README.md`.

**Release workflow** (requires developer confirmation at each step):

```bash
./scripts/bump-version.sh <version>
git add -A && git commit -m "Release v<version>"
git tag -a v<version> -m "Release v<version>"
git push origin main --tags
```

Pushing the tag triggers `.github/workflows/release.yml`, which builds the wheel/sdist
and creates a GitHub Release with artifacts.

## Source Control

Git operations require developer confirmation before execution:

- **Commits**: Always confirm with the developer before running `git commit`
- **Push**: Always confirm before pushing to remote
- Claude Code may stage files and draft commit messages as part of the workflow

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Arcaneum is a CLI tool for semantic and full-text search across Qdrant and MeiliSearch vector databases.
It provides Claude Code integration through slash commands and plugins.

This repository uses Recommendation Decisioning Records (RDRs) for detailed technical planning before implementation.

**Note**: This project uses [bd (beads)](https://github.com/steveyegge/beads) for issue tracking.
Use `bd` commands instead of markdown TODOs. See @AGENTS.md for workflow details.

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

When asked to search collections, use this exact syntax:

```bash
# Semantic search (most common)
arc search semantic "your query" --corpus CorpusName

# Multi-corpus search
arc search semantic "your query" --corpus Corp1 --corpus Corp2

# Full-text search
arc search text "your query" --corpus CorpusName

# List available collections
arc collection list
```

**IMPORTANT:** The subcommand (`semantic` or `text`) must come BEFORE the query.
Do NOT use `arc search --corpus` without a subcommand - that syntax is incorrect.

## Source Control

Git operations require developer confirmation before execution:

- **Commits**: Always confirm with the developer before running `git commit`
- **Push**: Always confirm before pushing to remote
- Claude Code may stage files and draft commit messages as part of the workflow

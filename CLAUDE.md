# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Arcaneum is a CLI tool for semantic and full-text search across Qdrant and MeiliSearch vector databases. It provides Claude Code integration through slash commands and plugins.

This repository uses Recommendation Data Records (RDRs) for detailed technical planning before implementation.

**Note**: This project uses [bd (beads)](https://github.com/steveyegge/beads) for issue tracking. Use `bd` commands instead of markdown TODOs. See @AGENTS.md for workflow details.

## Recommendation Data Records (RDRs)

RDRs are detailed implementation plans created **before** coding begins. They serve as planning documents, iteration artifacts, and AI collaboration tools.

- **Location**: `docs/rdr/`
- **Template**: `docs/rdr/TEMPLATE.md`
- **Complete guide**: `docs/rdr/README.md`

### When to Use RDRs

Create an RDR for complex implementations, architectural decisions, framework workarounds, or when you want to iterate on an approach before committing to code.

### Quick RDR Workflow

1. Create RDR using template for complex work
2. Iterate and refine based on feedback
3. Implement using RDR as guide
4. Update status and index when complete

**IMPORTANT**: RDRs are **immutable** once marked as implemented. They are historical design documents that show what was planned and why. Do NOT update RDRs to reflect current code state - they represent the design at implementation time.

See `docs/rdr/README.md` for complete RDR workflow, format details, and usage guidelines.

## Markdown Files

When creating or modifying markdown files in this project:

- **ALWAYS** run `markdownlint <file>` to validate markdown formatting before completing the task
- Fix any linting errors reported by markdownlint
- Follow the project's markdown style guidelines enforced by markdownlint

## Source Control

We use git to manage the project source. Claude Code has the following restrictions:

- **NEVER** run `git add` commands to stage files
- **NEVER** run `git commit` commands to create commits
- **NEVER** run `git push` commands
- Claude Code MAY draft commit messages when requested by the user
- All git operations for staging and committing must be performed by the user manually

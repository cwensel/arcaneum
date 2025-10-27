# CLAUDE.md

**Note**: This project uses [bd (beads)](https://github.com/steveyegge/beads) for issue tracking. Use `bd` commands instead of markdown TODOs. See AGENTS.md for workflow details.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Arcaneum is a Claude Code marketplace for skills and mcp servers to manage search across vector and full text databases. 

This repository uses Recommendation Data Records (RDRs) for detailed technical planning before implementation.

## Issue Tracking with Beads

This repository uses **Beads (bd)** for issue tracking instead of Markdown files.

- Database: `.beads/arcaneum.db`
- Issue prefix: `arcaneum` (issues named `arcaneum-1`, `arcaneum-2`, etc.)

### Common Beads Commands

```bash
bd stats              # Project statistics
bd list              # List all issues
bd ready             # Find ready-to-work tasks (no blockers)
bd show <issue-id>   # Show issue details
bd create            # Create new issue
bd update            # Update issue status/priority
bd close <issue-id>  # Mark issue as completed
```

**Note**: Always set context before write operations: `bd set_context <workspace_root>`

See resource `beads://quickstart` for detailed workflow guidance.

## Recommendation Data Records (RDRs)

RDRs are detailed implementation plans created **before** coding begins. They serve as planning documents, iteration artifacts, and AI collaboration tools.

- **Location**: `doc/rdr/`
- **Template**: `doc/rdr/TEMPLATE.md`
- **Complete guide**: `doc/rdr/README.md`

### When to Use RDRs

Create an RDR for complex implementations, architectural decisions, framework workarounds, or when you want to iterate on an approach before committing to code.

### Quick RDR Workflow

1. Create RDR using template for complex work
2. Iterate and refine based on feedback
3. Implement using RDR as guide
4. Update status and index when complete

See `doc/rdr/README.md` for complete RDR workflow, format details, and usage guidelines.

## Development Workflow

1. Use `bd ready` to find available tasks
2. Create RDRs for complex implementations (see `doc/rdr/README.md`)
3. Update Beads issues as work progresses
4. Preserve knowledge in RDRs for future reference

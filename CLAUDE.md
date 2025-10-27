# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Arcaneum is a Claude Code marketplace for skills and mcp servers to manage search across vector and full text databases.

This repository uses Recommendation Data Records (RDRs) for detailed technical planning before implementation.

**Note**: This project uses [bd (beads)](https://github.com/steveyegge/beads) for issue tracking. Use `bd` commands instead of markdown TODOs. See @AGENTS.md for workflow details.

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

## Source Control

We use git to manage the project source, but Claude may never create commits, but may create commit messages.

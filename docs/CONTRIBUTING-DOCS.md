# Documentation Style Guide

This guide ensures consistency across all Arcaneum documentation.

## File Naming Conventions

### Use Lowercase with Hyphens

**✅ Correct:**

- `quickstart.md`
- `offline-mode.md`
- `cli-reference.md`
- `source-code-indexing.md`

**❌ Incorrect:**

- `QUICKSTART.md` (breaks on Linux case-sensitive filesystems)
- `offline_mode.md` (use hyphens, not underscores)
- `CLIReference.md` (avoid camelCase)

**Rationale:** Lowercase with hyphens works on all operating systems (case-sensitive Linux, case-insensitive macOS/Windows).

### RDR Files

RDR files use uppercase prefix for easy identification:

- `RDR-001-project-structure.md`
- `RDR-002-qdrant-server-setup.md`

Pattern: `RDR-{number}-{kebab-case-title}.md`

## Command Invocation

### User-Facing Documentation

Use the simplest command format:

**✅ Correct:**

```bash
arc container start
arc collection create MyCode --model jina-code
arc search "query" --collection MyCode
```

**❌ Incorrect:**

```bash
python -m arcaneum.cli.main container start  # Too verbose
bin/arc container start  # Only for dev docs
./arc container start  # Confusing path
```

### Developer/Technical Documentation

When showing module structure or implementation details:

```bash
# OK in technical docs/RDRs
python -m arcaneum.cli.main doctor
python -m arcaneum.cli.index_pdfs
```

When showing development mode:

```bash
# OK when explicitly documenting dev setup
bin/arc --help
chmod +x bin/arc
```

### Slash Commands (commands/*.md)

Always use simple `arc` command format:

```bash
cd ${CLAUDE_PLUGIN_ROOT}
arc <command> $ARGUMENTS
```

## Link Conventions

### Internal Links

Use relative links and lowercase filenames:

**✅ Correct:**

```markdown
See [quickstart.md](guides/quickstart.md)
See [offline mode guide](../testing/offline-mode.md)
See [RDR process](docs/rdr/README.md)
```

**❌ Incorrect:**

```markdown
See [QUICKSTART.md](guides/QUICKSTART.md)  # Case mismatch
See offline-mode.md  # Not a link
See /docs/rdr/README.md  # Absolute path
```

### External Links

Use full URLs for external resources:

```markdown
[Install Docker](https://docs.docker.com/get-docker/)
[Qdrant Docs](https://qdrant.tech/documentation/)
```

## Code Example Conventions

### Show Simplest Working Example First

**✅ Correct:**

```markdown
## Quick Start

\`\`\`bash
arc container start
arc collection create MyCode --model jina-code --type code
arc index source ~/project --collection MyCode
\`\`\`

## Advanced Options

\`\`\`bash
arc collection create MyCode --model jina-code --type code --hnsw-m 32 --on-disk
\`\`\`
```

**❌ Incorrect:**

```markdown
## Quick Start

\`\`\`bash
# Start with all the complex options
arc collection create MyCode --model jina-code --type code --hnsw-m 32 --hnsw-ef 200 --on-disk
\`\`\`
```

### Use Consistent Command Names

- ✅ `arc container start` (current)
- ❌ `arc docker start` (old, renamed)
- ❌ `docker compose -f deploy/docker-compose.yml up -d` (too complex for user docs)

### Include Expected Output

Help users know what success looks like:

```markdown
\`\`\`bash
arc container start
\`\`\`

Expected output:
\`\`\`
[INFO] Starting container services...
Qdrant started successfully
  REST API: http://localhost:6333
\`\`\`
```

## Documentation Structure

### User Documentation (Humans Getting Started)

Located in:

- `README.md` - Main entry point, 5-minute quick start
- `docs/guides/` - Comprehensive guides
- `docs/testing/` - Testing and troubleshooting

**Purpose:** Help users get started and accomplish tasks.

**Tone:** Clear, concise, example-driven

**Command format:** Use `arc` (simplest)

### Contributor Documentation (Humans & Agents Contributing)

Located in:

- `CLAUDE.md` - AI agent instructions
- `AGENTS.md` - AI workflow with beads
- `CONTRIBUTING.md` - Human contributor guide
- `docs/rdr/` - Technical design documents
- `docs/reference/` - Technical reference

**Purpose:** Guide contributors through development workflow.

**Tone:** Technical, detailed, precise

**Command format:** Can use `python -m arcaneum.cli.main` when showing implementation

### Claude Code Integration

Located in:

- `commands/*.md` - Slash command definitions

**Purpose:** Enable Claude Code slash commands.

**Format:** Follow template with description, arguments, examples, execution

**Command format:** Use `arc` in execution section

## Documentation Categories

### When to Update Which Docs

**User makes a feature request:**

1. Create RDR (docs/rdr/) for complex features
2. Implement feature
3. Add to CLI reference (docs/guides/cli-reference.md)
4. Add slash command (commands/)
5. Update quickstart if it's a common workflow

**Bug fix:**

1. Create beads issue
2. Fix bug
3. Update troubleshooting section if relevant

**New CLI command added:**

1. Add to CLI reference (docs/guides/cli-reference.md)
2. Create slash command (commands/)
3. Update README.md if it's a major feature

## Common Pitfalls

### ❌ Don't

- Use uppercase in filenames (breaks on Linux)
- Reference files with wrong case (QUICKSTART.md vs quickstart.md)
- Use `python -m arcaneum.cli.main` in user-facing docs
- Use `arc docker` (renamed to `arc container`)
- Create documentation without examples
- Show complex examples before simple ones

### ✅ Do

- Use lowercase filenames with hyphens
- Test links work (especially on case-sensitive systems)
- Use `arc` in all user-facing examples
- Include expected output
- Show simple examples first, then advanced
- Keep quickstart truly quick (5 minutes max)

## Testing Documentation Changes

Before committing documentation changes:

```bash
# 1. Check all links use lowercase
grep -r "QUICKSTART\|OFFLINE-MODE\|TEST-COMMANDS" docs/

# 2. Check for old command references
grep -r "arc docker\|python -m arcaneum.cli.main" docs/ commands/

# 3. Verify files exist
ls -la docs/guides/quickstart.md
ls -la docs/testing/offline-mode.md

# 4. Test examples actually work
arc container start
arc collection list
```

## Documentation Checklist

When adding/updating documentation:

- [ ] Filenames are lowercase with hyphens
- [ ] Links use correct case and relative paths
- [ ] Examples use `arc` not `python -m arcaneum.cli.main`
- [ ] Examples use `arc container` not `arc docker`
- [ ] Simple examples shown before advanced
- [ ] Expected output included for key commands
- [ ] Links tested (click through to verify)
- [ ] Spell check and grammar check completed
- [ ] Consistent with other docs in same category

## Questions?

See:

- [CONTRIBUTING.md](../CONTRIBUTING.md) - General contribution guide
- [AGENTS.md](../AGENTS.md) - AI agent workflow
- [RDR Process](rdr/README.md) - Technical documentation process

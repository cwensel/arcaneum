# Contributing to Arcaneum

Thank you for your interest in contributing to Arcaneum! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git
- Docker (for running Qdrant and MeiliSearch)
- Basic understanding of vector databases and embeddings (helpful but not required)

### Development Setup

1. **Fork and Clone**

```bash
git clone https://github.com/cwensel/arcaneum
cd arcaneum
```

2. **Install in Development Mode**

```bash
pip install -e .
```

This installs Arcaneum in editable mode, so code changes take effect immediately.

3. **Start Services**

```bash
arc container start
```

4. **Verify Installation**

```bash
arc doctor
arc --help
```

5. **Run Tests**

```bash
pytest tests/
```

## Development Workflow

### Using Beads for Task Tracking

This project uses [beads](https://github.com/steveyegge/beads) for issue tracking.

**Check for ready work:**

```bash
bd ready
```

**Create a new task:**

```bash
bd create "Add feature X" --type feature --priority 2
```

**Claim a task:**

```bash
bd update arcaneum-123 --status in_progress --assignee YourName
```

**Complete a task:**

```bash
bd close arcaneum-123 --reason "Completed"
```

See **[AGENTS.md](AGENTS.md)** for complete beads workflow and AI agent guidelines.

### Using Recommendation Data Records (RDRs)

For complex features or architectural changes, create an RDR first:

1. **Copy the template:**

```bash
cp docs/rdr/TEMPLATE.md docs/rdr/RDR-0XX-your-feature.md
```

2. **Fill in the RDR sections:**
   - Problem statement
   - Requirements
   - Design decisions
   - Implementation plan
   - Testing strategy

3. **Iterate on the RDR** before implementing

4. **Implement following the RDR**

5. **Update RDR status** when complete

See **[docs/rdr/README.md](docs/rdr/README.md)** for detailed RDR process.

### When to Create an RDR

**Create an RDR for:**

- New major features (indexing, search, integrations)
- Architectural changes
- Complex algorithms or workflows
- Framework workarounds or tricky implementations

**Skip RDR for:**

- Bug fixes
- Documentation updates
- Simple refactoring
- Minor improvements

## Code Guidelines

### Python Style

- Follow PEP 8
- Use type hints where helpful
- Write docstrings for public functions
- Keep functions focused and testable

### CLI Design Principles

1. **CLI-First**: All functionality as CLI commands
2. **JSON Output**: Support `--json` flag for scripting
3. **Structured Errors**: Use exit codes and clear error messages
4. **Examples in Help**: Include usage examples in command help
5. **Slash Commands**: Every CLI command should have a corresponding slash command

### File Organization

```text
src/arcaneum/
├── cli/           # CLI command implementations
├── embeddings/    # Embedding generation
├── collections/   # Collection management
├── indexing/      # Indexing pipelines
├── search/        # Search logic
├── fulltext/      # MeiliSearch integration
└── schema/        # Shared schemas
```

### Testing

- Write tests for new functionality
- Aim for >80% code coverage
- Include both unit and integration tests
- Test edge cases and error conditions

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=arcaneum tests/

# Run specific test file
pytest tests/test_collections.py

# Run with verbose output
pytest -v tests/
```

## Documentation Guidelines

### When to Update Documentation

**For new CLI commands:**

1. Add to `docs/guides/cli-reference.md`
2. Create slash command in `commands/`
3. Update README.md if it's a major feature
4. Add examples to relevant guides

**For bug fixes:**

- Update troubleshooting sections if relevant
- Add to known issues if temporary workaround

**For features:**

- Update quickstart if it changes the basic workflow
- Add detailed guide if complex
- Update RDR status

### Documentation Style

Follow the **[Documentation Style Guide](docs/CONTRIBUTING-DOCS.md)** for:

- File naming conventions (lowercase with hyphens)
- Command invocation format (`arc` vs `python -m arcaneum.cli.main`)
- Link conventions (relative, lowercase)
- Code example best practices

### Testing Documentation

Before submitting documentation changes:

```bash
# Check for case-sensitivity issues
grep -r "QUICKSTART\|OFFLINE-MODE" docs/

# Check for old commands
grep -r "arc docker\|python -m arcaneum.cli.main" docs/ commands/

# Verify links work
# Click through all links in your changes
```

## Pull Request Process

### Before Submitting

1. **Create beads issue** for your work (if not already exists)
2. **Write/update tests** for your changes
3. **Update documentation** as needed
4. **Run tests** and verify they pass
5. **Check code style** (PEP 8)
6. **Update RDR** if implementing from an RDR

### PR Guidelines

**Title:** Use clear, descriptive titles

- ✅ "Add arc config commands for cache management"
- ❌ "Fix stuff"

**Description:** Include:

- Link to beads issue: "Resolves arcaneum-123"
- Brief summary of changes
- Testing performed
- Documentation updated (if applicable)

**Commits:**

- Write clear commit messages
- Reference beads issues: "Implement cache commands (arcaneum-162)"
- Keep commits focused (one logical change per commit)

### Review Process

1. Maintainers will review your PR
2. Address feedback and comments
3. Tests must pass
4. Documentation must be updated
5. Once approved, PR will be merged

## Branch Naming

Use descriptive branch names:

- `feature/cache-management` - New features
- `fix/case-sensitivity-docs` - Bug fixes
- `docs/update-quickstart` - Documentation updates
- `refactor/cli-structure` - Refactoring

## Common Tasks

### Adding a New CLI Command

1. **Create beads issue:**

```bash
bd create "Add arc foo command" --type feature --priority 2
```

2. **Create RDR** (if complex)

3. **Implement in src/arcaneum/cli/:**

```python
# Create src/arcaneum/cli/foo.py
@click.group(name='foo')
def foo_group():
    """Foo management"""
    pass
```

4. **Register in main.py:**

```python
from arcaneum.cli.foo import foo_group
cli.add_command(foo_group, name='foo')
```

5. **Add to CLI reference:**

```markdown
## Foo Management

\`\`\`bash
arc foo bar --option value
\`\`\`
```

6. **Create slash command:**

```markdown
---
description: Foo management
---

Runs: arc foo $ARGUMENTS
```

7. **Write tests:**

```python
# tests/test_foo.py
def test_foo_command():
    assert foo() == expected
```

8. **Close beads issue:**

```bash
bd close arcaneum-XXX --reason "Completed"
```

### Updating Documentation

1. **Read the style guide:** [docs/CONTRIBUTING-DOCS.md](docs/CONTRIBUTING-DOCS.md)
2. **Make changes** following conventions
3. **Test links** by clicking through
4. **Verify examples** actually work
5. **Commit with changes**

### Reporting Issues

Use beads for issue tracking:

```bash
# Report a bug
bd create "Bug: arc search fails with filter" --type bug --priority 1

# Suggest a feature
bd create "Feature: add arc backup command" --type feature --priority 3

# Document a task
bd create "Update RDR-005 with branch support" --type task --priority 2
```

Or use GitHub issues if you prefer (maintainers will create beads issues).

## Questions?

- **Setup issues?** Run `arc doctor` for diagnostics
- **Beads questions?** See [AGENTS.md](AGENTS.md)
- **RDR questions?** See [docs/rdr/README.md](docs/rdr/README.md)
- **Documentation questions?** See [docs/CONTRIBUTING-DOCS.md](docs/CONTRIBUTING-DOCS.md)
- **Still stuck?** Open a GitHub issue or discussion

## Code of Conduct

- Be respectful and constructive
- Help others learn
- Focus on the code, not the person
- Keep discussions professional

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

All contributors will be recognized in the project. Thank you for making Arcaneum better!

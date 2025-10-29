# Claude Code Plugin Marketplace: Local Testing Guide

This guide explains how to safely test the Arcaneum plugin marketplace in a local Claude Code installation before publishing to GitHub.

## Prerequisites

- Claude Code installed and running
- Local clone of the arcaneum repository
- Python 3.12+ installed
- Docker running (for Qdrant/MeiliSearch functionality)

## Directory Structure Compliance

Our plugin structure meets Claude Code best practices:

```
arcaneum/                           # Plugin root
├── .claude-plugin/                 # Plugin metadata (required)
│   ├── plugin.json                 # Plugin manifest ✓
│   └── marketplace.json            # Marketplace catalog ✓
├── commands/                       # Slash commands at root ✓
│   ├── create-collection.md
│   ├── index-pdfs.md
│   ├── index-source.md
│   ├── list-collections.md
│   ├── search.md
│   ├── search-text.md
│   ├── create-corpus.md
│   └── sync-directory.md
├── src/arcaneum/                   # Python implementation
│   ├── cli/                        # CLI commands
│   ├── indexing/                   # Indexing pipelines
│   ├── search/                     # Search logic
│   └── ...
├── docs/                           # Documentation
├── README.md
└── LICENSE
```

**Key Compliance Points:**
- ✅ `commands/` at root level (not nested in `.claude-plugin/`)
- ✅ All paths in `plugin.json` use relative `./` prefix
- ✅ `plugin.json` contains required fields (name, version, description)
- ✅ `marketplace.json` contains owner and plugins array
- ✅ Uses `${CLAUDE_PLUGIN_ROOT}` in slash commands

## Local Testing Workflow

### Option 1: Direct Local Path (Recommended for Development)

This is the safest method for active development and testing.

#### Step 1: Add Local Marketplace

From within Claude Code:

```
/plugin marketplace add /Users/chris.wensel/sandbox/arcaneum
```

**What This Does:**
- Registers the local directory as a marketplace
- Reads `.claude-plugin/marketplace.json`
- Makes plugins discoverable via `/plugin` menu
- Changes to files are reflected immediately (no reinstall needed for code changes)

#### Step 2: Install the Plugin

```
/plugin install arc@arcaneum-marketplace
```

**Verification:**
- Commands appear in `/help` or `/commands`
- Test a simple command: `/list-collections`

#### Step 3: Test Slash Commands

Test each command systematically:

```
# Collection management
/create-collection test-local --model stella --type code
/list-collections --verbose
/list-collections --json
/delete-collection test-local --confirm

# Model management
/list-models
/list-models --json

# Indexing (if you have test data)
/index-source /path/to/test/repo --collection test-code
/index-pdfs /path/to/test/pdfs --collection test-docs

# Search
/search "test query" --collection test-code --limit 5
/search "test query" --collection test-code --json
```

#### Step 4: Test Error Handling

Verify error codes and messages:

```
# Invalid model (should show error with available models)
/create-collection test --model invalid-model

# Nonexistent collection (should show error)
/search "query" --collection nonexistent

# Missing required argument (Click should show usage)
/create-collection test
```

#### Step 5: Test $ARGUMENTS Expansion

Verify arguments are passed correctly:

```
# Multiple flags
/index-source ~/code --collection MyCode --depth 1 --workers 8 --verbose

# JSON output
/list-collections --json

# Complex filter
/search "auth" --collection Code --filter "language=python" --limit 20
```

### Option 2: Git-Based Local Testing

If you want to simulate the GitHub installation flow:

#### Step 1: Create a Local Git Remote

```bash
# In arcaneum directory
git remote add local-test file:///Users/chris.wensel/sandbox/arcaneum
```

#### Step 2: Add as Git Marketplace

```
/plugin marketplace add file:///Users/chris.wensel/sandbox/arcaneum
```

This simulates how Claude Code would fetch from GitHub but uses your local repository.

### Option 3: Symlink Testing (Advanced)

Create a symlink in Claude's plugin directory:

```bash
# Find Claude's plugin directory
ls ~/.claude/plugins/marketplaces/

# Create symlink
ln -s /Users/chris.wensel/sandbox/arcaneum ~/.claude/plugins/marketplaces/arcaneum-dev

# In Claude Code
/plugin install arcaneum@arcaneum-dev
```

**Warning:** This bypasses Claude's normal installation flow. Use only for advanced debugging.

## Validation Checklist

Before each test session, verify:

### Plugin Manifest (`.claude-plugin/plugin.json`)

- [ ] Valid JSON syntax (no trailing commas, proper quotes)
- [ ] Required fields present: `name`, `version`, `description`
- [ ] All command paths use `./` prefix
- [ ] Commands array lists all 8 slash commands
- [ ] Version follows semantic versioning (e.g., `0.1.0`)

**Validation Command:**
```bash
# Validate JSON syntax
python -m json.tool .claude-plugin/plugin.json > /dev/null && echo "✓ Valid JSON"

# Check required fields
python -c "import json; d=json.load(open('.claude-plugin/plugin.json')); assert all(k in d for k in ['name','version','description']); print('✓ Required fields present')"
```

### Marketplace Manifest (`.claude-plugin/marketplace.json`)

- [ ] Valid JSON syntax
- [ ] `owner` object with `name` field
- [ ] `plugins` array with at least one entry
- [ ] Plugin `source` set to `"./"` (current directory)

**Validation Command:**
```bash
python -m json.tool .claude-plugin/marketplace.json > /dev/null && echo "✓ Valid JSON"
```

### Slash Commands

- [ ] All 8 files exist in `commands/` directory
- [ ] Each has YAML frontmatter with `description` and `argument-hint`
- [ ] Each uses `${CLAUDE_PLUGIN_ROOT}` in execution block
- [ ] Each uses `$ARGUMENTS` for parameter passing
- [ ] Examples are clear and runnable

**Validation Command:**
```bash
# Check all commands have frontmatter
for f in commands/*.md; do
  if ! head -3 "$f" | grep -q "^description:"; then
    echo "❌ Missing frontmatter in $f"
  fi
done
echo "✓ All commands have frontmatter"

# Check all use CLAUDE_PLUGIN_ROOT
if [ $(grep -l "CLAUDE_PLUGIN_ROOT" commands/*.md | wc -l) -eq $(ls commands/*.md | wc -l) ]; then
  echo "✓ All commands use CLAUDE_PLUGIN_ROOT"
fi
```

### Python CLI Verification

- [ ] CLI executes without errors: `python -m arcaneum.cli.main --help`
- [ ] Version check passes: `python --version` (should be 3.12+)
- [ ] Dependencies installed: `pip list | grep -E '(qdrant|fastembed|click|rich)'`
- [ ] Test collection creation works locally

**Validation Command:**
```bash
# Test CLI loads
python -m arcaneum.cli.main --help > /dev/null && echo "✓ CLI loads successfully"

# Test version
python -m arcaneum.cli.main --version && echo "✓ Version command works"
```

## Testing Scenarios

### Scenario 1: Fresh Installation

**Goal:** Verify a new user can install and use the plugin.

**Steps:**
1. Remove any existing marketplace: `/plugin marketplace remove arcaneum-marketplace`
2. Remove plugin if installed: `/plugin uninstall arc`
3. Add local marketplace: `/plugin marketplace add /path/to/arcaneum`
4. Install plugin: `/plugin install arc@arcaneum-marketplace`
5. Verify: `/help` shows arcaneum commands

**Expected Result:**
- All 8 slash commands appear
- Commands execute without errors
- Help text is clear and accurate

### Scenario 2: Update Testing

**Goal:** Verify plugin updates work correctly.

**Steps:**
1. Make a change to a command (e.g., add example to `search.md`)
2. In Claude Code: `/plugin marketplace update arcaneum-marketplace`
3. Test the updated command
4. Verify changes are reflected

**Expected Result:**
- Command updates appear without reinstall
- No stale cache issues

### Scenario 3: JSON Output Validation

**Goal:** Verify all commands support `--json` flag correctly.

**Test Commands:**
```
/list-collections --json
/list-models --json
/create-collection test-json --model stella --type code --json
/collection-info test-json --json
/delete-collection test-json --confirm --json
```

**Expected Result:**
- All output is valid JSON
- JSON follows standard format: `{status, message, data, errors}`
- Exit codes are correct (0 for success, 2 for invalid args, etc.)

### Scenario 4: Error Handling

**Goal:** Verify errors are user-friendly and actionable.

**Test Commands:**
```
# Invalid model
/create-collection test --model bad-model

# Nonexistent collection
/search "query" --collection nonexistent

# Missing required arg
/index-source
```

**Expected Result:**
- All errors have `[ERROR]` prefix
- Error messages are descriptive
- Suggested fixes are provided where applicable
- Exit codes are appropriate (2 for invalid args, 3 for not found)

### Scenario 5: Concurrent Operations

**Goal:** Verify multiple indexing operations can run simultaneously.

**Steps:**
1. Open two Claude Code sessions
2. In session 1: `/index-source /path/one --collection code`
3. In session 2: `/index-pdfs /path/two --collection docs`
4. Monitor both complete successfully

**Expected Result:**
- No resource conflicts
- Both operations complete successfully
- Progress tracking works in both sessions

### Scenario 6: Long-Running Operations

**Goal:** Verify progress tracking for indexing operations.

**Steps:**
1. Index a moderately large directory: `/index-source ~/sandbox/thirdparty --collection test`
2. Monitor Claude Code output

**Expected Result:**
- Progress messages use `[INFO]` prefix
- Percentage updates: `[INFO] Processing X/Y (Z%)`
- Final summary: `[INFO] Complete: X projects, Y files, Z chunks`
- Claude parses and displays progress correctly

## Troubleshooting

### Issue: Plugin Not Loading

**Symptoms:**
- Plugin doesn't appear in `/plugin` menu
- Commands don't show in `/help`

**Debug Steps:**
1. Check JSON syntax: `python -m json.tool .claude-plugin/plugin.json`
2. Verify marketplace path is correct
3. Check Claude Code logs (if available)
4. Try removing and re-adding marketplace

**Common Causes:**
- Invalid JSON syntax (trailing commas, unquoted keys)
- Wrong directory path
- Missing required fields in plugin.json

### Issue: Commands Not Found

**Symptoms:**
- `/help` doesn't show arcaneum commands
- Executing `/search` gives "command not found"

**Debug Steps:**
1. Verify `commands/` directory exists at plugin root
2. Check all .md files have proper frontmatter
3. Verify plugin is installed: `/plugin` and check enabled plugins
4. Try reinstalling: `/plugin uninstall arc` then `/plugin install arc@arcaneum-marketplace`

**Common Causes:**
- `commands/` in wrong location (must be at root)
- Missing frontmatter in .md files
- Plugin not actually installed (just marketplace added)

### Issue: Python Import Errors

**Symptoms:**
- Commands fail with `ModuleNotFoundError`
- "No module named 'arcaneum'" errors

**Debug Steps:**
1. Verify Python can find arcaneum: `python -c "import arcaneum; print(arcaneum.__version__)"`
2. Check you're in correct directory when testing
3. Verify dependencies installed: `pip list | grep arcaneum`
4. Test CLI directly: `python -m arcaneum.cli.main --help`

**Common Causes:**
- Not running from plugin root directory
- Virtual environment not activated
- Dependencies not installed
- PYTHONPATH issues

### Issue: $ARGUMENTS Not Expanding

**Symptoms:**
- Literal `$ARGUMENTS` appears in error messages
- Arguments not passed to CLI

**Debug Steps:**
1. Check command markdown has proper execution block
2. Verify `$ARGUMENTS` is not escaped or quoted
3. Test command with simple args first

**Common Causes:**
- Incorrect execution block format
- Shell quoting issues in .md file
- Missing execution block entirely

## Safe Testing Practices

### 1. Use Test Collections

Always test with disposable collections:

```bash
# Create test collection
arc create-collection test-dev --model stella --type code

# Test indexing with small dataset
arc index-source ~/small-test-repo --collection test-dev

# Clean up when done
arc delete-collection test-dev --confirm
```

### 2. Use Separate Qdrant Instance (Optional)

For complete isolation:

```bash
# Start test Qdrant on different port
docker run -d --name qdrant-test -p 6334:6333 qdrant/qdrant:v1.15.4

# Update slash commands to use test port temporarily
# Or use --qdrant-url flag (if implemented)
```

### 3. Version Your Test Changes

Create a feature branch for plugin changes:

```bash
git checkout -b plugin/test-slash-commands
# Make changes
# Test in Claude Code
# Commit if successful
git commit -m "Update slash command examples"
```

### 4. Document Test Results

Keep a testing log:

```markdown
## Test Session: 2025-10-28

- ✅ create-collection: Works with --json
- ✅ list-collections: Table format looks good
- ❌ search: Filter syntax unclear in help
  - Fixed: Updated search.md with filter examples
- ✅ index-source: Progress tracking works perfectly
```

## Automated Testing

### Validate Plugin Structure

Create a validation script `scripts/validate-plugin.sh`:

```bash
#!/bin/bash
set -e

echo "Validating Arcaneum plugin structure..."

# Check JSON files
echo "✓ Validating JSON syntax..."
python -m json.tool .claude-plugin/plugin.json > /dev/null
python -m json.tool .claude-plugin/marketplace.json > /dev/null

# Check required fields
echo "✓ Checking required fields..."
python -c "
import json
p = json.load(open('.claude-plugin/plugin.json'))
assert 'name' in p and 'version' in p and 'description' in p
m = json.load(open('.claude-plugin/marketplace.json'))
assert 'owner' in m and 'plugins' in m
"

# Check commands directory
echo "✓ Verifying commands directory..."
test -d commands || (echo "❌ commands/ directory missing" && exit 1)

# Check all commands have frontmatter
echo "✓ Checking command frontmatter..."
for f in commands/*.md; do
    head -3 "$f" | grep -q "^description:" || (echo "❌ Missing frontmatter in $f" && exit 1)
done

# Check CLAUDE_PLUGIN_ROOT usage
echo "✓ Verifying CLAUDE_PLUGIN_ROOT usage..."
count=$(grep -l "CLAUDE_PLUGIN_ROOT" commands/*.md | wc -l | tr -d ' ')
total=$(ls commands/*.md | wc -l | tr -d ' ')
test "$count" -eq "$total" || (echo "❌ Not all commands use CLAUDE_PLUGIN_ROOT" && exit 1)

# Test CLI loads
echo "✓ Testing CLI loads..."
python -m arcaneum.cli.main --help > /dev/null

echo ""
echo "✅ All validation checks passed!"
echo ""
echo "Ready for local testing in Claude Code:"
echo "  /plugin marketplace add $(pwd)"
echo "  /plugin install arc@arcaneum-marketplace"
```

Make it executable:
```bash
chmod +x scripts/validate-plugin.sh
./scripts/validate-plugin.sh
```

### Test Script for All Commands

Create `scripts/test-plugin-commands.sh`:

```bash
#!/bin/bash

echo "Testing all Arcaneum CLI commands..."
echo ""

# Test each command with --help
for cmd in create-collection list-collections collection-info delete-collection \
           list-models index-pdfs index-source search search-text \
           create-corpus sync-directory; do
    echo "Testing: arc $cmd --help"
    python -m arcaneum.cli.main "$cmd" --help > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "  ✓ $cmd"
    else
        echo "  ❌ $cmd failed"
    fi
done

echo ""
echo "Testing JSON output..."

# Test JSON output for commands that support it
python -m arcaneum.cli.main list-models --json > /dev/null 2>&1 && echo "  ✓ list-models --json"
python -m arcaneum.cli.main list-collections --json > /dev/null 2>&1 && echo "  ✓ list-collections --json"

echo ""
echo "Testing error handling..."

# Test invalid arguments (should exit with code 2)
python -m arcaneum.cli.main create-collection test --model invalid 2>&1 | grep -q "\[ERROR\]"
if [ $? -eq 0 ]; then
    echo "  ✓ Error messages use [ERROR] prefix"
fi

echo ""
echo "✅ Command testing complete!"
```

### Claude Code Integration Tests

Create `scripts/test-claude-integration.sh` to validate Claude-parseable output:

```bash
chmod +x scripts/test-claude-integration.sh
./scripts/test-claude-integration.sh
```

This script validates:
- JSON output is valid for all commands
- `$ARGUMENTS` expansion works correctly
- Progress messages follow `[INFO]` format
- Error messages use `[ERROR]` prefix
- Exit codes match documentation
- Plugin manifest structure
- Command file format (frontmatter, execution blocks)
- Output encoding (UTF-8 compliance)

**When to run:**
- Before committing changes
- After modifying command files
- Before creating a release
- When testing Claude Code integration

## Integration Testing Matrix

| Test Case | Command | Expected Result | Exit Code |
|-----------|---------|----------------|-----------|
| **Setup verification** | `/doctor --json` | Valid JSON with all checks | 0 or 1 |
| **Setup verbose** | `/doctor --verbose` | Table with detailed diagnostics | 0 or 1 |
| **Valid collection creation** | `/create-collection test --model stella --type code` | Collection created, JSON if --json | 0 |
| **Invalid model** | `/create-collection test --model bad` | Error with available models list | 2 |
| **List collections** | `/list-collections --json` | Valid JSON output | 0 |
| **Collection info** | `/collection-info nonexistent` | Collection not found error | 1 or 3 |
| **Delete with confirm** | `/delete-collection test --confirm --json` | Success JSON response | 0 |
| **List models** | `/list-models --json` | Valid JSON with all models | 0 |
| **Search valid** | `/search "test" --collection code --limit 5` | Results or empty results | 0 |
| **Search invalid collection** | `/search "test" --collection bad` | Collection not found | 3 |
| **Index source (small)** | `/index-source . --collection test --depth 0` | Progress messages with [INFO] | 0 |

## Common Testing Mistakes to Avoid

### ❌ Testing in Production Claude Code Instance

**Problem:** Changes affect your real Claude Code setup.

**Solution:** Use a separate Claude Code profile or test in a VM/container.

### ❌ Forgetting to Update marketplace.json Version

**Problem:** Claude Code may cache old version.

**Solution:**
- Update version in both files when making changes
- Use `/plugin marketplace update arcaneum-marketplace` to refresh

### ❌ Not Testing Clean Install

**Problem:** Your environment has cached state that new users won't have.

**Solution:**
1. Uninstall plugin completely
2. Remove marketplace
3. Clear any test collections
4. Reinstall fresh and test

### ❌ Hardcoding Paths

**Problem:** Commands fail when installed to different location.

**Solution:**
- Always use `${CLAUDE_PLUGIN_ROOT}` for plugin-relative paths
- Use `$ARGUMENTS` for user-provided paths

### ❌ Not Testing All Argument Combinations

**Problem:** Edge cases fail in production.

**Solution:**
- Test with no optional flags
- Test with all optional flags
- Test with conflicting flags
- Test with invalid values

## Pre-Release Checklist

Before publishing to GitHub:

### Code Quality
- [ ] All validation scripts pass
- [ ] Manual testing complete for all commands
- [ ] Error messages are clear and actionable
- [ ] JSON output validated for all commands
- [ ] Exit codes are correct

### Documentation
- [ ] README.md has installation instructions
- [ ] All commands documented with examples
- [ ] Troubleshooting guide is comprehensive
- [ ] CHANGELOG.md updated with changes

### Compliance
- [ ] plugin.json has correct version
- [ ] marketplace.json matches plugin version
- [ ] LICENSE file present
- [ ] No sensitive data in repository (API keys, credentials)

### Testing
- [ ] Tested fresh install workflow
- [ ] Tested update workflow
- [ ] Tested uninstall workflow
- [ ] Tested error scenarios
- [ ] Tested on clean Claude Code instance

### Performance
- [ ] Commands respond in < 1 second (except long-running indexing)
- [ ] Progress updates appear regularly during indexing
- [ ] No memory leaks in long-running operations

## Continuous Integration for Plugin

### GitHub Actions Workflow (Future)

Create `.github/workflows/validate-plugin.yml`:

```yaml
name: Validate Claude Code Plugin

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Validate JSON manifests
        run: |
          python -m json.tool .claude-plugin/plugin.json
          python -m json.tool .claude-plugin/marketplace.json

      - name: Validate CLI
        run: python -m arcaneum.cli.main --help

      - name: Run validation script
        run: ./scripts/validate-plugin.sh

      - name: Test CLI commands
        run: ./scripts/test-plugin-commands.sh
```

## Local Development Best Practices

### 1. Use Feature Branches

```bash
git checkout -b feature/improve-search-command
# Make changes
# Test in Claude Code
git commit -m "Improve search command documentation"
```

### 2. Test Incrementally

Don't batch all changes:
- Change one command → test → commit
- Add one feature → test → commit
- Fix one bug → test → commit

### 3. Keep Test Collections Clean

```bash
# At start of day
arc delete-collection test-dev --confirm 2>/dev/null || true

# At end of testing
arc delete-collection test-* --confirm 2>/dev/null || true
```

### 4. Document Bugs Immediately

Use beads to track issues found during testing:

```bash
bd create "Search filter syntax unclear in help text" \
  --type bug \
  --priority 1 \
  --json
```

## References

- **Claude Code Plugins**: https://docs.claude.com/en/docs/claude-code/plugins
- **Plugin Marketplaces**: https://docs.claude.com/en/docs/claude-code/plugin-marketplaces
- **Plugins Reference**: https://docs.claude.com/en/docs/claude-code/plugins-reference
- **Slash Commands**: https://docs.claude.com/en/docs/claude-code/slash-commands
- **RDR-006**: `docs/rdr/RDR-006-claude-code-integration.md`
- **Output Format**: `docs/reference/cli-output-format.md`
- **Compliance Report**: `docs/reference/plugin-compliance.md`
- **Beads Plugin**: https://github.com/steveyegge/beads (reference implementation)

## Quick Reference

### Essential Commands

```bash
# Validation
./scripts/validate-plugin.sh

# Local testing in Claude Code
/plugin marketplace add $(pwd)
/plugin install arc@arcaneum-marketplace
/help  # Verify commands appear

# Test a command
/arc:list-collections --json

# Update after changes
/plugin marketplace update arcaneum-marketplace

# Clean up
/plugin uninstall arc
/plugin marketplace remove arcaneum-marketplace
```

### Testing Checklist Summary

- [ ] JSON manifests valid
- [ ] All commands have frontmatter
- [ ] ${CLAUDE_PLUGIN_ROOT} used everywhere
- [ ] CLI loads without errors
- [ ] Fresh install works
- [ ] JSON output works
- [ ] Error handling works
- [ ] Progress tracking works
- [ ] Documentation is clear
- [ ] No sensitive data committed

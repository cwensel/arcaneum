#!/bin/bash
# Validate Arcaneum plugin structure for Claude Code compliance
# Based on: docs/plugin-marketplace-testing.md

set -e

echo "Validating Arcaneum plugin structure..."
echo ""

# Check JSON files
echo "✓ Validating JSON syntax..."
python -m json.tool .claude-plugin/plugin.json > /dev/null
python -m json.tool .claude-plugin/marketplace.json > /dev/null

# Check required fields in plugin.json
echo "✓ Checking plugin.json required fields..."
python -c "
import json
p = json.load(open('.claude-plugin/plugin.json'))
required = ['name', 'version', 'description']
for field in required:
    assert field in p, f'Missing required field: {field}'
print('  - name:', p['name'])
print('  - version:', p['version'])
print('  - description:', p['description'][:50] + '...')
"

# Check required fields in marketplace.json
echo "✓ Checking marketplace.json required fields..."
python -c "
import json
m = json.load(open('.claude-plugin/marketplace.json'))
assert 'owner' in m, 'Missing owner field'
assert 'plugins' in m, 'Missing plugins array'
assert len(m['plugins']) > 0, 'No plugins defined'
print('  - owner:', m['owner']['name'])
print('  - plugins:', len(m['plugins']))
"

# Check commands directory
echo "✓ Verifying commands directory..."
if [ ! -d commands ]; then
    echo "❌ commands/ directory missing at root"
    exit 1
fi

CMD_COUNT=$(ls commands/*.md 2>/dev/null | wc -l | tr -d ' ')
echo "  - Found $CMD_COUNT command files"

# Check all commands have frontmatter
echo "✓ Checking command frontmatter..."
for f in commands/*.md; do
    if ! head -3 "$f" | grep -q "^description:"; then
        echo "❌ Missing 'description:' in frontmatter: $f"
        exit 1
    fi
    if ! head -3 "$f" | grep -q "^argument-hint:"; then
        echo "❌ Missing 'argument-hint:' in frontmatter: $f"
        exit 1
    fi
done
echo "  - All commands have complete frontmatter"

# Check CLAUDE_PLUGIN_ROOT usage
echo "✓ Verifying CLAUDE_PLUGIN_ROOT usage..."
count=$(grep -l "CLAUDE_PLUGIN_ROOT" commands/*.md | wc -l | tr -d ' ')
total=$(ls commands/*.md | wc -l | tr -d ' ')
if [ "$count" -ne "$total" ]; then
    echo "❌ Not all commands use CLAUDE_PLUGIN_ROOT ($count/$total)"
    exit 1
fi
echo "  - All $total commands use \${CLAUDE_PLUGIN_ROOT}"

# Check $ARGUMENTS usage
echo "✓ Verifying \$ARGUMENTS usage..."
args_count=$(grep -l '\$ARGUMENTS' commands/*.md | wc -l | tr -d ' ')
if [ "$args_count" -ne "$total" ]; then
    echo "⚠️  Warning: Not all commands use \$ARGUMENTS ($args_count/$total)"
    echo "  - Some commands may use positional args instead"
fi

# Check paths use relative ./ prefix
echo "✓ Checking relative paths in plugin.json..."
python -c "
import json
p = json.load(open('.claude-plugin/plugin.json'))
if 'commands' in p:
    for cmd in p['commands']:
        if not cmd.startswith('./'):
            print(f'❌ Path should start with ./: {cmd}')
            exit(1)
print('  - All command paths use ./ prefix')
"

# Test CLI loads
echo "✓ Testing CLI loads..."
python -m arcaneum.cli.main --help > /dev/null
echo "  - Python CLI executes successfully"

# Test version command
echo "✓ Testing version command..."
VERSION=$(python -m arcaneum.cli.main --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
echo "  - CLI version: $VERSION"

# Check version consistency
echo "✓ Checking version consistency..."
python -c "
import json
p = json.load(open('.claude-plugin/plugin.json'))
m = json.load(open('.claude-plugin/marketplace.json'))
plugin_version = p['version']
marketplace_plugin_version = m['plugins'][0]['version']
assert plugin_version == marketplace_plugin_version, f'Version mismatch: {plugin_version} != {marketplace_plugin_version}'
print(f'  - Versions consistent: {plugin_version}')
"

echo ""
echo "✅ All validation checks passed!"
echo ""
echo "Plugin is ready for local testing in Claude Code:"
echo "  /plugin marketplace add $(pwd)"
echo "  /plugin install arc@arcaneum-marketplace"
echo ""
echo "To test commands:"
echo "  /arc:list-collections"
echo "  /arc:list-models --json"
echo "  /help  # See all available commands"

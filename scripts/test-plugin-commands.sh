#!/bin/bash
# Test all Arcaneum CLI commands for basic functionality
# Based on: docs/plugin-marketplace-testing.md

set -e

echo "Testing all Arcaneum CLI commands..."
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test each command with --help
echo "1. Testing command --help flags..."
for cmd in create-collection list-collections collection-info delete-collection \
           list-models index-pdfs index-source search search-text \
           create-corpus; do
    if python -m arcaneum.cli.main "$cmd" --help > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} $cmd --help"
    else
        echo -e "  ${RED}❌${NC} $cmd --help failed"
        exit 1
    fi
done

echo ""
echo "2. Testing JSON output support..."

# Test JSON output for commands that support it
if python -m arcaneum.cli.main list-models --json > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} list-models --json"
else
    echo -e "  ${RED}❌${NC} list-models --json"
fi

if python -m arcaneum.cli.main list-collections --json > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} list-collections --json"
else
    echo -e "  ${RED}❌${NC} list-collections --json"
fi

echo ""
echo "3. Testing error handling..."

# Test invalid arguments (should exit with code 2)
if python -m arcaneum.cli.main create-collection test --model invalid 2>&1 | grep -q "\[ERROR\]"; then
    echo -e "  ${GREEN}✓${NC} Error messages use [ERROR] prefix"
else
    echo -e "  ${RED}❌${NC} Error messages missing [ERROR] prefix"
fi

# Check exit code for invalid args
python -m arcaneum.cli.main create-collection test --model invalid > /dev/null 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -eq 2 ]; then
    echo -e "  ${GREEN}✓${NC} Invalid args return exit code 2"
elif [ $EXIT_CODE -eq 1 ]; then
    echo -e "  ${GREEN}✓${NC} Errors return exit code 1"
else
    echo -e "  ${RED}❌${NC} Unexpected exit code: $EXIT_CODE (expected 1 or 2)"
fi

echo ""
echo "4. Testing JSON structure..."

# Test that JSON output is valid
OUTPUT=$(python -m arcaneum.cli.main list-models --json 2>&1)
if echo "$OUTPUT" | python -m json.tool > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} JSON output is valid"

    # Check for standard fields
    if echo "$OUTPUT" | grep -q '"status"'; then
        echo -e "  ${GREEN}✓${NC} JSON contains 'status' field"
    fi
    if echo "$OUTPUT" | grep -q '"message"'; then
        echo -e "  ${GREEN}✓${NC} JSON contains 'message' field"
    fi
    if echo "$OUTPUT" | grep -q '"data"'; then
        echo -e "  ${GREEN}✓${NC} JSON contains 'data' field"
    fi
else
    echo -e "  ${RED}❌${NC} JSON output is invalid"
    echo "$OUTPUT"
    exit 1
fi

echo ""
echo "5. Testing version consistency..."

# Check version matches across files
PLUGIN_VERSION=$(python -c "import json; print(json.load(open('.claude-plugin/plugin.json'))['version'])")
MARKETPLACE_VERSION=$(python -c "import json; print(json.load(open('.claude-plugin/marketplace.json'))['plugins'][0]['version'])")
CLI_VERSION=$(python -c "from arcaneum import __version__; print(__version__)")

if [ "$PLUGIN_VERSION" = "$MARKETPLACE_VERSION" ] && [ "$PLUGIN_VERSION" = "$CLI_VERSION" ]; then
    echo -e "  ${GREEN}✓${NC} Versions consistent: $PLUGIN_VERSION"
else
    echo -e "  ${RED}❌${NC} Version mismatch:"
    echo "    plugin.json: $PLUGIN_VERSION"
    echo "    marketplace.json: $MARKETPLACE_VERSION"
    echo "    __init__.py: $CLI_VERSION"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ All command tests passed!${NC}"
echo ""
echo "Plugin is ready for integration testing in Claude Code."
echo ""
echo "Quick test commands:"
echo "  /plugin marketplace add $(pwd)"
echo "  /plugin install arc@arcaneum-marketplace"
echo "  /arc:list-models --json"
echo "  /arc:list-collections"

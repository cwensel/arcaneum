#!/usr/bin/env bash
set -euo pipefail

# Bump version across all project files
# Usage: ./scripts/bump-version.sh <new-version>
# Example: ./scripts/bump-version.sh 0.3.0

if [ $# -ne 1 ]; then
    echo "Usage: $0 <new-version>"
    echo "Example: $0 0.3.0"
    exit 1
fi

NEW_VERSION="$1"

# Validate version format (semver without v prefix)
if ! echo "$NEW_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "ERROR: Version must be in format X.Y.Z (e.g., 0.3.0)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Bumping version to $NEW_VERSION..."

# 1. Update pyproject.toml
sed -i.bak "s/^version = \".*\"/version = \"$NEW_VERSION\"/" "$PROJECT_ROOT/pyproject.toml"
rm -f "$PROJECT_ROOT/pyproject.toml.bak"
echo "  ✓ pyproject.toml"

# 2. Update .claude-plugin/plugin.json
sed -i.bak "s/\"version\": \".*\"/\"version\": \"$NEW_VERSION\"/" "$PROJECT_ROOT/.claude-plugin/plugin.json"
rm -f "$PROJECT_ROOT/.claude-plugin/plugin.json.bak"
echo "  ✓ .claude-plugin/plugin.json"

# 3. Update .claude-plugin/marketplace.json
sed -i.bak "s/\"version\": \".*\"/\"version\": \"$NEW_VERSION\"/" "$PROJECT_ROOT/.claude-plugin/marketplace.json"
rm -f "$PROJECT_ROOT/.claude-plugin/marketplace.json.bak"
echo "  ✓ .claude-plugin/marketplace.json"

# 4. Update src/arcaneum/__init__.py
sed -i.bak "s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" "$PROJECT_ROOT/src/arcaneum/__init__.py"
rm -f "$PROJECT_ROOT/src/arcaneum/__init__.py.bak"
echo "  ✓ src/arcaneum/__init__.py"

echo ""
echo "Version bumped to $NEW_VERSION in all files."
echo ""
echo "Next steps:"
echo "  git add -A && git commit -m \"Release v$NEW_VERSION\""
echo "  git tag -a v$NEW_VERSION -m \"Release v$NEW_VERSION\""
echo "  git push origin main --tags"

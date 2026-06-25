#!/bin/bash
# Test script for Arcaneum installation in fresh Ubuntu environment
# This script runs inside the Docker container

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
print_header() {
    echo ""
    echo "=================================================="
    echo "$1"
    echo "=================================================="
}

print_test() {
    echo ""
    echo "→ Testing: $1"
}

pass() {
    echo -e "${GREEN}✓ PASS:${NC} $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

fail() {
    echo -e "${RED}✗ FAIL:${NC} $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

warn() {
    echo -e "${YELLOW}⚠ WARN:${NC} $1"
}

# Start tests
print_header "Arcaneum Installation Test"
echo "Test environment: $(lsb_release -d | cut -f2)"
echo "Test user: $(whoami)"
echo "Home directory: $HOME"
echo "Working directory: $(pwd)"

# Test 1: Python version
print_test "Python version >= 3.12"
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if [ "$(echo "$PYTHON_VERSION >= 3.12" | bc)" -eq 1 ]; then
    pass "Python $PYTHON_VERSION is installed"
else
    fail "Python $PYTHON_VERSION is too old (need 3.12+)"
    exit 1
fi

# Test 2: System dependencies
print_test "System dependencies (tesseract, poppler)"
if command -v tesseract &> /dev/null; then
    pass "tesseract is installed: $(tesseract --version | head -n1)"
else
    fail "tesseract is not installed"
fi

if command -v pdfinfo &> /dev/null; then
    pass "poppler-utils is installed: $(pdfinfo -v 2>&1 | head -n1)"
else
    fail "poppler-utils is not installed"
fi

# Test 3: Git availability
print_test "Git availability"
if command -v git &> /dev/null; then
    pass "git is installed: $(git --version)"
else
    fail "git is not installed"
fi

# Test 4: Docker CLI availability
print_test "Docker CLI availability"
if command -v docker &> /dev/null; then
    pass "docker CLI is installed: $(docker --version)"
else
    fail "docker CLI is not installed"
fi

# Test 5: Docker daemon connectivity
print_test "Docker daemon connectivity"
if docker info &> /dev/null; then
    pass "Docker daemon is accessible"
else
    fail "Cannot connect to Docker daemon (is socket mounted?)"
fi

# Test 6: Build and install Arcaneum
print_test "Building Arcaneum wheel"
cd ~/arcaneum
rm -rf build dist src/arcaneum.egg-info
if python3 -m build --wheel; then
    pass "Arcaneum wheel built successfully"
else
    fail "Arcaneum wheel build failed"
    exit 1
fi

WHEEL_PATH="$(find dist -maxdepth 1 -name 'arcaneum-*.whl' -print -quit)"
if [ -z "$WHEEL_PATH" ]; then
    fail "Arcaneum wheel not found in dist/"
    exit 1
fi

print_test "Installing Arcaneum wheel with pipx"
if python3 -m pipx install "$WHEEL_PATH"; then
    pass "Arcaneum installed successfully from wheel"
else
    fail "Arcaneum wheel installation failed"
    exit 1
fi

# Test 7: Arc command availability
print_test "Arc command availability"
if command -v arc &> /dev/null; then
    pass "arc command is available: $(which arc)"
else
    fail "arc command not found in PATH"
    exit 1
fi

# Test 8: Arc doctor check
print_test "Running arc doctor"
echo ""
if arc doctor; then
    pass "arc doctor completed (see output above)"
else
    warn "arc doctor reported issues (expected: Qdrant not running yet)"
fi

# Test 9: Start Qdrant container
print_test "Starting Qdrant container"
echo ""
if arc container start; then
    pass "Qdrant container started successfully"
    # Give Qdrant a moment to fully start
    sleep 5
else
    fail "Failed to start Qdrant container"
fi

# Test 10: Check container status
print_test "Checking container status"
echo ""
if arc container status; then
    pass "Container status command works"
else
    fail "Container status check failed"
fi

# Test 11: Run arc doctor again with Qdrant running
print_test "Running arc doctor with Qdrant running"
echo ""
if arc doctor; then
    pass "arc doctor passed with Qdrant running"
else
    fail "arc doctor failed even with Qdrant running"
fi

# Test 12: Create test collection
print_test "Creating test collection"
TEST_COLLECTION="install-test-$(date +%s)"
if arc collection create "$TEST_COLLECTION" --model stella --json &> /dev/null; then
    pass "Created test collection: $TEST_COLLECTION"
else
    fail "Failed to create test collection"
fi

# Test 13: List collections
print_test "Listing collections"
if arc collection list --json &> /dev/null; then
    pass "Collection list command works"
else
    fail "Failed to list collections"
fi

# Test 14: Delete test collection
print_test "Deleting test collection"
if arc collection delete "$TEST_COLLECTION" --json &> /dev/null; then
    pass "Deleted test collection"
else
    warn "Failed to delete test collection (may need manual cleanup)"
fi

# Test 15: Stop container
print_test "Stopping Qdrant container"
if arc container stop; then
    pass "Container stopped successfully"
else
    warn "Failed to stop container gracefully"
fi

# Print summary
print_header "Test Results Summary"
echo ""
echo "Total tests: $((TESTS_PASSED + TESTS_FAILED))"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  ALL TESTS PASSED! ✓${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  SOME TESTS FAILED! ✗${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

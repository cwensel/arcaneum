#!/bin/bash
# Test script for Arcaneum installation on macOS
# This creates a fresh Python virtual environment and tests the installation

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${REPO_ROOT}/test-install-tmp"
VENV_DIR="${TEST_DIR}/venv"

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}=================================================="
    echo "$1"
    echo -e "==================================================${NC}"
}

print_test() {
    echo ""
    echo -e "${BLUE}→ Testing: $1${NC}"
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

# Cleanup function
cleanup() {
    if [ -d "$TEST_DIR" ]; then
        print_header "Cleaning up test environment"
        rm -rf "$TEST_DIR"
        echo "Test directory removed: $TEST_DIR"
    fi
}

# Register cleanup on exit
trap cleanup EXIT

# Start tests
print_header "Arcaneum macOS Installation Test"
echo "Test environment: macOS $(sw_vers -productVersion)"
echo "Test user: $(whoami)"
echo "Repository root: $REPO_ROOT"

# Test 1: Check Homebrew
print_test "Homebrew installation"
if command -v brew &> /dev/null; then
    pass "Homebrew is installed: $(brew --version | head -n1)"
else
    fail "Homebrew is not installed"
    echo ""
    echo "Install Homebrew from: https://brew.sh"
    exit 1
fi

# Test 2: Check Python version
print_test "Python version >= 3.12"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)

    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 12 ]; then
        pass "Python $PYTHON_VERSION is installed"
    else
        fail "Python $PYTHON_VERSION is too old (need 3.12+)"
        echo ""
        echo "Install Python 3.12+ with: brew install python@3.12"
        exit 1
    fi
else
    fail "Python 3 is not installed"
    exit 1
fi

# Test 3: Check Docker Desktop
print_test "Docker Desktop installation"
if command -v docker &> /dev/null; then
    pass "Docker is installed: $(docker --version)"
else
    fail "Docker is not installed"
    echo ""
    echo "Install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Test 4: Docker daemon running
print_test "Docker daemon status"
if docker info &> /dev/null; then
    pass "Docker daemon is running"
else
    fail "Docker daemon is not running"
    echo ""
    echo "Please start Docker Desktop"
    exit 1
fi

# Test 5: Check/Install tesseract
print_test "Tesseract OCR installation"
if command -v tesseract &> /dev/null; then
    pass "tesseract is installed: $(tesseract --version 2>&1 | head -n1)"
else
    warn "tesseract is not installed, attempting to install..."
    if brew install tesseract; then
        pass "tesseract installed successfully"
    else
        fail "Failed to install tesseract"
    fi
fi

# Test 6: Check/Install poppler
print_test "Poppler utilities installation"
if command -v pdfinfo &> /dev/null; then
    pass "poppler is installed: $(pdfinfo -v 2>&1 | head -n1)"
else
    warn "poppler is not installed, attempting to install..."
    if brew install poppler; then
        pass "poppler installed successfully"
    else
        fail "Failed to install poppler"
    fi
fi

# Test 7: Create test directory and venv
print_test "Creating test virtual environment"
mkdir -p "$TEST_DIR"
if python3 -m venv "$VENV_DIR"; then
    pass "Virtual environment created at: $VENV_DIR"
else
    fail "Failed to create virtual environment"
    exit 1
fi

# Activate venv for remaining tests
source "$VENV_DIR/bin/activate"

# Test 8: Upgrade pip
print_test "Upgrading pip in virtual environment"
if pip install --upgrade pip setuptools wheel &> /dev/null; then
    pass "pip upgraded successfully"
else
    fail "Failed to upgrade pip"
fi

# Test 9: Install Arcaneum
print_test "Installing Arcaneum from local source"
cd "$REPO_ROOT"
if pip install -e .; then
    pass "Arcaneum installed successfully"
else
    fail "Arcaneum installation failed"
    exit 1
fi

# Test 10: Arc command availability
print_test "Arc command availability"
if command -v arc &> /dev/null; then
    pass "arc command is available: $(which arc)"
else
    fail "arc command not found in PATH"
    exit 1
fi

# Test 11: Arc doctor check
print_test "Running arc doctor"
echo ""
if arc doctor; then
    pass "arc doctor completed (see output above)"
else
    warn "arc doctor reported issues (expected: Qdrant not running yet)"
fi

# Test 12: Start Qdrant container
print_test "Starting Qdrant container"
echo ""
if arc container start; then
    pass "Qdrant container started successfully"
    # Give Qdrant a moment to fully start
    sleep 5
else
    fail "Failed to start Qdrant container"
fi

# Test 13: Check container status
print_test "Checking container status"
echo ""
if arc container status; then
    pass "Container status command works"
else
    fail "Container status check failed"
fi

# Test 14: Run arc doctor again with Qdrant running
print_test "Running arc doctor with Qdrant running"
echo ""
if arc doctor; then
    pass "arc doctor passed with Qdrant running"
else
    fail "arc doctor failed even with Qdrant running"
fi

# Test 15: Create test collection
print_test "Creating test collection"
TEST_COLLECTION="install-test-$(date +%s)"
if arc collection create "$TEST_COLLECTION" --model stella --json &> /dev/null; then
    pass "Created test collection: $TEST_COLLECTION"
else
    fail "Failed to create test collection"
fi

# Test 16: List collections
print_test "Listing collections"
if arc collection list --json &> /dev/null; then
    pass "Collection list command works"
else
    fail "Failed to list collections"
fi

# Test 17: Delete test collection
print_test "Deleting test collection"
if arc collection delete "$TEST_COLLECTION" --json &> /dev/null; then
    pass "Deleted test collection"
else
    warn "Failed to delete test collection (may need manual cleanup)"
fi

# Test 18: Stop container
print_test "Stopping Qdrant container"
if arc container stop; then
    pass "Container stopped successfully"
else
    warn "Failed to stop container gracefully"
fi

# Deactivate venv
deactivate

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
    echo ""
    echo "Test environment will be cleaned up automatically."
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  SOME TESTS FAILED! ✗${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Test environment left at: $TEST_DIR"
    echo "To clean up manually: rm -rf $TEST_DIR"
    trap - EXIT  # Disable auto-cleanup on failure
    exit 1
fi

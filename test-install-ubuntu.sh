#!/bin/bash
# Runner script for Ubuntu installation tests
# This script builds Docker images and runs installation tests for Ubuntu 22.04 and 24.04

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_PREFIX="arcaneum-install-test"

# Track results
UBUNTU22_PASSED=false
UBUNTU24_PASSED=false

print_header() {
    echo ""
    echo -e "${BLUE}=================================================="
    echo "$1"
    echo -e "==================================================${NC}"
    echo ""
}

print_step() {
    echo ""
    echo -e "${BLUE}→ $1${NC}"
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

error() {
    echo -e "${RED}✗ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check prerequisites
print_header "Arcaneum Ubuntu Installation Test Runner"
echo "Repository root: $REPO_ROOT"

print_step "Checking prerequisites"

# Check Docker
if ! command -v docker &> /dev/null; then
    error "Docker is not installed"
    echo "Please install Docker Desktop (Mac) or Docker Engine (Linux)"
    exit 1
fi
success "Docker is installed: $(docker --version)"

# Check Docker daemon
if ! docker info &> /dev/null; then
    error "Docker daemon is not running"
    echo "Please start Docker Desktop or the Docker daemon"
    exit 1
fi
success "Docker daemon is running"

# Check for required files
if [ ! -f "$REPO_ROOT/tests/docker/Dockerfile.ubuntu22" ]; then
    error "Dockerfile.ubuntu22 not found"
    exit 1
fi
if [ ! -f "$REPO_ROOT/tests/docker/Dockerfile.ubuntu24" ]; then
    error "Dockerfile.ubuntu24 not found"
    exit 1
fi
if [ ! -f "$REPO_ROOT/tests/docker/test-installation.sh" ]; then
    error "test-installation.sh not found"
    exit 1
fi
success "All required test files found"

# Function to run test for a specific Ubuntu version
run_ubuntu_test() {
    local VERSION=$1
    local DOCKERFILE=$2
    local IMAGE_NAME="${IMAGE_PREFIX}:ubuntu${VERSION}"

    print_header "Testing Ubuntu ${VERSION} LTS"

    # Build image
    print_step "Building Docker image for Ubuntu ${VERSION}"
    if docker build \
        -f "$REPO_ROOT/tests/docker/$DOCKERFILE" \
        -t "$IMAGE_NAME" \
        "$REPO_ROOT"; then
        success "Image built successfully"
    else
        error "Failed to build Docker image"
        return 1
    fi

    # Run tests
    print_step "Running installation tests in container"
    echo ""

    if docker run \
        --rm \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -e DOCKER_HOST=unix:///var/run/docker.sock \
        "$IMAGE_NAME"; then
        success "Ubuntu ${VERSION} tests PASSED"
        return 0
    else
        error "Ubuntu ${VERSION} tests FAILED"
        return 1
    fi
}

# Run tests for Ubuntu 22.04
if run_ubuntu_test "22.04" "Dockerfile.ubuntu22"; then
    UBUNTU22_PASSED=true
else
    UBUNTU22_PASSED=false
fi

echo ""
echo "=========================================="
echo ""

# Run tests for Ubuntu 24.04
if run_ubuntu_test "24.04" "Dockerfile.ubuntu24"; then
    UBUNTU24_PASSED=true
else
    UBUNTU24_PASSED=false
fi

# Print final summary
print_header "Final Test Summary"

echo "Test Results:"
echo ""
if [ "$UBUNTU22_PASSED" = true ]; then
    echo -e "  Ubuntu 22.04 LTS: ${GREEN}✓ PASSED${NC}"
else
    echo -e "  Ubuntu 22.04 LTS: ${RED}✗ FAILED${NC}"
fi

if [ "$UBUNTU24_PASSED" = true ]; then
    echo -e "  Ubuntu 24.04 LTS: ${GREEN}✓ PASSED${NC}"
else
    echo -e "  Ubuntu 24.04 LTS: ${RED}✗ FAILED${NC}"
fi

echo ""

# Cleanup option
print_step "Cleanup"
echo "Docker images created:"
echo "  - ${IMAGE_PREFIX}:ubuntu22.04"
echo "  - ${IMAGE_PREFIX}:ubuntu24.04"
echo ""
read -p "Remove test images? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker rmi "${IMAGE_PREFIX}:ubuntu22.04" 2>/dev/null || true
    docker rmi "${IMAGE_PREFIX}:ubuntu24.04" 2>/dev/null || true
    success "Test images removed"
fi

# Exit with appropriate code
if [ "$UBUNTU22_PASSED" = true ] && [ "$UBUNTU24_PASSED" = true ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  ALL TESTS PASSED! ✓${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  SOME TESTS FAILED! ✗${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

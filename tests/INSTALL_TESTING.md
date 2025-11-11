# Installation Testing

This directory contains automated tests for verifying Arcaneum installation in fresh environments.

## Overview

The installation tests ensure that Arcaneum can be successfully installed and run on:

- **Ubuntu 22.04 LTS** (via Docker)
- **Ubuntu 24.04 LTS** (via Docker)
- **macOS** (via virtual environment)

These tests simulate a fresh installation from scratch, including:

- Python 3.12+ installation
- System dependencies (tesseract, poppler)
- Docker availability
- Arcaneum pip installation
- Basic functionality verification

## Test Files

### Ubuntu Testing (Docker-based)

- **`test-install-ubuntu.sh`** (root directory) - Main test runner for Ubuntu
  - Builds Docker images for Ubuntu 22.04 and 24.04
  - Runs installation tests in isolated containers
  - Reports results and optionally cleans up images

- **`tests/docker/Dockerfile.ubuntu22`** - Ubuntu 22.04 test image
  - Installs Python 3.12 via deadsnakes PPA
  - Installs system dependencies
  - Sets up test environment

- **`tests/docker/Dockerfile.ubuntu24`** - Ubuntu 24.04 test image
  - Uses built-in Python 3.12
  - Installs system dependencies
  - Sets up test environment

- **`tests/docker/test-installation.sh`** - Shared test script
  - Runs inside containers
  - Executes 15+ validation tests
  - Tests arc doctor, container management, collections

### macOS Testing

- **`test-install-mac.sh`** (root directory) - macOS test script
  - Creates fresh Python virtual environment
  - Installs/verifies system dependencies via Homebrew
  - Tests Arcaneum installation
  - Cleans up automatically on success

## Prerequisites

### For Ubuntu Testing (Docker)

- **Docker Desktop** (Mac/Windows) or **Docker Engine** (Linux)
- Docker daemon must be running
- Minimum 4GB disk space (for images and layers)
- Internet connection (for package downloads)

### For macOS Testing

- **macOS** (any recent version)
- **Homebrew** (<https://brew.sh>)
- **Docker Desktop for Mac** (<https://www.docker.com/products/docker-desktop>)
- **Python 3.12+** (can be installed during test)
- Internet connection

## Running the Tests

### Ubuntu Tests

From the repository root:

```bash
./test-install-ubuntu.sh
```

This will:

1. Check Docker availability
2. Build test images for Ubuntu 22.04 and 24.04
3. Run installation tests in each container
4. Display results
5. Optionally clean up Docker images

**Expected time**: 10-15 minutes (first run, includes downloads)

### macOS Tests

From the repository root:

```bash
./test-install-mac.sh
```

This will:

1. Check prerequisites (Homebrew, Docker, Python)
2. Install missing system dependencies (tesseract, poppler)
3. Create fresh virtual environment
4. Install and test Arcaneum
5. Clean up test environment

**Expected time**: 5-10 minutes (depending on dependencies)

## What Gets Tested

Both test suites verify:

1. **Python Version**: >= 3.12
2. **System Dependencies**: tesseract-ocr, poppler-utils
3. **Docker Availability**: CLI and daemon
4. **Installation**: pip install process
5. **Command Availability**: `arc` command in PATH
6. **Arc Doctor**: Health check passes
7. **Container Management**: Start/stop Qdrant
8. **Basic Operations**:
   - Collection creation
   - Collection listing
   - Collection deletion

## Test Output

### Success Example

```text
==================================================
  ALL TESTS PASSED! ✓
==================================================

Total tests: 15
Passed: 15
Failed: 0
```

### Failure Example

```text
==================================================
  SOME TESTS FAILED! ✗
==================================================

Total tests: 15
Passed: 12
Failed: 3

✗ FAIL: Docker daemon is not accessible
✗ FAIL: Failed to start Qdrant container
✗ FAIL: Container status check failed
```

## Troubleshooting

### Ubuntu Tests

**Problem**: "Cannot connect to Docker daemon"

**Solution**: Ensure Docker is running and you have permission to access it:

```bash
docker info  # Should succeed
```

**Problem**: "Build failed during package installation"

**Solution**: Check your internet connection. Some packages are large (Python, Docker CLI).

**Problem**: "Failed to start Qdrant container"

**Solution**: The test mounts `/var/run/docker.sock` into the container. Ensure Docker socket is accessible:

```bash
ls -l /var/run/docker.sock
```

### macOS Tests

**Problem**: "Homebrew is not installed"

**Solution**: Install Homebrew from <https://brew.sh>

**Problem**: "Python version too old"

**Solution**: Install Python 3.12+:

```bash
brew install python@3.12
```

**Problem**: "Docker Desktop is not running"

**Solution**: Start Docker Desktop application and wait for it to be ready.

**Problem**: "tesseract or poppler installation failed"

**Solution**: The script will attempt to install via Homebrew. If this fails, install manually:

```bash
brew install tesseract poppler
```

## Technical Details

### Docker Architecture

The Ubuntu tests use a "Docker-in-Docker" approach with host socket mounting:

- Container has Docker CLI installed
- Container mounts host's Docker socket (`/var/run/docker.sock`)
- When `arc container start` runs, it creates containers on the *host* Docker daemon
- This avoids the complexity of true Docker-in-Docker (DinD)

### Virtual Environment Isolation

The macOS test creates a temporary directory (`test-install-tmp/`) with a fresh Python virtual environment to ensure:

- No interference from existing packages
- Clean slate for testing
- Easy cleanup after testing

### Test Cleanup

- **Ubuntu**: Prompts to remove Docker images after tests
- **macOS**: Automatically removes test directory on success, keeps on failure for debugging

## CI/CD Integration (Future)

These tests are designed for one-time manual validation but could be integrated into CI/CD:

### GitHub Actions Example

```yaml
name: Installation Tests

on: [push, pull_request]

jobs:
  test-ubuntu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Ubuntu tests
        run: ./test-install-ubuntu.sh

  test-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run macOS tests
        run: ./test-install-mac.sh
```

## Maintenance

### Updating Test Scripts

When adding new Arcaneum features that require verification:

1. Add test cases to `tests/docker/test-installation.sh` (Ubuntu)
2. Add corresponding tests to `test-install-mac.sh` (macOS)
3. Update expected test counts in documentation
4. Test on both platforms

### Updating Python Requirements

If minimum Python version changes:

1. Update Dockerfiles (both Ubuntu versions)
2. Update test scripts (version check logic)
3. Update documentation

### Adding New Linux Distributions

To add support for other distributions:

1. Create new Dockerfile (e.g., `Dockerfile.fedora`)
2. Adjust package manager commands (dnf, yum, etc.)
3. Update `test-install-ubuntu.sh` to include new distribution
4. Test thoroughly

## Contact

For issues with the test scripts themselves (not Arcaneum installation issues):

- Check the Arcaneum issue tracker
- Review test script logs for detailed error messages
- Ensure all prerequisites are met before reporting issues

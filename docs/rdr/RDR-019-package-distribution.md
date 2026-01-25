# Recommendation 019: Arcaneum Package Distribution via GitHub Actions

## Metadata

- **Date**: 2026-01-19
- **Status**: Recommendation
- **Type**: Feature
- **Priority**: High
- **Related Issues**: None
- **Related Tests**: None (CI/CD configuration)

## Problem Statement

Arcaneum needs to be packaged and distributed to users in a way that:

1. Is simple for end users to install
2. Can be automated via GitHub Actions
3. Is compatible with the existing Claude Code plugin architecture
4. Supports macOS and Linux (Windows is not a priority)
5. Separates the Claude plugin from the CLI so users can install globally via pipx/Homebrew,
   then install the plugin separately which references the global `arc` command

### Platform Requirements

**Primary platforms**: macOS (Apple Silicon and Intel) and Linux

**Windows**: Not a priority. Arcaneum has runtime dependencies on Docker for Qdrant and
MeiliSearch services. While Docker Desktop exists for Windows, the tooling and workflow
are optimized for Unix-like environments. Users on Windows should consider WSL2.

**Rationale**: The target user base (developers using Claude Code for semantic search
over local corpora) predominantly uses macOS or Linux workstations.

## Context

### Background

Arcaneum is a Python 3.12+ CLI tool that provides semantic and full-text search capabilities
through Qdrant and MeiliSearch. It also functions as a Claude Code plugin with 5 auto-activating
skills located in `.claude-plugin/`.

Current state:

- Package version: 0.2.0
- Entry point: `arc` command via `arcaneum.cli.main:main`
- Build system: setuptools with `pyproject.toml`
- Heavy dependencies: PyTorch, sentence-transformers, transformers (~1.2GB installed, see analysis below)
- LLM embedding models: Downloaded on-demand (~500MB-1.5GB per model)

**Key Insight**: This is a specialized project not expecting wide adoption. All distribution
channels will be self-hosted (GitHub Releases, personal Homebrew tap, private PyPI index)
rather than official registries (PyPI, homebrew-core, Claude plugin directory).

### Technical Environment

- **Runtime**: Python 3.12+
- **Build Backend**: setuptools >= 61.0
- **CLI Framework**: Click 8.3.0+
- **Key Dependencies**: torch, sentence-transformers, qdrant-client, meilisearch
- **Claude Plugin**: `.claude-plugin/plugin.json` with marketplace.json

### Dependency Size Analysis

Total installed size: **~1.2GB** (excluding on-demand downloads)

#### ML/AI Dependencies (~650MB)

| Package                    | Installed Size | Purpose                            |
| -------------------------- | -------------- | ---------------------------------- |
| `torch`                    | 382 MB         | PyTorch ML framework               |
| `onnxruntime`              | 128 MB         | ONNX inference (FastEmbed backend) |
| `transformers`             | 102 MB         | HuggingFace model loading          |
| `sentence-transformers`    | 4 MB           | Embedding model wrapper            |
| `fastembed`                | 1 MB           | Fast ONNX embeddings               |
| `tokenizers`               | 8 MB           | Fast tokenization (Rust)           |
| `huggingface-hub`          | 5 MB           | Model download/caching             |
| `safetensors`              | 1 MB           | Safe model serialization           |

#### Document Processing Dependencies (~420MB)

| Package                    | Installed Size | Purpose                            |
| -------------------------- | -------------- | ---------------------------------- |
| `tree-sitter-language-pack`| 324 MB         | Source code parsing (40+ languages)|
| `opencv-python-headless`   | 99 MB          | Image processing for OCR           |
| `pymupdf`                  | 69 MB          | PDF text extraction                |
| `llama-index-core`         | 28 MB          | Document chunking utilities        |
| `Pillow`                   | 14 MB          | Image handling                     |
| `pdfminer-six`             | 8 MB           | PDF text extraction fallback       |

#### Core/Utility Dependencies (~50MB)

| Package                    | Installed Size | Purpose                            |
| -------------------------- | -------------- | ---------------------------------- |
| `numpy`                    | 32 MB          | Numerical operations               |
| `pygments`                 | 9 MB           | Syntax highlighting                |
| `qdrant-client`            | 5 MB           | Vector database client             |
| `pydantic`                 | 4 MB           | Data validation                    |
| `rich`                     | 2 MB           | CLI formatting                     |
| `psutil`                   | 2 MB           | System monitoring                  |
| `GitPython`                | 2 MB           | Git metadata extraction            |
| `click`                    | 1 MB           | CLI framework                      |
| `PyYAML`                   | 1 MB           | YAML parsing                       |
| `meilisearch`              | <1 MB          | Full-text search client            |
| `tqdm`                     | <1 MB          | Progress bars                      |
| Others                     | ~5 MB          | Various small utilities            |

#### Dependency Categories by Install Impact

| Category           | Size    | Required For                  | Could Be Optional? |
| ------------------ | ------- | ----------------------------- | ------------------ |
| PyTorch + ONNX     | ~510 MB | All embedding operations      | No (core feature)  |
| Tree-sitter        | ~324 MB | Source code indexing          | Yes (feature flag) |
| PDF Processing     | ~90 MB  | PDF indexing                  | Yes (feature flag) |
| OpenCV + Pillow    | ~113 MB | OCR, image processing         | Yes (feature flag) |
| Transformers/HF    | ~120 MB | Model loading                 | No (core feature)  |
| Core utilities     | ~50 MB  | Basic CLI operation           | No                 |

### On-Demand Downloads

Arcaneum downloads heavy components on first use, keeping initial install smaller:

| Component            | Size   | Trigger                  | Cache Location        |
| -------------------- | ------ | ------------------------ | --------------------- |
| Embedding Models     | ~1.5GB | First `arc search`       | `~/.arcaneum/models/` |
| Tree-sitter Grammars | ~50MB  | First `arc index source` | System cache          |
| Docker Images        | ~500MB | `arc container start`    | Docker daemon         |

Implementation: `EmbeddingClient.is_model_cached()` checks availability;
`local_files_only=True` prevents network calls when cached.

### Claude Code Plugin Compatibility Requirements

The distribution method must support:

- Plugin installation via `/plugin install owner/repo` (see [Claude Code Plugin Docs](https://docs.anthropic.com/en/docs/claude-code/plugins))
- The `.claude-plugin/` directory structure remaining intact
- Skills and commands being discoverable after installation
- **Separation of plugin from globally-installed CLI**: Plugin should assume `arc` is available
  in PATH, not bundle Python dependencies or ML models

## Research Findings

### Investigation Process

1. Analyzed Arcaneum's `pyproject.toml` and project structure
2. Researched Claude Code plugin distribution documentation
3. Investigated PyPI, Homebrew, npm, and binary distribution methods
4. Reviewed GitHub Actions workflows for each method
5. Studied multi-channel distribution strategies used by similar CLI+plugin projects
6. Analyzed on-demand download patterns in existing code

### Key Discoveries

1. **Claude Code Plugin Distribution** (see [Plugin Docs](https://docs.anthropic.com/en/docs/claude-code/plugins)):
   - Direct GitHub repository installation: `/plugin install owner/repo`
   - Custom marketplaces (self-hosted GitHub repos with `marketplace.json`)
   - Official plugin directory (requires approval, not pursued for this project)

2. **Self-Hosted Distribution is Required** for this specialized project:
   - No approval process required
   - Full control over release timing
   - GitHub Releases + personal Homebrew tap pattern works well

3. **Plugin in Same Repo, Separate Directory**:
   - Plugin lives in `.claude-plugin/` and `commands/` directories
   - Users install CLI globally first (pipx, Homebrew)
   - Then install plugin via `/plugin install cwensel/arcaneum`
   - Plugin's slash commands execute `arc` directly via Bash
   - No bundled Python dependencies in plugin - assumes global `arc`

4. **Optional MCP Server** (future consideration):
   - Could add `integrations/arcaneum-mcp/` for Claude Desktop users
   - Would provide typed tool interfaces as alternative to CLI
   - Higher token overhead than CLI approach (~10-50k vs 1-2k tokens)
   - Not required for Claude Code which has shell access

5. **Binary Distribution Impractical**:
   - Even with on-demand model downloads, binary would be ~400-500MB
   - PyTorch/ONNX runtime cannot be lazy-loaded
   - pipx/Homebrew handle Python dependencies better

## Proposed Solution

### Approach

Use a self-hosted multi-tier distribution strategy:

1. **Primary CLI**: GitHub Releases with wheel/sdist artifacts, installable via pipx
2. **Supplementary CLI**: Personal Homebrew tap (`cwensel/homebrew-arcaneum`)
3. **Claude Plugin**: Same repository, `/plugin install cwensel/arcaneum`
4. **Future Optional**: MCP server in `integrations/arcaneum-mcp/` for Claude Desktop

All distribution channels are self-hosted (no public PyPI, homebrew-core, or official plugin directory).

### Architecture: Plugin in Same Repository

The plugin remains in the main repository:

```text
arcaneum/
├── .claude-plugin/              # Plugin metadata
│   ├── plugin.json
│   ├── marketplace.json
│   └── skills/                  # Auto-activating skills
│       ├── arc-collection/
│       ├── arc-search/
│       └── ...
├── commands/                    # Slash commands (execute arc CLI)
│   ├── search.md
│   ├── collection.md
│   └── ...
├── src/arcaneum/               # CLI implementation
├── pyproject.toml              # CLI packaging
└── integrations/               # Future: MCP server
    └── arcaneum-mcp/           # Optional Claude Desktop integration
```

**Benefits**:

- Single source of truth - plugin and CLI versioned together
- Users install CLI globally first, then plugin references it
- Plugin installation is lightweight (just markdown files and metadata)
- MCP server can be added later as optional integration

### Technical Design

#### GitHub Release Workflow

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'  # Triggers on version tags like v0.2.0, v1.0.0

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install build dependencies
        run: pip install build

      - name: Build package
        run: python -m build

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            dist/*.whl
            dist/*.tar.gz
          generate_release_notes: true
```

#### pyproject.toml Enhancements

Add the following to `pyproject.toml`:

```toml
keywords = [
    "semantic-search",
    "full-text-search",
    "qdrant",
    "meilisearch",
    "vector-database",
    "embeddings",
    "claude-code",
    "ai-tools",
    "pdf-indexing",
    "code-indexing",
]

[project.urls]
Homepage = "https://github.com/cwensel/arcaneum"
Repository = "https://github.com/cwensel/arcaneum"
Documentation = "https://github.com/cwensel/arcaneum#readme"
Issues = "https://github.com/cwensel/arcaneum/issues"
Changelog = "https://github.com/cwensel/arcaneum/releases"
```

#### plugin.json Fixes

Update URLs in `.claude-plugin/plugin.json`:

```json
{
  "author": {
    "name": "Arcaneum Contributors",
    "url": "https://github.com/cwensel/arcaneum"
  },
  "homepage": "https://github.com/cwensel/arcaneum"
}
```

#### Version Coordination Script

Create `scripts/bump-version.sh` to update versions across:

- `pyproject.toml` (CLI version)
- `.claude-plugin/plugin.json` (plugin version)
- `.claude-plugin/marketplace.json` (marketplace version)

#### Git Tag and Release Strategy

Since distribution is via GitHub (not PyPI), users need a way to pin versions.

**Tag Format**: `v{MAJOR}.{MINOR}.{PATCH}` (e.g., `v0.2.0`, `v1.0.0`)

**Version Source of Truth**: The git tag is the source of truth for release versions. The version
in `pyproject.toml` and other files should match the tag being released.

**Release Workflow** (manual version bump):

```bash
# 1. Bump versions in all files
./scripts/bump-version.sh 0.3.0

# 2. Commit version changes
git add -A && git commit -m "Release v0.3.0"

# 3. Create and push tag (triggers CI)
git tag -a v0.3.0 -m "Release v0.3.0"
git push origin main --tags
```

GitHub Actions triggers on the tag push and creates the release with artifacts.

**Alternative: Tag-Derived Versioning** (automated):

For fully automated versioning, `pyproject.toml` can derive version from git tags using
[setuptools-scm](https://setuptools-scm.readthedocs.io/):

```toml
[build-system]
requires = ["setuptools>=61.0", "setuptools-scm"]

[project]
dynamic = ["version"]

[tool.setuptools_scm]
```

With this approach:

- Version is derived from the most recent git tag
- No manual version bumping needed in `pyproject.toml`
- Plugin JSON files still need manual updates (or a pre-tag hook)
- Development builds get versions like `0.2.1.dev4+g1234abc`

**Recommendation**: Start with manual version bumping for simplicity. The `bump-version.sh`
script ensures all files stay in sync. Consider setuptools-scm if release frequency increases.

**User Installation with Version Pinning**:

```bash
# Install specific version via pip
pip install "git+https://github.com/cwensel/arcaneum.git@v0.2.0"

# Install specific version via pipx
pipx install "git+https://github.com/cwensel/arcaneum.git@v0.2.0"

# Install from release artifact (wheel)
pipx install "https://github.com/cwensel/arcaneum/releases/download/v0.2.0/arcaneum-0.2.0-py3-none-any.whl"

# Homebrew (version determined by formula)
brew install cwensel/arcaneum/arcaneum
brew install cwensel/arcaneum/arcaneum@0.2  # If versioned formulas are created
```

**Plugin Version Pinning**:

Claude Code plugins installed via `/plugin install` track the default branch (main).
For version pinning, users can:

```bash
# Clone specific tag for plugin
git clone --branch v0.2.0 --depth 1 https://github.com/cwensel/arcaneum.git ~/.claude/plugins/arcaneum
```

**Branch Strategy**:

- `main`: Stable releases only, tagged versions
- `develop` (optional): Integration branch for PRs
- Feature branches: `feature/*`, `fix/*`

No `latest` tag - users should reference explicit versions or `main` for latest stable.

### User Installation Commands

```bash
# Step 1: Install CLI globally (choose one method)

# Option A: pip/pipx from GitHub (latest stable)
pip install "git+https://github.com/cwensel/arcaneum.git"
pipx install "git+https://github.com/cwensel/arcaneum.git"

# Option A (pinned version):
pip install "git+https://github.com/cwensel/arcaneum.git@v0.2.0"
pipx install "git+https://github.com/cwensel/arcaneum.git@v0.2.0"

# Option B: From release artifact (pinned)
pipx install "https://github.com/cwensel/arcaneum/releases/download/v0.2.0/arcaneum-0.2.0-py3-none-any.whl"

# Option C: Homebrew (after tap is created)
brew tap cwensel/arcaneum
brew install arcaneum

# Step 2: Install Claude plugin (in Claude Code)
/plugin install cwensel/arcaneum

# Verify installation
arc --version
arc doctor
```

**Note**: The plugin assumes `arc` is available in PATH. Slash commands like `/search` execute
`arc search $ARGUMENTS` directly via Bash. See "Git Tag and Release Strategy" for version pinning options.

## Alternatives Considered

### Alternative 1: Separate Plugin Repository

**Description**: Create a separate `cwensel/arcaneum-claude-plugin` repository containing only
the plugin files (`.claude-plugin/`, `commands/`, `skills/`).

**Pros**:

- Plugin repo is smaller (no Python source code)
- Independent release cycles possible
- Users can star/watch just the plugin

**Cons**:

- Two repositories to maintain
- Version coordination across repos is error-prone
- Plugin updates require syncing between repos
- Beads keeps plugin in main repo, not separate

**Reason for rejection**: Added maintenance overhead outweighs benefits. Plugin stays in main
repo. Users install CLI globally first, then plugin references it.

### Alternative 2: Binary Distribution with On-Demand Downloads

**Description**: Create standalone executables (via PyInstaller or Nuitka) that download ML
models on first use.

**Pros**:

- No Python required on user's system
- LLM models (~1.5GB) already downloaded on-demand

**Cons**:

- Still requires PyTorch/transformers runtime (~200MB+ in binary)
- Complex multi-platform build matrix (macOS x86/arm64, Linux)
- No automatic updates
- Long build times with PyInstaller/Nuitka
- tree-sitter native extensions complicate bundling

**Analysis of Binary Size** (based on measured `pip show` sizes and typical PyInstaller compression):

| Component                  | Installed | On-Demand? | Binary Impact                    |
| -------------------------- | --------- | ---------- | -------------------------------- |
| LLM Embedding Models       | N/A       | ✅ Yes     | Not bundled (downloaded on use)  |
| Tree-sitter Grammars       | 324 MB    | ✅ Partial | ~50MB bundled, rest on-demand    |
| PyTorch                    | 382 MB    | ❌ No      | ~150-200MB after stripping       |
| ONNX Runtime               | 128 MB    | ❌ No      | ~80MB (platform-specific)        |
| Transformers + HF          | 115 MB    | ❌ No      | ~60MB (code only, no models)     |
| OpenCV + Pillow            | 113 MB    | ❌ No      | ~50MB (native libs)              |
| PyMuPDF                    | 69 MB     | ❌ No      | ~40MB (native libs)              |
| Core Python deps           | 80 MB     | ❌ No      | ~50MB                            |
| **Estimated Binary Total** | 1.2 GB    | -          | **~400-500MB**                   |

**Breakdown by feature** (if optional deps were implemented):

| Feature Set                | Binary Size | Dependencies Included              |
| -------------------------- | ----------- | ---------------------------------- |
| Core CLI + Search          | ~300 MB     | PyTorch, ONNX, transformers, core  |
| + PDF Indexing             | +90 MB      | PyMuPDF, pdfminer, Pillow          |
| + Source Code Indexing     | +50 MB      | tree-sitter (bundled grammars)     |
| + OCR Support              | +50 MB      | OpenCV, pytesseract                |
| **Full Binary**            | ~500 MB     | All features                       |

**Reason for rejection**: Binary still 400-500MB after optimization. PyPI/pipx is simpler
and handles dependency management automatically. However, optional dependency groups
could reduce pip install size significantly for users who don't need all features.

### Alternative 3: Homebrew as Primary (Instead of Supplementary)

**Description**: Use Homebrew tap as the **primary** distribution method instead of GitHub Releases.

**Pros**:

- Familiar to macOS users
- Handles Python dependencies automatically
- Creates isolated virtualenv
- Can share large dependencies (PyTorch, numpy) with other Homebrew packages

**Cons**:

- Requires maintaining separate tap repository
- More complex formula for projects with many dependencies
- Linux support via Linuxbrew is less common than macOS
- No Windows support (acceptable given platform requirements)

**Assessment**: Homebrew is included as a **supplementary** distribution method. Making it
primary would exclude users who prefer pip/pipx workflows. The shared dependency benefit
only helps users who already have PyTorch installed via Homebrew (uncommon).

#### Homebrew External Dependencies Analysis

Homebrew has formulas for several large dependencies that could potentially reduce
distribution size:

| Homebrew Formula | Version | Can Use as Dependency? | Notes                          |
| ---------------- | ------- | ---------------------- | ------------------------------ |
| `pytorch`        | 2.9.1   | ✅ Yes                 | Installs Python bindings       |
| `numpy`          | 2.4.1   | ✅ Yes                 | Common dependency              |
| `opencv`         | 4.13.0  | ✅ Yes                 | Includes Python bindings       |
| `onnxruntime`    | 1.23.2  | ✅ Yes                 | ONNX inference runtime         |
| `tesseract`      | 5.5.2   | ✅ Yes                 | OCR engine (no Python binding) |

**How Homebrew `depends_on` works**:

A Homebrew formula can declare dependencies on other formulas:

```ruby
class Arcaneum < Formula
  depends_on "python@3.12"
  depends_on "pytorch"       # ~382 MB - not bundled, shared
  depends_on "numpy"         # ~32 MB - not bundled, shared
  depends_on "opencv"        # ~99 MB - not bundled, shared
  depends_on "onnxruntime"   # ~128 MB - not bundled, shared
```

**Potential savings**: ~640 MB of dependencies become shared system libraries rather
than bundled per-package. Users who already have these installed get instant installs.

**Caveats**:

- [Known issues](https://github.com/simonw/llm/issues/315) with PyTorch + Homebrew Python
  environment isolation
- Formula must use `depends_on` (references Homebrew formula) rather than `resource`
  (bundles PyPI package)
- Version pinning is harder - Homebrew updates dependencies independently
- Some packages (transformers, sentence-transformers) don't have Homebrew formulas

**Verdict**: Homebrew tap is included as supplementary distribution. Users who prefer
`brew install` can use it; others can use pipx with GitHub Releases. The shared dependency
benefit is theoretical - most users won't have PyTorch via Homebrew already.

### Alternative 4: Docker Image as Primary

**Description**: Publish Docker image with Arcaneum pre-installed.

**Pros**:

- Zero local dependencies
- Consistent environment
- Multi-architecture support

**Cons**:

- Requires Docker
- Large image size (~3GB with ML dependencies)
- Cannot be used as Claude Code plugin directly
- Awkward for interactive CLI use

**Reason for rejection**: Not suitable as primary CLI tool; poor Claude plugin compatibility.

## Trade-offs and Consequences

### Positive Consequences

- Users can install CLI with single command (pipx or Homebrew)
- Plugin installed separately, references global CLI (no bundled deps)
- Self-hosted distribution avoids approval processes
- Full support for target platforms (macOS, Linux)
- Proven multi-channel distribution pattern
- Single repo for CLI and plugin simplifies maintenance

### Negative Consequences

- Users must have Python 3.12+ installed
- Heavy dependencies (~1.2GB) may take time to install on first use
- LLM models downloaded on-demand add ~1.5GB on first search
- Two-step install: CLI first, then plugin

### Risks and Mitigations

- **Risk**: Plugin version mismatches with CLI version
  **Mitigation**: Version coordination script; plugin documents minimum CLI version

- **Risk**: Users install plugin without CLI
  **Mitigation**: Plugin README clearly states CLI prerequisite; slash commands fail gracefully
  with helpful error message

- **Risk**: Large dependency download times
  **Mitigation**: Document requirements clearly; consider optional dependency groups

## Implementation Plan

### Phase 1: Core Distribution (GitHub Releases)

1. Update `pyproject.toml` with URLs and keywords
2. Fix placeholder URLs in `.claude-plugin/plugin.json` and `marketplace.json`
3. Create GitHub Actions workflow for release artifacts (wheel + sdist)
4. Create `scripts/bump-version.sh` for coordinated versioning
5. Add installation documentation to README

**Plugin handling**: Plugin stays in main repo. Users run `/plugin install cwensel/arcaneum`
after installing CLI globally. Slash commands execute `arc` via Bash.

### Phase 2: Homebrew Tap (Supplementary)

1. Create `cwensel/homebrew-arcaneum` repository
2. Create Homebrew formula with Python virtualenv
3. Add workflow to auto-update formula on release
4. Test on macOS (Apple Silicon and Intel) and Linux

**Plugin handling**: Same as Phase 1. Homebrew installs CLI; plugin installed separately.

### Phase 3: Optional MCP Server (Future)

1. Create `integrations/arcaneum-mcp/` directory
2. Implement MCP server using FastMCP
3. Separate `pyproject.toml` for MCP package
4. Publish to GitHub Releases (not PyPI)

**Plugin handling**: MCP provides alternative to CLI for Claude Desktop users who
prefer typed tool interfaces over Bash execution.

### Common Tasks (All Phases)

- Sync versions: `pyproject.toml`, `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`
- Consider optional dependency groups for smaller installs (see Future Considerations)

## References

- [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [pipx Documentation](https://pipx.pypa.io/)
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [Homebrew Formula Cookbook](https://docs.brew.sh/Formula-Cookbook)
- [GitHub Actions - pypa/gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish)

## Notes

### Distribution Method Summary

| Method                 | User Experience                          | Role          | Platform Support |
| ---------------------- | ---------------------------------------- | ------------- | ---------------- |
| **GitHub Releases**    | `pipx install <release-url>`             | Primary CLI   | macOS/Linux ✅   |
| **Homebrew Tap**       | `brew install cwensel/arcaneum/arcaneum` | Supplementary | macOS/Linux ✅   |
| **Plugin (same repo)** | `/plugin install cwensel/arcaneum`       | Claude Code   | Any              |
| **MCP Server**         | `pip install arcaneum-mcp` (future)      | Optional      | Claude Desktop   |

### On-Demand Download Summary

| Component            | Download Trigger         | Size   | Cache Location        |
| -------------------- | ------------------------ | ------ | --------------------- |
| Embedding Models     | First `arc search`       | ~1.5GB | `~/.arcaneum/models/` |
| Tree-sitter Grammars | First `arc index source` | ~50MB  | System cache          |
| Qdrant/MeiliSearch   | `arc container start`    | ~500MB | Docker images         |

### Future Considerations

#### Optional Dependency Groups

Based on the dependency analysis, install size could be significantly reduced with
optional dependency groups in `pyproject.toml`:

```toml
[project.optional-dependencies]
# Current
dev = ["pytest", "black", "ruff"]
ocr = ["easyocr"]

# Proposed additions
pdf = ["pymupdf", "pymupdf4llm", "pdfplumber", "pdf2image"]
code = ["tree-sitter-language-pack", "llama-index-core", "GitPython"]
ocr-full = ["easyocr", "pytesseract", "opencv-python-headless"]
all = ["arcaneum[pdf,code,ocr-full]"]
```

**Estimated install sizes with optional groups**:

| Installation Command             | Size    | Features               |
| -------------------------------- | ------- | ---------------------- |
| `pipx install arcaneum`          | ~700 MB | Search only (core)     |
| `pipx install 'arcaneum[pdf]'`   | ~800 MB | + PDF indexing         |
| `pipx install 'arcaneum[code]'`  | ~1.0 GB | + Source code indexing |
| `pipx install 'arcaneum[all]'`   | ~1.2 GB | All features           |

#### MCP Server (Optional Integration)

An optional MCP server could be added in `integrations/arcaneum-mcp/`:

- Provides typed tool interfaces for Claude Desktop users
- Higher token overhead than CLI (~10-50k vs 1-2k tokens)
- Not required for Claude Code (which has shell access)
- Would use FastMCP framework

#### Other Considerations

- Evaluate conda-forge submission for scientific computing users
- Consider lazy-loading PyTorch to reduce CLI startup time
- Investigate torch-cpu vs torch-gpu optional variants

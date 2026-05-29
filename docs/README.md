# Arcaneum Documentation

## Quick Start

- **[Quick Start Guide](guides/quickstart.md)** - Complete walkthrough with troubleshooting (5-10 minutes)

## User Guides

- **[Arc CLI Reference](guides/cli-reference.md)** - Complete CLI command reference
- **[PDF Indexing Guide](guides/pdf-indexing.md)** - Detailed PDF indexing with OCR
- **[Claude Code Plugin Guide](guides/claude-code-plugin.md)** - Plugin testing and command reference
- **[Qdrant Migration Guide](guides/qdrant-migration.md)** - Move legacy bind mounts to Docker named volumes
- **[Persisted Schema Guide](guides/persisted-schema.md)** - Persisted payload schema policy

## Testing

- **[Test Commands](testing/test-commands.md)** - Quick copy/paste test commands
- **[Testing Guide](testing/testing.md)** - Step-by-step testing instructions
- **[Corporate Proxy](testing/corporate-proxy.md)** - SSL certificate workaround
- **[Offline Mode](testing/offline-mode.md)** - Cached-model setup for restricted networks
- **[Source Code Indexing Tests](testing/source-code-indexing.md)** - Source indexing test scenarios
- **[Unit Tests](testing/unit-tests.md)** - Unit-test notes and conventions

## Technical Specifications

- **[RDR Directory](rdr/)** - Recommendation Data Records (detailed technical specs)
  - [RDR-001](rdr/RDR-001-project-structure.md) - Project structure
  - [RDR-002](rdr/RDR-002-qdrant-server-setup.md) - Qdrant setup
  - [RDR-003](rdr/RDR-003-collection-creation.md) - Collection management
  - [RDR-004](rdr/RDR-004-pdf-bulk-indexing.md) - PDF indexing (this implementation)
  - [RDR-008](rdr/RDR-008-fulltext-search-server-setup.md) - MeiliSearch setup
  - [RDR-009](rdr/RDR-009-dual-indexing-strategy.md) - Dual-index corpora
  - See [rdr/README.md](rdr/README.md) for complete list
- **[Arcaneum 1.0 Release and Compatibility Policy](reference/release-compatibility-policy.md)** -
  release-readiness gates and the post-1.0 compatibility contract

## Directory Structure

```text
docs/
├── README.md              # This file
├── guides/                # User-facing documentation
│   ├── quickstart.md
│   ├── cli-reference.md
│   ├── pdf-indexing.md
│   ├── claude-code-plugin.md
│   ├── persisted-schema.md
│   └── qdrant-migration.md
├── testing/               # Testing documentation
│   ├── test-commands.md
│   ├── testing.md
│   ├── corporate-proxy.md
│   ├── source-code-indexing.md
│   ├── unit-tests.md
│   └── offline-mode.md
├── reference/             # Technical reference
│   └── release-compatibility-policy.md
└── rdr/                   # Technical specifications
    ├── README.md
    ├── TEMPLATE.md
    └── RDR-*.md
```

## For Developers

See individual RDRs in `rdr/` directory for detailed technical specifications and implementation rationale.

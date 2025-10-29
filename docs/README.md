# Arcaneum Documentation

## Quick Start

- **[Quick Start Guide](guides/quickstart.md)** - Complete walkthrough with troubleshooting (5-10 minutes)

## User Guides

- **[Arc CLI Reference](guides/arc-cli-reference.md)** - Complete CLI command reference
- **[PDF Indexing Guide](guides/pdf-indexing.md)** - Detailed PDF indexing with OCR

## Testing

- **[Test Commands](testing/test-commands.md)** - Quick copy/paste test commands
- **[Testing Guide](testing/testing.md)** - Step-by-step testing instructions
- **[Corporate Proxy](testing/corporate-proxy.md)** - SSL certificate workaround

## Technical Specifications

- **[RDR Directory](rdr/)** - Recommendation Data Records (detailed technical specs)
  - [RDR-001](rdr/RDR-001-project-structure.md) - Project structure
  - [RDR-002](rdr/RDR-002-qdrant-docker-compose.md) - Qdrant setup
  - [RDR-003](rdr/RDR-003-qdrant-collection-creation-cli.md) - Collection management
  - [RDR-004](rdr/RDR-004-pdf-bulk-indexing.md) - PDF indexing (this implementation)
  - See [rdr/README.md](rdr/README.md) for complete list

## Directory Structure

```
docs/
├── README.md              # This file
├── guides/                # User-facing documentation
│   ├── quickstart.md
│   ├── arc-cli-reference.md
│   └── pdf-indexing.md
├── testing/               # Testing documentation
│   ├── test-commands.md
│   ├── testing.md
│   └── corporate-proxy.md
└── rdr/                   # Technical specifications
    ├── README.md
    ├── TEMPLATE.md
    └── RDR-*.md
```

## For Developers

See individual RDRs in `rdr/` directory for detailed technical specifications and implementation rationale.

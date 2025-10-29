---
description: Index PDF files to Qdrant collection
argument-hint: <path> --collection <name> [options]
---

Index PDF files from a directory to a Qdrant collection for semantic search.

**Arguments:**

- <path>: Directory containing PDF files
- --collection <name>: Target Qdrant collection name (required)
- --model <model>: Embedding model (default: stella)
- --workers <n>: Parallel workers (default: 4)
- --no-ocr: Disable OCR (enabled by default for scanned documents)
- --ocr-language <lang>: OCR language code (default: eng)
- --force: Force reindex all files
- --verbose: Detailed progress output
- --json: Output JSON format

**Examples:**

```text
/index-pdfs /Documents/papers --collection Research
/index-pdfs /Scans --collection Archive --ocr-language fra
/index-pdfs /Books --collection Library --workers 8 --force
```

**Execution:**

```bash
cd ${CLAUDE_PLUGIN_ROOT}
arc index-pdfs $ARGUMENTS
```

**Note:** This command may take several minutes for large document collections.
I'll monitor the progress in real-time and show you:

- How many PDF files are discovered
- Processing status for each file with percentage completion
- OCR status for scanned documents
- Final summary with total files indexed and chunks created
- Any errors encountered during processing

Full implementation in RDR-004.

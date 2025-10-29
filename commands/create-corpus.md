---
description: Create both Qdrant collection and MeiliSearch index
argument-hint: <name> --type <source-code|pdf> [options]
---

Create a corpus (both Qdrant collection and MeiliSearch index) for dual search capabilities.

**Arguments:**

- <name>: Corpus name (required)
- --type <type>: Corpus type (source-code or pdf) (required)
- --models <models>: Embedding models, comma-separated (default: stella,jina)
- --json: Output JSON format

**Examples:**

```text
/create-corpus MyCode --type source-code
/create-corpus Research --type pdf --models stella,modernbert
```

**Execution:**

```bash
cd ${CLAUDE_PLUGIN_ROOT}
arc create-corpus $ARGUMENTS
```

**Note:** This command creates both Qdrant collection and MeiliSearch index for
dual search capabilities (semantic + full-text). I'll confirm both stores are
created successfully and show you the configuration for each.

**Next Step:** Use /sync-directory to index documents to this corpus.
Full implementation in RDR-009.

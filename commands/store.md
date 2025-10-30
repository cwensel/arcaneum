---
description: Store agent-generated content for long-term memory
argument-hint: <file|-for-stdin> --collection <name> [options]
---

Store agent-generated content (research, analysis, synthesized information) with rich
metadata. Content is persisted to disk for re-indexing and full-text retrieval, then
indexed to Qdrant for semantic search.

**Storage Location:** `~/.arcaneum/agent-memory/{collection}/`

**Options:**

- --collection: Target collection (required)
- --model: Embedding model (default: stella for documents)
- --title: Document title (added to frontmatter)
- --category: Document category (e.g., research, security, analysis)
- --tags: Comma-separated tags
- --metadata: Additional metadata as JSON
- --chunk-size: Target chunk size in tokens (overrides model default)
- --chunk-overlap: Overlap between chunks in tokens
- --verbose: Show detailed progress
- --json: Output in JSON format

**Examples:**

```text
/store analysis.md --collection Memory --title "Security Analysis" --category security
/store - --collection Research --title "Findings" --tags "research,important"
```

**Execution:**

```bash
cd ${CLAUDE_PLUGIN_ROOT}
arc store $ARGUMENTS
```

**How It Works:**

1. Accept content from file or stdin (`-`)
2. Extract/add rich metadata (title, category, tags, custom fields)
3. Semantic chunking preserving document structure
4. Generate embeddings (stella default: 1024D for documents)
5. Upload to Qdrant with metadata
6. Persist to disk: `~/.arcaneum/agent-memory/{collection}/{date}_{agent}_{slug}.md`
7. Generate YAML frontmatter with injection metadata (injection_id, injected_at, injected_by)

**Persistence:**

Content is always persisted for durability. This enables:

- Re-indexing: Update embeddings without losing original content
- Full-text retrieval: Access complete original documents
- Audit trail: Track what was stored and when (injection_id, timestamps)

**Filename Format:**

`YYYYMMDD_agent_slug.md` (e.g., `20251030_claude_security-analysis.md`)

**Use Cases:**

- AI agents storing research findings
- Preserving analysis results
- Collecting synthesized information
- Building knowledge bases from agent workflows

**Default Model:**

- stella (1024D, document-optimized)

**Related Commands:**

- /collection create - Create collection before storing (use --type markdown)
- /search semantic - Search stored content
- /index markdown - For indexing existing markdown directories (different use case)

**Implementation:**

- RDR-014: Markdown content indexing
- arcaneum-204: Direct injection persistence module

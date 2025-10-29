# Collections, Corpus, and Types in Arcaneum

## Overview

Arcaneum uses a **typed collection system** to ensure PDFs and source code are properly separated and searched.

## Key Concepts

### Collection

**Definition:** A typed vector database in Qdrant for semantic search

**Characteristics:**
- Stores embeddings and metadata
- **Must be typed** (pdf or code)
- Single search system (Qdrant only)
- Names are flexible, but types are enforced

**Creation:**
```bash
# PDF collection
arc create-collection docs --model stella --type pdf

# Source code collection
arc create-collection code --model stella --type code

# Names can be anything, but type is required
arc create-collection my-work-docs --model stella --type pdf
arc create-collection personal-code --model stella --type code
```

### Corpus (RDR-009)

**Definition:** A unified searchable dataset across BOTH Qdrant AND MeiliSearch

**Characteristics:**
- Dual indexing: Qdrant (semantic) + MeiliSearch (full-text)
- **Inherits type** from underlying collection
- Same name used for both systems
- Enables hybrid search workflows

**Creation:**
```bash
# Creates BOTH Qdrant collection AND MeiliSearch index
arc create-corpus docs --type pdf
# ├─> Qdrant collection 'docs' (type=pdf)
# └─> MeiliSearch index 'docs'

arc create-corpus code --type code
# ├─> Qdrant collection 'code' (type=code)
# └─> MeiliSearch index 'code'
```

## Type System

### Collection Types

| Type | Content | Commands |
|------|---------|----------|
| `pdf` | PDF documents, scanned docs | `index-pdfs` |
| `code` | Source code (15+ languages) | `index-source` |

### Type Enforcement

**At Creation:**
```bash
arc create-collection docs --model stella --type pdf
# Stores: {"collection_type": "pdf", ...}
```

**At Indexing:**
```bash
# ✓ Valid: type matches
arc index-pdfs ~/docs --collection docs
arc index-source ~/projects --collection code

# ✗ Invalid: type mismatch
arc index-pdfs ~/docs --collection code
# Error: Collection 'code' is type 'code', cannot index PDFs

arc index-source ~/projects --collection docs
# Error: Collection 'docs' is type 'pdf', cannot index source code
```

### Metadata Schema

**Collection-Level Metadata** (stored in Qdrant):
```python
{
  "collection_type": "pdf",      # or "code" - ENFORCED
  "model": "stella",
  "created_at": "2025-10-27",
  "created_by": "arcaneum"
}
```

**Point-Level Metadata** (each chunk):
```python
{
  "store_type": "pdf",           # or "source-code" - DESCRIPTIVE
  "filename": "document.pdf",
  "file_path": "/path/to/file",
  # ... plus type-specific fields
}
```

**Key Difference:**
- `collection_type`: Collection-level, **enforces** what can be indexed
- `store_type`: Point-level, **describes** what was indexed

## Relationship Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   COLLECTION (typed)                     │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Collection Metadata                                 │ │
│  │ - collection_type: "pdf" or "code" (ENFORCED)      │ │
│  │ - model: "stella"                                  │ │
│  │ - created_at: "2025-10-27"                         │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Points (chunks)                                     │ │
│  │ - store_type: "pdf" or "source-code" (DESCRIBES)   │ │
│  │ - embeddings (vectors)                             │ │
│  │ - content metadata                                 │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                CORPUS (typed, dual-system)               │
│  ┌────────────────────────────────────────────────────┐ │
│  │ Qdrant Collection (semantic search)                │ │
│  │ - collection_type enforced                         │ │
│  │ - embeddings for similarity                        │ │
│  └────────────────────────────────────────────────────┘ │
│                          +                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │ MeiliSearch Index (full-text search)               │ │
│  │ - same type as collection                          │ │
│  │ - text for exact matching                          │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Usage Patterns

### Pattern 1: Simple Semantic Search (Collection Only)

```bash
# Create typed collection
arc create-collection code --model stella --type code

# Index source code
arc index-source ~/projects --collection code

# Search (RDR-007)
arc search "authentication" --collection code
```

**Use when:** You only need semantic similarity search

### Pattern 2: Hybrid Search (Corpus)

```bash
# Create typed corpus (both Qdrant + MeiliSearch)
arc create-corpus code --type code

# Sync directory (indexes to both)
arc sync-directory ~/projects --corpus code

# Semantic search (Qdrant)
arc search "authentication" --collection code

# Full-text search (MeiliSearch)
arc search-text "def authenticate" --index code
```

**Use when:** You need both semantic similarity AND exact phrase matching

## Examples

### Separate Collections for Different Content

```bash
# Personal documents (PDFs)
arc create-collection personal-docs --model stella --type pdf
arc index-pdfs ~/Documents --collection personal-docs

# Work documents (PDFs)
arc create-collection work-docs --model bge --type pdf
arc index-pdfs ~/Work/PDFs --collection work-docs

# Personal code
arc create-collection personal-code --model stella --type code
arc index-source ~/Code --collection personal-code

# Work code
arc create-collection work-code --model stella --type code
arc index-source ~/Work/Projects --collection work-code
```

### Multi-Branch Code Collections

```bash
# Single collection can have multiple branches
arc create-collection my-app --model stella --type code

# Index main branch
cd ~/my-app
git checkout main
arc index-source ~/my-app --collection my-app
# Stores: my-app#main

# Index feature branch
git checkout feature-auth
arc index-source ~/my-app --collection my-app
# Stores: my-app#feature-auth

# Both branches coexist in same collection
# Query by branch using git_project_identifier filter
```

## Validation Examples

### Successful Operations

```bash
✓ arc create-collection docs --model stella --type pdf
✓ arc index-pdfs ~/documents --collection docs
✓ arc create-collection code --model stella --type code
✓ arc index-source ~/projects --collection code
```

### Type Mismatch Errors

```bash
# Create PDF collection
$ arc create-collection docs --model stella --type pdf

# Try to index code into it
$ arc index-source ~/projects --collection docs
❌ Error: Collection 'docs' is type 'pdf', cannot index source code content.
   Create a new collection with --type code.

# Create code collection
$ arc create-collection code --model stella --type code

# Try to index PDFs into it
$ arc index-pdfs ~/documents --collection code
❌ Error: Collection 'code' is type 'code', cannot index pdf content.
   Create a new collection with --type pdf.
```

### Untyped Collections (Backward Compatibility)

```bash
# Old collection without type
$ arc create-collection old-collection --model stella
# (no --type flag used)

# First index operation succeeds with warning
$ arc index-source ~/projects --collection old-collection
⚠️  Warning: Collection 'old-collection' has no type. Allowing code indexing.
    Consider recreating with --type flag.
✓ Indexed successfully

# Subsequent operations continue to work
# But best practice: recreate with type
```

## Best Practices

1. **Always specify --type** when creating collections
2. **Use separate collections** for PDFs and code
3. **Name collections clearly**: `docs`, `code`, `work-docs`, `my-code`
4. **Check collection info** before indexing: `arc collection-info <name>`
5. **Use corpus** for hybrid search, **collection** for semantic-only

## Terminology Summary

| Term | Meaning | Example |
|------|---------|---------|
| **Collection** | Typed Qdrant vector database | `arc create-collection code --type code` |
| **Corpus** | Dual-indexed (Qdrant + MeiliSearch) | `arc create-corpus docs --type pdf` |
| **Collection Type** | Enforced at creation (pdf/code) | `--type code` |
| **Store Type** | Metadata describing indexed content | `store_type: "source-code"` |
| **Collection Name** | User-chosen name (flexible) | `docs`, `code`, `my-work` |

## See Also

- **RDR-003**: Collection creation and management
- **RDR-004**: PDF indexing
- **RDR-005**: Source code indexing
- **RDR-009**: Corpus (dual indexing) workflow
- **[Source Code Indexing Testing](../testing/source-code-indexing.md)**: Testing guide

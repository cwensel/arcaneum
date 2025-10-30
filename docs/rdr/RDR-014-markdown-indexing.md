# Recommendation 014: Markdown Content Indexing with Directory Sync and Direct Injection

## Metadata

- **Date**: 2025-10-30
- **Status**: Implemented
- **Implemented**: 2025-10-30
- **Type**: Feature
- **Priority**: High
- **Related Issues**: arcaneum-199, arcaneum-204, arcaneum-211, arcaneum-212, arcaneum-213
- **Related Tests**: tests/unit/indexing/markdown/test_chunker.py

## Problem Statement

Create a markdown indexing system for Arcaneum that supports two distinct usage patterns:

1. **Directory Sync Mode**: Index entire directories of markdown files (documentation, notes,
   research) similar to how we handle PDFs (RDR-004) and source code (RDR-005). Support
   incremental updates via change detection.

2. **Direct Injection Mode**: Allow Claude (or other AI agents) to directly inject markdown
   content into a collection as long-term memory storage. This enables agents to store research
   results, summaries, or synthesized information. Injected content must be persisted to a local
   directory so it can be re-indexed if the collection is recreated.

The system must:

- Implement markdown-specific chunking that respects document structure (headers, code blocks, lists)
- Track rich metadata (creation date, source, tags, frontmatter)
- Persist directly-injected content to local storage for durability
- Integrate with existing collection/corpus architecture
- Support both Qdrant (semantic) and MeiliSearch (full-text) indexing
- Provide clean CLI interface for both modes

This addresses the need to index markdown content for knowledge management, documentation search, and AI agent memory systems.

## Context

### Background

Markdown is a critical content format for Arcaneum because:

- **Documentation**: Project READMEs, wikis, technical documentation
- **Knowledge Management**: Personal notes, research summaries, meeting notes
- **AI Agent Memory**: Synthesized information, research results, learned patterns
- **Research Artifacts**: RDRs, investigation notes, architecture decisions

**Two Distinct Usage Patterns**:

1. **Directory Sync**: User has existing markdown files (e.g., Obsidian vault, project docs)
   - Similar to PDF/source code indexing
   - File-based, tracked via file paths
   - Change detection via content hashing (SHA256)
   - Example: `arc index markdown ~/Documents/Notes --collection knowledge`

2. **Direct Injection**: Agent generates markdown during operation
   - Programmatic, created by AI during research/analysis
   - Must persist to disk for durability (can't rely on ephemeral memory)
   - Example: Agent researches "GPU acceleration patterns" → stores summary as markdown
   - Enables re-indexing if collection is deleted/recreated

**Design Questions**:

- How to chunk markdown preserving semantic boundaries?
- What metadata should be extracted from frontmatter?
- Where to persist directly-injected content?
- How to track injection source (agent name, timestamp, tags)?
- Should we support both collection (semantic-only) and corpus (dual-index)?
- How to handle markdown with embedded code blocks?

### Technical Environment

- **Python**: >= 3.12
- **Qdrant**: v1.15.4+ (from RDR-002)
- **MeiliSearch**: v1.24.0+ (from RDR-008)
- **Markdown Parsing**:
  - markdown-it-py >= 4.0.0 (parsing with structure awareness)
  - python-frontmatter >= 1.1.0 (YAML frontmatter extraction)
  - pygments >= 2.19.0 (code block language detection)
- **Embedding** (already present in project):
  - FastEmbed >= 0.7.3 (ONNX-based models)
  - sentence-transformers >= 3.3.1 (PyTorch-based models, MPS GPU support)
  - qdrant-client[fastembed] >= 1.15.0
- **Supporting Libraries**:
  - tenacity (retry logic)
  - rich (progress tracking)
  - llama-index-core >= 0.14.6 (already present, includes MarkdownNodeParser)

**Target Embedding Models** (verified from RDR-013):

- **stella_en_1.5B_v5**: 1024D, 512-1024 token chunks (recommended for general text)
  - Model: dunzhang/stella_en_1.5B_v5
  - Backend: sentence-transformers (MPS GPU support on Apple Silicon)
  - Status: ✅ Tested and working with GPU acceleration
- **jina-embeddings-v2-base-en**: 768D, 8K token context (alternative)
  - Backend: FastEmbed (ONNX Runtime)
- **bge-base-en-v1.5**: 768D, 512 token context (alternative)
  - Backend: FastEmbed (CoreML acceleration on Apple Silicon)

**Collection Type**: New `markdown` type (follows RDR-003 collection typing pattern)

## Research Findings

### Investigation Process

**Research completed** to inform this RDR:

1. **Markdown Parsing Libraries**: Analysis of markdown-it-py vs mistune vs commonmark
2. **Chunking Strategies**: Research on semantic chunking for markdown (headers, sections)
3. **Frontmatter Standards**: Investigation of YAML/TOML frontmatter conventions
4. **Persistence Patterns**: Analysis of storage patterns for agent-generated content
5. **Existing Patterns**: Review of RDR-004 (PDF), RDR-005 (source code), RDR-009 (corpus) for architectural consistency

### Key Discoveries

#### 1. Markdown-Specific Chunking Strategy

**Challenge**: Traditional token-based chunking breaks markdown semantics

**Semantic Chunking Approach**:

```text
Document Structure:
# Title (H1)
## Section 1 (H2)
### Subsection 1.1 (H3)
Content...
### Subsection 1.2 (H3)
Content...
## Section 2 (H2)
Content...

Chunking Strategy:
- Chunk 0: Title + Section 1 + Subsection 1.1 (if fits in token budget)
- Chunk 1: Subsection 1.2 (preserve heading context)
- Chunk 2: Section 2 (with heading)
```

**Key Principles**:

1. **Respect Headers**: Never split mid-section without preserving header context
2. **Code Block Integrity**: Keep code blocks intact (never split)
3. **List Coherence**: Keep list items together when possible
4. **Context Preservation**: Include parent headers in chunks for context

**Algorithm**:

```python
def chunk_markdown(doc: str, max_tokens: int = 512) -> List[Chunk]:
    """Semantic markdown chunking."""
    # 1. Parse into AST (heading, paragraph, code_block, list nodes)
    ast = parse_markdown(doc)

    # 2. Build section tree (hierarchical by header levels)
    sections = build_section_tree(ast)

    # 3. Chunk sections, preserving boundaries
    chunks = []
    for section in sections:
        if section.token_count <= max_tokens:
            # Entire section fits
            chunks.append(Chunk(
                content=section.full_text,
                headers=section.header_path,  # ["Title", "Section 1", "Subsection 1.1"]
                type="section"
            ))
        else:
            # Split section, preserving context
            sub_chunks = split_section(section, max_tokens)
            for sub_chunk in sub_chunks:
                # Include parent headers for context
                chunk_content = "\n".join(section.header_path) + "\n" + sub_chunk.text
                chunks.append(Chunk(
                    content=chunk_content,
                    headers=section.header_path,
                    type="subsection"
                ))

    return chunks
```

**Proven Results** (from ChunkHound research):

- 35% better retrieval accuracy for markdown vs naive token splitting
- Preserves semantic relationships between sections

#### 2. Frontmatter Metadata Extraction

**Standard Format** (YAML frontmatter):

```markdown
---
title: GPU Acceleration Patterns
date: 2025-10-30
tags: [gpu, performance, optimization]
author: Claude
source: research_session
project: arcaneum
---

# Content starts here
```

**Metadata Schema**:

```python
@dataclass
class MarkdownMetadata:
    # File-based fields (directory sync mode)
    file_path: Optional[str] = None  # Absolute path (if from file)
    filename: Optional[str] = None   # Basename
    file_size: Optional[int] = None  # Bytes
    file_hash: Optional[str] = None  # SHA256 hash (first 12 chars) for change detection

    # Injection fields (direct injection mode)
    injection_id: Optional[str] = None  # Unique ID for injected content
    injected_by: Optional[str] = None   # Agent name (e.g., "Claude", "GPT-4")
    injected_at: Optional[datetime] = None  # Creation timestamp

    # Frontmatter fields (extracted from YAML)
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    project: Optional[str] = None
    source: Optional[str] = None  # e.g., "research_session", "meeting_notes"

    # Content fields
    word_count: int = 0
    chunk_index: int = 0
    chunk_count: int = 0
    header_path: List[str] = field(default_factory=list)  # ["Section", "Subsection"]
    has_code_blocks: bool = False
    has_tables: bool = False
    has_lists: bool = False

    # Collection metadata
    embedding_model: str = "stella_en_1.5B_v5"
    store_type: str = "markdown"
```

#### 3. Direct Injection Persistence Pattern

**Storage Structure** (organized by collection):

```text
~/.arcaneum/agent-memory/
  knowledge/                              # Collection: knowledge
    2025-10-30_claude_gpu-acceleration.md
    2025-10-30_claude_vector-search-patterns.md
  memory/                                 # Collection: memory
    2025-10-30_gpt4_security-analysis.md
    2025-10-31_claude_api-patterns.md
  research/                               # Collection: research
    2025-10-29_claude_embedding-comparison.md
```

**Benefits**:

- Clear organization by collection/corpus
- Easy to find injected content for specific collection
- Can backup/restore collection-specific injections
- No intermingling of unrelated content

**Filename Convention**:

```text
{date}_{agent-name}_{slug}.md

Examples:
- 2025-10-30_claude_gpu-acceleration.md
- 2025-10-30_gpt4_api-design-patterns.md
```

**Standardized Frontmatter for Agent Memory**:

```yaml
---
# System-managed fields (auto-populated by injection handler)
injection_id: 5f3a2c8b-12ef-4a5d-9b1c-8d4e7f1a2c3d
injected_by: Claude
injected_at: 2025-10-30T14:23:45Z

# Required content fields
title: GPU Acceleration Patterns
category: research
tags: [gpu, performance, metal, cuda, vector-search]

# Optional organizational fields
project: arcaneum
collection: memory

# Optional context fields
source: research
context: Investigation of GPU acceleration for embedding generation
related_to: [arcaneum-183, arcaneum-199]

# Optional priority/status
priority: high
status: reviewed
---

# GPU Acceleration Patterns

Research findings on GPU acceleration...
```

**Field Definitions**:

**System-Managed (auto-populated)**:

- `injection_id`: UUID for tracking (generated automatically)
- `injected_by`: Agent name (e.g., "Claude")
- `injected_at`: ISO timestamp (auto-generated)

**Required Fields**:

- `title`: Document title (extracted from H1 if not provided)
- `category`: Primary category for classification
- `tags`: List of searchable keywords

**Optional Organizational**:

- `project`: Project identifier (e.g., "arcaneum")
- `collection`: Target collection name
- `source`: Source type (e.g., "research", "conversation", "analysis")
- `context`: Brief description of why/when created
- `related_to`: Links to beads issues or related documents

**Optional Metadata**:

- `priority`: Importance level (low, medium, high, critical)
- `status`: Document lifecycle state (draft, reviewed, final, archived)

**Category Guidelines**:

- `research`: Research findings, technology surveys, literature reviews
- `security-analysis`: Security assessments, vulnerability analyses
- `api-design`: API patterns, design decisions
- `architecture`: System design, architectural decisions
- `performance`: Performance analysis, optimization patterns
- `debugging`: Debug sessions, root cause analyses
- `meeting-notes`: Meeting summaries, action items
- `reference`: Reference materials, cheat sheets

**Tag Best Practices**:

- Use lowercase, hyphenated tags
- Include technology names (e.g., "qdrant", "python", "metal")
- Include domain concepts (e.g., "vector-search", "gpu-acceleration")
- Include project-specific terms
- 3-7 tags recommended for optimal searchability

**Prompt for AI Agents**:

When injecting content to agent-memory, use this template:

```yaml
---
# System fields (auto-populated - do not set manually)
injection_id: <auto-generated>
injected_by: <agent-name>
injected_at: <auto-generated>

# Required fields
title: <Clear, descriptive title>
category: <primary-category>
tags: [<keyword1>, <keyword2>, <keyword3>]

# Organizational (use when applicable)
project: <project-name>
collection: <collection-name>

# Context (recommended)
source: <source-type>
context: <brief description of why this was created>
related_to: [<issue-id1>, <issue-id2>]

# Optional metadata
priority: <low|medium|high|critical>
status: <draft|reviewed|final|archived>
---

# Your Title

Content here...
```

**Benefits**:

- **Durability**: Content persists across collection deletions
- **Re-indexing**: Can rebuild collection from stored files
- **Auditability**: Track what agents injected and when
- **Portability**: Standard markdown files, readable without Arcaneum
- **Searchability**: Rich metadata enables sophisticated filtering

#### 4. Integration with Existing Architecture

**Collection Type** (follows RDR-003 pattern):

```bash
# Create markdown collection
arc collection create notes --model stella --type markdown

# Validation: prevents indexing wrong content type
arc index pdfs ~/documents --collection notes
# ❌ Error: Collection 'notes' is type 'markdown', cannot index pdf content
```

**Corpus Support** (follows RDR-009 pattern):

```bash
# Create corpus (both Qdrant + MeiliSearch)
arc corpus create knowledge --type markdown

# Sync directory (dual-index)
arc corpus sync ~/Documents/Notes --corpus knowledge

# Search both ways
arc search "GPU acceleration" --collection knowledge  # Semantic
arc search text "Metal API" --index knowledge         # Exact
```

#### 5. Change Detection for Directory Sync

**Pattern** (follows RDR-004 metadata-based sync with content hashing):

```python
import hashlib
from pathlib import Path

def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file content (first 12 chars).

    Args:
        file_path: Path to file

    Returns:
        First 12 characters of SHA256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:12]


class MarkdownMetadataSync:
    """Query Qdrant for indexed markdown files (source of truth)."""

    def get_indexed_files(self, collection_name: str) -> Set[tuple]:
        """Get all (file_path, file_hash) pairs from Qdrant.

        Returns:
            Set of (file_path, file_hash) tuples for fast lookup
        """
        indexed = set()
        offset = None

        while True:
            points, offset = self.qdrant.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=["file_path", "file_hash"],
                with_vectors=False
            )

            if not points:
                break

            for point in points:
                if point.payload:
                    path = point.payload.get("file_path")
                    hash_val = point.payload.get("file_hash")
                    if path and hash_val:
                        indexed.add((path, hash_val))

            if offset is None:
                break

        return indexed

    def get_unindexed_files(self, collection_name: str,
                           file_list: List[Path]) -> List[Path]:
        """Filter file list to only unindexed or modified files.

        Computes hash for each file and checks against indexed set.

        Args:
            collection_name: Qdrant collection name
            file_list: List of markdown file paths

        Returns:
            List of files that need indexing (new or modified)
        """
        # Get all indexed (path, hash) pairs
        indexed = self.get_indexed_files(collection_name)

        # Filter to files not in indexed set
        unindexed = []
        for file_path in file_list:
            file_hash = compute_file_hash(file_path)
            if (str(file_path), file_hash) not in indexed:
                unindexed.append(file_path)

        logger.info(f"Found {len(unindexed)}/{len(file_list)} files to index")
        return unindexed
```

**Deletion Strategy**:

```python
# Delete all chunks for a specific file (filter-based, 50-500ms)
client.delete(
    collection_name="notes",
    points_selector=Filter(
        must=[FieldCondition(
            key="file_path",
            match=MatchValue("/path/to/file.md")
        )]
    )
)
```

**Benefits of Content Hashing**:

- Detects actual content changes (not just timestamp updates)
- Reliable across file copies, syncs, checkouts
- Matches PDF indexing implementation (architectural consistency)

#### 6. Code Block Handling

**Challenge**: Markdown often contains code blocks in multiple languages

**Solution**: Detect language and preserve syntax

```python
def extract_code_blocks(markdown: str) -> List[CodeBlock]:
    """Extract code blocks with language tags."""
    # Parse markdown AST
    tokens = markdown_parser.parse(markdown)

    code_blocks = []
    for token in tokens:
        if token.type == "fence" or token.type == "code_block":
            language = token.info.strip() or "text"  # e.g., "python", "bash"
            code = token.content
            code_blocks.append(CodeBlock(
                language=language,
                code=code,
                line_number=token.map[0] if token.map else None
            ))

    return code_blocks

# Store in metadata
metadata.has_code_blocks = len(code_blocks) > 0
metadata.code_languages = [b.language for b in code_blocks]
```

**Benefits**:

- Enables filtering: "show markdown chunks with Python code"
- Preserves syntax for display
- Can optionally index code separately (future enhancement)

## Proposed Solution

### Approach

Implement a **markdown indexing pipeline with dual-mode support**:

1. **Directory Sync Mode**: Discover markdown files recursively, extract frontmatter, chunk
   semantically, track changes via content hashing
2. **Direct Injection Mode**: Accept markdown content programmatically, persist to storage
   directory, generate metadata
3. **Semantic Chunking**: Parse markdown AST, respect headers/code blocks/lists
4. **Metadata-Based Sync**: Query Qdrant for indexed files (source of truth)
5. **Collection Type**: New `markdown` type alongside `pdf` and `code`
6. **Corpus Support**: Optional dual-indexing to Qdrant + MeiliSearch

### Technical Design

#### Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│              Markdown Indexing Pipeline                      │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
     Directory Sync Mode         Direct Injection Mode
     ───────────────────         ────────────────────
            │                               │
    discover markdown files       accept content + metadata
    + extract frontmatter         + generate injection ID
            │                     + persist to storage dir
            └───────────────┬─────────────┘
                            │
                   ┌────────▼─────────┐
                   │  Parse Markdown  │
                   │  Extract AST     │
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────────────┐
                   │ Semantic Chunking        │
                   │ (respect headers, code)  │
                   └────────┬─────────────────┘
                            │
                   ┌────────▼─────────────────┐
                   │ Embedding Generation     │
                   │ (FastEmbed + stella/jina)│
                   └────────┬─────────────────┘
                            │
                   ┌────────▼─────────────────┐
                   │ Upload to Qdrant         │
                   │ (+ MeiliSearch if corpus)│
                   └──────────────────────────┘
```

#### Core Components

**1. Markdown Discovery and Parsing**

```python
# src/arcaneum/indexing/markdown/discovery.py

import frontmatter
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

class MarkdownDiscovery:
    """Discover and parse markdown files."""

    def discover_files(self, input_path: str,
                      recursive: bool = True,
                      exclude_patterns: List[str] = None) -> List[Path]:
        """Find all markdown files in directory.

        Args:
            input_path: Directory to search
            recursive: Search subdirectories
            exclude_patterns: Patterns to exclude (e.g., ["node_modules", ".git"])

        Returns:
            List of markdown file paths
        """
        path = Path(input_path)

        if not path.is_dir():
            raise ValueError(f"Not a directory: {input_path}")

        # Find markdown files
        if recursive:
            files = list(path.rglob("*.md"))
        else:
            files = list(path.glob("*.md"))

        # Apply exclusions
        if exclude_patterns:
            files = [f for f in files
                    if not any(pattern in str(f) for pattern in exclude_patterns)]

        return sorted(files)

    def parse_frontmatter(self, file_path: Path) -> Dict:
        """Extract YAML frontmatter from markdown file.

        Returns dict with:
        - metadata: Parsed frontmatter (dict)
        - content: Markdown content without frontmatter (str)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)

        return {
            'metadata': dict(post.metadata),
            'content': post.content
        }
```

**2. Semantic Markdown Chunker**

```python
# src/arcaneum/indexing/markdown/chunker.py

from markdown_it import MarkdownIt
from typing import List
from dataclasses import dataclass

@dataclass
class MarkdownChunk:
    """A semantically-chunked markdown section."""
    content: str  # Full text including headers
    headers: List[str]  # Header path ["Section", "Subsection"]
    chunk_type: str  # "section", "subsection", "paragraph"
    start_line: int
    end_line: int
    has_code_blocks: bool = False
    has_tables: bool = False
    has_lists: bool = False

class MarkdownChunker:
    """Semantic chunking for markdown documents."""

    def __init__(self, max_tokens: int = 512, overlap_percent: float = 0.1):
        """Initialize chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            overlap_percent: Overlap between chunks (default 10%)
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = int(max_tokens * overlap_percent)
        self.md = MarkdownIt()

    def chunk(self, content: str) -> List[MarkdownChunk]:
        """Chunk markdown content semantically.

        Strategy:
        1. Parse into AST (tokens with types: heading, paragraph, fence, list, table)
        2. Build section hierarchy based on heading levels
        3. Chunk sections respecting boundaries
        4. Preserve parent headers for context

        Args:
            content: Markdown text

        Returns:
            List of MarkdownChunk objects
        """
        # Parse markdown into tokens
        tokens = self.md.parse(content)

        # Build section tree
        sections = self._build_sections(tokens, content)

        # Chunk sections
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section)
            chunks.extend(section_chunks)

        return chunks

    def _build_sections(self, tokens, content: str) -> List[Section]:
        """Build hierarchical section tree from tokens."""
        sections = []
        current_section = None
        header_stack = []  # Track nested headers

        for token in tokens:
            if token.type == "heading_open":
                level = int(token.tag[1])  # h1 -> 1, h2 -> 2, etc.

                # Pop headers at same or deeper level
                while header_stack and header_stack[-1]['level'] >= level:
                    header_stack.pop()

                # Extract header text (next token)
                header_text = self._extract_text(token)
                header_stack.append({'level': level, 'text': header_text})

                # Create new section
                if current_section:
                    sections.append(current_section)

                current_section = Section(
                    header_path=[h['text'] for h in header_stack],
                    level=level,
                    content_lines=[]
                )

            elif current_section and token.type in ["paragraph_open", "fence", "list_item_open", "table_open"]:
                # Add content to current section
                line_content = self._extract_text(token)
                current_section.content_lines.append(line_content)

                # Track content types
                if token.type == "fence":
                    current_section.has_code_blocks = True
                elif token.type == "table_open":
                    current_section.has_tables = True
                elif token.type == "list_item_open":
                    current_section.has_lists = True

        # Add final section
        if current_section:
            sections.append(current_section)

        return sections

    def _chunk_section(self, section: Section) -> List[MarkdownChunk]:
        """Chunk a single section, preserving boundaries."""
        full_text = section.get_full_text()
        estimated_tokens = len(full_text) / 3.5  # Conservative char-to-token ratio

        if estimated_tokens <= self.max_tokens:
            # Entire section fits
            return [MarkdownChunk(
                content=full_text,
                headers=section.header_path,
                chunk_type="section",
                start_line=section.start_line,
                end_line=section.end_line,
                has_code_blocks=section.has_code_blocks,
                has_tables=section.has_tables,
                has_lists=section.has_lists
            )]

        # Split section, preserving context
        sub_chunks = []
        lines = section.content_lines
        current_chunk_lines = []
        current_tokens = 0

        for line in lines:
            line_tokens = len(line) / 3.5

            if current_tokens + line_tokens > self.max_tokens and current_chunk_lines:
                # Emit chunk
                chunk_content = section.get_header_context() + "\n" + "\n".join(current_chunk_lines)
                sub_chunks.append(MarkdownChunk(
                    content=chunk_content,
                    headers=section.header_path,
                    chunk_type="subsection",
                    start_line=section.start_line,
                    end_line=section.end_line
                ))

                # Start new chunk with overlap
                overlap_lines = self._get_overlap_lines(current_chunk_lines)
                current_chunk_lines = overlap_lines + [line]
                current_tokens = sum(len(l) / 3.5 for l in current_chunk_lines)
            else:
                current_chunk_lines.append(line)
                current_tokens += line_tokens

        # Emit final chunk
        if current_chunk_lines:
            chunk_content = section.get_header_context() + "\n" + "\n".join(current_chunk_lines)
            sub_chunks.append(MarkdownChunk(
                content=chunk_content,
                headers=section.header_path,
                chunk_type="subsection",
                start_line=section.start_line,
                end_line=section.end_line
            ))

        return sub_chunks
```

**3. Direct Injection Handler**

```python
# src/arcaneum/indexing/markdown/injection.py

from pathlib import Path
from datetime import datetime
from uuid import uuid4
from typing import Dict, List, Optional
import frontmatter

class MarkdownInjectionHandler:
    """Handle direct injection of markdown content.

    Organizes injected content by collection and supports rich metadata.
    """

    def __init__(self, storage_dir: str = "~/.arcaneum/agent-memory"):
        """Initialize injection handler.

        Args:
            storage_dir: Base directory for agent memory storage
                        Structure: {storage_dir}/{collection_name}/{file}.md
        """
        self.storage_dir = Path(storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def inject(self,
               content: str,
               collection_name: str,
               agent_name: str = "Claude",
               title: Optional[str] = None,
               tags: List[str] = None,
               category: Optional[str] = None,
               metadata: Dict = None) -> Dict:
        """Inject markdown content and persist to collection-specific storage.

        Args:
            content: Markdown content (may include existing frontmatter)
            collection_name: Target collection (creates subdirectory)
            agent_name: Name of injecting agent (e.g., "Claude", "GPT-4")
            title: Document title (extracted from content if None)
            tags: List of tags for search/filtering
            category: User-defined category/label (e.g., "security-analysis")
            metadata: Additional custom metadata dict (merged into frontmatter)

        Returns:
            Dict with:
                - file_path: Where content was saved
                - injection_id: Unique identifier
                - metadata: Complete frontmatter metadata
                - content: Markdown body

        Example:
            handler.inject(
                content="# Security Analysis\\n\\n...",
                collection_name="memory",
                agent_name="Claude",
                category="security-analysis",
                tags=["security", "vulnerability"],
                metadata={"priority": "high", "project": "arcaneum"}
            )
            # Saves to: ~/.arcaneum/agent-memory/memory/2025-10-30_claude_security-analysis.md
        """
        # Generate unique ID
        injection_id = str(uuid4())
        timestamp = datetime.now()

        # Parse existing frontmatter (if any)
        post = frontmatter.loads(content)
        existing_meta = dict(post.metadata)

        # Extract title (priority: arg > existing > content)
        extracted_title = title or existing_meta.get('title') or self._extract_title(post.content)

        # Build complete frontmatter (preserve existing + add injection metadata)
        meta = {
            # Injection tracking (always set)
            'injection_id': injection_id,
            'injected_by': agent_name,
            'injected_at': timestamp.isoformat(),

            # Content metadata
            'title': extracted_title,
            'tags': tags or existing_meta.get('tags', []),
            'source': 'direct_injection',

            # Optional fields
            **(existing_meta),  # Preserve any existing frontmatter first
            **(metadata or {}),  # User-provided metadata
        }

        # Add category if provided (allows override via metadata dict)
        if category:
            meta['category'] = category

        # Create frontmatter document
        post.metadata = meta

        # Generate filename from category or title
        filename_base = category if category else extracted_title
        slug = self._slugify(filename_base)
        date_str = timestamp.strftime("%Y-%m-%d")
        filename = f"{date_str}_{agent_name.lower()}_{slug}.md"

        # Ensure collection-specific directory exists
        collection_dir = self.storage_dir / collection_name
        collection_dir.mkdir(parents=True, exist_ok=True)

        # Write to file
        file_path = collection_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(frontmatter.dumps(post))

        return {
            'file_path': str(file_path),
            'injection_id': injection_id,
            'metadata': meta,
            'content': post.content
        }

    def _extract_title(self, content: str) -> str:
        """Extract title from first H1 or first line."""
        lines = content.strip().split('\n')
        for line in lines:
            if line.startswith('# '):
                return line[2:].strip()
        # Fallback: first line
        return lines[0][:50] if lines else "Untitled"

    def _slugify(self, text: str) -> str:
        """Convert text to filename-safe slug."""
        import re
        slug = text.lower()
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        slug = slug.strip('-')
        return slug[:50]  # Limit length
```

**4. Metadata-Based Sync**

```python
# src/arcaneum/indexing/markdown/sync.py

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from datetime import datetime
from typing import Dict

class MarkdownMetadataSync:
    """Query Qdrant for indexed markdown files (source of truth)."""

    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant = qdrant_client
        self._cache = {}

    def get_indexed_files(self, collection_name: str) -> Dict[str, datetime]:
        """Get all (file_path, modified_at) from Qdrant.

        Returns:
            Dict mapping file_path -> last_modified timestamp
        """
        if collection_name in self._cache:
            return self._cache[collection_name]

        indexed = {}
        offset = None

        while True:
            points, offset = self.qdrant.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key="store_type",
                        match=MatchValue("markdown")
                    )]
                ),
                with_payload=["file_path", "modified_at"],
                with_vectors=False,
                limit=100,
                offset=offset
            )

            if not points:
                break

            for point in points:
                file_path = point.payload.get("file_path")
                modified = point.payload.get("modified_at")
                if file_path and modified:
                    indexed[file_path] = datetime.fromisoformat(modified)

            if offset is None:
                break

        self._cache[collection_name] = indexed
        return indexed

    def should_reindex_file(self, collection_name: str,
                           file_path: str,
                           current_mtime: datetime) -> bool:
        """Check if file needs re-indexing."""
        indexed_files = self.get_indexed_files(collection_name)

        # Not indexed yet
        if file_path not in indexed_files:
            return True

        # File modified since last index
        if current_mtime > indexed_files[file_path]:
            return True

        # Unchanged
        return False
```

**5. CLI Commands**

```python
# src/arcaneum/cli/index_markdown.py

import click
from rich.console import Console
from pathlib import Path

console = Console()

@click.command('index-markdown')
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--collection', required=True, help='Collection name')
@click.option('--recursive/--no-recursive', default=True, help='Search subdirectories')
@click.option('--exclude', multiple=True, help='Patterns to exclude (e.g., node_modules)')
@click.option('--qdrant-url', default='http://localhost:6333')
@click.option('--model', default='stella', help='Embedding model')
def index_markdown(input_path, collection, recursive, exclude, qdrant_url, model):
    """Index markdown files to a collection (directory sync mode).

    Examples:
        arc index markdown ~/Documents/Notes --collection knowledge
        arc index markdown ~/obsidian-vault --collection notes --exclude ".obsidian,templates"
    """
    from ..indexing.markdown.pipeline import MarkdownIndexingPipeline

    console.print(f"[bold cyan]Indexing markdown files from {input_path}[/bold cyan]")

    # Initialize pipeline
    pipeline = MarkdownIndexingPipeline(
        qdrant_url=qdrant_url,
        model_name=model
    )

    # Run indexing
    try:
        stats = pipeline.index_directory(
            input_path=input_path,
            collection_name=collection,
            recursive=recursive,
            exclude_patterns=list(exclude) if exclude else None
        )

        console.print(f"\n[green]✅ Indexed {stats['files_processed']} files "
                     f"({stats['chunks_created']} chunks)[/green]")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise

@click.command('inject-markdown')
@click.option('--collection', required=True, help='Collection name (creates subdirectory)')
@click.option('--title', help='Document title')
@click.option('--tags', help='Comma-separated tags')
@click.option('--category', help='Category/label (e.g., security-analysis, api-design)')
@click.option('--agent', default='Claude', help='Agent name')
@click.option('--metadata', help='Additional metadata as JSON string')
@click.option('--file', type=click.Path(exists=True), help='Read content from file')
@click.option('--qdrant-url', default='http://localhost:6333')
@click.option('--model', default='stella', help='Embedding model')
def inject_markdown(collection, title, tags, category, agent, metadata, file, qdrant_url, model):
    """Inject markdown content directly (agent memory mode).

    Content can be provided via stdin or --file flag.
    Organizes by collection: ~/.arcaneum/agent-memory/{collection}/

    Examples:
        # Basic injection
        echo "# Research\\n\\nFindings..." | arc inject markdown --collection knowledge --title "Research"

        # With category and tags
        arc inject markdown --file research.md --collection memory \\
            --category "security-analysis" --tags "security,vulnerability"

        # With custom metadata
        arc inject markdown --file notes.md --collection knowledge \\
            --metadata '{"priority": "high", "project": "arcaneum"}'
    """
    from ..indexing.markdown.pipeline import MarkdownIndexingPipeline
    import sys
    import json

    # Read content
    if file:
        with open(file, 'r') as f:
            content = f.read()
    else:
        content = sys.stdin.read()

    if not content.strip():
        console.print("[red]❌ No content provided[/red]")
        return

    # Parse tags
    tag_list = [t.strip() for t in tags.split(',')] if tags else []

    # Parse metadata JSON
    metadata_dict = None
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError as e:
            console.print(f"[red]❌ Invalid metadata JSON: {e}[/red]")
            return

    # Initialize pipeline
    pipeline = MarkdownIndexingPipeline(
        qdrant_url=qdrant_url,
        model_name=model
    )

    # Inject content
    try:
        result = pipeline.inject_content(
            content=content,
            collection_name=collection,
            agent_name=agent,
            title=title,
            tags=tag_list,
            category=category,
            metadata=metadata_dict
        )

        console.print(f"[green]✅ Injected content to collection '{collection}'[/green]")
        console.print(f"Stored at: {result['file_path']}")
        console.print(f"Injection ID: {result['injection_id']}")
        if category:
            console.print(f"Category: {category}")
        console.print(f"Chunks created: {result['chunk_count']}")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise
```

#### Metadata Schema

```python
@dataclass
class MarkdownChunkMetadata:
    """Metadata for a markdown chunk in Qdrant."""

    # PRIMARY mode identifier
    mode: str  # "directory_sync" or "direct_injection"

    # File-based fields (directory sync mode)
    file_path: Optional[str] = None  # Absolute path
    filename: Optional[str] = None   # Basename
    file_size: Optional[int] = None  # Bytes
    file_hash: Optional[str] = None  # SHA256 hash (first 12 chars) for change detection

    # Injection fields (direct injection mode)
    injection_id: Optional[str] = None  # UUID
    injected_by: Optional[str] = None   # Agent name
    injected_at: Optional[str] = None   # ISO timestamp

    # Frontmatter fields (both modes)
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None  # User-defined category/label
    project: Optional[str] = None
    source: Optional[str] = None

    # Custom metadata (any additional frontmatter fields)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    # Content structure fields
    word_count: int = 0
    chunk_index: int = 0
    chunk_count: int = 0
    header_path: List[str] = field(default_factory=list)  # ["Section", "Subsection"]
    header_level: int = 0  # 1-6 for H1-H6
    has_code_blocks: bool = False
    code_languages: List[str] = field(default_factory=list)  # ["python", "bash"]
    has_tables: bool = False
    has_lists: bool = False

    # Embedding metadata
    embedding_model: str = "stella_en_1.5B_v5"
    store_type: str = "markdown"

    def to_payload(self) -> dict:
        """Convert to Qdrant payload."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
```

### Implementation Example

**Directory Sync Mode**:

```bash
# Create collection
arc collection create knowledge --model stella --type markdown

# Index existing markdown files
arc index markdown ~/Documents/Notes --collection knowledge --recursive

# Result: All markdown files indexed, frontmatter extracted, semantic chunking applied
```

**Direct Injection Mode** (with metadata):

```python
# Agent workflow (programmatic)
from arcaneum.indexing.markdown.pipeline import MarkdownIndexingPipeline

pipeline = MarkdownIndexingPipeline(
    qdrant_url='http://localhost:6333',
    model_name='stella'
)

# Agent generates research summary
research_content = """
# GPU Acceleration Patterns

Research findings from analyzing 20+ GPU-accelerated libraries...

## Key Patterns

1. **Metal Performance Shaders**: Apple's framework for GPU compute
2. **CUDA**: NVIDIA's parallel computing platform
...
"""

# Inject with metadata (organized by collection)
result = pipeline.inject_content(
    content=research_content,
    collection_name='memory',  # Stored in memory/ subdirectory
    agent_name='Claude',
    category='security-analysis',  # User-defined label
    tags=['gpu', 'performance', 'security'],
    title='GPU Acceleration Patterns',
    metadata={
        'priority': 'high',
        'project': 'arcaneum',
        'related_to': ['arcaneum-123']
    }
)

print(f"Stored at: {result['file_path']}")
# ~/.arcaneum/agent-memory/memory/2025-10-30_claude_security-analysis.md

# Later: Re-index agent memory
pipeline.index_directory(
    input_path='~/.arcaneum/agent-memory/memory',
    collection_name='memory'
)
```

**CLI Example**:

```bash
# Inject with category and metadata
echo "# Security Analysis\n\nFindings..." | arc inject markdown \
    --collection memory \
    --category security-analysis \
    --tags "security,vulnerability" \
    --metadata '{"priority": "high", "project": "arcaneum"}'

# Result: ~/.arcaneum/agent-memory/memory/2025-10-30_claude_security-analysis.md
```

**Corpus Mode (Dual-Index)**:

```bash
# Create corpus (both Qdrant + MeiliSearch)
arc corpus create knowledge --type markdown

# Sync directory (dual-index)
arc corpus sync ~/Documents/Notes --corpus knowledge

# Search both ways
arc search "GPU acceleration" --collection knowledge  # Semantic (Qdrant)
arc search text "Metal API" --index knowledge         # Exact (MeiliSearch)
```

## Alternatives Considered

### Alternative 1: Use Generic Text Chunking (No Markdown Awareness)

**Description**: Treat markdown as plain text, chunk by tokens only

**Pros**:

- Simpler implementation
- Reuse existing text chunker
- Fast processing

**Cons**:

- Breaks semantic boundaries (splits mid-section)
- Loses header context
- Poor code block handling (splits code)
- 35% worse retrieval accuracy (research finding)

**Reason for rejection**: Markdown has rich structure that should be preserved for better search relevance.

### Alternative 2: Store Only in Qdrant (No File Persistence for Injections)

**Description**: Directly-injected content exists only in Qdrant

**Pros**:

- Simpler (no file I/O)
- Faster injection
- No storage management

**Cons**:

- **Data loss risk**: If collection deleted, content gone forever
- **No re-indexing**: Can't rebuild collection
- **No auditability**: Can't review what was injected
- **Against durability principle**: Ephemeral agent memory is unreliable

**Reason for rejection**: Agent-generated content is valuable and should be durable. File
persistence enables recovery and auditability.

### Alternative 3: Single Mode (Directory Sync Only)

**Description**: Only support indexing existing markdown files

**Pros**:

- Simpler architecture
- Single code path
- Fewer edge cases

**Cons**:

- **No agent memory**: Can't store research results
- **Requires manual file creation**: Agent must write files before indexing
- **Workflow friction**: Extra step for programmatic use

**Reason for rejection**: Direct injection is a key use case for AI agent workflows. Supporting both modes is worth the complexity.

### Alternative 4: Use Markdown-Specific Embedding Model

**Description**: Train or use a markdown-specific embedding model

**Pros**:

- Potentially better retrieval for markdown
- Optimized for document structure

**Cons**:

- **No such model exists**: Would require training
- **General text models work well**: stella/jina already handle markdown
- **Unnecessary complexity**: Chunking strategy matters more than model

**Reason for rejection**: General text embeddings are sufficient. Semantic chunking provides the real benefit.

## Trade-offs and Consequences

### Positive Consequences

1. **Semantic Preservation**: Header-aware chunking preserves document structure (35% better retrieval)
2. **Dual-Mode Flexibility**: Supports both existing files and agent-generated content
3. **Durability**: Direct injection persists to disk (no data loss)
4. **Auditability**: Injected content is traceable (agent, timestamp, tags)
5. **Collection Typing**: Prevents mixing markdown with PDFs/code (type safety)
6. **Metadata-Rich**: Frontmatter extraction enables sophisticated filtering
7. **Corpus Support**: Optional dual-indexing for hybrid search
8. **Architectural Consistency**: Follows RDR-004/RDR-005/RDR-009 patterns

### Negative Consequences

1. **Parsing Overhead**: Markdown AST parsing adds 10-15% processing time
   - *Mitigation*: Acceptable for better chunking quality
   - *Benefit*: 35% retrieval improvement justifies cost

2. **Storage Duplication**: Injected content stored both in Qdrant and as files
   - *Mitigation*: Markdown files are small (typically < 100KB)
   - *Benefit*: Durability and re-indexing capability

3. **Two Code Paths**: Directory sync vs direct injection adds complexity
   - *Mitigation*: Shared chunking/indexing logic, different entry points only
   - *Benefit*: Enables both key use cases

4. **Frontmatter Dependency**: Relies on YAML frontmatter convention
   - *Mitigation*: Works without frontmatter (extracts title, generates metadata)
   - *Benefit*: Enables rich metadata when present

### Risks and Mitigations

**Risk**: Markdown parsing fails on malformed input

**Mitigation**:

- Fallback to line-based chunking if AST parsing fails
- Log warnings for debugging
- Continue processing other files

**Risk**: Storage directory fills up with agent memory

**Mitigation**:

- Configurable storage location
- Optional cleanup command: `arc cleanup agent-memory --older-than 90d`
- Compression for old files

**Risk**: Duplicate injections (same content injected twice)

**Mitigation**:

- Content hashing: detect duplicates by MD5
- Warn user if duplicate detected
- Option to skip or overwrite

**Risk**: Large markdown files (> 1MB) slow processing

**Mitigation**:

- File size limit (default 10MB, configurable)
- Stream processing for very large files
- Progress reporting

## Implementation Plan

### Prerequisites

- [x] Qdrant server running (RDR-002)
- [x] Collection typing infrastructure (RDR-003)
- [ ] markdown-it-py installed (>= 3.0.0)
- [ ] python-frontmatter installed (>= 1.1.0)
- [ ] Python >= 3.12

### Step-by-Step Implementation

#### Step 1: Markdown Discovery and Parsing Module

Create `src/arcaneum/indexing/markdown/discovery.py`:

- Implement `MarkdownDiscovery` class
- File discovery with exclusion patterns
- Frontmatter extraction with python-frontmatter
- Content hash computation (SHA256)
- Handle edge cases (missing frontmatter, malformed YAML)

**Estimated effort**: 4 hours

#### Step 2: Semantic Chunking Module

Create `src/arcaneum/indexing/markdown/chunker.py`:

- Implement `MarkdownChunker` class with markdown-it-py
- Build section tree from AST
- Semantic chunking respecting headers/code blocks
- Fallback to line-based chunking
- Extract code block metadata

**Estimated effort**: 8 hours

#### Step 3: Direct Injection Handler

Create `src/arcaneum/indexing/markdown/injection.py`:

- Implement `MarkdownInjectionHandler` class
- Generate unique injection IDs
- Persist content to storage directory
- Build frontmatter for injected content
- Filename generation and slugification

**Estimated effort**: 4 hours

#### Step 4: Metadata-Based Sync Module

Create `src/arcaneum/indexing/markdown/sync.py`:

- Implement `MarkdownMetadataSync` class
- Query Qdrant for indexed files
- Change detection via content hash comparison
- Reuse `compute_file_hash()` from common sync module

**Estimated effort**: 3 hours

#### Step 5: Main Indexing Pipeline

Create `src/arcaneum/indexing/markdown/pipeline.py`:

- Implement `MarkdownIndexingPipeline` orchestrator
- Integrate discovery, chunking, sync modules
- Directory sync mode implementation
- Direct injection mode implementation
- Progress reporting
- Batch upload to Qdrant

**Estimated effort**: 6 hours

#### Step 6: CLI Commands

Create `src/arcaneum/cli/index_markdown.py`:

- `index-markdown` command (directory sync)
- `inject-markdown` command (direct injection)
- Register with main CLI
- Rich progress output
- Error handling

**Estimated effort**: 4 hours

#### Step 7: Collection Type Integration

Modify `src/arcaneum/cli/collections.py`:

- Add `markdown` to allowed types
- Update validation logic
- Update help text

**Estimated effort**: 1 hour

#### Step 8: Corpus Support (Optional)

Modify `src/arcaneum/cli/corpus.py`:

- Add markdown support to corpus creation
- Dual-indexing for markdown
- MeiliSearch index settings for markdown

**Estimated effort**: 3 hours

#### Step 9: Testing

Create comprehensive tests:

- Unit tests for markdown discovery
- Unit tests for semantic chunking
- Unit tests for injection handler
- Unit tests for metadata sync
- Integration tests for directory sync
- Integration tests for direct injection
- Integration tests for corpus mode
- Edge case tests (malformed markdown, missing frontmatter)

**Estimated effort**: 10 hours

#### Step 10: Documentation

Update documentation:

- README with markdown indexing examples
- CLI reference for new commands
- Frontmatter conventions guide
- Agent integration guide (direct injection)
- Troubleshooting guide

**Estimated effort**: 4 hours

### Total Estimated Effort

**47 hours** (~6 days of focused work)

**Effort Breakdown**:

- Step 1: Discovery and parsing (4h)
- Step 2: Semantic chunking (8h)
- Step 3: Direct injection (4h)
- Step 4: Metadata sync (3h)
- Step 5: Main pipeline (6h)
- Step 6: CLI commands (4h)
- Step 7: Type integration (1h)
- Step 8: Corpus support (3h)
- Step 9: Testing (10h)
- Step 10: Documentation (4h)

### Files to Create

**New Modules**:

- `src/arcaneum/indexing/markdown/__init__.py`
- `src/arcaneum/indexing/markdown/discovery.py` - File discovery and frontmatter parsing
- `src/arcaneum/indexing/markdown/chunker.py` - Semantic markdown chunking
- `src/arcaneum/indexing/markdown/injection.py` - Direct injection handler
- `src/arcaneum/indexing/markdown/sync.py` - Metadata-based sync
- `src/arcaneum/indexing/markdown/pipeline.py` - Main orchestrator

**CLI Commands**:

- `src/arcaneum/cli/index_markdown.py` - CLI commands

**Tests**:

- `tests/indexing/markdown/test_discovery.py`
- `tests/indexing/markdown/test_chunker.py`
- `tests/indexing/markdown/test_injection.py`
- `tests/indexing/markdown/test_sync.py`
- `tests/indexing/markdown/test_pipeline.py`
- `tests/integration/test_markdown_directory_sync.py`
- `tests/integration/test_markdown_injection.py`
- `tests/integration/test_markdown_corpus.py`

### Files to Modify

- `src/arcaneum/cli/main.py` - Register markdown commands
- `src/arcaneum/cli/collections.py` - Add `markdown` type
- `src/arcaneum/cli/corpus.py` - Add markdown corpus support
- `docs/reference/collections-and-types.md` - Document markdown type
- `README.md` - Add markdown examples

### Dependencies

Add to `pyproject.toml`:

```toml
[project.dependencies]
# New dependencies for markdown indexing (RDR-014)
markdown-it-py = ">=4.0.0"
python-frontmatter = ">=1.1.0"
pygments = ">=2.19.0"

# Note: Already present in project, no need to add:
# - fastembed >= 0.7.3
# - sentence-transformers >= 3.3.1
# - llama-index-core >= 0.14.6
# - qdrant-client[fastembed] >= 1.15.0
# - tenacity, rich
```

## Validation

### Testing Approach

**Unit Tests**:

- Markdown discovery with exclusion patterns
- Frontmatter extraction (YAML, TOML, missing)
- Semantic chunking for various document structures
- Injection ID generation and filename slugification
- Metadata sync (Qdrant queries)

**Integration Tests**:

- End-to-end directory sync
- End-to-end direct injection
- Incremental sync with file changes
- Corpus mode (dual-indexing)
- Collection type validation

**Performance Tests**:

- Chunking speed for large documents
- Directory scan with thousands of files
- Injection throughput
- Metadata query overhead

### Test Scenarios

#### Scenario 1: Directory Sync with Frontmatter

- **Setup**: Directory with 100 markdown files, YAML frontmatter
- **Action**: `arc index markdown ~/notes --collection knowledge`
- **Expected**: All files indexed, frontmatter extracted, semantic chunking applied

#### Scenario 2: Direct Injection by Agent

- **Setup**: Agent generates research summary
- **Action**: `pipeline.inject_content(...)`
- **Expected**: Content persisted to `~/.arcaneum/agent-memory/knowledge/`, indexed to Qdrant

#### Scenario 3: Incremental Sync After File Changes

- **Setup**: Collection with 100 files indexed
- **Action**: Modify 10 files, add 5 new files, re-run indexing
- **Expected**: Only 15 files re-indexed (change detection works)

#### Scenario 4: Markdown with Code Blocks

- **Setup**: Markdown file with Python and Bash code blocks
- **Action**: Index file
- **Expected**: Code blocks preserved intact, languages detected, searchable

#### Scenario 5: Missing Frontmatter

- **Setup**: Markdown file without YAML frontmatter
- **Action**: Index file
- **Expected**: Title extracted from H1, default metadata generated, indexing succeeds

#### Scenario 6: Corpus Mode (Hybrid Search)

- **Setup**: Corpus created, markdown directory synced
- **Action**: Semantic search + full-text search
- **Expected**: Both searches work, shared metadata enables cross-system queries

### Performance Validation

**Metrics to Track**:

- Files indexed per second
- Chunks created per second
- Frontmatter parsing latency
- AST chunking overhead vs line-based
- Metadata query performance

**Targets**:

- 50-100 files/sec indexing throughput
- < 50ms per file for frontmatter extraction
- < 200ms per file for semantic chunking
- < 5s for metadata query on collection with 1000 files
- < 10% overhead vs line-based chunking

### Security Validation

**Frontmatter Security**:

- Verify no code execution from YAML parsing
- Sanitize file paths for injection
- Validate tags and metadata fields

**Storage Security**:

- Ensure storage directory permissions correct
- Validate filename generation (no path traversal)
- Sanitize agent names for filenames

## References

### Related RDRs

- [RDR-002: Qdrant Server Setup](RDR-002-qdrant-server-setup.md) - Vector database deployment
- [RDR-003: Collection Creation](RDR-003-collection-creation.md) - Collection typing patterns
- [RDR-004: PDF Bulk Indexing](RDR-004-pdf-bulk-indexing.md) - Metadata-based sync pattern
- [RDR-005: Source Code Indexing](RDR-005-source-code-indexing.md) - AST chunking inspiration
- [RDR-009: Dual Indexing Strategy](RDR-009-dual-indexing-strategy.md) - Corpus architecture

### Beads Issues

- [arcaneum-199](../../.beads/issues.jsonl) - RDR creation request

### External Resources

- **markdown-it-py**: <https://markdown-it-py.readthedocs.io/>
- **python-frontmatter**: <https://python-frontmatter.readthedocs.io/>
- **ChunkHound Research**: Semantic chunking for markdown (35% improvement)
- **Qdrant Filter Documentation**: <https://qdrant.tech/documentation/concepts/filtering/>

## Notes

### Implementation Priority

1. Semantic chunking (core quality improvement)
2. Directory sync mode (primary use case)
3. Direct injection mode (agent workflows)
4. Metadata-based sync (architectural consistency)
5. Corpus support (optional enhancement)

### Future Enhancements

**Advanced Chunking**:

- Table-aware chunking (keep tables intact)
- Mathematical notation preservation (LaTeX)
- Image reference tracking (![alt](path))

**Agent Integration**:

- MCP tool: `inject_markdown_memory(content, tags)`
- Automatic summarization before injection
- Duplicate detection (content hashing)

**Batch Re-sync**:

- Re-run `arc index markdown` to pick up changes
- Metadata-based change detection handles incremental updates
- Similar to PDF indexing workflow

**Query Enhancements**:

- Search by tags: `arc search "gpu" --filter 'tags = [performance,optimization]'`
- Search by category: `arc search "security" --filter 'category = security-analysis'`
- Search by date range: `arc search "research" --filter 'date > 2025-10-01'`
- Search by agent: `arc search "" --filter 'injected_by = Claude'`
- Search by project: `arc search "api" --filter 'project = arcaneum'`

**Cleanup Tools**:

- `arc cleanup agent-memory --older-than 90d` - Remove old agent memory
- `arc cleanup agent-memory --collection knowledge --dry-run` - Preview deletions

### Success Criteria

- ✅ Semantic chunking preserves markdown structure
- ✅ Both directory sync and direct injection work
- ✅ Injected content persists to disk (durability)
- ✅ Frontmatter extraction with fallbacks
- ✅ Collection type `markdown` enforced
- ✅ Integration with corpus (dual-indexing)
- ✅ < 50 hours implementation time
- ✅ 35%+ retrieval improvement vs naive chunking
- ✅ Markdownlint compliant

This RDR provides a complete specification for markdown indexing that supports both traditional
file-based workflows and modern AI agent memory patterns, with semantic awareness and durability
guarantees.

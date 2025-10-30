"""
Semantic markdown chunking with structure-aware parsing (RDR-014).

This module provides intelligent markdown chunking that preserves document structure
using markdown-it-py AST parsing. Chunks respect semantic boundaries (headers, code blocks,
lists, tables) and preserve parent header context for better retrieval.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    from markdown_it import MarkdownIt
    from markdown_it.token import Token
    MARKDOWN_IT_AVAILABLE = True
except ImportError:
    MARKDOWN_IT_AVAILABLE = False
    logging.warning("markdown-it-py not available, semantic chunking will be limited")

logger = logging.getLogger(__name__)


@dataclass
class MarkdownChunk:
    """Represents a semantic markdown chunk with metadata."""
    text: str
    chunk_index: int
    token_count: int
    metadata: Dict
    header_path: List[str]  # Parent headers for context (e.g., ["# Intro", "## Background"])


class SemanticMarkdownChunker:
    """Chunks markdown with semantic awareness and parent header context preservation.

    Algorithm (RDR-014):
    1. Parse markdown to tokens using markdown-it-py (heading, paragraph, code_block, list, table)
    2. Build hierarchical section tree based on heading levels
    3. Chunk sections while preserving semantic boundaries
    4. Include parent headers in sub-chunks for context

    Target: 35% better retrieval accuracy vs naive token-based chunking.
    """

    # Character to token ratio (conservative estimate for markdown)
    CHARS_PER_TOKEN = 3.3

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_chars: Optional[int] = None,
        preserve_code_blocks: bool = True
    ):
        """Initialize semantic markdown chunker.

        Args:
            chunk_size: Target chunk size in tokens
                       - 512 tokens for 1K models (bge-base-en-v1.5)
                       - 1024 tokens for stella_en_1.5B_v5 (recommended)
                       - 2048+ tokens for jina-embeddings-v2-base-en (8K context)
            chunk_overlap: Overlap between chunks in tokens (default ~10%)
            max_chars: Maximum characters per chunk (None = auto-calculate)
            preserve_code_blocks: Keep code blocks intact (don't split mid-block)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chars = max_chars or int(chunk_size * self.CHARS_PER_TOKEN)
        self.preserve_code_blocks = preserve_code_blocks

        if not MARKDOWN_IT_AVAILABLE:
            logger.warning(
                "markdown-it-py not available. "
                "Semantic chunking will fall back to naive splitting."
            )

        # Initialize markdown-it parser
        self.md = MarkdownIt() if MARKDOWN_IT_AVAILABLE else None

    def chunk(self, text: str, metadata: Dict) -> List[MarkdownChunk]:
        """Chunk markdown using semantic structure.

        Args:
            text: Markdown text to chunk
            metadata: Base metadata to attach to all chunks

        Returns:
            List of MarkdownChunk objects with preserved structure
        """
        if not text or not text.strip():
            return []

        # Try semantic chunking if available
        if MARKDOWN_IT_AVAILABLE and self.md:
            try:
                return self._semantic_chunking(text, metadata)
            except Exception as e:
                logger.warning(f"Semantic chunking failed: {e}, falling back to naive chunking")
                # Fall through to naive chunking

        # Fallback to naive token-based chunking
        return self._naive_chunking(text, metadata)

    def _semantic_chunking(self, text: str, metadata: Dict) -> List[MarkdownChunk]:
        """Semantic chunking using markdown AST parsing.

        Algorithm:
        1. Parse to tokens (heading, paragraph, code_block, list, table)
        2. Build section tree based on heading hierarchy
        3. Chunk sections respecting boundaries
        4. Preserve parent header context
        """
        # Parse markdown to tokens
        tokens = self.md.parse(text)

        # Build hierarchical sections from tokens
        sections = self._build_sections(tokens, text)

        # Chunk sections while preserving boundaries
        chunks = self._chunk_sections(sections, metadata)

        logger.info(f"Created {len(chunks)} semantic chunks from {len(sections)} sections")
        return chunks

    def _build_sections(self, tokens: List[Token], source_text: str) -> List[Dict]:
        """Build hierarchical sections from markdown tokens.

        A section starts with a heading and contains all content until the next
        heading of equal or higher level.

        Returns:
            List of section dicts with: level, header, content, start_pos, end_pos
        """
        sections = []
        current_section = None
        header_stack = []  # Track parent headers for context

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Heading starts a new section
            if token.type == "heading_open":
                # Get heading level (h1=1, h2=2, etc.)
                level = int(token.tag[1])

                # Get heading text from next token (inline content)
                heading_text = ""
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    heading_text = tokens[i + 1].content

                # Save previous section if exists
                if current_section:
                    sections.append(current_section)

                # Update header stack (remove deeper levels)
                header_stack = [h for h in header_stack if h['level'] < level]
                header_stack.append({'level': level, 'text': heading_text})

                # Start new section
                current_section = {
                    'level': level,
                    'header': heading_text,
                    'header_path': [h['text'] for h in header_stack],
                    'content_parts': [],
                    'start_line': token.map[0] if token.map else 0,
                }

                # Skip to heading_close
                i += 3  # heading_open, inline, heading_close
                continue

            # Add content to current section
            if current_section is not None:
                content = self._extract_token_content(token, source_text)
                if content:
                    current_section['content_parts'].append({
                        'type': token.type,
                        'content': content,
                        'is_code_block': token.type in ['code_block', 'fence']
                    })

            i += 1

        # Save final section
        if current_section:
            sections.append(current_section)

        # If no headers found, create a single section with all content
        if not sections:
            sections = [{
                'level': 0,
                'header': '',
                'header_path': [],
                'content_parts': [{'type': 'text', 'content': source_text, 'is_code_block': False}],
                'start_line': 0,
            }]

        return sections

    def _extract_token_content(self, token: Token, source_text: str) -> str:
        """Extract content from a markdown token."""
        # For tokens with content, use it directly
        if token.content:
            return token.content

        # For tokens with map (line numbers), extract from source
        if token.map:
            lines = source_text.split('\n')
            start_line, end_line = token.map
            content_lines = lines[start_line:end_line]
            return '\n'.join(content_lines)

        return ""

    def _chunk_sections(self, sections: List[Dict], base_metadata: Dict) -> List[MarkdownChunk]:
        """Chunk sections while preserving semantic boundaries.

        Strategy:
        1. Try to keep sections intact if under chunk size
        2. Split large sections at paragraph/list boundaries
        3. Preserve code blocks (don't split)
        4. Include parent header context in each chunk
        """
        chunks = []
        chunk_index = 0

        for section in sections:
            # Build section text with header
            section_text = self._build_section_text(section)
            section_tokens = len(section_text) / self.CHARS_PER_TOKEN

            # Check if section has code blocks
            has_code = any(part.get('is_code_block', False) for part in section['content_parts'])

            # Section fits in one chunk
            if section_tokens <= self.chunk_size:
                chunk = self._create_chunk(
                    text=section_text,
                    chunk_index=chunk_index,
                    metadata=base_metadata,
                    header_path=section['header_path'],
                    has_code_blocks=has_code
                )
                chunks.append(chunk)
                chunk_index += 1

            else:
                # Section too large, need to split
                sub_chunks = self._split_large_section(
                    section, base_metadata, chunk_index
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)

        return chunks

    def _build_section_text(self, section: Dict) -> str:
        """Build text for a section including header and content."""
        parts = []

        # Add header if present
        if section['header']:
            header_prefix = '#' * section['level']
            parts.append(f"{header_prefix} {section['header']}")

        # Add content parts
        for part in section['content_parts']:
            parts.append(part['content'])

        return '\n\n'.join(parts)

    def _split_large_section(
        self, section: Dict, base_metadata: Dict, start_chunk_index: int
    ) -> List[MarkdownChunk]:
        """Split a large section into multiple chunks at semantic boundaries."""
        chunks = []
        current_parts = []
        current_tokens = 0
        current_has_code = False
        chunk_index = start_chunk_index

        # Add header to first chunk
        header_text = ""
        if section['header']:
            header_prefix = '#' * section['level']
            header_text = f"{header_prefix} {section['header']}"
            header_tokens = len(header_text) / self.CHARS_PER_TOKEN
            current_parts.append(header_text)
            current_tokens += header_tokens

        # Process content parts
        for part in section['content_parts']:
            part_text = part['content']
            part_tokens = len(part_text) / self.CHARS_PER_TOKEN
            is_code = part.get('is_code_block', False)

            # Code blocks: keep intact if possible
            if is_code and self.preserve_code_blocks:
                # If code block fits with current parts, add it
                if current_tokens + part_tokens <= self.chunk_size:
                    current_parts.append(part_text)
                    current_tokens += part_tokens
                    current_has_code = True
                else:
                    # Save current chunk if has content
                    if len(current_parts) > (1 if header_text else 0):
                        chunk_text = '\n\n'.join(current_parts)
                        chunk = self._create_chunk(
                            text=chunk_text,
                            chunk_index=chunk_index,
                            metadata=base_metadata,
                            header_path=section['header_path'],
                            has_code_blocks=current_has_code
                        )
                        chunks.append(chunk)
                        chunk_index += 1

                    # Start new chunk with header context + code block
                    current_parts = [header_text, part_text] if header_text else [part_text]
                    current_tokens = (len(header_text) / self.CHARS_PER_TOKEN if header_text else 0) + part_tokens
                    current_has_code = True

            # Regular content: can split if needed
            else:
                if current_tokens + part_tokens <= self.chunk_size:
                    current_parts.append(part_text)
                    current_tokens += part_tokens
                else:
                    # Save current chunk
                    if current_parts:
                        chunk_text = '\n\n'.join(current_parts)
                        chunk = self._create_chunk(
                            text=chunk_text,
                            chunk_index=chunk_index,
                            metadata=base_metadata,
                            header_path=section['header_path'],
                            has_code_blocks=current_has_code
                        )
                        chunks.append(chunk)
                        chunk_index += 1

                    # Start new chunk with header context
                    current_parts = [header_text, part_text] if header_text else [part_text]
                    current_tokens = (len(header_text) / self.CHARS_PER_TOKEN if header_text else 0) + part_tokens
                    current_has_code = False

        # Save final chunk
        if current_parts:
            chunk_text = '\n\n'.join(current_parts)
            chunk = self._create_chunk(
                text=chunk_text,
                chunk_index=chunk_index,
                metadata=base_metadata,
                header_path=section['header_path'],
                has_code_blocks=current_has_code
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self, text: str, chunk_index: int, metadata: Dict, header_path: List[str],
        has_code_blocks: bool = False
    ) -> MarkdownChunk:
        """Create a MarkdownChunk with metadata."""
        token_count = int(len(text) / self.CHARS_PER_TOKEN)

        # Detect code blocks from fence markers or indented code
        if not has_code_blocks:
            has_code_blocks = '```' in text or '~~~' in text or '\n    ' in text

        chunk_metadata = {
            **metadata,
            'chunk_index': chunk_index,
            'header_path': ' > '.join(header_path) if header_path else None,
            'has_code_blocks': has_code_blocks,
        }

        return MarkdownChunk(
            text=text,
            chunk_index=chunk_index,
            token_count=token_count,
            metadata=chunk_metadata,
            header_path=header_path
        )

    def _naive_chunking(self, text: str, metadata: Dict) -> List[MarkdownChunk]:
        """Fallback naive token-based chunking when markdown-it is unavailable."""
        chunks = []
        chunk_index = 0

        # Calculate character limits
        chunk_chars = int(self.chunk_size * self.CHARS_PER_TOKEN)
        overlap_chars = int(self.chunk_overlap * self.CHARS_PER_TOKEN)

        start = 0
        while start < len(text):
            end = start + chunk_chars

            # Try to break at paragraph boundary
            if end < len(text):
                # Look for double newline in last 20% of chunk
                search_start = end - int(chunk_chars * 0.2)
                para_end = text.rfind('\n\n', search_start, end)

                if para_end != -1:
                    end = para_end + 2

            chunk_text = text[start:end].strip()

            if chunk_text:
                token_count = int(len(chunk_text) / self.CHARS_PER_TOKEN)

                chunk_metadata = {
                    **metadata,
                    'chunk_index': chunk_index,
                    'chunk_start_char': start,
                    'chunk_end_char': end,
                    'semantic_chunking': False,
                }

                chunk = MarkdownChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    metadata=chunk_metadata,
                    header_path=[]
                )

                chunks.append(chunk)
                chunk_index += 1

            # Move start position (with overlap)
            start = end - overlap_chars

        logger.info(f"Created {len(chunks)} naive chunks (fallback mode)")
        return chunks


def chunk_markdown(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    metadata: Optional[Dict] = None
) -> List[MarkdownChunk]:
    """Convenience function to chunk markdown text.

    Args:
        text: Markdown text to chunk
        chunk_size: Target chunk size in tokens (default 512)
        chunk_overlap: Overlap between chunks in tokens (default 50)
        metadata: Optional base metadata dict

    Returns:
        List of MarkdownChunk objects
    """
    chunker = SemanticMarkdownChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return chunker.chunk(text, metadata or {})

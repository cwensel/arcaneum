"""
AST-aware code chunking with tree-sitter.

This module provides intelligent code chunking that preserves syntactic boundaries
using tree-sitter parsing with automatic fallback to line-based chunking (RDR-005).
"""

import os
import logging
from typing import List, Optional
from pathlib import Path

try:
    from llama_index.core.node_parser import CodeSplitter
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    logging.warning("llama-index-core not available, AST chunking will be limited")

from .types import Chunk

logger = logging.getLogger(__name__)


class ASTCodeChunker:
    """Chunks source code using AST parsing with fallback to line-based chunking.

    Supports 165+ languages via tree-sitter-language-pack.
    Uses the cAST (Context-Aware AST) algorithm:
    1. Parse code into AST using tree-sitter
    2. Recursively split nodes that exceed max_chunk_size
    3. Greedily merge adjacent small nodes to optimize chunk count
    4. Preserve syntactic boundaries (functions, classes, modules)

    Automatically falls back to line-based chunking if AST parsing fails.
    """

    # Language mapping: file extension -> tree-sitter language name
    # Covers 15+ primary languages plus many more supported by tree-sitter
    LANGUAGE_MAP = {
        # Primary languages (RDR-005)
        ".py": "python",
        ".java": "java",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".cs": "c_sharp",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".php": "php",
        ".rb": "ruby",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".scala": "scala",
        ".sc": "scala",
        ".swift": "swift",

        # Additional supported languages
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".xml": "xml",
        ".md": "markdown",
        ".sql": "sql",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".r": "r",
        ".R": "r",
        ".lua": "lua",
        ".vim": "vim",
        ".el": "elisp",
        ".clj": "clojure",
        ".ex": "elixir",
        ".exs": "elixir",
        ".erl": "erlang",
        ".hrl": "erlang",
        ".hs": "haskell",
        ".ml": "ocaml",
        ".nim": "nim",
        ".pl": "perl",
        ".pm": "perl",
        ".proto": "proto",
        ".thrift": "thrift",
    }

    # Character to token ratio (conservative estimate for code)
    CHARS_PER_TOKEN = 3.5

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 20,
        max_chars: Optional[int] = None,
        hard_max_chars: Optional[int] = None
    ):
        """Initialize AST code chunker.

        Args:
            chunk_size: Target chunk size in tokens
                       - 400 tokens for 8K context models (jina-v2-base-code)
                       - 2000-4000 tokens for 32K context models (jina-code-1.5b)
            chunk_overlap: Overlap between chunks in tokens (default 5%)
            max_chars: Maximum characters per chunk (None = auto-calculate)
            hard_max_chars: Absolute maximum chars per chunk based on embedding model's
                           max_seq_length. Chunks exceeding this are re-split to prevent
                           OOM during embedding. (None = no hard limit)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chars = max_chars or int(chunk_size * self.CHARS_PER_TOKEN)
        self.hard_max_chars = hard_max_chars

        if not LLAMA_INDEX_AVAILABLE:
            logger.warning(
                "llama-index-core not available. "
                "AST chunking will fall back to line-based for all files."
            )

    def chunk_code(
        self,
        file_path: str,
        code: str,
        force_line_based: bool = False
    ) -> List[Chunk]:
        """Chunk code using AST with automatic fallback to line-based.

        Args:
            file_path: Path to source file (used for language detection)
            code: Source code content
            force_line_based: If True, skip AST and use line-based chunking

        Returns:
            List of Chunk objects with content and extraction method

        The extraction method will be one of:
        - "ast_{language}": AST-based chunking succeeded
        - "line_based": AST parsing failed or was skipped, used line-based fallback
        """
        if not code or not code.strip():
            # Empty file
            return []

        # Detect language from file extension
        file_ext = Path(file_path).suffix.lower()
        language = self.LANGUAGE_MAP.get(file_ext, "text")

        # Try AST-based chunking if available and not forced to line-based
        if not force_line_based and LLAMA_INDEX_AVAILABLE and language != "text":
            try:
                chunks = self._chunk_with_ast(code, language)
                if chunks:
                    extraction_method = f"ast_{language}"
                    result = [Chunk(content=c, method=extraction_method) for c in chunks]
                    # Enforce hard_max_chars limit if set
                    if self.hard_max_chars:
                        result = self._enforce_hard_limit(result)
                    return result
            except Exception as e:
                logger.debug(f"AST parsing failed for {file_path} ({language}): {e}")
                # Fall through to line-based chunking

        # Fallback to line-based chunking
        logger.debug(f"Using line-based chunking for {file_path}")
        chunks = self._chunk_line_based(code)
        result = [Chunk(content=c, method="line_based") for c in chunks]

        # Enforce hard_max_chars limit if set - re-split any oversized chunks
        if self.hard_max_chars:
            result = self._enforce_hard_limit(result)

        return result

    def _chunk_with_ast(self, code: str, language: str) -> List[str]:
        """Chunk code using AST parsing via LlamaIndex CodeSplitter.

        Args:
            code: Source code content
            language: Tree-sitter language name

        Returns:
            List of code chunk strings

        Raises:
            Exception: If AST parsing fails
        """
        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("llama-index-core not available")

        try:
            # Create CodeSplitter with tree-sitter
            splitter = CodeSplitter(
                language=language,
                chunk_lines=self.chunk_size,  # Target lines (roughly equivalent to tokens)
                chunk_lines_overlap=self.chunk_overlap,
                max_chars=self.max_chars
            )

            # Split code into chunks
            chunks = splitter.split_text(code)

            if not chunks:
                raise ValueError("CodeSplitter returned empty chunks")

            # Filter out empty chunks (can happen with certain AST edge cases)
            chunks = [c for c in chunks if c and c.strip()]

            if not chunks:
                raise ValueError("All chunks were empty after filtering")

            return chunks

        except Exception as e:
            # Log but re-raise so caller can handle fallback
            logger.debug(f"AST splitting failed for {language}: {e}")
            raise

    def _split_long_line(self, line: str, max_chars: int) -> List[str]:
        """Split a very long line into smaller segments.

        Used for minified code where a single line can be the entire file.
        Splits at max_chars boundaries, preferring natural break points
        (semicolons, commas, braces) when possible.

        Args:
            line: The long line to split
            max_chars: Maximum characters per segment

        Returns:
            List of line segments
        """
        if len(line) <= max_chars:
            return [line]

        segments = []
        pos = 0

        while pos < len(line):
            # Take up to max_chars
            end = min(pos + max_chars, len(line))

            if end < len(line):
                # Try to find a natural break point (semicolon, comma, closing brace)
                # within the last 20% of the segment
                search_start = pos + int(max_chars * 0.8)
                best_break = -1

                for break_char in [';', ',', '}', ')', ']']:
                    break_pos = line.rfind(break_char, search_start, end)
                    if break_pos > best_break:
                        best_break = break_pos

                if best_break > search_start:
                    end = best_break + 1  # Include the break character

            segments.append(line[pos:end])
            pos = end

        return segments

    def _chunk_line_based(self, code: str) -> List[str]:
        """Fallback line-based chunking when AST parsing fails.

        Simple strategy:
        1. Split code into lines
        2. Split very long lines (e.g., minified code) into smaller segments
        3. Group lines into chunks of approximately chunk_size tokens
        4. Add overlap between chunks

        Args:
            code: Source code content

        Returns:
            List of code chunk strings
        """
        raw_lines = code.split('\n')

        if not raw_lines:
            return []

        # Maximum characters per line segment (to prevent memory issues)
        # Use chunk_size * CHARS_PER_TOKEN as the limit
        max_line_chars = int(self.chunk_size * self.CHARS_PER_TOKEN)

        # Pre-process: split very long lines (handles minified code)
        lines = []
        for line in raw_lines:
            if len(line) > max_line_chars:
                # Split long line into manageable segments
                segments = self._split_long_line(line, max_line_chars)
                lines.extend(segments)
            else:
                lines.append(line)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for line in lines:
            # Estimate tokens in line (chars / CHARS_PER_TOKEN)
            line_tokens = len(line) / self.CHARS_PER_TOKEN

            # Check if adding this line would exceed chunk size
            if current_tokens + line_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))

                # Start new chunk with overlap
                overlap_lines = max(0, self.chunk_overlap // 10)  # Rough overlap
                current_chunk = current_chunk[-overlap_lines:] if overlap_lines > 0 else []
                current_tokens = sum(len(l) / self.CHARS_PER_TOKEN for l in current_chunk)

            # Add line to current chunk
            current_chunk.append(line)
            current_tokens += line_tokens

        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        # Filter out empty chunks
        chunks = [c for c in chunks if c and c.strip()]

        return chunks if chunks else [code]  # Return original if no chunks created

    def _enforce_hard_limit(self, chunks: List[Chunk]) -> List[Chunk]:
        """Re-split any chunks exceeding hard_max_chars to prevent embedding OOM.

        This is a post-processing step that ensures no chunk exceeds the embedding
        model's capacity. Oversized chunks are split at natural break points
        (newlines, then punctuation) with overlap to preserve context continuity.

        Args:
            chunks: List of Chunk objects from AST or line-based chunking

        Returns:
            List of Chunk objects with all chunks <= hard_max_chars
        """
        if not self.hard_max_chars:
            return chunks

        # Calculate overlap for re-split chunks (15% of hard_max_chars, similar to normal chunking)
        overlap_chars = int(self.hard_max_chars * 0.15)

        result = []
        split_count = 0

        for chunk in chunks:
            if len(chunk.content) <= self.hard_max_chars:
                result.append(chunk)
            else:
                # Need to split this chunk with overlap
                split_count += 1
                sub_chunks = self._split_oversized_chunk(
                    chunk.content, self.hard_max_chars, overlap_chars
                )
                for sub in sub_chunks:
                    result.append(Chunk(content=sub, method=f"{chunk.method}_resplit"))

        if split_count > 0:
            logger.info(
                f"Re-split {split_count} oversized chunks exceeding {self.hard_max_chars} chars "
                f"(total chunks: {len(chunks)} -> {len(result)}, overlap: {overlap_chars} chars)"
            )

        return result

    def _split_oversized_chunk(self, text: str, max_chars: int, overlap_chars: int = 0) -> List[str]:
        """Split an oversized chunk into smaller pieces with overlap.

        Uses a sliding window approach to maintain context continuity across chunks.
        Each chunk (except the first) includes overlap_chars from the end of the
        previous chunk's content, providing context for large functions that span
        multiple chunks.

        Tries to split at natural boundaries in order of preference:
        1. Double newlines (paragraph breaks)
        2. Single newlines
        3. Sentence-ending punctuation followed by space
        4. Other punctuation (semicolons, braces)
        5. Hard cut at max_chars if no break point found

        Args:
            text: The oversized text to split
            max_chars: Maximum characters per resulting chunk
            overlap_chars: Number of characters to overlap between chunks

        Returns:
            List of text chunks, each <= max_chars, with overlap for context
        """
        if len(text) <= max_chars:
            return [text]

        result = []
        pos = 0

        while pos < len(text):
            # Determine how much text to consider for this chunk
            # For first chunk, use full max_chars
            # For subsequent chunks, we'll prepend overlap from previous
            is_first_chunk = (pos == 0)

            if is_first_chunk:
                available_chars = max_chars
            else:
                # Subsequent chunks: include overlap at start, so less room for new content
                available_chars = max_chars - overlap_chars

            end_pos = pos + available_chars

            if end_pos >= len(text):
                # Last chunk - take everything remaining
                if is_first_chunk:
                    result.append(text[pos:])
                else:
                    # Prepend overlap from previous chunk
                    overlap_start = max(0, pos - overlap_chars)
                    result.append(text[overlap_start:])
                break

            # Try to find a good split point within available space
            segment = text[pos:end_pos]
            split_offset = -1  # Offset within segment

            # Try double newline first (paragraph break)
            nl_pos = segment.rfind('\n\n')
            if nl_pos > available_chars * 0.5:  # Only if at least halfway through
                split_offset = nl_pos + 2

            # Try single newline
            if split_offset == -1:
                nl_pos = segment.rfind('\n')
                if nl_pos > available_chars * 0.3:
                    split_offset = nl_pos + 1

            # Try sentence endings
            if split_offset == -1:
                for ending in ['. ', '.\n', '? ', '?\n', '! ', '!\n']:
                    end_pos_in_seg = segment.rfind(ending)
                    if end_pos_in_seg > available_chars * 0.3:
                        split_offset = end_pos_in_seg + len(ending)
                        break

            # Try code-specific break points
            if split_offset == -1:
                for brk in [';\n', '},\n', '}\n', ',\n', '; ', '}, ', '} ']:
                    brk_pos = segment.rfind(brk)
                    if brk_pos > available_chars * 0.3:
                        split_offset = brk_pos + len(brk)
                        break

            # Hard cut if no good break point found
            if split_offset == -1:
                split_offset = available_chars

            # Build the chunk content
            if is_first_chunk:
                chunk_content = text[pos:pos + split_offset]
            else:
                # Prepend overlap from previous chunk's end
                overlap_start = max(0, pos - overlap_chars)
                chunk_content = text[overlap_start:pos + split_offset]

            result.append(chunk_content)

            # Move position forward (no overlap in position - overlap is in content)
            pos = pos + split_offset

        return result

    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension.

        Args:
            file_path: Path to source file

        Returns:
            Language name for tree-sitter, or None if unknown
        """
        file_ext = Path(file_path).suffix.lower()
        return self.LANGUAGE_MAP.get(file_ext)

    def supports_ast_chunking(self, file_path: str) -> bool:
        """Check if AST chunking is supported for this file.

        Args:
            file_path: Path to source file

        Returns:
            True if AST chunking is available for this file type
        """
        if not LLAMA_INDEX_AVAILABLE:
            return False

        language = self.detect_language(file_path)
        return language is not None and language != "text"

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of supported programming languages.

        Returns:
            List of language names supported by AST chunking
        """
        # Get unique language names from the mapping
        return sorted(set(cls.LANGUAGE_MAP.values()))

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions.

        Returns:
            List of file extensions supported by AST chunking
        """
        return sorted(cls.LANGUAGE_MAP.keys())


def chunk_code_file(
    file_path: str,
    chunk_size: int = 400,
    chunk_overlap: int = 20
) -> List[Chunk]:
    """Convenience function to chunk a code file.

    Args:
        file_path: Path to source code file
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        List of Chunk objects

    Raises:
        FileNotFoundError: If file does not exist
        IOError: If file cannot be read
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except UnicodeDecodeError:
        # Try with latin-1 as fallback
        with open(file_path, 'r', encoding='latin-1') as f:
            code = f.read()

    chunker = ASTCodeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_code(file_path, code)

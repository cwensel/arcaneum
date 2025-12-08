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
        max_chars: Optional[int] = None
    ):
        """Initialize AST code chunker.

        Args:
            chunk_size: Target chunk size in tokens
                       - 400 tokens for 8K context models (jina-v2-base-code)
                       - 2000-4000 tokens for 32K context models (jina-code-1.5b)
            chunk_overlap: Overlap between chunks in tokens (default 5%)
            max_chars: Maximum characters per chunk (None = auto-calculate)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chars = max_chars or int(chunk_size * self.CHARS_PER_TOKEN)

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
                    return [Chunk(content=c, method=extraction_method) for c in chunks]
            except Exception as e:
                logger.debug(f"AST parsing failed for {file_path} ({language}): {e}")
                # Fall through to line-based chunking

        # Fallback to line-based chunking
        logger.debug(f"Using line-based chunking for {file_path}")
        chunks = self._chunk_line_based(code)
        return [Chunk(content=c, method="line_based") for c in chunks]

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

    def _chunk_line_based(self, code: str) -> List[str]:
        """Fallback line-based chunking when AST parsing fails.

        Simple strategy:
        1. Split code into lines
        2. Group lines into chunks of approximately chunk_size tokens
        3. Add overlap between chunks

        Args:
            code: Source code content

        Returns:
            List of code chunk strings
        """
        lines = code.split('\n')

        if not lines:
            return []

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

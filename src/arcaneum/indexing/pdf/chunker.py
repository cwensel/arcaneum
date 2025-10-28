"""PDF chunking with semantic awareness and late chunking support (RDR-004)."""

from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_index: int
    token_count: int
    metadata: Dict


class PDFChunker:
    """Chunk PDF text with semantic awareness and late chunking support."""

    def __init__(
        self,
        model_config: Dict,
        overlap_percent: float = 0.15,
        late_chunking_enabled: bool = True,
        min_doc_tokens: int = 2000,
        max_doc_tokens: int = 8000
    ):
        """Initialize PDF chunker.

        Args:
            model_config: Model configuration with chunk_size, char_to_token_ratio, etc.
            overlap_percent: Overlap between chunks (default 0.15 = 15%)
            late_chunking_enabled: Enable late chunking for long documents
            min_doc_tokens: Minimum document length for late chunking
            max_doc_tokens: Maximum document length for late chunking
        """
        self.model_config = model_config
        self.overlap_percent = overlap_percent
        self.late_chunking_enabled = late_chunking_enabled
        self.min_doc_tokens = min_doc_tokens
        self.max_doc_tokens = max_doc_tokens

        self.chunk_size = model_config['chunk_size']
        self.chunk_overlap = int(self.chunk_size * overlap_percent)

    def chunk(self, text: str, metadata: Dict) -> List[Chunk]:
        """Chunk text using appropriate strategy.

        Strategies:
        1. Late chunking: For documents 2K-8K tokens (if supported by model)
        2. Traditional chunking: Token-aware splitting with overlap

        Args:
            text: Text to chunk
            metadata: Base metadata to attach to all chunks

        Returns:
            List of Chunk objects
        """
        # Estimate token count (rough approximation)
        char_to_token = self.model_config.get('char_to_token_ratio', 3.3)
        estimated_tokens = len(text) / char_to_token

        # Select chunking strategy
        if (self.late_chunking_enabled and
            self.model_config.get('late_chunking', False) and
            self.min_doc_tokens < estimated_tokens < self.max_doc_tokens):

            logger.info(f"Using late chunking (doc tokens: {estimated_tokens:.0f})")
            return self._late_chunking(text, metadata)

        else:
            logger.info(f"Using traditional chunking (doc tokens: {estimated_tokens:.0f})")
            return self._traditional_chunking(text, metadata)

    def _late_chunking(self, text: str, metadata: Dict) -> List[Chunk]:
        """Implement late chunking strategy.

        Note: This is a simplified example. Production implementation would:
        1. Embed entire document first
        2. Apply mean pooling to chunk-sized windows of token embeddings
        3. Return contextual chunk embeddings

        For jina-v3, use API parameter: late_chunking=True
        For stella/modernbert, implement custom mean pooling after embedding.
        """
        # For now, return traditional chunks with metadata flag
        # Actual late chunking happens in embedding phase
        chunks = self._traditional_chunking(text, metadata)

        # Mark chunks for late chunking processing
        for chunk in chunks:
            chunk.metadata['late_chunking'] = True

        return chunks

    def _traditional_chunking(self, text: str, metadata: Dict) -> List[Chunk]:
        """Traditional token-aware chunking with overlap."""
        chunks = []
        char_to_token = self.model_config.get('char_to_token_ratio', 3.3)

        # Calculate character limits
        chunk_chars = int(self.chunk_size * char_to_token)
        overlap_chars = int(self.chunk_overlap * char_to_token)

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_chars

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence boundary in last 20% of chunk
                search_start = end - int(chunk_chars * 0.2)
                sentence_end = text.rfind('. ', search_start, end)

                if sentence_end != -1:
                    end = sentence_end + 1  # Include the period

            chunk_text = text[start:end].strip()

            if chunk_text:
                # Estimate token count
                token_count = int(len(chunk_text) / char_to_token)

                # Calculate page number if page boundaries available
                page_number = self._calculate_page_number(start, metadata.get('page_boundaries'))

                chunk_metadata = {
                    **metadata,
                    'chunk_index': chunk_index,
                    'chunk_start_char': start,
                    'chunk_end_char': end,
                    'late_chunking': False,
                }

                # Add page_number if calculated
                if page_number is not None:
                    chunk_metadata['page_number'] = page_number

                chunk = Chunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    metadata=chunk_metadata
                )

                chunks.append(chunk)
                chunk_index += 1

            # Move start position (with overlap)
            start = end - overlap_chars

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def _calculate_page_number(self, chunk_start_char: int, page_boundaries: List[Dict]) -> Optional[int]:
        """Calculate which page a chunk belongs to based on its character position.

        Args:
            chunk_start_char: Starting character position of chunk
            page_boundaries: List of dicts with page_number, start_char, page_text_length

        Returns:
            Page number (1-indexed) or None if page_boundaries not available
        """
        if not page_boundaries:
            return None

        # Find the page this chunk starts in
        for page in page_boundaries:
            page_start = page['start_char']
            page_end = page_start + page['page_text_length']

            if page_start <= chunk_start_char < page_end:
                return page['page_number']

        # If not found (edge case), return last page
        # This handles chunks at exact page boundaries
        if page_boundaries:
            return page_boundaries[-1]['page_number']

        return None

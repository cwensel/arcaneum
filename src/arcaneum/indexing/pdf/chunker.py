"""PDF chunking with semantic awareness and late chunking support (RDR-004)."""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

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
        max_doc_tokens: int = 8000,
    ):
        """Initialize PDF chunker.

        Args:
            model_config: Model configuration with chunk_size, char_to_token_ratio, etc.
            overlap_percent: Overlap between chunks (default 0.15 = 15%)
            late_chunking_enabled: Enable late chunking for long documents
            min_doc_tokens: Minimum document length for late chunking
            max_doc_tokens: Maximum document length for late chunking
        """
        if not 0 <= overlap_percent < 1:
            raise ValueError("overlap_percent must be at least 0 and less than 1")

        self.model_config = model_config
        self.overlap_percent = overlap_percent
        self.late_chunking_enabled = late_chunking_enabled
        self.min_doc_tokens = min_doc_tokens
        self.max_doc_tokens = max_doc_tokens

        self.chunk_size = model_config["chunk_size"]
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
        char_to_token = self.model_config.get("char_to_token_ratio", 3.3)
        estimated_tokens = len(text) / char_to_token

        # Select chunking strategy
        if (
            self.late_chunking_enabled
            and self.model_config.get("late_chunking", False)
            and self.min_doc_tokens < estimated_tokens < self.max_doc_tokens
        ):
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
            chunk.metadata["late_chunking"] = True

        return chunks

    def _traditional_chunking(self, text: str, metadata: Dict) -> List[Chunk]:
        """Traditional token-aware chunking with overlap."""
        chunks = []
        char_to_token = self.model_config.get("char_to_token_ratio", 3.3)

        # Calculate character limits
        chunk_chars = max(1, int(self.chunk_size * char_to_token))
        overlap_chars = max(0, int(self.chunk_overlap * char_to_token))
        boundary_window = max(1, (chunk_chars + 4) // 5)

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_chars, len(text))
            if end < len(text):
                end = self._find_end_boundary(text, start, end, boundary_window)

            chunk_start, chunk_end = self._trim_span(text, start, end)
            chunk_text = text[chunk_start:chunk_end]

            if chunk_text:
                # Estimate token count
                token_count = int(len(chunk_text) / char_to_token)

                # Calculate page number if page boundaries available
                page_number = self._calculate_page_number(
                    chunk_start, metadata.get("page_boundaries")
                )

                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index,
                    "chunk_start_char": chunk_start,
                    "chunk_end_char": chunk_end,
                    "late_chunking": False,
                }

                # Add page_number if calculated
                if page_number is not None:
                    chunk_metadata["page_number"] = page_number

                chunk = Chunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    metadata=chunk_metadata,
                )

                chunks.append(chunk)
                chunk_index += 1

            if end >= len(text):
                break

            # Move start position (with overlap)
            overlap_end = chunk_end if chunk_text else end
            desired_start = max(start + 1, overlap_end - overlap_chars)
            next_start = self._snap_start_to_whitespace(
                text,
                desired_start,
                boundary_window,
                minimum=start + 1,
                maximum=overlap_end,
            )
            start = max(start + 1, next_start)

        chunk_count = len(chunks)
        for chunk in chunks:
            chunk.metadata["chunk_count"] = chunk_count

        logger.info(f"Created {chunk_count} chunks")
        return chunks

    @staticmethod
    def _find_end_boundary(text: str, start: int, end: int, window: int) -> int:
        """Choose the strongest boundary near the target chunk end."""
        search_start = max(start + 1, end - window)
        search_text = text[search_start : end + 1]

        paragraph_matches = list(re.finditer(r"\n[ \t]*\n", search_text))
        if paragraph_matches:
            return search_start + paragraph_matches[-1].start()

        sentence_matches = list(re.finditer(r"[.!?](?=\s)", search_text))
        if sentence_matches:
            return search_start + sentence_matches[-1].end()

        for boundary in range(end, search_start - 1, -1):
            if boundary < len(text) and text[boundary].isspace():
                return boundary
            if boundary > start and text[boundary - 1].isspace():
                return boundary

        return end

    @staticmethod
    def _snap_start_to_whitespace(
        text: str,
        desired: int,
        window: int,
        *,
        minimum: int,
        maximum: int,
    ) -> int:
        """Snap an overlap-derived start to the nearest word boundary."""
        desired = min(desired, len(text))
        if desired == 0 or (desired > 0 and text[desired - 1].isspace()):
            return desired

        search_start = max(minimum, desired - window)
        search_end = min(len(text), maximum + 1, desired + window + 1)
        candidates = []
        position = search_start
        while position < search_end:
            if not text[position].isspace():
                position += 1
                continue

            while position < len(text) and text[position].isspace():
                position += 1
            if minimum <= position <= maximum + 1:
                candidates.append(position)

        if candidates:
            return min(candidates, key=lambda candidate: (abs(candidate - desired), candidate))

        return desired

    @staticmethod
    def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
        """Return source offsets after removing edge whitespace."""
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    def _calculate_page_number(
        self, chunk_start_char: int, page_boundaries: List[Dict]
    ) -> Optional[int]:
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
            page_start = page["start_char"]
            page_end = page_start + page["page_text_length"]

            if page_start <= chunk_start_char < page_end:
                return page["page_number"]

        # If not found (edge case), return last page
        # This handles chunks at exact page boundaries
        if page_boundaries:
            return page_boundaries[-1]["page_number"]

        return None

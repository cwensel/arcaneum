"""PDF chunking with semantic awareness and late chunking support (RDR-004)."""

import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_REFERENCE_HEADINGS = {"references", "bibliography", "works cited", "literature cited"}
_ACKNOWLEDGEMENT_HEADINGS = {
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",
}
_CONTENTS_HEADINGS = {"contents", "table of contents"}
_BODY_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "methods",
    "methodology",
    "materials and methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
}


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    text: str
    chunk_index: int
    token_count: int
    metadata: Dict


class ChunkList(list):
    """Chunks plus file-level replacement-character omission details."""

    def __init__(
        self, chunks: Iterable[Chunk] = (), *, replacement_omissions: Optional[Dict] = None
    ):
        super().__init__(chunks)
        self.replacement_omissions = replacement_omissions or {
            "source_character_count": 0,
            "retained_character_count": 0,
            "character_count": 0,
            "replacement_ratio": 0.0,
            "degraded": False,
            "singleton_count": 0,
            "multi_character_run_count": 0,
            "multi_character_run_character_count": 0,
            "hard_boundary_count": 0,
            "hard_boundary_character_count": 0,
            "normalized_run_count": 0,
            "normalized_run_character_count": 0,
        }


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
        self.min_chunk_chars = model_config.get("min_chunk_chars", 200)
        if self.min_chunk_chars < 0:
            raise ValueError("min_chunk_chars must be at least 0")

    def chunk(self, text: str, metadata: Dict) -> ChunkList:
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

    def _late_chunking(self, text: str, metadata: Dict) -> ChunkList:
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

    def _traditional_chunking(self, text: str, metadata: Dict) -> ChunkList:
        """Traditional token-aware chunking with overlap.

        U+FFFD characters are normalized to spaces so nearby valid text stays
        searchable. Runs of two or more become hard omitted boundaries only
        when both adjacent regions contain enough meaningful text; other runs
        remain length-preserving spaces to avoid tiny-chunk amplification.
        Source offsets remain coordinates in the original extracted text.
        """
        char_to_token = self.model_config.get("char_to_token_ratio", 3.3)
        chunk_chars = max(1, int(self.chunk_size * char_to_token))
        overlap_chars = max(0, int(self.chunk_overlap * char_to_token))
        boundary_window = max(1, (chunk_chars + 4) // 5)

        regions, replacement_omissions = self._replacement_regions(text)
        section_markers = self._section_markers(text)
        spans = []
        for source_start, normalized_region in regions:
            region_end = source_start + len(normalized_region)
            section_offsets = [
                offset - source_start
                for offset, _, _ in section_markers
                if source_start <= offset < region_end
            ]
            segment_starts = sorted({0, *section_offsets})
            segment_ends = [*segment_starts[1:], len(normalized_region)]
            for segment_start, segment_end in zip(segment_starts, segment_ends):
                segment = normalized_region[segment_start:segment_end]
                region_spans = self._chunk_region_spans(
                    segment,
                    chunk_chars=chunk_chars,
                    overlap_chars=overlap_chars,
                    boundary_window=boundary_window,
                )
                # Merge only within one clean section. Replacement runs and
                # section headings are both hard semantic boundaries.
                region_spans = self._merge_fragment_spans(region_spans)
                spans.extend(
                    (
                        source_start + segment_start + chunk_start,
                        source_start + segment_start + chunk_end,
                    )
                    for chunk_start, chunk_end in region_spans
                )

        chunk_count = len(spans)
        if replacement_omissions["character_count"] and chunk_count == 0:
            replacement_omissions["degraded"] = True
        chunks = []
        for chunk_index, (chunk_start, chunk_end) in enumerate(spans):
            source_text = text[chunk_start:chunk_end]
            chunk_text = source_text.replace("\ufffd", " ")
            chunk_metadata = {
                **metadata,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "chunk_start_char": chunk_start,
                "chunk_end_char": chunk_end,
                "late_chunking": False,
            }
            normalized_count = source_text.count("\ufffd")
            if normalized_count:
                chunk_metadata.update(
                    {
                        "chunk_text_normalized": True,
                        "normalized_replacement_character_count": normalized_count,
                        "chunk_source_offset_semantics": "original_extracted_text_half_open",
                    }
                )
            page_number = self._calculate_page_number(chunk_start, metadata.get("page_boundaries"))
            if page_number is not None:
                chunk_metadata["page_number"] = page_number
            section_type, section_title, section_offset = self._section_at_offset(
                section_markers, chunk_start
            )
            chunk_metadata["section_type"] = section_type
            chunk_metadata["section_title"] = section_title
            chunk_metadata["section_offset"] = section_offset

            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    token_count=int(len(chunk_text) / char_to_token),
                    metadata=chunk_metadata,
                )
            )

        logger.info(f"Created {chunk_count} chunks")
        return ChunkList(chunks, replacement_omissions=replacement_omissions)

    @staticmethod
    def _section_type(title: str) -> str:
        normalized = re.sub(r"\s+", " ", title.strip().rstrip(":")).casefold()
        normalized = re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", normalized)
        if normalized in _REFERENCE_HEADINGS:
            return "references"
        if normalized in _ACKNOWLEDGEMENT_HEADINGS:
            return "acknowledgements"
        if normalized in _CONTENTS_HEADINGS:
            return "contents"
        if normalized.startswith("appendix"):
            return "appendix"
        return "body"

    @classmethod
    def _section_markers(cls, text: str) -> List[tuple[int, str, str]]:
        """Return conservative academic section headings and their source offsets."""
        markers = []
        known_bare = (
            _REFERENCE_HEADINGS | _ACKNOWLEDGEMENT_HEADINGS | _CONTENTS_HEADINGS | _BODY_HEADINGS
        )
        for line_match in re.finditer(r"(?m)^[^\r\n]+", text):
            raw_line = line_match.group(0)
            title = raw_line.strip()
            if not title:
                continue
            semantic_title = re.sub(r"^#{1,6}\s+", "", title).strip()
            normalized = re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", semantic_title.casefold()).rstrip(
                ":"
            )
            is_known_bare = normalized in known_bare or normalized.startswith("appendix")
            if not is_known_bare:
                continue
            offset = line_match.start() + len(raw_line) - len(raw_line.lstrip())
            markers.append((offset, cls._section_type(semantic_title), semantic_title))
        return markers

    @staticmethod
    def _section_at_offset(
        markers: List[tuple[int, str, str]], offset: int
    ) -> tuple[str, Optional[str], Optional[int]]:
        active = None
        for marker in markers:
            if marker[0] > offset:
                break
            active = marker
        if active is None:
            return "unknown", None, None
        return active[1], active[2], active[0]

    def _chunk_region_spans(
        self,
        text: str,
        *,
        chunk_chars: int,
        overlap_chars: int,
        boundary_window: int,
    ) -> List[tuple[int, int]]:
        """Return local spans for one region that cannot cross a hard boundary."""
        spans = []
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            if end < len(text):
                end = self._find_end_boundary(text, start, end, boundary_window)

            chunk_start, chunk_end = self._trim_span(text, start, end)
            if chunk_start < chunk_end:
                spans.append((chunk_start, chunk_end))

            if end >= len(text):
                break

            overlap_end = chunk_end if chunk_start < chunk_end else end
            desired_start = max(start + 1, overlap_end - overlap_chars)
            next_start = self._snap_start_to_whitespace(
                text,
                desired_start,
                boundary_window,
                minimum=start + 1,
                maximum=overlap_end,
            )
            start = max(start + 1, next_start)

        return spans

    def _replacement_regions(self, text: str) -> tuple[List[tuple[int, str]], Dict]:
        """Split only meaningful regions on U+FFFD runs.

        A run becomes a hard boundary only when the immediately adjacent
        clean regions are both at least ``max(32, min_chunk_chars)`` chars.
        Shorter neighbors would create low-value tiny chunks, so those runs
        are length-preserving-normalized to spaces instead. Only aggregate
        counts are retained to keep manifests bounded for hostile input.
        """
        matches = list(re.finditer(r"\ufffd{2,}", text))
        segment_starts = [0, *(match.end() for match in matches)]
        segment_ends = [*(match.start() for match in matches), len(text)]
        segments = [text[start:end] for start, end in zip(segment_starts, segment_ends)]
        safety_floor = max(32, self.min_chunk_chars)
        meaningful_character_counts = [
            sum(1 for character in segment if not character.isspace() and character != "\ufffd")
            for segment in segments
        ]
        hard_boundaries = [
            meaningful_character_counts[index] >= safety_floor
            and meaningful_character_counts[index + 1] >= safety_floor
            for index in range(len(matches))
        ]

        regions = []
        region_start = 0
        region_parts = [segments[0]]
        for index, match in enumerate(matches):
            if hard_boundaries[index]:
                raw_region = "".join(region_parts)
                if raw_region:
                    regions.append((region_start, raw_region.replace("\ufffd", " ")))
                region_start = match.end()
                region_parts = [segments[index + 1]]
            else:
                region_parts.extend((" " * (match.end() - match.start()), segments[index + 1]))

        raw_region = "".join(region_parts)
        if raw_region:
            regions.append((region_start, raw_region.replace("\ufffd", " ")))

        multi_character_run_character_count = sum(match.end() - match.start() for match in matches)
        hard_boundary_character_count = sum(
            match.end() - match.start()
            for match, is_hard in zip(matches, hard_boundaries)
            if is_hard
        )
        hard_boundary_count = sum(hard_boundaries)
        source_character_count = len(text)
        character_count = text.count("\ufffd")
        retained_character_count = source_character_count - character_count
        replacement_ratio = (
            character_count / source_character_count if source_character_count else 0.0
        )
        return regions, {
            "source_character_count": source_character_count,
            "retained_character_count": retained_character_count,
            "character_count": character_count,
            "replacement_ratio": replacement_ratio,
            "degraded": replacement_ratio > 0.05,
            "singleton_count": character_count - multi_character_run_character_count,
            "multi_character_run_count": len(matches),
            "multi_character_run_character_count": multi_character_run_character_count,
            "hard_boundary_count": hard_boundary_count,
            "hard_boundary_character_count": hard_boundary_character_count,
            "normalized_run_count": len(matches) - hard_boundary_count,
            "normalized_run_character_count": (
                multi_character_run_character_count - hard_boundary_character_count
            ),
        }

    def _merge_fragment_spans(self, spans: List[tuple[int, int]]) -> List[tuple[int, int]]:
        """Merge sub-floor source spans into an adjacent chunk.

        Leading and interior fragment runs attach forward. A trailing run
        attaches backward. If the whole document is shorter than the floor,
        all spans collapse to the one allowed sub-floor document chunk.
        """
        if not self.min_chunk_chars or len(spans) < 2:
            return spans

        groups: List[List[tuple[int, int]]] = []
        pending: List[tuple[int, int]] = []
        for span in spans:
            pending.append(span)
            if span[1] - span[0] >= self.min_chunk_chars:
                groups.append(pending)
                pending = []

        if pending:
            if groups:
                groups[-1].extend(pending)
            else:
                groups.append(pending)

        merged = [(group[0][0], max(end for _, end in group)) for group in groups]
        merged_count = len(spans) - len(merged)
        if merged_count:
            logger.info(
                "Merged %d PDF fragments below %d chars into adjacent chunks",
                merged_count,
                self.min_chunk_chars,
            )
        return merged

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

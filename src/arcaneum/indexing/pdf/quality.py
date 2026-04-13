"""Text quality scoring and layout detection for PDF extractions.

Used by --repair to identify indexed chunks with unreadable text
(replacement characters, encoding garbage, CID font mapping failures)
that need re-extraction with the updated pymupdf4llm auto-OCR.

Also provides multi-column layout detection for routing to advanced
extractors (RDR-022/023).
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache for detect_multi_column results, keyed by (file_path, file_mtime).
# Avoids repeated geometry analysis on the same file during a single process
# (e.g., corpus repair scanning 148 PDFs). Reset on process restart.
# See RDR-022 Component 2: "Detection results should be cached per-file."
_column_detection_cache: dict = {}

# High-frequency English stop words for readability detection
STOP_WORDS = frozenset({
    'the', 'of', 'and', 'is', 'in', 'to', 'for', 'a', 'that', 'with',
    'on', 'are', 'was', 'it', 'be', 'as', 'at', 'by', 'this', 'from',
    'or', 'an', 'have', 'has', 'not', 'but', 'can', 'which', 'their',
    'will', 'been', 'would', 'each', 'were', 'do', 'there',
})


def score_text(text: str) -> float:
    """Score text readability from 0.0 (garbage) to 1.0 (clean English).

    Combines four signals:
    - Replacement character ratio (U+FFFD) — dominant garbage signal
    - English stop word frequency — readable text has predictable stop word rates
    - ASCII printable ratio — clean text is mostly printable ASCII
    - Average word length — garbage tends to have very long or very short tokens

    Args:
        text: Text to score

    Returns:
        Float 0.0-1.0 where <0.3 indicates garbage text
    """
    if not text or not text.strip():
        return 0.0

    text_len = len(text)

    # Signal 1: Replacement character ratio (weight: 0.35)
    replacement_count = text.count('\ufffd')
    replacement_ratio = replacement_count / text_len
    replacement_score = max(0.0, 1.0 - (replacement_ratio * 10))

    # Signal 2: Stop word ratio (weight: 0.30)
    words = re.findall(r'[a-zA-Z]+', text.lower())
    if words:
        stop_count = sum(1 for w in words if w in STOP_WORDS)
        stop_ratio = stop_count / len(words)
        # English text typically has 20-40% stop words
        stop_score = min(1.0, stop_ratio / 0.15)
    else:
        stop_score = 0.0

    # Signal 3: ASCII printable ratio (weight: 0.20)
    printable_count = sum(1 for c in text if 32 <= ord(c) <= 126 or c in '\n\r\t')
    ascii_ratio = printable_count / text_len
    ascii_score = min(1.0, ascii_ratio / 0.7)

    # Signal 4: Average word length penalty (weight: 0.15)
    if words:
        avg_len = sum(len(w) for w in words) / len(words)
        if avg_len > 15 or avg_len < 2:
            length_score = 0.2
        elif avg_len > 10:
            length_score = 0.6
        else:
            length_score = 1.0
    else:
        length_score = 0.0

    score = (
        0.35 * replacement_score
        + 0.30 * stop_score
        + 0.20 * ascii_score
        + 0.15 * length_score
    )

    return round(min(1.0, max(0.0, score)), 3)


def score_chunks(chunks: list) -> float:
    """Average quality score across a list of chunk dicts.

    Args:
        chunks: List of dicts with 'text' keys

    Returns:
        Average score (0.0-1.0), or 0.0 if no text found
    """
    scores = [score_text(c['text']) for c in chunks if c.get('text')]
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def is_readable(text: str, threshold: float = 0.9) -> bool:
    """Check if text is readable (above quality threshold).

    Args:
        text: Text to check
        threshold: Minimum quality score (default 0.9)

    Returns:
        True if text scores above threshold
    """
    return score_text(text) >= threshold


def needs_ocr(text: str) -> bool:
    """Check if text has garbled content that OCR could fix.

    Distinct from is_readable(): a PDF with chart bullet characters (●●●)
    scores low on readability but OCR won't help — the extraction is correct.
    This function specifically checks for replacement characters (U+FFFD)
    and near-zero stop word frequency, which indicate broken font mappings
    that OCR can recover.

    Args:
        text: Text to check

    Returns:
        True if text appears garbled and OCR would likely improve it
    """
    if not text or not text.strip():
        return True

    text_len = len(text)

    # High replacement character ratio = broken font mapping → OCR helps
    replacement_ratio = text.count('\ufffd') / text_len
    if replacement_ratio > 0.05:
        return True

    # No English words at all — check if it's encoding garbage or just non-text content.
    # Non-text content (chart bullets ●, math symbols) has valid Unicode but no words.
    # Encoding garbage has high non-printable or non-Unicode-category chars.
    words = re.findall(r'[a-zA-Z]+', text.lower())
    if not words and text_len > 100:
        # Only flag as needing OCR if text is mostly non-printable ASCII
        # (not just Unicode symbols like ● which are valid extraction)
        printable_count = sum(1 for c in text if 32 <= ord(c) <= 126 or c in '\n\r\t')
        if printable_count / text_len < 0.1:
            return True

    # Has some words but zero stop words in a large sample = likely garbled
    if words and len(words) > 20:
        stop_count = sum(1 for w in words if w in STOP_WORDS)
        if stop_count == 0:
            return True

    return False


def has_column_interleaving_artifacts(text: str) -> bool:
    """Detect column-interleaving artifacts in extracted text (RDR-022 Phase B).

    Post-extraction quality check that catches multi-column PDFs missed by
    geometry-only detection. Checks for three specific artifact patterns
    produced when PyMuPDF4LLM reads across column boundaries.

    Args:
        text: Extracted text to check

    Returns:
        True if column-interleaving artifacts are detected
    """
    if not text or len(text) < 200:
        return False

    orphaned = _count_orphaned_headers(text)
    bracketed = _count_bracket_fragments(text)
    page_nums = _count_page_number_insertions(text)

    # Any single strong signal is sufficient
    if orphaned >= 3:
        logger.debug(f"Column artifact: {orphaned} orphaned single-letter headers")
        return True
    if bracketed >= 5:
        logger.debug(f"Column artifact: {bracketed} bracket-fragmented sequences")
        return True
    if page_nums >= 2:
        logger.debug(f"Column artifact: {page_nums} mid-sentence page number insertions")
        return True

    # Combined weak signals
    total = orphaned + bracketed + page_nums
    if total >= 4:
        logger.debug(f"Column artifact: combined score {total} "
                     f"(orphaned={orphaned}, brackets={bracketed}, page_nums={page_nums})")
        return True

    return False


def _count_orphaned_headers(text: str) -> int:
    """Count orphaned single-letter section headers mid-paragraph.

    Detects patterns like "study the dynamic **A** Algorithm DC-Tree" where
    a bold single letter from an adjacent column's article header is injected
    between sentences.

    Pattern: word(s) + bold/caps single letter + word(s) on the same line,
    where the letter doesn't fit the sentence context.
    """
    # Match: lowercase word, then bold single uppercase letter, then lowercase word
    # e.g., "dynamic **A** Algorithm" or "equilibrium **B** Binary"
    pattern = r'[a-z]+\s+\*\*[A-Z]\*\*\s+[A-Z][a-z]'
    return len(re.findall(pattern, text))


def _count_bracket_fragments(text: str) -> int:
    """Count bracket-fragmented text sequences.

    Detects patterns like "[whose][property][is][that]" produced when
    individual words near column boundaries are captured as separate
    bracketed tokens.

    Pattern: 3+ consecutive [word] tokens.
    """
    # Match sequences of 3+ bracketed single words
    pattern = r'(?:\[[a-zA-Z]+\]){3,}'
    return len(re.findall(pattern, text))


def _count_page_number_insertions(text: str) -> int:
    """Count abrupt page-number insertions mid-sentence.

    Detects patterns where a page number and possibly a running header
    appear in the middle of a sentence, like:
    "study the dynamic 8 Adwords Pricing model's equilibrium"

    Pattern: lowercase word + space + bare number + space + Capitalized words
    + space + lowercase continuation, where the number is 1-4 digits.
    """
    # Match: lowercase word, bare number (1-4 digits), capitalized phrase, lowercase word
    pattern = r'[a-z]+\s+\d{1,4}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+[a-z]+'
    return len(re.findall(pattern, text))


def detect_multi_column(pdf_path: Path, sample_pages: int = 10) -> bool:
    """Detect if a PDF has multi-column layout using page geometry heuristics (RDR-022).

    Samples up to `sample_pages` evenly-spaced pages and analyzes text block
    bounding boxes to identify dual-column layouts. A page is classified as
    dual-column when text blocks cluster into two distinct horizontal groups
    with a gap between them.

    Validated at 94% accuracy on a 17-PDF test set (RDR-022 spike).

    Classification criteria per page:
    - Both left and right groups have >= 2 text blocks
    - Gap between columns is 0-15% of page width (median edges)
    - Each column width is 25-55% of page width
    - Column width ratio > 0.6 (columns are roughly equal)
    - Blocks wider than 60% of page are excluded (spanning headers)

    A PDF is classified as multi-column if >= 30% of sampled pages are dual-column.

    Args:
        pdf_path: Path to the PDF file
        sample_pages: Number of pages to sample (default: 6)

    Returns:
        True if PDF appears to have multi-column layout
    """
    import pymupdf

    # Check cache (keyed by resolved path + mtime to invalidate on file change)
    try:
        cache_key = (str(pdf_path.resolve()), pdf_path.stat().st_mtime)
        if cache_key in _column_detection_cache:
            return _column_detection_cache[cache_key]
    except OSError:
        cache_key = None

    try:
        with pymupdf.open(pdf_path) as doc:
            page_count = len(doc)
            if page_count == 0:
                if cache_key:
                    _column_detection_cache[cache_key] = False
                return False

            # Select evenly-spaced sample pages
            if page_count <= sample_pages:
                page_indices = list(range(page_count))
            else:
                step = page_count / sample_pages
                page_indices = [int(i * step) for i in range(sample_pages)]

            multi_column_pages = 0

            for page_idx in page_indices:
                page = doc[page_idx]
                page_width = page.rect.width

                if page_width <= 0:
                    continue

                # Get text blocks: (x0, y0, x1, y1, text, block_no, block_type)
                blocks = page.get_text("blocks")

                # Filter to text blocks (type 0) with meaningful content
                # Exclude wide spanning blocks (>60% of page width) like headers,
                # which straddle both columns and distort gap/width calculations
                text_blocks = [
                    b for b in blocks
                    if b[6] == 0
                    and len(b[4].strip()) > 20
                    and (b[2] - b[0]) / page_width <= 0.60
                ]

                if len(text_blocks) < 4:
                    continue

                # Find the midpoint of the page
                mid_x = page_width / 2

                # Classify blocks into left and right groups
                left_blocks = []
                right_blocks = []
                for b in text_blocks:
                    x0, _, x1, _, _, _, _ = b
                    center_x = (x0 + x1) / 2
                    if center_x < mid_x:
                        left_blocks.append(b)
                    else:
                        right_blocks.append(b)

                # Need at least 2 blocks in each group
                if len(left_blocks) < 2 or len(right_blocks) < 2:
                    continue

                # Use median right-edges and left-edges for robustness against outliers
                left_right_edges = sorted(b[2] for b in left_blocks)
                right_left_edges = sorted(b[0] for b in right_blocks)
                left_max_x = left_right_edges[len(left_right_edges) // 2]  # median
                right_min_x = right_left_edges[len(right_left_edges) // 2]  # median

                # Gap between columns: 0-15% of page width
                # Use >= 0 (not > 1%) since median edges handle outliers
                gap = right_min_x - left_max_x
                gap_ratio = gap / page_width
                if gap_ratio < 0.0 or gap_ratio > 0.15:
                    continue

                # Column widths: 25-55% of page width each
                left_min_x = min(b[0] for b in left_blocks)
                right_max_x = max(b[2] for b in right_blocks)
                left_width = (left_max_x - left_min_x) / page_width
                right_width = (right_max_x - right_min_x) / page_width

                if left_width < 0.25 or left_width > 0.55:
                    continue
                if right_width < 0.25 or right_width > 0.55:
                    continue

                # Width ratio > 0.6 (columns roughly equal)
                width_ratio = min(left_width, right_width) / max(left_width, right_width)
                if width_ratio < 0.6:
                    continue

                multi_column_pages += 1

            # Classify as multi-column if >= 30% of sampled pages qualify.
            # Lower threshold accounts for front matter, index, and single-column
            # pages that appear in otherwise dual-column documents.
            threshold = max(2, len(page_indices) * 0.3)
            result = multi_column_pages >= threshold
            if cache_key:
                _column_detection_cache[cache_key] = result
            return result

    except Exception as e:
        logger.debug(f"Column detection failed for {pdf_path}: {e}")
        if cache_key:
            _column_detection_cache[cache_key] = False
        return False

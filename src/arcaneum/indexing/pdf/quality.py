"""Text quality scoring for detecting garbled PDF extractions.

Used by --repair to identify indexed chunks with unreadable text
(replacement characters, encoding garbage, CID font mapping failures)
that need re-extraction with the updated pymupdf4llm auto-OCR.
"""

import re
import logging

logger = logging.getLogger(__name__)

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

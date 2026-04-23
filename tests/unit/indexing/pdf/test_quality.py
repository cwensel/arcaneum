"""Tests for PDF quality heuristics, including dropout detection."""

from arcaneum.indexing.pdf.quality import (
    looks_like_dropout,
    needs_ocr,
    score_text,
)


class TestLooksLikeDropout:
    def test_watermark_only_extraction_flagged(self):
        # IEEE Xplore watermark scenario: ~1 KB across 8 pages
        watermark = (
            "Authorized licensed use limited to: Chris Wensel. "
            "Downloaded from IEEE Xplore.  Restrictions apply. \n"
        ) * 10
        assert looks_like_dropout(watermark, page_count=8) is True

    def test_realistic_paper_not_flagged(self):
        body = "x" * 39000
        assert looks_like_dropout(body, page_count=8) is False

    def test_single_page_not_flagged_even_if_sparse(self):
        # A legitimate 1-page abstract should not be flagged — the min_pages
        # guard exists specifically to avoid false positives here.
        assert looks_like_dropout("x" * 100, page_count=1) is False

    def test_empty_multi_page_flagged(self):
        assert looks_like_dropout("", page_count=8) is True

    def test_zero_page_count_not_flagged(self):
        assert looks_like_dropout("anything", page_count=0) is False

    def test_threshold_boundary(self):
        # Right at the density threshold (500 chars/page)
        assert looks_like_dropout("x" * 1000, page_count=2) is False
        assert looks_like_dropout("x" * 999, page_count=2) is True


class TestExistingSignalsMissDropout:
    """Regression: the watermark case passes score_text/needs_ocr cleanly,
    which is exactly why we needed a separate dropout detector."""

    def test_watermark_scores_high(self):
        watermark = (
            "Authorized licensed use limited to: Chris Wensel. "
            "Downloaded from IEEE Xplore.  Restrictions apply. \n"
        ) * 10
        # Clean English → high quality score, needs_ocr returns False
        assert score_text(watermark) > 0.9
        assert needs_ocr(watermark) is False


class TestSyncSoftQualityGate:
    """Sync uses score_text < 0.7 as a soft gate to catch files that slip
    past needs_ocr() but score poorly. This documents the threshold boundary
    between the hard gate (needs_ocr, for U+FFFD/no-stopwords garbage) and
    repair's threshold (0.9)."""

    def test_clean_english_passes_soft_gate(self):
        # Clean prose scores high — no OCR retry in sync.
        text = (
            "This paper presents a new approach to software inspection. "
            "The method is based on established principles and has been "
            "evaluated in multiple case studies with significant results. "
        ) * 8
        assert needs_ocr(text) is False
        assert score_text(text) >= 0.9

    def test_hard_garbage_caught_by_needs_ocr(self):
        # Pure gibberish (>100 words, no stop words) — caught by the existing
        # hard gate, no soft-gate behavior change needed.
        text = "zqxw zqxw " * 60
        assert needs_ocr(text) is True

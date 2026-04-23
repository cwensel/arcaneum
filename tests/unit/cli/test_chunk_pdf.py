"""Tests for chunk_pdf_file sync-path logic: dropout fallback and soft quality gate.

These tests exercise the chunk_pdf_file function in arcaneum.cli.sync with
mocked PDFExtractor and PDFChunker to verify:
  (a) dropout fallback triggers PDFExtractor(markdown_conversion=False) when
      looks_like_dropout returns True
  (b) extraction_floor is set when the fallback doesn't improve
  (c) soft quality gate (score_text < 0.7) triggers OCR for marginal text
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arcaneum.cli.sync import chunk_pdf_file


def _make_chunk(text, metadata):
    from arcaneum.indexing.pdf.chunker import Chunk
    return Chunk(text=text, chunk_index=0, token_count=len(text) // 4, metadata=metadata)


# Text long enough to pass the len(text.strip()) < 100 check in chunk_pdf_file
# and that needs_ocr() returns False for (clean English).
_REALISTIC_TEXT = (
    "This paper presents a new approach to software inspection. "
    "The method is based on established principles and has been "
    "evaluated in multiple case studies with significant results. "
) * 5  # ~400 chars — well above the 100-char minimum


@pytest.fixture
def fake_pdf(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    return pdf


@pytest.fixture
def model_config():
    return {"chunk_size": 8000, "char_to_token_ratio": 4.0}


class TestDropoutFallback:
    """When looks_like_dropout returns True, chunk_pdf_file should re-extract
    with PDFExtractor(markdown_conversion=False) and use the better result."""

    @patch("arcaneum.indexing.pdf.chunker.PDFChunker")
    @patch("arcaneum.indexing.pdf.extractor.PDFExtractor")
    @patch("arcaneum.indexing.pdf.quality.looks_like_dropout")
    @patch("arcaneum.indexing.pdf.quality.score_text")
    @patch("arcaneum.indexing.pdf.quality.needs_ocr")
    def test_fallback_uses_normalized_extraction(
        self, mock_needs_ocr, mock_score, mock_dropout, mock_extractor_cls, mock_chunker_cls,
        fake_pdf, model_config,
    ):
        mock_needs_ocr.return_value = False
        mock_score.return_value = 0.9  # High quality — no OCR trigger
        mock_dropout.return_value = True

        # Initial extraction returns sparse text (watermark)
        initial_extractor = MagicMock()
        initial_extractor.extract.return_value = (
            "Authorized licensed use limited to: Chris Wensel.\n" * 5,
            {"page_count": 8, "page_boundaries": []},
        )

        # Fallback extraction returns much more text (> 2x initial)
        fallback_extractor = MagicMock()
        fallback_extractor.extract.return_value = (
            _REALISTIC_TEXT * 5,  # ~2000 chars, much more than initial ~200
            {"page_count": 8, "page_boundaries": [], "fallback": True},
        )

        mock_extractor_cls.side_effect = [initial_extractor, fallback_extractor]

        chunker = MagicMock()
        chunker.chunk.return_value = [
            _make_chunk("recovered text", {"file_path": str(fake_pdf), "page_count": 8})
        ]
        mock_chunker_cls.return_value = chunker

        result = chunk_pdf_file(fake_pdf, model_config, use_ocr=False)

        # Fallback extractor was created with markdown_conversion=False
        mock_extractor_cls.assert_any_call(markdown_conversion=False)
        assert len(result) == 1

    @patch("arcaneum.indexing.pdf.chunker.PDFChunker")
    @patch("arcaneum.indexing.pdf.extractor.PDFExtractor")
    @patch("arcaneum.indexing.pdf.quality.looks_like_dropout")
    @patch("arcaneum.indexing.pdf.quality.score_text")
    @patch("arcaneum.indexing.pdf.quality.needs_ocr")
    def test_extraction_floor_set_when_bypass_fails(
        self, mock_needs_ocr, mock_score, mock_dropout, mock_extractor_cls, mock_chunker_cls,
        fake_pdf, model_config,
    ):
        mock_needs_ocr.return_value = False
        mock_score.return_value = 0.9  # High quality — no OCR trigger
        mock_dropout.return_value = True

        # Both extractions return sparse text — bypass doesn't improve enough
        initial_extractor = MagicMock()
        initial_extractor.extract.return_value = (
            "x" * 200,
            {"page_count": 8, "page_boundaries": []},
        )

        fallback_extractor = MagicMock()
        fallback_extractor.extract.return_value = (
            "x" * 300,  # Not 2x better than initial 200 chars
            {"page_count": 8, "page_boundaries": []},
        )

        mock_extractor_cls.side_effect = [initial_extractor, fallback_extractor]

        # Capture the metadata passed to chunker
        captured_meta = {}
        chunker = MagicMock()

        def fake_chunk(text, metadata):
            captured_meta.update(metadata)
            return [_make_chunk(text, metadata)]

        chunker.chunk.side_effect = fake_chunk
        mock_chunker_cls.return_value = chunker

        chunk_pdf_file(fake_pdf, model_config, use_ocr=False)

        # extraction_floor should be set because fallback didn't improve enough
        assert captured_meta.get("extraction_floor") is True


class TestSoftQualityGate:
    """score_text < 0.7 triggers OCR re-extraction for text that passes
    needs_ocr() but scores poorly (mis-mapped fonts yielding some English)."""

    @patch("arcaneum.indexing.pdf.chunker.PDFChunker")
    @patch("arcaneum.indexing.pdf.extractor.PDFExtractor")
    @patch("arcaneum.indexing.pdf.quality.looks_like_dropout")
    @patch("arcaneum.indexing.pdf.quality.score_text")
    @patch("arcaneum.indexing.pdf.quality.needs_ocr")
    def test_low_quality_triggers_ocr(
        self, mock_needs_ocr, mock_score, mock_dropout,
        mock_extractor_cls, mock_chunker_cls,
        fake_pdf, model_config,
    ):
        mock_needs_ocr.return_value = False
        mock_score.return_value = 0.55  # Below the 0.7 soft gate
        mock_dropout.return_value = False

        # Initial extraction
        initial_extractor = MagicMock()
        initial_extractor.extract.return_value = (
            _REALISTIC_TEXT,  # Long enough to pass the < 100 check
            {"page_count": 1, "page_boundaries": []},
        )

        # OCR re-extraction
        ocr_extractor = MagicMock()
        ocr_extractor.extract.return_value = (
            "clean OCR text with full content " * 20,
            {"page_count": 1, "page_boundaries": [], "ocr": True},
        )

        mock_extractor_cls.side_effect = [initial_extractor, ocr_extractor]

        chunker = MagicMock()
        chunker.chunk.return_value = [_make_chunk("ocr text", {"file_path": str(fake_pdf)})]
        mock_chunker_cls.return_value = chunker

        result = chunk_pdf_file(fake_pdf, model_config, use_ocr=False)

        # OCR extractor was created with use_ocr=True
        mock_extractor_cls.assert_any_call(use_ocr=True)
        assert len(result) == 1

    @patch("arcaneum.indexing.pdf.chunker.PDFChunker")
    @patch("arcaneum.indexing.pdf.extractor.PDFExtractor")
    @patch("arcaneum.indexing.pdf.quality.looks_like_dropout")
    @patch("arcaneum.indexing.pdf.quality.score_text")
    @patch("arcaneum.indexing.pdf.quality.needs_ocr")
    def test_high_quality_skips_ocr(
        self, mock_needs_ocr, mock_score, mock_dropout,
        mock_extractor_cls, mock_chunker_cls,
        fake_pdf, model_config,
    ):
        mock_needs_ocr.return_value = False
        mock_score.return_value = 0.85  # Above the 0.7 soft gate
        mock_dropout.return_value = False

        initial_extractor = MagicMock()
        initial_extractor.extract.return_value = (
            _REALISTIC_TEXT,  # Long enough to pass the < 100 check
            {"page_count": 1, "page_boundaries": []},
        )
        mock_extractor_cls.return_value = initial_extractor

        chunker = MagicMock()
        chunker.chunk.return_value = [_make_chunk("good text", {"file_path": str(fake_pdf)})]
        mock_chunker_cls.return_value = chunker

        result = chunk_pdf_file(fake_pdf, model_config, use_ocr=False)

        # Only one extractor created (initial), no OCR re-extraction
        assert mock_extractor_cls.call_count == 1
        assert len(result) == 1
"""Tests for chunk_pdf_file sync-path logic: dropout fallback and soft quality gate.

These tests exercise the chunk_pdf_file function in arcaneum.cli.sync with
mocked PDFExtractor and PDFChunker to verify:
  (a) dropout fallback triggers PDFExtractor(markdown_conversion=False) when
      looks_like_dropout returns True
  (b) extraction_floor is set when the fallback doesn't improve
  (c) soft quality gate (score_text < 0.7) triggers OCR for marginal text
"""

from unittest.mock import MagicMock, patch

import pytest

from arcaneum.cli.sync import _build_quality_manifest, chunk_pdf_file
from arcaneum.indexing.pdf.quality import is_replacement_heavy
from arcaneum.indexing.verify import _manifest_fidelity_state


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


def test_replacement_heavy_uses_existing_strict_five_percent_threshold():
    assert is_replacement_heavy("a" * 95 + "\ufffd" * 5) is False
    assert is_replacement_heavy("a" * 94 + "\ufffd" * 6) is True


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
        self,
        mock_needs_ocr,
        mock_score,
        mock_dropout,
        mock_extractor_cls,
        mock_chunker_cls,
        fake_pdf,
        model_config,
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
            {
                "page_count": 8,
                "page_boundaries": [],
                "extraction_method": "pymupdf_normalized",
                "fallback": True,
            },
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
        manifest = result[0]["metadata"]["quality_manifest"]
        assert manifest["source_hash"]
        assert manifest["chunk_count"] == 1
        assert manifest["fallback_method"] == "pymupdf_normalized"
        assert "dropout_recovered" in manifest["quality_warnings"]
        assert len(result) == 1

    @patch("arcaneum.indexing.pdf.chunker.PDFChunker")
    @patch("arcaneum.indexing.pdf.extractor.PDFExtractor")
    @patch("arcaneum.indexing.pdf.quality.looks_like_dropout")
    @patch("arcaneum.indexing.pdf.quality.score_text")
    @patch("arcaneum.indexing.pdf.quality.needs_ocr")
    def test_extraction_floor_set_when_bypass_fails(
        self,
        mock_needs_ocr,
        mock_score,
        mock_dropout,
        mock_extractor_cls,
        mock_chunker_cls,
        fake_pdf,
        model_config,
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

        result = chunk_pdf_file(fake_pdf, model_config, use_ocr=False)

        # extraction_floor should be set because fallback didn't improve enough
        assert captured_meta.get("extraction_floor") is True
        manifest = result.quality_manifest
        assert manifest["fallback_method"] == "normalized_bypass_no_improvement"
        assert "extraction_floor" in manifest["quality_warnings"]


class TestSoftQualityGate:
    """score_text < 0.7 triggers OCR re-extraction for text that passes
    needs_ocr() but scores poorly (mis-mapped fonts yielding some English)."""

    @patch("arcaneum.indexing.pdf.chunker.PDFChunker")
    @patch("arcaneum.indexing.pdf.extractor.PDFExtractor")
    @patch("arcaneum.indexing.pdf.quality.looks_like_dropout")
    @patch("arcaneum.indexing.pdf.quality.score_text")
    @patch("arcaneum.indexing.pdf.quality.needs_ocr")
    def test_low_quality_triggers_ocr(
        self,
        mock_needs_ocr,
        mock_score,
        mock_dropout,
        mock_extractor_cls,
        mock_chunker_cls,
        fake_pdf,
        model_config,
    ):
        mock_needs_ocr.return_value = False
        mock_score.return_value = 0.55  # Below the 0.7 soft gate
        mock_dropout.return_value = False

        # Initial extraction
        initial_extractor = MagicMock()
        original_boundaries = [
            {"page_number": 1, "start_char": 0, "page_text_length": len(_REALISTIC_TEXT)}
        ]
        initial_extractor.extract.return_value = (
            _REALISTIC_TEXT,  # Long enough to pass the < 100 check
            {
                "extraction_method": "pymupdf4llm_markdown",
                "page_count": 1,
                "page_boundaries": original_boundaries,
            },
        )

        # OCR re-extraction
        ocr_extractor = MagicMock()
        ocr_text = "clean OCR text with full content " * 20
        ocr_extractor.extract.return_value = (
            ocr_text,
            {
                "extraction_method": "pymupdf4llm_ocr",
                "page_count": 1,
                "page_boundaries": [
                    {"page_number": 1, "start_char": 0, "page_text_length": len(ocr_text)}
                ],
                "ocr_confidence": 72.0,
                "ocr_pages_processed": 1,
                "ocr_pages_failed": 0,
            },
        )

        mock_extractor_cls.side_effect = [initial_extractor, ocr_extractor]

        captured = {}
        chunker = MagicMock()

        def fake_chunk(text, metadata):
            captured["text"] = text
            captured["metadata"] = dict(metadata)
            return [_make_chunk(text, metadata)]

        chunker.chunk.side_effect = fake_chunk
        mock_chunker_cls.return_value = chunker

        result = chunk_pdf_file(fake_pdf, model_config, use_ocr=False)

        # OCR extractor was created with use_ocr=True
        mock_extractor_cls.assert_any_call(use_ocr=True)
        assert captured["text"] == _REALISTIC_TEXT
        assert "clean OCR text with full content" not in captured["text"]
        assert captured["metadata"]["ocr_merge_strategy"] == "page_quality_selection"
        assert captured["metadata"]["ocr_triggered_by"] == "quality"
        manifest = result.quality_manifest
        assert manifest["ocr"]["triggered"] is True
        assert manifest["ocr"]["reason"] == "quality"
        assert manifest["ocr"]["confidence"] == 72.0
        assert manifest["extraction_candidates"][0]["chosen_source"] == "embedded"
        assert captured["metadata"]["ocr_confidence"] == 72.0
        assert captured["metadata"]["page_boundaries"][0] == original_boundaries[0]
        assert len(result) == 1

    @patch("arcaneum.indexing.pdf.chunker.PDFChunker")
    @patch("arcaneum.indexing.pdf.extractor.PDFExtractor")
    def test_empty_text_ocr_retry_records_empty_trigger(
        self,
        mock_extractor_cls,
        mock_chunker_cls,
        fake_pdf,
        model_config,
    ):
        initial_extractor = MagicMock()
        initial_extractor.extract.return_value = (
            "",
            {"extraction_method": "pymupdf4llm_markdown", "page_count": 1, "page_boundaries": []},
        )

        ocr_text = "recovered OCR text " * 20
        ocr_extractor = MagicMock()
        ocr_extractor.extract.return_value = (
            ocr_text,
            {
                "extraction_method": "pymupdf4llm_ocr",
                "page_count": 1,
                "page_boundaries": [
                    {"page_number": 1, "start_char": 0, "page_text_length": len(ocr_text)}
                ],
            },
        )
        mock_extractor_cls.side_effect = [initial_extractor, ocr_extractor]

        captured = {}
        chunker = MagicMock()

        def fake_chunk(text, metadata):
            captured["metadata"] = dict(metadata)
            return [_make_chunk(text, metadata)]

        chunker.chunk.side_effect = fake_chunk
        mock_chunker_cls.return_value = chunker

        result = chunk_pdf_file(fake_pdf, model_config, use_ocr=False)

        assert captured["metadata"]["ocr_triggered_by"] == "empty"
        assert len(result) == 1

    @patch("arcaneum.indexing.pdf.chunker.PDFChunker")
    @patch("arcaneum.indexing.pdf.extractor.PDFExtractor")
    @patch("arcaneum.indexing.pdf.quality.looks_like_dropout")
    @patch("arcaneum.indexing.pdf.quality.score_text")
    @patch("arcaneum.indexing.pdf.quality.needs_ocr")
    def test_high_quality_skips_ocr(
        self,
        mock_needs_ocr,
        mock_score,
        mock_dropout,
        mock_extractor_cls,
        mock_chunker_cls,
        fake_pdf,
        model_config,
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


def test_code_quality_manifest_distinguishes_ast_fallback(tmp_path):
    source = tmp_path / "fallback.py"
    source.write_text("print('fallback')\n")

    manifest = _build_quality_manifest(
        file_path=source,
        corpus_type="code",
        source_hash="abc123",
        chunk_count=1,
        metadata={"method": "line_based"},
        extraction_method="line_based",
    )

    assert manifest["extractor"] == "code"
    assert manifest["extraction_method"] == "line_based"
    assert manifest["fallback_method"] == "line_based"
    assert manifest["source_hash"] == "abc123"


def test_quality_manifest_marks_forced_ocr(tmp_path):
    source = tmp_path / "forced.pdf"
    source.write_bytes(b"%PDF-1.4\n")

    manifest = _build_quality_manifest(
        file_path=source,
        corpus_type="pdf",
        source_hash="abc123",
        chunk_count=1,
        metadata={"extraction_method": "pymupdf4llm_ocr"},
    )

    assert manifest["ocr"]["triggered"] is True
    assert manifest["ocr"]["reason"] == "forced"
    assert manifest["dropped_chunk_count"] == 0
    assert manifest["dropped_chunk_reason"] is None


def test_quality_manifest_marks_degraded_ocr_as_recovery_exhausted(tmp_path):
    source = tmp_path / "degraded.pdf"
    source.write_bytes(b"%PDF-1.4\n")

    manifest = _build_quality_manifest(
        file_path=source,
        corpus_type="pdf",
        source_hash="abc123",
        chunk_count=1,
        metadata={
            "ocr_triggered_by": "garbled",
            "replacement_omissions": {
                "source_character_count": 100,
                "retained_character_count": 10,
                "character_count": 90,
                "replacement_ratio": 0.9,
                "degraded": True,
            },
        },
    )

    assert "replacement_fidelity_degraded" in manifest["quality_warnings"]
    assert "replacement_recovery_exhausted" in manifest["quality_warnings"]


@patch("arcaneum.indexing.pdf.extractor.PDFExtractor")
@patch("arcaneum.indexing.pdf.quality.looks_like_dropout", return_value=False)
@patch("arcaneum.indexing.pdf.quality.needs_ocr", return_value=True)
def test_failed_ocr_attempt_converges_without_repair_loop(
    _needs_ocr,
    _dropout,
    extractor_cls,
    fake_pdf,
    model_config,
):
    text = ("readable source text " * 20) + ("\ufffd" * 40)
    initial_extractor = MagicMock()
    initial_extractor.extract.return_value = (
        text,
        {"page_count": 1, "page_boundaries": []},
    )
    ocr_extractor = MagicMock()
    ocr_extractor.extract.side_effect = RuntimeError("unbounded detail " * 1000)
    extractor_cls.side_effect = [initial_extractor, ocr_extractor]

    result = chunk_pdf_file(fake_pdf, model_config)

    assert len(result) == 1
    manifest = result.quality_manifest
    assert manifest["ocr"]["triggered"] is True
    assert manifest["ocr"]["reason"] == "garbled"
    assert manifest["ocr"]["attempt_failed"] is True
    assert "ocr_attempt_failed" in manifest["quality_warnings"]
    assert "replacement_recovery_exhausted" in manifest["quality_warnings"]
    assert "unbounded detail" not in str(manifest)
    assert _manifest_fidelity_state(manifest, len(result)) == (True, True, True, False)


@patch("arcaneum.indexing.pdf.extractor.PDFExtractor")
@patch("arcaneum.indexing.pdf.quality.looks_like_dropout", return_value=False)
@patch("arcaneum.indexing.pdf.quality.score_text", return_value=0.9)
@patch("arcaneum.indexing.pdf.quality.needs_ocr", return_value=False)
def test_replacement_runs_split_chunks_and_preserve_clean_text(
    _needs_ocr,
    _score_text,
    _dropout,
    extractor_cls,
    fake_pdf,
    model_config,
):
    first = _REALISTIC_TEXT
    second = _REALISTIC_TEXT.replace("paper", "study")
    text = first + "\ufffd" * 20 + second
    extractor_cls.return_value.extract.return_value = (
        text,
        {"page_count": 1, "page_boundaries": []},
    )

    result = chunk_pdf_file(fake_pdf, model_config)

    assert [chunk["text"] for chunk in result] == [first.strip(), second.strip()]
    assert [chunk["metadata"]["chunk_index"] for chunk in result] == [0, 1]
    assert {chunk["metadata"]["chunk_count"] for chunk in result} == {2}
    manifest = result.quality_manifest
    assert manifest["chunk_count"] == 2
    assert manifest["replacement_omissions"]["source_character_count"] == len(text)
    assert manifest["replacement_omissions"]["retained_character_count"] == len(text) - 20
    assert manifest["replacement_omissions"]["character_count"] == 20
    assert manifest["replacement_omissions"]["replacement_ratio"] == 20 / len(text)
    assert manifest["replacement_omissions"]["degraded"] is False
    assert manifest["replacement_omissions"]["hard_boundary_count"] == 1
    assert "replacement_characters_omitted" in manifest["quality_warnings"]


@patch("arcaneum.indexing.pdf.extractor.PDFExtractor")
@patch("arcaneum.indexing.pdf.quality.looks_like_dropout", return_value=False)
@patch("arcaneum.indexing.pdf.quality.score_text", return_value=0.9)
@patch("arcaneum.indexing.pdf.quality.needs_ocr", return_value=False)
def test_all_replacement_text_keeps_an_auditable_manifest(
    _needs_ocr,
    _score_text,
    _dropout,
    extractor_cls,
    fake_pdf,
    model_config,
):
    extractor_cls.return_value.extract.return_value = (
        "\ufffd" * 200,
        {"page_count": 1, "page_boundaries": []},
    )

    result = chunk_pdf_file(fake_pdf, model_config)

    assert result == []
    assert result.quality_manifest["chunk_count"] == 0
    assert result.quality_manifest["replacement_omissions"]["character_count"] == 200
    assert result.quality_manifest["replacement_omissions"]["source_character_count"] == 200
    assert result.quality_manifest["replacement_omissions"]["retained_character_count"] == 0
    assert result.quality_manifest["replacement_omissions"]["replacement_ratio"] == 1.0
    assert result.quality_manifest["replacement_omissions"]["degraded"] is True
    assert result.quality_manifest["replacement_omissions"]["hard_boundary_count"] == 0
    assert result.quality_manifest["replacement_omissions"]["normalized_run_count"] == 1

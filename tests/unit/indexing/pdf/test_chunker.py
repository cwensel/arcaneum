"""Regression tests for PDF chunk boundary selection."""

import pytest

from arcaneum.indexing.pdf.chunker import PDFChunker


def _chunker(*, chunk_size: int, overlap_percent: float = 0.15) -> PDFChunker:
    return PDFChunker(
        {"chunk_size": chunk_size, "char_to_token_ratio": 1},
        overlap_percent=overlap_percent,
        late_chunking_enabled=False,
    )


def test_chunk_edges_preserve_whole_words_with_overlap():
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"

    chunks = _chunker(chunk_size=20, overlap_percent=0.25).chunk(text, {})

    assert len(chunks) > 2
    for chunk in chunks:
        start = chunk.metadata["chunk_start_char"]
        end = chunk.metadata["chunk_end_char"]
        assert chunk.text == text[start:end]
        assert start == 0 or text[start - 1].isspace()
        assert end == len(text) or text[end].isspace()

    assert all(
        current.metadata["chunk_start_char"] < previous.metadata["chunk_end_char"]
        for previous, current in zip(chunks, chunks[1:])
    )


def test_end_boundary_prefers_paragraph_over_later_sentence():
    text = "alpha bravo charlie xx\n\nY. Echo foxtrot golf hotel india"
    paragraph_end = text.index("\n\n")

    chunks = _chunker(chunk_size=28, overlap_percent=0).chunk(text, {})

    assert chunks[0].text == text[:paragraph_end]
    assert chunks[0].metadata["chunk_end_char"] == paragraph_end


def test_end_boundary_falls_back_to_sentence_before_whitespace():
    text = "alpha bravo charlie. Delta echo foxtrot golf hotel"
    sentence_end = text.index(".") + 1

    chunks = _chunker(chunk_size=23, overlap_percent=0).chunk(text, {})

    assert chunks[0].text == text[:sentence_end]
    assert chunks[0].metadata["chunk_end_char"] == sentence_end


def test_unbroken_token_hard_cuts_and_makes_progress():
    text = "x" * 55

    chunks = _chunker(chunk_size=20, overlap_percent=0.25).chunk(text, {})

    assert [len(chunk.text) for chunk in chunks] == [20, 20, 20, 10]
    assert [chunk.metadata["chunk_start_char"] for chunk in chunks] == [0, 15, 30, 45]
    assert chunks[-1].metadata["chunk_end_char"] == len(text)


@pytest.mark.parametrize("overlap_percent", [-0.1, 1])
def test_out_of_range_overlap_is_rejected(overlap_percent):
    with pytest.raises(ValueError, match="overlap_percent"):
        _chunker(chunk_size=20, overlap_percent=overlap_percent)


def test_trimmed_span_drives_page_provenance_at_page_boundary():
    page_one = "alpha bravo charlie"
    separator = "\n\n"
    page_two = "delta echo foxtrot golf"
    text = page_one + separator + page_two
    page_two_start = len(page_one) + len(separator)
    metadata = {
        "page_boundaries": [
            {"page_number": 1, "start_char": 0, "page_text_length": len(page_one)},
            {
                "page_number": 2,
                "start_char": page_two_start,
                "page_text_length": len(page_two),
            },
        ]
    }

    chunks = _chunker(chunk_size=23, overlap_percent=0).chunk(text, metadata)

    assert chunks[1].text.startswith("delta")
    assert chunks[1].metadata["chunk_start_char"] == page_two_start
    assert chunks[1].metadata["page_number"] == 2
    assert (
        chunks[1].text
        == text[chunks[1].metadata["chunk_start_char"] : chunks[1].metadata["chunk_end_char"]]
    )

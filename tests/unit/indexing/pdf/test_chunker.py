"""Regression tests for PDF chunk boundary selection."""

import json

import pytest

from arcaneum.indexing.pdf.chunker import PDFChunker


def _chunker(
    *, chunk_size: int, overlap_percent: float = 0.15, min_chunk_chars: int = 0
) -> PDFChunker:
    return PDFChunker(
        {
            "chunk_size": chunk_size,
            "char_to_token_ratio": 1,
            "min_chunk_chars": min_chunk_chars,
        },
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


def test_negative_fragment_floor_is_rejected():
    with pytest.raises(ValueError, match="min_chunk_chars"):
        _chunker(chunk_size=20, min_chunk_chars=-1)


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


def test_reference_fragments_merge_and_recompute_final_chunk_metadata():
    text = "References\n\n" + " ".join(
        f"{chr(65 + index % 26)}. B. Author. Distributed Systems Study {index}. "
        f"Conf., 20{index:02d}."
        for index in range(24)
    )
    page_two_start = len(text) // 2
    metadata = {
        "page_boundaries": [
            {
                "page_number": 1,
                "start_char": 0,
                "page_text_length": page_two_start,
            },
            {
                "page_number": 2,
                "start_char": page_two_start,
                "page_text_length": len(text) - page_two_start,
            },
        ]
    }

    chunks = _chunker(chunk_size=220, overlap_percent=0, min_chunk_chars=200).chunk(text, metadata)

    assert len(chunks) > 1
    assert all(len(chunk.text) >= 200 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        start = chunk.metadata["chunk_start_char"]
        end = chunk.metadata["chunk_end_char"]
        assert chunk.text == text[start:end]
        assert chunk.token_count == len(chunk.text)
        assert chunk.metadata["chunk_index"] == chunk.chunk_index
        assert chunk.metadata["chunk_count"] == len(chunks)
        assert chunk.metadata["page_number"] == (1 if start < page_two_start else 2)


def test_document_shorter_than_fragment_floor_is_the_only_exception():
    text = " ".join(
        [
            "A. B. Author. Distributed Systems Study.",
            "C. D. Author. Reliable Storage Study.",
            "E. F. Author. Network Protocol Study.",
        ]
    )

    chunks = _chunker(chunk_size=50, overlap_percent=0, min_chunk_chars=200).chunk(text, {})

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert len(chunks[0].text) < 200


def test_single_replacements_become_spaces_without_losing_source_offsets():
    text = "alpha\ufffdbeta gamma\ufffddelta"

    chunks = _chunker(chunk_size=100, overlap_percent=0).chunk(text, {})

    assert [chunk.text for chunk in chunks] == ["alpha beta gamma delta"]
    assert chunks[0].metadata["chunk_start_char"] == 0
    assert chunks[0].metadata["chunk_end_char"] == len(text)
    assert chunks[0].metadata["chunk_text_normalized"] is True
    assert chunks[0].metadata["normalized_replacement_character_count"] == 2
    assert (
        chunks[0].metadata["chunk_source_offset_semantics"] == "original_extracted_text_half_open"
    )
    assert chunks.replacement_omissions == {
        "source_character_count": len(text),
        "retained_character_count": len(text) - 2,
        "character_count": 2,
        "replacement_ratio": 2 / len(text),
        "degraded": True,
        "singleton_count": 2,
        "multi_character_run_count": 0,
        "multi_character_run_character_count": 0,
        "hard_boundary_count": 0,
        "hard_boundary_character_count": 0,
        "normalized_run_count": 0,
        "normalized_run_character_count": 0,
    }


def test_replacement_runs_are_omitted_hard_boundaries():
    left = ("alpha bravo " * 6).strip()
    right = ("delta echo " * 6).strip()
    text = left + "\ufffd\ufffd\ufffd" + right
    boundary_start = text.index("\ufffd")

    chunks = _chunker(chunk_size=200, overlap_percent=0, min_chunk_chars=50).chunk(text, {})

    assert [chunk.text for chunk in chunks] == [left, right]
    assert chunks[0].metadata["chunk_end_char"] == boundary_start
    assert chunks[1].metadata["chunk_start_char"] == boundary_start + 3
    assert chunks.replacement_omissions == {
        "source_character_count": len(text),
        "retained_character_count": len(text) - 3,
        "character_count": 3,
        "replacement_ratio": 3 / len(text),
        "degraded": False,
        "singleton_count": 0,
        "multi_character_run_count": 1,
        "multi_character_run_character_count": 3,
        "hard_boundary_count": 1,
        "hard_boundary_character_count": 3,
        "normalized_run_count": 0,
        "normalized_run_character_count": 0,
    }


def test_overlap_never_crosses_a_replacement_boundary():
    left = " ".join(f"left{index}" for index in range(20))
    right = " ".join(f"right{index}" for index in range(20))
    text = left + "\ufffd\ufffd" + right
    boundary_start = len(left)
    boundary_end = boundary_start + 2

    chunks = _chunker(chunk_size=35, overlap_percent=0.25).chunk(text, {})

    assert len(chunks) > 4
    assert all(
        chunk.metadata["chunk_end_char"] <= boundary_start
        or chunk.metadata["chunk_start_char"] >= boundary_end
        for chunk in chunks
    )
    assert all("\ufffd" not in chunk.text for chunk in chunks)


def test_many_single_replacements_do_not_create_many_regions():
    text = "\ufffd".join(f"word{index}" for index in range(50))

    chunks = _chunker(chunk_size=len(text), overlap_percent=0).chunk(text, {})

    assert len(chunks) == 1
    assert "\ufffd" not in chunks[0].text
    assert chunks.replacement_omissions["singleton_count"] == 49
    assert chunks.replacement_omissions["hard_boundary_count"] == 0


def test_only_replacement_runs_produce_no_indexable_chunks_with_audit_details():
    text = "\ufffd" * 12

    chunks = _chunker(chunk_size=100, overlap_percent=0).chunk(text, {})

    assert chunks == []
    assert chunks.replacement_omissions["character_count"] == 12
    assert chunks.replacement_omissions["hard_boundary_character_count"] == 0
    assert chunks.replacement_omissions["normalized_run_character_count"] == 12


def test_zero_indexable_text_is_degraded_even_below_replacement_ratio_threshold():
    text = (" " * 100) + "\ufffd" + (" " * 100)

    chunks = _chunker(chunk_size=1000, overlap_percent=0).chunk(text, {})

    assert chunks == []
    assert chunks.replacement_omissions["replacement_ratio"] < 0.05
    assert chunks.replacement_omissions["degraded"] is True


def test_repeated_short_regions_do_not_amplify_chunks_or_manifest_size():
    text = ("a\ufffd\ufffd") * 10_000

    chunks = _chunker(chunk_size=1000, overlap_percent=0).chunk(text, {})

    assert 20 <= len(chunks) <= 40
    assert all("\ufffd" not in chunk.text for chunk in chunks)
    assert chunks.replacement_omissions["multi_character_run_count"] == 10_000
    assert chunks.replacement_omissions["source_character_count"] == len(text)
    assert chunks.replacement_omissions["retained_character_count"] == 10_000
    assert chunks.replacement_omissions["replacement_ratio"] == 2 / 3
    assert chunks.replacement_omissions["degraded"] is True
    assert chunks.replacement_omissions["hard_boundary_count"] == 0
    assert chunks.replacement_omissions["normalized_run_count"] == 10_000
    assert "boundaries" not in chunks.replacement_omissions
    assert len(json.dumps(chunks.replacement_omissions)) < 512


def test_whitespace_heavy_regions_do_not_become_hard_boundaries():
    sparse_region = "a" + (" " * 100) + "b"
    text = "\ufffd\ufffd".join(sparse_region for _ in range(1000))

    chunks = _chunker(chunk_size=1000, overlap_percent=0).chunk(text, {})

    assert 50 <= len(chunks) <= 150
    assert chunks.replacement_omissions["multi_character_run_count"] == 999
    assert chunks.replacement_omissions["hard_boundary_count"] == 0
    assert chunks.replacement_omissions["normalized_run_count"] == 999


def test_page_attribution_uses_original_offset_after_hard_boundary():
    left = ("left page text " * 10).strip()
    right = ("right page text " * 10).strip()
    right_start = len(left) + 3
    metadata = {
        "page_boundaries": [
            {"page_number": 1, "start_char": 0, "page_text_length": len(left)},
            {
                "page_number": 2,
                "start_char": right_start,
                "page_text_length": len(right),
            },
        ]
    }

    chunks = _chunker(chunk_size=1000, overlap_percent=0).chunk(
        left + "\ufffd\ufffd\ufffd" + right, metadata
    )

    assert [chunk.metadata["page_number"] for chunk in chunks] == [1, 2]
    assert chunks[1].metadata["chunk_start_char"] == right_start


def test_late_chunking_path_preserves_replacement_boundaries():
    left = ("left context " * 15).strip()
    right = ("right context " * 15).strip()
    chunker = PDFChunker(
        {
            "chunk_size": 1000,
            "char_to_token_ratio": 1,
            "min_chunk_chars": 0,
            "late_chunking": True,
        },
        overlap_percent=0,
        late_chunking_enabled=True,
        min_doc_tokens=10,
        max_doc_tokens=1000,
    )

    chunks = chunker.chunk(left + "\ufffd\ufffd" + right, {})

    assert [chunk.text for chunk in chunks] == [left, right]
    assert all(chunk.metadata["late_chunking"] is True for chunk in chunks)
    assert chunks.replacement_omissions["hard_boundary_count"] == 1

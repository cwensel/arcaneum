"""Selecting indexed files by stored metadata for `arc corpus sync --from-index`."""

from types import SimpleNamespace

import pytest

from arcaneum.indexing.index_selector import (
    IndexSelectorError,
    parse_index_selector,
    select_indexed_paths,
)


class _FakeQdrant:
    """Serve scroll pages the way qdrant-client does: (points, next_offset)."""

    def __init__(self, payloads, page_size=2):
        self.payloads = payloads
        self.page_size = page_size
        self.calls = []

    def scroll(self, collection_name, limit, offset=None, with_payload=True, with_vectors=False):
        self.calls.append({"collection_name": collection_name, "with_payload": with_payload})
        start = offset or 0
        page = self.payloads[start : start + self.page_size]
        next_offset = start + self.page_size
        points = [SimpleNamespace(payload=p) for p in page]
        return points, (next_offset if next_offset < len(self.payloads) else None)


def _chunk(path, method="pymupdf4llm_markdown", app_version="0.10.0", **manifest):
    return {
        "file_path": path,
        "app_version": app_version,
        "quality_manifest": {"extraction_method": method, **manifest},
    }


def test_method_shorthand_matches_a_component_of_combined_methods():
    qdrant = _FakeQdrant(
        [
            _chunk("/p/plain.pdf", "pymupdf_normalized"),
            _chunk("/p/plain.pdf", "pymupdf_normalized"),
            _chunk("/p/merged.pdf", "pymupdf4llm_markdown+pymupdf_normalized"),
            _chunk("/p/selected.pdf", "page_selected:pymupdf4llm_markdown+pymupdf4llm_ocr"),
            _chunk("/p/clean.pdf", "pymupdf4llm_markdown"),
        ]
    )

    paths = select_indexed_paths(qdrant, "Papers", "method=pymupdf_normalized")

    assert paths == ["/p/merged.pdf", "/p/plain.pdf"]


def test_selection_maps_alias_manifests_to_their_canonical_path():
    qdrant = _FakeQdrant(
        [
            _chunk("/p/canonical.pdf", "pymupdf_normalized"),
            {
                "metadata_type": "file_manifest",
                "file_path": "/p/alias.pdf",
                "canonical_path": "/p/canonical.pdf",
                "quality_manifest": {"extraction_method": "pymupdf_normalized"},
            },
        ]
    )

    assert select_indexed_paths(qdrant, "Papers", "method=pymupdf_normalized") == [
        "/p/canonical.pdf"
    ]


def test_warning_shorthand_matches_list_membership():
    qdrant = _FakeQdrant(
        [
            _chunk("/p/a.pdf", quality_warnings=["empty_pages", "low_text_pages"]),
            _chunk("/p/b.pdf", quality_warnings=["replacement_characters_omitted"]),
        ]
    )

    assert select_indexed_paths(qdrant, "Papers", "warning=low_text_pages") == ["/p/a.pdf"]


def test_ocr_and_fallback_shorthands_read_nested_manifest_fields():
    qdrant = _FakeQdrant(
        [
            _chunk("/p/garbled.pdf", ocr={"reason": "garbled"}, fallback_reason=None),
            _chunk(
                "/p/timeout.pdf",
                "pymupdf_normalized",
                ocr={"reason": None},
                fallback_reason="LayoutWorkerTimeout",
            ),
        ]
    )

    assert select_indexed_paths(qdrant, "Papers", "ocr=garbled") == ["/p/garbled.pdf"]
    assert select_indexed_paths(qdrant, "Papers", "fallback=layoutworkertimeout") == [
        "/p/timeout.pdf"
    ]


def test_version_shorthand_compares_numerically():
    qdrant = _FakeQdrant(
        [
            _chunk("/p/old.md", app_version="0.8.3"),
            _chunk("/p/mid.md", app_version="0.9.0"),
            _chunk("/p/new.md", app_version="0.10.0"),
        ]
    )

    # A string comparison would order "0.10.0" before "0.9.0".
    assert select_indexed_paths(qdrant, "Notes", "version<0.10.0") == ["/p/mid.md", "/p/old.md"]
    assert select_indexed_paths(qdrant, "Notes", "version>=0.9.0") == ["/p/mid.md", "/p/new.md"]
    assert select_indexed_paths(qdrant, "Notes", "version=0.10") == ["/p/new.md"]


def test_terms_are_combined_with_and():
    qdrant = _FakeQdrant(
        [
            _chunk("/p/old-normalized.pdf", "pymupdf_normalized", app_version="0.9.0"),
            _chunk("/p/new-normalized.pdf", "pymupdf_normalized", app_version="0.10.0"),
        ]
    )

    assert select_indexed_paths(qdrant, "Papers", "method=pymupdf_normalized,version<0.10.0") == [
        "/p/old-normalized.pdf"
    ]


def test_dotted_paths_select_fields_without_a_shorthand():
    qdrant = _FakeQdrant(
        [
            _chunk("/p/a.pdf", page_coverage={"page_count": 12}),
            _chunk("/p/b.pdf", page_coverage={"page_count": 3}),
        ]
    )

    assert select_indexed_paths(
        qdrant, "Papers", "quality_manifest.page_coverage.page_count=3"
    ) == ["/p/b.pdf"]


def test_field_absent_from_every_chunk_is_an_error_not_an_empty_selection():
    qdrant = _FakeQdrant([{"file_path": "/p/notes.md", "app_version": "0.10.0"}])

    with pytest.raises(IndexSelectorError, match="quality_manifest.extraction_method"):
        select_indexed_paths(qdrant, "Notes", "method=pymupdf_normalized")


def test_present_field_with_no_matching_value_selects_nothing():
    qdrant = _FakeQdrant([_chunk("/p/clean.pdf")])

    assert select_indexed_paths(qdrant, "Papers", "method=pymupdf_normalized") == []


@pytest.mark.parametrize(
    "expression, message",
    [
        ("", "empty"),
        ("method", "key=value"),
        ("method=", "value"),
        ("method<pymupdf_normalized", "version"),
    ],
)
def test_malformed_selectors_are_rejected(expression, message):
    with pytest.raises(IndexSelectorError, match=message):
        parse_index_selector(expression)


def test_negated_key_selects_points_that_do_not_record_the_field():
    """`!fallback` means the field is absent, as in Kubernetes label selectors.

    Manifests written before fallback_reason existed lack the key entirely,
    which is how stale normalized extractions are told apart from files that
    fell back after the column-order fix.
    """
    qdrant = _FakeQdrant(
        [
            _chunk("/p/before-fix.pdf", "pymupdf_normalized"),
            _chunk("/p/after-fix.pdf", "pymupdf_normalized", fallback_reason="LayoutWorkerTimeout"),
            _chunk("/p/clean.pdf", "pymupdf4llm_markdown", fallback_reason=None),
        ]
    )

    assert select_indexed_paths(qdrant, "Papers", "method=pymupdf_normalized,!fallback") == [
        "/p/before-fix.pdf"
    ]


def test_negated_key_absent_everywhere_is_not_an_error():
    qdrant = _FakeQdrant([_chunk("/p/a.pdf", "pymupdf_normalized")])

    assert select_indexed_paths(qdrant, "Papers", "!fallback") == ["/p/a.pdf"]


def test_negated_key_takes_no_value():
    with pytest.raises(IndexSelectorError, match="no value"):
        parse_index_selector("!fallback=LayoutWorkerTimeout")


def test_negated_key_needs_the_record_that_would_hold_the_field():
    # A point without any quality manifest says nothing about fallback_reason.
    qdrant = _FakeQdrant(
        [
            {"file_path": "/p/no-manifest.pdf", "app_version": "0.10.0"},
            _chunk("/p/old.pdf", "pymupdf_normalized"),
        ]
    )

    assert select_indexed_paths(qdrant, "Papers", "!fallback") == ["/p/old.pdf"]

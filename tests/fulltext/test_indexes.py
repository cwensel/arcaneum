"""Unit tests for index configuration templates (RDR-008)."""

import pytest
from arcaneum.fulltext.indexes import (
    SOURCE_CODE_SETTINGS,
    PDF_DOCS_SETTINGS,
    MARKDOWN_DOCS_SETTINGS,
    get_index_settings,
    get_available_index_types,
)


class TestIndexSettings:
    """Tests for index settings templates."""

    def test_source_code_settings_structure(self):
        """Test source code settings have required fields."""
        assert "searchableAttributes" in SOURCE_CODE_SETTINGS
        assert "filterableAttributes" in SOURCE_CODE_SETTINGS
        assert "typoTolerance" in SOURCE_CODE_SETTINGS
        assert "pagination" in SOURCE_CODE_SETTINGS

    def test_source_code_searchable_attributes(self):
        """Test source code searchable attributes include expected fields."""
        searchable = SOURCE_CODE_SETTINGS["searchableAttributes"]
        assert "content" in searchable
        assert "filename" in searchable
        assert "function_names" in searchable
        assert "class_names" in searchable

    def test_source_code_filterable_attributes(self):
        """Test source code filterable attributes include expected fields."""
        filterable = SOURCE_CODE_SETTINGS["filterableAttributes"]
        assert "language" in filterable
        assert "project" in filterable
        assert "file_path" in filterable

    def test_source_code_typo_tolerance(self):
        """Test source code has higher typo thresholds for code accuracy."""
        typo = SOURCE_CODE_SETTINGS["typoTolerance"]
        assert typo["enabled"] is True
        # Higher thresholds for code (7/12 vs 5/9 for docs)
        assert typo["minWordSizeForTypos"]["oneTypo"] >= 7
        assert typo["minWordSizeForTypos"]["twoTypos"] >= 12

    def test_source_code_no_stop_words(self):
        """Test source code preserves all words (no stop words)."""
        assert SOURCE_CODE_SETTINGS["stopWords"] == []

    def test_pdf_docs_settings_structure(self):
        """Test PDF docs settings have required fields."""
        assert "searchableAttributes" in PDF_DOCS_SETTINGS
        assert "filterableAttributes" in PDF_DOCS_SETTINGS
        assert "sortableAttributes" in PDF_DOCS_SETTINGS
        assert "typoTolerance" in PDF_DOCS_SETTINGS
        assert "stopWords" in PDF_DOCS_SETTINGS

    def test_pdf_docs_has_stop_words(self):
        """Test PDF docs have common stop words configured."""
        stop_words = PDF_DOCS_SETTINGS["stopWords"]
        assert len(stop_words) > 0
        assert "the" in stop_words
        assert "a" in stop_words

    def test_pdf_docs_sortable_by_page(self):
        """Test PDF docs can be sorted by page number."""
        assert "page_number" in PDF_DOCS_SETTINGS["sortableAttributes"]

    def test_markdown_docs_settings_structure(self):
        """Test markdown docs settings have required fields."""
        assert "searchableAttributes" in MARKDOWN_DOCS_SETTINGS
        assert "filterableAttributes" in MARKDOWN_DOCS_SETTINGS
        assert "headings" in MARKDOWN_DOCS_SETTINGS["searchableAttributes"]


class TestGetIndexSettings:
    """Tests for get_index_settings function."""

    def test_get_source_code_settings(self):
        """Test getting source code settings by canonical name."""
        settings = get_index_settings("source-code")
        assert settings == SOURCE_CODE_SETTINGS

    def test_get_pdf_docs_settings(self):
        """Test getting PDF docs settings by canonical name."""
        settings = get_index_settings("pdf-docs")
        assert settings == PDF_DOCS_SETTINGS

    def test_get_markdown_docs_settings(self):
        """Test getting markdown docs settings by canonical name."""
        settings = get_index_settings("markdown-docs")
        assert settings == MARKDOWN_DOCS_SETTINGS

    def test_get_settings_by_alias_code(self):
        """Test getting settings by 'code' alias."""
        settings = get_index_settings("code")
        assert settings == SOURCE_CODE_SETTINGS

    def test_get_settings_by_alias_pdf(self):
        """Test getting settings by 'pdf' alias."""
        settings = get_index_settings("pdf")
        assert settings == PDF_DOCS_SETTINGS

    def test_get_settings_by_alias_markdown(self):
        """Test getting settings by 'markdown' alias."""
        settings = get_index_settings("markdown")
        assert settings == MARKDOWN_DOCS_SETTINGS

    def test_get_settings_returns_copy(self):
        """Test that get_index_settings returns a copy, not the original."""
        settings1 = get_index_settings("code")
        settings2 = get_index_settings("code")

        # Modify one, should not affect the other
        settings1["searchableAttributes"].append("test_field")
        assert "test_field" not in settings2["searchableAttributes"]

    def test_get_unknown_type_raises_error(self):
        """Test that unknown index type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_index_settings("unknown-type")
        assert "Unknown index type" in str(exc_info.value)
        assert "unknown-type" in str(exc_info.value)


class TestGetAvailableIndexTypes:
    """Tests for get_available_index_types function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        types = get_available_index_types()
        assert isinstance(types, list)

    def test_includes_canonical_types(self):
        """Test that canonical types are included."""
        types = get_available_index_types()
        assert "source-code" in types
        assert "pdf-docs" in types
        assert "markdown-docs" in types

    def test_includes_aliases(self):
        """Test that aliases are included."""
        types = get_available_index_types()
        assert "code" in types
        assert "pdf" in types
        assert "markdown" in types

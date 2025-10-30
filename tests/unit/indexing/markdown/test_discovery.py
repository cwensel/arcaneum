"""
Unit tests for markdown file discovery (RDR-014).

Tests validate:
- Recursive file discovery
- Frontmatter extraction
- Content hashing
- File metadata extraction
- Graceful handling of missing frontmatter
- Exclusion patterns
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from arcaneum.indexing.markdown.discovery import (
    MarkdownDiscovery,
    MarkdownFileMetadata,
    discover_markdown_files,
    compute_file_hash,
    FRONTMATTER_AVAILABLE
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def sample_markdown_tree(temp_dir):
    """Create a sample markdown file tree for testing."""
    # Create directory structure
    (temp_dir / "docs").mkdir()
    (temp_dir / "docs" / "guides").mkdir()
    (temp_dir / "node_modules").mkdir()

    # File with frontmatter
    with_frontmatter = """---
title: Sample Document
author: Test Author
tags: [test, sample, demo]
category: documentation
project: arcaneum
created_at: 2025-10-30
custom_field: custom value
---

# Introduction

This is a sample markdown file with frontmatter.

## Section 1

Content for section 1.
"""
    (temp_dir / "docs" / "sample.md").write_text(with_frontmatter)

    # File without frontmatter
    without_frontmatter = """# Simple Document

This is a markdown file without frontmatter.

Just plain content.
"""
    (temp_dir / "docs" / "simple.md").write_text(without_frontmatter)

    # Nested file
    nested_content = """---
title: Nested Guide
tags: guide, nested
---

# Nested Guide

This is in a subdirectory.
"""
    (temp_dir / "docs" / "guides" / "nested.md").write_text(nested_content)

    # File to exclude
    excluded = """# Excluded

This should be excluded.
"""
    (temp_dir / "node_modules" / "excluded.md").write_text(excluded)

    # Different extension
    (temp_dir / "docs" / "readme.markdown").write_text("# Readme\n\nMarkdown extension test.")

    return temp_dir


class TestMarkdownDiscovery:
    """Test suite for MarkdownDiscovery class."""

    def test_initialization(self):
        """Test discovery initializes with correct defaults."""
        discovery = MarkdownDiscovery()
        assert '.md' in discovery.extensions
        assert '.markdown' in discovery.extensions
        assert discovery.exclude_patterns == []

    def test_initialization_custom_extensions(self):
        """Test discovery accepts custom extensions."""
        discovery = MarkdownDiscovery(extensions=['.md', '.txt'])
        assert discovery.extensions == ['.md', '.txt']

    def test_initialization_exclude_patterns(self):
        """Test discovery accepts exclude patterns."""
        discovery = MarkdownDiscovery(exclude_patterns=['**/node_modules/**'])
        assert '**/node_modules/**' in discovery.exclude_patterns

    def test_discover_files_recursive(self, sample_markdown_tree):
        """Test recursive file discovery."""
        discovery = MarkdownDiscovery()
        files = discovery.discover_files(sample_markdown_tree, recursive=True)

        # Should find all .md and .markdown files
        filenames = [f.name for f in files]
        assert 'sample.md' in filenames
        assert 'simple.md' in filenames
        assert 'nested.md' in filenames
        assert 'readme.markdown' in filenames
        assert 'excluded.md' in filenames  # Not excluded yet

    def test_discover_files_non_recursive(self, sample_markdown_tree):
        """Test non-recursive file discovery."""
        discovery = MarkdownDiscovery()
        docs_dir = sample_markdown_tree / "docs"
        files = discovery.discover_files(docs_dir, recursive=False)

        filenames = [f.name for f in files]
        assert 'sample.md' in filenames
        assert 'simple.md' in filenames
        assert 'nested.md' not in filenames  # Should not find nested
        assert 'readme.markdown' in filenames

    def test_discover_files_with_exclusion(self, sample_markdown_tree):
        """Test file discovery with exclusion patterns."""
        discovery = MarkdownDiscovery(exclude_patterns=['**/node_modules/**'])
        files = discovery.discover_files(sample_markdown_tree, recursive=True)

        filenames = [f.name for f in files]
        assert 'excluded.md' not in filenames

    def test_discover_files_nonexistent_directory(self):
        """Test discovery raises error for nonexistent directory."""
        discovery = MarkdownDiscovery()
        with pytest.raises(FileNotFoundError):
            discovery.discover_files(Path("/nonexistent/directory"))

    def test_discover_files_not_a_directory(self, temp_dir):
        """Test discovery raises error when path is not a directory."""
        file_path = temp_dir / "test.md"
        file_path.write_text("# Test")

        discovery = MarkdownDiscovery()
        with pytest.raises(ValueError):
            discovery.discover_files(file_path)

    @pytest.mark.skipif(not FRONTMATTER_AVAILABLE, reason="python-frontmatter not available")
    def test_extract_metadata_with_frontmatter(self, sample_markdown_tree):
        """Test metadata extraction from file with frontmatter."""
        discovery = MarkdownDiscovery()
        file_path = sample_markdown_tree / "docs" / "sample.md"

        metadata = discovery.extract_metadata(file_path)

        assert metadata.file_name == "sample.md"
        assert metadata.file_path == str(file_path.absolute())
        assert metadata.file_size > 0
        assert len(metadata.content_hash) == 64  # SHA256 hex length
        assert metadata.modified_time > 0

        # Frontmatter fields
        assert metadata.has_frontmatter is True
        assert metadata.title == "Sample Document"
        assert metadata.author == "Test Author"
        assert "test" in metadata.tags
        assert "sample" in metadata.tags
        assert metadata.category == "documentation"
        assert metadata.project == "arcaneum"
        # created_at might be parsed as datetime object or string
        assert metadata.created_at is not None
        assert metadata.custom_metadata.get('custom_field') == "custom value"

    def test_extract_metadata_without_frontmatter(self, sample_markdown_tree):
        """Test metadata extraction from file without frontmatter."""
        discovery = MarkdownDiscovery()
        file_path = sample_markdown_tree / "docs" / "simple.md"

        metadata = discovery.extract_metadata(file_path)

        assert metadata.file_name == "simple.md"
        assert metadata.file_size > 0
        assert len(metadata.content_hash) == 64
        assert metadata.has_frontmatter is False
        assert metadata.title is None
        assert metadata.author is None
        assert metadata.tags == []

    def test_extract_metadata_tags_formats(self, temp_dir):
        """Test that tags can be parsed from different formats."""
        # Array format
        array_tags = """---
tags: [tag1, tag2, tag3]
---
Content"""
        (temp_dir / "array.md").write_text(array_tags)

        # Comma-separated string
        string_tags = """---
tags: tag1, tag2, tag3
---
Content"""
        (temp_dir / "string.md").write_text(string_tags)

        discovery = MarkdownDiscovery()

        if FRONTMATTER_AVAILABLE:
            meta1 = discovery.extract_metadata(temp_dir / "array.md")
            assert len(meta1.tags) == 3
            assert "tag1" in meta1.tags

            meta2 = discovery.extract_metadata(temp_dir / "string.md")
            assert len(meta2.tags) == 3
            assert "tag1" in meta2.tags

    def test_extract_metadata_nonexistent_file(self):
        """Test extraction raises error for nonexistent file."""
        discovery = MarkdownDiscovery()
        with pytest.raises(FileNotFoundError):
            discovery.extract_metadata(Path("/nonexistent/file.md"))

    def test_discover_and_extract(self, sample_markdown_tree):
        """Test combined discover and extract operation."""
        discovery = MarkdownDiscovery(exclude_patterns=['**/node_modules/**'])
        metadata_list = discovery.discover_and_extract(sample_markdown_tree, recursive=True)

        assert len(metadata_list) > 0
        filenames = [m.file_name for m in metadata_list]
        assert 'sample.md' in filenames
        assert 'simple.md' in filenames
        assert 'excluded.md' not in filenames

    def test_content_hash_consistency(self, temp_dir):
        """Test that content hash is consistent for same content."""
        content = "# Test\n\nSame content"
        file1 = temp_dir / "file1.md"
        file2 = temp_dir / "file2.md"

        file1.write_text(content)
        file2.write_text(content)

        discovery = MarkdownDiscovery()
        meta1 = discovery.extract_metadata(file1)
        meta2 = discovery.extract_metadata(file2)

        assert meta1.content_hash == meta2.content_hash

    def test_content_hash_changes(self, temp_dir):
        """Test that content hash changes when content changes."""
        file_path = temp_dir / "test.md"
        file_path.write_text("# Version 1")

        discovery = MarkdownDiscovery()
        meta1 = discovery.extract_metadata(file_path)

        # Modify content
        file_path.write_text("# Version 2")
        meta2 = discovery.extract_metadata(file_path)

        assert meta1.content_hash != meta2.content_hash

    def test_read_file_content(self, sample_markdown_tree):
        """Test reading file content without frontmatter."""
        file_path = sample_markdown_tree / "docs" / "sample.md"
        content = MarkdownDiscovery.read_file_content(file_path)

        # Should not include frontmatter
        if FRONTMATTER_AVAILABLE:
            assert '---' not in content  # Frontmatter stripped
            assert '# Introduction' in content

    def test_read_file_with_frontmatter(self, sample_markdown_tree):
        """Test reading file with frontmatter extraction."""
        file_path = sample_markdown_tree / "docs" / "sample.md"
        content, frontmatter_dict = MarkdownDiscovery.read_file_with_frontmatter(file_path)

        if FRONTMATTER_AVAILABLE:
            assert '---' not in content  # Body only
            assert '# Introduction' in content
            assert frontmatter_dict.get('title') == "Sample Document"
        else:
            assert content  # Should still return content
            assert frontmatter_dict == {}

    def test_unicode_handling(self, temp_dir):
        """Test handling of unicode content."""
        unicode_content = """---
title: Unicode Test
---

# Test with émoji 😀

Content with spéciål characters.
"""
        file_path = temp_dir / "unicode.md"
        file_path.write_text(unicode_content, encoding='utf-8')

        discovery = MarkdownDiscovery()
        metadata = discovery.extract_metadata(file_path)

        assert metadata.file_name == "unicode.md"
        if FRONTMATTER_AVAILABLE:
            assert metadata.title == "Unicode Test"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_discover_markdown_files(self, sample_markdown_tree):
        """Test discover_markdown_files convenience function."""
        metadata_list = discover_markdown_files(
            sample_markdown_tree,
            recursive=True,
            exclude_patterns=['**/node_modules/**']
        )

        assert len(metadata_list) > 0
        filenames = [m.file_name for m in metadata_list]
        assert 'sample.md' in filenames

    def test_compute_file_hash(self, temp_dir):
        """Test compute_file_hash convenience function."""
        content = "# Test content"
        file_path = temp_dir / "test.md"
        file_path.write_text(content)

        hash1 = compute_file_hash(file_path)
        hash2 = compute_file_hash(file_path)

        assert len(hash1) == 64  # SHA256 hex
        assert hash1 == hash2  # Consistent


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_file(self, temp_dir):
        """Test handling of empty markdown file."""
        file_path = temp_dir / "empty.md"
        file_path.write_text("")

        discovery = MarkdownDiscovery()
        metadata = discovery.extract_metadata(file_path)

        assert metadata.file_name == "empty.md"
        assert metadata.file_size == 0
        assert len(metadata.content_hash) == 64

    def test_large_file(self, temp_dir):
        """Test handling of large markdown file."""
        large_content = "# Large File\n\n" + ("paragraph\n\n" * 10000)
        file_path = temp_dir / "large.md"
        file_path.write_text(large_content)

        discovery = MarkdownDiscovery()
        metadata = discovery.extract_metadata(file_path)

        assert metadata.file_size > 100000
        assert len(metadata.content_hash) == 64

    def test_malformed_frontmatter(self, temp_dir):
        """Test handling of malformed frontmatter."""
        malformed = """---
title: Test
invalid yaml: [unclosed
---

Content"""
        file_path = temp_dir / "malformed.md"
        file_path.write_text(malformed)

        discovery = MarkdownDiscovery()
        # Should not raise, should handle gracefully
        metadata = discovery.extract_metadata(file_path)
        assert metadata.file_name == "malformed.md"

    def test_no_markdown_files(self, temp_dir):
        """Test discovery in directory with no markdown files."""
        (temp_dir / "test.txt").write_text("Not markdown")

        discovery = MarkdownDiscovery()
        files = discovery.discover_files(temp_dir)

        assert len(files) == 0

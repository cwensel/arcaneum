"""
Unit tests for semantic markdown chunker (RDR-014).

Tests validate:
- Document structure preservation
- Parent header context
- Nested heading handling
- Code block integrity
- Configurable chunk sizes
- Fallback behavior
"""

import pytest

from arcaneum.indexing.markdown.chunker import (
    MARKDOWN_IT_AVAILABLE,
    SemanticMarkdownChunker,
    chunk_markdown,
)


class TestSemanticMarkdownChunker:
    """Test suite for SemanticMarkdownChunker."""

    @staticmethod
    def _assert_non_frontmatter_lines_represented(source_text, chunks):
        chunk_text = "\n".join(chunk.text for chunk in chunks)
        lines = source_text.splitlines()
        in_frontmatter = bool(lines and lines[0] == "---")

        for line_number, line in enumerate(lines, start=1):
            if in_frontmatter:
                if line_number > 1 and line == "---":
                    in_frontmatter = False
                continue
            if not line.strip():
                continue
            assert line in chunk_text, f"line {line_number} missing from chunks: {line!r}"

    def test_initialization(self):
        """Test chunker initializes with correct defaults."""
        chunker = SemanticMarkdownChunker()
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50
        assert chunker.preserve_code_blocks is True
        assert chunker.max_chars == int(512 * 3.3)

    def test_initialization_custom_params(self):
        """Test chunker accepts custom parameters."""
        chunker = SemanticMarkdownChunker(
            chunk_size=1024,
            chunk_overlap=100,
            max_chars=5000,
            hard_max_chars=8000,
            preserve_code_blocks=False
        )
        assert chunker.chunk_size == 1024
        assert chunker.chunk_overlap == 100
        assert chunker.max_chars == 5000
        assert chunker.hard_max_chars == 8000
        assert chunker.preserve_code_blocks is False

    def test_empty_text(self):
        """Test chunker handles empty text gracefully."""
        chunker = SemanticMarkdownChunker()
        chunks = chunker.chunk("", {})
        assert chunks == []

        chunks = chunker.chunk("   \n\n  ", {})
        assert chunks == []

    def test_simple_paragraph(self):
        """Test chunker handles simple paragraph without headers."""
        text = "This is a simple paragraph with some text content."
        chunker = SemanticMarkdownChunker(chunk_size=100)
        chunks = chunker.chunk(text, {'source': 'test'})

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0
        assert chunks[0].metadata['source'] == 'test'

    def test_single_header_section(self):
        """Test chunker preserves single header with content."""
        text = """# Introduction

This is the introduction section with some content.
It has multiple lines of text."""

        chunker = SemanticMarkdownChunker(chunk_size=100)
        chunks = chunker.chunk(text, {})

        assert len(chunks) == 1
        assert '# Introduction' in chunks[0].text
        assert 'introduction section' in chunks[0].text
        assert chunks[0].header_path == ['Introduction']

    @pytest.mark.skipif(not MARKDOWN_IT_AVAILABLE, reason="markdown-it-py not available")
    def test_intro_before_first_heading_is_preserved(self):
        """Preamble text before the first heading must not be dropped."""
        text = """Intro line before any heading.
Still part of the document preamble.

# Section

Section content."""

        chunker = SemanticMarkdownChunker(chunk_size=100)
        chunks = chunker.chunk(text, {})

        assert chunks[0].text.startswith("Intro line before any heading.")
        assert "Still part of the document preamble." in chunks[0].text
        self._assert_non_frontmatter_lines_represented(text, chunks)

    @pytest.mark.skipif(not MARKDOWN_IT_AVAILABLE, reason="markdown-it-py not available")
    def test_raw_markdown_spans_preserve_inline_structure(self):
        """Links, images, HTML, and code fences stay in chunk text verbatim."""
        text = """---
title: Raw Span Fixture
---
# Assets

Read [the guide](https://example.test/guide) before deploying.

![architecture](../assets/arch.png)

<div data-kind="note">raw html</div>

```python
print("[literal markdown](not-a-link)")
```
"""

        chunker = SemanticMarkdownChunker(chunk_size=1000)
        chunks = chunker.chunk(text, {})
        chunk_text = "\n".join(chunk.text for chunk in chunks)

        assert "[the guide](https://example.test/guide)" in chunk_text
        assert "![architecture](../assets/arch.png)" in chunk_text
        assert '<div data-kind="note">raw html</div>' in chunk_text
        assert 'print("[literal markdown](not-a-link)")' in chunk_text
        self._assert_non_frontmatter_lines_represented(text, chunks)

    def test_nested_headers(self):
        """Test chunker handles nested header hierarchy correctly."""
        text = """# Main Title

Introduction content.

## Subsection 1

Content for subsection 1.

## Subsection 2

Content for subsection 2.

### Deep section

Deep content here."""

        chunker = SemanticMarkdownChunker(chunk_size=100)
        chunks = chunker.chunk(text, {})

        # Verify header paths are preserved
        header_paths = [c.header_path for c in chunks]

        # Check that nested structure is captured
        assert any('Main Title' in path for path in header_paths)
        assert any('Subsection 1' in path for path in header_paths)
        assert any('Subsection 2' in path for path in header_paths)
        assert any('Deep section' in path for path in header_paths)

    def test_parent_header_context_preservation(self):
        """Test that parent headers are included in sub-section chunks."""
        text = """# Chapter 1

## Section 1.1

Content for section 1.1.

### Subsection 1.1.1

Detailed content here."""

        chunker = SemanticMarkdownChunker(chunk_size=50)
        chunks = chunker.chunk(text, {})

        # Find chunk with deepest nesting
        deep_chunks = [c for c in chunks if 'Subsection 1.1.1' in c.header_path]
        assert len(deep_chunks) > 0

        # Verify full path is preserved
        deep_chunk = deep_chunks[0]
        assert 'Chapter 1' in deep_chunk.header_path
        assert 'Section 1.1' in deep_chunk.header_path
        assert 'Subsection 1.1.1' in deep_chunk.header_path

    def test_code_block_preservation(self):
        """Test that code blocks remain intact and aren't split."""
        text = """# Code Example

Here's some code:

```python
def hello_world():
    print("Hello, world!")
    return 42
```

More text after code."""

        chunker = SemanticMarkdownChunker(chunk_size=50, preserve_code_blocks=True)
        chunks = chunker.chunk(text, {})

        # Find chunk with code block
        code_chunks = [c for c in chunks if '```python' in c.text or 'def hello_world' in c.text]
        assert len(code_chunks) > 0

        # Verify code block is complete in at least one chunk
        has_complete_code = any(
            'def hello_world():' in c.text and 'return 42' in c.text
            for c in code_chunks
        )
        assert has_complete_code

    def test_code_block_metadata(self):
        """Test that chunks with code blocks are marked in metadata."""
        text = """# Example

Some text.

```javascript
const x = 42;
```

More text."""

        chunker = SemanticMarkdownChunker(chunk_size=100)
        chunks = chunker.chunk(text, {})

        # Find chunk with code
        code_chunks = [c for c in chunks if 'const x = 42' in c.text]
        assert len(code_chunks) > 0
        assert code_chunks[0].metadata.get('has_code_blocks') is True

    def test_large_section_splitting(self):
        """Test that large sections are split at semantic boundaries."""
        # Create a large section that exceeds chunk size
        paragraphs = [f"This is paragraph {i} with some content." for i in range(20)]
        text = f"""# Large Section

{chr(10).join(paragraphs)}"""

        chunker = SemanticMarkdownChunker(chunk_size=50)
        chunks = chunker.chunk(text, {})

        # Should create multiple chunks
        assert len(chunks) > 1

        # All chunks should have the header in their path
        for chunk in chunks:
            assert 'Large Section' in chunk.header_path

    def test_hard_max_splits_oversized_semantic_chunk(self):
        """Oversized markdown chunks are windowed instead of truncated."""
        prefix = "A" * 450
        middle = "B" * 450
        tail = "TAIL_SENTINEL"
        text = f"# Huge Paragraph\n\n{prefix}{middle}{tail}"

        chunker = SemanticMarkdownChunker(
            chunk_size=1000,
            chunk_overlap=10,
            hard_max_chars=300,
        )
        chunks = chunker.chunk(text, {})

        assert len(chunks) > 1
        assert all(len(chunk.text) <= 300 for chunk in chunks)
        assert any(tail in chunk.text for chunk in chunks)
        assert any(chunk.metadata.get('hard_split') is True for chunk in chunks)

    def test_hard_max_windowing_uses_overlap(self):
        """Hard-max windows preserve overlap so boundary context is retained."""
        text = "# Huge Paragraph\n\n" + "".join(str(i % 10) for i in range(900))

        chunker = SemanticMarkdownChunker(
            chunk_size=1000,
            chunk_overlap=30,
            hard_max_chars=300,
        )
        chunks = chunker.chunk(text, {})

        split_chunks = [chunk for chunk in chunks if chunk.metadata.get('hard_split')]
        assert len(split_chunks) > 1

        first = split_chunks[0]
        second = split_chunks[1]
        assert second.metadata['chunk_start_char'] < first.metadata['chunk_end_char']

    def test_chunk_indices(self):
        """Test that chunk indices are sequential."""
        text = """# Section 1

Content 1.

# Section 2

Content 2.

# Section 3

Content 3."""

        chunker = SemanticMarkdownChunker(chunk_size=50)
        chunks = chunker.chunk(text, {})

        # Verify indices are sequential
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_token_count_estimation(self):
        """Test that token counts are reasonably estimated."""
        text = "This is a test sentence with approximately ten tokens in it."
        chunker = SemanticMarkdownChunker()
        chunks = chunker.chunk(text, {})

        assert len(chunks) == 1
        # Token count should be roughly chars/3.3
        expected_tokens = len(text) / 3.3
        assert abs(chunks[0].token_count - expected_tokens) < 5

    def test_metadata_propagation(self):
        """Test that base metadata is propagated to all chunks."""
        text = """# Section 1

Content 1.

# Section 2

Content 2."""

        base_metadata = {
            'file_path': '/test/doc.md',
            'source': 'test',
            'priority': 'high'
        }

        chunker = SemanticMarkdownChunker(chunk_size=50)
        chunks = chunker.chunk(text, base_metadata)

        for chunk in chunks:
            assert chunk.metadata['file_path'] == '/test/doc.md'
            assert chunk.metadata['source'] == 'test'
            assert chunk.metadata['priority'] == 'high'

    def test_list_handling(self):
        """Test that markdown lists are handled properly."""
        text = """# Todo List

Here are the tasks:

- First task
- Second task
- Third task

Done."""

        chunker = SemanticMarkdownChunker(chunk_size=100)
        chunks = chunker.chunk(text, {})

        # Should keep list together if possible
        list_chunks = [c for c in chunks if '- First task' in c.text]
        if list_chunks:
            # Ideally all list items in same chunk
            assert '- Second task' in list_chunks[0].text or len(chunks) > 1

    def test_table_handling(self):
        """Test that markdown tables are preserved."""
        text = """# Data

| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |

End."""

        chunker = SemanticMarkdownChunker(chunk_size=100)
        chunks = chunker.chunk(text, {})

        # Find chunk with table
        table_chunks = [c for c in chunks if '| Column 1 |' in c.text]
        assert len(table_chunks) > 0

    @pytest.mark.skipif(not MARKDOWN_IT_AVAILABLE, reason="markdown-it-py not available")
    def test_semantic_chunking_used(self):
        """Test that semantic chunking is used when markdown-it-py available."""
        text = """# Test

Content here."""

        chunker = SemanticMarkdownChunker()
        chunks = chunker.chunk(text, {})

        # The naive fallback path tags chunks with semantic_chunking=False; the
        # semantic path does not set the key at all. Verifying "not False"
        # distinguishes the two paths correctly.
        assert chunks[0].metadata.get('semantic_chunking', True) is not False

    def test_naive_chunking_fallback(self):
        """Test that naive chunking works when markdown-it-py unavailable."""
        text = "A " * 1000  # Create long text

        # chunk_overlap must be < chunk_size to avoid infinite loop in _naive_chunking
        chunker = SemanticMarkdownChunker(chunk_size=100, chunk_overlap=10)

        # Force naive chunking by setting md to None
        original_md = chunker.md
        chunker.md = None

        try:
            chunks = chunker.chunk(text, {})
            assert len(chunks) > 1
            assert all(c.chunk_index == i for i, c in enumerate(chunks))
        finally:
            chunker.md = original_md

    def test_header_path_formatting(self):
        """Test that header paths are formatted correctly in metadata."""
        text = """# Level 1

## Level 2

### Level 3

Content."""

        chunker = SemanticMarkdownChunker(chunk_size=100)
        chunks = chunker.chunk(text, {})

        # Find deepest chunk
        deep_chunks = [c for c in chunks if len(c.header_path) == 3]
        if deep_chunks:
            # Check metadata has formatted path
            assert 'Level 1 > Level 2 > Level 3' in deep_chunks[0].metadata.get('header_path', '')


class TestChunkMarkdownFunction:
    """Test the convenience function chunk_markdown."""

    def test_chunk_markdown_with_params(self):
        """Test chunk_markdown with custom parameters."""
        text = "# Test\n\n" + ("Content. " * 500)
        chunks = chunk_markdown(text, chunk_size=100, chunk_overlap=10)

        assert len(chunks) > 1

    def test_chunk_markdown_with_metadata(self):
        """Test chunk_markdown with metadata."""
        text = "# Test\n\nContent."
        metadata = {'file': 'test.md'}
        chunks = chunk_markdown(text, metadata=metadata)

        assert chunks[0].metadata['file'] == 'test.md'

    def test_chunk_markdown_accepts_hard_max_chars(self):
        """Convenience wrapper forwards hard_max_chars."""
        text = "# Test\n\n" + ("Content " * 200)
        chunks = chunk_markdown(
            text,
            chunk_size=1000,
            chunk_overlap=10,
            hard_max_chars=200,
        )

        assert len(chunks) > 1
        assert all(len(chunk.text) <= 200 for chunk in chunks)


class TestAcceptanceCriteria:
    """Validate acceptance criteria from RDR-014."""

    def test_handles_nested_headings(self):
        """Acceptance: Handles nested headings correctly."""
        text = """# H1
## H2
### H3
#### H4
##### H5
###### H6

Content at each level."""

        chunks = chunk_markdown(text, chunk_size=100)

        # All heading levels should be captured
        all_headers = []
        for c in chunks:
            all_headers.extend(c.header_path)

        assert 'H1' in all_headers
        assert 'H6' in all_headers

    def test_configurable_chunk_size(self):
        """Acceptance: Configurable chunk size.

        Uses structured markdown with multiple sections so the chunker has
        semantic boundaries to split on.
        """
        # Generate markdown with many small sections
        sections = []
        for i in range(20):
            sections.append(f"## Section {i}\n\n" + "Word " * 50)
        text = "\n\n".join(sections)

        # Small chunks
        small_chunks = chunk_markdown(text, chunk_size=50)

        # Large chunks
        large_chunks = chunk_markdown(text, chunk_size=500)

        # Large chunk size should produce fewer chunks
        assert len(large_chunks) < len(small_chunks)

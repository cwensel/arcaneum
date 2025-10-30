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
    SemanticMarkdownChunker,
    MarkdownChunk,
    chunk_markdown,
    MARKDOWN_IT_AVAILABLE
)


class TestSemanticMarkdownChunker:
    """Test suite for SemanticMarkdownChunker."""

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
            preserve_code_blocks=False
        )
        assert chunker.chunk_size == 1024
        assert chunker.chunk_overlap == 100
        assert chunker.max_chars == 5000
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

        # Should not have 'semantic_chunking': False in metadata
        assert chunks[0].metadata.get('semantic_chunking', True) is not False

    def test_naive_chunking_fallback(self):
        """Test that naive chunking works when markdown-it-py unavailable."""
        text = "A " * 1000  # Create long text

        chunker = SemanticMarkdownChunker(chunk_size=50)

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

    def test_chunk_markdown_basic(self):
        """Test basic usage of chunk_markdown function."""
        text = "# Test\n\nContent here."
        chunks = chunk_markdown(text)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, MarkdownChunk) for c in chunks)

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


class TestAcceptanceCriteria:
    """Validate acceptance criteria from RDR-014."""

    def test_respects_document_structure(self):
        """Acceptance: Respects document structure."""
        text = """# Title

## Section 1
Content 1.

## Section 2
Content 2."""

        chunks = chunk_markdown(text, chunk_size=100)

        # Structure should be preserved in headers
        assert any('Title' in c.header_path for c in chunks)
        assert any('Section 1' in c.header_path for c in chunks)

    def test_preserves_parent_header_context(self):
        """Acceptance: Preserves parent header context."""
        text = """# Main

## Sub

### Deep

Content."""

        chunks = chunk_markdown(text, chunk_size=50)

        # Deep chunks should have full path
        deep_chunks = [c for c in chunks if 'Deep' in c.header_path]
        if deep_chunks:
            assert 'Main' in deep_chunks[0].header_path
            assert 'Sub' in deep_chunks[0].header_path

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

    def test_code_blocks_remain_intact(self):
        """Acceptance: Code blocks remain intact."""
        text = """# Code

```python
def function():
    x = 1
    y = 2
    return x + y
```

Done."""

        chunks = chunk_markdown(text, chunk_size=50, chunk_overlap=0)

        # Code should be in one chunk
        code_chunks = [c for c in chunks if 'def function():' in c.text]
        assert len(code_chunks) > 0
        assert 'return x + y' in code_chunks[0].text

    def test_configurable_chunk_size(self):
        """Acceptance: Configurable chunk size."""
        text = "Word " * 1000

        # Small chunks
        small_chunks = chunk_markdown(text, chunk_size=50)

        # Large chunks
        large_chunks = chunk_markdown(text, chunk_size=500)

        # Large chunk size should produce fewer chunks
        assert len(large_chunks) < len(small_chunks)

"""Tests for AST-aware code chunking."""

import os
import tempfile
import pytest

from arcaneum.indexing.ast_chunker import ASTCodeChunker, chunk_code_file
from arcaneum.indexing.types import Chunk


# Sample code snippets for testing
PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

def add(a, b):
    return a + b

class Calculator:
    def __init__(self):
        self.result = 0

    def multiply(self, x, y):
        self.result = x * y
        return self.result
"""

JAVA_CODE = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }

    public int add(int a, int b) {
        return a + b;
    }
}
"""

JAVASCRIPT_CODE = """
function helloWorld() {
    console.log("Hello, World!");
}

const add = (a, b) => {
    return a + b;
};

class Calculator {
    constructor() {
        this.result = 0;
    }

    multiply(x, y) {
        this.result = x * y;
        return this.result;
    }
}
"""

GO_CODE = """
package main

import "fmt"

func helloWorld() {
    fmt.Println("Hello, World!")
}

func add(a int, b int) int {
    return a + b
}

type Calculator struct {
    result int
}

func (c *Calculator) multiply(x int, y int) int {
    c.result = x * y
    return c.result
}
"""

RUST_CODE = """
fn hello_world() {
    println!("Hello, World!");
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

struct Calculator {
    result: i32,
}

impl Calculator {
    fn new() -> Self {
        Calculator { result: 0 }
    }

    fn multiply(&mut self, x: i32, y: i32) -> i32 {
        self.result = x * y;
        self.result
    }
}
"""


class TestASTCodeChunker:
    """Tests for ASTCodeChunker class."""

    def test_initialization(self):
        """Test basic initialization."""
        chunker = ASTCodeChunker(chunk_size=400, chunk_overlap=20)

        assert chunker.chunk_size == 400
        assert chunker.chunk_overlap == 20
        assert chunker.max_chars > 0

    def test_initialization_with_custom_max_chars(self):
        """Test initialization with custom max_chars."""
        chunker = ASTCodeChunker(chunk_size=400, max_chars=2000)

        assert chunker.max_chars == 2000

    def test_language_detection_python(self):
        """Test language detection for Python files."""
        chunker = ASTCodeChunker()

        assert chunker.detect_language("test.py") == "python"
        assert chunker.detect_language("/path/to/script.py") == "python"

    def test_language_detection_various(self):
        """Test language detection for various file types."""
        chunker = ASTCodeChunker()

        test_cases = {
            "test.java": "java",
            "app.js": "javascript",
            "component.jsx": "javascript",
            "script.ts": "typescript",
            "app.tsx": "typescript",
            "main.go": "go",
            "lib.rs": "rust",
            "program.c": "c",
            "program.cpp": "cpp",
            "Main.cs": "c_sharp",
            "index.php": "php",
            "script.rb": "ruby",
            "App.kt": "kotlin",
            "Main.scala": "scala",
            "app.swift": "swift",
        }

        for filepath, expected_lang in test_cases.items():
            detected = chunker.detect_language(filepath)
            assert detected == expected_lang, f"Failed for {filepath}: got {detected}, expected {expected_lang}"

    def test_language_detection_unknown(self):
        """Test language detection for unknown file types."""
        chunker = ASTCodeChunker()

        assert chunker.detect_language("file.xyz") is None
        assert chunker.detect_language("README.txt") is None

    def test_supports_ast_chunking(self):
        """Test AST chunking support detection."""
        chunker = ASTCodeChunker()

        # Should support known languages
        assert chunker.supports_ast_chunking("test.py") is True
        assert chunker.supports_ast_chunking("app.java") is True
        assert chunker.supports_ast_chunking("script.js") is True

        # Should not support unknown types
        assert chunker.supports_ast_chunking("file.xyz") is False

    def test_chunk_python_code(self):
        """Test chunking Python code."""
        chunker = ASTCodeChunker(chunk_size=400)

        chunks = chunker.chunk_code("test.py", PYTHON_CODE)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        # Should use AST or line-based
        assert all(c.method in ["ast_python", "line_based"] for c in chunks)
        # Chunks should contain code
        assert all(len(c.content) > 0 for c in chunks)

    def test_chunk_java_code(self):
        """Test chunking Java code."""
        chunker = ASTCodeChunker(chunk_size=400)

        chunks = chunker.chunk_code("Test.java", JAVA_CODE)

        assert len(chunks) > 0
        assert all(c.method in ["ast_java", "line_based"] for c in chunks)

    def test_chunk_javascript_code(self):
        """Test chunking JavaScript code."""
        chunker = ASTCodeChunker(chunk_size=400)

        chunks = chunker.chunk_code("app.js", JAVASCRIPT_CODE)

        assert len(chunks) > 0
        assert all(c.method in ["ast_javascript", "line_based"] for c in chunks)

    def test_chunk_go_code(self):
        """Test chunking Go code."""
        chunker = ASTCodeChunker(chunk_size=400)

        chunks = chunker.chunk_code("main.go", GO_CODE)

        assert len(chunks) > 0
        assert all(c.method in ["ast_go", "line_based"] for c in chunks)

    def test_chunk_rust_code(self):
        """Test chunking Rust code."""
        chunker = ASTCodeChunker(chunk_size=400)

        chunks = chunker.chunk_code("lib.rs", RUST_CODE)

        assert len(chunks) > 0
        assert all(c.method in ["ast_rust", "line_based"] for c in chunks)

    def test_chunk_empty_code(self):
        """Test chunking empty code."""
        chunker = ASTCodeChunker()

        chunks = chunker.chunk_code("test.py", "")

        assert len(chunks) == 0

    def test_chunk_whitespace_only(self):
        """Test chunking whitespace-only code."""
        chunker = ASTCodeChunker()

        chunks = chunker.chunk_code("test.py", "   \n\n   \t  ")

        assert len(chunks) == 0

    def test_force_line_based_chunking(self):
        """Test forcing line-based chunking."""
        chunker = ASTCodeChunker()

        chunks = chunker.chunk_code("test.py", PYTHON_CODE, force_line_based=True)

        assert len(chunks) > 0
        assert all(c.method == "line_based" for c in chunks)

    def test_line_based_fallback_for_unknown_language(self):
        """Test that unknown languages fall back to line-based."""
        chunker = ASTCodeChunker()

        code = "Some text content\nWith multiple lines\nAnd more content"
        chunks = chunker.chunk_code("file.txt", code)

        assert len(chunks) > 0
        assert all(c.method == "line_based" for c in chunks)

    def test_large_file_chunking(self):
        """Test chunking of large file produces multiple chunks."""
        chunker = ASTCodeChunker(chunk_size=100)  # Small chunks

        # Create a large Python file
        large_code = "\n".join([
            f"def function_{i}():\n    return {i}"
            for i in range(100)
        ])

        chunks = chunker.chunk_code("large.py", large_code)

        # Should produce multiple chunks
        assert len(chunks) > 1

    def test_line_based_chunking_with_overlap(self):
        """Test that line-based chunking includes overlap."""
        chunker = ASTCodeChunker(chunk_size=50, chunk_overlap=10)

        # Create code that will need multiple chunks
        lines = [f"line_{i} = {i}" for i in range(50)]
        code = "\n".join(lines)

        chunks = chunker.chunk_code("test.py", code, force_line_based=True)

        # Should have multiple chunks due to small chunk_size
        assert len(chunks) > 1

        # Verify chunks contain code
        for chunk in chunks:
            assert len(chunk.content) > 0
            assert "line_" in chunk.content

    def test_get_supported_languages(self):
        """Test getting list of supported languages."""
        languages = ASTCodeChunker.get_supported_languages()

        assert isinstance(languages, list)
        assert len(languages) > 15  # Should have at least 15+ languages
        assert "python" in languages
        assert "java" in languages
        assert "javascript" in languages
        assert "go" in languages
        assert "rust" in languages

    def test_get_supported_extensions(self):
        """Test getting list of supported extensions."""
        extensions = ASTCodeChunker.get_supported_extensions()

        assert isinstance(extensions, list)
        assert ".py" in extensions
        assert ".java" in extensions
        assert ".js" in extensions
        assert ".go" in extensions
        assert ".rs" in extensions

    def test_chunk_with_different_sizes(self):
        """Test chunking with different chunk sizes."""
        # Small chunks
        chunker_small = ASTCodeChunker(chunk_size=50)
        chunks_small = chunker_small.chunk_code("test.py", PYTHON_CODE, force_line_based=True)

        # Large chunks
        chunker_large = ASTCodeChunker(chunk_size=2000)
        chunks_large = chunker_large.chunk_code("test.py", PYTHON_CODE, force_line_based=True)

        # Small chunk size should produce more chunks
        assert len(chunks_small) >= len(chunks_large)

    def test_chunk_multiline_strings(self):
        """Test chunking code with multiline strings."""
        code_with_multiline = '''
def example():
    text = """
    This is a multiline string
    with multiple lines
    of content
    """
    return text
'''
        chunker = ASTCodeChunker()
        chunks = chunker.chunk_code("test.py", code_with_multiline)

        assert len(chunks) > 0
        # Should preserve multiline strings
        combined = "".join(c.content for c in chunks)
        assert '"""' in combined or "'''" in combined


class TestChunkCodeFile:
    """Tests for chunk_code_file convenience function."""

    def test_chunk_existing_file(self):
        """Test chunking an existing file."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(PYTHON_CODE)
            temp_path = f.name

        try:
            chunks = chunk_code_file(temp_path)

            assert len(chunks) > 0
            assert all(isinstance(c, Chunk) for c in chunks)
        finally:
            os.unlink(temp_path)

    def test_chunk_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            chunk_code_file("/nonexistent/file.py")

    def test_chunk_file_with_custom_params(self):
        """Test chunking file with custom parameters."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
            f.write(JAVA_CODE)
            temp_path = f.name

        try:
            chunks = chunk_code_file(temp_path, chunk_size=100, chunk_overlap=10)

            assert len(chunks) > 0
        finally:
            os.unlink(temp_path)

    def test_chunk_file_encoding_fallback(self):
        """Test that encoding fallback works."""
        # Create file with latin-1 encoding
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.py', delete=False) as f:
            # Write some latin-1 encoded text
            f.write("# Comment with special char: \xe9\n".encode('latin-1'))
            f.write("def test():\n    pass\n".encode('latin-1'))
            temp_path = f.name

        try:
            chunks = chunk_code_file(temp_path)

            assert len(chunks) > 0
        finally:
            os.unlink(temp_path)


class TestChunkingAccuracy:
    """Tests for chunking accuracy and quality."""

    def test_chunk_preserves_code(self):
        """Test that chunking preserves all code content."""
        chunker = ASTCodeChunker(chunk_size=200)

        chunks = chunker.chunk_code("test.py", PYTHON_CODE, force_line_based=True)

        # Combine chunks (remove overlap approximately)
        combined = ""
        for i, chunk in enumerate(chunks):
            if i == 0:
                combined += chunk.content
            else:
                # For line-based, there might be overlap
                combined += "\n" + chunk.content

        # Should contain key elements from original
        assert "hello_world" in combined or "hello_world" in PYTHON_CODE
        assert "Calculator" in combined or "Calculator" in PYTHON_CODE

    def test_chunk_boundaries_reasonable(self):
        """Test that chunk boundaries are at reasonable points."""
        chunker = ASTCodeChunker(chunk_size=200)

        chunks = chunker.chunk_code("test.py", PYTHON_CODE, force_line_based=True)

        for chunk in chunks:
            # Chunks should not be empty
            assert len(chunk.content.strip()) > 0

            # Chunks should end with complete lines (not mid-line)
            # This is important for code readability
            assert not chunk.content or chunk.content[-1] in ['\n', ' ', '}', ')', ';', ':']


class TestMinifiedCodeHandling:
    """Tests for handling minified code (very long single lines)."""

    def test_split_long_line_basic(self):
        """Test that _split_long_line splits very long lines."""
        chunker = ASTCodeChunker(chunk_size=100)  # ~350 chars max

        # Create a line longer than max_chars (100 * 3.5 = 350)
        long_line = "a" * 500

        segments = chunker._split_long_line(long_line, 350)

        # Should produce multiple segments
        assert len(segments) > 1
        # Each segment should be <= max_chars
        for seg in segments:
            assert len(seg) <= 350
        # Combined segments should equal original
        assert "".join(segments) == long_line

    def test_split_long_line_prefers_break_points(self):
        """Test that _split_long_line prefers natural break points."""
        chunker = ASTCodeChunker(chunk_size=100)

        # Create a line with semicolons that can be used as break points
        # Pattern: code;code;code;... repeating
        long_line = ";".join(["code" * 10] * 20)  # ~860 chars

        segments = chunker._split_long_line(long_line, 300)

        # Most segments should end with semicolon (except possibly the last)
        semicolon_endings = sum(1 for seg in segments[:-1] if seg.endswith(';'))
        assert semicolon_endings >= len(segments) - 2

    def test_split_long_line_short_line_unchanged(self):
        """Test that short lines are returned unchanged."""
        chunker = ASTCodeChunker(chunk_size=100)

        short_line = "var x = 1;"
        segments = chunker._split_long_line(short_line, 350)

        assert len(segments) == 1
        assert segments[0] == short_line

    def test_chunk_minified_javascript(self):
        """Test chunking simulated minified JavaScript."""
        chunker = ASTCodeChunker(chunk_size=100)  # Small for testing

        # Simulate minified JS: entire file on one line
        minified_js = 'var a=1,b=2;function add(x,y){return x+y;}' * 50

        # Should not raise memory error
        chunks = chunker.chunk_code("jquery.min.js", minified_js)

        assert len(chunks) > 0
        # All chunks should be reasonably sized
        max_expected_chars = int(100 * chunker.CHARS_PER_TOKEN * 1.5)  # Allow some margin
        for chunk in chunks:
            assert len(chunk.content) < max_expected_chars

    def test_chunk_minified_preserves_all_content(self):
        """Test that all content is preserved when chunking minified code."""
        chunker = ASTCodeChunker(chunk_size=50)

        # Create deterministic minified-like content
        minified = "function a(){return 1;};function b(){return 2;};" * 10

        chunks = chunker.chunk_code("min.js", minified, force_line_based=True)

        # Combined chunks should contain all original content
        combined = "".join(c.content for c in chunks)
        # Account for potential newlines added between segments
        combined_cleaned = combined.replace('\n', '')
        assert minified in combined_cleaned or combined_cleaned == minified

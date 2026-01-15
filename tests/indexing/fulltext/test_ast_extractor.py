"""Unit tests for ASTFunctionExtractor (RDR-011).

Tests the AST-based function/class extraction for full-text indexing.
"""

import pytest
from pathlib import Path

from arcaneum.indexing.fulltext.ast_extractor import (
    ASTFunctionExtractor,
    CodeDefinition,
    DEFINITION_TYPES,
    TREE_SITTER_AVAILABLE,
)


class TestCodeDefinition:
    """Tests for CodeDefinition dataclass."""

    def test_line_count_single_line(self):
        """Test line count for single-line definition."""
        defn = CodeDefinition(
            name="test",
            qualified_name="test",
            code_type="function",
            start_line=5,
            end_line=5,
            content="def test(): pass",
            file_path="/test.py"
        )
        assert defn.line_count == 1

    def test_line_count_multi_line(self):
        """Test line count for multi-line definition."""
        defn = CodeDefinition(
            name="test",
            qualified_name="test",
            code_type="function",
            start_line=10,
            end_line=25,
            content="def test():\n    pass",
            file_path="/test.py"
        )
        assert defn.line_count == 16

    def test_all_fields(self):
        """Test that all fields are properly set."""
        defn = CodeDefinition(
            name="my_func",
            qualified_name="MyClass.my_func",
            code_type="method",
            start_line=1,
            end_line=10,
            content="def my_func(self): pass",
            file_path="/path/to/file.py"
        )
        assert defn.name == "my_func"
        assert defn.qualified_name == "MyClass.my_func"
        assert defn.code_type == "method"
        assert defn.start_line == 1
        assert defn.end_line == 10
        assert defn.content == "def my_func(self): pass"
        assert defn.file_path == "/path/to/file.py"


class TestASTFunctionExtractorLanguageDetection:
    """Tests for language detection."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return ASTFunctionExtractor()

    def test_detect_python(self, extractor):
        """Test Python detection."""
        assert extractor.detect_language("test.py") == "python"

    def test_detect_javascript(self, extractor):
        """Test JavaScript detection."""
        assert extractor.detect_language("test.js") == "javascript"
        assert extractor.detect_language("test.jsx") == "javascript"

    def test_detect_typescript(self, extractor):
        """Test TypeScript detection."""
        assert extractor.detect_language("test.ts") == "typescript"
        assert extractor.detect_language("test.tsx") == "typescript"

    def test_detect_java(self, extractor):
        """Test Java detection."""
        assert extractor.detect_language("Test.java") == "java"

    def test_detect_go(self, extractor):
        """Test Go detection."""
        assert extractor.detect_language("main.go") == "go"

    def test_detect_rust(self, extractor):
        """Test Rust detection."""
        assert extractor.detect_language("lib.rs") == "rust"

    def test_detect_c_cpp(self, extractor):
        """Test C/C++ detection."""
        assert extractor.detect_language("main.c") == "c"
        assert extractor.detect_language("header.h") == "c"
        assert extractor.detect_language("main.cpp") == "cpp"
        assert extractor.detect_language("main.cc") == "cpp"
        assert extractor.detect_language("main.cxx") == "cpp"
        assert extractor.detect_language("header.hpp") == "cpp"

    def test_detect_csharp(self, extractor):
        """Test C# detection."""
        assert extractor.detect_language("Program.cs") == "c_sharp"

    def test_detect_ruby(self, extractor):
        """Test Ruby detection."""
        assert extractor.detect_language("script.rb") == "ruby"

    def test_detect_php(self, extractor):
        """Test PHP detection."""
        assert extractor.detect_language("index.php") == "php"

    def test_detect_swift(self, extractor):
        """Test Swift detection."""
        assert extractor.detect_language("main.swift") == "swift"

    def test_detect_kotlin(self, extractor):
        """Test Kotlin detection."""
        assert extractor.detect_language("Main.kt") == "kotlin"
        assert extractor.detect_language("build.kts") == "kotlin"

    def test_detect_scala(self, extractor):
        """Test Scala detection."""
        assert extractor.detect_language("Main.scala") == "scala"

    def test_detect_unknown(self, extractor):
        """Test unknown extension returns None."""
        assert extractor.detect_language("file.unknown") is None
        assert extractor.detect_language("file.xyz") is None

    def test_detect_case_insensitive(self, extractor):
        """Test detection is case insensitive for extensions."""
        assert extractor.detect_language("TEST.PY") == "python"
        assert extractor.detect_language("Main.JAVA") == "java"


class TestASTFunctionExtractorSupport:
    """Tests for support checking methods."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return ASTFunctionExtractor()

    def test_supports_ast_extraction_python(self, extractor):
        """Test AST support for Python."""
        if TREE_SITTER_AVAILABLE:
            assert extractor.supports_ast_extraction("test.py") is True
        else:
            assert extractor.supports_ast_extraction("test.py") is False

    def test_supports_ast_extraction_unknown(self, extractor):
        """Test AST support for unknown language."""
        assert extractor.supports_ast_extraction("file.xyz") is False

    def test_get_supported_languages(self):
        """Test getting list of supported languages."""
        languages = ASTFunctionExtractor.get_supported_languages()
        assert isinstance(languages, list)
        assert len(languages) > 0
        # Check some expected languages
        assert "python" in languages
        assert "javascript" in languages
        assert "java" in languages

    def test_get_supported_extensions(self):
        """Test getting list of supported extensions."""
        extensions = ASTFunctionExtractor.get_supported_extensions()
        assert isinstance(extensions, list)
        assert len(extensions) > 0
        # Check some expected extensions
        assert ".py" in extensions
        assert ".js" in extensions
        assert ".java" in extensions


class TestASTFunctionExtractorBasicExtraction:
    """Tests for basic definition extraction."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return ASTFunctionExtractor()

    def test_extract_empty_code(self, extractor):
        """Test extraction from empty code returns empty list."""
        result = extractor.extract_definitions("test.py", "")
        assert result == []

    def test_extract_whitespace_only(self, extractor):
        """Test extraction from whitespace returns empty list."""
        result = extractor.extract_definitions("test.py", "   \n\t\n   ")
        assert result == []

    def test_extract_unsupported_extension(self, extractor):
        """Test extraction falls back to module for unsupported extensions."""
        code = "some content here"
        result = extractor.extract_definitions("file.xyz", code)

        assert len(result) == 1
        assert result[0].code_type == "module"
        assert result[0].name == "module"
        assert code in result[0].content

    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    def test_extract_python_function(self, extractor):
        """Test extraction of Python function."""
        code = '''def hello():
    print("Hello, World!")
'''
        result = extractor.extract_definitions("test.py", code)

        # Should have function definition
        functions = [d for d in result if d.code_type == "function"]
        assert len(functions) == 1
        assert functions[0].name == "hello"
        assert functions[0].qualified_name == "hello"
        assert functions[0].start_line == 1
        assert "def hello" in functions[0].content

    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    def test_extract_python_class(self, extractor):
        """Test extraction of Python class."""
        code = '''class MyClass:
    def __init__(self):
        pass

    def method(self):
        pass
'''
        result = extractor.extract_definitions("test.py", code)

        # Should have class definition
        classes = [d for d in result if d.code_type == "class"]
        assert len(classes) == 1
        assert classes[0].name == "MyClass"

    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    def test_extract_python_nested_methods(self, extractor):
        """Test extraction of nested methods in Python class."""
        code = '''class MyClass:
    def method_one(self):
        pass

    def method_two(self):
        pass
'''
        result = extractor.extract_definitions("test.py", code)

        # Should have class and methods
        classes = [d for d in result if d.code_type == "class"]
        functions = [d for d in result if d.code_type == "function"]

        assert len(classes) == 1
        # Methods are nested inside class, should have qualified names
        methods = [d for d in result if "MyClass." in d.qualified_name and d.code_type == "function"]
        assert len(methods) == 2

        method_names = [m.name for m in methods]
        assert "method_one" in method_names
        assert "method_two" in method_names

    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    def test_extract_python_decorated_function(self, extractor):
        """Test extraction of decorated Python function."""
        code = '''@decorator
def decorated_func():
    pass
'''
        result = extractor.extract_definitions("test.py", code)

        functions = [d for d in result if d.code_type == "function"]
        assert len(functions) == 1
        assert functions[0].name == "decorated_func"
        # Decorator should be included in content
        assert "@decorator" in functions[0].content

    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    def test_extract_python_module_level_code(self, extractor):
        """Test extraction includes module-level code."""
        code = '''import os
import sys

CONSTANT = 42

def my_function():
    pass
'''
        result = extractor.extract_definitions("test.py", code)

        # Should have function and module-level code
        modules = [d for d in result if d.code_type == "module"]
        functions = [d for d in result if d.code_type == "function"]

        assert len(functions) == 1
        assert len(modules) == 1
        # Module should contain imports and constant
        assert "import" in modules[0].content

    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    def test_extract_line_numbers_1_indexed(self, extractor):
        """Test that line numbers are 1-indexed."""
        code = '''# Line 1
# Line 2
def func():
    pass
# Line 5
'''
        result = extractor.extract_definitions("test.py", code)

        functions = [d for d in result if d.code_type == "function"]
        assert len(functions) == 1
        # Function starts at line 3
        assert functions[0].start_line == 3


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
class TestASTFunctionExtractorJavaScript:
    """Tests for JavaScript extraction."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return ASTFunctionExtractor()

    def test_extract_js_function_declaration(self, extractor):
        """Test extraction of JavaScript function declaration."""
        code = '''function greet(name) {
    console.log("Hello, " + name);
}
'''
        result = extractor.extract_definitions("test.js", code)

        functions = [d for d in result if d.code_type == "function"]
        assert len(functions) >= 1
        assert any(f.name == "greet" for f in functions)

    def test_extract_js_class(self, extractor):
        """Test extraction of JavaScript class."""
        code = '''class Person {
    constructor(name) {
        this.name = name;
    }

    greet() {
        return "Hello";
    }
}
'''
        result = extractor.extract_definitions("test.js", code)

        classes = [d for d in result if d.code_type == "class"]
        assert len(classes) >= 1


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
class TestASTFunctionExtractorTypeScript:
    """Tests for TypeScript extraction."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return ASTFunctionExtractor()

    def test_extract_ts_interface(self, extractor):
        """Test extraction of TypeScript interface."""
        code = '''interface User {
    id: number;
    name: string;
}
'''
        result = extractor.extract_definitions("test.ts", code)

        interfaces = [d for d in result if d.code_type == "interface"]
        assert len(interfaces) >= 1


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
class TestASTFunctionExtractorJava:
    """Tests for Java extraction."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return ASTFunctionExtractor()

    def test_extract_java_class_and_methods(self, extractor):
        """Test extraction of Java class and methods."""
        code = '''public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello");
    }

    public void greet() {
        System.out.println("Hi");
    }
}
'''
        result = extractor.extract_definitions("HelloWorld.java", code)

        classes = [d for d in result if d.code_type == "class"]
        methods = [d for d in result if d.code_type == "method"]

        assert len(classes) >= 1
        assert len(methods) >= 1


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
class TestASTFunctionExtractorGo:
    """Tests for Go extraction."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return ASTFunctionExtractor()

    def test_extract_go_function(self, extractor):
        """Test extraction of Go function."""
        code = '''package main

func main() {
    fmt.Println("Hello")
}

func greet(name string) string {
    return "Hello, " + name
}
'''
        result = extractor.extract_definitions("main.go", code)

        functions = [d for d in result if d.code_type == "function"]
        assert len(functions) >= 2


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
class TestASTFunctionExtractorRust:
    """Tests for Rust extraction."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return ASTFunctionExtractor()

    def test_extract_rust_function(self, extractor):
        """Test extraction of Rust function."""
        code = '''fn main() {
    println!("Hello");
}

fn greet(name: &str) -> String {
    format!("Hello, {}", name)
}
'''
        result = extractor.extract_definitions("main.rs", code)

        functions = [d for d in result if d.code_type == "function"]
        assert len(functions) >= 2

    def test_extract_rust_struct(self, extractor):
        """Test extraction of Rust struct."""
        code = '''struct Person {
    name: String,
    age: u32,
}
'''
        result = extractor.extract_definitions("lib.rs", code)

        classes = [d for d in result if d.code_type == "class"]
        assert len(classes) >= 1


class TestDefinitionTypes:
    """Tests for DEFINITION_TYPES constant."""

    def test_python_types(self):
        """Test Python definition types."""
        assert "python" in DEFINITION_TYPES
        py_types = DEFINITION_TYPES["python"]
        assert "function_definition" in py_types
        assert "class_definition" in py_types

    def test_javascript_types(self):
        """Test JavaScript definition types."""
        assert "javascript" in DEFINITION_TYPES
        js_types = DEFINITION_TYPES["javascript"]
        assert "function_declaration" in js_types
        assert "class_declaration" in js_types

    def test_java_types(self):
        """Test Java definition types."""
        assert "java" in DEFINITION_TYPES
        java_types = DEFINITION_TYPES["java"]
        assert "method_declaration" in java_types
        assert "class_declaration" in java_types

    def test_all_languages_have_types(self):
        """Test that all languages have at least one definition type."""
        for lang, types in DEFINITION_TYPES.items():
            assert len(types) > 0, f"Language {lang} has no definition types"


class TestASTFunctionExtractorEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return ASTFunctionExtractor()

    def test_extract_malformed_code(self, extractor):
        """Test extraction handles malformed code gracefully."""
        code = '''def broken(
    # Missing closing parenthesis
'''
        # Should not raise, should fall back to module
        result = extractor.extract_definitions("test.py", code)
        assert len(result) >= 1

    def test_extract_unicode_content(self, extractor):
        """Test extraction handles unicode content."""
        code = '''def greet():
    print("Hello, 世界!")
    print("Привет, мир!")
    print("🎉")
'''
        if TREE_SITTER_AVAILABLE:
            result = extractor.extract_definitions("test.py", code)
            functions = [d for d in result if d.code_type == "function"]
            assert len(functions) == 1
            assert "世界" in functions[0].content

    def test_extract_very_long_function(self, extractor):
        """Test extraction of very long function."""
        lines = ["def long_function():"]
        for i in range(1000):
            lines.append(f"    x_{i} = {i}")
        lines.append("    return x_999")
        code = "\n".join(lines)

        if TREE_SITTER_AVAILABLE:
            result = extractor.extract_definitions("test.py", code)
            functions = [d for d in result if d.code_type == "function"]
            assert len(functions) == 1
            assert functions[0].line_count > 1000

    def test_extract_deeply_nested(self, extractor):
        """Test extraction of deeply nested definitions."""
        code = '''class Outer:
    class Inner:
        def method(self):
            def nested():
                pass
'''
        if TREE_SITTER_AVAILABLE:
            result = extractor.extract_definitions("test.py", code)

            # Should have outer class, inner class, method, nested function
            assert len(result) >= 3

    def test_file_path_stored_correctly(self, extractor):
        """Test that file path is stored in definitions."""
        code = "def test(): pass"
        file_path = "/some/path/to/test.py"

        result = extractor.extract_definitions(file_path, code)

        for defn in result:
            assert defn.file_path == file_path

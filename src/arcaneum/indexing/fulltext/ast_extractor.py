"""AST-based function/class extractor for full-text indexing (RDR-011).

This module extracts discrete function/class definitions with line ranges
using tree-sitter directly. This is different from ast_chunker.py which
creates overlapping chunks for embeddings.

Key differences from ast_chunker.py:
- ast_chunker.py: Creates overlapping chunks for embeddings (Qdrant)
- ast_extractor.py: Extracts discrete definitions with line ranges (MeiliSearch)
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    from tree_sitter_language_pack import get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("tree-sitter-language-pack not available, AST extraction will be limited")

logger = logging.getLogger(__name__)


@dataclass
class CodeDefinition:
    """Function/class definition with location.

    Represents a discrete code unit (function, class, method, or module)
    extracted from source code with precise line ranges.
    """
    name: str
    qualified_name: str
    code_type: str  # "function", "class", "method", "module", "interface"
    start_line: int  # 1-indexed for user display
    end_line: int
    content: str
    file_path: str

    @property
    def line_count(self) -> int:
        """Number of lines in this definition."""
        return self.end_line - self.start_line + 1


# Language -> node types that represent definitions
# Maps tree-sitter node types to our code_type values
DEFINITION_TYPES: Dict[str, Dict[str, str]] = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
        "decorated_definition": "decorated",  # Special handling needed
    },
    "javascript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "arrow_function": "function",
        "generator_function_declaration": "function",
    },
    "typescript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "arrow_function": "function",
    },
    "java": {
        "method_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "class",
        "constructor_declaration": "method",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "class",  # structs, interfaces
    },
    "rust": {
        "function_item": "function",
        "impl_item": "class",
        "struct_item": "class",
        "trait_item": "interface",
        "enum_item": "class",
    },
    "c": {
        "function_definition": "function",
        "struct_specifier": "class",
    },
    "cpp": {
        "function_definition": "function",
        "class_specifier": "class",
        "struct_specifier": "class",
    },
    "c_sharp": {
        "method_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "struct_declaration": "class",
    },
    "ruby": {
        "method": "method",
        "class": "class",
        "module": "module",
    },
    "php": {
        "function_definition": "function",
        "method_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "trait_declaration": "class",
    },
    "swift": {
        "function_declaration": "function",
        "class_declaration": "class",
        "struct_declaration": "class",
        "protocol_declaration": "interface",
    },
    "kotlin": {
        "function_declaration": "function",
        "class_declaration": "class",
        "object_declaration": "class",
        "interface_declaration": "interface",
    },
    "scala": {
        "function_definition": "function",
        "class_definition": "class",
        "object_definition": "class",
        "trait_definition": "interface",
    },
}


class ASTFunctionExtractor:
    """Extract function/class definitions using tree-sitter directly.

    This extractor uses tree-sitter AST parsing to identify and extract
    discrete code definitions (functions, classes, methods) with precise
    line ranges for full-text indexing.

    Unlike ASTCodeChunker (ast_chunker.py) which creates overlapping chunks
    for embeddings, this extractor produces discrete, non-overlapping
    definitions suitable for MeiliSearch full-text search.
    """

    # Language mapping: file extension -> tree-sitter language name
    # Reused from ast_chunker.py for consistency
    LANGUAGE_MAP: Dict[str, str] = {
        # Primary languages (RDR-005)
        ".py": "python",
        ".java": "java",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".cs": "c_sharp",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".php": "php",
        ".rb": "ruby",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".scala": "scala",
        ".sc": "scala",
        ".swift": "swift",

        # Additional supported languages
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".r": "r",
        ".R": "r",
        ".lua": "lua",
        ".vim": "vim",
        ".el": "elisp",
        ".clj": "clojure",
        ".ex": "elixir",
        ".exs": "elixir",
        ".erl": "erlang",
        ".hrl": "erlang",
        ".hs": "haskell",
        ".ml": "ocaml",
        ".nim": "nim",
        ".pl": "perl",
        ".pm": "perl",
    }

    def __init__(self):
        """Initialize AST function extractor."""
        if not TREE_SITTER_AVAILABLE:
            logger.warning(
                "tree-sitter-language-pack not available. "
                "AST extraction will fall back to module-level for all files."
            )

    def extract_definitions(
        self,
        file_path: str,
        code: str
    ) -> List[CodeDefinition]:
        """Extract function/class definitions with line ranges.

        Uses tree-sitter directly (NOT LlamaIndex CodeSplitter) to parse
        the AST and extract discrete function/class definitions.

        Args:
            file_path: Path to source file (used for language detection)
            code: Source code content

        Returns:
            List of CodeDefinition objects. Fallback to module-level
            if AST parsing fails or language is unsupported.
        """
        if not code or not code.strip():
            return []

        file_ext = Path(file_path).suffix.lower()
        language = self.LANGUAGE_MAP.get(file_ext)

        if not language or language not in DEFINITION_TYPES:
            # Fallback: entire file as "module"
            logger.debug(f"No AST support for {file_ext}, using module-level")
            return [self._create_module_definition(file_path, code)]

        if not TREE_SITTER_AVAILABLE:
            logger.debug("tree-sitter not available, using module-level")
            return [self._create_module_definition(file_path, code)]

        try:
            parser = get_parser(language)
            tree = parser.parse(bytes(code, "utf8"))

            definitions: List[CodeDefinition] = []
            self._extract_from_node(
                tree.root_node,
                code,
                language,
                file_path,
                definitions,
                parent_name=None
            )

            # Add module-level code if any exists outside definitions
            module_code = self._extract_module_code(code, definitions, file_path)
            if module_code:
                definitions.append(module_code)

            if not definitions:
                # No definitions found, fall back to module-level
                return [self._create_module_definition(file_path, code)]

            return definitions

        except Exception as e:
            logger.warning(f"AST extraction failed for {file_path}: {e}")
            return [self._create_module_definition(file_path, code)]

    def _extract_from_node(
        self,
        node,
        code: str,
        language: str,
        file_path: str,
        definitions: List[CodeDefinition],
        parent_name: Optional[str]
    ):
        """Recursively extract definitions from AST nodes.

        Args:
            node: tree-sitter Node to process
            code: Full source code (for reference)
            language: Tree-sitter language name
            file_path: Path to source file
            definitions: List to append found definitions to
            parent_name: Name of parent definition (for qualified names)
        """
        def_types = DEFINITION_TYPES.get(language, {})

        # Handle Python decorated definitions specially
        if language == "python" and node.type == "decorated_definition":
            # Find the actual definition inside the decorator
            for child in node.children:
                if child.type in def_types:
                    self._process_definition_node(
                        child, node, code, language, file_path, definitions, parent_name, def_types
                    )
                    return
            # If no definition found, process children normally
            for child in node.children:
                self._extract_from_node(child, code, language, file_path, definitions, parent_name)
            return

        if node.type in def_types:
            self._process_definition_node(
                node, node, code, language, file_path, definitions, parent_name, def_types
            )
        else:
            # Not a definition node, but might contain definitions
            for child in node.children:
                self._extract_from_node(child, code, language, file_path, definitions, parent_name)

    def _process_definition_node(
        self,
        definition_node,
        source_node,
        code: str,
        language: str,
        file_path: str,
        definitions: List[CodeDefinition],
        parent_name: Optional[str],
        def_types: Dict[str, str]
    ):
        """Process a definition node and its nested definitions.

        Args:
            definition_node: The actual definition node (function_definition, etc.)
            source_node: The node to use for line ranges (may include decorators)
            code: Full source code
            language: Tree-sitter language name
            file_path: Path to source file
            definitions: List to append found definitions to
            parent_name: Name of parent definition (for qualified names)
            def_types: Definition types mapping for this language
        """
        # Get the name from the identifier child
        name = self._extract_name(definition_node)

        # Build qualified name (e.g., "MyClass.method")
        qualified = f"{parent_name}.{name}" if parent_name else name

        # Line numbers: tree-sitter uses 0-indexed, convert to 1-indexed
        # Use source_node for line ranges (includes decorators)
        start_line = source_node.start_point[0] + 1
        end_line = source_node.end_point[0] + 1

        code_type = def_types.get(definition_node.type, "unknown")

        definitions.append(CodeDefinition(
            name=name,
            qualified_name=qualified,
            code_type=code_type,
            start_line=start_line,
            end_line=end_line,
            content=source_node.text.decode("utf8"),
            file_path=file_path
        ))

        # Recurse into nested definitions (e.g., methods in classes)
        for child in definition_node.children:
            self._extract_from_node(
                child, code, language, file_path,
                definitions, parent_name=qualified
            )

    def _extract_name(self, node) -> str:
        """Extract the name from a definition node.

        Args:
            node: tree-sitter Node representing a definition

        Returns:
            Name of the definition, or "anonymous" if not found
        """
        # Try common field names for identifiers
        for field_name in ["name", "identifier"]:
            name_node = node.child_by_field_name(field_name)
            if name_node:
                return name_node.text.decode("utf8")

        # Fallback: look for identifier child directly
        for child in node.children:
            if child.type in ("identifier", "name"):
                return child.text.decode("utf8")

        return "anonymous"

    def _extract_module_code(
        self,
        code: str,
        definitions: List[CodeDefinition],
        file_path: str
    ) -> Optional[CodeDefinition]:
        """Extract module-level code not inside any definition.

        Creates a "module" definition for code that exists outside
        of functions/classes (imports, constants, module docstrings, etc.).

        Args:
            code: Full source code
            definitions: Already extracted definitions
            file_path: Path to source file

        Returns:
            CodeDefinition for module-level code, or None if none exists
        """
        if not definitions:
            return None

        # Get line ranges covered by definitions
        covered_lines: Set[int] = set()
        for defn in definitions:
            for line in range(defn.start_line, defn.end_line + 1):
                covered_lines.add(line)

        # Find uncovered lines
        lines = code.splitlines()
        total_lines = len(lines)

        uncovered_content: List[str] = []
        uncovered_start: Optional[int] = None
        uncovered_end: Optional[int] = None

        for line_num in range(1, total_lines + 1):
            if line_num not in covered_lines:
                line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                # Skip empty lines at start/end
                if line_content.strip() or uncovered_content:
                    uncovered_content.append(line_content)
                    if uncovered_start is None:
                        uncovered_start = line_num
                    uncovered_end = line_num

        # Remove trailing empty lines
        while uncovered_content and not uncovered_content[-1].strip():
            uncovered_content.pop()
            if uncovered_content:
                uncovered_end = uncovered_start + len(uncovered_content) - 1

        if not uncovered_content or not any(line.strip() for line in uncovered_content):
            return None

        module_content = "\n".join(uncovered_content)

        return CodeDefinition(
            name="module",
            qualified_name="module",
            code_type="module",
            start_line=uncovered_start or 1,
            end_line=uncovered_end or len(lines),
            content=module_content,
            file_path=file_path
        )

    def _create_module_definition(
        self,
        file_path: str,
        code: str
    ) -> CodeDefinition:
        """Create a module-level definition for the entire file.

        Used as fallback when AST parsing fails or for unsupported languages.

        Args:
            file_path: Path to source file
            code: Full source code

        Returns:
            CodeDefinition representing the entire file as a module
        """
        lines = code.splitlines()
        return CodeDefinition(
            name="module",
            qualified_name="module",
            code_type="module",
            start_line=1,
            end_line=len(lines) if lines else 1,
            content=code,
            file_path=file_path
        )

    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension.

        Args:
            file_path: Path to source file

        Returns:
            Language name for tree-sitter, or None if unknown
        """
        file_ext = Path(file_path).suffix.lower()
        return self.LANGUAGE_MAP.get(file_ext)

    def supports_ast_extraction(self, file_path: str) -> bool:
        """Check if AST extraction is supported for this file.

        Args:
            file_path: Path to source file

        Returns:
            True if AST extraction is available for this file type
        """
        if not TREE_SITTER_AVAILABLE:
            return False

        language = self.detect_language(file_path)
        return language is not None and language in DEFINITION_TYPES

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Get list of languages with definition extraction support.

        Returns:
            List of language names that support function/class extraction
        """
        return sorted(DEFINITION_TYPES.keys())

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions.

        Returns:
            List of file extensions that support AST extraction
        """
        # Only return extensions for languages with definition support
        supported = set()
        for ext, lang in cls.LANGUAGE_MAP.items():
            if lang in DEFINITION_TYPES:
                supported.add(ext)
        return sorted(supported)

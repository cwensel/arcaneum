"""Guard: markdown decisions must not key off ``Path.suffix`` (kata t88p).

``Path("session.md.zst").suffix`` is ``".zst"``, not ``".md"``. Any code that
compares a raw suffix against a markdown extension therefore silently
misclassifies compressed markdown - it does not raise, it just decides "not
markdown" and drops the file, or stores ``".zst"`` as chunk metadata.

That is precisely the failure this repo already shipped once: five copies of
the markdown extension set drifted apart and the narrower one silently dropped
tracked files. Documenting the convention is not enough to stop a sixth copy
appearing, so this test enforces it.

Deliberately narrow. Raw ``.suffix`` is correct almost everywhere - code and
PDF corpora have no compressed variants, so ``ast_chunker``, ``code_indexer``,
``ast_extractor`` and ``pdf_indexer`` all use it legitimately. Only markdown
carries a double extension, so only *markdown-coupled* suffix use is flagged.
A blanket ban would fire on ~20 correct call sites and be turned off within a
week.
"""

import re
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[4] / "src" / "arcaneum"

#: The module that defines the convention is allowed to implement it.
EXEMPT_FILES = {"text_source.py"}

#: Markdown extensions written as literals. A raw-suffix comparison against any
#: of these is the bug this guard exists to catch.
_MD_LITERAL = r'"\.(?:md|markdown|mdown|mkd|mkdn)"'

#: Names that hold a markdown extension *set*; membership tests against them
#: have the same defect as comparing to a literal.
_MD_SET_NAMES = r"(?:MARKDOWN_EXTENSIONS|PLAIN_MARKDOWN_EXTENSIONS|md_extensions)"

PATTERNS = [
    # `path.suffix == ".md"` / `!=` / `in (".md", ...)`, either operand order.
    re.compile(rf"\.suffix(?:\.lower\(\))?\s*(?:==|!=|\bin\b)\s*[\(\[{{]?[^\n]*{_MD_LITERAL}"),
    re.compile(rf"{_MD_LITERAL}[^\n]*\s*(?:==|!=)\s*[^\n]*\.suffix"),
    # `path.suffix in MARKDOWN_EXTENSIONS` — set membership, same defect.
    re.compile(rf"\.suffix(?:\.lower\(\))?\s*(?:\bin\b|==)\s*{_MD_SET_NAMES}"),
]

FIX_HINT = (
    "Use logical_suffix(path) from arcaneum.indexing.common.text_source, which "
    "returns '.md' for 'a.md.zst'. For allowlist membership use "
    "extension_allowed(path, allowed) (arcaneum.cli.utils) or _corpus_extension(path) "
    "(arcaneum.cli.sync). Raw .suffix stays correct for code/PDF paths, which have "
    "no compressed variants."
)


def _iter_source_files():
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path.name in EXEMPT_FILES:
            continue
        yield path


def _strip_comments(line: str) -> str:
    """Drop trailing comments so prose about the bug does not trip the guard."""
    in_single = in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


def _find_violations():
    violations = []
    for path in _iter_source_files():
        for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = _strip_comments(raw_line)
            if ".suffix" not in line:
                continue
            if any(pattern.search(line) for pattern in PATTERNS):
                rel = path.relative_to(SRC_ROOT.parents[1])
                violations.append(f"{rel}:{lineno}: {raw_line.strip()}")
    return violations


class TestNoRawSuffixOnMarkdownPaths:
    def test_source_tree_is_clean(self):
        """No markdown decision may key off Path.suffix."""
        violations = _find_violations()
        assert not violations, (
            "Raw Path.suffix used for a markdown decision - compressed markdown "
            "(.md.zst) reports '.zst' and would be silently misclassified:\n\n"
            + "\n".join(f"  {v}" for v in violations)
            + f"\n\n{FIX_HINT}"
        )


class TestGuardActuallyDetects:
    """The guard is only worth having if it fires. Pin its detection."""

    def test_detects_equality_against_literal(self):
        line = 'if path.suffix == ".md":'
        assert any(p.search(line) for p in PATTERNS)

    def test_detects_lowered_equality(self):
        line = 'if path.suffix.lower() == ".markdown":'
        assert any(p.search(line) for p in PATTERNS)

    def test_detects_membership_in_literal_tuple(self):
        line = 'if f.suffix.lower() in (".md", ".markdown"):'
        assert any(p.search(line) for p in PATTERNS)

    def test_detects_membership_in_named_set(self):
        line = "md_files = [f for f in files if f.suffix in MARKDOWN_EXTENSIONS]"
        assert any(p.search(line) for p in PATTERNS)

    def test_detects_reversed_operand_order(self):
        line = 'if ".md" == path.suffix:'
        assert any(p.search(line) for p in PATTERNS)

    def test_ignores_code_extensions(self):
        """Code corpora have no compressed variants; raw suffix is correct."""
        line = 'language = ext_to_lang.get(file_path.suffix.lower(), "unknown")'
        assert not any(p.search(line) for p in PATTERNS)

    def test_ignores_pdf(self):
        line = 'pdf_files = [f for f in file_list if f.suffix.lower() == ".pdf"]'
        assert not any(p.search(line) for p in PATTERNS)

    def test_ignores_write_path_suffix_use(self):
        """injection.py picks a filename for a file arc creates, never reads."""
        line = "suffix = base_path.suffix"
        assert not any(p.search(line) for p in PATTERNS)

    def test_ignores_commented_prose(self):
        line = '    # Path.suffix reports ".zst" for a ".md" file, hence this helper'
        assert not any(p.search(_strip_comments(line)) for p in PATTERNS)

    def test_scans_a_nonempty_tree(self):
        """A guard that silently scans nothing passes forever."""
        assert len(list(_iter_source_files())) > 20

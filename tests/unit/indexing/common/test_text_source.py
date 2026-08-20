"""Unit tests for codec-aware text reading (kata t88p).

Covers the shared helpers that let a ``.md.zst`` file behave exactly like its
uncompressed ``.md`` twin:

- ``logical_suffix()`` sees through the compression suffix
- ``read_text_source()`` transparently decompresses
- the markdown extension set is a single shared constant
"""

from pathlib import Path

import pytest

from arcaneum.indexing.common.text_source import (
    COMPRESSION_SUFFIXES,
    MARKDOWN_EXTENSIONS,
    is_compressed,
    logical_suffix,
    read_text_source,
)

SAMPLE = "---\ntitle: Doc\ntags: [a, b]\n---\n\n# Heading\n\nBody text.\n"


def _write_zst(path: Path, text: str) -> Path:
    import zstandard

    path.write_bytes(zstandard.ZstdCompressor(level=19).compress(text.encode("utf-8")))
    return path


class TestLogicalSuffix:
    def test_plain_markdown_unchanged(self):
        assert logical_suffix(Path("a.md")) == ".md"
        assert logical_suffix(Path("a.markdown")) == ".markdown"

    def test_sees_through_zst(self):
        assert logical_suffix(Path("a.md.zst")) == ".md"
        assert logical_suffix(Path("a.markdown.zst")) == ".markdown"

    def test_case_insensitive(self):
        assert logical_suffix(Path("A.MD.ZST")) == ".md"

    def test_bare_compressed_file_keeps_its_suffix(self):
        # Nothing logical underneath — do not invent one.
        assert logical_suffix(Path("archive.zst")) == ".zst"

    def test_no_suffix(self):
        assert logical_suffix(Path("README")) == ""

    def test_unrelated_double_extension_untouched(self):
        assert logical_suffix(Path("a.tar.gz")) == ".gz"


class TestIsCompressed:
    def test_detects_zst(self):
        assert is_compressed(Path("a.md.zst")) is True

    def test_plain_is_not(self):
        assert is_compressed(Path("a.md")) is False

    def test_zst_is_a_known_suffix(self):
        assert ".zst" in COMPRESSION_SUFFIXES


class TestMarkdownExtensions:
    def test_includes_plain_and_compressed_forms(self):
        assert ".md" in MARKDOWN_EXTENSIONS
        assert ".markdown" in MARKDOWN_EXTENSIONS
        assert ".md.zst" in MARKDOWN_EXTENSIONS
        assert ".markdown.zst" in MARKDOWN_EXTENSIONS

    def test_every_plain_extension_has_a_compressed_twin(self):
        plain = {e for e in MARKDOWN_EXTENSIONS if not e.endswith(".zst")}
        for ext in plain:
            assert f"{ext}.zst" in MARKDOWN_EXTENSIONS


class TestReadTextSource:
    def test_reads_plain_file(self, tmp_path):
        p = tmp_path / "a.md"
        p.write_text(SAMPLE, encoding="utf-8")
        assert read_text_source(p) == SAMPLE

    def test_reads_zst_file(self, tmp_path):
        p = _write_zst(tmp_path / "a.md.zst", SAMPLE)
        assert read_text_source(p) == SAMPLE

    def test_compressed_and_plain_twins_are_identical(self, tmp_path):
        plain = tmp_path / "a.md"
        plain.write_text(SAMPLE, encoding="utf-8")
        comp = _write_zst(tmp_path / "b.md.zst", SAMPLE)
        assert read_text_source(plain) == read_text_source(comp)

    def test_latin1_fallback_on_undecodable_bytes(self, tmp_path):
        p = tmp_path / "a.md"
        p.write_bytes(b"caf\xe9 latin-1 only\n")
        assert "caf" in read_text_source(p)

    def test_errors_replace_is_honored(self, tmp_path):
        p = tmp_path / "a.md"
        p.write_bytes(b"caf\xe9\n")
        assert read_text_source(p, errors="replace") is not None

    def test_compressed_latin1_fallback(self, tmp_path):
        import zstandard

        p = tmp_path / "a.md.zst"
        p.write_bytes(zstandard.ZstdCompressor().compress(b"caf\xe9 latin-1\n"))
        assert "caf" in read_text_source(p)

    def test_corrupt_zst_raises_rather_than_indexing_garbage(self, tmp_path):
        """Never fall back to raw bytes - that would index compressed noise."""
        import zstandard

        p = tmp_path / "a.md.zst"
        p.write_bytes(b"not a zstd frame at all")
        with pytest.raises(zstandard.ZstdError):
            read_text_source(p)

    def test_corrupt_zst_error_names_the_file(self, tmp_path):
        """Sync catches per-file, so the message is the user's only signal."""
        import zstandard

        p = tmp_path / "broken.md.zst"
        p.write_bytes(b"not a zstd frame at all")
        with pytest.raises(zstandard.ZstdError) as excinfo:
            read_text_source(p)
        assert "broken.md.zst" in str(excinfo.value)

    @pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75, 0.9, 0.99])
    def test_truncated_zst_raises_never_returns_partial(self, tmp_path, fraction):
        """A half-written archive must fail loudly, not yield partial content.

        stream_reader treats a truncated frame as a clean EOF, which would
        index a quietly incomplete document: no error, plausible chunk count,
        missing text. Regression guard for that silent-corruption path.
        """
        import random
        import string

        import zstandard

        random.seed(88)
        body = "\n".join(
            "".join(random.choices(string.ascii_letters + " ", k=70)) for _ in range(3000)
        )
        text = SAMPLE + body
        full = zstandard.ZstdCompressor(level=19).compress(text.encode("utf-8"))

        p = tmp_path / "truncated.md.zst"
        p.write_bytes(full[: int(len(full) * fraction)])
        with pytest.raises(zstandard.ZstdError):
            read_text_source(p)

    def test_intact_archive_round_trips_exactly(self, tmp_path):
        """The truncation guard must not reject valid archives."""
        import random
        import string

        import zstandard

        random.seed(88)
        body = "\n".join(
            "".join(random.choices(string.ascii_letters + " ", k=70)) for _ in range(3000)
        )
        text = SAMPLE + body
        p = tmp_path / "intact.md.zst"
        p.write_bytes(zstandard.ZstdCompressor(level=19).compress(text.encode("utf-8")))
        assert read_text_source(p) == text

    def test_empty_zst_file_raises(self, tmp_path):
        import zstandard

        p = tmp_path / "empty.md.zst"
        p.write_bytes(b"")
        with pytest.raises(zstandard.ZstdError):
            read_text_source(p)

    def test_large_content_round_trips(self, tmp_path):
        big = SAMPLE + ("line of transcript text\n" * 20000)
        p = _write_zst(tmp_path / "big.md.zst", big)
        assert read_text_source(p) == big

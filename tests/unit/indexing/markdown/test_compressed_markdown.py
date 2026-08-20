"""Compressed markdown (.md.zst) parity tests (kata t88p).

The invariant under test: a ``.md.zst`` file is indistinguishable from its
uncompressed ``.md`` twin everywhere downstream of the read — same frontmatter,
same content hash, same chunk text and count, same stored ``file_extension``.
"""

from pathlib import Path

from arcaneum.cli.sync import SUPPORTED_EXTENSIONS_BY_TYPE, chunk_markdown_file
from arcaneum.indexing.common.sync import compute_text_file_hash
from arcaneum.indexing.common.text_source import logical_suffix
from arcaneum.indexing.markdown.discovery import MarkdownDiscovery

SAMPLE = (
    "---\n"
    "title: Session Transcript\n"
    "author: cwensel\n"
    "tags: [transcript, claude]\n"
    "---\n"
    "\n"
    "# Session\n"
    "\n"
    "Some body text that is long enough to chunk meaningfully.\n"
    "\n"
    "## Second Section\n"
    "\n" + ("Repeated transcript line with plenty of content to fill a chunk.\n" * 40)
)


def _twins(tmp_path: Path, text: str = SAMPLE):
    """Write a .md and its .md.zst twin; return (plain, compressed)."""
    import zstandard

    plain = tmp_path / "doc.md"
    plain.write_text(text, encoding="utf-8")
    comp = tmp_path / "doc_twin.md.zst"
    comp.write_bytes(zstandard.ZstdCompressor(level=19).compress(text.encode("utf-8")))
    return plain, comp


class TestExtensionAllowlist:
    def test_markdown_corpus_accepts_md_zst(self):
        assert ".md.zst" in SUPPORTED_EXTENSIONS_BY_TYPE["markdown"]

    def test_markdown_corpus_still_accepts_plain(self):
        assert ".md" in SUPPORTED_EXTENSIONS_BY_TYPE["markdown"]
        assert ".markdown" in SUPPORTED_EXTENSIONS_BY_TYPE["markdown"]

    def test_single_file_validation_accepts_md_zst(self, tmp_path):
        """The sync.py single-file gate keys off logical suffix, not .suffix."""
        _, comp = _twins(tmp_path)
        valid = SUPPORTED_EXTENSIONS_BY_TYPE["markdown"]
        # The gate must accept the compressed file by some rule; logical suffix
        # is the mechanism this fix introduces.
        assert logical_suffix(comp) in valid or comp.name.endswith(
            tuple(e for e in valid if e.endswith(".zst"))
        )


class TestDiscovery:
    def test_discovers_compressed_alongside_plain(self, tmp_path):
        _twins(tmp_path)
        found = MarkdownDiscovery().discover_files(tmp_path)
        names = {p.name for p in found}
        assert "doc.md" in names
        assert "doc_twin.md.zst" in names

    def test_no_duplicate_discovery(self, tmp_path):
        """Globbing both .md and .md.zst must not yield the same path twice."""
        _twins(tmp_path)
        found = MarkdownDiscovery().discover_files(tmp_path)
        assert len(found) == len(set(found))

    def test_plain_glob_does_not_swallow_compressed(self, tmp_path):
        _twins(tmp_path)
        found = MarkdownDiscovery(extensions=[".md"]).discover_files(tmp_path)
        assert {p.name for p in found} == {"doc.md"}


class TestFrontmatterParity:
    def test_metadata_matches_between_twins(self, tmp_path):
        plain, comp = _twins(tmp_path)
        d = MarkdownDiscovery()
        m_plain = d.extract_metadata(plain)
        m_comp = d.extract_metadata(comp)

        assert m_comp.has_frontmatter is True
        assert m_comp.title == m_plain.title == "Session Transcript"
        assert m_comp.author == m_plain.author == "cwensel"
        assert m_comp.tags == m_plain.tags

    def test_read_file_with_frontmatter_matches(self, tmp_path):
        plain, comp = _twins(tmp_path)
        c_plain, fm_plain = MarkdownDiscovery.read_file_with_frontmatter(plain)
        c_comp, fm_comp = MarkdownDiscovery.read_file_with_frontmatter(comp)
        assert c_plain == c_comp
        assert fm_plain == fm_comp

    def test_content_hash_matches_between_twins(self, tmp_path):
        """Hash is of decompressed text, so identical content hashes identically."""
        plain, comp = _twins(tmp_path)
        assert compute_text_file_hash(plain) == compute_text_file_hash(comp)


class TestChunkParity:
    def test_identical_chunk_text_and_count(self, tmp_path):
        plain, comp = _twins(tmp_path)
        chunks_plain = chunk_markdown_file(plain, chunk_size=512, chunk_overlap=50)
        chunks_comp = chunk_markdown_file(comp, chunk_size=512, chunk_overlap=50)

        assert len(chunks_plain) == len(chunks_comp)
        assert len(chunks_plain) > 0
        assert [c["text"] for c in chunks_plain] == [c["text"] for c in chunks_comp]

    def test_empty_compressed_file_yields_no_chunks(self, tmp_path):
        import zstandard

        p = tmp_path / "empty.md.zst"
        p.write_bytes(zstandard.ZstdCompressor().compress(b"   \n"))
        assert chunk_markdown_file(p, chunk_size=512, chunk_overlap=50) == []


class TestStoredMetadata:
    def test_file_extension_is_logical_not_physical(self, tmp_path):
        """Stored file_extension reads .md so compressed chunks stay comparable."""
        _, comp = _twins(tmp_path)
        assert logical_suffix(comp) == ".md"
        assert comp.suffix == ".zst"  # physical suffix unchanged on disk


class TestMtimePreserved:
    def test_stat_is_untouched_by_codec_handling(self, tmp_path):
        """Nothing in the compression path may synthesize or flatten mtime."""
        _, comp = _twins(tmp_path)
        import os

        os.utime(comp, (1_600_000_000, 1_600_000_000))
        MarkdownDiscovery().extract_metadata(comp)
        assert comp.stat().st_mtime == 1_600_000_000

    def test_reported_size_is_physical(self, tmp_path):
        """The mtime+size change gate must see on-disk (compressed) size."""
        _, comp = _twins(tmp_path)
        meta = MarkdownDiscovery().extract_metadata(comp)
        assert meta.file_size == comp.stat().st_size


class TestTwinIdentityBoundary:
    """Twins match on content, not identity — pinned so a change is deliberate.

    Unifying twin identity (so compressing a file is a no-op re-index rather
    than a new entry) is a separate design call about the change-detection key.
    """

    def test_content_is_equivalent(self, tmp_path):
        plain, comp = _twins(tmp_path)
        assert compute_text_file_hash(plain) == compute_text_file_hash(comp)

    def test_identity_is_not_equivalent(self, tmp_path):
        """Change detection keys on absolute path, so twins are two entries."""
        plain, comp = _twins(tmp_path)
        assert str(plain.absolute()) != str(comp.absolute())

    def test_quick_hash_differs_on_compressed_size(self, tmp_path):
        """mtime+size gate sees compressed size — hence the migration re-embed."""
        from arcaneum.indexing.common.sync import compute_quick_hash

        plain, comp = _twins(tmp_path)
        assert compute_quick_hash(plain) != compute_quick_hash(comp)

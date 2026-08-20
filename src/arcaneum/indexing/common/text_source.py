"""Codec-aware text reading and logical extensions (kata t88p).

Markdown archives can be stored zstd-compressed (``session.md.zst``) so large
transcript corpora index without keeping an uncompressed copy on disk. A
compressed file must be indistinguishable from its plain twin everywhere
downstream of the read: same frontmatter, same content hash, same chunk text.

Two things make that work, and both live here so they cannot drift apart:

``logical_suffix(path)``
    ``Path("a.md.zst").suffix`` is ``".zst"``, which is useless for deciding
    corpus type, chunker, or stored metadata. ``logical_suffix`` peels the
    compression suffix and returns ``".md"``.

``read_text_source(path)``
    Reads text, transparently decompressing when the file carries a known
    compression suffix. Everything downstream only ever sees a ``str``.

The markdown extension set also lives here. It was previously duplicated across
five call sites and had already drifted once (the sync allowlist carried a
narrower set than discovery, silently dropping tracked files); a single
constant is what keeps ``.md.zst`` from re-introducing that class of bug.

Note on size and mtime: only *content* is decompressed. ``stat()`` is never
proxied or synthesized, so the incremental-sync change gate keeps seeing the
real on-disk (compressed) size and the real mtime. That is deliberate —
compressed size is the correct change signal, and mtime carries ordering
signal for ``--order newest`` and resume-phase classification.

Scope note: these helpers make a compressed file equivalent to its plain twin
in *content* (same hash, same chunks, same stored extension). They do not make
it the same *file*. Change detection keys on absolute path, so ``doc.md`` and
``doc.md.zst`` remain two tracked entries: syncing a directory holding both
indexes the content twice, and replacing a ``.md`` with its ``.md.zst`` leaves
the original's chunks orphaned until a ``--parity`` sweep reaps them.
Unifying twin identity is deliberately out of scope here.
"""

import logging
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)

#: Compression suffixes recognized on top of a logical extension.
COMPRESSION_SUFFIXES: Set[str] = {".zst"}

#: Uncompressed markdown extensions.
PLAIN_MARKDOWN_EXTENSIONS: Set[str] = {".md", ".markdown", ".mdown", ".mkd", ".mkdn"}

#: Every markdown extension, plain and compressed. Single source of truth for
#: discovery defaults, corpus-type validation, and file-list filtering.
MARKDOWN_EXTENSIONS: Set[str] = PLAIN_MARKDOWN_EXTENSIONS | {
    f"{ext}{codec}" for ext in PLAIN_MARKDOWN_EXTENSIONS for codec in COMPRESSION_SUFFIXES
}


def is_compressed(path: Path) -> bool:
    """True when the path carries a recognized compression suffix."""
    return path.suffix.lower() in COMPRESSION_SUFFIXES


def logical_suffix(path: Path) -> str:
    """Return the content-bearing extension, seeing through compression.

    ``a.md.zst`` -> ``.md``; ``a.md`` -> ``.md``. A bare ``archive.zst`` has no
    logical extension underneath, so its own suffix is returned unchanged
    rather than inventing one.

    Args:
        path: Path to inspect. Need not exist.

    Returns:
        Lowercased extension including the leading dot, or ``""`` when the
        path has no suffix at all.
    """
    suffix = path.suffix.lower()
    if suffix not in COMPRESSION_SUFFIXES:
        return suffix

    inner = Path(path.stem).suffix.lower()
    # `archive.zst` — nothing underneath; keep the physical suffix.
    return inner or suffix


def logical_name(path: Path) -> str:
    """Return the filename with any compression suffix removed.

    ``a.md.zst`` -> ``a.md``. Used where a display or logical filename should
    not leak the storage codec.
    """
    return path.stem if is_compressed(path) else path.name


def read_text_source(
    path: Path,
    encoding: str = "utf-8",
    errors: Optional[str] = None,
) -> str:
    """Read a text file, transparently decompressing compressed sources.

    Mirrors ``Path.read_text`` semantics, including the latin-1 fallback used
    across the indexing paths: when ``errors`` is None and the bytes are not
    valid UTF-8, the read is retried as latin-1 rather than raising.

    Args:
        path: File to read. May be plain or ``.zst``-compressed.
        encoding: Primary text encoding.
        errors: Decoding error policy. When given (e.g. ``"replace"``) it is
            applied directly and no latin-1 fallback is attempted.

    Returns:
        Decoded text content.

    Raises:
        zstandard.ZstdError: The file carries a ``.zst`` suffix but is not a
            valid zstd frame. Surfaced rather than swallowed so a truncated or
            corrupt archive fails loudly instead of indexing as garbage.
    """
    raw = _read_bytes(path)

    if errors is not None:
        return raw.decode(encoding, errors=errors)

    try:
        return raw.decode(encoding)
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decode failed for {path}, trying latin-1")
        return raw.decode("latin-1")


def _read_bytes(path: Path) -> bytes:
    """Return file bytes, decompressing when the suffix says to."""
    if not is_compressed(path):
        return path.read_bytes()

    import zstandard

    try:
        # decompressobj (not stream_reader) because it exposes `.eof`, which is
        # true only once a frame has properly terminated.  stream_reader treats
        # a truncated frame as a clean end-of-file: a half-written .md.zst
        # would decompress to whatever bytes it happens to contain and index as
        # a successful but quietly incomplete document - no error, plausible
        # chunk count, missing content.  Checking `.eof` turns that silent
        # corruption into a loud per-file failure.
        #
        # Reads in chunks so a large archive never needs the whole compressed
        # image and its expansion resident at once.
        dctx = zstandard.ZstdDecompressor()
        dobj = dctx.decompressobj()
        out = bytearray()
        with path.open("rb") as fh:
            while chunk := fh.read(262144):
                out += dobj.decompress(chunk)
        if not dobj.eof:
            raise zstandard.ZstdError("archive ends mid-frame (truncated, or still being written)")
        return bytes(out)
    except zstandard.ZstdError as e:
        # Name the file and the cause. Callers catch this per-file and keep
        # going, so the message is the only signal the user gets about which
        # archive is bad. Never fall back to reading the raw bytes as text -
        # that would index compressed garbage as if it were content.
        raise zstandard.ZstdError(
            f"{path}: not a readable zstd archive ({e}). "
            "Re-create it from the source markdown, or remove it from the sync path."
        ) from e

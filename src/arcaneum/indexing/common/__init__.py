"""Common indexing utilities (RDR-004)."""

from .sync import MetadataBasedSync, compute_file_hash, compute_text_file_hash
from .text_source import (
    COMPRESSION_SUFFIXES,
    MARKDOWN_EXTENSIONS,
    PLAIN_MARKDOWN_EXTENSIONS,
    is_compressed,
    logical_name,
    logical_suffix,
    read_text_source,
)

__all__ = [
    "MetadataBasedSync",
    "compute_file_hash",
    "compute_text_file_hash",
    "COMPRESSION_SUFFIXES",
    "MARKDOWN_EXTENSIONS",
    "PLAIN_MARKDOWN_EXTENSIONS",
    "is_compressed",
    "logical_name",
    "logical_suffix",
    "read_text_source",
]

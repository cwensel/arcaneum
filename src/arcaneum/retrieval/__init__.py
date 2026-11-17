"""Print-free retrieval core shared by the CLI and the MCP server (RDR-026).

Each operation returns a result dataclass whose ``to_data()`` is the JSON
``data`` both front ends emit, and fails with an ``ArcaneumError`` carrying a
stable ``code`` (``arcaneum.cli.errors``). The core writes nothing to stdout,
starts no container, persists nothing from a read, and downloads an embedding
model only when the caller allows it. It uses neither rich nor click.
"""

from .corpora import CorpusListing, list_corpora
from .passage import PassageResult, get_passage
from .semantic import SemanticSearchResult, search_semantic
from .text import TextSearchResult, search_text

__all__ = [
    "CorpusListing",
    "PassageResult",
    "SemanticSearchResult",
    "TextSearchResult",
    "get_passage",
    "list_corpora",
    "search_semantic",
    "search_text",
]

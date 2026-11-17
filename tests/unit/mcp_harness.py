"""Run `arc` against fake Qdrant and MeiliSearch backends for the MCP server tests.

usage: python mcp_harness.py <scenario> <arc arguments...>

The tests drive `arc mcp` (and `arc corpus list --json`, `arc search text
--json`, `arc search semantic --json` and `arc search passage --json`, for the
CLI-parity checks) as a real child process over pipes. A scenario swaps the
client factories the retrieval core builds its clients from, and the query
embedder's model loader, so no service runs and no model loads:

- ``fake``: three corpora, both services up; each index holds ``DOCUMENTS``
  chunks of ``CONTENT_CHARS`` characters, each in a file of its own, and each
  corpus records the source root ``/src/<corpus>``. Each corpus also holds
  ``book.md``, ``BOOK_CHUNKS`` overlapping chunks of one file under UUID ids,
  which only a ``file_path`` filter or a passage lookup reaches
- ``many``: 200 corpora, for the output cap
- ``noisy``: ``fake``, but the listing prints to stdout through every channel a
  library might use (print, a raw fd 1 write, logging, warnings, rich)
- ``stall``: the listing and the text search block, for the per-call timeout
- ``uncached``: ``fake``, but no query model is in the local cache
- ``embed-stall``: ``fake``, but loading a query model blocks
- ``qdrant-down`` / ``meili-down``: that service refuses connections
- ``real``: no fakes (start-up checks that never call a tool)

Every fake Qdrant or MeiliSearch write and every fake model load prints a line
to stderr (``fake qdrant write: …``, ``fake meili write: …``, ``fake model
load: …``) for the tests to count.
Collections carry no prompt policy, so a semantic search may stamp one.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
import warnings
from types import SimpleNamespace

CORPORA = {
    "Alpha": {"collection_type": "pdf", "model": "arctic-m", "description": "Alpha docs"},
    "Beta": {"collection_type": "markdown", "model": "arctic-m", "description": "Beta notes"},
    "Gamma": {"collection_type": "code", "model": "jina-code", "description": "Gamma code"},
}

DOCUMENTS = 30
CONTENT_CHARS = 2000

METADATA_POINT_ID = "00000000-0000-0000-0000-000000000001"
BOOK_CHUNKS = 6
BOOK_STEP, BOOK_SIZE = 150, 200  # words; each chunk overlaps the next by 50
BOOK_WORDS = [f"w{n:04d}" for n in range(BOOK_STEP * (BOOK_CHUNKS - 1) + BOOK_SIZE)]


def book_path(corpus: str) -> str:
    return f"/src/{corpus}/book.md"


def book_chunk_id(corpus: str, i: int) -> str:
    return f"00000000-0000-4000-8{list(CORPORA).index(corpus):03d}-{i:012d}"


def book_text(first: int, last: int) -> str:
    """The stitched text of book chunks ``first`` through ``last``."""
    return " ".join(BOOK_WORDS[first * BOOK_STEP : last * BOOK_STEP + BOOK_SIZE])


def _book(corpus: str) -> list[dict]:
    """The chunks of one multi-chunk file, as MeiliSearch documents."""
    if corpus not in CORPORA:
        return []
    return [
        {
            "id": book_chunk_id(corpus, i),
            "file_path": book_path(corpus),
            "chunk_index": i,
            "section": "Book",
            "content": book_text(i, i),
        }
        for i in range(BOOK_CHUNKS)
    ]


def _book_point(document: dict) -> SimpleNamespace:
    """A book chunk as Qdrant stores it: its text under ``text``."""
    payload = {key: value for key, value in document.items() if key not in ("id", "content")}
    payload["text"] = document["content"]
    return SimpleNamespace(id=document["id"], payload=payload)


def _matches(payload: dict, conditions) -> bool:
    for condition in conditions:
        value = payload.get(condition.key)
        if getattr(condition, "match", None) is not None and value != condition.match.value:
            return False
        bounds = getattr(condition, "range", None)
        if bounds is not None:
            if not isinstance(value, int):
                return False
            if (bounds.gte is not None and value < bounds.gte) or (
                bounds.lte is not None and value > bounds.lte
            ):
                return False
    return True


def _documents(corpus: str) -> list[dict]:
    """The chunks a fake index holds, best match first."""
    documents = []
    for i in range(DOCUMENTS):
        content = f"{corpus} chunk {i} " + "lorem " * (CONTENT_CHARS // 6)
        documents.append(
            {
                "id": f"{corpus}-{i}",
                "file_path": f"/src/{corpus}/doc{i}.md",
                "section": f"Part {i}",
                "language": "python" if i % 2 else "markdown",
                "content": content[:CONTENT_CHARS],
                "_formatted": {"content": content[:CONTENT_CHARS]},
                "_rankingScore": round(1 - i / 100, 2),
            }
        )
    return documents


def _report(line: str) -> None:
    os.write(2, (line + "\n").encode())


def semantic_score(corpus_rank: int, i: int) -> float:
    """A fake similarity: best chunk first, corpora interleaved by score."""
    return round(0.95 - i / 40 - corpus_rank / 200, 4)


class FakeQdrant:
    def __init__(self, metadata, down=False):
        self.metadata = metadata
        self.down = down

    def collection_exists(self, name):
        if self.down:
            raise ConnectionError("[Errno 61] Connection refused")
        return name in self.metadata

    def query_points(
        self, collection_name, query_filter=None, limit=10, offset=0, score_threshold=None, **_kw
    ):
        rank = list(self.metadata).index(collection_name)
        conditions = (query_filter.must or []) if query_filter is not None else []
        documents = _documents(collection_name)
        if conditions:
            documents += _book(collection_name)
        points = []
        for i, document in enumerate(documents):
            payload = {
                key: value
                for key, value in document.items()
                if key not in ("id", "_formatted", "_rankingScore")
            }
            if any(payload.get(c.key) != c.match.value for c in conditions):
                continue
            score = semantic_score(rank, i)
            if score_threshold is not None and score < score_threshold:
                continue
            points.append(SimpleNamespace(id=document["id"], score=score, payload=payload))
        return SimpleNamespace(points=points[offset : offset + limit])

    def upsert(self, collection_name, **_kwargs):
        _report(f"fake qdrant write: upsert {collection_name}")

    def create_payload_index(self, collection_name, **_kwargs):
        _report(f"fake qdrant write: create_payload_index {collection_name}")

    def get_collections(self):
        if self.down:
            raise ConnectionError("[Errno 61] Connection refused")
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.metadata])

    def get_collection(self, _name):
        return SimpleNamespace(
            points_count=4,
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=768, distance="Cosine"))
            ),
        )

    def facet(self, collection_name, key, **_kwargs):
        return SimpleNamespace(
            hits=[SimpleNamespace(value=f"/{collection_name}/{i}", count=1) for i in range(2)]
        )

    def scroll(self, collection_name=None, scroll_filter=None, **_kwargs):
        if self.down:
            raise ConnectionError("[Errno 61] Connection refused")
        conditions = (scroll_filter.must or []) if scroll_filter is not None else []
        if not any(condition.key == "file_path" for condition in conditions):
            return [], None
        points = [_book_point(document) for document in _book(collection_name)]
        return [point for point in points if _matches(point.payload, conditions)], None

    def retrieve(self, collection_name, ids=(), **_kwargs):
        if self.down:
            raise ConnectionError("[Errno 61] Connection refused")
        wanted = {str(point_id) for point_id in ids}
        if METADATA_POINT_ID in wanted:
            payload = {"is_metadata": True, "source_roots": [f"/src/{collection_name}"]}
            return [SimpleNamespace(id=METADATA_POINT_ID, payload=payload)]
        return [
            _book_point(document) for document in _book(collection_name) if document["id"] in wanted
        ]


class FakeMeili:
    def __init__(self, names, healthy=True, stall=False):
        self.names = list(names)
        self.healthy = healthy
        self.stall = stall

    def health_check(self):
        return self.healthy

    def list_indexes(self):
        return [{"uid": name} for name in self.names]

    def get_stats(self):
        return {"indexes": {name: {"numberOfDocuments": 4} for name in self.names}}

    def index_exists(self, name):
        return name in self.names

    def get_index_settings(self, _name):
        return {"filterableAttributes": ["file_path", "language"]}

    def search(self, name, _query, filter=None, limit=20, offset=0, **_kwargs):
        if self.stall:
            time.sleep(60)
        documents = _documents(name)
        if filter:
            match = re.fullmatch(r'(\w+) = "([^"]*)"', filter)
            if match is None or match.group(1) not in ("file_path", "language"):
                raise RuntimeError(f"invalid_search_filter: Attribute `{filter}` is not filterable")
            documents = [
                d for d in documents + _book(name) if d.get(match.group(1)) == match.group(2)
            ]
        return {
            "hits": documents[offset : offset + limit],
            "estimatedTotalHits": len(documents),
            "processingTimeMs": 1,
        }

    def get_document(self, name, document_id):
        for document in _documents(name) + _book(name):
            if document["id"] == document_id:
                return dict(document)
        return None

    def update_index_settings(self, name, _settings):
        _report(f"fake meili write: update_index_settings {name}")


def _chatter():
    print("print chatter")
    sys.stdout.flush()
    os.write(1, b"raw fd chatter\n")
    logging.getLogger("arcaneum.test.noisy").warning("log chatter")
    warnings.warn("warning chatter", stacklevel=1)
    from rich.console import Console

    Console().print("rich chatter")


class FakeModel:
    def query_embed(self, texts):
        for _text in texts:
            yield [0.1, 0.2, 0.3]


def fake_embedding_client(scenario: str):
    """The query embedder's model loader: no model loads, and none downloads."""

    class FakeEmbeddingClient:
        def __init__(self, cache_dir=None, allow_download=True, **_kwargs):
            self.allow_download = allow_download
            self.models = {}

        def is_model_cached(self, _model_key):
            return scenario != "uncached"

        def get_model(self, model_key):
            if model_key not in self.models:
                if scenario == "embed-stall":
                    time.sleep(60)
                if not self.is_model_cached(model_key):
                    _report(f"fake model download: {model_key}")
                _report(f"fake model load: {model_key}")
                self.models[model_key] = FakeModel()
            return self.models[model_key]

        @property
        def local_models(self):
            return sorted(self.models)

        def unload_model(self, model_key):
            self.models.pop(model_key, None)

        def close(self):
            pass

    return FakeEmbeddingClient


def install(scenario: str, command: list[str]) -> None:
    if scenario == "real":
        return

    import arcaneum.cli.utils as cli_utils
    import arcaneum.retrieval.corpora as corpora

    if scenario == "many":
        metadata = {
            f"Corpus{i:03d}": {"collection_type": "pdf", "description": "x" * 40}
            for i in range(200)
        }
    else:
        metadata = dict(CORPORA)

    qdrant = FakeQdrant(metadata, down=scenario == "qdrant-down")
    meili = FakeMeili(metadata, healthy=scenario != "meili-down", stall=scenario == "stall")

    def get_collection_metadata(_client, name):
        if scenario == "noisy":
            _chatter()
        if scenario == "stall":
            time.sleep(60)
        return metadata.get(name, {})

    corpora.get_collection_metadata = get_collection_metadata
    cli_utils.create_qdrant_client = lambda *args, **kwargs: qdrant

    if command[:1] == ["mcp"] or command[:2] == ["search", "semantic"]:
        import arcaneum.search.embedder as search_embedder

        search_embedder.EmbeddingClient = fake_embedding_client(scenario)
        search_embedder.SearchEmbedder.detect_collection_model = staticmethod(
            lambda _client, name, _vector_name=None: metadata[name]["model"]
        )
    cli_utils.create_meili_client = lambda *args, **kwargs: meili
    if command[:1] == ["corpus"]:
        import arcaneum.cli.corpus as cli_corpus

        cli_corpus.create_qdrant_client = cli_utils.create_qdrant_client
        cli_corpus.create_meili_client = cli_utils.create_meili_client
    if command[:2] == ["search", "text"]:
        import arcaneum.cli.fulltext as cli_fulltext

        cli_fulltext.create_qdrant_client = cli_utils.create_qdrant_client
        cli_fulltext.create_meili_client = cli_utils.create_meili_client
    if command[:2] in (["search", "semantic"], ["search", "passage"]):
        import arcaneum.cli.search as cli_search

        cli_search.create_qdrant_client = cli_utils.create_qdrant_client
        cli_search.create_meili_client = cli_utils.create_meili_client


def main() -> int:
    scenario, command = sys.argv[1], sys.argv[2:]
    install(scenario, command)
    from arcaneum.cli.main import main as arc_main

    sys.argv = ["arc", *command]
    return arc_main()


if __name__ == "__main__":
    sys.exit(main())

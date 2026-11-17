"""Run `arc` against fake Qdrant and MeiliSearch backends for the MCP server tests.

usage: python mcp_harness.py <scenario> <arc arguments...>

The tests drive `arc mcp` (and `arc corpus list --json`, for the CLI-parity
check) as a real child process over pipes. A scenario swaps the client
factories the retrieval core builds its clients from, so no service runs:

- ``fake``: three corpora, both services up
- ``many``: 200 corpora, for the output cap
- ``noisy``: ``fake``, but the listing prints to stdout through every channel a
  library might use (print, a raw fd 1 write, logging, warnings, rich)
- ``stall``: the listing blocks, for the per-call timeout
- ``qdrant-down`` / ``meili-down``: that service refuses connections
- ``real``: no fakes (start-up checks that never call a tool)
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from types import SimpleNamespace

CORPORA = {
    "Alpha": {"collection_type": "pdf", "model": "arctic-m", "description": "Alpha docs"},
    "Beta": {"collection_type": "markdown", "model": "arctic-m", "description": "Beta notes"},
    "Gamma": {"collection_type": "code", "model": "jina-code", "description": "Gamma code"},
}


class FakeQdrant:
    def __init__(self, metadata, down=False):
        self.metadata = metadata
        self.down = down

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

    def scroll(self, **_kwargs):
        return [], None


class FakeMeili:
    def __init__(self, names, healthy=True):
        self.names = list(names)
        self.healthy = healthy

    def health_check(self):
        return self.healthy

    def list_indexes(self):
        return [{"uid": name} for name in self.names]

    def get_stats(self):
        return {"indexes": {name: {"numberOfDocuments": 4} for name in self.names}}


def _chatter():
    print("print chatter")
    sys.stdout.flush()
    os.write(1, b"raw fd chatter\n")
    logging.getLogger("arcaneum.test.noisy").warning("log chatter")
    warnings.warn("warning chatter", stacklevel=1)
    from rich.console import Console

    Console().print("rich chatter")


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
    meili = FakeMeili(metadata, healthy=scenario != "meili-down")

    def get_collection_metadata(_client, name):
        if scenario == "noisy":
            _chatter()
        if scenario == "stall":
            time.sleep(60)
        return metadata.get(name, {})

    corpora.get_collection_metadata = get_collection_metadata
    cli_utils.create_qdrant_client = lambda *args, **kwargs: qdrant
    cli_utils.create_meili_client = lambda *args, **kwargs: meili
    if command[:1] == ["corpus"]:
        import arcaneum.cli.corpus as cli_corpus

        cli_corpus.create_qdrant_client = cli_utils.create_qdrant_client
        cli_corpus.create_meili_client = cli_utils.create_meili_client


def main() -> int:
    scenario, command = sys.argv[1], sys.argv[2:]
    install(scenario, command)
    from arcaneum.cli.main import main as arc_main

    sys.argv = ["arc", *command]
    return arc_main()


if __name__ == "__main__":
    sys.exit(main())

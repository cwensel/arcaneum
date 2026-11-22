"""``arc corpus migrate-text`` on a real Qdrant 1.20 and MeiliSearch (RDR-027 Q9).

Each test builds a MeiliSearch corpus whose chunks are in both stores under
the same ids, as sync writes them, and removes both when done. Scenarios 4
and 5 of RDR-027's Testing Strategy, the dry run, a rollback after a file was
removed on qdrant, and the refusals.

Requires QDRANT_URL (default http://localhost:6333) at 1.20 or later and
MEILISEARCH_URL (default http://localhost:7700, key MEILISEARCH_API_KEY);
skips otherwise.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid

import pytest

from arcaneum.cli import migrate_text
from arcaneum.cli.migrate_text import MigrationError
from arcaneum.fulltext.backend import MeiliTextBackend
from arcaneum.indexing.collection_metadata import get_collection_metadata, set_collection_metadata
from arcaneum.retrieval.qdrant_text import QdrantTextBackend
from arcaneum.schema.document import DualIndexDocument

pytestmark = pytest.mark.integration

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
MEILISEARCH_URL = os.environ.get("MEILISEARCH_URL", "http://localhost:7700")

FILES = {
    "/docs/wal.pdf": [
        "The write-ahead log flushes every dirty page before the checkpoint record completes.",
        "Recovery replays the write-ahead log from the last checkpoint toward the crash point.",
        "Group commit batches several transactions into one durable log flush.",
    ],
    "/docs/btree.pdf": [
        "A B-tree node splits when an insertion overflows its fixed page capacity.",
        "Latch coupling lets readers descend the tree while writers hold sibling latches.",
        "### ** `` __",
    ],
    "/docs/mvcc.pdf": [
        "Snapshot isolation gives each transaction a consistent view of committed versions.",
        "Vacuum reclaims tuple versions that no running snapshot can still observe.",
    ],
}
ZERO_TOKEN_TEXT = "### ** `` __"
PHRASE = '"write-ahead log"'


def _chunk_id(path: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path}#{index}"))


def _docs():
    for path, texts in FILES.items():
        for index, text in enumerate(texts):
            yield DualIndexDocument(
                id=_chunk_id(path, index),
                content=text,
                file_path=path,
                filename=path.rsplit("/", 1)[-1],
                file_extension=".pdf",
                chunk_index=index,
                page_number=index + 1,
                vectors={"stub": [1.0, float(index)]},
            )


@pytest.fixture
def qdrant():
    from qdrant_client import QdrantClient

    client = QdrantClient(url=QDRANT_URL, timeout=30)
    try:
        version = client.info().version
    except Exception as e:
        pytest.skip(f"Qdrant not available at {QDRANT_URL}: {e}")
    if tuple(int(p) for p in version.split(".")[:2]) < (1, 20):
        pytest.skip(f"Qdrant {version} has no BM25 text index; 1.20 is required")
    return client


@pytest.fixture
def meili():
    from arcaneum.fulltext.client import FullTextClient
    from arcaneum.paths import get_meilisearch_api_key

    client = FullTextClient(
        MEILISEARCH_URL, os.environ.get("MEILISEARCH_API_KEY") or get_meilisearch_api_key()
    )
    if not client.health_check():
        pytest.skip(f"MeiliSearch not available at {MEILISEARCH_URL}")
    return client


@pytest.fixture
def corpus(qdrant, meili):
    """A meili pdf corpus with every chunk in both stores under one id."""
    from qdrant_client.http import models

    name = f"arc_migrate_it_{uuid.uuid4().hex[:10]}"
    qdrant.create_collection(
        name,
        vectors_config={"stub": models.VectorParams(size=2, distance=models.Distance.COSINE)},
    )
    set_collection_metadata(qdrant, name, "pdf", "arctic-m")
    docs = list(_docs())
    QdrantTextBackend(qdrant).add(name, docs)
    backend = MeiliTextBackend(meili)
    backend.ensure(name, "pdf")
    backend.add(name, docs)
    yield name
    if qdrant.collection_exists(name):
        qdrant.delete_collection(name)
    backend.drop(name)


def _meili_ids(meili, name, query=PHRASE):
    return [hit["id"] for hit in MeiliTextBackend(meili).search(name, query, limit=50).hits]


def _backend(qdrant, name):
    return migrate_text.text_backend_of(get_collection_metadata(qdrant, name))


def _qdrant_digest(qdrant, name) -> str:
    points, _next = qdrant.scroll(name, limit=1000, with_payload=True, with_vectors=True)
    rows = sorted(json.dumps([str(p.id), p.payload, p.vector], sort_keys=True) for p in points)
    schema = sorted((qdrant.get_collection(name).payload_schema or {}).keys())
    return hashlib.sha256(json.dumps([rows, schema]).encode()).hexdigest()


def _meili_digest(meili, name) -> str:
    docs = MeiliTextBackend(meili).iter_documents(
        name, ["id", "content", "file_path", "chunk_index"]
    )
    rows = sorted(json.dumps(d, sort_keys=True) for d in docs)
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def test_round_trip_keeps_phrase_hit_ids(qdrant, meili, corpus):
    """Scenario 4: --to qdrant, then --to meili, answers a phrase query as before."""
    before = _meili_ids(meili, corpus)
    assert len(before) == 2

    report = migrate_text.migrate_to_qdrant(qdrant, meili, corpus)

    assert report["switched"] is True
    assert _backend(qdrant, corpus) == "qdrant"
    stamp = get_collection_metadata(qdrant, corpus)[migrate_text.MIGRATION_FIELD]
    assert stamp["to"] == "qdrant" and stamp["qdrant_version"] and stamp["arc_version"]
    assert report["coverage"]["status"] == "synced"
    assert report["gate"]["failed"] == 0 and report["gate"]["sample"] == 7
    assert report["reconcile"]["flagged"] == 1  # the zero-token chunk
    flagged = qdrant.retrieve(corpus, ids=[_chunk_id("/docs/btree.pdf", 2)], with_payload=True)
    assert flagged[0].payload["text_tokens"] is False
    qdrant_ids = [h["id"] for h in QdrantTextBackend(qdrant).search(corpus, PHRASE).hits]
    assert sorted(qdrant_ids) == sorted(before)

    back = migrate_text.migrate_to_meili(qdrant, meili, corpus)

    assert back["switched"] is True
    assert _backend(qdrant, corpus) == "meili"
    assert _meili_ids(meili, corpus) == before


def test_interrupted_migration_stays_on_meili_and_reruns(qdrant, meili, corpus, monkeypatch):
    """Scenario 5: a fault before the switch leaves meili; a rerun writes nothing to reconcile."""

    def fault(*_args, **_kwargs):
        raise RuntimeError("injected before the switch")

    with monkeypatch.context() as patched:
        patched.setattr(migrate_text, "update_collection_metadata", fault)
        with pytest.raises(RuntimeError, match="injected"):
            migrate_text.migrate_to_qdrant(qdrant, meili, corpus)

    assert _backend(qdrant, corpus) == "meili"

    rerun = migrate_text.migrate_to_qdrant(qdrant, meili, corpus)

    assert rerun["reconcile"]["copied"] == 0
    assert rerun["reconcile"]["flagged"] == 0
    assert rerun["switched"] is True
    assert _backend(qdrant, corpus) == "qdrant"


def test_dry_run_writes_nothing(qdrant, meili, corpus):
    qdrant_before, meili_before = _qdrant_digest(qdrant, corpus), _meili_digest(meili, corpus)

    report = migrate_text.migrate_to_qdrant(qdrant, meili, corpus, dry_run=True)

    assert report["switched"] is False
    assert report["reconcile"]["flagged"] == 1
    # The tokenful chunk count the text index must reach.
    assert report["coverage"]["searchable_chunks"] == 7
    assert _qdrant_digest(qdrant, corpus) == qdrant_before
    assert _meili_digest(meili, corpus) == meili_before


def test_reconcile_copies_text_qdrant_lacks(qdrant, meili, corpus):
    """A point without text gets MeiliSearch's content; its vector is untouched."""
    chunk = _chunk_id("/docs/mvcc.pdf", 1)
    qdrant.delete_payload(corpus, keys=["text"], points=[chunk], wait=True)
    (vector_before,) = qdrant.retrieve(corpus, ids=[chunk], with_vectors=True)

    report = migrate_text.migrate_to_qdrant(qdrant, meili, corpus)

    assert report["reconcile"]["copied"] == 1
    (point,) = qdrant.retrieve(corpus, ids=[chunk], with_payload=True, with_vectors=True)
    assert point.payload["text"] == FILES["/docs/mvcc.pdf"][1]
    assert point.vector == vector_before.vector


def test_rollback_after_finalize_and_removed_file_leaves_no_stale_document(qdrant, meili, corpus):
    migrate_text.migrate_to_qdrant(qdrant, meili, corpus)
    QdrantTextBackend(qdrant).delete_files(corpus, ["/docs/mvcc.pdf"])  # removed while on qdrant

    finalized = migrate_text.finalize(qdrant, meili, corpus)
    assert finalized["deleted"] is True
    assert not MeiliTextBackend(meili).exists(corpus)

    migrate_text.migrate_to_meili(qdrant, meili, corpus)

    backend = MeiliTextBackend(meili)
    assert not backend.has_file(corpus, "/docs/mvcc.pdf")
    assert backend.coverage(corpus, files=False).chunks == 6
    assert "file_path" in backend.filterable_attributes(corpus)  # recreated with pdf settings


class _OldQdrant:
    def __init__(self, client):
        self._client = client

    def info(self):
        from types import SimpleNamespace

        return SimpleNamespace(version="1.19.2")

    def __getattr__(self, name):
        return getattr(self._client, name)


def test_refusals_name_the_way_forward(qdrant, meili, corpus):
    with pytest.raises(MigrationError, match="arc container upgrade"):
        migrate_text.migrate_to_qdrant(_OldQdrant(qdrant), meili, corpus)
    with pytest.raises(MigrationError, match="not on qdrant"):
        migrate_text.finalize(qdrant, meili, corpus)
    with pytest.raises(MigrationError, match="never migrated"):
        migrate_text.migrate_to_meili(qdrant, meili, corpus)

    migrate_text.migrate_to_qdrant(qdrant, meili, corpus)
    with pytest.raises(MigrationError, match="already on qdrant"):
        migrate_text.migrate_to_qdrant(qdrant, meili, corpus)


def test_parity_mismatch_names_the_repair(qdrant, meili, corpus):
    MeiliTextBackend(meili).delete_files(corpus, ["/docs/wal.pdf"])

    with pytest.raises(MigrationError, match=f"arc corpus repair {corpus}"):
        migrate_text.migrate_to_qdrant(qdrant, meili, corpus)
    assert _backend(qdrant, corpus) == "meili"


def test_meilisearch_only_chunk_leaves_the_corpus_on_meili(qdrant, meili, corpus):
    """Per-file counts agree, but one MeiliSearch chunk has no Qdrant point."""
    old = _chunk_id("/docs/wal.pdf", 2)
    (point,) = qdrant.retrieve(corpus, ids=[old], with_payload=True, with_vectors=True)
    qdrant.delete(corpus, points_selector=[old], wait=True)
    from qdrant_client.http import models

    qdrant.upsert(
        corpus,
        points=[
            models.PointStruct(id=str(uuid.uuid4()), vector=point.vector, payload=point.payload)
        ],
        wait=True,
    )

    with pytest.raises(MigrationError, match="arc corpus repair") as refused:
        migrate_text.migrate_to_qdrant(qdrant, meili, corpus)

    assert refused.value.report["reconcile"]["meili_only"] == 1
    assert _backend(qdrant, corpus) == "meili"

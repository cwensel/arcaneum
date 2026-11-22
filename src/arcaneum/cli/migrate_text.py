"""``arc corpus migrate-text``: move a corpus's text search between backends (RDR-027 Q9).

``--to qdrant`` checks the stores agree, copies MeiliSearch's text into any
chunk point that lacks it, builds Q3's indexes, proves every sampled chunk is
findable, and switches the corpus as its last write, so an interrupted run
leaves the corpus on ``meili`` and a rerun completes. MeiliSearch's index is
kept until ``--finalize`` deletes it. ``--to meili`` switches back, clears the
retained index (or recreates it) and refills it from Qdrant through repair's
backfill. Every vector and chunk id stays; nothing is re-extracted or
re-embedded.

The ``migrate_*`` functions take both clients and return a report; the
command wraps them in the corpus write lock and the JSON envelope.
"""

from __future__ import annotations

import contextlib
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..cli.errors import ArcaneumError, ResourceNotFoundError
from ..fulltext.backend import MeiliTextBackend
from ..indexing.collection_metadata import (
    get_collection_metadata,
    text_backend_of,
    update_collection_metadata,
)
from ..retrieval.qdrant_text import (
    TEXT_FIELD,
    TEXT_TOKENS_FIELD,
    QdrantTextBackend,
    _chunk_filter,
    _point_id,
    tokenize,
    zero_token,
)
from ..schema.document import current_app_version

# The metadata key of the migration stamp, beside ``text_backend`` (Q2).
MIGRATION_FIELD = "text_migration"

GATE_FRACTION = 0.01  # Q9 step 4: a 1% sample of tokenful chunks...
GATE_MINIMUM = 200  # ...of at least 200
GATE_SEED = 27
SPAN_TOKENS = 6  # the assumptions spike's span length (C4)
OVERLAP_QUERIES = 20  # spans also run through both backends, for the report
REPORT_SAMPLE = 20  # ids a report names per finding
GREEN_TIMEOUT_S = 3600.0
_BATCH = 1000

Say = Callable[[str], None]


def _quiet(_message: str) -> None:
    pass


class MigrationError(ArcaneumError):
    """The migration stopped; the message names the command that unblocks it.

    ``report`` holds what the run found before it stopped.
    """

    def __init__(self, message: str, report: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.report = report or {}


# ---------------------------------------------------------------------------
# Reconcile (Q9 step 2)
# ---------------------------------------------------------------------------


@dataclass
class Reconcile:
    """MeiliSearch's documents against Qdrant's chunk points, by id and text."""

    scanned: int = 0
    equal: int = 0
    copied: int = 0  # points given MeiliSearch's content
    flagged: int = 0  # zero-token points given text_tokens: false
    unflagged: int = 0  # tokenful points that carried a stale flag
    stubs: List[str] = field(default_factory=list)  # documents without content
    meili_only: List[str] = field(default_factory=list)  # documents with no point
    tokenful_ids: List[Any] = field(default_factory=list)

    def to_data(self) -> Dict[str, Any]:
        return {
            "scanned": self.scanned,
            "equal": self.equal,
            "copied": self.copied,
            "flagged": self.flagged,
            "unflagged": self.unflagged,
            "stubs": len(self.stubs),
            "stub_ids": self.stubs[:REPORT_SAMPLE],
            "meili_only": len(self.meili_only),
            "meili_only_ids": self.meili_only[:REPORT_SAMPLE],
            "tokenful_chunks": len(self.tokenful_ids),
        }


def _batches(items: Iterable[Any], size: int = _BATCH) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def reconcile(
    qdrant,
    meili: MeiliTextBackend,
    corpus: str,
    corpus_type: Optional[str],
    *,
    dry_run: bool = False,
) -> Reconcile:
    """Stream MeiliSearch's ``id``/``content`` against the points of the same ids.

    A point whose ``text`` is missing or differs gets MeiliSearch's content;
    every zero-token chunk scanned gets ``text_tokens: false`` (Q8). The
    vector and id are never touched. Under ``dry_run`` the counts are what
    the writes would be, and nothing is written.
    """
    result = Reconcile()
    documents = meili.iter_documents(corpus, ["id", "content"], batch_size=_BATCH)
    for batch in _batches(documents):
        ids = {str(doc.get("id")): _point_id(doc.get("id")) for doc in batch}
        points = qdrant.retrieve(
            collection_name=corpus,
            ids=[pid for pid in ids.values() if pid is not None],
            with_payload=[TEXT_FIELD, TEXT_TOKENS_FIELD],
            with_vectors=False,
        )
        by_id = {str(p.id): p for p in points}
        to_flag: List[Any] = []
        to_unflag: List[Any] = []
        for doc in batch:
            result.scanned += 1
            doc_id = str(doc.get("id"))
            point = by_id.get(str(ids[doc_id])) if ids[doc_id] is not None else None
            content = doc.get("content")
            if point is None:
                if content:
                    result.meili_only.append(doc_id)
                else:
                    result.stubs.append(doc_id)
                continue
            payload = point.payload or {}
            text = payload.get(TEXT_FIELD)
            if not content:
                # A stub: skipped, but its point's own text decides its flag.
                result.stubs.append(doc_id)
                content = text or ""
            elif text != content:
                result.copied += 1
                if not dry_run:
                    qdrant.set_payload(
                        collection_name=corpus,
                        payload={TEXT_FIELD: content},
                        points=[point.id],
                        wait=True,
                    )
            else:
                result.equal += 1
            flagged = payload.get(TEXT_TOKENS_FIELD) is False
            if zero_token(content, corpus_type):
                if not flagged:
                    to_flag.append(point.id)
            else:
                result.tokenful_ids.append(point.id)
                if flagged:
                    to_unflag.append(point.id)
        result.flagged += len(to_flag)
        result.unflagged += len(to_unflag)
        if dry_run:
            continue
        if to_flag:
            qdrant.set_payload(
                collection_name=corpus,
                payload={TEXT_TOKENS_FIELD: False},
                points=to_flag,
                wait=True,
            )
        if to_unflag:
            qdrant.delete_payload(
                collection_name=corpus, keys=[TEXT_TOKENS_FIELD], points=to_unflag, wait=True
            )
    return result


# ---------------------------------------------------------------------------
# The gate (Q9 step 4)
# ---------------------------------------------------------------------------


def gate_sample_size(tokenful: int) -> int:
    """1% of the tokenful chunks, at least 200, at most all of them."""
    return min(tokenful, max(GATE_MINIMUM, math.ceil(tokenful * GATE_FRACTION)))


def distinctive_span(tokens: List[str], idf: Dict[str, float]) -> List[str]:
    """The ``SPAN_TOKENS`` consecutive tokens with the highest summed IDF."""
    if len(tokens) <= SPAN_TOKENS:
        return list(tokens)
    best = max(
        range(len(tokens) - SPAN_TOKENS + 1),
        key=lambda i: sum(idf.get(t, 0.0) for t in tokens[i : i + SPAN_TOKENS]),
    )
    return tokens[best : best + SPAN_TOKENS]


def _phrase_condition(phrase: str):
    from qdrant_client.http import models

    return models.FieldCondition(key=TEXT_FIELD, match=models.MatchPhrase(phrase=phrase))


def _scroll_texts(qdrant, corpus: str) -> Iterable[str]:
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=corpus,
            scroll_filter=_chunk_filter(),
            limit=_BATCH,
            offset=offset,
            with_payload=[TEXT_FIELD],
            with_vectors=False,
        )
        for point in points:
            yield (point.payload or {}).get(TEXT_FIELD) or ""
        if offset is None:
            return


def prove(
    qdrant,
    meili: MeiliTextBackend,
    corpus: str,
    corpus_type: Optional[str],
    tokenful_ids: List[Any],
) -> Dict[str, Any]:
    """Count each sampled chunk by its own id AND its span's phrase; report ranking.

    The gate passes when every count is 1. The top-10 rate and the
    cross-backend overlap are reported, never gated: a span is often shared
    by other chunks (assumptions spike C4).
    """
    from qdrant_client.http import models

    sample = random.Random(GATE_SEED).sample(
        sorted(tokenful_ids, key=str), gate_sample_size(len(tokenful_ids))
    )
    tokens: Dict[Any, List[str]] = {}
    for batch in _batches(sample, 256):
        for point in qdrant.retrieve(
            collection_name=corpus, ids=batch, with_payload=[TEXT_FIELD], with_vectors=False
        ):
            tokens[point.id] = tokenize((point.payload or {}).get(TEXT_FIELD) or "", corpus_type)

    # Document frequency of the sampled chunks' tokens only, over every chunk.
    wanted = {t for ts in tokens.values() for t in ts}
    df: Counter = Counter()
    docs = 0
    for text in _scroll_texts(qdrant, corpus):
        docs += 1
        df.update(set(tokenize(text, corpus_type)) & wanted)
    idf = {t: math.log(max(docs, 1) / max(df[t], 1)) for t in wanted}

    failures: List[Dict[str, Any]] = []
    top10 = 0
    spans: List[str] = []
    for chunk_id in sample:
        phrase = " ".join(distinctive_span(tokens.get(chunk_id, []), idf))
        spans.append(phrase)
        own = models.Filter(
            must=[models.HasIdCondition(has_id=[chunk_id]), _phrase_condition(phrase)]
        )
        if qdrant.count(collection_name=corpus, count_filter=own, exact=True).count != 1:
            failures.append({"id": str(chunk_id), "span": phrase})
        hits = qdrant.query_points(
            collection_name=corpus,
            query=models.TextQuery(text=phrase),
            using=TEXT_FIELD,
            query_filter=_chunk_filter(models.Filter(must=[_phrase_condition(phrase)])),
            limit=10,
            with_payload=False,
            with_vectors=False,
        ).points
        top10 += any(str(hit.id) == str(chunk_id) for hit in hits)

    qdrant_text = QdrantTextBackend(qdrant, {corpus: corpus_type})
    overlaps = []
    for phrase in spans[:OVERLAP_QUERIES]:
        query = f'"{phrase}"'
        a = {h["id"] for h in meili.search(corpus, query, limit=10).hits}
        b = {h["id"] for h in qdrant_text.search(corpus, query, limit=10).hits}
        overlaps.append(len(a & b) / max(len(a), len(b)) if a or b else 1.0)

    return {
        "sample": len(sample),
        "passed": len(sample) - len(failures),
        "failed": len(failures),
        "failures": failures[:REPORT_SAMPLE],
        "top10_rate": round(top10 / len(sample), 4) if sample else None,
        "overlap_queries": len(overlaps),
        "top10_overlap": round(sum(overlaps) / len(overlaps), 4) if overlaps else None,
    }


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def _metadata(qdrant, corpus: str) -> Dict[str, Any]:
    if not qdrant.collection_exists(corpus):
        raise ResourceNotFoundError(
            f"Corpus '{corpus}' not found. List corpora with: arc corpus list"
        )
    return get_collection_metadata(qdrant, corpus) or {}


def _qdrant_version(qdrant) -> str:
    from .doctor import QDRANT_TEXT_MIN_VERSION, _version_tuple

    version = str(qdrant.info().version)
    if _version_tuple(version)[:2] < QDRANT_TEXT_MIN_VERSION:
        minimum = ".".join(str(part) for part in QDRANT_TEXT_MIN_VERSION)
        raise MigrationError(
            f"Qdrant {version} is below {minimum}, which a qdrant corpus needs. "
            "Upgrade with: arc container upgrade"
        )
    return version


def _collection_status(qdrant, corpus: str) -> str:
    status = getattr(qdrant.get_collection(corpus), "status", None)
    return str(getattr(status, "value", status)).lower()


def _check_parity(qdrant, meili_client, corpus: str, corpus_type: Optional[str]) -> Dict[str, Any]:
    """``arc corpus verify``'s per-item chunk comparison; a mismatch names the repair."""
    from .corpus import _compare_parity_items, _scan_meili_parity_items, _scan_qdrant_parity_items

    parity = _compare_parity_items(
        _scan_qdrant_parity_items(qdrant, corpus, corpus_type),
        _scan_meili_parity_items(meili_client, corpus, corpus_type, ids_only=True),
    )
    if parity["status"] != "synced":
        counts = ", ".join(f"{n} {s}" for s, n in parity["summary"].items() if n and s != "synced")
        raise MigrationError(
            f"Qdrant and MeiliSearch disagree for '{corpus}' ({counts}). "
            f"Repair with: arc corpus repair {corpus}, then rerun the migration",
            {"parity": parity},
        )
    return parity


def _stamp(to: str, **fields: Any) -> Dict[str, Any]:
    return {
        "to": to,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arc_version": current_app_version(),
        **fields,
    }


# ---------------------------------------------------------------------------
# --to qdrant
# ---------------------------------------------------------------------------


def migrate_to_qdrant(
    qdrant, meili_client, corpus: str, *, dry_run: bool = False, say: Say = _quiet
) -> Dict[str, Any]:
    """Q9 steps 1-5; the switch is the last write. Raises ``MigrationError`` on a stop."""
    version = _qdrant_version(qdrant)
    metadata = _metadata(qdrant, corpus)
    if text_backend_of(metadata) == "qdrant":
        raise MigrationError(
            f"Corpus '{corpus}' is already on qdrant. Roll back with: "
            f"arc corpus migrate-text {corpus} --to meili"
        )
    corpus_type = metadata.get("collection_type")
    meili = MeiliTextBackend(meili_client)
    meili.health()
    if not meili.exists(corpus):
        raise MigrationError(
            f"MeiliSearch has no index '{corpus}' to migrate from. "
            f"Rebuild it with: arc corpus repair {corpus}"
        )
    status = _collection_status(qdrant, corpus)
    if status != "green":
        raise MigrationError(
            f"Collection '{corpus}' is {status}, not green. Wait for Qdrant's optimizer, then rerun"
        )
    report: Dict[str, Any] = {
        "corpus": corpus,
        "to": "qdrant",
        "dry_run": dry_run,
        "qdrant_version": version,
        "switched": False,
    }
    say(f"Checking parity of '{corpus}'...")
    report["parity"] = _check_parity(qdrant, meili_client, corpus, corpus_type)["status"]

    say("Reconciling MeiliSearch's text against Qdrant's chunk points...")
    result = reconcile(qdrant, meili, corpus, corpus_type, dry_run=dry_run)
    report["reconcile"] = result.to_data()
    if result.meili_only:
        raise MigrationError(
            f"MeiliSearch holds {len(result.meili_only)} chunk(s) of '{corpus}' that Qdrant lacks; "
            f"the corpus stays on meili. Repair with: arc corpus repair {corpus}",
            report,
        )

    if dry_run:
        # The coverage the text index must reach (Q8).
        report["coverage"] = {"searchable_chunks": len(result.tokenful_ids)}
        report["text_backend"] = "meili"
        return report

    say("Building the text index...")
    from ..indexing.common.sync import MetadataBasedSync

    MetadataBasedSync(qdrant).ensure_corpus_payload_indexes(corpus)
    text_index = QdrantTextBackend(qdrant, {corpus: corpus_type})
    text_index.ensure(corpus, corpus_type)
    if not text_index.wait_green(corpus, timeout=GREEN_TIMEOUT_S):
        raise MigrationError(
            f"Collection '{corpus}' did not turn green within {GREEN_TIMEOUT_S:.0f}s; "
            "the corpus stays on meili. Rerun once Qdrant's optimizer finishes",
            report,
        )
    coverage = text_index.coverage(corpus, files=False)
    report["coverage"] = coverage.to_data()
    if not coverage.complete:
        raise MigrationError(
            f"The text index of '{corpus}' holds {coverage.indexed or 0} of "
            f"{coverage.searchable_chunks} tokenful chunks; the corpus stays on meili. "
            f"Repair with: arc corpus repair {corpus}, then rerun the migration",
            report,
        )

    say("Proving every sampled chunk is findable by its own text...")
    report["gate"] = prove(qdrant, meili, corpus, corpus_type, result.tokenful_ids)
    if report["gate"]["failed"]:
        raise MigrationError(
            f"{report['gate']['failed']} of {report['gate']['sample']} sampled chunks of "
            f"'{corpus}' did not match their own span; the corpus stays on meili",
            report,
        )

    stamp = _stamp(
        "qdrant",
        qdrant_version=version,
        chunks=coverage.chunks,
        searchable_chunks=coverage.searchable_chunks,
        text_chunks=coverage.indexed,
        copied=result.copied,
        stubs=len(result.stubs),
        gate_sample=report["gate"]["sample"],
    )
    update_collection_metadata(qdrant, corpus, text_backend="qdrant", **{MIGRATION_FIELD: stamp})
    report.update(switched=True, text_backend="qdrant", stamp=stamp)
    return report


# ---------------------------------------------------------------------------
# --to meili
# ---------------------------------------------------------------------------


def migrate_to_meili(
    qdrant,
    meili_client,
    corpus: str,
    *,
    dry_run: bool = False,
    verbose: bool = False,
    quiet: bool = True,
    say: Say = _quiet,
) -> Dict[str, Any]:
    """Switch back to meili, clear (or recreate) the index, and refill it from Qdrant.

    A ``meili`` corpus is refused unless it carries a migration stamp, so a
    rerun after an interrupted rollback proceeds.
    """
    from rich.progress import Progress

    from .sync import _backfill_qdrant_to_meili

    metadata = _metadata(qdrant, corpus)
    if text_backend_of(metadata) != "qdrant" and not metadata.get(MIGRATION_FIELD):
        raise MigrationError(
            f"Corpus '{corpus}' is on meili and was never migrated; there is nothing to roll back"
        )
    corpus_type = metadata.get("collection_type")
    meili = MeiliTextBackend(meili_client)
    meili.health()
    files = {
        path: n
        for path, n in QdrantTextBackend(qdrant, {corpus: corpus_type})
        .coverage(corpus, files=True)
        .files.items()
        if path
    }
    report: Dict[str, Any] = {
        "corpus": corpus,
        "to": "meili",
        "dry_run": dry_run,
        "files": len(files),
        "chunks": sum(files.values()),
        "index_existed": meili.exists(corpus),
        "switched": False,
    }
    if dry_run:
        report["text_backend"] = text_backend_of(metadata)
        return report

    stamp = _stamp("meili", refilled=False)
    update_collection_metadata(qdrant, corpus, text_backend="meili", **{MIGRATION_FIELD: stamp})
    report.update(switched=True, text_backend="meili")

    # Chunks changed or removed while on qdrant must leave nothing stale.
    if report["index_existed"]:
        say(f"Clearing the retained MeiliSearch index '{corpus}'...")
        meili.clear(corpus)
    else:
        say(f"Recreating the MeiliSearch index '{corpus}'...")
        meili.ensure(corpus, corpus_type)

    say(f"Refilling MeiliSearch from Qdrant ({len(files)} files)...")
    with Progress(disable=quiet) as progress:
        task = progress.add_task("Backfilling...", total=len(files))
        _files_ok, chunks_ok, files_failed = _backfill_qdrant_to_meili(
            qdrant, meili_client, corpus, sorted(files), verbose, quiet, progress, task
        )
    report["refill"] = {"chunks": chunks_ok, "files_failed": files_failed}
    held = meili.coverage(corpus, files=False).chunks
    if files_failed or held != report["chunks"]:
        raise MigrationError(
            f"MeiliSearch holds {held} of {report['chunks']} chunks of '{corpus}' "
            "after the refill. "
            f"Rerun with: arc corpus migrate-text {corpus} --to meili",
            report,
        )
    stamp = {**stamp, "refilled": True, "files": len(files), "chunks": held}
    update_collection_metadata(qdrant, corpus, **{MIGRATION_FIELD: stamp})
    report["stamp"] = stamp
    return report


# ---------------------------------------------------------------------------
# --finalize
# ---------------------------------------------------------------------------


def finalize(qdrant, meili_client, corpus: str, *, dry_run: bool = False) -> Dict[str, Any]:
    """Delete the retained MeiliSearch index of a qdrant corpus."""
    metadata = _metadata(qdrant, corpus)
    if text_backend_of(metadata) != "qdrant":
        raise MigrationError(
            f"Corpus '{corpus}' is not on qdrant; --finalize would delete the index it searches"
        )
    meili = MeiliTextBackend(meili_client)
    meili.health()
    existed = meili.exists(corpus)
    report = {"corpus": corpus, "finalize": True, "dry_run": dry_run, "existed": existed}
    if dry_run:
        report["deleted"] = False
        return report
    meili.drop(corpus)
    stamp = {**(metadata.get(MIGRATION_FIELD) or {}), "finalized_at": _stamp("qdrant")["at"]}
    update_collection_metadata(qdrant, corpus, **{MIGRATION_FIELD: stamp})
    report["deleted"] = existed
    return report


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def migrate_text_command(
    corpus: str,
    to: Optional[str],
    do_finalize: bool,
    dry_run: bool,
    verbose: bool,
    output_json: bool,
) -> None:
    """``arc corpus migrate-text``, under the corpus write lock unless ``--dry-run``.

    ``--to meili`` and ``--finalize`` need MeiliSearch, so they are refused,
    before any client is made, while ``meilisearch.enabled`` is false.
    """
    from rich.console import Console

    from ..config import resolve_meili_settings
    from .corpus_lock import acquire_corpus_lock
    from .output import print_json
    from .utils import create_meili_client, create_qdrant_client

    console = Console(stderr=output_json)

    def say(message: str) -> None:
        if not output_json:
            console.print(f"[dim]{message}[/dim]")

    def fail(e: MigrationError) -> None:
        if output_json:
            print_json("error", str(e), data=e.report, errors=[e.to_error()])
        else:
            print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(e.exit_code)

    if do_finalize or to == "meili":
        settings = resolve_meili_settings()
        if not settings.enabled:
            step = "--finalize" if do_finalize else "--to meili"
            fail(
                MigrationError(
                    f"migrate-text {step} needs MeiliSearch: {settings.disabled_reason()}",
                    {"corpus": corpus},
                )
            )

    qdrant = create_qdrant_client()
    meili_client = create_meili_client()
    lock = contextlib.nullcontext() if dry_run else acquire_corpus_lock(corpus, quiet=output_json)
    try:
        with lock:
            if do_finalize:
                report = finalize(qdrant, meili_client, corpus, dry_run=dry_run)
            elif to == "qdrant":
                report = migrate_to_qdrant(qdrant, meili_client, corpus, dry_run=dry_run, say=say)
            else:
                report = migrate_to_meili(
                    qdrant,
                    meili_client,
                    corpus,
                    dry_run=dry_run,
                    verbose=verbose,
                    quiet=output_json,
                    say=say,
                )
    except MigrationError as e:
        fail(e)

    message = _summary(report)
    if output_json:
        print_json("success", message, data=report)
        return
    console.print(f"[green]{message}[/green]")
    for key in ("reconcile", "coverage", "gate", "refill"):
        if report.get(key):
            details = ", ".join(
                f"{k}={v}" for k, v in report[key].items() if not isinstance(v, (list, dict))
            )
            console.print(f"  {key}: {details}")


def _summary(report: Dict[str, Any]) -> str:
    corpus = report["corpus"]
    if report.get("finalize"):
        if report["dry_run"]:
            return (
                f"Would delete MeiliSearch index '{corpus}'"
                if report["existed"]
                else (f"MeiliSearch holds no index '{corpus}'")
            )
        return (
            f"Deleted MeiliSearch index '{corpus}'"
            if report["deleted"]
            else (f"MeiliSearch held no index '{corpus}'")
        )
    if report["dry_run"]:
        return f"Dry run: '{corpus}' can move to {report['to']}; nothing was written"
    return f"Corpus '{corpus}' now searches text on {report['to']}"

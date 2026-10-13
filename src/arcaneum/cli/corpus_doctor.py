"""`arc corpus doctor`: read-only health sweep across corpora.

For each corpus, runs the same checks as `arc corpus verify` (persisted
content verification and Qdrant/MeiliSearch parity) plus orphan
classification, then names the command that fixes each finding. Nothing is
written. Commands that migrate policy or could delete documents for files
on an unmounted volume are listed as optional, never as required.
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from .errors import ResourceNotFoundError
from .output import print_error, print_json

console = Console()

# Corpora checked at once.
DOCTOR_WORKERS = 4

# Verifier counters worth naming when explaining why files need re-indexing.
_REPAIR_DETAIL_COUNTERS = (
    "incomplete_items",
    "duplicate_items",
    "garbled_items",
    "dropout_items",
    "duplicate_source_groups",
    "quality_manifest_gaps",
)


def diagnose_corpus(
    name: str,
    presence: str,
    verification,
    parity: Optional[Dict[str, Any]],
    orphans: Optional[Dict[str, List[str]]],
) -> Dict[str, Any]:
    """Turn check results into findings and the commands that fix them.

    Args:
        name: Corpus name
        presence: "both", "qdrant_only", or "meili_only"
        verification: CollectionVerificationResult, or None when not run
        parity: Result of corpus._compare_parity_items, an
            {"status": "unknown", "reason": ...} dict, or None when not run
        orphans: Result of sync._classify_meili_orphans for MeiliSearch-only
            files missing from disk, or None

    Each finding has a severity: "error" (a check failed), "action" (needs a
    fix), "optional" (a deliberate choice, e.g. policy migration), or "info".
    """
    from ..indexing.verify import STALE_EXTRACTION_SELECTOR, file_verification_breakdown

    findings: List[Dict[str, Any]] = []
    orphans = orphans or {"prunable": [], "dir_missing": [], "referenced": []}
    repair = f"arc corpus repair {name}" + (" --prune-orphans" if orphans["prunable"] else "")

    def add(code, severity, message, count=None, command=None, **extra):
        finding = {"code": code, "severity": severity, "message": message, "command": command}
        if count is not None:
            finding["count"] = count
        finding.update(extra)
        findings.append(finding)

    if presence == "qdrant_only":
        add(
            "qdrant_only_corpus",
            "action",
            "Corpus exists only in Qdrant; create its MeiliSearch index and backfill",
            command=f"arc corpus parity {name} --create-missing",
        )
    elif presence == "meili_only":
        add(
            "meili_only_corpus",
            "action",
            "Corpus exists only in MeiliSearch; create the Qdrant collection with "
            f"`arc corpus create {name} --type <pdf|code|markdown>` and re-sync its sources",
        )

    if verification is not None:
        if verification.errors:
            add(
                "verification_errors",
                "error",
                "; ".join(verification.errors),
                count=len(verification.errors),
            )
        breakdown = file_verification_breakdown(verification, include_stale_policy=False)
        # Files whose only defect is a pre-fix normalized extraction get a
        # targeted re-extract; repair would also fix them, but scans for more.
        stale_extraction_only = set(breakdown.get("stale_extraction_only_paths", []))
        other_repairable = [
            p for p in breakdown["repairable_paths"] if p not in stale_extraction_only
        ]
        if other_repairable:
            details = {
                counter: getattr(verification, counter, 0)
                for counter in _REPAIR_DETAIL_COUNTERS
                if getattr(verification, counter, 0)
            }
            add(
                "repairable_files",
                "action",
                f"{len(other_repairable)} files need re-indexing",
                count=len(other_repairable),
                command=repair,
                details=details,
            )
        if stale_extraction_only:
            add(
                "stale_extraction",
                "action",
                f"{len(stale_extraction_only)} PDFs were extracted before the column-order "
                "fix, so two-column pages are indexed as interleaved lines; re-extract them",
                count=len(stale_extraction_only),
                command=f"arc corpus sync {name} --from-index '{STALE_EXTRACTION_SELECTOR}'",
            )
        if breakdown["policy_only_files"]:
            add(
                "policy_only_files",
                "optional",
                f"{breakdown['policy_only_files']} files are healthy but indexed under a "
                "stale policy; migrate them only if you want the current policy",
                count=breakdown["policy_only_files"],
                command=f"arc corpus repair {name} --include-stale-policy",
            )
        if breakdown["recovery_exhausted_files"]:
            add(
                "recovery_exhausted_files",
                "info",
                f"{breakdown['recovery_exhausted_files']} files are degraded after OCR; "
                "automatic recovery is exhausted",
                count=breakdown["recovery_exhausted_files"],
            )

    if parity is not None:
        if parity["status"] == "unknown":
            add("parity_unknown", "error", parity["reason"])
        elif parity["status"] == "mismatch":
            orphaned = {path for paths in orphans.values() for path in paths}
            fixable = [item for item in parity["items"] if item["id"] not in orphaned]
            if fixable:
                add(
                    "parity_mismatch",
                    "action",
                    f"{len(fixable)} items differ between Qdrant and MeiliSearch",
                    count=len(fixable),
                    command=repair,
                )

    if orphans["prunable"]:
        add(
            "orphans_prunable",
            "action",
            f"{len(orphans['prunable'])} files are gone from Qdrant and disk but still "
            "in MeiliSearch",
            count=len(orphans["prunable"]),
            command=repair,
        )
    if orphans["dir_missing"]:
        add(
            "orphans_dir_missing",
            "action",
            f"{len(orphans['dir_missing'])} files are only in MeiliSearch and their "
            "directory is missing; mount the volume and re-check, or prune them if "
            "they were deleted",
            count=len(orphans["dir_missing"]),
            command=f"arc corpus repair {name} --force-prune",
            optional=True,
        )
    if orphans["referenced"]:
        add(
            "orphans_referenced",
            "action",
            f"{len(orphans['referenced'])} MeiliSearch-only paths are still named as a "
            "duplicate PDF's canonical copy; review manually",
            count=len(orphans["referenced"]),
        )

    def unique(commands):
        return list(dict.fromkeys(c for c in commands if c))

    required = [
        f["command"] for f in findings if f["severity"] == "action" and not f.get("optional")
    ]
    optional = [f["command"] for f in findings if f["severity"] == "optional" or f.get("optional")]
    if any(f["severity"] == "error" for f in findings):
        status = "error"
    elif any(f["severity"] == "action" for f in findings):
        status = "needs_action"
    else:
        status = "ok"

    for finding in findings:
        finding.pop("optional", None)
    return {
        "corpus": name,
        "status": status,
        "findings": findings,
        "commands": unique(required),
        "optional_commands": unique(optional),
    }


def _open_clients():
    """Return (qdrant, meili), failing if either service is unreachable.

    A sweep with one service down would report every corpus as one-sided.
    """
    from .sync import get_meili_client
    from .utils import create_qdrant_client

    qdrant = create_qdrant_client()
    qdrant.get_collections()
    meili = get_meili_client()
    if not meili.health_check():
        raise ResourceNotFoundError("MeiliSearch server not available")
    return qdrant, meili


def _discover_corpora(qdrant, meili) -> Dict[str, str]:
    """Map every corpus name to "both", "qdrant_only", or "meili_only"."""
    qdrant_names = {c.name for c in qdrant.get_collections().collections}
    meili_names = {idx["uid"] for idx in meili.list_indexes()}
    return {
        name: (
            "both"
            if name in qdrant_names and name in meili_names
            else ("qdrant_only" if name in qdrant_names else "meili_only")
        )
        for name in sorted(qdrant_names | meili_names)
    }


def _check_corpus(qdrant, meili, name: str, presence: str) -> Dict[str, Any]:
    """Run the read-only checks for one corpus and diagnose the results."""
    from ..indexing.collection_metadata import get_collection_type
    from ..indexing.verify import CollectionVerifier
    from . import corpus as corpus_cli
    from . import sync as sync_cli

    verification = None
    parity = None
    orphans = None

    if presence != "meili_only":
        try:
            verification = CollectionVerifier(qdrant).verify_collection(name)
        except Exception as e:
            verification = _FailedVerification(f"Verification failed: {e}")

    if presence == "both":
        try:
            collection_type = getattr(verification, "collection_type", None) or (
                get_collection_type(qdrant, name)
            )
            # Reuse the verifier's point counts; scan only when it failed or
            # verified per project (code).
            qdrant_items = None
            if not isinstance(verification, _FailedVerification):
                qdrant_items = corpus_cli._qdrant_parity_items_from_verification(verification)
            if qdrant_items is None:
                qdrant_items = corpus_cli._scan_qdrant_parity_items(qdrant, name, collection_type)
            parity = corpus_cli._compare_parity_items(
                qdrant_items,
                corpus_cli._scan_meili_parity_items(meili, name, collection_type, ids_only=True),
            )
            # Code corpora compare per git project, not per file, so orphaned
            # files cannot be told apart here; repair's file-level pass handles them.
            if collection_type != "code":
                gone = [
                    item["id"]
                    for item in parity["items"]
                    if item["status"] == "meili_only" and not Path(item["id"]).exists()
                ]
                if gone:
                    orphans = sync_cli._classify_meili_orphans(
                        qdrant, sync_cli.MetadataBasedSync(qdrant), name, gone
                    )
        except Exception as e:
            parity = {"status": "unknown", "reason": f"Parity scan failed: {e}"}

    return diagnose_corpus(name, presence, verification, parity, orphans)


class _FailedVerification:
    """Stands in for a verification result when the verifier raised."""

    collection_type = None
    files: List[Any] = []

    def __init__(self, error: str):
        self.errors = [error]

    def get_items_needing_repair(self, include_stale_policy: bool = True) -> List[str]:
        return []


_STATUS_STYLE = {
    "ok": ("green", "✓", "ok"),
    "needs_action": ("yellow", "!", "needs action"),
    "error": ("red", "✗", "error"),
}


def _check_all(qdrant, meili, presence_by_name: Dict[str, str], output_json: bool) -> List:
    """Check corpora concurrently; return results in presence_by_name order.

    Each check is dominated by waiting on Qdrant and MeiliSearch scans, so a
    few threads overlap them without contending for the CPU.
    """
    if not output_json:
        console.print(f"[dim]Checking {len(presence_by_name)} corpora...[/dim]")
    workers = max(1, min(DOCTOR_WORKERS, len(presence_by_name)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="corpus-doctor") as pool:
        futures = {
            pool.submit(_check_corpus, qdrant, meili, name, presence): name
            for name, presence in presence_by_name.items()
        }
        results = {}
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
            if not output_json:
                console.print(f"[dim]  checked {name}[/dim]")
    return [results[name] for name in presence_by_name]


def corpus_doctor_command(
    names: tuple,
    verbose: bool,
    output_json: bool,
    strict: bool = False,
):
    """Check every corpus (or the named ones) and list the commands to fix them.

    Args:
        names: Corpus names to check; all corpora when empty
        verbose: Show every finding, including informational ones
        output_json: Output one JSON document
        strict: Exit 1 when any corpus needs action or a check failed
    """
    try:
        qdrant, meili = _open_clients()
        presence_by_name = _discover_corpora(qdrant, meili)
    except Exception as e:
        print_error(f"Cannot check corpora: {e}", output_json)
        sys.exit(ResourceNotFoundError.exit_code)

    if names:
        unknown = [n for n in names if n not in presence_by_name]
        if unknown:
            print_error(f"Corpus not found: {', '.join(unknown)}", output_json)
            sys.exit(ResourceNotFoundError.exit_code)
        presence_by_name = {n: presence_by_name[n] for n in names}

    # The verifier logs a warning per inconsistent file, which would bury the
    # report across a whole sweep; its findings still appear as counts.
    verify_logger = logging.getLogger("arcaneum.indexing.verify")
    previous_level = verify_logger.level
    if not verbose:
        verify_logger.setLevel(logging.ERROR)
    try:
        results = _check_all(qdrant, meili, presence_by_name, output_json)
    finally:
        verify_logger.setLevel(previous_level)

    summary = {
        status: sum(1 for r in results if r["status"] == status)
        for status in ("ok", "needs_action", "error")
    }
    commands = [c for r in results for c in r["commands"]]
    optional_commands = [c for r in results for c in r["optional_commands"]]

    if output_json:
        status = (
            "error" if summary["error"] else "warning" if summary["needs_action"] else "success"
        )
        print_json(
            status,
            f"Checked {len(results)} corpora: {summary['ok']} ok, "
            f"{summary['needs_action']} need action, {summary['error']} errors",
            data={
                "corpora": results,
                "summary": summary,
                "commands": commands,
                "optional_commands": optional_commands,
            },
        )
    else:
        console.print()
        for result in results:
            color, icon, label = _STATUS_STYLE[result["status"]]
            console.print(f"[{color}]{icon} {result['corpus']}: {label}[/{color}]")
            for finding in result["findings"]:
                if finding["severity"] == "info" and not verbose:
                    continue
                console.print(f"    • {finding['message']}")
                if verbose and finding.get("details"):
                    details = ", ".join(f"{k}: {v}" for k, v in finding["details"].items())
                    console.print(f"      [dim]{details}[/dim]")
        console.print(
            f"\n{summary['ok']} ok, {summary['needs_action']} need action, "
            f"{summary['error']} errors"
        )
        if commands:
            console.print("\n[bold]Run (preview each with --dry-run first):[/bold]")
            for command in commands:
                console.print(f"  {command}")
        if optional_commands:
            console.print("\n[bold]Optional, review before running:[/bold]")
            for command in optional_commands:
                console.print(f"  {command}")

    if strict and (summary["needs_action"] or summary["error"]):
        sys.exit(1)

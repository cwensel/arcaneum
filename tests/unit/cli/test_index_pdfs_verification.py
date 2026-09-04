"""PDF verification reporting distinguishes actionable and exhausted failures."""

from io import StringIO

from rich.console import Console

from arcaneum.cli import index_pdfs, sync
from arcaneum.indexing.verify import CollectionVerificationResult, FileVerificationResult


def _result(*files: FileVerificationResult) -> CollectionVerificationResult:
    return CollectionVerificationResult(
        collection_name="Papers",
        collection_type="pdf",
        total_points=len(files),
        total_items=len(files),
        complete_items=0,
        incomplete_items=len(files),
        is_healthy=False,
        files=list(files),
    )


def _capture_report(monkeypatch, result):
    output = StringIO()
    monkeypatch.setattr(
        index_pdfs,
        "console",
        Console(file=output, force_terminal=False, color_system=None),
    )
    breakdown = index_pdfs._report_pdf_verification_issues(result, repair_flag="--force")
    return output.getvalue(), breakdown


def test_exhausted_only_report_has_no_repair_or_missing_claim(monkeypatch):
    exhausted = FileVerificationResult(
        file_path="/papers/exhausted.pdf",
        expected_chunks=1,
        actual_chunks=1,
        is_complete=False,
        fidelity_degraded=True,
        recovery_exhausted=True,
        repair_recommended=False,
    )

    output, breakdown = _capture_report(monkeypatch, _result(exhausted))

    assert "1 unhealthy files" in output
    assert "automatic recovery is exhausted" in output
    assert "0 incomplete" not in output
    assert "no longer exist" not in output
    assert "--force" not in output
    assert breakdown["repairable_files"] == 0
    assert breakdown["degraded_files"] == 1
    assert breakdown["recovery_exhausted_files"] == 1
    assert sync._file_verification_breakdown(_result(exhausted))["repairable_files"] == 0


def test_mixed_report_recommends_repair_only_for_actionable_file(monkeypatch):
    repairable = FileVerificationResult(
        file_path="/papers/incomplete.pdf",
        expected_chunks=2,
        actual_chunks=1,
        is_complete=False,
        repair_recommended=True,
    )
    exhausted = FileVerificationResult(
        file_path="/papers/exhausted.pdf",
        expected_chunks=1,
        actual_chunks=1,
        is_complete=False,
        fidelity_degraded=True,
        recovery_exhausted=True,
        repair_recommended=False,
    )

    output, breakdown = _capture_report(monkeypatch, _result(repairable, exhausted))

    assert "2 unhealthy files" in output
    assert "1 files can be repaired" in output
    assert "/papers/incomplete.pdf" in output
    assert "automatic recovery is exhausted" in output
    assert "--force" in output
    assert breakdown["repairable_files"] == 1
    assert breakdown["degraded_files"] == 1
    assert breakdown["recovery_exhausted_files"] == 1
    sync_breakdown = sync._file_verification_breakdown(_result(repairable, exhausted))
    assert sync_breakdown["repairable_paths"] == ["/papers/incomplete.pdf"]
    assert sync_breakdown["recovery_exhausted_paths"] == ["/papers/exhausted.pdf"]


def test_repair_breakdown_can_defer_policy_only_files():
    policy_only = FileVerificationResult(
        file_path="/papers/legacy.pdf",
        expected_chunks=1,
        actual_chunks=1,
        is_complete=False,
        stale_policy=True,
        repair_recommended=True,
        policy_only_repair=True,
    )
    corrupt_and_stale = FileVerificationResult(
        file_path="/papers/corrupt.pdf",
        expected_chunks=2,
        actual_chunks=1,
        is_complete=False,
        stale_policy=True,
        repair_recommended=True,
        policy_only_repair=False,
    )

    breakdown = sync._file_verification_breakdown(
        _result(policy_only, corrupt_and_stale), include_stale_policy=False
    )

    assert breakdown["repairable_paths"] == ["/papers/corrupt.pdf"]
    assert breakdown["policy_only_paths"] == ["/papers/legacy.pdf"]
    assert breakdown["policy_only_files"] == 1
    legacy_breakdown = index_pdfs._verification_breakdown(
        _result(policy_only, corrupt_and_stale), include_stale_policy=False
    )
    assert legacy_breakdown == breakdown


def test_policy_only_report_defers_force_reindex(monkeypatch):
    policy_only = FileVerificationResult(
        file_path="/papers/legacy.pdf",
        expected_chunks=1,
        actual_chunks=1,
        is_complete=False,
        stale_policy=True,
        repair_recommended=True,
        policy_only_repair=True,
    )

    output, breakdown = _capture_report(monkeypatch, _result(policy_only))

    assert breakdown["repairable_files"] == 0
    assert breakdown["policy_only_files"] == 1
    assert "policy migration is deferred" in output
    assert "Re-run with --force" not in output


def test_report_reuses_precomputed_breakdown(monkeypatch):
    repairable = FileVerificationResult(
        file_path="/papers/incomplete.pdf",
        expected_chunks=2,
        actual_chunks=1,
        is_complete=False,
        repair_recommended=True,
    )
    result = _result(repairable)
    breakdown = index_pdfs._verification_breakdown(result, include_stale_policy=False)
    output = StringIO()
    monkeypatch.setattr(
        index_pdfs,
        "console",
        Console(file=output, force_terminal=False, color_system=None),
    )

    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("breakdown recomputed")

    monkeypatch.setattr(index_pdfs, "_verification_breakdown", fail_if_recomputed)

    returned = index_pdfs._report_pdf_verification_issues(
        result,
        repair_flag="--force",
        breakdown=breakdown,
    )

    assert returned is breakdown
    assert "/papers/incomplete.pdf" in output.getvalue()

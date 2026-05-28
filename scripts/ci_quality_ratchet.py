"""Ratcheted CI quality checks for existing lint and format backlogs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

KNOWN_AUDIT_VULNS = {
    ("transformers", "PYSEC-2025-217"),
    ("transformers", "PYSEC-2025-214"),
    ("transformers", "PYSEC-2025-218"),
    ("transformers", "PYSEC-2025-211"),
    ("transformers", "PYSEC-2025-212"),
    ("transformers", "PYSEC-2025-213"),
    ("transformers", "PYSEC-2025-215"),
    ("transformers", "PYSEC-2025-216"),
    ("transformers", "CVE-2026-1839"),
}


def run_check(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def lint_count() -> tuple[int, str]:
    result = run_check(["ruff", "check", ".", "--output-format", "json"])
    if result.returncode not in (0, 1):
        return -1, result.stderr or result.stdout
    try:
        return len(json.loads(result.stdout or "[]")), result.stderr
    except json.JSONDecodeError as exc:
        return -1, f"{exc}\n{result.stdout}\n{result.stderr}"


def format_count() -> tuple[int, str]:
    result = run_check(["ruff", "format", "--check", "."])
    if result.returncode not in (0, 1):
        return -1, result.stderr or result.stdout
    return len(re.findall(r"^Would reformat:", result.stdout, flags=re.MULTILINE)), result.stderr


def audit_findings() -> tuple[set[tuple[str, str]], str]:
    cache_dir = os.environ.get("PIP_AUDIT_CACHE_DIR", "/tmp/arcaneum-pip-audit-cache")
    result = run_check(
        [
            "pip-audit",
            ".",
            "--skip-editable",
            "--format",
            "json",
            "--desc",
            "off",
            "--aliases",
            "off",
            "--cache-dir",
            cache_dir,
        ]
    )
    if result.returncode not in (0, 1):
        return {("__audit_error__", str(result.returncode))}, result.stderr or result.stdout
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return {("__parse_error__", str(exc))}, f"{exc}\n{result.stdout}\n{result.stderr}"

    findings: set[tuple[str, str]] = set()
    for dependency in data.get("dependencies", []):
        name = dependency.get("name", "").lower()
        for vuln in dependency.get("vulns", []):
            findings.add((name, vuln.get("id", "")))
    return findings, result.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lint-baseline", type=int, required=True)
    parser.add_argument("--format-baseline", type=int, required=True)
    args = parser.parse_args()

    failures: list[str] = []

    lint_total, lint_detail = lint_count()
    print(f"ruff check . violations: {lint_total}/{args.lint_baseline}")
    if lint_detail:
        print(lint_detail, file=sys.stderr)
    if lint_total < 0 or lint_total > args.lint_baseline:
        failures.append("ruff lint violations exceeded the ratchet baseline")

    format_total, format_detail = format_count()
    print(f"ruff format --check . reformat count: {format_total}/{args.format_baseline}")
    if format_detail:
        print(format_detail, file=sys.stderr)
    if format_total < 0 or format_total > args.format_baseline:
        failures.append("ruff format violations exceeded the ratchet baseline")

    audit_findings_total, audit_detail = audit_findings()
    unexpected_audit_findings = audit_findings_total - KNOWN_AUDIT_VULNS
    resolved_audit_findings = KNOWN_AUDIT_VULNS - audit_findings_total
    print(f"pip-audit . vulnerabilities: {len(audit_findings_total)}/{len(KNOWN_AUDIT_VULNS)}")
    if audit_detail:
        print(audit_detail, file=sys.stderr)
    if resolved_audit_findings:
        resolved = ", ".join(f"{name}:{vuln}" for name, vuln in sorted(resolved_audit_findings))
        print(f"resolved audit baseline entries: {resolved}", file=sys.stderr)
        failures.append("pip-audit baseline includes resolved vulnerabilities")
    if unexpected_audit_findings:
        unexpected = ", ".join(f"{name}:{vuln}" for name, vuln in sorted(unexpected_audit_findings))
        print(f"unexpected audit findings: {unexpected}", file=sys.stderr)
        failures.append("pip-audit found unbaselined vulnerabilities")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

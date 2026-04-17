#!/usr/bin/env python3
"""Format release notes from git log, grouped by conventional-commit type.

Usage: format-release-notes.py <range> <prev_tag> <tag> <repo_url>
"""
import re
import subprocess
import sys

TYPE_HEADINGS = [
    ("feat", "Features"),
    ("fix", "Bug Fixes"),
    ("perf", "Performance"),
    ("refactor", "Refactoring"),
    ("docs", "Documentation"),
    ("test", "Tests"),
    ("build", "Build"),
    ("ci", "CI"),
    ("chore", "Chores"),
    ("style", "Style"),
    ("revert", "Reverts"),
]
CONVENTIONAL_TYPES = {t for t, _ in TYPE_HEADINGS}
CC_PATTERN = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]+)\))?(?P<breaking>!)?:\s*(?P<desc>.+)$"
)


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__, file=sys.stderr)
        return 1
    range_spec, prev_tag, tag, repo_url = sys.argv[1:]

    log = subprocess.check_output(
        ["git", "log", "--no-merges", "--pretty=format:%H%x09%s", range_spec],
        text=True,
    )

    grouped: dict[str, list[tuple[str, str, str | None, bool]]] = {
        t: [] for t, _ in TYPE_HEADINGS
    }
    other: list[tuple[str, str]] = []
    breaking_any = False

    for line in log.splitlines():
        if not line.strip():
            continue
        sha, subject = line.split("\t", 1)
        short = sha[:7]
        match = CC_PATTERN.match(subject)
        if match and match["type"] in CONVENTIONAL_TYPES:
            is_breaking = bool(match["breaking"])
            breaking_any = breaking_any or is_breaking
            grouped[match["type"]].append(
                (short, match["desc"], match["scope"], is_breaking)
            )
        else:
            other.append((short, subject))

    lines: list[str] = []

    if breaking_any:
        lines.append("## ⚠ Breaking Changes")
        lines.append("")
        for type_key, _ in TYPE_HEADINGS:
            for short, desc, scope, is_breaking in grouped[type_key]:
                if is_breaking:
                    scope_s = f"**{scope}:** " if scope else ""
                    lines.append(f"- {scope_s}{desc} ({short})")
        lines.append("")

    for type_key, heading in TYPE_HEADINGS:
        entries = grouped[type_key]
        if not entries:
            continue
        lines.append(f"## {heading}")
        lines.append("")
        for short, desc, scope, _ in entries:
            scope_s = f"**{scope}:** " if scope else ""
            lines.append(f"- {scope_s}{desc} ({short})")
        lines.append("")

    if other:
        lines.append("## Other")
        lines.append("")
        for short, subject in other:
            lines.append(f"- {subject} ({short})")
        lines.append("")

    if prev_tag:
        lines.append(f"**Full Changelog**: {repo_url}/compare/{prev_tag}...{tag}")
    else:
        lines.append(f"**Full Changelog**: {repo_url}/commits/{tag}")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

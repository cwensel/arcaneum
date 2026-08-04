#!/usr/bin/env python3
"""Run or compare versioned accelerator benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arcaneum.benchmarks.accelerator import (
    compare_results,
    render_summary,
    run_reference_benchmark,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "fixtures" / "accelerator-v1" / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"))
    args = parser.parse_args()

    if args.compare:
        values = [json.loads(path.read_text(encoding="utf-8")) for path in args.compare]
        print(json.dumps(compare_results(*values), indent=2, sort_keys=True))
        return 0

    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    result = run_reference_benchmark(args.manifest, iterations=args.iterations)
    summary = render_summary(result)
    if args.output:
        write_json(args.output, result)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

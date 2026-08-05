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
    write_result,
)
from arcaneum.benchmarks.coreml import run_coreml_qualification
from arcaneum.benchmarks.cuda import run_cuda_qualification
from arcaneum.benchmarks.mlx import run_mlx_feasibility
from arcaneum.benchmarks.mps import run_mps_qualification

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks" / "fixtures" / "accelerator-v1" / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--backend",
        choices=("reference-cpu", "mps", "cuda", "coreml", "mlx"),
        default="reference-cpu",
    )
    parser.add_argument("--model", default="jina-code-st")
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "arcaneum" / "models"))
    parser.add_argument("--soak-batches", type=int, default=100)
    parser.add_argument("--soak-texts", type=int, default=1500)
    parser.add_argument("--soak-seconds", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--coreml-cache-dir", type=Path, default=Path.home() / ".cache" / "arcaneum" / "coreml"
    )
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"))
    args = parser.parse_args()

    if args.compare:
        values = [json.loads(path.read_text(encoding="utf-8")) for path in args.compare]
        print(json.dumps(compare_results(*values), indent=2, sort_keys=True))
        return 0

    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.token_budget < 1:
        parser.error("--token-budget must be at least 1")
    if args.soak_texts < 0 or args.soak_seconds < 0:
        parser.error("soak targets cannot be negative")
    if args.backend == "mps":
        result = run_mps_qualification(
            args.manifest,
            model=args.model,
            cache_dir=args.cache_dir,
            iterations=args.iterations,
            soak_batches=args.soak_batches,
            batch_size=args.batch_size,
            timeout=args.timeout,
        )
    elif args.backend == "cuda":
        result = run_cuda_qualification(
            args.manifest,
            model=args.model,
            cache_dir=args.cache_dir,
            iterations=args.iterations,
            soak_texts=args.soak_texts,
            soak_seconds=args.soak_seconds,
            batch_size=args.batch_size,
            token_budget=args.token_budget,
            timeout=args.timeout,
        )
    elif args.backend == "coreml":
        result = run_coreml_qualification(
            args.manifest,
            model=args.model,
            cache_dir=args.cache_dir,
            compiled_cache_dir=args.coreml_cache_dir,
            iterations=args.iterations,
            soak_texts=args.soak_texts,
        )
    elif args.backend == "mlx":
        result = run_mlx_feasibility(args.manifest, cache_dir=args.cache_dir)
    else:
        result = run_reference_benchmark(args.manifest, iterations=args.iterations)
    summary = render_summary(result)
    if args.output:
        write_result(args.output, result)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

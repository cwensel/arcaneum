#!/usr/bin/env python3
"""Compare persistent and restart-per-document PDF layout worker execution."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from arcaneum.indexing.pdf.layout_worker import LayoutRequest, PDFLayoutWorker


def _request(pdf_path: Path, layout: bool) -> LayoutRequest:
    return LayoutRequest(
        pdf_path=str(pdf_path.resolve()),
        layout=layout,
        ignore_images=True,
        preserve_images=False,
        use_ocr=False,
    )


def _persistent(requests: list[LayoutRequest]) -> dict:
    timings = []
    pids = []
    started = time.perf_counter()
    with PDFLayoutWorker() as worker:
        for request in requests:
            request_started = time.perf_counter()
            result = worker.convert(request)
            timings.append(time.perf_counter() - request_started)
            pids.append(result["worker_pid"])
        generation = worker.generation
    return {
        "total_seconds": time.perf_counter() - started,
        "document_seconds": timings,
        "median_document_seconds": statistics.median(timings),
        "worker_pids": sorted(set(pids)),
        "worker_generations": generation,
    }


def _restart_each(requests: list[LayoutRequest]) -> dict:
    timings = []
    pids = []
    started = time.perf_counter()
    for request in requests:
        request_started = time.perf_counter()
        with PDFLayoutWorker() as worker:
            result = worker.convert(request)
            pids.append(result["worker_pid"])
        timings.append(time.perf_counter() - request_started)
    return {
        "total_seconds": time.perf_counter() - started,
        "document_seconds": timings,
        "median_document_seconds": statistics.median(timings),
        "worker_pids": sorted(set(pids)),
        "worker_generations": len(pids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="+", type=Path)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--layout", choices=("on", "off"), default="on")
    args = parser.parse_args()
    missing = [str(path) for path in args.pdf if not path.is_file()]
    if missing:
        parser.error(f"PDFs do not exist: {', '.join(missing)}")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    requests = [
        _request(pdf_path, args.layout == "on")
        for _ in range(args.iterations)
        for pdf_path in args.pdf
    ]
    persistent = _persistent(requests)
    restart_each = _restart_each(requests)
    report = {
        "documents": len(requests),
        "layout": args.layout,
        "persistent": persistent,
        "restart_each": restart_each,
        "persistent_speedup": (
            restart_each["total_seconds"] / persistent["total_seconds"]
            if persistent["total_seconds"]
            else None
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

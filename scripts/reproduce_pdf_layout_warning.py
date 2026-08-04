#!/usr/bin/env python3
"""Reproduce PyMuPDF layout teardown diagnostics in an isolated process.

This script deliberately imports no Arcaneum embedding modules. The parent
captures the child's stdout and stderr verbatim and emits one JSON report that
is suitable for an upstream issue attachment.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _child(pdf_path: Path, layout: bool) -> int:
    import pymupdf4llm

    pymupdf4llm.use_layout(layout)
    pages = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        ignore_images=True,
        write_images=False,
        force_text=True,
        table_strategy="lines_strict",
        use_ocr=False,
    )
    payload = {
        "layout_enabled": bool(getattr(pymupdf4llm, "_use_layout", False)),
        "page_count": len(pages),
        "text_characters": sum(len(page["text"]) for page in pages),
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pymupdf": _version("PyMuPDF"),
            "pymupdf4llm": _version("pymupdf4llm"),
            "pymupdf-layout": _version("pymupdf-layout"),
            "torch": _version("torch"),
        },
    }
    print("ARCANEUM_LAYOUT_REPRO=" + json.dumps(payload, sort_keys=True))
    return 0


def _parent(pdf_path: Path, layout: bool) -> int:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(pdf_path.resolve()),
        "--layout",
        "on" if layout else "off",
        "--child",
    ]
    started = time.perf_counter()
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    report = {
        "command": command,
        "duration_seconds": round(time.perf_counter() - started, 6),
        "exit_code": result.returncode,
        "layout_requested": layout,
        "pdf": str(pdf_path.resolve()),
        "stderr": result.stderr,
        "stdout": result.stdout,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--layout", choices=("on", "off"), default="on")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error(f"PDF does not exist: {args.pdf}")
    layout = args.layout == "on"
    return _child(args.pdf, layout) if args.child else _parent(args.pdf, layout)


if __name__ == "__main__":
    raise SystemExit(main())

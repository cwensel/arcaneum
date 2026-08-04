"""Real-process coverage for the PDF layout warning reproducer."""

import json
import subprocess
import sys
from pathlib import Path

import pymupdf
import pytest


@pytest.mark.parametrize("layout", ["on", "off"])
def test_pdf_layout_reproducer_captures_process_streams(tmp_path, layout):
    pdf_path = tmp_path / "two-column.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Left column heading")
        page.insert_text((320, 72), "Right column heading")
        page.insert_text((72, 100), "A small real-document layout fixture.")
        document.save(pdf_path)

    script = Path(__file__).parents[2] / "scripts" / "reproduce_pdf_layout_warning.py"
    result = subprocess.run(
        [sys.executable, str(script), str(pdf_path), "--layout", layout],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["exit_code"] == 0
    assert report["layout_requested"] is (layout == "on")
    assert "ARCANEUM_LAYOUT_REPRO=" in report["stdout"]
    child = json.loads(report["stdout"].split("ARCANEUM_LAYOUT_REPRO=", 1)[1])
    assert child["layout_enabled"] is (layout == "on")
    assert child["page_count"] == 1
    assert child["text_characters"] > 0
    assert child["versions"]["pymupdf4llm"] != "not-installed"

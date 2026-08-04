"""Real spawned-worker coverage using a generated PDF."""

from __future__ import annotations

import sys

import pymupdf

from arcaneum.indexing.pdf.layout_worker import LayoutRequest, PDFLayoutWorker


def test_real_worker_reuses_child_and_returns_plain_page_data(tmp_path):
    pdf_path = tmp_path / "two-pages.pdf"
    document = pymupdf.open()
    for text in ("First page", "Second page"):
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(pdf_path)
    document.close()

    request = LayoutRequest(
        pdf_path=str(pdf_path),
        layout=False,
        ignore_images=True,
        preserve_images=False,
        use_ocr=False,
    )
    # Importing the parent protocol must not initialize the native layout stack.
    assert "pymupdf4llm" not in sys.modules

    with PDFLayoutWorker(timeout_seconds=30) as worker:
        first = worker.convert(request)
        second = worker.convert(request)

        assert first["worker_pid"] == second["worker_pid"] == worker.pid
        assert worker.generation == 1
        assert worker.completed_requests == 2
        assert first["page_count"] == 2
        assert [page["metadata"]["page_number"] for page in first["pages"]] == [1, 2]
        assert all(isinstance(page["text"], str) for page in first["pages"])

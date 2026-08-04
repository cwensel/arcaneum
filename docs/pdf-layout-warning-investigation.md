# PyMuPDF Layout Teardown Warning Investigation

## Classification

The `PyInterpreter.cpp` diagnostics about deallocating a tensor or untyped
storage with live Python references originate in the PyTorch-backed
`pymupdf-layout` analyzer during PDF extraction. They occur before Arcaneum
embedding and can be reproduced by calling `pymupdf4llm.to_markdown()` in a
fresh process that imports no Arcaneum embedding service.

This classifies the warning as an upstream native-object teardown/lifetime
issue in the PDF layout path, not an embedding accelerator failure. Parsing a
document once with `page_chunks=True` (commit `4be6178`) removes the repeated
per-page analyzer construction introduced after `8e2b143`; it reduces the
number and cost of teardown events but cannot guarantee that the upstream
analyzer emits no diagnostic when the document-level analyzer is destroyed.

## Reproduction

Run both modes against the same affected document:

```console
python scripts/reproduce_pdf_layout_warning.py document.pdf --layout on > layout-on.json
python scripts/reproduce_pdf_layout_warning.py document.pdf --layout off > layout-off.json
```

The script launches a clean child process, directly selects PyMuPDF4LLM layout
mode, performs one whole-document conversion, and captures the exact child
stdout, stderr, exit status, duration, Python/platform details, and installed
versions of PyMuPDF, PyMuPDF4LLM, pymupdf-layout, and Torch. It does not import
Arcaneum's embedding modules or contact an embedding service.

PyMuPDF4LLM's supported `use_layout(False)` function sets its process-global
layout policy to false and clears `pymupdf._get_layout`; `use_layout(True)`
imports `pymupdf.layout` and calls `pymupdf.layout.activate()`. Arcaneum's
`PDFExtractor(use_layout_analysis=False)` previously changed only metadata and
did not call this function. It now selects the requested mode around the
conversion, serializes access to that process-global setting, and restores the
prior setting afterward.

## Quality and performance comparison

Do not decide whether to disable layout from one warning or one PDF. Use a
redistributable corpus containing single- and multi-column prose, tables,
headers/footers, lists, figures, scanned pages, and long documents. Run both
modes in fresh processes and record:

- cold and warm wall time, peak RSS, completion/failure count, and stderr;
- page count, output characters/tokens, heading/list/table preservation, and
  reading order;
- task-level retrieval relevance using identical chunking and embeddings.

Review a blinded sample for reading order and structure, and set explicit
acceptance thresholds before choosing a default. A warning-free run alone is
not evidence that non-layout extraction preserves acceptable quality.

## Upstream-ready report

Attach the smallest redistributable PDF that reproduces the warning plus the
two JSON reports. Include the complete stderr (without filtering), exact
versions and platform from the reports, whether the warning occurs with layout
off, repeat count, process exit status, and whether output remains usable.
State that the reproducer makes one `page_chunks=True` call and uses no
SentenceTransformers or Arcaneum embedding code. If the document cannot be
shared, reduce it while retaining the warning and describe the reduction.

Until upstream teardown is proven safe in a soak test, treat the diagnostic as
a PDF layout reliability signal. Suppressing stderr would hide evidence and is
not a resolution. If it correlates with crashes, corruption, or accumulating
memory, isolate layout extraction in the dedicated process proposed by kata
`4x5y`.

## Runtime containment

Arcaneum now performs PyMuPDF4LLM conversion in one persistent spawned process.
The child exclusively imports and owns pymupdf4llm, pymupdf-layout, and their
PyTorch state; the parent receives only page text, page metadata, and structured
errors. Native parser output remains in the child. A timeout or crash terminates
and joins that child before normalized, non-layout PyMuPDF extraction begins.
The next document starts a clean replacement worker.

Use the benchmark harness to quantify model-startup amortization on a
representative corpus:

```console
PYTHONPATH=src python scripts/benchmark_pdf_layout_worker.py \
  document-a.pdf document-b.pdf --iterations 3 --layout on
```

The JSON report compares one persistent worker with restarting a worker for
every document and records total time, per-document time, process identities,
and the observed persistence speedup. Keep results tied to the input corpus and
dependency/platform versions; do not generalize one machine's ratio into a
product-wide performance claim.

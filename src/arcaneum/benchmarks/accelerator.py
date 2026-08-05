"""Versioned accelerator benchmark baseline and validated result contract.

The reference backend deliberately performs a deterministic CPU workload instead
of downloading a model.  It exercises the result contract in ordinary CI; real
backend adapters can emit the same contract in later qualification katas.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_VERSION = "1.0.0"
HARNESS_VERSION = "1.0.0"
RESULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "schema"
    / "accelerator-result-v1.schema.json"
)


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    length_class: str
    text: str
    repetitions: int


@lru_cache(maxsize=1)
def load_result_schema() -> dict[str, Any]:
    """Load and meta-validate the canonical Draft 2020-12 result schema."""
    schema = json.loads(RESULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


@lru_cache(maxsize=1)
def _result_validator() -> Draft202012Validator:
    return Draft202012Validator(load_result_schema(), format_checker=FormatChecker())


def _json_path(parts: Any) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def validate_result(value: dict[str, Any], *, label: str = "result") -> None:
    """Raise a path-qualified error when a benchmark result violates the schema."""
    errors = sorted(
        _result_validator().iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{_json_path(error.absolute_path)}: {error.message}" for error in errors
        )
        raise ValueError(f"{label} failed accelerator result schema validation: {details}")


def load_manifest(path: Path) -> tuple[dict[str, Any], list[Fixture]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    fixtures = []
    for item in manifest["fixtures"]:
        text = (root / item["path"]).read_text(encoding="utf-8")
        fixtures.append(Fixture(item["id"], item["length_class"], text, item.get("repetitions", 1)))
    return manifest, fixtures


def manifest_digest(path: Path) -> str:
    manifest, fixtures = load_manifest(path)
    canonical = {
        "manifest_version": manifest["manifest_version"],
        "fixtures": [
            {
                "id": fixture.fixture_id,
                "length_class": fixture.length_class,
                "repetitions": fixture.repetitions,
                "text_sha256": hashlib.sha256(fixture.text.encode()).hexdigest(),
            }
            for fixture in fixtures
        ],
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reference_embed(texts: list[str], dimensions: int) -> np.ndarray:
    """Return stable, normalized vectors with a non-trivial CPU cost."""
    rows = []
    for text in texts:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        raw = hashlib.shake_256(seed + text.encode("utf-8")).digest(dimensions * 4)
        vector = np.frombuffer(raw, dtype="<u4").astype(np.float32)
        vector = (vector / np.float32(2**32 - 1)) * 2 - 1
        vector /= np.linalg.norm(vector)
        rows.append(vector)
    return np.stack(rows)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def run_reference_benchmark(
    manifest_path: Path, *, iterations: int = 3, dimensions: int = 64
) -> dict[str, Any]:
    manifest, fixtures = load_manifest(manifest_path)
    texts = [fixture.text for fixture in fixtures for _ in range(fixture.repetitions)]
    process = psutil.Process()
    peak_rss = process.memory_info().rss

    cold_start = time.perf_counter()
    reference = _reference_embed(texts, dimensions)
    cold_seconds = time.perf_counter() - cold_start
    peak_rss = max(peak_rss, process.memory_info().rss)

    latencies = []
    for _ in range(iterations):
        started = time.perf_counter()
        actual = _reference_embed(texts, dimensions)
        latencies.append(time.perf_counter() - started)
        peak_rss = max(peak_rss, process.memory_info().rss)

    differences = np.abs(reference - actual)
    warm_seconds = sum(latencies)
    return {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "run": {
            "id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-reference-cpu"),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
        },
        "environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "cpu": platform.processor() or "unknown",
            "logical_cpu_count": os.cpu_count(),
            "memory_bytes": psutil.virtual_memory().total,
            "dependencies": {"numpy": np.__version__, "psutil": psutil.__version__},
        },
        "backend": {
            "name": "reference-cpu",
            "device": "cpu",
            "model": "deterministic-shake256",
            "precision": "float32",
            "dimensions": dimensions,
        },
        "fixture": {
            "manifest_version": manifest["manifest_version"],
            "manifest_sha256": manifest_digest(manifest_path),
            "length_classes": sorted({fixture.length_class for fixture in fixtures}),
            "unique_texts": len(fixtures),
            "total_texts": len(texts),
        },
        "performance": {
            "cold_start_seconds": cold_seconds,
            "warm_total_seconds": warm_seconds,
            "warm_iterations": iterations,
            "throughput_texts_per_second": len(texts) * iterations / warm_seconds,
            "latency_seconds": {
                "samples": latencies,
                "p50": statistics.median(latencies),
                "p95": _percentile(latencies, 0.95),
            },
            "peak_rss_bytes": peak_rss,
        },
        "reliability": {
            "attempted_batches": iterations + 1,
            "completed_batches": iterations + 1,
            "failures": 0,
            "fallbacks": 0,
            "restarts": 0,
        },
        "correctness": {
            "reference_backend": "reference-cpu",
            "shape": list(actual.shape),
            "finite": bool(np.isfinite(actual).all()),
            "max_absolute_error": float(differences.max()),
            "mean_absolute_error": float(differences.mean()),
            "minimum_cosine_similarity": 1.0,
            "passed": bool(np.array_equal(reference, actual)),
        },
    }


def compare_results(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    validate_result(baseline, label="baseline")
    validate_result(candidate, label="candidate")
    if baseline["schema_version"] != candidate["schema_version"]:
        raise ValueError("result schema versions differ")
    if baseline["fixture"]["manifest_sha256"] != candidate["fixture"]["manifest_sha256"]:
        raise ValueError("fixture manifests differ")
    for field in ("model", "precision", "dimensions"):
        if baseline["backend"][field] != candidate["backend"][field]:
            raise ValueError(f"backend {field} differs")
    base_rate = baseline["performance"]["throughput_texts_per_second"]
    candidate_rate = candidate["performance"]["throughput_texts_per_second"]
    return {
        "schema_version": baseline["schema_version"],
        "baseline": baseline["backend"]["name"],
        "candidate": candidate["backend"]["name"],
        "speedup": candidate_rate / base_rate,
        "correctness_passed": candidate["correctness"]["passed"],
        "candidate_failures": candidate["reliability"]["failures"],
        "candidate_fallbacks": candidate["reliability"]["fallbacks"],
        "candidate_restarts": candidate["reliability"]["restarts"],
    }


def render_summary(result: dict[str, Any]) -> str:
    perf = result["performance"]
    reliable = result["reliability"]
    correct = result["correctness"]
    return "\n".join(
        [
            f"Accelerator benchmark schema {result['schema_version']}",
            f"Backend: {result['backend']['name']} ({result['backend']['device']})",
            f"Fixture: {result['fixture']['total_texts']} texts; "
            f"classes={','.join(result['fixture']['length_classes'])}",
            f"Cold start: {perf['cold_start_seconds']:.6f}s",
            f"Warm throughput: {perf['throughput_texts_per_second']:.2f} texts/s",
            f"Warm latency p50/p95: {perf['latency_seconds']['p50']:.6f}s / "
            f"{perf['latency_seconds']['p95']:.6f}s",
            f"Peak RSS: {perf['peak_rss_bytes']} bytes",
            f"Reliability: failures={reliable['failures']} fallbacks={reliable['fallbacks']} "
            f"restarts={reliable['restarts']}",
            f"Correctness: {'PASS' if correct['passed'] else 'FAIL'} "
            f"max_abs_error={correct['max_absolute_error']:.3g}",
        ]
    )


def write_result(path: Path, value: dict[str, Any]) -> None:
    """Atomically persist one schema-valid result; invalid data never touches disk."""
    validate_result(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Backward-compatible name for :func:`write_result`."""
    write_result(path, value)

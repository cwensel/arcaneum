"""Opt-in CoreML/FastEmbed qualification with compiled-model cache evidence."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil

from arcaneum.benchmarks.accelerator import (
    HARNESS_VERSION,
    SCHEMA_VERSION,
    load_manifest,
    manifest_digest,
)
from arcaneum.embeddings.client import EMBEDDING_MODELS, _write_coreml_sentinel

SPEEDUP_GATE = 1.25
SOAK_TARGET_TEXTS = 100_000
SOAK_TARGET_SECONDS = 3 * 60 * 60
DEFAULT_BUCKETS = (1, 2, 4, 8, 16, 32)


def coreml_provider_options(
    cache_dir: Path, *, profile_compute_plan: bool = True
) -> dict[str, str]:
    options = {
        "ModelFormat": "MLProgram",
        "RequireStaticInputShapes": "1",
        "ModelCacheDirectory": str(cache_dir.expanduser().resolve()),
        "SpecializationStrategy": "FastPrediction",
    }
    if profile_compute_plan:
        options["ProfileComputePlan"] = "1"
    return options


def route_bucketed(
    texts: list[str],
    encode: Callable[[list[str]], np.ndarray],
    buckets: tuple[int, ...] = DEFAULT_BUCKETS,
) -> tuple[np.ndarray, list[int]]:
    """Encode fixed-count buckets and restore order; padding rows are discarded."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32), []
    ordered = sorted(enumerate(texts), key=lambda item: len(item[1]), reverse=True)
    rows: list[np.ndarray | None] = [None] * len(texts)
    used, offset = [], 0
    while offset < len(ordered):
        remaining = len(ordered) - offset
        bucket = next((value for value in buckets if value >= remaining), buckets[-1])
        take = min(bucket, remaining)
        chunk = ordered[offset : offset + take]
        padded = [text for _, text in chunk] + [chunk[-1][1]] * (bucket - take)
        encoded = np.asarray(encode(padded), dtype=np.float32)
        if encoded.ndim != 2 or encoded.shape[0] != bucket:
            raise RuntimeError(f"CoreML bucket {bucket} returned shape {encoded.shape}")
        for (index, _), vector in zip(chunk, encoded[:take], strict=True):
            rows[index] = np.array(vector, dtype=np.float32, copy=True)
        used.append(bucket)
        offset += take
    return np.stack(rows), used  # type: ignore[arg-type]


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _fixture(path: Path) -> dict[str, Any]:
    manifest, fixtures = load_manifest(path)
    return {
        "manifest_version": manifest["manifest_version"],
        "manifest_sha256": manifest_digest(path),
        "length_classes": sorted({f.length_class for f in fixtures}),
        "unique_texts": len(fixtures),
        "total_texts": sum(f.repetitions for f in fixtures),
    }


def _empty(path: Path, model: str, reason: str, compiled_cache: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "run": {
            "id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-coreml"),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "status": "inconclusive",
        },
        "environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "cpu": platform.processor() or "unknown",
            "logical_cpu_count": os.cpu_count(),
            "memory_bytes": psutil.virtual_memory().total,
            "dependencies": {
                n: _version(n) for n in ("numpy", "psutil", "onnxruntime", "fastembed")
            },
        },
        "backend": {
            "name": "onnxruntime-coreml",
            "device": "coreml",
            "model": model,
            "precision": "float32",
            "dimensions": 0,
        },
        "fixture": _fixture(path),
        "performance": {
            "cold_start_seconds": 0.0,
            "warm_total_seconds": 0.0,
            "warm_iterations": 0,
            "throughput_texts_per_second": 0.0,
            "latency_seconds": {"samples": [], "p50": 0.0, "p95": 0.0},
            "peak_rss_bytes": psutil.Process().memory_info().rss,
            "compiled_cache_reused": False,
        },
        "reliability": {
            "attempted_batches": 0,
            "completed_batches": 0,
            "failures": 1,
            "fallbacks": 0,
            "restarts": 0,
        },
        "correctness": {
            "reference_backend": "onnxruntime-cpu",
            "shape": [0, 0],
            "finite": False,
            "max_absolute_error": 0.0,
            "mean_absolute_error": 0.0,
            "minimum_cosine_similarity": 0.0,
            "passed": False,
        },
        "qualification": {
            "decision": "experimental",
            "reason": reason,
            "speedup_gate": SPEEDUP_GATE,
            "soak_target_batches": None,
            "soak_target_texts": SOAK_TARGET_TEXTS,
            "soak_target_seconds": SOAK_TARGET_SECONDS,
            "provider_options": coreml_provider_options(compiled_cache),
            "provider_placement": "unknown",
            "static_shape_scope": "batch-count buckets; tokenizer sequence shape is dynamic",
            "exact_rerun": (
                "ARC_RUN_COREML_QUALIFICATION=1 PYTHONPATH=$PWD/src python "
                "scripts/benchmark_accelerators.py --backend coreml --model bge-small "
                "--output benchmarks/results/coreml-local.json"
            ),
        },
    }


def _providers(model: Any) -> list[str]:
    seen, stack, visited = [], [model], set()
    while stack:
        value = stack.pop()
        if id(value) in visited:
            continue
        visited.add(id(value))
        getter = getattr(value, "get_providers", None)
        if callable(getter):
            seen.extend(map(str, getter()))
        for name in ("model", "onnx_model", "session", "_session"):
            child = getattr(value, name, None)
            if child is not None:
                stack.append(child)
    return list(dict.fromkeys(seen))


def run_coreml_qualification(
    manifest_path: Path,
    *,
    model: str,
    cache_dir: str,
    compiled_cache_dir: Path,
    iterations: int = 3,
    soak_texts: int = 0,
) -> dict[str, Any]:
    result = _empty(manifest_path, model, "CoreML qualification not run", compiled_cache_dir)
    if os.environ.get("ARC_RUN_COREML_QUALIFICATION") != "1":
        result["qualification"]["reason"] = (
            "set ARC_RUN_COREML_QUALIFICATION=1; hardware benchmark is opt-in"
        )
        return result
    try:
        import onnxruntime as ort
        from fastembed import TextEmbedding

        if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
            raise RuntimeError("CoreML qualification requires Apple Silicon macOS")
        if "CoreMLExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CoreMLExecutionProvider is unavailable in this onnxruntime build")
        config = EMBEDDING_MODELS.get(model)
        if not config or config.get("backend") != "fastembed":
            raise RuntimeError(f"{model!r} is not a strategic FastEmbed model")
        compiled_cache_dir.mkdir(parents=True, exist_ok=True)
        before = {p.relative_to(compiled_cache_dir) for p in compiled_cache_dir.rglob("*")}
        texts = [f.text for f in load_manifest(manifest_path)[1] for _ in range(f.repetitions)]
        tick = time.perf_counter()
        cpu = TextEmbedding(
            config["name"],
            cache_dir=cache_dir,
            local_files_only=True,
            providers=["CPUExecutionProvider"],
        )
        cpu_load = time.perf_counter() - tick
        reference, _ = route_bucketed(texts, lambda batch: np.stack(list(cpu.embed(batch))))
        _write_coreml_sentinel(model)
        tick = time.perf_counter()
        coreml = TextEmbedding(
            config["name"],
            cache_dir=cache_dir,
            local_files_only=True,
            providers=[
                ("CoreMLExecutionProvider", coreml_provider_options(compiled_cache_dir)),
                "CPUExecutionProvider",
            ],
        )
        actual, used = route_bucketed(texts, lambda batch: np.stack(list(coreml.embed(batch))))
        cold = time.perf_counter() - tick
        latencies = []
        for _ in range(iterations):
            tick = time.perf_counter()
            actual, used = route_bucketed(texts, lambda b: np.stack(list(coreml.embed(b))))
            latencies.append(time.perf_counter() - tick)
        completed_soak = 0
        while completed_soak < soak_texts:
            route_bucketed(texts, lambda b: np.stack(list(coreml.embed(b))))
            completed_soak += len(texts)
        provider_list = _providers(coreml)
        placement = (
            "hybrid-coreml-cpu"
            if {"CoreMLExecutionProvider", "CPUExecutionProvider"} <= set(provider_list)
            else "coreml"
            if provider_list == ["CoreMLExecutionProvider"]
            else "unknown"
        )
        diff = np.abs(reference - actual)
        cosine = np.sum(reference * actual, axis=1) / np.maximum(
            np.linalg.norm(reference, axis=1) * np.linalg.norm(actual, axis=1),
            np.finfo(np.float32).eps,
        )
        tick = time.perf_counter()
        route_bucketed(texts, lambda b: np.stack(list(cpu.embed(b))))
        cpu_warm = time.perf_counter() - tick
        speedup, correct = (
            cpu_warm / statistics.mean(latencies),
            bool(np.isfinite(actual).all() and cosine.min() >= 0.999),
        )
        after = {p.relative_to(compiled_cache_dir) for p in compiled_cache_dir.rglob("*")}
        decision = (
            "qualified"
            if speedup >= SPEEDUP_GATE
            and correct
            and completed_soak >= SOAK_TARGET_TEXTS
            and placement == "coreml"
            else "experimental"
        )
        result["run"]["status"] = "completed"
        result["backend"]["dimensions"] = actual.shape[1]
        result["performance"].update(
            {
                "cold_start_seconds": cold,
                "warm_total_seconds": sum(latencies),
                "warm_iterations": iterations,
                "throughput_texts_per_second": len(texts) * iterations / sum(latencies),
                "latency_seconds": {
                    "samples": latencies,
                    "p50": statistics.median(latencies),
                    "p95": max(latencies),
                },
                "cpu_cold_start_seconds": cpu_load,
                "cpu_warm_seconds": cpu_warm,
                "speedup_over_cpu": speedup,
                "compiled_cache_reused": bool(before),
                "compiled_cache_entries_created": len(after - before),
            }
        )
        result["reliability"].update(
            {
                "attempted_batches": iterations + 2,
                "completed_batches": iterations + 2,
                "failures": 0,
                "completed_soak_texts": completed_soak,
            }
        )
        result["correctness"] = {
            "reference_backend": "onnxruntime-cpu",
            "shape": list(actual.shape),
            "finite": bool(np.isfinite(actual).all()),
            "max_absolute_error": float(diff.max()),
            "mean_absolute_error": float(diff.mean()),
            "minimum_cosine_similarity": float(cosine.min()),
            "passed": correct,
        }
        result["qualification"].update(
            {
                "decision": decision,
                "provider_placement": placement,
                "providers_reported": provider_list,
                "buckets_used": used,
                "reason": (
                    f"speedup={speedup:.3f}; numerical_pass={correct}; "
                    f"soak_texts={completed_soak}/{SOAK_TARGET_TEXTS}; "
                    f"placement={placement}"
                ),
            }
        )
        return result
    except Exception as exc:
        result["qualification"]["reason"] = f"{type(exc).__name__}: {exc}"
        return result

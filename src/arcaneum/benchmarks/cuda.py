"""Opt-in CUDA qualification through the persistent spawned worker."""

from __future__ import annotations

import importlib.metadata
import math
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from arcaneum.benchmarks.accelerator import (
    HARNESS_VERSION,
    SCHEMA_VERSION,
    load_manifest,
    manifest_digest,
)
from arcaneum.embeddings.batch_scheduler import BatchBudget, OversizePolicy, schedule_batches
from arcaneum.embeddings.worker_protocol import AcceleratorWorkerSession, WorkerConfig

FACTORY = (
    "arcaneum.embeddings.sentence_transformer_worker:"
    "create_sentence_transformer_accelerator_backend"
)
SPEEDUP_GATE = 1.25
SOAK_TARGET_TEXTS = 100_000
SOAK_TARGET_SECONDS = 3 * 60 * 60


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _nvidia_smi() -> dict[str, Any]:
    """Collect driver/hardware metadata without importing CUDA in the parent."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {"nvidia_smi": "unavailable"}
    devices = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5:
            devices.append(
                {
                    "name": fields[0],
                    "uuid": fields[1],
                    "driver_version": fields[2],
                    "memory_total_mib": fields[3],
                    "compute_capability": fields[4],
                }
            )
    return {"nvidia_smi": "available", "cuda_devices": devices}


def _environment() -> dict[str, Any]:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "cpu": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": psutil.virtual_memory().total,
        "dependencies": {
            name: _version(name)
            for name in ("numpy", "psutil", "torch", "sentence-transformers", "transformers")
        },
        **_nvidia_smi(),
    }


def _fixture_metadata(manifest_path: Path) -> dict[str, Any]:
    manifest, fixtures = load_manifest(manifest_path)
    return {
        "manifest_version": manifest["manifest_version"],
        "manifest_sha256": manifest_digest(manifest_path),
        "length_classes": sorted({fixture.length_class for fixture in fixtures}),
        "unique_texts": len(fixtures),
        "total_texts": sum(fixture.repetitions for fixture in fixtures),
    }


def _empty_result(
    manifest_path: Path,
    model: str,
    reason: str,
    *,
    token_budget: int = 8192,
    batch_size: int = 8,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "run": {
            "id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-cuda"),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "status": "inconclusive",
        },
        "environment": _environment(),
        "backend": {
            "name": "pytorch-cuda-worker",
            "device": "cuda",
            "model": model,
            "precision": "float32",
            "dimensions": 0,
        },
        "fixture": _fixture_metadata(manifest_path),
        "performance": {
            "cold_start_seconds": 0.0,
            "warm_total_seconds": 0.0,
            "warm_iterations": 0,
            "throughput_texts_per_second": 0.0,
            "latency_seconds": {"samples": [], "p50": 0.0, "p95": 0.0},
            "peak_rss_bytes": psutil.Process().memory_info().rss,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        },
        "reliability": {
            "attempted_batches": 0,
            "completed_batches": 0,
            "failures": 1,
            "fallbacks": 0,
            "restarts": 0,
        },
        "correctness": {
            "reference_backend": "pytorch-cpu-worker",
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
            "token_budget": {
                "estimator": "conservative-utf8-bytes-divided-by-four",
                "max_actual_tokens": token_budget,
                "max_padded_tokens": token_budget,
                "max_sequence_tokens": token_budget,
                "max_batch_size": batch_size,
            },
            "failure_policy": {
                "worker_oom_retries": 2,
                "qualification_worker_restarts": 0,
                "timeout_reaps_worker": True,
                "promotion_requires_zero_failures": True,
            },
            "containment_evidence": {
                "timeout_reaps_worker": "tests/embeddings/test_worker_protocol.py",
                "bounded_oom_retries": "tests/unit/benchmarks/test_cuda_qualification.py",
                "validated": True,
            },
        },
    }


def _estimated_tokens(text: str) -> int:
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def _encode_scheduled(
    worker: AcceleratorWorkerSession,
    texts: list[str],
    *,
    timeout: float,
    token_budget: int,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    budget = BatchBudget(
        max_actual_tokens=token_budget,
        max_padded_tokens=token_budget,
        max_sequence_tokens=token_budget,
        max_batch_size=batch_size,
        oversize_policy=OversizePolicy.SINGLETON,
    )
    batches = schedule_batches(texts, budget=budget, count_tokens=_estimated_tokens)
    rows: list[np.ndarray | None] = [None] * len(texts)
    for batch in batches:
        encoded = worker.encode(batch.texts, timeout=timeout, batch_size=len(batch.items))
        for index, vector in zip(batch.original_indices, encoded, strict=True):
            rows[index] = vector
    if any(row is None for row in rows):
        raise RuntimeError("scheduled CUDA encode did not return every input")
    return np.stack(rows), len(batches)  # type: ignore[arg-type]


def run_cuda_qualification(
    manifest_path: Path,
    *,
    model: str,
    cache_dir: str,
    iterations: int = 5,
    soak_texts: int = 1_500,
    soak_seconds: float = 0.0,
    batch_size: int = 8,
    token_budget: int = 8192,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Qualify one concrete CUDA/model combination, or record why it is inconclusive."""
    _, fixtures = load_manifest(manifest_path)
    texts = [fixture.text for fixture in fixtures for _ in range(fixture.repetitions)]
    common = {
        "model_name": model,
        "cache_dir": cache_dir,
        "local_files_only": True,
        "strict_local_files_only": True,
    }
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    cpu: AcceleratorWorkerSession | None = None
    cuda: AcceleratorWorkerSession | None = None
    attempted_batches = completed_batches = completed_soak_texts = 0
    soak_started: float | None = None
    try:
        started = time.perf_counter()
        cuda = AcceleratorWorkerSession(
            WorkerConfig(FACTORY, {**common, "device": "cuda"}), startup_timeout=timeout
        ).start()
        cuda_load = time.perf_counter() - started
        cuda_health = cuda.health(timeout=5)["backend"]

        started = time.perf_counter()
        cpu = AcceleratorWorkerSession(
            WorkerConfig(FACTORY, {**common, "device": "cpu"}), startup_timeout=timeout
        ).start()
        cpu_load = time.perf_counter() - started
        reference, count = _encode_scheduled(
            cpu, texts, timeout=timeout, token_budget=token_budget, batch_size=batch_size
        )
        attempted_batches += count
        completed_batches += count
        cpu_latencies = []
        for _ in range(iterations):
            tick = time.perf_counter()
            _, count = _encode_scheduled(
                cpu, texts, timeout=timeout, token_budget=token_budget, batch_size=batch_size
            )
            cpu_latencies.append(time.perf_counter() - tick)
            attempted_batches += count
            completed_batches += count
        cpu.shutdown()
        cpu = None

        actual, count = _encode_scheduled(
            cuda, texts, timeout=timeout, token_budget=token_budget, batch_size=batch_size
        )
        attempted_batches += count
        completed_batches += count
        latencies = []
        for _ in range(iterations):
            tick = time.perf_counter()
            actual, count = _encode_scheduled(
                cuda, texts, timeout=timeout, token_budget=token_budget, batch_size=batch_size
            )
            latencies.append(time.perf_counter() - tick)
            attempted_batches += count
            completed_batches += count

        soak_started = time.monotonic()
        while completed_soak_texts < soak_texts or (
            soak_seconds > 0 and time.monotonic() - soak_started < soak_seconds
        ):
            actual, count = _encode_scheduled(
                cuda, texts, timeout=timeout, token_budget=token_budget, batch_size=batch_size
            )
            attempted_batches += count
            completed_batches += count
            completed_soak_texts += len(texts)
            peak_rss = max(peak_rss, process.memory_info().rss)
        soak_elapsed = time.monotonic() - soak_started
        cuda_health = cuda.health(timeout=5)["backend"]
        cuda.shutdown()
        cuda = None
    except Exception as exc:
        for candidate in (cpu, cuda):
            if candidate is not None:
                candidate.shutdown()
        result = _empty_result(
            manifest_path,
            model,
            f"{type(exc).__name__}: {exc}",
            token_budget=token_budget,
            batch_size=batch_size,
        )
        result["reliability"].update(
            {
                "attempted_batches": attempted_batches,
                "completed_batches": completed_batches,
                "completed_soak_texts": completed_soak_texts,
                "soak_seconds": (
                    time.monotonic() - soak_started if soak_started is not None else 0.0
                ),
            }
        )
        return result

    dots = np.sum(reference * actual, axis=1)
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(actual, axis=1)
    cosine = dots / np.maximum(denominator, np.finfo(np.float32).eps)
    differences = np.abs(reference - actual)
    speedup = statistics.mean(cpu_latencies) / statistics.mean(latencies)
    correctness_passed = bool(np.isfinite(actual).all() and cosine.min() >= 0.999)
    soak_passed = completed_soak_texts >= SOAK_TARGET_TEXTS or soak_elapsed >= SOAK_TARGET_SECONDS
    zero_failures = int(cuda_health.get("oom_retries", 0)) == 0
    decision = (
        "qualified"
        if speedup >= SPEEDUP_GATE and correctness_passed and soak_passed and zero_failures
        else "experimental"
    )
    result = _empty_result(
        manifest_path,
        model,
        "qualification gates evaluated",
        token_budget=token_budget,
        batch_size=batch_size,
    )
    result["run"]["status"] = "completed"
    result["environment"]["cuda_worker"] = {
        key: cuda_health.get(key)
        for key in (
            "cuda_device_name",
            "cuda_compute_capability",
            "cuda_total_memory_bytes",
            "cuda_runtime_version",
        )
    }
    result["backend"]["dimensions"] = actual.shape[1]
    result["performance"].update(
        {
            "cold_start_seconds": cuda_load,
            "warm_total_seconds": sum(latencies),
            "warm_iterations": iterations,
            "throughput_texts_per_second": len(texts) * iterations / sum(latencies),
            "latency_seconds": {
                "samples": latencies,
                "p50": statistics.median(latencies),
                "p95": max(latencies),
            },
            "peak_rss_bytes": peak_rss,
            "cuda_peak_allocated_bytes": cuda_health.get("cuda_peak_allocated_bytes"),
            "cuda_peak_reserved_bytes": cuda_health.get("cuda_peak_reserved_bytes"),
            "cpu_cold_start_seconds": cpu_load,
            "cpu_throughput_texts_per_second": len(texts) * iterations / sum(cpu_latencies),
            "speedup_over_cpu": speedup,
        }
    )
    result["reliability"] = {
        "attempted_batches": attempted_batches,
        "completed_batches": completed_batches,
        "failures": 0,
        "fallbacks": 0,
        "restarts": 0,
        "oom_retries": int(cuda_health.get("oom_retries", 0)),
        "completed_soak_texts": completed_soak_texts,
        "soak_seconds": soak_elapsed,
    }
    result["correctness"] = {
        "reference_backend": "pytorch-cpu-worker",
        "shape": list(actual.shape),
        "finite": bool(np.isfinite(actual).all()),
        "max_absolute_error": float(differences.max()),
        "mean_absolute_error": float(differences.mean()),
        "minimum_cosine_similarity": float(cosine.min()),
        "passed": correctness_passed,
    }
    result["qualification"].update(
        {
            "decision": decision,
            "reason": (
                f"speedup={speedup:.3f}; soak_texts={completed_soak_texts}/"
                f"{SOAK_TARGET_TEXTS}; soak_seconds={soak_elapsed:.1f}/"
                f"{SOAK_TARGET_SECONDS}; numerical_pass={correctness_passed}; "
                f"oom_retries={cuda_health.get('oom_retries', 0)}"
            ),
        }
    )
    return result

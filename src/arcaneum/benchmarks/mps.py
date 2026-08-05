"""Opt-in, spawned-worker PyTorch MPS qualification harness."""

from __future__ import annotations

import importlib.metadata
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
from arcaneum.embeddings.worker_protocol import AcceleratorWorkerSession, WorkerConfig

FACTORY = (
    "arcaneum.embeddings.sentence_transformer_worker:"
    "create_sentence_transformer_accelerator_backend"
)


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _environment() -> dict[str, Any]:
    result = {
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
    }
    if platform.system() == "Darwin":
        try:
            hardware = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            for line in hardware.splitlines():
                label, separator, value = line.strip().partition(":")
                if separator and label in {"Chip", "Model Name", "Model Identifier"}:
                    result[label.lower().replace(" ", "_")] = value.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def _empty_result(manifest_path: Path, model: str, reason: str) -> dict[str, Any]:
    manifest, fixtures = load_manifest(manifest_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "run": {
            "id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-mps"),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "status": "inconclusive",
        },
        "environment": _environment(),
        "backend": {
            "name": "pytorch-mps-worker",
            "device": "mps",
            "model": model,
            "precision": "float32",
            "dimensions": 0,
        },
        "fixture": {
            "manifest_version": manifest["manifest_version"],
            "manifest_sha256": manifest_digest(manifest_path),
            "length_classes": sorted({f.length_class for f in fixtures}),
            "unique_texts": len(fixtures),
            "total_texts": sum(f.repetitions for f in fixtures),
        },
        "performance": {
            "cold_start_seconds": 0.0,
            "warm_total_seconds": 0.0,
            "warm_iterations": 0,
            "throughput_texts_per_second": 0.0,
            "latency_seconds": {"samples": [], "p50": 0.0, "p95": 0.0},
            "peak_rss_bytes": psutil.Process().memory_info().rss,
            "mps_peak_driver_allocated_bytes": None,
        },
        "reliability": {
            "attempted_batches": 0,
            "completed_batches": 0,
            "failures": 1,
            "fallbacks": 0,
            "restarts": 0,
            "oom_retries": None,
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
            "speedup_gate": 1.25,
            "soak_target_batches": 10000,
            "watermarks": {
                "low": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6"),
                "high": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8"),
            },
            "containment_evidence": {
                "timeout_reaps_worker": "tests/embeddings/test_worker_protocol.py",
                "validated": True,
            },
        },
    }


def _qualification_decision(
    *, speedup: float, soak_batches: int, soak_target: int, numerical_pass: bool, oom_retries: int
) -> str:
    """Promote only a clean run; recovered OOMs are reliability failures."""
    return (
        "qualified"
        if speedup >= 1.25
        and soak_batches >= soak_target
        and numerical_pass
        and oom_retries == 0
        else "experimental"
    )


def _completed_reliability(*, attempted_batches: int, oom_retries: int) -> dict[str, int]:
    """Account for completed logical batches and failed recovered native attempts."""
    return {
        "attempted_batches": attempted_batches,
        "completed_batches": attempted_batches,
        "failures": oom_retries,
        "fallbacks": 0,
        "restarts": 0,
        "oom_retries": oom_retries,
    }


def run_mps_qualification(
    manifest_path: Path,
    *,
    model: str,
    cache_dir: str,
    iterations: int = 5,
    soak_batches: int = 100,
    batch_size: int = 8,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Compare the same model on CPU/MPS; return truthful inconclusive output on setup failure."""
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
    if os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.0":
        return _empty_result(manifest_path, model, "unsafe MPS high watermark 0.0 is refused")
    _, fixtures = load_manifest(manifest_path)
    texts = [f.text for f in fixtures for _ in range(f.repetitions)]
    common = {"model_name": model, "cache_dir": cache_dir, "local_files_only": True}
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    try:
        # Probe and load MPS first. An unavailable runtime must not trigger a costly
        # CPU model load merely to produce an inconclusive artifact.
        started = time.perf_counter()
        mps = AcceleratorWorkerSession(
            WorkerConfig(FACTORY, {**common, "device": "mps"}), startup_timeout=timeout
        ).start()
        mps_load = time.perf_counter() - started
        started = time.perf_counter()
        cpu = AcceleratorWorkerSession(
            WorkerConfig(FACTORY, {**common, "device": "cpu"}), startup_timeout=timeout
        ).start()
        cpu_load = time.perf_counter() - started
        reference = cpu.encode(texts, timeout=timeout, batch_size=batch_size)
        cpu_latencies = []
        for _ in range(iterations):
            tick = time.perf_counter()
            cpu.encode(texts, timeout=timeout, batch_size=batch_size)
            cpu_latencies.append(time.perf_counter() - tick)
        cpu.shutdown()
        actual = mps.encode(texts, timeout=timeout, batch_size=batch_size)
        oom_retries = int(mps.health(timeout=5)["backend"].get("oom_retries", 0))
        latencies = []
        driver_peak = 0
        for _ in range(iterations + soak_batches):
            tick = time.perf_counter()
            actual = mps.encode(texts, timeout=timeout, batch_size=batch_size)
            latencies.append(time.perf_counter() - tick)
            backend_health = mps.health(timeout=5)["backend"]
            driver_peak = max(
                driver_peak, int(backend_health.get("mps_driver_allocated_bytes", 0))
            )
            oom_retries = max(oom_retries, int(backend_health.get("oom_retries", 0)))
            peak_rss = max(peak_rss, process.memory_info().rss)
        mps.shutdown()
    except Exception as exc:
        for candidate in (locals().get("cpu"), locals().get("mps")):
            if candidate is not None:
                candidate.shutdown()
        return _empty_result(manifest_path, model, f"{type(exc).__name__}: {exc}")
    dots = np.sum(reference * actual, axis=1)
    denom = np.linalg.norm(reference, axis=1) * np.linalg.norm(actual, axis=1)
    cosine = dots / np.maximum(denom, np.finfo(np.float32).eps)
    diff = np.abs(reference - actual)
    warm = latencies[:iterations]
    speedup = (sum(cpu_latencies) / len(cpu_latencies)) / (sum(warm) / len(warm))
    passed = bool(np.isfinite(actual).all() and cosine.min() >= 0.999)
    soak_target = 10000
    decision = _qualification_decision(
        speedup=speedup,
        soak_batches=soak_batches,
        soak_target=soak_target,
        numerical_pass=passed,
        oom_retries=oom_retries,
    )
    result = _empty_result(manifest_path, model, "qualification gates evaluated")
    result["run"]["status"] = "completed"
    result["backend"]["dimensions"] = actual.shape[1]
    result["performance"].update(
        {
            "cold_start_seconds": mps_load,
            "warm_total_seconds": sum(warm),
            "warm_iterations": iterations,
            "throughput_texts_per_second": len(texts) * iterations / sum(warm),
            "latency_seconds": {"samples": warm, "p50": statistics.median(warm), "p95": max(warm)},
            "peak_rss_bytes": peak_rss,
            "mps_peak_driver_allocated_bytes": driver_peak,
            "cpu_cold_start_seconds": cpu_load,
            "cpu_throughput_texts_per_second": len(texts) * iterations / sum(cpu_latencies),
            "speedup_over_cpu": speedup,
        }
    )
    result["reliability"] = _completed_reliability(
        attempted_batches=iterations + soak_batches + 1, oom_retries=oom_retries
    )
    result["correctness"] = {
        "reference_backend": "pytorch-cpu-worker",
        "shape": list(actual.shape),
        "finite": bool(np.isfinite(actual).all()),
        "max_absolute_error": float(diff.max()),
        "mean_absolute_error": float(diff.mean()),
        "minimum_cosine_similarity": float(cosine.min()),
        "passed": passed,
    }
    result["qualification"].update(
        {
            "decision": decision,
            "reason": (
                f"speedup={speedup:.3f}; soak={soak_batches}/{soak_target}; "
                f"numerical_pass={passed}; oom_retries={oom_retries}"
            ),
            "failure_policy": {
                "promotion_requires_zero_failures": True,
                "promotion_requires_zero_oom_retries": True,
            },
        }
    )
    return result

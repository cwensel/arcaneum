"""Offline MLX feasibility inventory for strategic embedding models.

This module deliberately does not install packages, convert weights, or contact a
model registry.  It records whether a useful MLX qualification can be performed
from the runtime and model assets already present on the machine.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from arcaneum.benchmarks.accelerator import (
    HARNESS_VERSION,
    SCHEMA_VERSION,
    load_manifest,
    manifest_digest,
)

SPEEDUP_GATE = 1.25
STRATEGIC_MODELS = {
    "minilm": {
        "repository": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "architecture": "BertModel",
        "tokenizer": "BERT WordPiece",
        "pooling": "attention-mask-aware mean pooling, then L2 normalization",
        "prompt_policy": "no query or document prompt",
        "source_precision": "float32 safetensors",
        "dimensions": 384,
        "conversion_assessment": "architecturally simple baseline; conversion still unverified",
    },
    "jina-code-st": {
        "repository": "jinaai/jina-embeddings-v2-base-code",
        "revision": "516f4baf13dec4ddddda8631e019b5737c8bc250",
        "architecture": "JinaBertForMaskedLM (remote model code)",
        "tokenizer": "Jina/BERT tokenizer assets",
        "pooling": "model-defined sentence embedding; parity must be measured",
        "prompt_policy": "retrieval.query / retrieval.passage task policy",
        "source_precision": "float32 safetensors/ONNX cache assets",
        "dimensions": 768,
        "conversion_assessment": "custom remote-code architecture requires a dedicated MLX port",
    },
    "qwen3-embed": {
        "repository": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "architecture": "Qwen3ForCausalLM",
        "tokenizer": "Qwen tokenizer with model chat/template assets",
        "pooling": "last token, including the query prompt, then L2 normalization",
        "prompt_policy": "query prompt_name=query; documents remain unprefixed",
        "source_precision": "source safetensors precision; MLX quantization not selected",
        "dimensions": 1024,
        "conversion_assessment": (
            "base architecture is plausible; embedding pooling parity unverified"
        ),
    },
}


def _package(name: str, module: str) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"installed": importlib.util.find_spec(module) is not None, "version": version}


def _repository_dir(cache_root: Path, repository: str) -> Path:
    return cache_root / ("models--" + repository.replace("/", "--"))


def _display_path(path: Path) -> str:
    try:
        return str(Path("~") / path.resolve().relative_to(Path.home().resolve()))
    except ValueError:
        return str(path)


def inventory_mlx(cache_root: Path) -> dict[str, Any]:
    """Inventory local runtimes and pinned model snapshots without network access."""
    packages = {
        "mlx": _package("mlx", "mlx"),
        "mlx-lm": _package("mlx-lm", "mlx_lm"),
        "mlx-embeddings": _package("mlx-embeddings", "mlx_embeddings"),
        "transformers": _package("transformers", "transformers"),
        "sentence-transformers": _package("sentence-transformers", "sentence_transformers"),
    }
    models = []
    for alias, facts in STRATEGIC_MODELS.items():
        repository_dir = _repository_dir(cache_root, facts["repository"])
        snapshot = repository_dir / "snapshots" / facts["revision"]
        converted = []
        if repository_dir.is_dir():
            converted = sorted(
                str(path.relative_to(repository_dir))
                for path in repository_dir.rglob("*")
                if path.is_file() and ("mlx" in path.name.lower() or "mlx" in path.parts)
            )
        models.append(
            {
                "alias": alias,
                **facts,
                "source_snapshot_cached": snapshot.is_dir(),
                "converted_mlx_assets": converted,
                "ready_for_offline_probe": bool(
                    packages["mlx"]["installed"] and snapshot.is_dir() and converted
                ),
            }
        )
    return {"packages": packages, "cache_root": _display_path(cache_root), "models": models}


def run_mlx_feasibility(manifest_path: Path, *, cache_dir: str) -> dict[str, Any]:
    """Return a schema-compatible, machine-readable MLX go/no-go artifact."""
    inventory = inventory_mlx(Path(cache_dir).expanduser())
    runtime_available = inventory["packages"]["mlx"]["installed"]
    runnable = [model["alias"] for model in inventory["models"] if model["ready_for_offline_probe"]]
    if not runtime_available:
        reason = "defer: MLX runtime is not installed; no performance or parity run was attempted"
    elif not runnable:
        reason = "defer: no strategic model has locally converted MLX assets"
    else:
        reason = "defer: runnable assets exist, but an embedding parity adapter is not implemented"
    manifest, fixtures = load_manifest(manifest_path)
    now = datetime.now(timezone.utc)
    return {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "run": {
            "id": now.strftime("%Y%m%dT%H%M%SZ-mlx-feasibility"),
            "recorded_at": now.isoformat(),
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
            "dependencies": inventory["packages"],
        },
        "backend": {
            "name": "mlx-feasibility",
            "device": "apple-silicon",
            "model": "strategic-model-inventory",
            "precision": "not-selected",
            "dimensions": 0,
        },
        "fixture": {
            "manifest_version": manifest["manifest_version"],
            "manifest_sha256": manifest_digest(manifest_path),
            "length_classes": sorted({fixture.length_class for fixture in fixtures}),
            "unique_texts": len(fixtures),
            "total_texts": sum(fixture.repetitions for fixture in fixtures),
        },
        "performance": {
            "cold_start_seconds": 0.0,
            "warm_total_seconds": 0.0,
            "warm_iterations": 0,
            "throughput_texts_per_second": 0.0,
            "latency_seconds": {"samples": [], "p50": 0.0, "p95": 0.0},
            "peak_rss_bytes": psutil.Process().memory_info().rss,
        },
        "reliability": {
            "attempted_batches": 0,
            "completed_batches": 0,
            "failures": 0,
            "fallbacks": 0,
            "restarts": 0,
        },
        "correctness": {
            "reference_backend": "sentence-transformers-cpu",
            "shape": [0, 0],
            "finite": False,
            "max_absolute_error": 0.0,
            "mean_absolute_error": 0.0,
            "minimum_cosine_similarity": 0.0,
            "passed": False,
        },
        "qualification": {
            "decision": "defer",
            "reason": reason,
            "speedup_gate": SPEEDUP_GATE,
            "soak_target_batches": 10_000,
            "network_access": "prohibited by this probe",
            "runnable_models": runnable,
            "inventory": inventory,
            "exact_rerun": (
                "PYTHONPATH=$PWD/src python scripts/benchmark_accelerators.py "
                "--backend mlx --output benchmarks/results/mlx-local.json"
            ),
        },
    }

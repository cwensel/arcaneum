"""Versioned, deny-by-default embedding backend capability selection."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

CapabilityState = Literal["stable", "experimental", "unavailable", "rejected"]


@dataclass(frozen=True)
class BackendSelection:
    backend: str
    state: CapabilityState
    device: str
    model: str
    evidence_version: str
    evidence: str
    fallback_reason: str | None = None
    worker_restart_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_capabilities() -> dict[str, Any]:
    path = Path(__file__).with_name("capabilities-v1.json")
    return json.loads(path.read_text(encoding="utf-8"))


def coreml_provider_options(cache_dir: Path) -> dict[str, str]:
    """Production CoreML options matching the qualified evaluation path."""
    return {
        "ModelFormat": "MLProgram",
        "RequireStaticInputShapes": "1",
        "ModelCacheDirectory": str(cache_dir.expanduser().resolve()),
        "SpecializationStrategy": "FastPrediction",
        "ProfileComputePlan": "1",
    }


def platform_key() -> str:
    system = sys.platform
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "darwin-arm64"
    if system.startswith("linux") and machine in {"x86_64", "amd64"}:
        return "linux-x86_64"
    return f"{system}-{machine or 'unknown'}"


def _rule(backend: str, model: str, model_backend: str, current_platform: str):
    matrix = load_capabilities()
    for rule in matrix["rules"]:
        if rule["backend"] != backend or rule["platform"] not in {"*", current_platform}:
            continue
        if "models" in rule and model not in rule["models"]:
            continue
        if "model_backend" in rule and rule["model_backend"] != model_backend:
            continue
        return matrix, rule
    return matrix, None


def select_backend(
    *,
    model: str,
    model_backend: str,
    requested_device: str,
    allow_experimental: bool = False,
    current_platform: str | None = None,
) -> BackendSelection:
    """Select a backend; only stable rules are eligible without explicit opt-in."""
    current_platform = current_platform or platform_key()
    if requested_device == "cpu":
        candidate = "fastembed-cpu" if model_backend == "fastembed" else "pytorch-cpu"
    elif requested_device == "mps":
        candidate = "onnxruntime-coreml" if model_backend == "fastembed" else "pytorch-mps"
    elif requested_device == "cuda":
        candidate = (
            "pytorch-cuda" if model_backend == "sentence-transformers" else "onnxruntime-cuda"
        )
    elif requested_device == "mlx":
        candidate = "mlx"
    else:
        candidate = f"unknown-{requested_device}"
    matrix, rule = _rule(candidate, model, model_backend, current_platform)
    version = matrix["evidence_version"]
    if rule is None:
        state: CapabilityState = "unavailable"
        reason = f"no capability rule for {candidate}/{model} on {current_platform}"
        if requested_device != "cpu":
            cpu = select_backend(
                model=model,
                model_backend=model_backend,
                requested_device="cpu",
                current_platform=current_platform,
            )
            return BackendSelection(
                cpu.backend, cpu.state, "cpu", model, version, cpu.evidence, reason
            )
        return BackendSelection(candidate, state, "cpu", model, version, "none", reason)
    state = rule["state"]
    if state == "stable" or (state == "experimental" and allow_experimental):
        return BackendSelection(
            candidate, state, requested_device, model, version, rule["evidence"]
        )
    reason = (
        f"{candidate} is {state}; explicit experimental opt-in required"
        if state == "experimental"
        else f"{candidate} is {state}"
    )
    cpu = select_backend(
        model=model,
        model_backend=model_backend,
        requested_device="cpu",
        current_platform=current_platform,
    )
    return BackendSelection(cpu.backend, cpu.state, "cpu", model, version, cpu.evidence, reason)

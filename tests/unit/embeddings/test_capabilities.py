import re
from pathlib import Path

from arcaneum.embeddings.capabilities import load_capabilities, select_backend

ROOT = Path(__file__).resolve().parents[3]


def test_matrix_is_versioned_and_has_only_known_states():
    matrix = load_capabilities()
    assert matrix["schema_version"] == "1.0.0"
    assert {rule["state"] for rule in matrix["rules"]} <= {
        "stable",
        "experimental",
        "unavailable",
        "rejected",
    }


def test_every_evidence_path_exists():
    matrix = load_capabilities()
    paths = [
        match
        for rule in matrix["rules"]
        for match in re.findall(r"(?:benchmarks|docs)/[A-Za-z0-9_.\-/]+", rule["evidence"])
    ]
    assert paths
    assert all((ROOT / path).is_file() for path in paths)


def test_cpu_is_deterministic_stable_path():
    selected = select_backend(model="arctic-m", model_backend="fastembed", requested_device="cpu")
    assert (selected.backend, selected.state, selected.device) == ("fastembed-cpu", "stable", "cpu")


def test_experimental_mps_requires_explicit_opt_in():
    denied = select_backend(
        model="jina-code-st",
        model_backend="sentence-transformers",
        requested_device="mps",
        current_platform="darwin-arm64",
    )
    allowed = select_backend(
        model="jina-code-st",
        model_backend="sentence-transformers",
        requested_device="mps",
        allow_experimental=True,
        current_platform="darwin-arm64",
    )
    assert denied.backend == "pytorch-cpu" and denied.fallback_reason
    assert (allowed.backend, allowed.state) == ("pytorch-mps", "experimental")


def test_unqualified_coreml_model_remains_experimental():
    selected = select_backend(
        model="arctic-m",
        model_backend="fastembed",
        requested_device="mps",
        allow_experimental=True,
        current_platform="darwin-arm64",
    )
    assert selected.backend == "onnxruntime-coreml"
    assert selected.state == "experimental"


def test_deferred_mlx_backend_is_truthfully_unavailable_even_with_opt_in():
    selected = select_backend(
        model="minilm",
        model_backend="sentence-transformers",
        requested_device="mlx",
        allow_experimental=True,
        current_platform="darwin-arm64",
    )
    assert selected.backend == "pytorch-cpu"
    assert "unavailable" in selected.fallback_reason


def test_unknown_cuda_fastembed_combination_falls_back_to_cpu():
    selected = select_backend(
        model="bge-small",
        model_backend="fastembed",
        requested_device="cuda",
        allow_experimental=True,
        current_platform="linux-x86_64",
    )
    assert selected.backend == "fastembed-cpu"
    assert selected.state == "stable"
    assert "no capability rule" in selected.fallback_reason

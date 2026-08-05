import json
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from arcaneum.benchmarks.accelerator import (
    SCHEMA_VERSION,
    compare_results,
    load_manifest,
    load_result_schema,
    manifest_digest,
    render_summary,
    run_reference_benchmark,
    validate_result,
    write_result,
)

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "benchmarks" / "fixtures" / "accelerator-v1" / "manifest.json"
EXAMPLE = ROOT / "benchmarks" / "results" / "example-accelerator-v1.json"


def test_manifest_has_representative_length_classes_and_stable_digest():
    _, fixtures = load_manifest(MANIFEST)

    assert {fixture.length_class for fixture in fixtures} == {
        "short",
        "medium",
        "long",
        "oversized",
    }
    assert manifest_digest(MANIFEST) == (
        "da62785afe275cff1b607cebb62f415a9c21089c842a4fe18a86c45e8199e165"
    )


def test_reference_cpu_result_covers_schema_and_is_correct():
    result = run_reference_benchmark(MANIFEST, iterations=2)
    schema = load_result_schema()

    assert result["schema_version"] == SCHEMA_VERSION
    assert set(schema["required"]) <= result.keys()
    assert result["backend"]["device"] == "cpu"
    assert result["fixture"]["total_texts"] == 15
    assert result["performance"]["cold_start_seconds"] >= 0
    assert result["performance"]["latency_seconds"]["p95"] >= 0
    assert result["performance"]["peak_rss_bytes"] > 0
    assert result["reliability"] == {
        "attempted_batches": 3,
        "completed_batches": 3,
        "failures": 0,
        "fallbacks": 0,
        "restarts": 0,
    }
    assert result["correctness"]["passed"] is True
    assert "Warm throughput:" in render_summary(result)


def test_schema_is_valid_and_accepts_every_checked_in_result():
    assert load_result_schema()["$schema"].endswith("draft/2020-12/schema")

    result_paths = sorted((ROOT / "benchmarks" / "results").glob("*.json"))
    assert result_paths
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        validate_result(result, label=path.name)


def test_benchmark_module_import_does_not_require_optional_validator():
    script = """
import importlib.abc
import sys

class BlockJsonschema(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jsonschema' or fullname.startswith('jsonschema.'):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockJsonschema())
import arcaneum.benchmarks.accelerator as accelerator
try:
    accelerator.load_result_schema()
except RuntimeError as exc:
    assert 'arcaneum[benchmarks]' in str(exc)
else:
    raise AssertionError('validation unexpectedly succeeded without jsonschema')
"""
    subprocess.run([sys.executable, "-c", script], check=True, cwd=ROOT)


def test_built_wheel_contains_canonical_result_schema(tmp_path):
    wheel_dir = tmp_path / "wheel"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_dir),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob("*.whl"))

    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")
        archive.extractall(installed)

    assert names.count("arcaneum/benchmarks/schemas/accelerator-result-v1.schema.json") == 1
    assert "Provides-Extra: benchmarks" in metadata
    assert 'Requires-Dist: jsonschema[format]' in metadata
    probe = """
import json
import sys
from importlib import resources

sys.path.insert(0, sys.argv[1])
schema = json.loads(
    resources.files('arcaneum.benchmarks')
    .joinpath('schemas/accelerator-result-v1.schema.json')
    .read_text(encoding='utf-8')
)
assert schema['properties']['schema_version']['const'] == '1.0.0'
"""
    subprocess.run(
        [sys.executable, "-I", "-c", probe, str(installed)],
        check=True,
        cwd=tmp_path,
    )


def test_schema_rejects_invalid_nested_result_data():
    result = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    result["reliability"]["failures"] = -1

    with pytest.raises(ValueError, match=r"\$\.reliability\.failures"):
        validate_result(result)


def test_write_result_validates_before_creating_output(tmp_path):
    result = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    output = tmp_path / "missing-parent" / "result.json"
    result["performance"]["latency_seconds"]["p95"] = -1

    with pytest.raises(ValueError, match=r"\$\.performance\.latency_seconds\.p95"):
        write_result(output, result)

    assert not output.exists()
    assert not output.parent.exists()


def test_write_result_accepts_valid_result(tmp_path):
    result = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    output = tmp_path / "result.json"

    write_result(output, result)

    assert json.loads(output.read_text(encoding="utf-8")) == result


@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_comparison_validates_both_inputs_before_comparing(side):
    baseline = run_reference_benchmark(MANIFEST, iterations=1)
    candidate = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    invalid = deepcopy(baseline if side == "baseline" else candidate)
    invalid["run"]["recorded_at"] = "not-a-date"
    if side == "baseline":
        baseline = invalid
    else:
        candidate = invalid

    with pytest.raises(ValueError, match=rf"{side}.*\$\.run\.recorded_at"):
        compare_results(baseline, candidate)


def test_example_accelerator_result_is_comparable():
    baseline = run_reference_benchmark(MANIFEST, iterations=1)
    candidate = json.loads(EXAMPLE.read_text())

    comparison = compare_results(baseline, candidate)

    assert comparison["candidate"] == "example-accelerator"
    assert comparison["speedup"] > 0
    assert comparison["correctness_passed"] is True


def test_comparison_rejects_fixture_drift():
    baseline = run_reference_benchmark(MANIFEST, iterations=1)
    candidate = json.loads(EXAMPLE.read_text())
    candidate["fixture"]["manifest_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="fixture manifests differ"):
        compare_results(baseline, candidate)


def test_comparison_rejects_incompatible_model_policy():
    baseline = run_reference_benchmark(MANIFEST, iterations=1)
    candidate = json.loads(EXAMPLE.read_text())
    candidate["backend"]["precision"] = "float16"

    with pytest.raises(ValueError, match="backend precision differs"):
        compare_results(baseline, candidate)


def test_cli_writes_machine_and_human_results(tmp_path):
    output = tmp_path / "result.json"
    summary = tmp_path / "summary.txt"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_accelerators.py"),
            "--iterations",
            "1",
            "--output",
            str(output),
            "--summary",
            str(summary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(output.read_text())["run"]["status"] == "completed"
    assert "Correctness: PASS" in summary.read_text()
    assert "reference-cpu" in completed.stdout

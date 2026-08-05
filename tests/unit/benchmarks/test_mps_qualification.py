import json
from pathlib import Path

from arcaneum.benchmarks.mps import (
    _completed_reliability,
    _empty_result,
    _qualification_decision,
    run_mps_qualification,
)

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "benchmarks" / "fixtures" / "accelerator-v1" / "manifest.json"
SCHEMA = ROOT / "benchmarks" / "schema" / "accelerator-result-v1.schema.json"


def test_inconclusive_result_is_truthful_and_captures_environment():
    result = _empty_result(MANIFEST, "jina-code-st", "torch unavailable")
    schema = json.loads(SCHEMA.read_text())
    assert set(schema["required"]) <= result.keys()
    assert result["run"]["status"] == "inconclusive"
    assert result["qualification"]["decision"] == "experimental"
    assert result["correctness"]["passed"] is False
    assert result["environment"]["memory_bytes"] > 0
    assert result["reliability"]["oom_retries"] is None


def test_unsafe_unlimited_watermark_is_refused(monkeypatch):
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    result = run_mps_qualification(MANIFEST, model="jina-code-st", cache_dir="/unused")
    assert result["run"]["status"] == "inconclusive"
    assert "refused" in result["qualification"]["reason"]
    assert result["reliability"]["completed_batches"] == 0


def test_nonzero_oom_retries_block_mps_qualification():
    clean = _qualification_decision(
        speedup=2.0,
        soak_batches=10000,
        soak_target=10000,
        numerical_pass=True,
        oom_retries=0,
    )
    recovered_oom = _qualification_decision(
        speedup=2.0,
        soak_batches=10000,
        soak_target=10000,
        numerical_pass=True,
        oom_retries=1,
    )

    assert clean == "qualified"
    assert recovered_oom == "experimental"


def test_recovered_oom_is_recorded_as_retry_and_failure():
    reliability = _completed_reliability(attempted_batches=12, oom_retries=2)

    assert reliability["completed_batches"] == 12
    assert reliability["oom_retries"] == 2
    assert reliability["failures"] == 2

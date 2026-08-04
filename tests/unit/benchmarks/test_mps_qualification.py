import json
from pathlib import Path

from arcaneum.benchmarks.mps import _empty_result, run_mps_qualification

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


def test_unsafe_unlimited_watermark_is_refused(monkeypatch):
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    result = run_mps_qualification(MANIFEST, model="jina-code-st", cache_dir="/unused")
    assert result["run"]["status"] == "inconclusive"
    assert "refused" in result["qualification"]["reason"]
    assert result["reliability"]["completed_batches"] == 0

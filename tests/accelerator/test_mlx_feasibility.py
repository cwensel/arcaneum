from pathlib import Path

from arcaneum.benchmarks import mlx

MANIFEST = Path("benchmarks/fixtures/accelerator-v1/manifest.json")


def test_inventory_reports_pinned_snapshots_and_no_converted_assets(monkeypatch, tmp_path):
    monkeypatch.setattr(
        mlx,
        "_package",
        lambda name, module: {"installed": name == "mlx", "version": "1.0"},
    )
    model = mlx.STRATEGIC_MODELS["minilm"]
    snapshot = (
        tmp_path
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
        / model["revision"]
    )
    snapshot.mkdir(parents=True)

    inventory = mlx.inventory_mlx(tmp_path)

    minilm = next(value for value in inventory["models"] if value["alias"] == "minilm")
    assert minilm["source_snapshot_cached"] is True
    assert minilm["converted_mlx_assets"] == []
    assert minilm["ready_for_offline_probe"] is False


def test_feasibility_defers_when_runtime_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        mlx,
        "_package",
        lambda name, module: {"installed": False, "version": None},
    )

    result = mlx.run_mlx_feasibility(MANIFEST, cache_dir=str(tmp_path))

    assert result["run"]["status"] == "inconclusive"
    assert result["qualification"]["decision"] == "defer"
    assert "runtime is not installed" in result["qualification"]["reason"]
    assert result["performance"]["warm_iterations"] == 0
    assert result["reliability"]["attempted_batches"] == 0
    assert result["correctness"]["passed"] is False


def test_converted_asset_requires_runtime_and_pinned_source(monkeypatch, tmp_path):
    monkeypatch.setattr(
        mlx,
        "_package",
        lambda name, module: {"installed": name == "mlx", "version": "1.0"},
    )
    model = mlx.STRATEGIC_MODELS["qwen3-embed"]
    root = tmp_path / "models--Qwen--Qwen3-Embedding-0.6B"
    snapshot = root / "snapshots" / model["revision"]
    snapshot.mkdir(parents=True)
    (snapshot / "weights-mlx.safetensors").write_bytes(b"test")

    inventory = mlx.inventory_mlx(tmp_path)

    qwen = next(value for value in inventory["models"] if value["alias"] == "qwen3-embed")
    assert qwen["ready_for_offline_probe"] is True
    assert qwen["converted_mlx_assets"] == [
        f"snapshots/{model['revision']}/weights-mlx.safetensors"
    ]

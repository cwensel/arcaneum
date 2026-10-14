"""`arc models workers` (RDR-025 Step 5)."""

import json
from unittest.mock import patch

from click.testing import CliRunner

from arcaneum.cli.main import cli
from arcaneum.embeddings.remote.identity import local_identity
from arcaneum.embeddings.remote.pool import EndpointStatus

DEPENDENCIES = {
    "torch": "2.14.1+cpu",
    "onnxruntime": "1.31.0",
    "fastembed": "0.8.0",
    "sentence-transformers": "5.7.0",
    "transformers": "4.57.0",
}


def statuses():
    identity = local_identity(None)
    return {
        "dev-0:7300": EndpointStatus(
            "dev-0:7300",
            "up",
            arcaneum_version=identity["arcaneum_version"],
            code_fingerprint=identity["code_fingerprint"],
            resident_model="arctic-m",
            busy=True,
            dependencies=DEPENDENCIES,
        ),
        "dev-1:7300": EndpointStatus(
            "dev-1:7300",
            "mismatch",
            "version mismatch: code_fingerprint local 7c47b1d53982… remote 000000000000…",
            arcaneum_version="0.9.0",
            code_fingerprint="0" * 64,
            dependencies=DEPENDENCIES,
            mismatches=["code_fingerprint"],
        ),
        "dev-2:7300": EndpointStatus("dev-2:7300", "unreachable", "unreachable (refused)"),
    }


def run(args, monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "remote_embedding:\n  endpoints:\n"
        "    - {host: dev-0, bind: tailscale}\n    - {host: dev-1}\n    - {host: dev-2}\n"
    )
    monkeypatch.delenv("ARC_EMBED_ENDPOINTS", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    known = statuses()
    with (
        patch("arcaneum.embeddings.remote.config._default_config_path", return_value=config),
        patch(
            "arcaneum.cli.remote_embedding.probe_endpoint",
            side_effect=lambda endpoint, cfg: known[endpoint.name],
        ),
    ):
        return CliRunner().invoke(cli, ["models", "workers", *args])


def test_workers_json_reports_each_endpoint(monkeypatch, tmp_path):
    result = run(["--json"], monkeypatch, tmp_path)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    data = payload["data"]
    assert data["local"]["code_fingerprint"] == local_identity(None)["code_fingerprint"]
    workers = {worker["endpoint"]: worker for worker in data["workers"]}
    assert workers["dev-0:7300"]["state"] == "up"
    assert workers["dev-0:7300"]["resident_model"] == "arctic-m"
    assert workers["dev-0:7300"]["busy"] is True
    assert workers["dev-0:7300"]["bind"] == "tailscale"
    assert workers["dev-0:7300"]["models"] == ["qwen3-embed", "arctic-m"]
    assert workers["dev-0:7300"]["dependencies"]["torch"] == "2.14.1+cpu"
    assert workers["dev-1:7300"]["state"] == "mismatch"
    assert workers["dev-1:7300"]["arcaneum_version"] == "0.9.0"
    assert workers["dev-2:7300"]["state"] == "unreachable"
    assert payload["status"] == "warning"


def test_workers_table_names_states_versions_and_models(monkeypatch, tmp_path):
    result = run([], monkeypatch, tmp_path)

    assert result.exit_code == 0, result.output
    for text in ("dev-0:7300", "arctic-m", "mismatch", "0.9.0", "unreachable", "2.14.1+cpu"):
        assert text in result.output
    assert "arc models deploy" in result.output


def test_workers_without_endpoints_explains_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_EMBED_ENDPOINTS", "")
    result = CliRunner().invoke(cli, ["models", "workers"])

    assert result.exit_code == 0, result.output
    assert "No remote embedding endpoints" in result.output
    assert "remote_embedding" in result.output

"""`arc models deploy` command wiring (RDR-025 Step 6)."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from arcaneum.cli.main import cli
from arcaneum.embeddings.remote.deploy import DeployError, WheelInfo

WHEEL = WheelInfo(Path("/tmp/arcaneum-0.10.0-py3-none-any.whl"), "c" * 64, "0.10.0", "0" * 64)


def invoke(args, monkeypatch, *, deploy_ok=True, wheel_error=None):
    monkeypatch.setenv("ARC_EMBED_ENDPOINTS", "dev-0,dev-1")
    with (
        patch(
            "arcaneum.cli.remote_embedding.resolve_wheel",
            side_effect=wheel_error,
            return_value=WHEEL,
        ) as resolve,
        patch("arcaneum.cli.remote_embedding.Deployer") as deployer,
    ):
        deployer.return_value.deploy.return_value = deploy_ok
        result = CliRunner().invoke(cli, ["models", "deploy", *args])
    return result, resolve, deployer


def test_deploy_defaults_to_the_release_wheel_and_all_endpoints(monkeypatch):
    result, resolve, deployer = invoke([], monkeypatch)

    assert result.exit_code == 0, result.output
    assert resolve.call_args.kwargs == {"build": False, "wheel": None}
    kwargs = deployer.call_args.kwargs
    assert (kwargs["dry_run"], kwargs["refresh"], kwargs["verify"]) == (False, False, True)
    deployer.return_value.deploy.assert_called_once_with(())
    assert "0.10.0" in result.output and "c" * 12 in result.output


def test_deploy_options_reach_the_deployer(monkeypatch, tmp_path):
    wheel = tmp_path / "arcaneum-0.10.0-py3-none-any.whl"
    wheel.write_bytes(b"x")
    result, resolve, deployer = invoke(
        ["dev-1", "--wheel", str(wheel), "--refresh", "--dry-run", "--no-verify"], monkeypatch
    )

    assert result.exit_code == 0, result.output
    assert resolve.call_args.kwargs == {"build": False, "wheel": wheel}
    kwargs = deployer.call_args.kwargs
    assert (kwargs["dry_run"], kwargs["refresh"], kwargs["verify"]) == (True, True, False)
    deployer.return_value.deploy.assert_called_once_with(("dev-1",))


def test_build_and_wheel_are_mutually_exclusive(monkeypatch, tmp_path):
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"x")
    result, _, deployer = invoke(["--build", "--wheel", str(wheel)], monkeypatch)

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
    deployer.assert_not_called()


def test_failed_hosts_exit_non_zero(monkeypatch):
    result, _, _ = invoke([], monkeypatch, deploy_ok=False)

    assert result.exit_code == 1


def test_refused_wheel_is_reported(monkeypatch):
    error = DeployError("release wheel differs; deploy it with --build")
    result, _, deployer = invoke([], monkeypatch, wheel_error=error)

    assert result.exit_code == 1
    assert "--build" in result.output
    deployer.assert_not_called()


def test_deploy_without_endpoints_explains_configuration(monkeypatch):
    monkeypatch.setenv("ARC_EMBED_ENDPOINTS", "")
    result = CliRunner().invoke(cli, ["models", "deploy"])

    assert result.exit_code == 1
    assert "No remote embedding endpoints" in result.output

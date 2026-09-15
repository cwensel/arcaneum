"""Tests for the live, comprehensive CLI description."""

import json
from unittest.mock import patch

from click.testing import CliRunner

from arcaneum import __version__
from arcaneum.cli.main import cli


def test_help_all_renders_nested_command_help():
    result = CliRunner().invoke(cli, ["--help-all"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "=== cli ===" in result.output
    assert "=== cli search semantic ===" in result.output
    assert "=== cli corpus hook install ===" in result.output
    assert "--score-threshold" in result.output
    assert "--changed-since" in result.output


def test_help_all_json_is_a_machine_readable_command_manifest():
    result = CliRunner().invoke(cli, ["--json", "--help-all"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert payload["program"] == "cli"
    assert payload["version"] == __version__

    semantic = next(
        command for command in payload["commands"] if command["path"] == "cli search semantic"
    )
    score_threshold = next(
        option for option in semantic["options"] if "--score-threshold" in option["flags"]
    )
    assert score_threshold["type"]["param_type"] == "Float"
    assert score_threshold["required"] is False


def test_help_all_skips_startup_migration():
    with patch("arcaneum.migrations.run_migration_if_needed") as migration:
        result = CliRunner().invoke(cli, ["--help-all"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    migration.assert_not_called()

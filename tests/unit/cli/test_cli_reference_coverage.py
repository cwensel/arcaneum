"""Guard that every CLI subcommand and flag appears in the CLI reference.

Three commands (doctor, models list, store) shipped in October 2025 and stayed
undocumented for roughly ten months, because nothing failed when a new
subcommand skipped its docs. These tests make that omission loud.
"""

from pathlib import Path

import click
import pytest

from arcaneum.cli.main import cli

CLI_REFERENCE = Path(__file__).resolve().parents[3] / "docs" / "guides" / "cli-reference.md"


def _walk(cmd, path):
    """Yield (command_path, click_command) for every leaf subcommand."""
    if isinstance(cmd, click.Group):
        for name, sub in sorted(cmd.commands.items()):
            yield from _walk(sub, path + [name])
    else:
        yield " ".join(path), cmd


def _leaves():
    return list(_walk(cli, ["arc"]))


@pytest.fixture(scope="module")
def reference_text():
    return CLI_REFERENCE.read_text()


def test_every_subcommand_is_in_the_cli_reference(reference_text):
    missing = [path for path, _ in _leaves() if path not in reference_text]
    assert not missing, "Subcommands missing from docs/guides/cli-reference.md: " + ", ".join(
        missing
    )


def test_every_long_flag_is_in_the_cli_reference(reference_text):
    missing = []
    for path, cmd in _leaves():
        for param in cmd.params:
            if not isinstance(param, click.Option) or param.hidden:
                continue
            for name in param.opts + param.secondary_opts:
                if name.startswith("--") and name not in reference_text:
                    missing.append(f"{path} {name}")
    assert not missing, "Options missing from docs/guides/cli-reference.md: " + ", ".join(missing)


def test_required_options_are_marked_required(reference_text):
    """A required option the docs never flag reads as optional."""
    import re

    unmarked = []
    for path, cmd in _leaves():
        for param in cmd.params:
            if not isinstance(param, click.Option) or not param.required:
                continue
            names = [n for n in param.opts if n.startswith("--")]
            if not names:
                continue
            name = names[0]
            nearby = re.findall(re.escape(name) + r"[^\n]{0,140}", reference_text)
            if not any("required" in line.lower() for line in nearby):
                unmarked.append(f"{path} {name}")
    assert not unmarked, "Required options not documented as required: " + ", ".join(unmarked)

"""The marketplace installs only the self-contained plugin directory.

Claude Code copies a plugin's ``source`` directory into its plugin cache,
gitignored content included. A ``"./"`` source copies the whole working tree
(``.venv``, ``.worktrees``, ``_repos``, ``src``, ``tests``), so the plugin lives
in ``plugin/`` and the marketplace entry points there.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
PLUGIN_DIR = "./plugin"
ALLOWED_TOP_LEVEL = {
    ".claude-plugin",
    "skills",
    "commands",
    "README.md",
    "settings.example.json",
}
EXPECTED_SKILLS = 5
EXPECTED_COMMANDS = 10


def plugin_layout_errors(root: Path) -> list[str]:
    """Return every way ``root``'s marketplace entry would ship more than the plugin."""
    errors = []
    marketplace = json.loads(
        (root / ".claude-plugin/marketplace.json").read_text(encoding="utf-8")
    )
    source = marketplace["plugins"][0]["source"]
    if not isinstance(source, str) or not source.startswith("./"):
        return [f"source must be a ./-relative path: {source!r}"]
    if ".." in Path(source).parts:
        return [f"source must not contain '..': {source!r}"]
    plugin = (root / source).resolve()
    if plugin == root.resolve():
        return [f"source is the repository root: {source!r}"]
    if not plugin.is_dir():
        return [f"source is not a directory: {source!r}"]

    manifest = plugin / ".claude-plugin/plugin.json"
    if not manifest.is_file():
        errors.append("missing .claude-plugin/plugin.json under the plugin")
    elif "skills" in json.loads(manifest.read_text(encoding="utf-8")):
        errors.append("plugin.json declares 'skills'; skills/ is discovered by default")

    extra = {p.name for p in plugin.iterdir()} - ALLOWED_TOP_LEVEL - {".DS_Store"}
    if extra:
        errors.append(f"unexpected top-level entries in the plugin: {sorted(extra)}")

    skills = sorted(plugin.glob("skills/*/SKILL.md"))
    if len(skills) != EXPECTED_SKILLS:
        errors.append(f"expected {EXPECTED_SKILLS} skills, found {len(skills)}")
    commands = sorted(plugin.glob("commands/*.md"))
    if len(commands) != EXPECTED_COMMANDS:
        errors.append(f"expected {EXPECTED_COMMANDS} commands, found {len(commands)}")
    return errors


def test_marketplace_ships_only_the_plugin_directory():
    marketplace = json.loads(
        (ROOT / ".claude-plugin/marketplace.json").read_text(encoding="utf-8")
    )
    assert marketplace["plugins"][0]["source"] == PLUGIN_DIR
    assert plugin_layout_errors(ROOT) == []


def test_plugin_manifest_version_matches_marketplace():
    marketplace = json.loads(
        (ROOT / ".claude-plugin/marketplace.json").read_text(encoding="utf-8")
    )
    plugin = json.loads(
        (ROOT / "plugin/.claude-plugin/plugin.json").read_text(encoding="utf-8")
    )
    assert plugin["name"] == marketplace["plugins"][0]["name"]
    assert plugin["version"] == marketplace["plugins"][0]["version"]


def _scratch_repo(tmp_path: Path, source: str) -> Path:
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin/marketplace.json").write_text(
        json.dumps({"plugins": [{"name": "arc", "source": source}]}),
        encoding="utf-8",
    )
    plugin = tmp_path / "plugin"
    (plugin / ".claude-plugin").mkdir(parents=True)
    (plugin / ".claude-plugin/plugin.json").write_text(
        json.dumps({"name": "arc", "version": "0.0.0"}), encoding="utf-8"
    )
    for i in range(EXPECTED_SKILLS):
        (plugin / f"skills/s{i}").mkdir(parents=True)
        (plugin / f"skills/s{i}/SKILL.md").write_text("x", encoding="utf-8")
    (plugin / "commands").mkdir()
    for i in range(EXPECTED_COMMANDS):
        (plugin / f"commands/c{i}.md").write_text("x", encoding="utf-8")
    return tmp_path


def test_scratch_layout_is_accepted(tmp_path):
    assert plugin_layout_errors(_scratch_repo(tmp_path, PLUGIN_DIR)) == []


@pytest.mark.parametrize("source", ["./", ".", "plugin", "./plugin/..", "../arcaneum"])
def test_refuses_a_source_that_is_not_the_plugin_directory(tmp_path, source):
    assert plugin_layout_errors(_scratch_repo(tmp_path, source))


@pytest.mark.parametrize("stray", ["src", "tests", ".venv", ".worktrees", "_repos"])
def test_refuses_repository_content_inside_the_plugin(tmp_path, stray):
    repo = _scratch_repo(tmp_path, PLUGIN_DIR)
    (repo / "plugin" / stray).mkdir()
    assert plugin_layout_errors(repo)


def test_refuses_a_skills_key_in_the_plugin_manifest(tmp_path):
    repo = _scratch_repo(tmp_path, PLUGIN_DIR)
    (repo / "plugin/.claude-plugin/plugin.json").write_text(
        json.dumps({"name": "arc", "version": "0.0.0", "skills": "./skills"}),
        encoding="utf-8",
    )
    assert plugin_layout_errors(repo)

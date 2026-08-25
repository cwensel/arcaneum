"""Interactive `arc corpus hook install` walkthrough (follow-on to kata vq0n).

Running `arc corpus hook install` with no corpus name is the natural thing to
type before you know the options, so it guides instead of erroring: pick or
create a corpus, choose hook points, and offer the initial backfill a hook
alone would never do.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from arcaneum.cli import hooks
from arcaneum.cli.main import cli


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="git hooks are shell scripts; not a supported target",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@e.com")
    _git(root, "config", "user.name", "T")
    (root / "a.py").write_text("print('a')\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(root)
    return root


def _hook(repo: Path, name: str = "post-commit") -> Path:
    return repo / ".git" / "hooks" / name


# --- repo content type inference ---------------------------------------------


def test_infers_code_from_mostly_source_files(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    for name in ("a.py", "b.py", "c.ts", "readme.md"):
        (root / name).write_text("x\n")
    assert hooks.infer_corpus_type(root) == "code"


def test_infers_markdown_from_mostly_docs(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    for name in ("a.md", "b.md", "c.md", "setup.py"):
        (root / name).write_text("x\n")
    assert hooks.infer_corpus_type(root) == "markdown"


def test_infers_pdf_from_mostly_pdfs(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    for name in ("a.pdf", "b.pdf", "notes.md"):
        (root / name).write_text("x\n")
    assert hooks.infer_corpus_type(root) == "pdf"


def test_inference_ignores_dot_directories(tmp_path):
    """.git holds far more files than the tree; it must not skew the guess."""
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.md").write_text("x\n")
    noise = root / ".git"
    noise.mkdir()
    for i in range(50):
        (noise / f"{i}.py").write_text("x\n")
    assert hooks.infer_corpus_type(root) == "markdown"


def test_inference_falls_back_to_code_when_undecidable(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    (root / "data.bin").write_text("x\n")
    assert hooks.infer_corpus_type(root) == "code"


def test_inference_on_an_empty_tree_does_not_crash(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    assert hooks.infer_corpus_type(root) in {"code", "markdown", "pdf"}


# --- suggested hook points ----------------------------------------------------


def test_suggests_post_merge_when_the_repo_has_a_remote(repo):
    _git(repo, "remote", "add", "origin", "https://example.com/r.git")
    assert hooks.suggested_hooks(repo) == ["post-commit", "post-merge"]


def test_suggests_only_post_commit_without_a_remote(repo):
    assert hooks.suggested_hooks(repo) == ["post-commit"]


# --- the interactive flow -----------------------------------------------------


def _invoke(args, corpora, input_text, sync=None):
    """Run the CLI with corpus discovery stubbed and stdin scripted."""
    with patch("arcaneum.cli.hooks.list_corpus_names", return_value=corpora):
        with patch("arcaneum.cli.main._run_backfill_sync", sync or (lambda *a, **k: None)):
            return CliRunner().invoke(cli, args, input=input_text, catch_exceptions=False)


def test_bare_install_picks_an_existing_corpus(repo):
    result = _invoke(
        ["corpus", "hook", "install"],
        corpora=["Docs", "CodeBase"],
        # pick corpus 2, accept suggested hooks, decline backfill
        input_text="2\n\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert "Docs" in result.output and "CodeBase" in result.output
    assert hooks.BLOCK_START.format(corpus="CodeBase") in _hook(repo).read_text()


def test_bare_install_lists_the_available_corpora(repo):
    result = _invoke(["corpus", "hook", "install"], corpora=["Docs"], input_text="1\n\nn\n")
    assert result.exit_code == 0, result.output
    assert "Docs" in result.output


def test_bare_install_offers_to_create_when_none_exist(repo):
    """With no corpora at all, creating one is the only sensible path."""
    created = []

    def fake_create(name, corpus_type, models, description, output_json):
        created.append((name, corpus_type))

    with patch("arcaneum.cli.corpus.create_corpus_command", fake_create):
        result = _invoke(
            ["corpus", "hook", "install"],
            corpora=[],
            # name the new corpus, accept inferred type, accept hooks, decline backfill
            input_text="MyRepo\n\n\nn\n",
        )

    assert result.exit_code == 0, result.output
    assert created == [("MyRepo", "code")], "should infer 'code' from the .py repo"
    assert hooks.BLOCK_START.format(corpus="MyRepo") in _hook(repo).read_text()


def test_created_corpus_type_can_be_overridden(repo):
    created = []

    def fake_create(name, corpus_type, models, description, output_json):
        created.append((name, corpus_type))

    with patch("arcaneum.cli.corpus.create_corpus_command", fake_create):
        result = _invoke(
            ["corpus", "hook", "install"],
            corpora=[],
            input_text="MyRepo\nmarkdown\n\nn\n",
        )

    assert result.exit_code == 0, result.output
    assert created == [("MyRepo", "markdown")]


def test_interactive_installs_the_suggested_hook_points(repo):
    _git(repo, "remote", "add", "origin", "https://example.com/r.git")

    result = _invoke(["corpus", "hook", "install"], corpora=["Docs"], input_text="1\n\nn\n")

    assert result.exit_code == 0, result.output
    assert _hook(repo, "post-commit").exists()
    assert _hook(repo, "post-merge").exists(), "a repo with a remote should also cover pulls"


def test_hook_points_can_be_overridden_at_the_prompt(repo):
    result = _invoke(
        ["corpus", "hook", "install"],
        corpora=["Docs"],
        input_text="1\npost-commit,post-rewrite\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert _hook(repo, "post-commit").exists()
    assert _hook(repo, "post-rewrite").exists()
    assert not _hook(repo, "post-merge").exists()


def test_offers_the_initial_backfill_sync(repo):
    """A hook only indexes future commits; existing files need one real sync."""
    calls = []

    result = _invoke(
        ["corpus", "hook", "install"],
        corpora=["Docs"],
        input_text="1\n\ny\n",
        sync=lambda corpus, path, **k: calls.append((corpus, str(path))),
    )

    assert result.exit_code == 0, result.output
    assert calls == [("Docs", str(repo.resolve()))]


def test_corpus_picker_has_no_default(repo):
    """Which corpus a repo feeds must be chosen, never silently defaulted.

    An exhausted or mistyped stdin previously fell through to corpus #1,
    wiring the repo into an unrelated corpus without the user ever saying so.
    """
    with patch("arcaneum.cli.hooks.list_corpus_names", return_value=["Alpha", "Beta"]):
        result = CliRunner().invoke(cli, ["corpus", "hook", "install"], input="")

    assert result.exit_code != 0
    assert not _hook(repo).exists(), "EOF must not install into a defaulted corpus"


def test_invalid_picker_input_does_not_fall_through_to_a_default(repo):
    """A mistyped answer must not silently install into the wrong corpus."""
    with patch("arcaneum.cli.hooks.list_corpus_names", return_value=["Alpha", "Beta"]):
        # A name where a number belongs, then stdin runs dry.
        result = CliRunner().invoke(cli, ["corpus", "hook", "install"], input="Gamma\n")

    assert result.exit_code != 0
    assert not _hook(repo).exists()


def test_declining_the_backfill_still_prints_the_command(repo):
    result = _invoke(["corpus", "hook", "install"], corpora=["Docs"], input_text="1\n\nn\n")
    assert result.exit_code == 0, result.output
    assert "arc corpus sync Docs" in result.output


# --- non-interactive behavior is unchanged ------------------------------------


def test_naming_a_corpus_skips_every_prompt(repo):
    """The existing non-interactive path must not start prompting."""
    result = CliRunner().invoke(
        cli, ["corpus", "hook", "install", "Docs"], input="", catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert hooks.BLOCK_START.format(corpus="Docs") in _hook(repo).read_text()
    assert "?" not in result.output


def test_yes_flag_skips_prompts_with_a_bare_invocation(repo):
    """--yes takes the defaults so scripts and agents never block on stdin."""
    with patch("arcaneum.cli.hooks.list_corpus_names", return_value=["Docs"]):
        result = CliRunner().invoke(
            cli, ["corpus", "hook", "install", "Docs", "--yes"], catch_exceptions=False
        )
    assert result.exit_code == 0, result.output
    assert _hook(repo).exists()


def test_bare_install_without_a_tty_errors_rather_than_hanging(repo):
    """No corpus, no stdin, no --yes: fail with guidance instead of blocking."""
    with patch("arcaneum.cli.hooks.list_corpus_names", return_value=["Docs"]):
        result = CliRunner().invoke(cli, ["corpus", "hook", "install"], input="")
    assert result.exit_code != 0


def test_bare_install_outside_a_repo_errors_before_prompting(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["corpus", "hook", "install"], input="1\n")
    assert result.exit_code != 0


# --- --yes must not reopen the silent-wrong-corpus hazard (roborev 6105) ------


def test_yes_with_many_corpora_refuses_to_guess(repo):
    """--yes must not pick the alphabetically-first of several unrelated corpora."""
    with patch("arcaneum.cli.hooks.list_corpus_names", return_value=["Alpha", "Beta"]):
        result = CliRunner().invoke(cli, ["corpus", "hook", "install", "--yes"])

    assert result.exit_code != 0
    assert not _hook(repo).exists()
    # The error should tell the user how to proceed.
    assert "hook install" in result.output


def test_yes_auto_picks_when_exactly_one_corpus_exists(repo):
    """One corpus is unambiguous, so taking it is a safe default."""
    with patch("arcaneum.cli.hooks.list_corpus_names", return_value=["Solo"]):
        result = CliRunner().invoke(cli, ["corpus", "hook", "install", "--yes"])

    assert result.exit_code == 0, result.output
    assert hooks.BLOCK_START.format(corpus="Solo") in _hook(repo).read_text()


def test_yes_with_no_corpora_errors(repo):
    with patch("arcaneum.cli.hooks.list_corpus_names", return_value=[]):
        result = CliRunner().invoke(cli, ["corpus", "hook", "install", "--yes"])
    assert result.exit_code != 0


def test_yes_installs_the_suggested_hook_points(repo):
    _git(repo, "remote", "add", "origin", "https://example.com/r.git")
    with patch("arcaneum.cli.hooks.list_corpus_names", return_value=["Solo"]):
        result = CliRunner().invoke(cli, ["corpus", "hook", "install", "--yes"])

    assert result.exit_code == 0, result.output
    assert _hook(repo, "post-commit").exists()
    assert _hook(repo, "post-merge").exists()


# --- interactive hook-point validation (roborev 6105) ------------------------


def test_a_typo_in_hook_points_installs_nothing(repo):
    """Validate before installing, so a bad name cannot half-install the set."""
    result = _invoke(
        ["corpus", "hook", "install"],
        corpora=["Docs"],
        input_text="1\npost-commit,post-comit\nn\n",
    )

    assert result.exit_code != 0
    assert not _hook(repo, "post-commit").exists(), "must fail before writing anything"


def test_multi_hook_success_message_lists_every_path(repo):
    """Reporting one path while claiming several hooks is misleading."""
    _git(repo, "remote", "add", "origin", "https://example.com/r.git")
    result = _invoke(["corpus", "hook", "install"], corpora=["Docs"], input_text="1\n\nn\n")

    assert result.exit_code == 0, result.output
    assert "post-commit" in result.output
    assert "post-merge" in result.output


def test_inference_does_not_walk_into_dot_directories(tmp_path):
    """Pruning, not filtering: a big .venv must not be traversed at all.

    Filtering only the count still walks every vendored file, so the sample
    limit bounds counted work rather than total work (roborev 6105).
    """
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.md").write_text("x\n")
    heavy = root / ".venv" / "lib"
    heavy.mkdir(parents=True)
    for i in range(200):
        (heavy / f"{i}.py").write_text("x\n")

    visited = []
    real_walk = os.walk

    def counting_walk(top, *args, **kwargs):
        for dirpath, dirnames, filenames in real_walk(top, *args, **kwargs):
            visited.append(dirpath)
            yield dirpath, dirnames, filenames

    with patch("arcaneum.cli.hooks.os.walk", counting_walk):
        assert hooks.infer_corpus_type(root) == "markdown"

    assert not any(".venv" in v for v in visited), "dot-dirs must be pruned from the walk"

"""MeiliSearch is optional: `meilisearch.enabled: false` keeps every command off it (arcaneum#183)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner
from fulltext.conftest import FakeFullTextClient, LocalQdrant

from arcaneum.indexing.collection_metadata import set_collection_metadata


class RefusingMeili:
    """A MeiliSearch client no call may reach."""

    def __getattr__(self, name):
        raise AssertionError(f"MeiliSearch was called: {name}")


@pytest.fixture
def new_user(tmp_path, monkeypatch):
    """A fresh home: no MeiliSearch key, no config.yaml."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("MEILISEARCH_API_KEY", raising=False)
    return tmp_path / "arcaneum"


@pytest.fixture
def disabled(new_user):
    """An existing user who turned MeiliSearch off."""
    new_user.mkdir(parents=True, exist_ok=True)
    (new_user / "meilisearch.key").write_text("an-existing-meilisearch-key\n")
    (new_user / "config.yaml").write_text("meilisearch:\n  enabled: false\n")
    return new_user


@pytest.fixture
def enabled(new_user):
    new_user.mkdir(parents=True, exist_ok=True)
    (new_user / "config.yaml").write_text("meilisearch:\n  enabled: true\n")
    return new_user


def _qdrant_with(*corpora):
    """A local Qdrant holding each ``(name, text_backend)`` corpus."""
    qdrant = LocalQdrant()
    for name, backend in corpora:
        _add_corpus(qdrant, name, backend)
    return qdrant


def _add_corpus(qdrant, name, backend):
    from arcaneum.retrieval.qdrant_text import QdrantTextBackend

    qdrant.create_collection_with_vector(name)
    set_collection_metadata(qdrant, name, "pdf", "stella", text_backend=backend)
    if backend == "qdrant":
        QdrantTextBackend(qdrant).ensure(name, "pdf")


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def _compose_env(args, **kwargs):
    from arcaneum.cli.docker import run_compose_command

    with (
        patch("arcaneum.cli.docker.get_compose_file", return_value="/compose.yml"),
        patch("subprocess.run") as run,
    ):
        run.return_value = MagicMock(returncode=0)
        run_compose_command(args, **kwargs)
    return run.call_args.kwargs["env"]


class TestComposeProfile:
    def test_disabled_leaves_the_meilisearch_profile_out(self, disabled, monkeypatch):
        monkeypatch.delenv("COMPOSE_PROFILES", raising=False)
        assert "meilisearch" not in _compose_env(["up", "-d"]).get("COMPOSE_PROFILES", "")

    def test_enabled_passes_the_meilisearch_profile(self, enabled, monkeypatch):
        monkeypatch.delenv("COMPOSE_PROFILES", raising=False)
        assert _compose_env(["up", "-d"])["COMPOSE_PROFILES"] == "meilisearch"

    def test_down_always_passes_the_profile(self, disabled, monkeypatch):
        monkeypatch.delenv("COMPOSE_PROFILES", raising=False)
        assert _compose_env(["down"])["COMPOSE_PROFILES"] == "meilisearch"

    def test_a_users_own_profiles_are_kept(self, enabled, monkeypatch):
        monkeypatch.setenv("COMPOSE_PROFILES", "extra")
        assert _compose_env(["up", "-d"])["COMPOSE_PROFILES"] == "extra,meilisearch"

    def test_the_compose_file_puts_meilisearch_behind_the_profile(self):
        from pathlib import Path

        import arcaneum

        compose = Path(arcaneum.__file__).resolve().parents[2] / "deploy" / "docker-compose.yml"
        services = yaml.safe_load(compose.read_text())["services"]
        assert services["meilisearch"]["profiles"] == ["meilisearch"]
        assert "profiles" not in services["qdrant"]


# ---------------------------------------------------------------------------
# arc container start / restart / status / upgrade / backup / restore
# ---------------------------------------------------------------------------


def _start(qdrant=None, output_json=False):
    from arcaneum.cli.docker import start_command

    with (
        patch("arcaneum.cli.docker.check_docker_available", return_value=True),
        patch("arcaneum.cli.docker._qdrant_version_blocker", return_value=None),
        patch("arcaneum.cli.docker._meili_version_blocker") as meili_blocker,
        patch("arcaneum.cli.docker.get_compose_file", return_value="/compose.yml"),
        patch("arcaneum.cli.docker.get_resource_limit_env", return_value={}),
        patch("arcaneum.cli.docker.check_qdrant_health", return_value=True),
        patch("arcaneum.cli.docker.check_meilisearch_health", return_value=False),
        patch("arcaneum.cli.utils.create_qdrant_client", return_value=qdrant or _qdrant_with()),
        patch("subprocess.run") as run,
        patch("time.sleep"),
    ):
        run.return_value = MagicMock(returncode=0)
        meili_blocker.return_value = None
        try:
            start_command.callback(output_json=output_json)
            code = 0
        except SystemExit as exc:
            code = exc.code
    return SimpleNamespace(run=run, meili_blocker=meili_blocker, code=code)


class TestContainerStart:
    def test_new_user_starts_qdrant_only_and_records_the_choice(self, new_user, capsys):
        started = _start()

        assert started.code == 0
        assert not started.meili_blocker.called
        env = started.run.call_args.kwargs["env"]
        assert "meilisearch" not in env.get("COMPOSE_PROFILES", "")
        assert env["MEILISEARCH_API_KEY"] == ""
        assert not (new_user / "meilisearch.key").exists()
        out = capsys.readouterr().out
        assert "MeiliSearch" not in out.replace("meilisearch.enabled", "")
        assert yaml.safe_load((new_user / "config.yaml").read_text()) == {
            "meilisearch": {"enabled": False},
            "corpus": {"default_text_backend": "qdrant"},
        }

    def test_existing_user_keeps_meilisearch_and_records_it(self, new_user):
        new_user.mkdir(parents=True)
        (new_user / "meilisearch.key").write_text("an-existing-meilisearch-key\n")

        started = _start()

        assert started.meili_blocker.called
        env = started.run.call_args.kwargs["env"]
        assert env["COMPOSE_PROFILES"] == "meilisearch"
        assert env["MEILISEARCH_API_KEY"] == "an-existing-meilisearch-key"
        assert yaml.safe_load((new_user / "config.yaml").read_text())["meilisearch"] == {
            "enabled": True
        }

    def test_disabled_with_a_meili_corpus_is_refused_by_name(self, disabled, capsys):
        started = _start(qdrant=_qdrant_with(("Papers", "meili"), ("Code", "qdrant")))

        assert started.code == 1
        err = capsys.readouterr().err
        assert "arc corpus migrate-text Papers --to qdrant" in err
        assert "Code" not in err.split("MeiliSearch:")[-1]

    def test_json_reports_meilisearch_disabled(self, new_user, capsys):
        _start(output_json=True)
        payload = json.loads(capsys.readouterr().out)
        assert payload["data"]["services"]["meilisearch"] == {"enabled": False}


class TestContainerRestartStatusUpgrade:
    def test_restart_reports_no_meilisearch_health(self, disabled, capsys):
        from arcaneum.cli.docker import restart_command

        with (
            patch("arcaneum.cli.docker.check_docker_available", return_value=True),
            patch("arcaneum.cli.docker.run_compose_command", return_value=MagicMock()),
            patch("arcaneum.cli.docker.check_qdrant_health", return_value=True),
            patch("arcaneum.cli.docker.check_meilisearch_health") as meili_health,
            patch("time.sleep"),
        ):
            restart_command.callback(output_json=True)
        assert not meili_health.called
        payload = json.loads(capsys.readouterr().out)
        assert payload["data"]["services"]["meilisearch"] == {"enabled": False}

    def test_status_prints_no_meilisearch_health_line(self, disabled, capsys):
        from arcaneum.cli.docker import status_command

        with (
            patch("arcaneum.cli.docker.check_docker_available", return_value=True),
            patch("arcaneum.cli.docker.run_compose_command", return_value=MagicMock(stdout="")),
            patch("arcaneum.cli.docker.check_qdrant_health", return_value=True),
            patch("arcaneum.cli.docker.check_meilisearch_health") as meili_health,
            patch("subprocess.run", side_effect=OSError("no docker")),
        ):
            try:
                status_command.callback(output_json=False)
            except OSError:
                pass
        assert not meili_health.called
        assert "MeiliSearch: Unhealthy" not in capsys.readouterr().out

    def test_upgrade_skips_meilisearch(self, disabled, capsys):
        from arcaneum.cli.docker import upgrade_command

        with (
            patch("arcaneum.cli.docker.check_docker_available", return_value=True),
            patch("arcaneum.cli.docker._upgrade_qdrant", return_value={"upgraded": False}),
            patch("arcaneum.cli.docker._upgrade_meilisearch") as upgrade_meili,
        ):
            upgrade_command.callback(meilisearch_url="http://meili", dry_run=False, output_json=True)
        assert not upgrade_meili.called
        payload = json.loads(capsys.readouterr().out)
        assert payload["data"]["qdrant"] == {"upgraded": False}
        assert payload["data"]["meilisearch"] == {"enabled": False}


class TestBackupRestore:
    def test_backup_skips_meilisearch(self, disabled, tmp_path):
        from arcaneum.cli.docker import backup_command

        with (
            patch("arcaneum.cli.docker.check_docker_available", return_value=True),
            patch("arcaneum.cli.docker._backup_qdrant", return_value=[]),
            patch("arcaneum.cli.docker._backup_meilisearch") as backup_meili,
            patch("arcaneum.cli.docker._meilisearch_headers") as headers,
        ):
            backup_command.callback(
                output=str(tmp_path / "backup"),
                qdrant_url="http://qdrant",
                meilisearch_url="http://meili",
                qdrant_container="qdrant",
                qdrant_timeout=300,
                skip_meilisearch=False,
                output_json=True,
            )
        assert not backup_meili.called
        assert not headers.called
        manifest = json.loads((tmp_path / "backup" / "manifest.json").read_text())
        assert manifest["meilisearch_url"] is None

    def test_restore_says_what_it_skipped(self, disabled, tmp_path, capsys):
        from arcaneum.cli.docker import restore_command

        backup = tmp_path / "backup"
        backup.mkdir()
        (backup / "manifest.json").write_text(
            json.dumps({"qdrant": [], "meilisearch": [{"index": "Papers"}, {"index": "Code"}]})
        )
        with (
            patch("arcaneum.cli.docker.check_docker_available", return_value=True),
            patch("arcaneum.cli.docker._restore_qdrant"),
            patch("arcaneum.cli.docker._restore_meilisearch") as restore_meili,
        ):
            restore_command.callback(
                backup_directory=str(backup),
                qdrant_url="http://qdrant",
                meilisearch_url="http://meili",
                qdrant_container="qdrant",
                qdrant_timeout=300,
                meilisearch_timeout=1800,
                skip_meilisearch=False,
                output_json=True,
            )
        assert not restore_meili.called
        data = json.loads(capsys.readouterr().out)["data"]
        assert data["meilisearch_indexes"] == 0
        assert any("Papers, Code" in warning for warning in data["warnings"])
        assert any("meilisearch.enabled" in warning for warning in data["warnings"])


# ---------------------------------------------------------------------------
# arc corpus list / create, arc doctor, migrate-text, search text
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus_services(monkeypatch):
    from arcaneum.cli import corpus

    qdrant = LocalQdrant()
    state = {"meili": FakeFullTextClient()}
    monkeypatch.setattr(corpus, "create_qdrant_client", lambda: qdrant)
    monkeypatch.setattr(corpus, "create_meili_client", lambda: state["meili"])
    monkeypatch.setattr(corpus, "interaction_logger", MagicMock())
    return qdrant, state


class TestCorpusList:
    def test_no_meilisearch_error_when_disabled_and_down(self, disabled, corpus_services):
        from arcaneum.cli.main import cli

        qdrant, state = corpus_services
        state["meili"] = RefusingMeili()
        _add_corpus(qdrant, "Code", "qdrant")

        result = CliRunner().invoke(cli, ["corpus", "list"])

        assert result.exit_code == 0, result.output
        assert "MeiliSearch" not in result.output
        assert "ERROR" not in result.output

    def test_a_meili_corpus_is_named_with_its_migration(self, disabled, corpus_services):
        from arcaneum.cli.main import cli

        qdrant, state = corpus_services
        state["meili"] = RefusingMeili()
        _add_corpus(qdrant, "Papers", "meili")

        result = CliRunner().invoke(cli, ["corpus", "list"])

        assert "arc corpus migrate-text Papers --to qdrant" in result.output


def _create(name, *extra):
    from arcaneum.cli.main import cli

    return CliRunner().invoke(
        cli, ["corpus", "create", name, "--type", "pdf", "--models", "stella", "--json", *extra]
    )


class TestCorpusCreate:
    def test_a_new_user_gets_a_qdrant_corpus(self, new_user, corpus_services):
        _qdrant, state = corpus_services
        state["meili"] = RefusingMeili()

        result = _create("Docs")

        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["data"]["text_backend"] == "qdrant"

    def test_the_configured_default_is_used(self, enabled, corpus_services):
        (enabled / "config.yaml").write_text(
            "meilisearch:\n  enabled: true\ncorpus:\n  default_text_backend: qdrant\n"
        )
        result = _create("Docs")
        assert json.loads(result.output)["data"]["text_backend"] == "qdrant"

    def test_meili_is_refused_while_disabled(self, disabled, corpus_services):
        qdrant, state = corpus_services
        state["meili"] = RefusingMeili()

        result = _create("Docs", "--text-backend", "meili")

        assert result.exit_code != 0
        assert "meilisearch.enabled" in result.output + str(result.exception)
        assert not qdrant.collection_exists("Docs")


class TestDoctor:
    def test_a_fresh_home_creates_no_key_and_needs_no_meilisearch(self, new_user):
        from arcaneum.cli.doctor import check_meilisearch_connection

        with patch("arcaneum.fulltext.client.FullTextClient") as client:
            status, message = check_meilisearch_connection()

        assert status is not False
        assert "disabled" in message.lower()
        assert "start" not in message.lower()
        assert not client.called
        assert not (new_user / "meilisearch.key").exists()

    def test_text_backends_name_a_meili_corpus_while_disabled(self, disabled):
        from arcaneum.cli.doctor import check_text_backends

        qdrant = _qdrant_with(("Papers", "meili"), ("Code", "qdrant"))
        with patch("arcaneum.cli.doctor.create_qdrant_client", return_value=qdrant):
            status, message = check_text_backends()

        assert status is False
        assert "arc corpus migrate-text Papers --to qdrant" in message


class TestMigrateText:
    @pytest.mark.parametrize("args", [["--to", "meili"], ["--finalize"]])
    def test_needs_meilisearch_enabled(self, disabled, args):
        from arcaneum.cli.main import cli

        with (
            patch("arcaneum.cli.utils.create_qdrant_client") as qdrant,
            patch("arcaneum.cli.utils.create_meili_client") as meili,
        ):
            result = CliRunner().invoke(cli, ["corpus", "migrate-text", "Docs", *args])

        assert result.exit_code != 0
        assert "meilisearch.enabled" in result.output
        assert not qdrant.called
        assert not meili.called


class TestSearchText:
    def test_qdrant_corpora_search_without_meilisearch(self, disabled):
        from arcaneum.retrieval.text import search_text

        qdrant = _qdrant_with(("Code", "qdrant"))
        result = search_text("anything", ["Code"], qdrant_client=qdrant)
        assert result.corpora == ["Code"]

    def test_a_meili_corpus_is_refused_naming_the_setting(self, disabled):
        from arcaneum.cli.errors import ArcaneumError
        from arcaneum.retrieval.text import search_text

        qdrant = _qdrant_with(("Papers", "meili"))
        with pytest.raises(ArcaneumError, match="meilisearch.enabled") as exc_info:
            search_text("anything", ["Papers"], qdrant_client=qdrant, text_backend=RefusingMeili())
        assert "migrate-text Papers --to qdrant" in str(exc_info.value)

    def test_unread_qdrant_reports_the_qdrant_error(self, disabled):
        from arcaneum.cli.errors import ServiceDownError
        from arcaneum.retrieval.text import search_text

        qdrant = MagicMock()
        qdrant.retrieve.side_effect = ConnectionError("connection refused")
        with pytest.raises(ServiceDownError, match="Qdrant"):
            search_text("anything", ["Papers"], qdrant_client=qdrant, text_backend=RefusingMeili())

    def test_a_legacy_collection_without_metadata_is_refused_naming_the_setting(self, disabled):
        from arcaneum.cli.errors import ArcaneumError
        from arcaneum.retrieval.text import search_text

        qdrant = _qdrant_with()
        qdrant.create_collection_with_vector("Legacy")  # no metadata point: reads as meili
        with pytest.raises(ArcaneumError, match="meilisearch.enabled") as exc_info:
            search_text("anything", ["Legacy"], qdrant_client=qdrant, text_backend=RefusingMeili())
        assert "migrate-text Legacy --to qdrant" in str(exc_info.value)

    def test_a_legacy_collection_beside_a_qdrant_corpus_is_not_skipped(self, disabled):
        from arcaneum.cli.errors import ArcaneumError
        from arcaneum.retrieval.text import search_text

        qdrant = _qdrant_with(("Code", "qdrant"))
        qdrant.create_collection_with_vector("Legacy")
        with pytest.raises(ArcaneumError, match="migrate-text Legacy --to qdrant"):
            search_text(
                "anything", ["Code", "Legacy"], qdrant_client=qdrant, text_backend=RefusingMeili()
            )

    def test_an_absent_corpus_is_unknown(self, disabled):
        from arcaneum.cli.errors import UnknownCorpusError
        from arcaneum.retrieval.text import search_text

        with pytest.raises(UnknownCorpusError):
            search_text(
                "anything", ["Nope"], qdrant_client=_qdrant_with(), text_backend=RefusingMeili()
            )


class TestPassage:
    def test_a_chunk_qdrant_lacks_is_not_looked_up_in_meilisearch(self, disabled):
        from arcaneum.cli.errors import ResourceNotFoundError
        from arcaneum.retrieval.passage import get_passage

        qdrant = _qdrant_with(("Code", "qdrant"))
        with pytest.raises(ResourceNotFoundError):
            get_passage(
                "Code",
                "00000000-0000-0000-0000-000000000001",
                qdrant_client=qdrant,
                text_backend=RefusingMeili(),
            )

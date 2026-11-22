"""Whether MeiliSearch is enabled, and the text backend new corpora get (arcaneum#183).

``meilisearch.enabled`` and ``corpus.default_text_backend`` in config.yaml win;
absent, a stored MeiliSearch key (or ``MEILISEARCH_API_KEY``) reads as an
existing user (MeiliSearch on, new corpora on meili), and its absence as a new
user (MeiliSearch off, new corpora on qdrant).
"""

import os
import stat

import pytest
import yaml

from arcaneum.config import persist_meili_settings, resolve_meili_settings


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fresh config home: no key, no config.yaml, no key in the environment."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("MEILISEARCH_API_KEY", raising=False)
    return tmp_path / "arcaneum"


def _key(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "meilisearch.key").write_text("an-existing-meilisearch-key\n")


def _config(home, text):
    home.mkdir(parents=True, exist_ok=True)
    path = home / "config.yaml"
    path.write_text(text)
    return path


class TestInference:
    def test_new_user_without_a_key_gets_qdrant_only(self, home):
        settings = resolve_meili_settings(home / "config.yaml")
        assert settings.enabled is False
        assert settings.default_text_backend == "qdrant"
        assert settings.inferred is True

    def test_existing_key_file_keeps_meilisearch(self, home):
        _key(home)
        settings = resolve_meili_settings(home / "config.yaml")
        assert settings.enabled is True
        assert settings.default_text_backend == "meili"

    def test_key_in_the_environment_counts_as_an_existing_user(self, home, monkeypatch):
        monkeypatch.setenv("MEILISEARCH_API_KEY", "an-environment-meilisearch-key")
        settings = resolve_meili_settings(home / "config.yaml")
        assert settings.enabled is True
        assert settings.default_text_backend == "meili"

    def test_explicit_keys_win_over_a_stored_key(self, home):
        _key(home)
        path = _config(
            home,
            "meilisearch:\n  enabled: false\ncorpus:\n  default_text_backend: qdrant\n",
        )
        settings = resolve_meili_settings(path)
        assert settings.enabled is False
        assert settings.default_text_backend == "qdrant"
        assert settings.inferred is False

    def test_explicit_enabled_wins_for_a_new_user(self, home):
        path = _config(home, "meilisearch:\n  enabled: true\n")
        settings = resolve_meili_settings(path)
        assert settings.enabled is True
        assert settings.default_text_backend == "meili"

    def test_disabled_without_a_backend_defaults_new_corpora_to_qdrant(self, home):
        _key(home)
        path = _config(home, "meilisearch:\n  enabled: false\n")
        assert resolve_meili_settings(path).default_text_backend == "qdrant"

    def test_explicit_backend_wins_over_inference(self, home):
        _key(home)
        path = _config(home, "corpus:\n  default_text_backend: qdrant\n")
        settings = resolve_meili_settings(path)
        assert settings.enabled is True
        assert settings.default_text_backend == "qdrant"

    @pytest.mark.parametrize(
        "text, key",
        [
            ("meilisearch:\n  enabled: maybe\n", "meilisearch.enabled"),
            ("corpus:\n  default_text_backend: elastic\n", "corpus.default_text_backend"),
        ],
    )
    def test_invalid_values_name_the_setting(self, home, text, key):
        path = _config(home, text)
        with pytest.raises(ValueError, match=key):
            resolve_meili_settings(path)

    def test_resolving_writes_nothing(self, home):
        resolve_meili_settings(home / "config.yaml")
        assert not (home / "config.yaml").exists()
        assert not (home / "meilisearch.key").exists()


class TestPersistence:
    def test_creates_a_missing_config_with_the_inferred_values(self, home):
        path = home / "config.yaml"
        persist_meili_settings(resolve_meili_settings(path))
        data = yaml.safe_load(path.read_text())
        assert data == {
            "meilisearch": {"enabled": False},
            "corpus": {"default_text_backend": "qdrant"},
        }
        # The choice now holds even once a key exists.
        _key(home)
        assert resolve_meili_settings(path).enabled is False

    def test_appends_to_an_existing_config_keeping_its_content(self, home):
        _key(home)
        original = "# my settings\nqdrant:\n  url: http://example:6333  # remote box\n"
        path = _config(home, original)
        os.chmod(path, 0o600)
        persist_meili_settings(resolve_meili_settings(path))
        text = path.read_text()
        assert text.startswith(original)
        data = yaml.safe_load(text)
        assert data["qdrant"] == {"url": "http://example:6333"}
        assert data["meilisearch"] == {"enabled": True}
        assert data["corpus"] == {"default_text_backend": "meili"}
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_appends_after_a_file_without_a_trailing_newline(self, home):
        path = _config(home, "qdrant:\n  url: http://example:6333")
        persist_meili_settings(resolve_meili_settings(path))
        data = yaml.safe_load(path.read_text())
        assert data["qdrant"] == {"url": "http://example:6333"}
        assert data["meilisearch"] == {"enabled": False}

    def test_explicit_values_are_not_rewritten(self, home):
        text = "meilisearch:\n  enabled: true\ncorpus:\n  default_text_backend: qdrant\n"
        path = _config(home, text)
        notes = persist_meili_settings(resolve_meili_settings(path))
        assert path.read_text() == text
        assert notes == []

    def test_a_section_without_the_leaf_is_left_alone_and_named(self, home):
        text = "meilisearch:\n  url: http://localhost:7700\n"
        path = _config(home, text)
        notes = persist_meili_settings(resolve_meili_settings(path))
        data = yaml.safe_load(path.read_text())
        assert data["meilisearch"] == {"url": "http://localhost:7700"}
        assert data["corpus"] == {"default_text_backend": "qdrant"}
        assert any("meilisearch.enabled: false" in note for note in notes)

    def test_is_idempotent(self, home):
        path = home / "config.yaml"
        persist_meili_settings(resolve_meili_settings(path))
        first = path.read_text()
        persist_meili_settings(resolve_meili_settings(path))
        assert path.read_text() == first


class TestKeyGeneration:
    def test_no_key_is_generated_while_meilisearch_is_disabled(self, home):
        from arcaneum.paths import get_meilisearch_api_key

        assert get_meilisearch_api_key() == ""
        assert not (home / "meilisearch.key").exists()

    def test_a_key_is_generated_once_meilisearch_is_enabled(self, home):
        from arcaneum.paths import get_meilisearch_api_key

        _config(home, "meilisearch:\n  enabled: true\n")
        key = get_meilisearch_api_key()
        assert len(key) >= 16
        assert (home / "meilisearch.key").read_text().strip() == key

    def test_an_existing_key_is_returned_even_when_disabled(self, home):
        from arcaneum.paths import get_meilisearch_api_key

        _key(home)
        _config(home, "meilisearch:\n  enabled: false\n")
        assert get_meilisearch_api_key() == "an-existing-meilisearch-key"

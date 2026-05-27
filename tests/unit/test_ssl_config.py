"""Unit tests for SSL verification policy."""

from unittest.mock import patch

from arcaneum.ssl_config import configure_ssl_from_env


def test_ssl_verification_is_strict_by_default(monkeypatch):
    monkeypatch.delenv("ARC_SSL_VERIFY", raising=False)

    with patch("arcaneum.ssl_config.disable_ssl_verification") as disable_ssl:
        assert configure_ssl_from_env() is False

    disable_ssl.assert_not_called()


def test_ssl_verification_can_be_disabled_explicitly(monkeypatch):
    monkeypatch.setenv("ARC_SSL_VERIFY", "false")

    with patch(
        "arcaneum.ssl_config.disable_ssl_verification",
        return_value=True,
    ) as disable_ssl:
        assert configure_ssl_from_env() is True

    disable_ssl.assert_called_once_with(quiet=True)

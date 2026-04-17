"""Unit tests for arcaneum.cli.search_merge helpers."""

import logging

import pytest

from arcaneum.cli.errors import ResourceNotFoundError
from arcaneum.cli.search_merge import fetch_from_corpora, per_corpus_limit


class TestPerCorpusLimit:
    """Tests for per_corpus_limit()."""

    def test_single_corpus_no_headroom(self):
        # With one corpus there is no cross-corpus merge, so no over-fetch.
        assert per_corpus_limit(["only"], limit=10, offset=0) == 10

    def test_single_corpus_adds_offset(self):
        assert per_corpus_limit(["only"], limit=10, offset=5) == 15

    def test_multi_corpus_doubles(self):
        # Multi-corpus over-fetches 2x so merged top-N stays stable.
        assert per_corpus_limit(["a", "b"], limit=10, offset=0) == 20

    def test_multi_corpus_doubles_with_offset(self):
        assert per_corpus_limit(["a", "b", "c"], limit=10, offset=5) == 30

    def test_empty_corpora_single_corpus_path(self):
        # len([]) == 0 is not > 1, so falls through the single-corpus branch.
        assert per_corpus_limit([], limit=10, offset=0) == 10


class _Missing(Exception):
    """Sentinel exception used as the 'missing corpus' signal in tests."""


def _is_missing(exc: Exception) -> bool:
    return isinstance(exc, _Missing)


class TestFetchFromCorpora:
    """Tests for fetch_from_corpora()."""

    def test_all_present(self):
        logger = logging.getLogger("test")

        def fetch(name):
            return f"hits-{name}"

        results, missing = fetch_from_corpora(
            ["a", "b", "c"], fetch, _is_missing, logger, verbose=False
        )
        assert results == ["hits-a", "hits-b", "hits-c"]
        assert missing == []

    def test_partial_missing_multi_corpus_skips(self):
        logger = logging.getLogger("test")

        def fetch(name):
            if name == "b":
                raise _Missing(name)
            return f"hits-{name}"

        results, missing = fetch_from_corpora(
            ["a", "b", "c"], fetch, _is_missing, logger, verbose=False
        )
        assert results == ["hits-a", "hits-c"]
        assert missing == ["b"]

    def test_single_corpus_missing_raises(self):
        logger = logging.getLogger("test")

        def fetch(name):
            raise _Missing(name)

        with pytest.raises(ResourceNotFoundError) as exc_info:
            fetch_from_corpora(
                ["only"], fetch, _is_missing, logger, verbose=False
            )
        assert "'only'" in str(exc_info.value)

    def test_all_corpora_missing_raises(self):
        logger = logging.getLogger("test")

        def fetch(name):
            raise _Missing(name)

        with pytest.raises(ResourceNotFoundError) as exc_info:
            fetch_from_corpora(
                ["a", "b", "c"], fetch, _is_missing, logger, verbose=False
            )
        msg = str(exc_info.value)
        assert "a" in msg and "b" in msg and "c" in msg

    def test_non_missing_exception_propagates(self):
        logger = logging.getLogger("test")

        class OtherError(RuntimeError):
            pass

        def fetch(name):
            raise OtherError("boom")

        with pytest.raises(OtherError):
            fetch_from_corpora(
                ["a", "b"], fetch, _is_missing, logger, verbose=False
            )

    def test_verbose_logs_skip_warning(self, caplog):
        logger = logging.getLogger("test_verbose_logs_skip_warning")

        def fetch(name):
            if name == "b":
                raise _Missing(name)
            return f"hits-{name}"

        with caplog.at_level(logging.WARNING, logger=logger.name):
            fetch_from_corpora(
                ["a", "b"], fetch, _is_missing, logger, verbose=True
            )
        assert any("'b'" in rec.message for rec in caplog.records)

    def test_non_verbose_does_not_log(self, caplog):
        logger = logging.getLogger("test_non_verbose_does_not_log")

        def fetch(name):
            if name == "b":
                raise _Missing(name)
            return f"hits-{name}"

        with caplog.at_level(logging.WARNING, logger=logger.name):
            fetch_from_corpora(
                ["a", "b"], fetch, _is_missing, logger, verbose=False
            )
        assert caplog.records == []

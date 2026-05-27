"""Unit tests for remote SentenceTransformer model loading policy."""

import re

import pytest

from arcaneum.embeddings.client import (
    EMBEDDING_MODELS,
    TRUST_REMOTE_CODE_ALLOWLIST,
    _sentence_transformer_load_kwargs,
)


REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def test_sentence_transformer_models_are_pinned_to_revisions():
    st_models = {
        key: config
        for key, config in EMBEDDING_MODELS.items()
        if config.get("backend") == "sentence-transformers"
    }

    assert st_models
    for model_key, config in st_models.items():
        assert REVISION_RE.match(config.get("revision", "")), model_key


def test_trusted_remote_code_requires_allowlisted_pinned_revision():
    config = {
        "name": "unreviewed/model",
        "backend": "sentence-transformers",
        "trust_remote_code": True,
        "revision": "0" * 40,
    }

    with pytest.raises(ValueError, match="not allowlisted"):
        _sentence_transformer_load_kwargs(
            "unsafe",
            config,
            cache_folder="/tmp/models",
            local_files_only=False,
            device="cpu",
        )


def test_trusted_remote_code_kwargs_include_allowlisted_revision():
    config = EMBEDDING_MODELS["stella"]

    kwargs = _sentence_transformer_load_kwargs(
        "stella",
        config,
        cache_folder="/tmp/models",
        local_files_only=False,
        device="cpu",
    )

    assert kwargs["trust_remote_code"] is True
    assert kwargs["revision"] == TRUST_REMOTE_CODE_ALLOWLIST[config["name"]]


def test_non_remote_code_models_do_not_enable_trust_remote_code():
    config = EMBEDDING_MODELS["nomic-code"]

    kwargs = _sentence_transformer_load_kwargs(
        "nomic-code",
        config,
        cache_folder="/tmp/models",
        local_files_only=True,
        device="cpu",
    )

    assert kwargs["trust_remote_code"] is False
    assert kwargs["revision"] == config["revision"]

"""Unit tests for remote SentenceTransformer model loading policy."""

import builtins
import logging
import re

import pytest

from arcaneum.embeddings.client import (
    EMBEDDING_MODELS,
    TRUST_REMOTE_CODE_ALLOWLIST,
    EmbeddingClient,
    _sentence_transformer_load_kwargs,
    get_embedding_prompt_policy,
    get_embedding_prompt_policies,
    model_key_for_name,
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


def test_sentence_transformer_prompt_policy_records_retrieval_semantics():
    assert get_embedding_prompt_policy("jina-code")["query"]["method"] == "query_embed"
    assert get_embedding_prompt_policy("jina-code")["document"]["method"] == "embed"
    assert get_embedding_prompt_policy("jina-code-st")["query"]["task"] == "retrieval.query"
    assert get_embedding_prompt_policy("jina-code-st")["document"]["task"] == "retrieval.passage"
    assert get_embedding_prompt_policy("stella")["query"]["prompt_name"] == "s2p_query"
    assert get_embedding_prompt_policy("e5-base")["document"]["prompt"] == "passage: "


def test_default_code_model_uses_fastembed_backend():
    config = EMBEDDING_MODELS["jina-code"]

    assert config["backend"] == "fastembed"
    assert config["name"] == "jinaai/jina-embeddings-v2-base-code"


def test_ambiguous_provider_model_name_does_not_choose_backend_alias():
    assert model_key_for_name("jinaai/jina-embeddings-v2-base-code") is None


def test_legacy_provider_model_name_stamps_default_prompt_policy():
    policies = get_embedding_prompt_policies("jinaai/jina-embeddings-v2-base-code")

    assert policies == {"jina-code": get_embedding_prompt_policy("jina-code")}


def test_legacy_bge_provider_model_name_stamps_alias_prompt_policy():
    policies = get_embedding_prompt_policies("BAAI/bge-large-en-v1.5")

    assert policies == {"bge-large": get_embedding_prompt_policy("bge-large")}


def test_unknown_provider_model_name_fails_prompt_policy_stamping():
    with pytest.raises(ValueError, match="Unknown model: unknown/provider-model"):
        get_embedding_prompt_policies("unknown/provider-model")


def test_qwen3_embed_uses_native_backend_without_remote_code():
    config = EMBEDDING_MODELS["qwen3-embed"]

    assert config["name"] == "Qwen/Qwen3-Embedding-0.6B"
    assert config["backend"] == "sentence-transformers"
    assert config["dimensions"] == 1024
    assert config["recommended_for"] == "pdf"
    assert not config.get("trust_remote_code", False)
    assert REVISION_RE.match(config["revision"])

    kwargs = _sentence_transformer_load_kwargs(
        "qwen3-embed",
        config,
        cache_folder="/tmp/models",
        local_files_only=True,
        device="cpu",
    )

    assert kwargs["trust_remote_code"] is False
    assert kwargs["revision"] == config["revision"]


def test_qwen3_embed_prompt_policy_uses_builtin_query_prompt():
    policy = get_embedding_prompt_policy("qwen3-embed")

    assert policy["query"]["prompt_name"] == "query"
    assert "prompt_name" not in policy["document"]


def test_stella_is_deprecated_in_favor_of_qwen3_embed():
    config = EMBEDDING_MODELS["stella"]

    assert config["deprecated"] is True
    assert config["superseded_by"] == "qwen3-embed"
    assert config.get("recommended_for") is None


def test_get_model_warns_once_for_deprecated_models(caplog):
    client = EmbeddingClient(cache_dir="/tmp/models", use_gpu=False)
    sentinel = object()
    client._models["stella"] = sentinel

    with caplog.at_level(logging.WARNING, logger="arcaneum.embeddings.client"):
        assert client.get_model("stella") is sentinel
        assert client.get_model("stella") is sentinel

    warnings = [r for r in caplog.records if "deprecated" in r.getMessage()]
    assert len(warnings) == 1
    assert "qwen3-embed" in warnings[0].getMessage()


def test_sentence_transformer_backend_reports_missing_extra(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("missing optional backend")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    client = EmbeddingClient(cache_dir="/tmp/models", use_gpu=False)
    with pytest.raises(RuntimeError, match=r"arcaneum\[sentence-transformers\]"):
        client.get_model("stella")

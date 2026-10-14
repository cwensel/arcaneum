"""Handshake identity carried by initialize/initialized (RDR-025)."""

import arcaneum
from arcaneum.embeddings.client import EMBEDDING_MODELS
from arcaneum.embeddings.remote.fingerprint import code_fingerprint
from arcaneum.embeddings.remote.identity import (
    dependency_versions,
    identity_mismatches,
    local_identity,
)
from arcaneum.embeddings.worker_protocol import PROTOCOL_VERSION


def test_model_identity_carries_every_handshake_field_from_code():
    identity = local_identity("arctic-m")

    assert identity == {
        "protocol_version": PROTOCOL_VERSION,
        "arcaneum_version": arcaneum.__version__,
        "code_fingerprint": code_fingerprint(),
        "model": "arctic-m",
        "revision": EMBEDDING_MODELS["arctic-m"]["revision"],
        "prompt_policy_version": 1,
        "dimensions": 768,
    }


def test_status_identity_has_no_model_fields():
    assert local_identity(None) == {
        "protocol_version": PROTOCOL_VERSION,
        "arcaneum_version": arcaneum.__version__,
        "code_fingerprint": code_fingerprint(),
    }


def test_identity_mismatches_name_each_differing_field():
    local = local_identity("qwen3-embed")
    remote = {**local, "code_fingerprint": "f" * 64, "dimensions": 768}

    assert identity_mismatches(local, local) == []
    assert identity_mismatches(local, remote) == ["code_fingerprint", "dimensions"]
    assert identity_mismatches(local, {"protocol_version": PROTOCOL_VERSION}) == [
        "arcaneum_version",
        "code_fingerprint",
        "model",
        "revision",
        "prompt_policy_version",
        "dimensions",
    ]


def test_dependency_versions_report_each_inference_package():
    versions = dependency_versions()

    assert set(versions) == {
        "torch",
        "onnxruntime",
        "fastembed",
        "sentence-transformers",
        "transformers",
    }
    assert versions["fastembed"]

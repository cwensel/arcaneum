"""Stable extraction and chunking policy identities for incremental indexing."""

from typing import Any, Mapping, Optional

_POLICY_IDS = {
    "pdf": ("pdf-extraction:v2", "pdf-chunking:v3"),
    "markdown": ("markdown-extraction:v1", "markdown-semantic:v2"),
    "code": ("code-extraction:v1", "code-ast:v1"),
}


def build_indexing_policy(
    corpus_type: str,
    config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return the active policy identity and relevant reproducibility settings."""
    extraction_id, chunking_id = _POLICY_IDS.get(
        corpus_type, (f"{corpus_type}-extraction:v1", f"{corpus_type}-chunking:v1")
    )
    config = config or {}
    relevant_config = {
        key: config[key]
        for key in (
            "chunk_size",
            "chunk_overlap",
            "char_to_token_ratio",
            "min_chunk_chars",
            "late_chunking",
        )
        if key in config
    }
    return {
        "schema_version": 1,
        "corpus_type": corpus_type,
        "extraction": {"id": extraction_id, "implementation_version": extraction_id},
        "chunking": {
            "id": chunking_id,
            "implementation_version": chunking_id,
            "config": relevant_config,
        },
    }


def policy_is_current(
    policy: object,
    corpus_type: str,
    expected_policy: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return whether a persisted policy uses the active corpus-scoped identities."""
    if not isinstance(policy, dict):
        return False
    active = expected_policy or build_indexing_policy(corpus_type)
    identities_match = (
        policy.get("corpus_type") == corpus_type
        and (policy.get("extraction") or {}).get("id") == active["extraction"]["id"]
        and (policy.get("chunking") or {}).get("id") == active["chunking"]["id"]
    )
    stored_config = (policy.get("chunking") or {}).get("config") or {}
    active_config = (active.get("chunking") or {}).get("config") or {}
    return identities_match and (
        not stored_config or not active_config or stored_config == active_config
    )

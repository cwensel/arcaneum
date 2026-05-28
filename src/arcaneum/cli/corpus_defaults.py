"""Shared corpus defaults used by CLI command modules."""

DEFAULT_MODELS_BY_CORPUS_TYPE = {
    # FastEmbed CPU defaults keep the common document pipeline stable on
    # Apple Silicon while still using a modern retrieval model.
    "pdf": "arctic-m",
    "markdown": "arctic-m",
    # Keep code default lightweight. Larger code models remain opt-in.
    "code": "jina-code",
}

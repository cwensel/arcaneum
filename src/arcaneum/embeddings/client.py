"""Embedding client utilities with FastEmbed (RDR-002)."""

from fastembed import TextEmbedding
from typing import Dict, List
import os

# Model configurations with dimensions
EMBEDDING_MODELS = {
    "stella": {
        "name": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
    },
    "modernbert": {
        "name": "nomic-ai/modernbert-embed-base",
        "dimensions": 1024,
    },
    "bge": {
        "name": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
    },
    "jina": {
        "name": "jinaai/jina-embeddings-v2-base-code",
        "dimensions": 768,
    },
}


class EmbeddingClient:
    """Manages embedding model instances with caching."""

    def __init__(self, cache_dir: str = "./models_cache", verify_ssl: bool = True):
        """Initialize embedding client.

        Args:
            cache_dir: Directory to cache downloaded models
            verify_ssl: Whether to verify SSL certificates (set False for self-signed certs)

        Note: SSL configuration must be done before creating EmbeddingClient.
              Use ssl_config.check_and_configure_ssl() or disable_ssl_verification() first.
        """
        self.cache_dir = cache_dir
        self.verify_ssl = verify_ssl
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
        self._models: Dict[str, TextEmbedding] = {}

    def get_model(self, model_name: str) -> TextEmbedding:
        """Get or initialize embedding model.

        Args:
            model_name: Model identifier (stella, jina, modernbert, bge)

        Returns:
            TextEmbedding instance

        Raises:
            ValueError: If model_name is not recognized
        """
        if model_name not in EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(EMBEDDING_MODELS.keys())}"
            )

        if model_name not in self._models:
            config = EMBEDDING_MODELS[model_name]
            self._models[model_name] = TextEmbedding(
                model_name=config["name"],
                cache_dir=self.cache_dir,
            )
        return self._models[model_name]

    def embed(self, texts: List[str], model_name: str) -> List[List[float]]:
        """Generate embeddings for texts using specified model.

        Processes in batches of 100 to prevent FastEmbed hangs on large batches.

        Args:
            texts: List of text strings to embed
            model_name: Model identifier (stella, jina, modernbert, bge)

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If model_name is not recognized
        """
        model = self.get_model(model_name)

        # Process in batches to prevent FastEmbed hangs
        BATCH_SIZE = 100
        all_embeddings = []
        total_texts = len(texts)

        # Just embed - batching now handled by callers (uploader.py, source_code_pipeline.py)
        # They control the line updates to maintain progress consistency
        all_embeddings = list(model.embed(texts))

        return all_embeddings

    def get_dimensions(self, model_name: str) -> int:
        """Get vector dimensions for a model.

        Args:
            model_name: Model identifier

        Returns:
            Number of dimensions

        Raises:
            ValueError: If model_name is not recognized
        """
        if model_name not in EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(EMBEDDING_MODELS.keys())}"
            )
        return EMBEDDING_MODELS[model_name]["dimensions"]

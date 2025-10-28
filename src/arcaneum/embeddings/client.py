"""Embedding client utilities with FastEmbed (RDR-002)."""

from fastembed import TextEmbedding
from typing import Dict, List
import os

# Model configurations with dimensions
# Note: "stella" is an alias for bge-large since actual stella (dunzhang/stella_en_1.5B_v5)
# is not available in FastEmbed
# Model configurations with multiple backends
EMBEDDING_MODELS = {
    # Code-specific models (SentenceTransformers)
    "jina-code": {
        "name": "jinaai/jina-embeddings-v2-base-code",
        "dimensions": 768,
        "backend": "sentence-transformers",
        "description": "Code-specific (768D, 8K context, best for source code)",
        "available": True,
        "recommended_for": "code"
    },

    # General purpose models (SentenceTransformers)
    "stella": {
        "name": "dunzhang/stella_en_1.5B_v5",
        "dimensions": 1024,
        "backend": "sentence-transformers",
        "description": "General purpose (1024D, high quality for docs/PDFs)",
        "available": True,
        "recommended_for": "pdf"
    },

    # BGE models (FastEmbed - fast ONNX inference)
    "bge-large": {
        "name": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
        "backend": "fastembed",
        "description": "BGE Large (1024D, general purpose, fast)",
        "available": True
    },
    "bge": {  # Alias
        "name": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
        "backend": "fastembed",
        "description": "BGE Large (alias for bge-large)",
        "available": True
    },
    "bge-base": {
        "name": "BAAI/bge-base-en-v1.5",
        "dimensions": 768,
        "backend": "fastembed",
        "description": "BGE Base (768D, balanced)",
        "available": True
    },
    "bge-small": {
        "name": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "backend": "fastembed",
        "description": "BGE Small (384D, fastest)",
        "available": True
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

    def get_model(self, model_name: str):
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
            backend = config.get("backend", "fastembed")

            if backend == "fastembed":
                self._models[model_name] = TextEmbedding(
                    model_name=config["name"],
                    cache_dir=self.cache_dir,
                )
            elif backend == "sentence-transformers":
                from sentence_transformers import SentenceTransformer
                model_obj = SentenceTransformer(config["name"], cache_folder=self.cache_dir)
                model_obj._backend = "sentence-transformers"
                self._models[model_name] = model_obj

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

        # Handle different backends
        if hasattr(model, '_backend') and model._backend == "sentence-transformers":
            # SentenceTransformers: use encode()
            embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=False)
            return [emb.tolist() for emb in embeddings]
        else:
            # FastEmbed: use embed()
            return list(model.embed(texts))

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

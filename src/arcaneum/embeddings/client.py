"""Embedding client utilities with FastEmbed (RDR-002)."""

from fastembed import TextEmbedding
from typing import Dict, List
import os
from arcaneum.paths import get_models_dir

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

    # Jina models (FastEmbed)
    "jina-v3": {
        "name": "jinaai/jina-embeddings-v3",
        "dimensions": 1024,
        "backend": "fastembed",
        "description": "Jina v3 (1024D, multilingual ~100, 8K context, 2024)",
        "available": True,
        "recommended_for": "multilingual"
    },
    "jina-base-en": {
        "name": "jinaai/jina-embeddings-v2-base-en",
        "dimensions": 768,
        "backend": "fastembed",
        "description": "Jina v2 Base English (768D, 8K context, English-only)",
        "available": True
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

    def __init__(self, cache_dir: str = None, verify_ssl: bool = True):
        """Initialize embedding client.

        Args:
            cache_dir: Directory to cache downloaded models (defaults to ~/.arcaneum/models)
            verify_ssl: Whether to verify SSL certificates (set False for self-signed certs)

        Note: SSL configuration must be done before creating EmbeddingClient.
              Use ssl_config.check_and_configure_ssl() or disable_ssl_verification() first.
        """
        self.cache_dir = cache_dir or str(get_models_dir())
        self.verify_ssl = verify_ssl
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = self.cache_dir
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
                import sys

                # Check if model is cached to avoid unnecessary network calls
                is_cached = self.is_model_cached(model_name)

                # Show loading indicator for models that take time
                if not is_cached:
                    print("   Downloading model files...", flush=True, file=sys.stderr)

                # Use local_files_only if model is cached to prevent network calls
                # This is the official FastEmbed parameter for offline mode
                self._models[model_name] = TextEmbedding(
                    model_name=config["name"],
                    cache_dir=self.cache_dir,
                    local_files_only=is_cached  # Skip network access if cached
                )
            elif backend == "sentence-transformers":
                from sentence_transformers import SentenceTransformer
                import sys

                # Check if model is cached to avoid unnecessary network calls
                is_cached = self.is_model_cached(model_name)

                # Show loading indicator for models that take time
                if not is_cached:
                    print("   Downloading model files...", flush=True, file=sys.stderr)

                # SentenceTransformer handles download progress automatically via HuggingFace
                # Use local_files_only if cached to prevent network calls to HuggingFace Hub
                model_obj = SentenceTransformer(
                    config["name"],
                    cache_folder=self.cache_dir,
                    local_files_only=is_cached  # Skip HuggingFace Hub check if cached
                )
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
            # Disable progress bar - pipeline handles progress display
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

    def is_model_cached(self, model_name: str) -> bool:
        """Check if a model is already cached locally.

        Args:
            model_name: Model identifier

        Returns:
            True if model is cached, False if needs download

        Raises:
            ValueError: If model_name is not recognized
        """
        if model_name not in EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(EMBEDDING_MODELS.keys())}"
            )

        config = EMBEDDING_MODELS[model_name]
        backend = config.get("backend", "fastembed")
        model_path = config["name"]

        if backend == "sentence-transformers":
            # Check HuggingFace cache (models--<org>--<model>)
            safe_model_name = model_path.replace("/", "--")
            model_dir = os.path.join(self.cache_dir, f"models--{safe_model_name}")
            return os.path.exists(model_dir) and os.path.isdir(model_dir)
        else:
            # FastEmbed uses HuggingFace cache structure with models-- prefix
            # The actual cached model name may differ from config (e.g., qdrant/bge-large-en-v1.5-onnx)
            # So we check for any model directory that starts with the model base name
            model_base = model_path.split("/")[-1].replace("-", "_")  # e.g., "bge_large_en_v1"

            # Check for exact match first (models--org--model format)
            safe_model_name = model_path.replace("/", "--")
            model_dir = os.path.join(self.cache_dir, f"models--{safe_model_name}")
            if os.path.exists(model_dir) and os.path.isdir(model_dir):
                return True

            # Check for FastEmbed wrapped versions (models--qdrant--model-onnx format)
            # List all model directories and check for similar names
            if os.path.exists(self.cache_dir):
                for item in os.listdir(self.cache_dir):
                    item_path = os.path.join(self.cache_dir, item)
                    if os.path.isdir(item_path) and item.startswith("models--"):
                        # Check if this directory contains the model name parts
                        item_lower = item.lower().replace("-", "_").replace(".", "_")
                        model_parts = model_path.lower().replace("-", "_").replace(".", "_").replace("/", "_").split("_")
                        # If most of the model name parts are in the directory name, consider it a match
                        if sum(1 for part in model_parts if len(part) > 2 and part in item_lower) >= len([p for p in model_parts if len(p) > 2]) * 0.6:
                            return True

            return False

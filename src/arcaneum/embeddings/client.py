"""Embedding client utilities with FastEmbed (RDR-002)."""

from fastembed import TextEmbedding
from typing import Dict, List
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from arcaneum.paths import get_models_dir
import threading

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
    """Manages embedding model instances with caching and GPU acceleration (RDR-013 Phase 2)."""

    def __init__(self, cache_dir: str = None, verify_ssl: bool = True, use_gpu: bool = False):
        """Initialize embedding client.

        Args:
            cache_dir: Directory to cache downloaded models (defaults to ~/.arcaneum/models)
            verify_ssl: Whether to verify SSL certificates (set False for self-signed certs)
            use_gpu: Enable GPU acceleration (MPS for Apple Silicon, CUDA for NVIDIA)
                     Default: False (CPU only for backward compatibility)

        Note: SSL configuration must be done before creating EmbeddingClient.
              Use ssl_config.check_and_configure_ssl() or disable_ssl_verification() first.

        GPU Support (RDR-013):
            - SentenceTransformers models (stella, jina-code): MPS on Apple Silicon, CUDA on NVIDIA
            - FastEmbed models (bge-*): CoreML on Apple Silicon (partial support)
        """
        self.cache_dir = cache_dir or str(get_models_dir())
        self.verify_ssl = verify_ssl
        self.use_gpu = use_gpu
        self._device = self._detect_device() if use_gpu else "cpu"
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = self.cache_dir
        self._models: Dict[str, TextEmbedding] = {}

        # Thread lock for GPU operations (prevents segfaults when multiple threads use GPU models)
        self._gpu_lock = threading.Lock() if use_gpu else None

    def _detect_device(self) -> str:
        """Detect best available GPU device (RDR-013 Phase 2).

        Returns:
            "mps" for Apple Silicon, "cuda" for NVIDIA, "cpu" if no GPU available
        """
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"  # Apple Silicon GPU
            elif torch.cuda.is_available():
                return "cuda"  # NVIDIA GPU
        except ImportError:
            pass
        return "cpu"

    def get_device_info(self) -> Dict[str, str]:
        """Get information about the device being used (RDR-013 Phase 2).

        Returns:
            Dictionary with device information
        """
        return {
            "device": self._device,
            "gpu_enabled": self.use_gpu,
            "gpu_available": self._device != "cpu"
        }

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

                # Configure ONNX Runtime providers for GPU (RDR-013 Phase 2)
                providers = None
                if self.use_gpu and self._device == "mps":
                    try:
                        import onnxruntime as ort
                        available_providers = ort.get_available_providers()
                        if "CoreMLExecutionProvider" in available_providers:
                            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                            # Note: Some models may show CoreML warnings but will run in hybrid mode
                    except Exception:
                        pass  # Fallback to CPU if CoreML setup fails

                # Use local_files_only if model is cached to prevent network calls
                # This is the official FastEmbed parameter for offline mode
                self._models[model_name] = TextEmbedding(
                    model_name=config["name"],
                    cache_dir=self.cache_dir,
                    local_files_only=is_cached,  # Skip network access if cached
                    providers=providers  # GPU acceleration if available
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
                # GPU acceleration via device parameter (RDR-013 Phase 2)
                model_obj = SentenceTransformer(
                    config["name"],
                    cache_folder=self.cache_dir,
                    local_files_only=is_cached,  # Skip HuggingFace Hub check if cached
                    device=self._device  # "mps", "cuda", or "cpu"
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
        # Use lock for GPU operations to prevent thread-unsafe access
        if self._gpu_lock:
            with self._gpu_lock:
                return self._embed_impl(texts, model_name)
        else:
            return self._embed_impl(texts, model_name)

    def _embed_impl(self, texts: List[str], model_name: str) -> List[List[float]]:
        """Internal implementation of embedding (called with or without lock).

        Args:
            texts: List of text strings to embed
            model_name: Model identifier

        Returns:
            List of embedding vectors
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
            # Process in batches to prevent hangs
            BATCH_SIZE = 100
            all_embeddings = []

            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                batch_embeddings = list(model.embed(batch))
                all_embeddings.extend(batch_embeddings)

            return all_embeddings

    def embed_parallel(
        self,
        texts: List[str],
        model_name: str,
        max_workers: int = 4,
        batch_size: int = 200,
        timeout: int = 300
    ) -> List[List[float]]:
        """Generate embeddings in parallel batches using ThreadPoolExecutor.

        This method provides 2-4x speedup by processing multiple embedding batches
        concurrently. Particularly effective for large text collections.

        Args:
            texts: List of text strings to embed
            model_name: Model identifier (stella, jina, modernbert, bge)
            max_workers: Number of concurrent workers (default: 4)
            batch_size: Chunk size for each batch (default: 200)
            timeout: Timeout in seconds for batch processing (default: 300)

        Returns:
            List of embedding vectors in original order

        Raises:
            ValueError: If model_name is not recognized

        Example:
            >>> client = EmbeddingClient()
            >>> texts = ["text1", "text2", ..., "text1000"]
            >>> embeddings = client.embed_parallel(texts, "stella", max_workers=4)
        """
        # Pre-allocate result list to maintain order
        all_embeddings = [None] * len(texts)

        # Thread-safe: get model before parallel processing
        # Model loading is done once, shared across threads
        _ = self.get_model(model_name)

        # Create batches with their start indices
        batches = []
        for start_idx in range(0, len(texts), batch_size):
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            batches.append((start_idx, end_idx, batch_texts))

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch jobs
            future_to_batch = {}
            for start_idx, end_idx, batch_texts in batches:
                future = executor.submit(self.embed, batch_texts, model_name)
                future_to_batch[future] = (start_idx, end_idx)

            # Collect results as they complete
            for future in as_completed(future_to_batch):
                start_idx, end_idx = future_to_batch[future]
                try:
                    batch_embeddings = future.result(timeout=timeout)
                    # Place results in correct position
                    all_embeddings[start_idx:end_idx] = batch_embeddings
                except TimeoutError:
                    # Log timeout error
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Batch {start_idx}-{end_idx} timed out (exceeded {timeout}s)")
                    # Fill with None to indicate failure
                    all_embeddings[start_idx:end_idx] = [None] * (end_idx - start_idx)
                except Exception as e:
                    # Log error but don't fail entire batch
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Batch {start_idx}-{end_idx} failed: {e}")
                    # Fill with None to indicate failure
                    all_embeddings[start_idx:end_idx] = [None] * (end_idx - start_idx)

        # Check for any failures
        if None in all_embeddings:
            failed_indices = [i for i, emb in enumerate(all_embeddings) if emb is None]
            raise RuntimeError(
                f"Failed to generate embeddings for {len(failed_indices)} texts at indices: {failed_indices[:10]}..."
            )

        return all_embeddings

    def release_model(self, model_name: str):
        """Release a specific model from memory to free resources.

        Args:
            model_name: Model identifier to release

        Note:
            After calling this, the model will be reloaded on next use.
            GPU cache (CUDA/MPS) is cleared if GPU was enabled.
        """
        if model_name in self._models:
            model = self._models[model_name]

            # Delete the model object
            del self._models[model_name]
            del model

            # Force garbage collection
            import gc
            gc.collect()

            # Clear GPU cache if using GPU
            if self.use_gpu and self._device != "cpu":
                self._clear_gpu_cache()

            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Released model: {model_name}")

    def release_all_models(self):
        """Release all loaded models from memory.

        Note:
            Models will be reloaded on next use.
            GPU cache (CUDA/MPS) is cleared if GPU was enabled.
        """
        model_names = list(self._models.keys())

        for model_name in model_names:
            model = self._models[model_name]
            del model

        self._models.clear()

        # Force garbage collection
        import gc
        gc.collect()

        # Clear GPU cache if using GPU
        if self.use_gpu and self._device != "cpu":
            self._clear_gpu_cache()

        import logging
        logger = logging.getLogger(__name__)
        if model_names:
            logger.info(f"Released {len(model_names)} models: {', '.join(model_names)}")

    def _clear_gpu_cache(self):
        """Clear GPU memory cache (CUDA or MPS).

        Note:
            This helps free GPU memory after releasing models.
        """
        try:
            import torch
            if self._device == "cuda":
                torch.cuda.empty_cache()
                import logging
                logger = logging.getLogger(__name__)
                logger.debug("Cleared CUDA cache")
            elif self._device == "mps":
                torch.mps.empty_cache()
                import logging
                logger = logging.getLogger(__name__)
                logger.debug("Cleared MPS cache")
        except Exception as e:
            # Not fatal if cache clearing fails
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Could not clear GPU cache: {e}")

    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models.

        Returns:
            List of model names currently in memory
        """
        return list(self._models.keys())

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

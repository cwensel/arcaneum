"""Embedding client utilities with FastEmbed (RDR-002)."""

from fastembed import TextEmbedding
from typing import Dict, List, Optional
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from arcaneum.paths import get_models_dir
import threading
import logging
import time

logger = logging.getLogger(__name__)

# Model configurations with dimensions
# Note: "stella" is an alias for bge-large since actual stella (dunzhang/stella_en_1.5B_v5)
# is not available in FastEmbed
#
# Model configurations with multiple backends
#
# IMPORTANT - Memory and Batch Size Configuration:
# ================================================
# Each SentenceTransformers model MUST specify "params_billions" for automatic batch sizing.
# Batch size is derived from model size to prevent GPU OOM errors:
#
#   params_billions >= 1.0  → batch_size = 16  (large models like stella, jina-code-1.5b)
#   params_billions >= 0.3  → batch_size = 32  (medium models like jina-code-0.5b)
#   params_billions <  0.3  → batch_size = 128 (small models)
#
# The batch size calculation happens in memory.py:get_batch_size_for_model_params()
# This prevents the common bug of adding a new large model without adjusting batch size.
#
# See also: memory.py for the batch size derivation logic
EMBEDDING_MODELS = {
    # Code-specific models (SentenceTransformers)
    "jina-code": {
        "name": "jinaai/jina-embeddings-v2-base-code",
        "dimensions": 768,
        "backend": "sentence-transformers",
        "description": "Code-specific (768D, 8K context, legacy v2 model)",
        "available": True,
        "recommended_for": "code",
        "params_billions": 0.137,  # ~137M params
        "max_seq_length": 8192,  # Limit attention memory: O(batch × seq_len²)
    },
    "jina-code-0.5b": {
        "name": "jinaai/jina-code-embeddings-0.5b",
        "dimensions": 896,
        "backend": "sentence-transformers",
        "description": "Code-specific SOTA (896D, 32K context, Sept 2025, fast)",
        "available": True,
        "recommended_for": "code",
        "params_billions": 0.5,  # 500M params, Qwen2 attention needs ~4GB per batch
        # Limit seq_length to control attention memory: O(batch × seq_len²)
        # Model supports 32K but was trained on 512; 8192 is recommended max
        # See: https://huggingface.co/jinaai/jina-code-embeddings-0.5b
        "max_seq_length": 8192
    },
    "jina-code-1.5b": {
        "name": "jinaai/jina-code-embeddings-1.5b",
        "dimensions": 1536,
        "backend": "sentence-transformers",
        "description": "Code-specific SOTA (1536D, 32K context, Sept 2025, highest quality)",
        "available": True,
        "recommended_for": "code",
        "params_billions": 1.5,  # 1.5B params
        "max_seq_length": 8192  # Same as 0.5b - limit attention memory
    },
    "codesage-large": {
        "name": "codesage/codesage-large",
        "dimensions": 1024,
        "backend": "sentence-transformers",
        "description": "CodeSage V2 (1024D, 9 languages, Dec 2024)",
        "available": True,
        "recommended_for": "code",
        "params_billions": 0.4,  # ~400M params
        "max_seq_length": 8192,  # Limit attention memory: O(batch × seq_len²)
    },
    "nomic-code": {
        "name": "nomic-ai/nomic-embed-code",
        "dimensions": 3584,
        "backend": "sentence-transformers",
        "description": "Nomic Code (3584D, 7B params, 6 languages, 2025)",
        "available": True,
        "recommended_for": "code",
        "params_billions": 7.0,  # 7B params - very large
        "max_seq_length": 8192,  # Limit attention memory: O(batch × seq_len²)
    },

    # General purpose models (SentenceTransformers)
    "stella": {
        "name": "dunzhang/stella_en_1.5B_v5",
        "dimensions": 1024,
        "backend": "sentence-transformers",
        "description": "General purpose (1024D, high quality for docs/PDFs)",
        "available": True,
        "recommended_for": "pdf",
        "params_billions": 1.5,  # 1.5B params
        "mps_max_batch": 2,  # MPS needs small batches to avoid system lockups on unified memory
        # Note: Model default max_seq_length=512, don't override
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

    # Additional general purpose models
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "backend": "sentence-transformers",
        "description": "MiniLM (384D, lightweight, fast)",
        "available": True,
        "params_billions": 0.022,  # ~22M params
    },
    "gte-base": {
        "name": "thenlper/gte-base",
        "dimensions": 768,
        "backend": "sentence-transformers",
        "description": "GTE Base (768D, general purpose retrieval)",
        "available": True,
        "params_billions": 0.110,  # ~110M params
    },
    "e5-base": {
        "name": "intfloat/e5-base-v2",
        "dimensions": 768,
        "backend": "sentence-transformers",
        "description": "E5 Base v2 (768D, multilingual, strong performance)",
        "available": True,
        "params_billions": 0.110,  # ~110M params
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

        # GPU operations use single-threaded batching for optimal performance
        # GPU models have built-in parallelism; ThreadPoolExecutor + locks cause serialization
        # See: RDR-013 Phase 2, arcaneum-m7hg
        self._gpu_lock = None  # Deprecated: no longer needed with single-threaded embedding

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

    def _get_optimal_batch_size(self, model_name: str) -> int:
        """Calculate optimal batch size based on model and device (arcaneum-i7oa).

        GPU models can process much larger batches efficiently. This method
        calculates optimal batch sizes based on model dimensions and GPU availability.

        Memory analysis shows 1024-text batches use <5MB (<0.1% of typical GPU memory),
        so we're not constrained by GPU memory - we can use larger batches for efficiency.

        IMPORTANT: MPS (Apple Silicon) with large models needs much smaller batches
        to avoid system lockups due to unified memory exhaustion.

        Args:
            model_name: Model identifier

        Returns:
            Optimal batch size for this model
        """
        if not self.use_gpu:
            return 256  # Conservative for CPU

        # MPS with large models: use smaller batch sizes to avoid system lockups
        # Unified memory architecture means GPU memory pressure affects entire system
        # Note: outer batch size mainly affects RAM, inner batch (mps_max_batch) affects GPU
        if self._device == "mps":
            model_config = EMBEDDING_MODELS.get(model_name, {})
            params_billions = model_config.get("params_billions", 0)
            if params_billions >= 1.0:
                return 128  # Large models (stella 1.5B)
            elif params_billions >= 0.3:
                return 256  # Medium models
            else:
                return 512  # Small models

        dimensions = self.get_dimensions(model_name)

        # CUDA: Adaptive sizing based on model dimensions (arcaneum-i7oa)
        # Larger batches = fewer kernel launches = better GPU utilization
        if dimensions <= 384:
            return 1024  # Small models (bge-small: 384D)
        elif dimensions <= 768:
            return 768   # Medium models (jina-code: 768D, bge-base: 768D)
        else:
            return 512   # Large models (stella: 1024D, bge-large: 1024D)

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
                try:
                    self._models[model_name] = TextEmbedding(
                        model_name=config["name"],
                        cache_dir=self.cache_dir,
                        local_files_only=is_cached,  # Skip network access if cached
                        providers=providers  # GPU acceleration if available
                    )
                except Exception as e:
                    # Detect and report network/SSL errors with helpful messages
                    error_msg = str(e).lower()
                    if "ssl" in error_msg or "certificate" in error_msg:
                        raise RuntimeError(
                            f"SSL certificate verification failed while downloading model '{model_name}'.\n"
                            f"For corporate proxies with self-signed certificates, run:\n"
                            f"  export ARC_SSL_VERIFY=false\n\n"
                            f"Original error: {e}"
                        ) from e
                    elif "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
                        raise RuntimeError(
                            f"Network connection failed while downloading model '{model_name}'.\n"
                            f"Please check your internet connection. If using a VPN, try disabling it.\n\n"
                            f"Original error: {e}"
                        ) from e
                    else:
                        # Re-raise other errors as-is
                        raise
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
                # trust_remote_code=True allows custom model architectures like stella
                model_obj = None
                last_error = None

                # Try with local_files_only=True first if cache exists (fast path, no network)
                if is_cached:
                    try:
                        model_obj = SentenceTransformer(
                            config["name"],
                            cache_folder=self.cache_dir,
                            local_files_only=True,  # Skip HuggingFace Hub check if cached
                            device=self._device,  # "mps", "cuda", or "cpu"
                            trust_remote_code=True  # Required for stella and other custom models
                        )
                        model_obj._backend = "sentence-transformers"
                        # Apply max_seq_length limit if configured (arcaneum-mem-leak)
                        # This controls attention memory: O(batch × seq_len²)
                        if "max_seq_length" in config:
                            original_max = model_obj.max_seq_length
                            model_obj.max_seq_length = config["max_seq_length"]
                            logger.info(f"Set {model_name} max_seq_length: {original_max} → {config['max_seq_length']}")
                        self._models[model_name] = model_obj
                    except Exception as e:
                        # If local_files_only fails, cache may be incomplete (e.g., missing custom code)
                        # Save error and try with network access
                        last_error = e

                # If not cached or local_files_only failed, try with network access
                if model_obj is None:
                    try:
                        # If we're retrying after local_files_only failure, show message
                        if last_error is not None:
                            print("   Downloading additional model files...", flush=True, file=sys.stderr)

                        model_obj = SentenceTransformer(
                            config["name"],
                            cache_folder=self.cache_dir,
                            local_files_only=False,  # Allow network access to complete download
                            device=self._device,  # "mps", "cuda", or "cpu"
                            trust_remote_code=True  # Required for stella and other custom models
                        )
                        model_obj._backend = "sentence-transformers"
                        # Apply max_seq_length limit if configured (arcaneum-mem-leak)
                        if "max_seq_length" in config:
                            original_max = model_obj.max_seq_length
                            model_obj.max_seq_length = config["max_seq_length"]
                            logger.info(f"Set {model_name} max_seq_length: {original_max} → {config['max_seq_length']}")
                        self._models[model_name] = model_obj
                    except Exception as e:
                        # Detect and report network/SSL errors with helpful messages
                        error_msg = str(e).lower()
                        if "ssl" in error_msg or "certificate" in error_msg:
                            raise RuntimeError(
                                f"SSL certificate verification failed while downloading model '{model_name}'.\n"
                                f"For corporate proxies with self-signed certificates, run:\n"
                                f"  export ARC_SSL_VERIFY=false\n\n"
                                f"Original error: {e}"
                            ) from e
                        elif "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
                            raise RuntimeError(
                                f"Network connection failed while downloading model '{model_name}'.\n"
                                f"Please check your internet connection. If using a VPN, try disabling it.\n\n"
                                f"Original error: {e}"
                            ) from e
                        else:
                            # Re-raise other errors as-is
                            raise

        return self._models[model_name]

    def embed(self, texts: List[str], model_name: str, batch_size: int = 512, max_internal_batch: int = None) -> List[List[float]]:
        """Generate embeddings for texts using specified model.

        Processes in batches to optimize GPU utilization.

        Note: Single-threaded for GPU models - ThreadPoolExecutor with locks causes serialization.
        GPU models have internal parallelism within batch processing. See arcaneum-m7hg.

        Args:
            texts: List of text strings to embed
            model_name: Model identifier (stella, jina, modernbert, bge)
            batch_size: Batch size for model.encode() (default: 512 for GPU optimization)
            max_internal_batch: Optional maximum for internal batch size (for OOM recovery)

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If model_name is not recognized
        """
        # No lock needed - single-threaded embedding is faster for GPU models
        # GPU parallelism is via large batches (256-512), not thread-level parallelism
        return self._embed_impl(texts, model_name, batch_size=batch_size, max_internal_batch=max_internal_batch)

    def _embed_impl(self, texts: List[str], model_name: str, batch_size: int = 512, max_internal_batch: int = None) -> List[List[float]]:
        """Internal implementation of embedding (called with or without lock).

        Optimized GPU→CPU transfer strategies for reduced overhead (arcaneum-ppa2).

        For SentenceTransformers:
        - Use convert_to_numpy=True to leverage model's optimized GPU→CPU path
        - Use conservative internal batch_size for MPS memory constraints
        - Return numpy rows (not converted to lists) - faster serialization to Qdrant
        - Qdrant accepts both lists and numpy arrays as vectors

        Args:
            texts: List of text strings to embed
            model_name: Model identifier
            batch_size: Batch size for model.encode() (default: 512, but see internal_batch_size logic)
            max_internal_batch: Optional maximum for internal batch size (for OOM recovery)

        Returns:
            List of embedding vectors (as lists or arrays)
        """
        model = self.get_model(model_name)

        # Handle different backends
        if hasattr(model, '_backend') and model._backend == "sentence-transformers":
            # SentenceTransformers: use encode() with convert_to_numpy=True (arcaneum-ppa2)
            # This uses the model's optimized GPU→CPU transfer path.
            # Potential 10-20% speedup on embeddings by reducing tensor→list conversion overhead.

            # CRITICAL: model.encode() batch_size controls GPU memory usage
            # Use dynamic batch sizing based on available memory at runtime
            # This replaces the previous hard-coded values (8/32/64) which caused
            # excessive kernel launches and poor GPU utilization
            if self._device in ("mps", "cuda"):
                from ..utils.memory import get_gpu_memory_info, estimate_safe_batch_size_v2

                available_bytes, _, device_type = get_gpu_memory_info()
                if available_bytes:
                    internal_batch_size = estimate_safe_batch_size_v2(
                        model_name=model_name,
                        available_gpu_bytes=available_bytes,
                        pipeline_overhead_gb=0.3,
                        safety_factor=0.6,
                        device_type=self._device
                    )
                    logger.debug(
                        f"Dynamic internal_batch_size={internal_batch_size} for "
                        f"model.encode() on {self._device} "
                        f"(available: {available_bytes / (1024**3):.1f}GB)"
                    )
                else:
                    # Fallback if memory detection fails
                    internal_batch_size = 64
                    logger.debug(f"Memory detection failed, using fallback internal_batch_size={internal_batch_size}")

                # Apply max_internal_batch limit if specified (for OOM recovery)
                if max_internal_batch is not None and max_internal_batch < internal_batch_size:
                    logger.debug(f"Applying OOM recovery limit: {internal_batch_size} → {max_internal_batch}")
                    internal_batch_size = max_internal_batch
            else:
                # CPU: Use conservative batches
                internal_batch_size = min(batch_size, 256)

            # Disable progress bar - pipeline handles progress display
            # Wrap in MPS error handling for graceful degradation
            try:
                embeddings = model.encode(
                    texts,
                    batch_size=internal_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
            except Exception as e:
                error_msg = str(e).lower()
                # Detect MPS "not enough space" errors (macOS Metal shader graph allocation failures)
                if self._device == "mps" and ("enough space" in error_msg or "mpsgraph" in error_msg):
                    # Try with minimal batch size
                    if internal_batch_size > 1:
                        logger.warning(
                            f"MPS memory error with batch_size={internal_batch_size}, retrying with batch_size=1. "
                            f"Consider using ARC_NO_GPU=1 for CPU mode."
                        )
                        embeddings = model.encode(
                            texts,
                            batch_size=1,
                            show_progress_bar=False,
                            convert_to_numpy=True
                        )
                    else:
                        # batch_size=1 still failing, suggest CPU fallback
                        logger.error(
                            f"MPS failed even with batch_size=1. Use ARC_NO_GPU=1 for CPU mode."
                        )
                        raise RuntimeError(
                            f"MPS GPU memory exhausted. Run with ARC_NO_GPU=1 to use CPU instead."
                        ) from e
                else:
                    raise
            # Return numpy arrays directly - Qdrant Python client accepts numpy.ndarray natively
            # Removing .tolist() conversion saves 5-15% overhead on embeddings (arcaneum-zfch)
            return embeddings
        else:
            # FastEmbed: use embed()
            # Process in batches to prevent hangs
            BATCH_SIZE = 100

            # Pre-allocate result array to avoid incremental list extensions (arcaneum-knl6)
            # First batch determines embedding dimensions
            first_batch = texts[:min(BATCH_SIZE, len(texts))]
            first_embeddings = list(model.embed(first_batch))

            if not first_embeddings:
                return []

            # Get dimensions from first embedding
            dim = len(first_embeddings[0])

            # Pre-allocate numpy array with correct shape and dtype
            import numpy as np
            all_embeddings = np.zeros((len(texts), dim), dtype=np.float32)

            # Fill first batch
            for idx, emb in enumerate(first_embeddings):
                all_embeddings[idx] = emb

            # Process remaining batches and fill array slices (arcaneum-knl6)
            offset = len(first_embeddings)
            for i in range(BATCH_SIZE, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                batch_embeddings = list(model.embed(batch))
                batch_size = len(batch_embeddings)
                all_embeddings[offset:offset + batch_size] = batch_embeddings
                offset += batch_size
                # Release batch references to prevent memory accumulation in loop scope
                del batch_embeddings
                del batch

            return all_embeddings

    def embed_parallel(
        self,
        texts: List[str],
        model_name: str,
        max_workers: int = 4,
        batch_size: int = None,
        timeout: int = 300,
        progress_callback: callable = None,
        on_batch_complete: callable = None,
        accumulate: bool = True
    ) -> Optional[List[List[float]]]:
        """Generate embeddings with batched processing.

        Note: Despite the name "parallel", GPU mode uses SEQUENTIAL batching. The "parallel"
        in the name refers to GPU hardware parallelism WITHIN each batch, not across batches.

        Strategy:
        - GPU models: Sequential batching (one batch at a time) with adaptive sizing (512-1024)
          GPU hardware parallelism processes N chunks within each batch simultaneously
          ThreadPoolExecutor adds overhead without benefit for GPU
        - CPU models: Optional ThreadPoolExecutor across batches (rarely used)

        Current implementation: Sequential batch processing for GPU.
        Large batch sizes (512-1024) maximize GPU utilization (arcaneum-i7oa).

        Streaming mode (accumulate=False):
        When accumulate=False and on_batch_complete is provided, embeddings are passed to the
        callback after each batch and not accumulated in memory. This reduces memory usage
        from O(total_chunks) to O(batch_size), enabling processing of arbitrarily large files.

        Args:
            texts: List of text strings to embed
            model_name: Model identifier (stella, jina, modernbert, bge)
            max_workers: Number of concurrent workers (ignored for GPU, kept for API compatibility)
            batch_size: Chunk size for batches (default: None = auto-optimal, can override with explicit value)
            timeout: Timeout in seconds (ignored in single-threaded mode)
            progress_callback: Optional callback(batch_idx, total_batches) called after each batch completes
            on_batch_complete: Optional callback(batch_idx, start_idx, embeddings) for streaming mode.
                Called after each batch with the batch embeddings. Use with accumulate=False for
                memory-efficient streaming where caller handles each batch (e.g., upload to Qdrant).
            accumulate: If True (default), return all embeddings. If False, don't accumulate
                embeddings in memory - caller must use on_batch_complete to handle each batch.
                Returns None when accumulate=False.

        Returns:
            List of embedding vectors in original order, or None if accumulate=False

        Raises:
            ValueError: If model_name is not recognized

        Note:
            After profiling (arcaneum-c128), single-threaded approach is faster for GPU models
            due to GPU's internal parallelism within batch. See arcaneum-m7hg for details.
            Adaptive batch sizing (arcaneum-i7oa) uses 512-1024 for GPU models to maximize throughput.

        Example:
            >>> client = EmbeddingClient()
            >>> texts = ["text1", "text2", ..., "text1000"]
            >>> embeddings = client.embed_parallel(texts, "stella")  # Uses optimal batch size
            >>> embeddings = client.embed_parallel(texts, "stella", batch_size=256)  # Override
            >>> # Streaming mode - process each batch without accumulating
            >>> def handle_batch(batch_idx, start_idx, embeddings):
            ...     upload_to_qdrant(embeddings)
            >>> client.embed_parallel(texts, "stella", on_batch_complete=handle_batch, accumulate=False)
        """
        # Get model once
        _ = self.get_model(model_name)

        # Use adaptive batch sizing if not explicitly provided (arcaneum-i7oa)
        if batch_size is None:
            batch_size = self._get_optimal_batch_size(model_name)

        # Log batch configuration in debug mode
        logger.debug(f"Embedding {len(texts)} texts with batch_size={batch_size}, use_gpu={self.use_gpu}, device={self._device}")

        # Note: GPU memory warning moved to CLI level (index_pdfs.py, index_source.py)
        # where we have more context about user intent and can distinguish explicit vs auto-tuned batch sizes

        # For GPU models: sequential batching (no ThreadPoolExecutor)
        # GPU hardware provides parallelism WITHIN each batch, not across batches
        if self.use_gpu:
            # Sequential batch processing: one batch completes before next begins
            # GPU hardware processes all N chunks in a batch simultaneously
            #
            # Memory optimization: Pre-allocate numpy array instead of list.extend()
            # List.extend() over-allocates by 25-50% during growth, causing memory bloat.
            # Pre-allocation uses exact memory needed. (arcaneum-q6by)
            # Only allocate if accumulating results.
            import numpy as np
            import gc
            dim = self.get_dimensions(model_name)
            if accumulate:
                all_embeddings = np.zeros((len(texts), dim), dtype=np.float32)
            else:
                all_embeddings = None

            total_batches = (len(texts) + batch_size - 1) // batch_size
            batch_idx = 0
            offset = 0
            chunks_embedded = 0  # Track actual chunks processed for progress

            # Memory leak prevention: Clear MPS/CUDA cache based on model size
            # Large models (stella, jina-code-1.5b, nomic-code) need aggressive
            # cache clearing because attention mechanism allocates large contiguous blocks.
            # (arcaneum-mem-leak)
            #
            # Cache clearing strategy is derived from params_billions:
            #   >= 1.0B params: clear BEFORE each batch (large models)
            #   >= 0.3B params: clear every 3 batches (medium models)
            #   <  0.3B params: clear every 10 batches (small models)
            model_config = EMBEDDING_MODELS.get(model_name, {})
            params_billions = model_config.get("params_billions")

            # Determine cache clearing strategy based on model params
            if params_billions is not None and params_billions >= 1.0:
                cache_clear_interval = 1  # Every batch
                clear_before_batch = True
                model_size_category = "large"
            elif params_billions is not None and params_billions >= 0.3:
                cache_clear_interval = 3  # Every 3 batches
                clear_before_batch = True
                model_size_category = "medium"
            else:
                cache_clear_interval = 10  # Every 10 batches
                clear_before_batch = False
                model_size_category = "small"

            logger.debug(f"Model {model_name} params={params_billions}B ({model_size_category}), cache_clear_interval={cache_clear_interval}, clear_before={clear_before_batch}")

            # OOM recovery: track effective batch size across all batches
            # Start with requested batch_size, reduce on OOM until we reach minimum
            effective_batch_size = batch_size
            min_batch_size = 8  # Minimum viable batch size before giving up

            # For large models, clear GPU cache BEFORE first batch to ensure maximum
            # available memory. This is critical when processing multiple files as
            # memory from PDF extraction may not be fully released yet.
            if clear_before_batch:
                gc.collect()
                self._clear_gpu_cache()
                logger.debug("Cleared GPU cache before first batch")

            for start_idx in range(0, len(texts), batch_size):
                # For memory-hungry models, clear cache BEFORE embedding to maximize
                # available memory for attention allocations (arcaneum-mem-leak)
                if clear_before_batch and batch_idx > 0:
                    gc.collect()
                    self._clear_gpu_cache()
                    logger.debug(f"Cleared GPU cache before batch {batch_idx + 1}")

                batch_start_time = time.time()
                end_idx = min(start_idx + batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                actual_batch_size = len(batch_texts)

                # OOM recovery: retry with progressively smaller internal batches (arcaneum-mem-leak)
                # The max_internal_batch parameter limits the batch size passed to model.encode()
                # Keep halving until we reach min_batch_size or succeed
                batch_embeddings = None
                current_max_internal = effective_batch_size
                batch_retry_count = 0

                while batch_embeddings is None:
                    try:
                        # Synchronize GPU before embedding to surface any pending async errors
                        # Metal/MPS errors can be raised asynchronously after previous operations
                        self._sync_gpu_if_needed()

                        result = self.embed(
                            batch_texts, model_name,
                            batch_size=batch_size,
                            max_internal_batch=current_max_internal if current_max_internal != batch_size else None
                        )

                        # Synchronize again after embedding to catch any errors before considering batch done
                        self._sync_gpu_if_needed()

                        # Validate embeddings - Metal OOM can corrupt results without raising exceptions
                        # The errors are printed to stderr but embeddings may contain NaN/garbage
                        if not self._validate_embeddings(result, len(batch_texts), model_name):
                            raise RuntimeError("GPU produced invalid embeddings (likely OOM corruption)")

                        batch_embeddings = result

                    except Exception as e:
                        # Detect GPU OOM from various sources:
                        # - PyTorch: "out of memory", "CUDA out of memory"
                        # - Metal/MPS: "Insufficient Memory", "kIOGPUCommandBufferCallbackErrorOutOfMemory"
                        # - Generic: "command buffer exited with error status"
                        # - Our validation: "invalid embeddings"
                        error_str = str(e).lower()
                        is_oom = any(pattern in error_str for pattern in [
                            "out of memory",
                            "insufficient memory",
                            "kiogpucommandbuffercallbackerroroutofmemory",
                            "command buffer exited with error status",
                            "mps backend out of memory",
                            "cuda error: out of memory",
                            "invalid embeddings",  # Our validation error
                            "oom corruption",
                        ])

                        # Keep retrying with smaller batches until we hit minimum
                        if is_oom and current_max_internal > min_batch_size:
                            batch_retry_count += 1
                            # Halve the batch size (more gradual than /4)
                            new_max = max(min_batch_size, current_max_internal // 2)

                            # Single clean warning message
                            import sys
                            print(
                                f"\n⚠ GPU OOM: batch size {current_max_internal} → {new_max}, retrying...",
                                file=sys.stderr, flush=True
                            )
                            logger.debug(
                                f"OOM at batch {batch_idx + 1}, reducing internal batch size: "
                                f"{current_max_internal} → {new_max} (attempt {batch_retry_count})"
                            )
                            current_max_internal = new_max
                            effective_batch_size = new_max  # Remember for future batches

                            # Clear cache and wait for GPU to recover
                            gc.collect()
                            self._clear_gpu_cache()
                            # Brief pause to let GPU recover
                            time.sleep(0.5)
                        elif is_oom:
                            # Already at minimum batch size - provide helpful error message
                            raise RuntimeError(
                                f"GPU out of memory even at minimum batch size ({min_batch_size}).\n\n"
                                f"Suggestions:\n"
                                f"  1. Use CPU instead: --no-gpu\n"
                                f"  2. Close other GPU-intensive applications\n"
                                f"  3. Try a smaller model (e.g., bge-small instead of jina-code)\n"
                                f"  4. Reduce chunk count by filtering files\n\n"
                                f"Original error: {e}"
                            ) from e
                        else:
                            # Not OOM, re-raise original error
                            raise

                batch_elapsed = time.time() - batch_start_time
                chunks_embedded += actual_batch_size
                logger.debug(f"Batch {batch_idx + 1}/{total_batches}: {actual_batch_size} chunks embedded in {batch_elapsed:.2f}s ({actual_batch_size/batch_elapsed:.1f} chunks/s)")

                # Call batch complete callback if provided (for streaming upload)
                if on_batch_complete:
                    on_batch_complete(batch_idx, start_idx, batch_embeddings)

                # Fill pre-allocated array in place (no list over-allocation)
                # Only if accumulating results
                if accumulate:
                    all_embeddings[offset:offset + actual_batch_size] = batch_embeddings
                offset += actual_batch_size

                batch_idx += 1
                if progress_callback:
                    # Pass extended progress info: batch_idx, total_batches, effective_batch_size, chunks_done, total_chunks
                    # Callback can accept 2 args (legacy) or 5 args (extended)
                    import inspect
                    sig = inspect.signature(progress_callback)
                    if len(sig.parameters) >= 5:
                        progress_callback(batch_idx, total_batches, effective_batch_size, chunks_embedded, len(texts))
                    else:
                        progress_callback(batch_idx, total_batches)

                # CRITICAL: Delete batch_embeddings after each iteration to prevent memory leak
                # Without this, the variable persists in loop scope and accumulates memory
                # This must happen BEFORE the periodic cleanup check (arcaneum-mem-leak)
                del batch_embeddings
                del batch_texts

                # Periodic GPU cache clearing to prevent memory leak (arcaneum-mem-leak)
                # MPS/CUDA cache allocations for reuse, but this causes OOM on long jobs.
                # For models with clear_before_batch, this is redundant but harmless.
                if not clear_before_batch and batch_idx % cache_clear_interval == 0:
                    gc.collect()
                    self._clear_gpu_cache()
                    logger.debug(f"Cleared GPU cache after batch {batch_idx}")

            # Final cleanup
            gc.collect()
            self._clear_gpu_cache()

            return all_embeddings

        # For CPU models: ThreadPoolExecutor can provide speedup
        # This is experimental - benchmark results will determine if kept
        else:
            # Pre-allocate result list to maintain order (only if accumulating)
            if accumulate:
                all_embeddings = [None] * len(texts)
            else:
                all_embeddings = None

            # Create batches with their start indices
            batches = []
            for start_idx in range(0, len(texts), batch_size):
                end_idx = min(start_idx + batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                batches.append((start_idx, end_idx, batch_texts))

            # Process batches in parallel for CPU models
            total_batches = len(batches)
            completed_batches = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batch jobs
                future_to_batch = {}
                for batch_idx, (start_idx, end_idx, batch_texts) in enumerate(batches):
                    future = executor.submit(self.embed, batch_texts, model_name)
                    future_to_batch[future] = (batch_idx, start_idx, end_idx)

                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    batch_idx, start_idx, end_idx = future_to_batch[future]
                    try:
                        batch_embeddings = future.result(timeout=timeout)

                        # Call batch complete callback if provided (for streaming upload)
                        # Note: batches may complete out of order with ThreadPoolExecutor
                        if on_batch_complete:
                            on_batch_complete(batch_idx, start_idx, batch_embeddings)

                        # Place results in correct position (only if accumulating)
                        if accumulate:
                            all_embeddings[start_idx:end_idx] = batch_embeddings
                        completed_batches += 1
                        if progress_callback:
                            progress_callback(completed_batches, total_batches)
                    except TimeoutError:
                        # Log timeout error
                        logger.error(f"Batch {start_idx}-{end_idx} timed out (exceeded {timeout}s)")
                        # Fill with None to indicate failure (only if accumulating)
                        if accumulate:
                            all_embeddings[start_idx:end_idx] = [None] * (end_idx - start_idx)
                    except Exception as e:
                        # Log error but don't fail entire batch
                        logger.error(f"Batch {start_idx}-{end_idx} failed: {e}")
                        # Fill with None to indicate failure (only if accumulating)
                        if accumulate:
                            all_embeddings[start_idx:end_idx] = [None] * (end_idx - start_idx)

            # Memory cleanup: Clear futures dictionary to release references (arcaneum-64yl)
            # Future objects hold references to results and callbacks that prevent GC
            del future_to_batch
            del batches
            import gc
            gc.collect()

            # Check for any failures (handle both list and numpy array cases)
            # Only check if accumulating
            if accumulate:
                failed_indices = [i for i, emb in enumerate(all_embeddings) if emb is None]
                if failed_indices:
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

        if model_names:
            logger.info(f"Released {len(model_names)} models: {', '.join(model_names)}")

    def _clear_gpu_cache(self):
        """Clear GPU memory cache (CUDA or MPS).

        Best practice: synchronize() before empty_cache() to ensure all
        GPU operations complete before releasing memory. (arcaneum-mem-leak)

        Note:
            This helps free GPU memory after releasing models or between batches.
        """
        try:
            import torch
            if self._device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache")
            elif self._device == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
                logger.debug("Cleared MPS cache")
        except Exception as e:
            # Not fatal if cache clearing fails
            logger.debug(f"Could not clear GPU cache: {e}")

    def _sync_gpu_if_needed(self):
        """Synchronize GPU to surface any pending async errors.

        Metal/MPS operations can raise errors asynchronously after the Python
        call returns. Calling synchronize() forces any pending GPU operations
        to complete and raises any errors that occurred.

        This is lighter weight than _clear_gpu_cache() - it only syncs, doesn't
        clear memory.
        """
        if not self.use_gpu or self._device == "cpu":
            return

        try:
            import torch
            if self._device == "cuda":
                torch.cuda.synchronize()
            elif self._device == "mps":
                torch.mps.synchronize()
        except Exception as e:
            # Re-raise as this might be a GPU OOM we need to catch
            raise

    def _validate_embeddings(self, embeddings, expected_count: int, model_name: str) -> bool:
        """Validate embeddings are not corrupted by GPU OOM.

        Metal/MPS OOM errors can corrupt embeddings without raising Python exceptions.
        The errors are printed to stderr but the function returns garbage data.

        Args:
            embeddings: The embeddings array to validate
            expected_count: Expected number of embeddings
            model_name: Model name for dimension lookup

        Returns:
            True if embeddings are valid, False if corrupted
        """
        import numpy as np

        try:
            # Check for None
            if embeddings is None:
                logger.debug("Embeddings validation failed: None returned")
                return False

            # Convert to numpy if needed
            if hasattr(embeddings, 'numpy'):
                embeddings = embeddings.numpy()
            elif not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            # Check shape
            expected_dims = self.get_dimensions(model_name)
            if len(embeddings.shape) != 2:
                logger.debug(f"Embeddings validation failed: wrong shape {embeddings.shape}")
                return False

            if embeddings.shape[0] != expected_count:
                logger.debug(f"Embeddings validation failed: count mismatch {embeddings.shape[0]} vs {expected_count}")
                return False

            if embeddings.shape[1] != expected_dims:
                logger.debug(f"Embeddings validation failed: dims mismatch {embeddings.shape[1]} vs {expected_dims}")
                return False

            # Check for NaN or Inf values (common with GPU memory corruption)
            if np.any(np.isnan(embeddings)):
                logger.debug("Embeddings validation failed: contains NaN values")
                return False

            if np.any(np.isinf(embeddings)):
                logger.debug("Embeddings validation failed: contains Inf values")
                return False

            # Check for all-zero vectors (another sign of corruption)
            zero_vectors = np.all(embeddings == 0, axis=1)
            if np.any(zero_vectors):
                zero_count = np.sum(zero_vectors)
                logger.debug(f"Embeddings validation failed: {zero_count} all-zero vectors")
                return False

            return True

        except Exception as e:
            logger.debug(f"Embeddings validation failed with error: {e}")
            return False

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

            # Check if main model cache exists
            if not (os.path.exists(model_dir) and os.path.isdir(model_dir)):
                return False

            # For models with trust_remote_code=True, also check transformers_modules cache
            # These models may have custom Python code in a separate cache location
            # Example: jina models store custom code in ~/.cache/huggingface/modules/transformers_modules/
            # or ~/.arcaneum/models/modules/transformers_modules/
            # We conservatively return False to allow network access for downloading custom code
            # This ensures models work correctly even with custom architectures

            # Check two possible locations for transformers_modules:
            # 1. Inside cache_dir (e.g., ~/.arcaneum/models/modules/)
            # 2. Sibling to cache_dir (e.g., ~/.cache/huggingface/modules/)
            transformers_modules_dir = os.path.join(self.cache_dir, "modules", "transformers_modules")
            if not os.path.exists(transformers_modules_dir):
                # Try sibling directory
                transformers_modules_dir = os.path.join(
                    os.path.dirname(self.cache_dir),
                    "modules",
                    "transformers_modules"
                )

            # If transformers_modules directory doesn't exist at all, model may need custom code
            # Return False to allow download attempt
            if not os.path.exists(transformers_modules_dir):
                return False

            # Check if there's a cached module for this model's organization
            # Extract org name from model path (e.g., "jinaai" from "jinaai/jina-embeddings-v2-base-code")
            if "/" in model_path:
                org_name = model_path.split("/")[0]
                org_modules_dir = os.path.join(transformers_modules_dir, org_name)

                # If org directory doesn't exist, model may need custom code
                if not os.path.exists(org_modules_dir):
                    return False

            # Both main cache and transformers_modules exist, model is fully cached
            return True
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

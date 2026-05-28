"""Embedding client utilities with FastEmbed (RDR-002)."""

import atexit
import logging
import os
import platform
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, TimeoutError, wait
from typing import Any, Dict, List, Optional

from fastembed import TextEmbedding

from arcaneum.paths import get_models_dir

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
        "revision": "516f4baf13dec4ddddda8631e019b5737c8bc250",
        "trust_remote_code": True,
        "dimensions": 768,
        "backend": "sentence-transformers",
        "description": "Code-specific (768D, 2K effective context, legacy v2 model)",
        "available": True,
        "recommended_for": "code",
        "params_billions": 0.137,  # ~137M params
        "query_task": "retrieval.query",
        "document_task": "retrieval.passage",
        # Attention memory is O(batch × seq_len² × heads). 8192 was producing
        # multi-GB Metal driver allocations on files with one long chunk in a
        # mixed batch (jetsam SIGKILL territory). AST chunks max ~400 tokens;
        # line-based fallback chunks max ~2000 tokens; truncating beyond 2048
        # is wasted overhead and the 4× reduction here cuts attention memory
        # 16× in the worst case.
        "max_seq_length": 2048,
        "mps_max_batch": 16,  # MPS: conservative batch to handle files with many long chunks
    },
    "jina-code-0.5b": {
        "name": "jinaai/jina-code-embeddings-0.5b",
        "revision": "4db235132dafbe56a8b9c5f59b59795ecf58a4a7",
        "dimensions": 896,
        "backend": "sentence-transformers",
        "description": "Code-specific SOTA (896D, 32K context, Sept 2025, fast)",
        "available": True,
        "recommended_for": "code",
        "params_billions": 0.5,  # 500M params, Qwen2 attention needs ~4GB per batch
        "query_task": "retrieval.query",
        "document_task": "retrieval.passage",
        # Limit seq_length to control attention memory: O(batch × seq_len²)
        # Model supports 32K but was trained on 512; 8192 is recommended max
        # See: https://huggingface.co/jinaai/jina-code-embeddings-0.5b
        "max_seq_length": 8192,
        "mps_max_batch": 8,  # MPS needs conservative batches due to unified memory
    },
    "jina-code-1.5b": {
        "name": "jinaai/jina-code-embeddings-1.5b",
        "revision": "39aeb4fb9b60f930934c78ae5d749a46287c248a",
        "dimensions": 1536,
        "backend": "sentence-transformers",
        "description": "Code-specific SOTA (1536D, 32K context, Sept 2025, highest quality)",
        "available": True,
        "recommended_for": "code",
        "params_billions": 1.5,  # 1.5B params
        "query_task": "retrieval.query",
        "document_task": "retrieval.passage",
        "max_seq_length": 8192,  # Same as 0.5b - limit attention memory
        "mps_max_batch": 2,  # MPS needs very small batches for 1.5B model (like stella)
    },
    "codesage-large": {
        "name": "codesage/codesage-large",
        "revision": "d672216d9b5cf6bc1babc53cca5f32cff2825c48",
        "trust_remote_code": True,
        "dimensions": 1024,
        "backend": "sentence-transformers",
        "description": "CodeSage V2 (1024D, 9 languages, Dec 2024)",
        "available": True,
        "recommended_for": "code",
        "params_billions": 0.4,  # ~400M params
        "max_seq_length": 8192,  # Limit attention memory: O(batch × seq_len²)
        "mps_max_batch": 8,  # MPS needs conservative batches due to unified memory
    },
    "nomic-code": {
        "name": "nomic-ai/nomic-embed-code",
        "revision": "11114029805cee545ef111d5144b623787462a52",
        "dimensions": 3584,
        "backend": "sentence-transformers",
        "description": "Nomic Code (3584D, 7B params, 6 languages, 2025)",
        "available": True,
        "recommended_for": "code",
        "params_billions": 7.0,  # 7B params - very large
        "max_seq_length": 8192,  # Limit attention memory: O(batch × seq_len²)
        "mps_max_batch": 1,  # MPS: 7B model needs single-item batches to avoid OOM
    },

    # General purpose models (SentenceTransformers)
    "stella": {
        "name": "dunzhang/stella_en_1.5B_v5",
        "revision": "7817065102fd9e1b031fe874e910c01f40b2f001",
        "trust_remote_code": True,
        "dimensions": 1024,
        "backend": "sentence-transformers",
        "description": "General purpose (1024D, high quality for docs/PDFs)",
        "available": True,
        "recommended_for": "pdf",
        "params_billions": 1.5,  # 1.5B params
        "query_prompt_name": "s2p_query",
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
    "arctic-m": {
        "name": "snowflake/snowflake-arctic-embed-m",
        "dimensions": 768,
        "backend": "fastembed",
        "description": "Snowflake Arctic Embed M (768D, stable retrieval default)",
        "available": True,
        "recommended_for": "docs",
    },
    "mxbai-large": {
        "name": "mixedbread-ai/mxbai-embed-large-v1",
        "dimensions": 1024,
        "backend": "fastembed",
        "description": "Mixedbread Embed Large (1024D, high-quality English retrieval)",
        "available": True,
        "recommended_for": "docs",
    },

    # Additional general purpose models
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "dimensions": 384,
        "backend": "sentence-transformers",
        "description": "MiniLM (384D, lightweight, fast)",
        "available": True,
        "params_billions": 0.022,  # ~22M params
    },
    "gte-base": {
        "name": "thenlper/gte-base",
        "revision": "c078288308d8dee004ab72c6191778064285ec0c",
        "dimensions": 768,
        "backend": "sentence-transformers",
        "description": "GTE Base (768D, general purpose retrieval)",
        "available": True,
        "params_billions": 0.110,  # ~110M params
    },
    "e5-base": {
        "name": "intfloat/e5-base-v2",
        "revision": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        "dimensions": 768,
        "backend": "sentence-transformers",
        "description": "E5 Base v2 (768D, multilingual, strong performance)",
        "available": True,
        "params_billions": 0.110,  # ~110M params
        "query_prompt": "query: ",
        "document_prompt": "passage: ",
    },
}


TRUST_REMOTE_CODE_ALLOWLIST = {
    "jinaai/jina-embeddings-v2-base-code": "516f4baf13dec4ddddda8631e019b5737c8bc250",
    "codesage/codesage-large": "d672216d9b5cf6bc1babc53cca5f32cff2825c48",
    "dunzhang/stella_en_1.5B_v5": "7817065102fd9e1b031fe874e910c01f40b2f001",
}


def _sentence_transformer_load_kwargs(
    model_key: str,
    config: Dict,
    *,
    cache_folder: str,
    local_files_only: bool,
    device: str,
) -> Dict:
    """Build SentenceTransformer load kwargs with pinned remote-code policy."""
    model_id = config["name"]
    revision = config.get("revision")
    if not revision:
        raise ValueError(f"SentenceTransformer model '{model_key}' must pin a revision")

    trust_remote_code = bool(config.get("trust_remote_code", False))
    if trust_remote_code:
        allowlisted_revision = TRUST_REMOTE_CODE_ALLOWLIST.get(model_id)
        if allowlisted_revision is None:
            raise ValueError(
                f"SentenceTransformer model '{model_key}' enables trust_remote_code "
                f"but '{model_id}' is not allowlisted"
            )
        if revision != allowlisted_revision:
            raise ValueError(
                f"SentenceTransformer model '{model_key}' enables trust_remote_code "
                "with a revision that does not match the allowlist"
            )

    return {
        "cache_folder": cache_folder,
        "local_files_only": local_files_only,
        "device": device,
        "revision": revision,
        "trust_remote_code": trust_remote_code,
    }


def _unknown_model_error(model_name: str) -> ValueError:
    """Build the canonical ValueError for an unknown embedding model."""
    return ValueError(
        f"Unknown model: {model_name}. "
        f"Available models: {list(EMBEDDING_MODELS.keys())}"
    )


def model_key_for_name(model_name: str) -> Optional[str]:
    """Return the registry key for either a model key or provider model name."""
    if model_name in EMBEDDING_MODELS:
        return model_name
    for key, config in EMBEDDING_MODELS.items():
        if config.get("name") == model_name:
            return key
    return None


def get_embedding_prompt_policy(model_name: str) -> Dict[str, Any]:
    """Return the stable query/document prompt policy for a configured model."""
    model_key = model_key_for_name(model_name)
    if model_key is None:
        raise _unknown_model_error(model_name)

    config = EMBEDDING_MODELS[model_key]
    backend = config.get("backend", "fastembed")
    policy = {
        "version": 1,
        "model": model_key,
        "backend": backend,
        "document": {},
        "query": {},
    }

    if backend == "fastembed":
        policy["document"]["method"] = "embed"
        policy["query"]["method"] = "query_embed"
    else:
        policy["document"]["method"] = "encode"
        policy["query"]["method"] = "encode"

    for role in ("document", "query"):
        for field in ("prompt", "prompt_name", "task"):
            value = config.get(f"{role}_{field}")
            if value:
                policy[role][field] = value

    return policy


def get_embedding_prompt_policies(model_names: str | List[str]) -> Dict[str, Dict[str, Any]]:
    """Return prompt policies keyed by registry model key for one or more models."""
    if isinstance(model_names, str):
        names = [m.strip() for m in model_names.split(",") if m.strip()]
    else:
        names = list(model_names)

    policies: Dict[str, Dict[str, Any]] = {}
    for name in names:
        model_key = model_key_for_name(name)
        if model_key is not None:
            policies[model_key] = get_embedding_prompt_policy(model_key)
    return policies


def _prompted_texts(texts: List[str], policy: Dict[str, Any], role: str) -> List[str]:
    prompt = policy.get(role, {}).get("prompt")
    if not prompt:
        return texts
    return [f"{prompt}{text}" for text in texts]


def _sentence_transformer_encode_kwargs(
    model_name: str,
    prompt_type: str,
) -> Dict[str, Any]:
    """Return SentenceTransformer encode kwargs for task/prompt-name policies."""
    policy = get_embedding_prompt_policy(model_name)
    role_policy = policy.get(prompt_type, {})
    kwargs: Dict[str, Any] = {}
    if role_policy.get("prompt_name"):
        kwargs["prompt_name"] = role_policy["prompt_name"]
    if role_policy.get("task"):
        kwargs["task"] = role_policy["task"]
    return kwargs


class EmbeddingClient:
    """Manages embedding model instances with caching and GPU acceleration (RDR-013 Phase 2)."""

    def __init__(self, cache_dir: str = None, use_gpu: bool = False, cpu_workers: int = None):
        """Initialize embedding client.

        Args:
            cache_dir: Directory to cache downloaded models (defaults to ~/.arcaneum/models)
            use_gpu: Enable GPU acceleration (MPS for Apple Silicon, CUDA for NVIDIA)
                     Default: False (CPU only for backward compatibility)
            cpu_workers: Number of batch workers for parallel embedding in CPU mode
                        Default: 1 (conservative, prevents system crashes from thread over-subscription)

        GPU Support (RDR-013):
            - SentenceTransformers models (stella, jina-code): MPS on Apple Silicon, CUDA on NVIDIA
            - FastEmbed models (bge-*): CoreML on Apple Silicon (partial support)

        CPU Mode Optimization:
            When use_gpu=False, the client processes batches sequentially (cpu_workers=1)
            but uses OMP/MKL threads for parallelism within each batch. This avoids
            thread over-subscription that can cause system crashes with large models.
            Use --cpu-workers to increase if your system can handle more parallelism.
        """
        self.cache_dir = cache_dir or str(get_models_dir())
        self.use_gpu = use_gpu
        self._device = self._detect_device() if use_gpu else "cpu"
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = self.cache_dir
        self._models: Dict[str, TextEmbedding] = {}

        # Set when a GPU encode times out — prevents further GPU use in this session.
        # After a timeout, a daemon thread is still running on the GPU; any new Metal
        # command buffers would conflict and cause a fatal assertion (SIGABRT).
        self._gpu_poisoned = False

        # CPU fallback models: model_name → SentenceTransformer on device="cpu"
        # Lazy-loaded when _gpu_poisoned is True, so remaining files can still be processed.
        self._cpu_fallback_models = {}

        # Deferred GPU cleanup: model_name → (thread, model_ref) for daemon threads
        # still running on GPU after timeout. Cleaned up when thread completes. (RDR-020)
        self._pending_gpu_cleanup = {}

        # Register atexit handler to join daemon threads before Python destroys the
        # MPS allocator. Without this, a daemon thread still running model.encode()
        # on Metal will SIGSEGV when __cxa_finalize tears down MPSAllocator. (RDR-020)
        atexit.register(self._atexit_join_gpu_threads)

        # CPU parallelization settings
        # Default to 1 worker (sequential batching) to avoid thread over-subscription.
        # With cpu_workers=1, we let OMP/MKL handle parallelism within each batch.
        # This is safer: 1 batch × N OMP threads vs N batches × M OMP threads competing.
        # Use --cpu-workers to increase if your system can handle it.
        if cpu_workers is not None:
            self._cpu_workers = max(1, cpu_workers)
        else:
            self._cpu_workers = 1  # Conservative default to prevent system crashes

        # Configure thread environment for CPU mode
        if not use_gpu:
            self._configure_cpu_threading()

    def _experimental_coreml_enabled(self) -> bool:
        """Return True when the user explicitly opts into FastEmbed CoreML.

        CoreMLExecutionProvider can be unstable for large transformer ONNX
        models on Apple Silicon because ORT may split the graph into many
        CoreML/CPU partitions and allocate large native unified-memory buffers
        outside Python's normal accounting. GPU is opt-in for stability, and
        this specific FastEmbed/CoreML provider pair requires an additional
        explicit opt-in.
        """
        return os.environ.get("ARC_EXPERIMENTAL_COREML", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _resolve_fastembed_providers(self, model_name: str):
        """Choose ONNX Runtime providers for FastEmbed models.

        FastEmbed uses ONNX Runtime rather than PyTorch. On macOS Apple Silicon,
        the available GPU provider is CoreMLExecutionProvider, which is not
        stable enough to enable automatically for transformer embedding models.
        Returning CPUExecutionProvider here still allows the rest of the client
        to keep GPU enabled for other model backends in the same run.
        """
        if not (self.use_gpu and self._device == "mps"):
            return None

        is_apple_silicon = sys.platform == "darwin" and platform.machine().lower() in {
            "arm64",
            "aarch64",
        }
        if is_apple_silicon and not self._experimental_coreml_enabled():
            message = (
                f"GPU requested, but FastEmbed/CoreML is experimental for '{model_name}'. "
                "Using CPUExecutionProvider. Set ARC_EXPERIMENTAL_COREML=1 to opt in."
            )
            logger.info(message)
            print(f"   {message}", file=sys.stderr, flush=True)
            return ["CPUExecutionProvider"]

        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            if "CoreMLExecutionProvider" in available_providers:
                return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            logger.debug("CoreML provider detection failed for FastEmbed", exc_info=True)

        return ["CPUExecutionProvider"]

    def _system_memory_available_gb(self) -> Optional[float]:
        """Return currently available system memory in GB, or None if unknown."""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            return None

    def _min_system_available_gb(self) -> float:
        """Configured free-memory floor for accelerator work.

        Apple Silicon uses unified memory: accelerator pressure can starve the
        entire OS, not just the Python process. Default to a conservative floor
        and allow power users to tune it without adding another CLI flag.
        """
        raw = os.environ.get("ARC_MIN_SYSTEM_AVAILABLE_GB", "4")
        try:
            return max(0.0, float(raw))
        except ValueError:
            logger.warning("Invalid ARC_MIN_SYSTEM_AVAILABLE_GB=%r; using 4GB", raw)
            return 4.0

    def _maybe_disable_gpu_for_memory_pressure(self, model_name: str) -> bool:
        """Disable further GPU work if system memory is already too low.

        Returns True when GPU was newly disabled. This guard runs before a batch
        starts so the process can fall back while the system is still responsive.
        """
        if not self.use_gpu or self._device == "cpu" or self._gpu_poisoned:
            return False

        available_gb = self._system_memory_available_gb()
        min_available_gb = self._min_system_available_gb()
        if available_gb is None or available_gb >= min_available_gb:
            return False

        self._gpu_poisoned = True
        if model_name in self._models:
            # Drop the active accelerator model reference so subsequent
            # get_model() calls load the CPU fallback for SentenceTransformers.
            # FastEmbed models already use CPUExecutionProvider by default on
            # Apple Silicon, so this is mainly for PyTorch MPS/CUDA models.
            self._models.pop(model_name, None)
            try:
                import gc
                gc.collect()
                self._clear_gpu_cache()
            except Exception:
                logger.debug("GPU cleanup after memory-pressure fallback failed", exc_info=True)

        logger.warning(
            "Disabling GPU for this session before embedding '%s': "
            "system available memory %.2fGB is below floor %.2fGB",
            model_name,
            available_gb,
            min_available_gb,
        )
        print(
            f"  Low system memory ({available_gb:.1f}GB available) — "
            f"falling back to CPU for remaining embedding work.",
            file=sys.stderr,
            flush=True,
        )
        return True

    def _get_cpu_fallback_model(self, model_name: str):
        """Load a fresh SentenceTransformer on CPU for fallback after GPU poisoning.

        Creates a completely new model instance on CPU — no shared state with the
        GPU model. Tries local_files_only=True first to preserve offline
        fallback when local files are complete, then mirrors the main
        SentenceTransformers loader by retrying with network access after a
        local load failure.
        Cached in _cpu_fallback_models so it's only loaded once per model.
        """
        if model_name in self._cpu_fallback_models:
            return self._cpu_fallback_models[model_name]

        from sentence_transformers import SentenceTransformer
        config = EMBEDDING_MODELS[model_name]
        try:
            model = SentenceTransformer(
                config["name"],
                **_sentence_transformer_load_kwargs(
                    model_name,
                    config,
                    cache_folder=self.cache_dir,
                    local_files_only=True,
                    device="cpu",
                ),
            )
        except Exception:
            model = SentenceTransformer(
                config["name"],
                **_sentence_transformer_load_kwargs(
                    model_name,
                    config,
                    cache_folder=self.cache_dir,
                    local_files_only=False,
                    device="cpu",
                ),
            )
        if "max_seq_length" in config:
            model.max_seq_length = config["max_seq_length"]
        # Mark backend so _embed_impl routes to encode() path, not embed() (FastEmbed)
        model._backend = "sentence-transformers"
        self._cpu_fallback_models[model_name] = model
        return model

    # CPU fallback encode sizing: keep peak memory bounded when a client that
    # started in GPU mode transitions to CPU after poisoning. Full-file encode
    # on a transformer with 8K max_seq_length at batch=32 with unbounded
    # OMP/tokenizer threads can drive RSS into jetsam-kill territory on macOS.
    _CPU_FALLBACK_OUTER_BATCH = 32
    _CPU_FALLBACK_INNER_BATCH = 8

    def _ensure_cpu_fallback_threading(self):
        """Constrain OMP/MKL/tokenizer threads before running a CPU encode.

        Two separate concerns:

        1. _configure_cpu_threading() sets OMP/MKL env vars, but those are
           only read by PyTorch at torch-import time. A GPU-started client
           has already imported torch before this runs, so env-var changes
           here are cosmetic for torch's own thread pool.
        2. torch.set_num_threads() / set_num_interop_threads() mutate the
           live thread pool and must be called here to actually get parallel
           CPU encode. Without this, MPS-started processes can end up with
           torch.get_num_threads() == 1, producing single-core CPU encodes
           that run for minutes per file while the daemon thread still
           holds MPS state.

        Idempotent: env vars are only set if absent; torch setters are
        cheap and the values are stable across calls.
        """
        self._configure_cpu_threading()

        # Mutate torch's live thread pool directly — env vars alone are too
        # late once torch is imported. cpu_count - 2 leaves headroom for
        # the hung MPS daemon thread and the main process.
        #
        # Note: torch.set_num_threads() is process-global; once the client has
        # fallen back after GPU poisoning, the _gpu_poisoned flag remains sticky
        # for the rest of the session.
        try:
            import torch
            available_cores = os.cpu_count() or 4
            target_threads = max(1, available_cores - 2)
            if self._cpu_workers > 1:
                target_threads = max(1, available_cores // self._cpu_workers)
            torch.set_num_threads(target_threads)
            try:
                torch.set_num_interop_threads(max(1, target_threads // 2))
            except RuntimeError:
                # set_num_interop_threads only accepted before parallel work begins;
                # if torch has already dispatched inter-op work, this raises.
                pass
            logger.debug(
                f"CPU fallback: torch.set_num_threads({target_threads}) "
                f"for {available_cores} cores, cpu_workers={self._cpu_workers}"
            )
        except Exception as e:
            logger.debug(f"Could not set torch thread count: {e}")

    def _encode_on_cpu_fallback(
        self,
        cpu_model,
        texts: List[str],
        model_name: str,
        prompt_type: str,
    ):
        """Run model.encode on CPU with bounded memory.

        Used in two paths: (1) explicit CPU mode (use_gpu=False or _device=="cpu"),
        and (2) post-poisoning fallback when MPS/CUDA has been disabled mid-session.
        Splits `texts` into outer batches and uses a small inner batch_size so peak
        RSS stays bounded regardless of how many chunks the caller passes. Returns
        a numpy array matching what a single cpu_model.encode() call would have
        returned.
        """
        import numpy as np

        self._ensure_cpu_fallback_threading()

        encode_kwargs = _sentence_transformer_encode_kwargs(model_name, prompt_type)

        if len(texts) <= self._CPU_FALLBACK_OUTER_BATCH:
            return cpu_model.encode(
                texts,
                batch_size=self._CPU_FALLBACK_INNER_BATCH,
                show_progress_bar=False,
                convert_to_numpy=True,
                **encode_kwargs,
            )

        chunks = []
        for start in range(0, len(texts), self._CPU_FALLBACK_OUTER_BATCH):
            end = min(start + self._CPU_FALLBACK_OUTER_BATCH, len(texts))
            chunks.append(cpu_model.encode(
                texts[start:end],
                batch_size=self._CPU_FALLBACK_INNER_BATCH,
                show_progress_bar=False,
                convert_to_numpy=True,
                **encode_kwargs,
            ))
        return np.concatenate(chunks, axis=0)

    def _try_deferred_gpu_cleanup(self) -> bool:
        """Reclaim GPU resources from completed daemon threads (RDR-020).

        After a GPU timeout, the daemon thread holds a closure reference to the model.
        Once the thread completes, we can safely delete the model ref and clear GPU cache.

        Returns:
            True if any cleanup occurred, False otherwise.
        """
        if not self._pending_gpu_cleanup:
            return False

        import gc
        cleaned = False
        finished = []

        for name, (thread, model_ref) in self._pending_gpu_cleanup.items():
            if not thread.is_alive():
                finished.append(name)
                del model_ref
                cleaned = True
                logger.info(f"Daemon thread for '{name}' completed, releasing GPU model ref.")

        for name in finished:
            del self._pending_gpu_cleanup[name]

        if cleaned:
            gc.collect()
            # Safe to clear GPU cache now — no active command buffers
            self._clear_gpu_cache()
            logger.info("Deferred GPU cleanup complete.")

        return cleaned

    def _atexit_join_gpu_threads(self):
        """Join pending daemon threads at process exit to prevent SIGSEGV (RDR-020).

        When Python exits, __cxa_finalize destroys the MPS allocator. If a daemon
        thread is still running model.encode() on Metal, it will SIGSEGV trying to
        use the destroyed allocator. This handler waits for daemon threads to finish
        before Python's cleanup runs.

        Tradeoff: the daemon thread holds a closure reference to the GPU model
        (~3 GB), so GPU memory is not reclaimed until the thread completes. If the
        Metal command buffer is permanently stuck, the 300s join timeout here will
        still expire and we proceed to interpreter shutdown — which may then
        SIGSEGV during allocator teardown. A stuck GPU thread is already a
        terminal state for this process; the warning logged below is the signal
        that shutdown may be noisy. We do not force-terminate because a clean
        exit is preferable whenever the GPU does in fact drain within the window.
        """
        if not self._pending_gpu_cleanup:
            return

        for name, (thread, _model_ref) in list(self._pending_gpu_cleanup.items()):
            if thread.is_alive():
                logger.info(f"Waiting for GPU daemon thread '{name}' to finish before exit...")
                thread.join(timeout=300)  # 5 min max wait
                if thread.is_alive():
                    logger.warning(f"GPU daemon thread '{name}' did not finish within 300s.")

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

    def _configure_cpu_threading(self):
        """Configure thread environment for optimal CPU parallelism.

        Sets environment variables for ONNX Runtime and PyTorch to use
        appropriate thread counts for CPU-only mode. This improves throughput
        when running with --no-gpu by allowing better CPU utilization.

        Strategy: With default cpu_workers=1 (sequential batching), we let
        OMP/MKL use most available cores for parallelism within each batch.
        This avoids thread over-subscription that causes system crashes.
        """
        # Calculate OMP threads: use most cores for within-batch parallelism
        # Leave 2 cores for system tasks to prevent complete system lockup
        available_cores = os.cpu_count() or 4
        omp_threads = max(1, available_cores - 2)

        # If user specified multiple cpu_workers, reduce OMP threads proportionally
        # to avoid over-subscription (workers × OMP threads should stay <= cores)
        if self._cpu_workers > 1:
            omp_threads = max(1, available_cores // self._cpu_workers)

        cpu_threads = str(omp_threads)

        # OMP_NUM_THREADS controls OpenMP parallelism used by PyTorch/ONNX
        if 'OMP_NUM_THREADS' not in os.environ:
            os.environ['OMP_NUM_THREADS'] = cpu_threads
            logger.debug(f"Set OMP_NUM_THREADS={cpu_threads} for CPU parallelism (cores={available_cores}, workers={self._cpu_workers})")

        # MKL_NUM_THREADS for Intel MKL (used by NumPy/PyTorch on Intel CPUs)
        if 'MKL_NUM_THREADS' not in os.environ:
            os.environ['MKL_NUM_THREADS'] = cpu_threads

        # Disable tokenizers parallelism by default - it adds another layer of threads
        # that can cause over-subscription. Only enable if explicitly set.
        if 'TOKENIZERS_PARALLELISM' not in os.environ:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            logger.debug("Disabled TOKENIZERS_PARALLELISM to prevent over-subscription")

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

        CPU Mode: Uses larger batches (512) since memory isn't constrained by GPU.
        This reduces Python overhead from batch processing loops.

        Args:
            model_name: Model identifier

        Returns:
            Optimal batch size for this model
        """
        if not self.use_gpu:
            return 512  # Larger batches for CPU (no GPU memory constraints)

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
            raise _unknown_model_error(model_name)

        # When GPU is poisoned, return CPU fallback for sentence-transformers models
        # instead of loading a new GPU model (RDR-020).
        if self._gpu_poisoned:
            config = EMBEDDING_MODELS[model_name]
            if config.get("backend") == "sentence-transformers":
                return self._get_cpu_fallback_model(model_name)

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

                # Configure ONNX Runtime providers. GPU remains enabled by
                # default for PyTorch-backed models, but FastEmbed/CoreML is
                # opt-in because graph partitioning can exhaust Apple unified
                # memory outside Python's RSS accounting.
                providers = self._resolve_fastembed_providers(model_name)

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
                import sys

                from sentence_transformers import SentenceTransformer

                # Warn about large models on MPS - risk of system lockup
                params_billions = config.get("params_billions", 0)
                if self._device == "mps" and params_billions >= 1.0:
                    logger.warning(
                        f"Loading {model_name} ({params_billions}B params) on MPS. "
                        "Large models can put heavy pressure on Apple unified memory. "
                        "For the stable default, omit --gpu or use a smaller FastEmbed model "
                        "(e.g., arctic-m)."
                    )
                    print(
                        f"   Warning: --gpu requested for {model_name} ({params_billions}B params) on MPS.\n"
                        f"     This may put heavy pressure on Apple unified memory.\n"
                        f"     For the stable default, omit --gpu or use --models arctic-m.",
                        flush=True, file=sys.stderr
                    )

                # Check if model is cached to avoid unnecessary network calls
                is_cached = self.is_model_cached(model_name)

                # Show loading indicator for models that take time
                if not is_cached:
                    print("   Downloading model files...", flush=True, file=sys.stderr)

                # SentenceTransformer handles download progress automatically via HuggingFace.
                # Model revisions are pinned, and trust_remote_code is only enabled for
                # allowlisted model/revision pairs.
                model_obj = None
                last_error = None

                # Try with local_files_only=True first if cache exists (fast path, no network)
                if is_cached:
                    try:
                        model_obj = SentenceTransformer(
                            config["name"],
                            **_sentence_transformer_load_kwargs(
                                model_name,
                                config,
                                cache_folder=self.cache_dir,
                                local_files_only=True,
                                device=self._device,
                            ),
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
                            **_sentence_transformer_load_kwargs(
                                model_name,
                                config,
                                cache_folder=self.cache_dir,
                                local_files_only=False,
                                device=self._device,
                            ),
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

    def embed(
        self,
        texts: List[str],
        model_name: str,
        batch_size: int = 512,
        max_internal_batch: int = None,
        prompt_type: str = "document",
    ) -> List[List[float]]:
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
        return self._embed_impl(
            texts,
            model_name,
            batch_size=batch_size,
            max_internal_batch=max_internal_batch,
            prompt_type=prompt_type,
        )

    def _embed_impl(
        self,
        texts: List[str],
        model_name: str,
        batch_size: int = 512,
        max_internal_batch: int = None,
        prompt_type: str = "document",
    ) -> List[List[float]]:
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
        if prompt_type not in {"document", "query"}:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        model_config = EMBEDDING_MODELS.get(model_name, {})
        if model_config.get("backend") == "sentence-transformers":
            self._maybe_disable_gpu_for_memory_pressure(model_name)

        model = self.get_model(model_name)

        prompt_policy = get_embedding_prompt_policy(model_name)
        prompt = prompt_policy.get(prompt_type, {}).get("prompt", "")

        # Pre-truncate texts that exceed safe character limit to prevent OOM
        # Generated code (OpenAPI, protobuf) can have high token density where
        # 1 char ≈ 1 token. Use conservative ratio of 2 chars/token.
        # This prevents tokenizer from allocating massive buffers before truncation.
        max_seq_length = model_config.get("max_seq_length", 8192)
        max_chars = max_seq_length * 2  # Conservative: assume 0.5 tokens/char worst case
        max_source_chars = max_chars - len(prompt)
        if max_source_chars <= 0:
            raise RuntimeError(
                f"Embedding prompt for {model_name}/{prompt_type} is longer than "
                f"the safe character limit ({max_chars} chars)."
            )

        # Log chunk sizes for debugging OOM issues
        max_text_len = max(len(t) for t in texts) if texts else 0

        if max_text_len > max_chars * 0.8:
            logger.warning(
                f"Large chunks detected: max={max_text_len} chars, limit={max_chars} chars "
                f"(model={model_name}, max_seq_length={max_seq_length})"
            )

        truncated_count = 0
        safe_texts = []
        for text in texts:
            if len(text) > max_source_chars:
                safe_texts.append(text[:max_source_chars])
                truncated_count += 1
            else:
                safe_texts.append(text)

        if truncated_count > 0:
            logger.warning(
                f"Embedding safety clipped {truncated_count}/{len(texts)} oversized texts before "
                f"embedding; content beyond {max_source_chars} chars is not represented in vectors. "
                f"This indicates upstream chunking should split smaller chunks "
                f"(model={model_name}, max_seq_length={max_seq_length})."
            )

        texts = _prompted_texts(safe_texts, prompt_policy, prompt_type)

        # Additional safeguard: if we still have very large texts after truncation,
        # something is wrong - refuse to process to prevent OOM
        remaining_large = [i for i, t in enumerate(texts) if len(t) > max_chars]
        if remaining_large:
            raise RuntimeError(
                f"BUG: {len(remaining_large)} texts still exceed {max_chars} chars after truncation. "
                f"Sizes: {[len(texts[i]) for i in remaining_large[:5]]}"
            )

        # Handle different backends
        if hasattr(model, '_backend') and model._backend == "sentence-transformers":
            # SentenceTransformers: use encode() with convert_to_numpy=True (arcaneum-ppa2)
            # This uses the model's optimized GPU→CPU transfer path.
            # Potential 10-20% speedup on embeddings by reducing tensor→list conversion overhead.

            # CRITICAL: model.encode() batch_size controls GPU memory usage
            # Use dynamic batch sizing based on available memory at runtime
            # This replaces the previous hard-coded values (8/32/64) which caused
            # excessive kernel launches and poor GPU utilization
            if self._device in ("mps", "cuda") and not self._gpu_poisoned:
                from ..utils.memory import estimate_safe_batch_size_v2, get_gpu_memory_info

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

            # For files with many chunks, process in outer batches to avoid buffer allocation failures
            # Metal/MPS can fail with "Invalid buffer size" when trying to allocate output buffers
            # for hundreds of embeddings at once, even with small internal batch_size
            MAX_OUTER_BATCH = 128  # Process at most 128 texts per model.encode() call
            import numpy as np

            # Sort by length so each internal batch pads to similar-length sequences,
            # not to the longest sequence in a heterogenous mix. Without this, one
            # near-max-length chunk in a batch of 16 inflates attention memory by
            # the full padded shape (batch × max_seq² × heads). Files with mixed
            # short and long chunks were the worst offenders for MPS driver growth.
            # We unsort the results before returning so callers see original order.
            sort_idx = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            sorted_texts = [texts[i] for i in sort_idx]

            if len(sorted_texts) > MAX_OUTER_BATCH:
                # Process in outer batches to avoid large buffer allocations
                import gc
                logger.debug(f"Large input ({len(sorted_texts)} texts), processing in {MAX_OUTER_BATCH}-text outer batches")

                dim = self.get_dimensions(model_name)
                sorted_embeddings = np.zeros((len(sorted_texts), dim), dtype=np.float32)
                offset = 0

                for start_idx in range(0, len(sorted_texts), MAX_OUTER_BATCH):
                    end_idx = min(start_idx + MAX_OUTER_BATCH, len(sorted_texts))
                    batch_texts = sorted_texts[start_idx:end_idx]

                    # Clear cache before each outer batch to prevent fragmentation
                    if self._device in ("mps", "cuda") and not self._gpu_poisoned and start_idx > 0:
                        gc.collect()
                        self._clear_gpu_cache()

                    batch_embeddings = self._encode_with_oom_recovery(
                        model, batch_texts, internal_batch_size, model_name, prompt_type
                    )
                    sorted_embeddings[offset:offset + len(batch_texts)] = batch_embeddings
                    offset += len(batch_texts)

                    # Release batch references
                    del batch_embeddings
                    del batch_texts
            else:
                # Small input - process all at once
                sorted_embeddings = self._encode_with_oom_recovery(
                    model, sorted_texts, internal_batch_size, model_name, prompt_type
                )

            # Unsort back to original order. sort_idx[i] = original index of
            # sorted position i, so embeddings[sort_idx[i]] = sorted_embeddings[i].
            embeddings = np.zeros_like(sorted_embeddings)
            for sorted_pos, orig_pos in enumerate(sort_idx):
                embeddings[orig_pos] = sorted_embeddings[sorted_pos]

            # Validate embeddings before returning - MPS OOM can corrupt results without raising
            # The Metal driver may print errors to stderr but return garbage embeddings
            if not self._validate_embeddings(embeddings, len(texts), model_name):
                raise RuntimeError(
                    "GPU produced invalid embeddings (likely OOM corruption). "
                    "Omit --gpu to use the stable CPU default."
                )

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

            # Validate embeddings before returning
            if not self._validate_embeddings(all_embeddings, len(texts), model_name):
                raise RuntimeError("FastEmbed produced invalid embeddings")

            return all_embeddings

    def _encode_with_oom_recovery(
        self,
        model,
        texts: List[str],
        internal_batch_size: int,
        model_name: str,
        prompt_type: str = "document",
        encode_timeout: int = 120,
    ):
        """Encode texts with OOM recovery for MPS/CUDA.

        Metal/MPS OOM errors can occur in two ways:
        1. Python exception is raised (we catch and retry)
        2. Error printed to stderr but function returns corrupted data (we validate and retry)
        3. GPU hangs indefinitely retrying at the C++ level (we timeout via thread)

        This method handles all three cases.

        Args:
            model: SentenceTransformer model
            texts: List of texts to encode
            internal_batch_size: Batch size for model.encode()
            model_name: Model name for logging
            encode_timeout: Maximum seconds to wait for a single encode call (default: 120)

        Returns:
            numpy array of embeddings

        Raises:
            RuntimeError: If GPU memory is exhausted even at batch_size=1, or encode times out
        """
        # Attempt deferred GPU cleanup between files (RDR-020). Cleanup only
        # releases completed daemon-thread model refs; poisoning remains sticky
        # because re-entering MPS/CUDA after a native timeout is unsafe.
        if self._gpu_poisoned:
            self._try_deferred_gpu_cleanup()

        if self._gpu_poisoned:
            cpu_model = self._get_cpu_fallback_model(model_name)
            logger.info(f"GPU poisoned, falling back to CPU for {len(texts)} texts")
            return self._encode_on_cpu_fallback(cpu_model, texts, model_name, prompt_type)

        # CPU short-circuit: the daemon-thread + timeout + poisoning machinery below
        # only makes sense for MPS/CUDA hangs at the Metal/CUDA C++ level. On CPU the
        # 120s timeout misfires on legitimate slow encodes, and the "fallback" path
        # would spawn a second CPU encode that competes with the still-running first
        # one for OMP threads and RAM. Run inline with bounded batching instead.
        if self._device == "cpu":
            return self._encode_on_cpu_fallback(model, texts, model_name, prompt_type)

        import gc
        import sys

        mps_oom_patterns = [
            "enough space",
            "mpsgraph",
            "mps backend out of memory",
            "command buffer exited with error",
            "invalid buffer size",  # Metal buffer allocation failure
        ]

        def try_encode(batch_size: int):
            """Try to encode and validate, returning None if OOM/corruption detected.

            Runs model.encode() in a daemon thread with a timeout to prevent
            infinite hangs when the GPU retries failed Metal command buffers
            at the C++ level (where Python's try/except can't intervene).
            """
            # Use a container to pass result/exception back from thread
            container = {'result': None, 'error': None}

            def _run_encode():
                try:
                    self._sync_gpu_if_needed()

                    result = model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        **_sentence_transformer_encode_kwargs(model_name, prompt_type),
                    )

                    self._sync_gpu_if_needed()
                    container['result'] = result
                except Exception as e:
                    container['error'] = e

            thread = threading.Thread(target=_run_encode, daemon=True)
            thread.start()
            thread.join(timeout=encode_timeout)

            if thread.is_alive():
                # Thread is stuck - GPU is hanging.
                # Do NOT call _clear_gpu_cache() here — the daemon thread is still
                # executing model.encode() on the GPU. Clearing the cache while Metal
                # command buffers are in-flight causes a fatal assertion:
                #   "commit an already committed command buffer" → SIGABRT
                # The daemon thread will eventually finish or die on its own.
                #
                # Poison the GPU so no further encode attempts touch it this session.
                self._gpu_poisoned = True
                logger.warning(
                    f"model.encode() timed out after {encode_timeout}s at batch_size={batch_size} "
                    f"for {len(texts)} texts — GPU likely hung on Metal OOM retry loop. "
                    f"GPU is now disabled for this session, falling back to CPU."
                )
                import sys
                print(
                    "  GPU encode timed out — falling back to CPU for remaining work.",
                    file=sys.stderr, flush=True
                )

                # Release GPU model from self._models to prevent OOM (RDR-020).
                # The daemon thread holds a closure reference to `model`, so the actual
                # GPU memory (~3GB) stays until the thread completes. But removing from
                # self._models prevents get_model() from returning it and prevents loading
                # a second GPU copy.
                if model_name in self._models:
                    gpu_model_ref = self._models.pop(model_name)
                    self._pending_gpu_cleanup[model_name] = (thread, gpu_model_ref)
                    logger.info(
                        f"Removed GPU model '{model_name}' from active models. "
                        f"Daemon thread holds closure ref; deferred cleanup pending."
                    )

                # Fall back to CPU for these texts instead of raising
                cpu_model = self._get_cpu_fallback_model(model_name)
                return self._encode_on_cpu_fallback(cpu_model, texts, model_name, prompt_type)

            # Thread completed - check for exceptions
            if container['error'] is not None:
                e = container['error']
                error_msg = str(e).lower()
                is_mps_oom = self._device == "mps" and any(p in error_msg for p in mps_oom_patterns)
                if is_mps_oom:
                    logger.debug(f"MPS OOM exception at batch_size={batch_size}: {e}")
                    return None  # Signal to retry
                else:
                    raise e  # Non-OOM error, propagate

            result = container['result']

            # Validate - Metal OOM can corrupt results without raising exceptions
            if not self._validate_embeddings(result, len(texts), model_name):
                logger.debug(f"Embeddings corrupted at batch_size={batch_size}")
                return None  # Treat as OOM, caller will retry

            return result

        # Try with requested batch size
        result = try_encode(internal_batch_size)
        if result is not None:
            return result

        # OOM or corruption detected - clear cache and retry with batch_size=1
        logger.debug(f"OOM/corruption at batch_size={internal_batch_size}, clearing cache and retrying with batch_size=1")
        print(f"  (GPU memory pressure, reducing batch {internal_batch_size} → 1...)", file=sys.stderr, flush=True)

        gc.collect()
        self._clear_gpu_cache()
        time.sleep(0.5)  # Brief pause for GPU recovery

        result = try_encode(1)
        if result is not None:
            return result

        # Still failing - try one more time after aggressive cleanup
        logger.debug("Still failing at batch_size=1, aggressive cleanup and final retry")
        gc.collect()
        self._clear_gpu_cache()
        time.sleep(1.0)  # Longer pause

        result = try_encode(1)
        if result is not None:
            return result

        # Give up
        raise RuntimeError(
            "MPS GPU memory exhausted even at batch_size=1. "
            "Omit --gpu to use the stable CPU default."
        )

    def embed_parallel(
        self,
        texts: List[str],
        model_name: str,
        max_workers: int = None,
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
        - CPU models: ThreadPoolExecutor across batches using cpu_workers (default: cpu_count // 2)
          Larger batch sizes (512) reduce Python overhead

        Current implementation: Sequential batch processing for GPU.
        Large batch sizes (512-1024) maximize GPU utilization (arcaneum-i7oa).

        Streaming mode (accumulate=False):
        When accumulate=False and on_batch_complete is provided, embeddings are passed to the
        callback after each batch and not accumulated in memory. This reduces memory usage
        from O(total_chunks) to O(batch_size), enabling processing of arbitrarily large files.

        Args:
            texts: List of text strings to embed
            model_name: Model identifier (stella, jina, modernbert, bge)
            max_workers: Number of concurrent workers for CPU mode (default: None = use cpu_workers
                        from __init__, ignored for GPU mode)
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
            CPU mode uses ThreadPoolExecutor with cpu_workers (configurable via --cpu-workers flag).

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
            import gc

            import numpy as np
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
                self._maybe_disable_gpu_for_memory_pressure(model_name)

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

                    except KeyboardInterrupt:
                        # User pressed Ctrl-C - clean up and re-raise
                        # This handles interrupts that arrive between GPU operations
                        logger.debug("KeyboardInterrupt received during embedding")
                        gc.collect()
                        self._clear_gpu_cache()
                        raise
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

                            # Brief message - Metal/CUDA already dumped verbose error
                            import sys
                            print(
                                f"  (GPU memory pressure, reducing batch {current_max_internal} → {new_max}...)",
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
                                f"  1. Use CPU instead: omit --gpu\n"
                                f"  2. Close other GPU-intensive applications\n"
                                f"  3. Try a smaller model (e.g., arctic-m or jina-code)\n"
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

        # For CPU models: ThreadPoolExecutor provides speedup via multi-batch parallelism
        # CPU workers configurable via --cpu-workers flag (default: cpu_count // 2)
        else:
            # Pre-allocate result list to maintain order (only if accumulating)
            if accumulate:
                all_embeddings = [None] * len(texts)
            else:
                all_embeddings = None

            # Process batches in parallel for CPU models
            # Use explicit max_workers if provided, otherwise use configured cpu_workers
            effective_workers = max_workers if max_workers is not None else self._cpu_workers
            total_batches = (len(texts) + batch_size - 1) // batch_size
            completed_batches = 0
            logger.debug(
                f"CPU mode: processing {total_batches} batches with {effective_workers} workers"
            )
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                # Keep only a bounded set of batch slices and futures live. This
                # preserves streaming mode's O(batch_size) memory contract while
                # still allowing CPU workers to overlap batches.
                future_to_batch = {}
                completed_by_batch = {}
                next_batch_to_submit = 0
                next_batch_to_emit = 0

                def submit_next_batch():
                    nonlocal next_batch_to_submit
                    if next_batch_to_submit >= total_batches:
                        return
                    start_idx = next_batch_to_submit * batch_size
                    end_idx = min(start_idx + batch_size, len(texts))
                    batch_texts = texts[start_idx:end_idx]
                    future = executor.submit(self.embed, batch_texts, model_name)
                    future_to_batch[future] = (next_batch_to_submit, start_idx, end_idx)
                    next_batch_to_submit += 1

                def replenish_window():
                    while (
                        len(future_to_batch) < effective_workers
                        and len(completed_by_batch) < effective_workers
                        and next_batch_to_submit < total_batches
                    ):
                        submit_next_batch()

                replenish_window()

                while future_to_batch:
                    done, _ = wait(future_to_batch, return_when=FIRST_COMPLETED)
                    for future in done:
                        batch_idx, start_idx, end_idx = future_to_batch.pop(future)
                        try:
                            completed_by_batch[batch_idx] = (
                                start_idx,
                                end_idx,
                                future.result(timeout=timeout),
                                None,
                            )
                        except KeyboardInterrupt:
                            # User pressed Ctrl-C - cancel remaining futures and re-raise
                            logger.debug("KeyboardInterrupt received during CPU embedding")
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise
                        except TimeoutError:
                            logger.error(
                                f"Batch {start_idx}-{end_idx} timed out (exceeded {timeout}s)"
                            )
                            completed_by_batch[batch_idx] = (start_idx, end_idx, None, "timeout")
                        except Exception as e:
                            logger.error(f"Batch {start_idx}-{end_idx} failed: {e}")
                            completed_by_batch[batch_idx] = (start_idx, end_idx, None, "error")

                    replenish_window()

                    while next_batch_to_emit in completed_by_batch:
                        start_idx, end_idx, batch_embeddings, error = completed_by_batch.pop(
                            next_batch_to_emit
                        )
                        if error:
                            if accumulate:
                                all_embeddings[start_idx:end_idx] = [None] * (end_idx - start_idx)
                            next_batch_to_emit += 1
                            replenish_window()
                            continue

                        # Call batch complete callback if provided (for streaming upload)
                        if on_batch_complete:
                            on_batch_complete(next_batch_to_emit, start_idx, batch_embeddings)

                        # Place results in correct position (only if accumulating)
                        if accumulate:
                            all_embeddings[start_idx:end_idx] = batch_embeddings
                        completed_batches += 1
                        if progress_callback:
                            progress_callback(completed_batches, total_batches)
                        next_batch_to_emit += 1
                        replenish_window()

            # Memory cleanup: Clear futures dictionary to release references (arcaneum-64yl)
            # Future objects hold references to results and callbacks that prevent GC
            del future_to_batch
            del completed_by_batch
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

        import torch
        if self._device == "cuda":
            torch.cuda.synchronize()
        elif self._device == "mps":
            torch.mps.synchronize()

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

            # Check for extreme L2 norms - embeddings should be roughly unit normalized
            # Most embedding models produce normalized or near-normalized vectors
            norms = np.linalg.norm(embeddings, axis=1)
            if np.any(norms < 0.01):  # Suspiciously small (near-zero)
                small_count = np.sum(norms < 0.01)
                logger.debug(f"Embeddings validation failed: {small_count} vectors with tiny norm (<0.01)")
                return False

            if np.any(norms > 1000):  # Suspiciously large
                large_count = np.sum(norms > 1000)
                logger.debug(f"Embeddings validation failed: {large_count} vectors with huge norm (>1000)")
                return False

            # Check for duplicate embeddings (GPU may copy same buffer to multiple outputs on OOM)
            if expected_count > 1:
                # Check if all embeddings are identical (catastrophic failure)
                if np.allclose(embeddings[0], embeddings, rtol=1e-5, atol=1e-8):
                    logger.debug("Embeddings validation failed: all embeddings are identical")
                    return False

                # Check for suspiciously low variance across embeddings
                # Different texts should produce different embeddings
                variance = np.var(embeddings, axis=0).mean()
                if variance < 1e-10:
                    logger.debug(f"Embeddings validation failed: suspiciously low variance ({variance:.2e})")
                    return False

            return True

        except Exception as e:
            logger.debug(f"Embeddings validation failed with error: {e}")
            return False

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
            raise _unknown_model_error(model_name)
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
            raise _unknown_model_error(model_name)

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

            if not config.get("trust_remote_code", False):
                return True

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

"""Embedding client utilities with FastEmbed (RDR-002)."""

import atexit
import logging
import os
import platform
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, TimeoutError, wait
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastembed import TextEmbedding

from arcaneum.embeddings.batch_scheduler import (
    BatchBudget,
    BatchResultCollector,
    OversizePolicy,
    schedule_batches,
)
from arcaneum.embeddings.capabilities import (
    BackendSelection,
    coreml_provider_options,
    select_backend,
)
from arcaneum.embeddings.worker_protocol import (
    AcceleratorWorkerSession,
    WorkerConfig,
    WorkerCrashedError,
    WorkerProtocolError,
    WorkerTimeoutError,
)
from arcaneum.paths import get_models_dir, get_state_dir

logger = logging.getLogger(__name__)


class _AcceleratorModelProxy:
    """Parent-side marker; contains no model or native runtime state."""

    _backend = "sentence-transformers"


def _coreml_sentinel_path():
    """Path of the crash sentinel marking an in-flight CoreML session."""
    return get_state_dir() / "coreml-session.json"


def _pid_is_alive(pid: int) -> bool:
    """Best-effort liveness check for the pid recorded in the sentinel."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except Exception:
        pass
    return True


def _write_coreml_sentinel(model_name: str) -> None:
    """Record that this process is about to run experimental CoreML.

    The sentinel is removed on clean interpreter exit. A SIGKILL (e.g. the OS
    reclaiming memory) skips atexit, so a leftover sentinel with a dead pid
    means the previous CoreML run was killed — the next EmbeddingClient warns
    about it via _warn_if_previous_coreml_session_killed().
    """
    import json
    from datetime import datetime, timezone

    try:
        _coreml_sentinel_path().write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "model": model_name,
                    "started": datetime.now(timezone.utc).isoformat(),
                }
            )
        )
        atexit.register(_remove_coreml_sentinel)
    except Exception:
        logger.debug("Could not write CoreML crash sentinel", exc_info=True)


def _remove_coreml_sentinel() -> None:
    """Remove this process's CoreML sentinel (clean-exit path)."""
    import json

    try:
        sentinel = _coreml_sentinel_path()
        if not sentinel.exists():
            return
        if json.loads(sentinel.read_text()).get("pid") == os.getpid():
            sentinel.unlink()
    except Exception:
        logger.debug("Could not remove CoreML crash sentinel", exc_info=True)


# Model configurations with dimensions and multiple backends
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
    # Code-specific models
    "jina-code": {
        "name": "jinaai/jina-embeddings-v2-base-code",
        "dimensions": 768,
        "backend": "fastembed",
        "description": "Code-specific FastEmbed default (768D, lightweight v2 model)",
        "available": True,
        "recommended_for": "code",
    },
    "jina-code-st": {
        "name": "jinaai/jina-embeddings-v2-base-code",
        "revision": "516f4baf13dec4ddddda8631e019b5737c8bc250",
        "trust_remote_code": True,
        "dimensions": 768,
        "backend": "sentence-transformers",
        "description": "Code-specific legacy SentenceTransformers path (768D, 2K context)",
        "available": True,
        "recommended_for": "code",
        "install_extra": "sentence-transformers",
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
        "install_extra": "sentence-transformers",
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
        "install_extra": "sentence-transformers",
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
        "install_extra": "sentence-transformers",
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
        "install_extra": "sentence-transformers",
        "params_billions": 7.0,  # 7B params - very large
        "max_seq_length": 8192,  # Limit attention memory: O(batch × seq_len²)
        "mps_max_batch": 1,  # MPS: 7B model needs single-item batches to avoid OOM
    },
    # General purpose models (SentenceTransformers)
    "qwen3-embed": {
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "dimensions": 1024,
        "backend": "sentence-transformers",
        "description": "Qwen3 Embedding 0.6B (1024D, 32K context, multilingual, no remote code)",
        "available": True,
        "recommended_for": "pdf",
        "install_extra": "sentence-transformers",
        "params_billions": 0.6,  # 600M params
        # The model's config_sentence_transformers.json defines the "query"
        # prompt (instruction-prefixed); documents are embedded unprefixed.
        "query_prompt_name": "query",
        # Model supports 32K context; cap to bound attention memory O(batch × seq_len²)
        "max_seq_length": 8192,
        "mps_max_batch": 8,  # MPS needs conservative batches due to unified memory
    },
    "gemma-embed": {
        "name": "google/embeddinggemma-300m",
        "revision": "57c266a740f537b4dc058e1b0cda161fd15afa75",
        "dimensions": 768,
        "backend": "sentence-transformers",
        "description": "EmbeddingGemma 300M (768D, 2K context, multilingual, gated HF repo)",
        "available": True,
        "recommended_for": "pdf",
        "install_extra": "sentence-transformers",
        "params_billions": 0.3,  # 300M params
        # Downloading requires accepting Google's license on Hugging Face and
        # an authenticated token (hf auth login).
        "gated": True,
        # EmbeddingGemma requires task prefixes on BOTH roles (unlike qwen3-embed,
        # which prefixes only queries). Literal prompts, applied by _prompted_texts;
        # adding prompt_name as well would double-prefix.
        "query_prompt": "task: search result | query: ",
        "document_prompt": "title: none | text: ",
        "max_seq_length": 2048,  # native context
        "mps_max_batch": 16,  # half qwen3-embed's params → 2x its batch
    },
    "stella": {
        "name": "dunzhang/stella_en_1.5B_v5",
        "revision": "7817065102fd9e1b031fe874e910c01f40b2f001",
        "trust_remote_code": True,
        "dimensions": 1024,
        "backend": "sentence-transformers",
        "description": "DEPRECATED: use qwen3-embed (unmaintained remote code)",
        "available": True,
        "deprecated": True,
        "superseded_by": "qwen3-embed",
        "install_extra": "sentence-transformers",
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
        "recommended_for": "multilingual",
    },
    "jina-base-en": {
        "name": "jinaai/jina-embeddings-v2-base-en",
        "dimensions": 768,
        "backend": "fastembed",
        "description": "Jina v2 Base English (768D, 8K context, English-only)",
        "available": True,
    },
    # BGE models (FastEmbed - fast ONNX inference)
    "bge-large": {
        "name": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
        "backend": "fastembed",
        "description": "BGE Large (1024D, general purpose, fast)",
        "available": True,
    },
    "bge": {  # Alias
        "name": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
        "backend": "fastembed",
        "description": "BGE Large (alias for bge-large)",
        "available": True,
    },
    "bge-base": {
        "name": "BAAI/bge-base-en-v1.5",
        "dimensions": 768,
        "backend": "fastembed",
        "description": "BGE Base (768D, balanced)",
        "available": True,
    },
    "bge-small": {
        "name": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "backend": "fastembed",
        "description": "BGE Small (384D, fastest)",
        "available": True,
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
        "install_extra": "sentence-transformers",
        "params_billions": 0.022,  # ~22M params
    },
    "gte-base": {
        "name": "thenlper/gte-base",
        "revision": "c078288308d8dee004ab72c6191778064285ec0c",
        "dimensions": 768,
        "backend": "sentence-transformers",
        "description": "GTE Base (768D, general purpose retrieval)",
        "available": True,
        "install_extra": "sentence-transformers",
        "params_billions": 0.110,  # ~110M params
    },
    "e5-base": {
        "name": "intfloat/e5-base-v2",
        "revision": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        "dimensions": 768,
        "backend": "sentence-transformers",
        "description": "E5 Base v2 (768D, multilingual, strong performance)",
        "available": True,
        "install_extra": "sentence-transformers",
        "params_billions": 0.110,  # ~110M params
        "query_prompt": "query: ",
        "document_prompt": "passage: ",
    },
}

LEGACY_PROMPT_POLICY_MODEL_ALIASES = {
    "jinaai/jina-embeddings-v2-base-code": "jina-code",
    "BAAI/bge-large-en-v1.5": "bge-large",
}


TRUST_REMOTE_CODE_ALLOWLIST = {
    "jinaai/jina-embeddings-v2-base-code": "516f4baf13dec4ddddda8631e019b5737c8bc250",
    "codesage/codesage-large": "d672216d9b5cf6bc1babc53cca5f32cff2825c48",
    "dunzhang/stella_en_1.5B_v5": "7817065102fd9e1b031fe874e910c01f40b2f001",
}


def _ensure_dynamic_cache_compat(cache_cls=None) -> None:
    """Restore DynamicCache.get_usable_length removed in transformers 4.54.

    Stella's pinned remote code (dunzhang/stella_en_1.5B_v5 @ 7817065) calls
    get_usable_length(), which the transformers 4.54 cache refactor removed.
    get_seq_length() is the direct replacement; delegating to it produces
    bit-identical embeddings (verified against a transformers 4.53 baseline).
    No-op on transformers <4.54 where the method still exists. transformers
    5.x is incompatible with the remote code for unrelated reasons (nested
    rope config), hence the <5.0 cap in pyproject.toml.
    """
    if cache_cls is None:
        try:
            from transformers.cache_utils import DynamicCache as cache_cls
        except ImportError:
            return

    if not hasattr(cache_cls, "get_usable_length"):
        cache_cls.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(
            layer_idx
        )


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
        f"Unknown model: {model_name}. Available models: {list(EMBEDDING_MODELS.keys())}"
    )


def model_key_for_name(model_name: str) -> Optional[str]:
    """Return the registry key for either a model key or provider model name."""
    if model_name in EMBEDDING_MODELS:
        return model_name
    matches = [key for key, config in EMBEDDING_MODELS.items() if config.get("name") == model_name]
    if len(matches) == 1:
        return matches[0]
    return None


def prompt_policy_model_key_for_name(model_name: str) -> Optional[str]:
    """Return the prompt-policy model key, including migration aliases."""
    return model_key_for_name(model_name) or LEGACY_PROMPT_POLICY_MODEL_ALIASES.get(model_name)


def get_embedding_prompt_policy(model_name: str) -> Dict[str, Any]:
    """Return the stable query/document prompt policy for a configured model."""
    model_key = prompt_policy_model_key_for_name(model_name)
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
        model_key = prompt_policy_model_key_for_name(name)
        if model_key is None:
            raise _unknown_model_error(name)
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


def _sentence_transformers_missing_error(model_name: str, config: Dict) -> RuntimeError:
    """Build the install guidance for optional SentenceTransformers models."""
    extra = config.get("install_extra", "sentence-transformers")
    return RuntimeError(
        f"Embedding model '{model_name}' uses the optional sentence-transformers backend. "
        f"Install it with 'arcaneum[{extra}]' or choose a FastEmbed model such as "
        "'jina-code' or 'arctic-m'."
    )


class EmbeddingClient:
    """Manages embedding model instances with caching and GPU acceleration (RDR-013 Phase 2)."""

    def __init__(
        self,
        cache_dir: str = None,
        use_gpu: bool = False,
        cpu_workers: int = None,
        allow_experimental_coreml: bool = False,
    ):
        """Initialize embedding client.

        Args:
            cache_dir: Directory to cache downloaded models (defaults to ~/.arcaneum/models)
            use_gpu: Enable GPU acceleration (MPS for Apple Silicon, CUDA for NVIDIA)
                     Default: False (CPU only for backward compatibility)
            cpu_workers: Number of batch workers for parallel embedding in CPU mode
                        Default: 1 (conservative, prevents system crashes from thread over-subscription)
            allow_experimental_coreml: Authorize the experimental CoreML provider for
                        FastEmbed models on Apple Silicon. Set True when the user
                        explicitly opted into GPU (e.g., passed --gpu); paths that
                        enable GPU implicitly should leave this False so CoreML
                        stays gated behind ARC_EXPERIMENTAL_COREML.

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
        self._allow_experimental_coreml = allow_experimental_coreml
        self._warn_if_previous_coreml_session_killed()
        self._device = self._detect_device() if use_gpu else "cpu"
        self._backend_selections: dict[str, BackendSelection] = {}
        self._backend_fallback_reasons: dict[str, str] = {}
        self._worker_restart_count = 0
        self._worker_failure_count = 0
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = self.cache_dir
        self._models: Dict[str, TextEmbedding] = {}

        # Deprecated models warn once per client instance, not per embed call.
        self._deprecation_warned: set = set()

        # Sticky after accelerator failure so later work uses the stable CPU path.
        self._gpu_poisoned = False

        # CPU fallback models: model_name → SentenceTransformer on device="cpu"
        # Lazy-loaded when _gpu_poisoned is True, so remaining files can still be processed.
        self._cpu_fallback_models = {}

        # Persistent spawned workers: model name → child process session.  The
        # parent stores only protocol handles; native runtime/model state stays in
        # the child and can be synchronously terminated and reaped.
        self._accelerator_workers: dict[str, AcceleratorWorkerSession] = {}
        atexit.register(self.close)

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

    def _warn_if_previous_coreml_session_killed(self) -> None:
        """Report a CoreML session that never exited cleanly (likely OS kill).

        A SIGKILL from memory pressure gives the dying process no chance to
        warn, so the warning is issued here, on the next EmbeddingClient
        construction, based on the leftover crash sentinel.
        """
        import json

        try:
            sentinel = _coreml_sentinel_path()
            if not sentinel.exists():
                return
            data = json.loads(sentinel.read_text())
            pid = data.get("pid")
            if pid is not None and _pid_is_alive(pid):
                return
            message = (
                f"A previous run using experimental CoreML "
                f"(model '{data.get('model', 'unknown')}', started {data.get('started', 'unknown')}) "
                "did not exit cleanly — likely killed by the OS due to memory exhaustion. "
                "Re-run without --gpu for stable CPU embedding."
            )
            logger.info(message)
            print(f"⚠ {message}", file=sys.stderr, flush=True)
            sentinel.unlink()
        except Exception:
            logger.debug("CoreML crash sentinel check failed", exc_info=True)

    def _gated_model_hint(self, model_name: str, config: Dict, error: Exception) -> Optional[str]:
        """Explain the license/auth steps when a gated HF repo refuses a download.

        Gated repos (e.g. google/embeddinggemma-300m) return 401/403 until the
        user accepts the license on Hugging Face and authenticates locally.
        """
        if not config.get("gated"):
            return None
        error_msg = str(error).lower()
        if not any(
            marker in error_msg for marker in ("401", "403", "gated", "unauthorized", "restricted")
        ):
            return None
        return (
            f"Model '{model_name}' is a gated Hugging Face repo and requires access approval.\n"
            f"1. Accept the license at https://huggingface.co/{config['name']}\n"
            f"2. Authenticate this machine with: hf auth login\n\n"
            f"Original error: {error}"
        )

    def _experimental_coreml_enabled(self) -> bool:
        """Return True when the user explicitly opts into FastEmbed CoreML.

        CoreMLExecutionProvider can be unstable for large transformer ONNX
        models on Apple Silicon because ORT may split the graph into many
        CoreML/CPU partitions and allocate large native unified-memory buffers
        outside Python's normal accounting. An explicit --gpu flag counts as
        the opt-in (allow_experimental_coreml); paths that enable GPU
        implicitly still require ARC_EXPERIMENTAL_COREML.
        """
        if self._allow_experimental_coreml:
            return True
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
        selection = self._backend_selections.get(model_name) or self._select_model_backend(
            model_name
        )
        if selection.backend != "onnxruntime-coreml":
            message = selection.fallback_reason or "CoreML combination is not qualified"
            logger.info("Using CPUExecutionProvider for '%s': %s", model_name, message)
            if not self.use_gpu:
                return ["CPUExecutionProvider"]
            print(
                f"   GPU requested, but FastEmbed/CoreML is experimental for "
                f"'{model_name}'. Using CPUExecutionProvider. "
                f"Set ARC_EXPERIMENTAL_COREML=1 to opt in. {message}",
                file=sys.stderr,
                flush=True,
            )
            return ["CPUExecutionProvider"]

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
                warning = (
                    f"Using experimental CoreML for '{model_name}'. This can exhaust "
                    "system memory; if this process is killed by the OS, re-run "
                    "without --gpu."
                )
                logger.info(warning)
                print(f"⚠ {warning}", file=sys.stderr, flush=True)
                _write_coreml_sentinel(model_name)
                compiled_cache = Path(get_models_dir()).parent / "coreml-compiled"
                return [
                    ("CoreMLExecutionProvider", coreml_provider_options(compiled_cache)),
                    "CPUExecutionProvider",
                ]
        except Exception:
            logger.debug("CoreML provider detection failed for FastEmbed", exc_info=True)

        return ["CPUExecutionProvider"]

    def _system_memory_available_gb(self) -> Optional[float]:
        """Return currently available system memory in GB, or None if unknown."""
        try:
            import psutil

            return psutil.virtual_memory().available / (1024**3)
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
        self._drop_accelerator_worker(model_name)
        if model_name in self._models:
            # Drop the active accelerator model reference so subsequent
            # get_model() calls load the CPU fallback for SentenceTransformers.
            # FastEmbed models already use CPUExecutionProvider by default on
            # Apple Silicon, so this is mainly for PyTorch MPS/CUDA models.
            self._models.pop(model_name, None)

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

        config = EMBEDDING_MODELS[model_name]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise _sentence_transformers_missing_error(model_name, config) from e
        _ensure_dynamic_cache_compat()
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
           that run for minutes per file.

        Idempotent: env vars are only set if absent; torch setters are
        cheap and the values are stable across calls.
        """
        self._configure_cpu_threading()

        # Mutate torch's live thread pool directly — env vars alone are too
        # late once torch is imported. cpu_count - 2 leaves headroom for
        # indexing orchestration and the parent process.
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
            chunks.append(
                cpu_model.encode(
                    texts[start:end],
                    batch_size=self._CPU_FALLBACK_INNER_BATCH,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    **encode_kwargs,
                )
            )
        return np.concatenate(chunks, axis=0)

    def close(self) -> None:
        """Shut down and reap every accelerator child; safe to call repeatedly."""
        first_error: BaseException | None = None
        workers = list(self._accelerator_workers.items())
        for model_name, worker in workers:
            try:
                worker.shutdown()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            finally:
                if not worker.is_alive:
                    self._accelerator_workers.pop(model_name, None)
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "EmbeddingClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _drop_accelerator_worker(self, model_name: str) -> None:
        worker = self._accelerator_workers.get(model_name)
        if worker is not None:
            worker.shutdown()
            if not worker.is_alive:
                self._accelerator_workers.pop(model_name, None)

    def _get_accelerator_worker(self, model_name: str) -> AcceleratorWorkerSession:
        worker = self._accelerator_workers.get(model_name)
        if worker is not None and worker.is_alive:
            return worker
        if worker is not None:
            worker.shutdown()
            self._accelerator_workers.pop(model_name, None)
            self._worker_restart_count += 1
        config = WorkerConfig(
            "arcaneum.embeddings.sentence_transformer_worker:"
            "create_sentence_transformer_accelerator_backend",
            {
                "model_name": model_name,
                "device": self._device,
                "cache_dir": self.cache_dir,
                "local_files_only": self.is_model_cached(model_name),
            },
        )
        worker = AcceleratorWorkerSession(config, startup_timeout=120.0).start()
        self._accelerator_workers[model_name] = worker
        return worker

    def _detect_device(self) -> str:
        """Detect best available GPU device (RDR-013 Phase 2).

        Returns:
            "mps" for Apple Silicon, "cuda" for NVIDIA, "cpu" if no GPU available
        """
        if sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
            return "mps"
        if shutil.which("nvidia-smi"):
            return "cuda"
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
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = cpu_threads
            logger.debug(
                f"Set OMP_NUM_THREADS={cpu_threads} for CPU parallelism (cores={available_cores}, workers={self._cpu_workers})"
            )

        # MKL_NUM_THREADS for Intel MKL (used by NumPy/PyTorch on Intel CPUs)
        if "MKL_NUM_THREADS" not in os.environ:
            os.environ["MKL_NUM_THREADS"] = cpu_threads

        # Disable tokenizers parallelism by default - it adds another layer of threads
        # that can cause over-subscription. Only enable if explicitly set.
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            logger.debug("Disabled TOKENIZERS_PARALLELISM to prevent over-subscription")

    def get_device_info(self) -> Dict[str, str]:
        """Get information about the device being used (RDR-013 Phase 2).

        Returns:
            Dictionary with device information
        """
        return {
            "device": self._device,
            "gpu_enabled": self.use_gpu,
            "gpu_available": self._device != "cpu",
        }

    def get_backend_diagnostics(self, model_name: str) -> Dict[str, Any]:
        """Return verbose-safe capability and fallback state for one model."""
        selection = self._backend_selections.get(model_name)
        if selection is None and model_name in EMBEDDING_MODELS:
            selection = self._select_model_backend(model_name)
        if selection is None:
            return {
                "model": model_name,
                "state": "unavailable",
                "fallback_reason": "unknown embedding model",
                "worker_restart_count": self._worker_restart_count,
            }
        result = selection.as_dict()
        result["fallback_reason"] = self._backend_fallback_reasons.get(
            model_name, selection.fallback_reason
        )
        result["worker_restart_count"] = self._worker_restart_count
        result["worker_failure_count"] = self._worker_failure_count
        return result

    def _select_model_backend(self, model_name: str) -> BackendSelection:
        config = EMBEDDING_MODELS[model_name]
        requested = self._device if self.use_gpu else "cpu"
        selection = select_backend(
            model=model_name,
            model_backend=config.get("backend", "fastembed"),
            requested_device=requested,
            # --gpu remains the explicit opt-in for experimental acceleration.
            allow_experimental=(
                self._experimental_coreml_enabled()
                if config.get("backend") == "fastembed"
                else self.use_gpu
            ),
        )
        self._backend_selections[model_name] = selection
        if selection.fallback_reason:
            self._backend_fallback_reasons[model_name] = selection.fallback_reason
        return selection

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
            return 768  # Medium models (jina-code: 768D, bge-base: 768D)
        else:
            return 512  # Large models (stella: 1024D, bge-large: 1024D)

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

        selection = self._select_model_backend(model_name)
        config = EMBEDDING_MODELS[model_name]

        deprecated_config = EMBEDDING_MODELS[model_name]
        if deprecated_config.get("deprecated") and model_name not in self._deprecation_warned:
            self._deprecation_warned.add(model_name)
            replacement = deprecated_config.get("superseded_by")
            hint = f" Use '{replacement}' for new corpora." if replacement else ""
            message = (
                f"Embedding model '{model_name}' is deprecated; existing corpora "
                f"continue to work.{hint}"
            )
            logger.warning(message)
            print(f"   Warning: {message}", file=sys.stderr, flush=True)

        # When GPU is poisoned, return CPU fallback for sentence-transformers models
        # instead of loading a new GPU model (RDR-020).
        if self._gpu_poisoned or (
            self.use_gpu
            and config.get("backend") == "sentence-transformers"
            and selection.device == "cpu"
        ):
            if config.get("backend") == "sentence-transformers":
                return self._get_cpu_fallback_model(model_name)

        # A GPU SentenceTransformer is represented in the parent by a marker
        # only. Loading and native imports happen when the spawned worker starts.
        if config.get("backend") == "sentence-transformers" and self._device in ("mps", "cuda"):
            return _AcceleratorModelProxy()

        if model_name not in self._models:
            backend = config.get("backend", "fastembed")

            if backend == "fastembed":
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

                model_obj = None
                last_error = None

                if is_cached:
                    try:
                        model_obj = TextEmbedding(
                            model_name=config["name"],
                            cache_dir=self.cache_dir,
                            local_files_only=True,
                            providers=providers,  # GPU acceleration if available
                        )
                    except Exception as e:
                        last_error = e

                if model_obj is None:
                    if last_error is not None:
                        print(
                            "   Downloading additional model files...", flush=True, file=sys.stderr
                        )
                    try:
                        model_obj = TextEmbedding(
                            model_name=config["name"],
                            cache_dir=self.cache_dir,
                            local_files_only=False,
                            providers=providers,  # GPU acceleration if available
                        )
                    except Exception as e:
                        last_error = e

                if model_obj is None and self._is_missing_fastembed_artifact_error(last_error):
                    purged = self._purge_fastembed_model_cache(model_name)
                    if purged:
                        print(
                            "   Cached model files were incomplete; redownloading...",
                            flush=True,
                            file=sys.stderr,
                        )
                        try:
                            model_obj = TextEmbedding(
                                model_name=config["name"],
                                cache_dir=self.cache_dir,
                                local_files_only=False,
                                providers=providers,
                            )
                        except Exception as e:
                            last_error = e

                if model_obj is not None:
                    self._models[model_name] = model_obj
                else:
                    # Detect and report network/SSL errors with helpful messages
                    error_msg = str(last_error).lower()
                    if "ssl" in error_msg or "certificate" in error_msg:
                        raise RuntimeError(
                            f"SSL certificate verification failed while downloading model '{model_name}'.\n"
                            f"For corporate proxies with self-signed certificates, run:\n"
                            f"  export ARC_SSL_VERIFY=false\n\n"
                            f"Original error: {last_error}"
                        ) from last_error
                    elif (
                        "connection" in error_msg
                        or "network" in error_msg
                        or "timeout" in error_msg
                    ):
                        raise RuntimeError(
                            f"Network connection failed while downloading model '{model_name}'.\n"
                            f"Please check your internet connection. If using a VPN, try disabling it.\n\n"
                            f"Original error: {last_error}"
                        ) from last_error
                    else:
                        # Re-raise other errors as-is
                        raise last_error
            elif backend == "sentence-transformers":
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as e:
                    raise _sentence_transformers_missing_error(model_name, config) from e
                _ensure_dynamic_cache_compat()

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
                        flush=True,
                        file=sys.stderr,
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
                            logger.info(
                                f"Set {model_name} max_seq_length: {original_max} → {config['max_seq_length']}"
                            )
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
                            print(
                                "   Downloading additional model files...",
                                flush=True,
                                file=sys.stderr,
                            )

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
                            logger.info(
                                f"Set {model_name} max_seq_length: {original_max} → {config['max_seq_length']}"
                            )
                        self._models[model_name] = model_obj
                    except Exception as e:
                        # Detect and report gated/network/SSL errors with helpful messages
                        gated_hint = self._gated_model_hint(model_name, config, e)
                        error_msg = str(e).lower()
                        if gated_hint:
                            raise RuntimeError(gated_hint) from e
                        elif "ssl" in error_msg or "certificate" in error_msg:
                            raise RuntimeError(
                                f"SSL certificate verification failed while downloading model '{model_name}'.\n"
                                f"For corporate proxies with self-signed certificates, run:\n"
                                f"  export ARC_SSL_VERIFY=false\n\n"
                                f"Original error: {e}"
                            ) from e
                        elif (
                            "connection" in error_msg
                            or "network" in error_msg
                            or "timeout" in error_msg
                        ):
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
        elif max_text_len > max_chars * 0.8:
            logger.info(
                f"Large chunks are near the embedding safety bound but no clipping was needed: "
                f"max={max_text_len} chars, limit={max_chars} chars "
                f"(model={model_name}, max_seq_length={max_seq_length}). "
                f"Upstream chunking/windowing preserved the full text for indexing."
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
        if hasattr(model, "_backend") and model._backend == "sentence-transformers":
            # SentenceTransformers: use encode() with convert_to_numpy=True (arcaneum-ppa2)
            # This uses the model's optimized GPU→CPU transfer path.
            # Potential 10-20% speedup on embeddings by reducing tensor→list conversion overhead.

            # CRITICAL: model.encode() batch_size controls GPU memory usage
            # Use dynamic batch sizing based on available memory at runtime
            # This replaces the previous hard-coded values (8/32/64) which caused
            # excessive kernel launches and poor GPU utilization
            if self._device in ("mps", "cuda") and not self._gpu_poisoned:
                # Runtime memory probes import torch and initialize native state in
                # the parent. Use the model's conservative qualified cap; token/shape
                # scheduling supplies the finer-grained bound outside this client.
                internal_batch_size = int(
                    model_config.get("mps_max_batch", min(batch_size, 64))
                    if self._device == "mps"
                    else min(batch_size, self.get_optimal_batch_size(model_name))
                )

                # Apply max_internal_batch limit if specified (for OOM recovery)
                if max_internal_batch is not None and max_internal_batch < internal_batch_size:
                    logger.debug(
                        f"Applying OOM recovery limit: {internal_batch_size} → {max_internal_batch}"
                    )
                    internal_batch_size = max_internal_batch
            else:
                # CPU: Use conservative batches
                internal_batch_size = min(batch_size, 256)

            # For files with many chunks, process in outer batches to avoid buffer allocation failures
            # Metal/MPS can fail with "Invalid buffer size" when trying to allocate output buffers
            # for hundreds of embeddings at once, even with small internal batch_size
            MAX_OUTER_BATCH = 128  # Process at most 128 texts per model.encode() call
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

                logger.debug(
                    f"Large input ({len(sorted_texts)} texts), processing in {MAX_OUTER_BATCH}-text outer batches"
                )

                dim = self.get_dimensions(model_name)
                sorted_embeddings = np.zeros((len(sorted_texts), dim), dtype=np.float32)
                offset = 0

                for start_idx in range(0, len(sorted_texts), MAX_OUTER_BATCH):
                    end_idx = min(start_idx + MAX_OUTER_BATCH, len(sorted_texts))
                    batch_texts = sorted_texts[start_idx:end_idx]

                    batch_embeddings = self._encode_with_oom_recovery(
                        model, batch_texts, internal_batch_size, model_name, prompt_type
                    )
                    sorted_embeddings[offset : offset + len(batch_texts)] = batch_embeddings
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
            # FastEmbed: schedule by token and padded shape, then restore caller order.
            if not texts:
                return []
            max_batch = min(batch_size, 100)
            max_sequence = int(model_config.get("max_seq_length", 512))
            backend_model = model.model
            if getattr(backend_model, "tokenizer", None) is None:
                backend_model.load_onnx_model()
            tokenizer = backend_model.tokenizer

            def count_tokens(text: str) -> int:
                encoding = tokenizer.encode(text, add_special_tokens=True)
                return sum(encoding.attention_mask)

            def truncate_tokens(text: str, limit: int) -> str:
                encoding = tokenizer.encode(text, add_special_tokens=False)
                token_ids = list(encoding.ids[: max(0, limit - 2)])
                candidate = tokenizer.decode(token_ids, skip_special_tokens=True)
                while token_ids and count_tokens(candidate) > limit:
                    token_ids.pop()
                    candidate = tokenizer.decode(token_ids, skip_special_tokens=True)
                return candidate

            budget = BatchBudget(
                max_actual_tokens=max_sequence * max_batch,
                max_padded_tokens=max_sequence * max_batch,
                max_sequence_tokens=max_sequence,
                max_batch_size=max_batch,
                oversize_policy=OversizePolicy.TRUNCATE,
            )
            batches = schedule_batches(
                texts,
                budget=budget,
                count_tokens=count_tokens,
                truncate=truncate_tokens,
            )
            collector = BatchResultCollector(len(texts))
            for scheduled in batches:
                rows = np.asarray(list(model.embed(scheduled.texts)), dtype=np.float32)
                collector.add(scheduled, rows)
            all_embeddings = collector.finalize()

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
        3. Native GPU work hangs indefinitely (the parent times out and kills its process)

        The persistent child handles recoverable OOM retries. The parent validates
        returned CPU arrays and enforces the kill/reap boundary for hangs or crashes.

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
        if self._gpu_poisoned:
            cpu_model = self._get_cpu_fallback_model(model_name)
            logger.info(f"GPU poisoned, falling back to CPU for {len(texts)} texts")
            return self._encode_on_cpu_fallback(cpu_model, texts, model_name, prompt_type)

        # CPU short-circuit: process timeout and poisoning only make sense for
        # MPS/CUDA hangs at the native level. On CPU the
        # 120s timeout misfires on legitimate slow encodes, and the "fallback" path
        # would spawn a second CPU encode that competes with the still-running first
        # one for OMP threads and RAM. Run inline with bounded batching instead.
        if self._device == "cpu":
            return self._encode_on_cpu_fallback(model, texts, model_name, prompt_type)

        try:
            worker = self._get_accelerator_worker(model_name)
            model_config = EMBEDDING_MODELS[model_name]
            max_sequence = int(model_config.get("max_seq_length", 512))
            token_budget = int(
                model_config.get("accelerator_token_budget", max_sequence * internal_batch_size)
            )
            result = worker.encode(
                texts,
                timeout=encode_timeout,
                batch_size=internal_batch_size,
                max_sequence_tokens=max_sequence,
                token_budget=token_budget,
                **_sentence_transformer_encode_kwargs(model_name, prompt_type),
            )
            if not self._validate_embeddings(result, len(texts), model_name):
                # Malformed/corrupt native output invalidates this worker just as a
                # crash does. Reap it before constructing any CPU model.
                self._drop_accelerator_worker(model_name)
                raise WorkerProtocolError("accelerator produced invalid embeddings")
            return result
        except (WorkerTimeoutError, WorkerCrashedError, WorkerProtocolError) as exc:
            # AcceleratorWorkerSession has already terminated and joined on every
            # request failure. shutdown() is idempotent and covers validation above.
            self._drop_accelerator_worker(model_name)
            self._gpu_poisoned = True
            self._models.pop(model_name, None)
            self._worker_failure_count += 1
            self._backend_fallback_reasons[model_name] = (
                f"accelerator worker reaped after {type(exc).__name__}: {exc}"
            )
            logger.warning(
                "Accelerator worker failed for '%s'; falling back to CPU after reap: %s",
                model_name,
                exc,
            )
            print(
                "  GPU worker stopped — falling back to CPU for remaining work.",
                file=sys.stderr,
                flush=True,
            )
            cpu_model = self._get_cpu_fallback_model(model_name)
            return self._encode_on_cpu_fallback(cpu_model, texts, model_name, prompt_type)

    def embed_parallel(
        self,
        texts: List[str],
        model_name: str,
        max_workers: int = None,
        batch_size: int = None,
        timeout: int = 300,
        progress_callback: callable = None,
        on_batch_complete: callable = None,
        accumulate: bool = True,
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
        logger.debug(
            f"Embedding {len(texts)} texts with batch_size={batch_size}, use_gpu={self.use_gpu}, device={self._device}"
        )

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

            logger.debug(
                f"Model {model_name} params={params_billions}B ({model_size_category}), cache_clear_interval={cache_clear_interval}, clear_before={clear_before_batch}"
            )

            # OOM recovery: track effective batch size across all batches
            # Start with requested batch_size, reduce on OOM until we reach minimum
            effective_batch_size = batch_size
            min_batch_size = 8  # Minimum viable batch size before giving up

            # For large models, clear GPU cache BEFORE first batch to ensure maximum
            # available memory. This is critical when processing multiple files as
            # memory from PDF extraction may not be fully released yet.
            if clear_before_batch:
                gc.collect()

            for start_idx in range(0, len(texts), batch_size):
                self._maybe_disable_gpu_for_memory_pressure(model_name)

                # For memory-hungry models, clear cache BEFORE embedding to maximize
                # available memory for attention allocations (arcaneum-mem-leak)
                if clear_before_batch and batch_idx > 0:
                    gc.collect()

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
                        result = self.embed(
                            batch_texts,
                            model_name,
                            batch_size=batch_size,
                            max_internal_batch=current_max_internal
                            if current_max_internal != batch_size
                            else None,
                        )

                        # Validate embeddings - Metal OOM can corrupt results without raising exceptions
                        # The errors are printed to stderr but embeddings may contain NaN/garbage
                        if not self._validate_embeddings(result, len(batch_texts), model_name):
                            raise RuntimeError(
                                "GPU produced invalid embeddings (likely OOM corruption)"
                            )

                        batch_embeddings = result

                    except KeyboardInterrupt:
                        # User pressed Ctrl-C - clean up and re-raise
                        # This handles interrupts that arrive between GPU operations
                        logger.debug("KeyboardInterrupt received during embedding")
                        self._drop_accelerator_worker(model_name)
                        raise
                    except Exception as e:
                        # Detect GPU OOM from various sources:
                        # - PyTorch: "out of memory", "CUDA out of memory"
                        # - Metal/MPS: "Insufficient Memory", "kIOGPUCommandBufferCallbackErrorOutOfMemory"
                        # - Generic: "command buffer exited with error status"
                        # - Our validation: "invalid embeddings"
                        error_str = str(e).lower()
                        is_oom = any(
                            pattern in error_str
                            for pattern in [
                                "out of memory",
                                "insufficient memory",
                                "kiogpucommandbuffercallbackerroroutofmemory",
                                "command buffer exited with error status",
                                "mps backend out of memory",
                                "cuda error: out of memory",
                                "invalid embeddings",  # Our validation error
                                "oom corruption",
                            ]
                        )

                        # Keep retrying with smaller batches until we hit minimum
                        if is_oom and current_max_internal > min_batch_size:
                            batch_retry_count += 1
                            # Halve the batch size (more gradual than /4)
                            new_max = max(min_batch_size, current_max_internal // 2)

                            # Brief message - Metal/CUDA already dumped verbose error
                            import sys

                            print(
                                f"  (GPU memory pressure, reducing batch {current_max_internal} → {new_max}...)",
                                file=sys.stderr,
                                flush=True,
                            )
                            logger.debug(
                                f"OOM at batch {batch_idx + 1}, reducing internal batch size: "
                                f"{current_max_internal} → {new_max} (attempt {batch_retry_count})"
                            )
                            current_max_internal = new_max
                            effective_batch_size = new_max  # Remember for future batches

                            # Native cache recovery happens inside the worker.
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
                logger.debug(
                    f"Batch {batch_idx + 1}/{total_batches}: {actual_batch_size} chunks embedded in {batch_elapsed:.2f}s ({actual_batch_size / batch_elapsed:.1f} chunks/s)"
                )

                # Call batch complete callback if provided (for streaming upload)
                if on_batch_complete:
                    on_batch_complete(batch_idx, start_idx, batch_embeddings)

                # Fill pre-allocated array in place (no list over-allocation)
                # Only if accumulating results
                if accumulate:
                    all_embeddings[offset : offset + actual_batch_size] = batch_embeddings
                offset += actual_batch_size

                batch_idx += 1
                if progress_callback:
                    # Pass extended progress info: batch_idx, total_batches, effective_batch_size, chunks_done, total_chunks
                    # Callback can accept 2 args (legacy) or 5 args (extended)
                    import inspect

                    sig = inspect.signature(progress_callback)
                    if len(sig.parameters) >= 5:
                        progress_callback(
                            batch_idx,
                            total_batches,
                            effective_batch_size,
                            chunks_embedded,
                            len(texts),
                        )
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

            # Final cleanup
            gc.collect()

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
            if hasattr(embeddings, "numpy"):
                embeddings = embeddings.numpy()
            elif not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            # Check shape
            expected_dims = self.get_dimensions(model_name)
            if len(embeddings.shape) != 2:
                logger.debug(f"Embeddings validation failed: wrong shape {embeddings.shape}")
                return False

            if embeddings.shape[0] != expected_count:
                logger.debug(
                    f"Embeddings validation failed: count mismatch {embeddings.shape[0]} vs {expected_count}"
                )
                return False

            if embeddings.shape[1] != expected_dims:
                logger.debug(
                    f"Embeddings validation failed: dims mismatch {embeddings.shape[1]} vs {expected_dims}"
                )
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
                logger.debug(
                    f"Embeddings validation failed: {small_count} vectors with tiny norm (<0.01)"
                )
                return False

            if np.any(norms > 1000):  # Suspiciously large
                large_count = np.sum(norms > 1000)
                logger.debug(
                    f"Embeddings validation failed: {large_count} vectors with huge norm (>1000)"
                )
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
                    logger.debug(
                        f"Embeddings validation failed: suspiciously low variance ({variance:.2e})"
                    )
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
            transformers_modules_dir = os.path.join(
                self.cache_dir, "modules", "transformers_modules"
            )
            if not os.path.exists(transformers_modules_dir):
                # Try sibling directory
                transformers_modules_dir = os.path.join(
                    os.path.dirname(self.cache_dir), "modules", "transformers_modules"
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
            if self._fastembed_model_cache_complete(model_dir, model_path):
                return True

            # Check for FastEmbed wrapped versions (models--qdrant--model-onnx format)
            # List all model directories and check for similar names
            if os.path.exists(self.cache_dir):
                for item in os.listdir(self.cache_dir):
                    item_path = os.path.join(self.cache_dir, item)
                    if os.path.isdir(item_path) and item.startswith("models--"):
                        # Check if this directory contains the model name parts
                        item_lower = item.lower().replace("-", "_").replace(".", "_")
                        model_parts = (
                            model_path.lower()
                            .replace("-", "_")
                            .replace(".", "_")
                            .replace("/", "_")
                            .split("_")
                        )
                        # If most of the model name parts are in the directory name, consider it a match
                        if (
                            sum(1 for part in model_parts if len(part) > 2 and part in item_lower)
                            >= len([p for p in model_parts if len(p) > 2]) * 0.6
                        ):
                            if self._fastembed_model_cache_complete(item_path, model_path):
                                return True

            return False

    def _fastembed_model_cache_complete(self, model_dir: str, model_path: str) -> bool:
        """Return True when a FastEmbed cache has its backend model artifact."""
        if not (os.path.exists(model_dir) and os.path.isdir(model_dir)):
            return False

        required_file = self._fastembed_required_model_file(model_path)
        if required_file is None:
            return True

        # Defensive check for non-standard cache layouts; real HuggingFace
        # caches keep model files under snapshots/<revision>/ (checked below).
        if os.path.isfile(os.path.join(model_dir, required_file)):
            return True

        snapshots_dir = os.path.join(model_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            return False

        for revision in os.listdir(snapshots_dir):
            candidate = os.path.join(snapshots_dir, revision, required_file)
            if os.path.isfile(candidate):
                return True

        return False

    @staticmethod
    def _is_missing_fastembed_artifact_error(error: Optional[BaseException]) -> bool:
        """Detect incomplete local FastEmbed caches from ONNX Runtime errors."""
        if error is None:
            return False

        message = str(error).lower()
        # Require a ".onnx" artifact mention so runtime-library paths like
        # libonnxruntime_providers_shared.so don't trigger a cache purge.
        return ("no_suchfile" in message or "file doesn't exist" in message) and ".onnx" in message

    def _purge_fastembed_model_cache(self, model_name: str) -> bool:
        """Remove cached FastEmbed directories for a model so download can heal."""
        if model_name not in EMBEDDING_MODELS:
            raise _unknown_model_error(model_name)

        config = EMBEDDING_MODELS[model_name]
        model_path = config["name"]
        safe_model_name = model_path.replace("/", "--")
        candidates = {os.path.join(self.cache_dir, f"models--{safe_model_name}")}

        if os.path.exists(self.cache_dir):
            for item in os.listdir(self.cache_dir):
                item_path = os.path.join(self.cache_dir, item)
                if not (os.path.isdir(item_path) and item.startswith("models--")):
                    continue
                if self._cache_dir_name_matches_model(item, model_path):
                    candidates.add(item_path)

        purged = False
        for candidate in candidates:
            try:
                if os.path.isdir(candidate):
                    shutil.rmtree(candidate)
                    purged = True
            except OSError:
                logger.warning(
                    "Failed to remove incomplete model cache %s", candidate, exc_info=True
                )

        return purged

    @staticmethod
    def _cache_dir_name_matches_model(dir_name: str, model_path: str) -> bool:
        """Match an HF cache dir (models--<org>--<name>) to a model.

        Purging is destructive, so this requires the directory's name tokens to
        equal the model's — tolerating a different org and an "onnx" wrapper
        suffix (FastEmbed re-uploads, e.g. qdrant/<name>-onnx) but never a
        sibling model that merely shares most name tokens.
        """
        repo_name = dir_name.split("--")[-1]

        def _tokens(value: str) -> set:
            normalized = value.lower().replace("-", "_").replace(".", "_")
            return {part for part in normalized.split("_") if len(part) > 2}

        model_tokens = _tokens(model_path.split("/")[-1])
        dir_tokens = _tokens(repo_name) - {"onnx"}
        return bool(model_tokens) and dir_tokens == model_tokens

    @staticmethod
    def _fastembed_required_model_file(model_path: str) -> Optional[str]:
        """Look up the model file FastEmbed needs for a supported model."""
        try:
            supported_models = TextEmbedding.list_supported_models()
        except Exception:
            logger.debug("Could not inspect FastEmbed supported models", exc_info=True)
            return None

        for model in supported_models:
            if model.get("model") == model_path:
                return model.get("model_file")

        return None

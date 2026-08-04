"""Child-only SentenceTransformers accelerator backend.

This module is named by import path in :class:`WorkerConfig`; the parent never
imports it.  Consequently torch, SentenceTransformers, the model, and all native
accelerator state are constructed and destroyed in the spawned child process.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Any

import numpy as np


class SentenceTransformerAcceleratorBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        import torch

        requested_device = config["device"]
        if requested_device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "PyTorch MPS is not available (built="
                f"{torch.backends.mps.is_built()}); qualification cannot run"
            )
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is not available; qualification cannot run")
        # Importing client is safe in this spawned child and reuses the canonical,
        # pinned model and prompt policies.  It must never be imported by the parent.
        from sentence_transformers import SentenceTransformer

        from arcaneum.embeddings.client import (
            EMBEDDING_MODELS,
            _ensure_dynamic_cache_compat,
            _sentence_transformer_load_kwargs,
        )

        self.model_name = config["model_name"]
        self.device = config["device"]
        self.model_config = EMBEDDING_MODELS[self.model_name]
        _ensure_dynamic_cache_compat()

        local_only = bool(config["local_files_only"])
        try:
            self.model = SentenceTransformer(
                self.model_config["name"],
                **_sentence_transformer_load_kwargs(
                    self.model_name,
                    self.model_config,
                    cache_folder=config["cache_dir"],
                    local_files_only=local_only,
                    device=self.device,
                ),
            )
        except Exception:
            if not local_only or config.get("strict_local_files_only"):
                raise
            # Match the historical loader: prefer a complete offline cache, then
            # allow Hugging Face to fill missing artifacts.
            self.model = SentenceTransformer(
                self.model_config["name"],
                **_sentence_transformer_load_kwargs(
                    self.model_name,
                    self.model_config,
                    cache_folder=config["cache_dir"],
                    local_files_only=False,
                    device=self.device,
                ),
            )
        if "max_seq_length" in self.model_config:
            self.model.max_seq_length = self.model_config["max_seq_length"]
        self._encodes = 0
        self._oom_retries = 0

    def _sync(self) -> None:
        import torch

        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    def _clear(self) -> None:
        import torch

        gc.collect()
        if self.device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _encode_once(self, texts: list[str], batch_size: int, **policy: Any) -> np.ndarray:
        self._sync()
        result = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            **policy,
        )
        self._sync()
        return np.asarray(result, dtype=np.float32)

    def encode(self, texts: list[str], **options: Any) -> np.ndarray:
        batch_size = max(1, int(options.get("batch_size", 1)))
        policy = {
            key: options[key] for key in ("task", "prompt_name") if options.get(key) is not None
        }
        oom_markers = (
            "enough space",
            "mpsgraph",
            "mps backend out of memory",
            "command buffer exited with error",
            "invalid buffer size",
            "cuda out of memory",
            "cuda error: out of memory",
        )
        attempts = (batch_size, 1, 1)
        last_error: BaseException | None = None
        for index, attempt_size in enumerate(attempts):
            try:
                result = self._encode_once(texts, attempt_size, **policy)
                self._encodes += 1
                return result
            except BaseException as exc:
                last_error = exc
                if not any(marker in str(exc).lower() for marker in oom_markers):
                    raise
                if index == len(attempts) - 1:
                    break
                self._oom_retries += 1
                self._clear()
                time.sleep(0.5 if index == 0 else 1.0)
        raise RuntimeError(
            "accelerator memory exhausted after batch-size reduction"
        ) from last_error

    def health(self) -> dict[str, Any]:
        result = {
            "pid": os.getpid(),
            "model": self.model_name,
            "device": self.device,
            "model_loads": 1,
            "encodes": self._encodes,
            "oom_retries": self._oom_retries,
        }
        if self.device == "mps":
            import torch

            result["mps_current_allocated_bytes"] = int(torch.mps.current_allocated_memory())
            result["mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
            result["mps_recommended_max_bytes"] = int(torch.mps.recommended_max_memory())
        elif self.device == "cuda":
            import torch

            properties = torch.cuda.get_device_properties(torch.cuda.current_device())
            result.update(
                {
                    "cuda_device_name": properties.name,
                    "cuda_compute_capability": [properties.major, properties.minor],
                    "cuda_total_memory_bytes": int(properties.total_memory),
                    "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
                    "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "cuda_runtime_version": torch.version.cuda,
                }
            )
        return result

    def close(self) -> None:
        self.model = None
        self._clear()


def create_sentence_transformer_accelerator_backend(
    config: dict[str, Any],
) -> SentenceTransformerAcceleratorBackend:
    return SentenceTransformerAcceleratorBackend(config)

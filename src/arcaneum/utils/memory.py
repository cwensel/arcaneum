"""Memory management utilities for Arcaneum."""

import psutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> tuple[Optional[int], Optional[int], Optional[str]]:
    """Get GPU memory information (available, total, device_type).

    Returns:
        Tuple of (available_bytes, total_bytes, device_type) or (None, None, None) if no GPU
    """
    try:
        import torch

        if torch.cuda.is_available():
            # CUDA device
            free_mem, total_mem = torch.cuda.mem_get_info()
            return (free_mem, total_mem, "cuda")
        elif torch.backends.mps.is_available():
            # MPS (Apple Silicon) - less precise info available
            # MPS doesn't expose memory info directly, use heuristics
            # Approximate based on system memory dedicated to GPU
            try:
                # Try to get current allocation
                allocated = torch.mps.driver_allocated_memory()
                # Estimate total as system unified memory (conservative: 70% of RAM)
                mem = psutil.virtual_memory()
                estimated_total = int(mem.total * 0.7)  # 70% of RAM for GPU
                available = estimated_total - allocated
                return (available, estimated_total, "mps")
            except Exception:
                # Fallback: assume 70% of system RAM available for MPS
                mem = psutil.virtual_memory()
                estimated_total = int(mem.total * 0.7)
                return (estimated_total, estimated_total, "mps")
        else:
            return (None, None, None)
    except ImportError:
        return (None, None, None)
    except Exception as e:
        logger.debug(f"Failed to get GPU memory info: {e}")
        return (None, None, None)


def estimate_safe_batch_size(
    model_dimensions: int,
    available_gpu_bytes: int,
    safety_factor: float = 0.5
) -> int:
    """Estimate safe batch size based on available GPU memory.

    DEPRECATED: Use estimate_safe_batch_size_v2() for more accurate estimates.
    This function overestimates memory usage (~2000x) and is kept for backward compatibility.

    Args:
        model_dimensions: Embedding model output dimensions (e.g., 1024)
        available_gpu_bytes: Available GPU memory in bytes
        safety_factor: Fraction of available memory to use (default: 0.5 for conservative estimate)

    Returns:
        Recommended safe batch size
    """
    # Rough heuristic: Each embedded item needs ~10MB for 1024D models
    # This includes model activations, gradients, and temporary buffers
    # Scale proportionally with dimensions
    bytes_per_item = (model_dimensions / 1024.0) * 10 * 1024 * 1024  # 10MB base for 1024D

    # Apply safety factor
    usable_memory = available_gpu_bytes * safety_factor

    # Calculate batch size
    estimated_batch = int(usable_memory / bytes_per_item)

    # Floor at reasonable minimum (8) and cap at reasonable maximum (1024)
    return max(8, min(estimated_batch, 1024))


def estimate_safe_batch_size_v2(
    model_name: str,
    available_gpu_bytes: int,
    pipeline_overhead_gb: float = 2.0,
    safety_factor: float = 0.6
) -> int:
    """Estimate safe batch size with model-aware memory calculations.

    This version accounts for:
    - Model weights (loaded once, not per-batch): 0.3-2.5GB depending on model
    - Pipeline overhead (PDF extraction, chunking): ~2GB
    - Activation memory per item: 3-8MB (empirical measurements)

    Based on empirical observation: "1024-text batches use <5MB" from embeddings/client.py:149
    This suggests per-item activation memory is much smaller than previously estimated.

    Memory model:
        Total = ModelWeights (one-time) + PipelineOverhead + (BatchSize × ActivationPerItem)

    Args:
        model_name: Model identifier (stella, jina-code, bge-large, bge-base, bge-small)
        available_gpu_bytes: Available GPU memory in bytes
        pipeline_overhead_gb: Memory reserved for PDF processing (default: 2GB)
        safety_factor: Fraction of memory to use after overhead (default: 0.6 = 60%)

    Returns:
        Recommended safe batch size (8-1024)

    Example:
        >>> # 10GB available, stella model
        >>> estimate_safe_batch_size_v2("stella", 10 * 1024**3)
        >>> # Returns: ~550 (10GB - 2.5GB model - 2GB pipeline = 5.5GB × 0.6 / 8MB)
    """
    # Model weights (one-time GPU memory allocation)
    MODEL_WEIGHTS_GB = {
        'stella': 2.5,        # 1.5B parameters
        'jina': 0.5,          # ~110M parameters
        'jina-code': 0.5,
        'bge-large': 0.8,
        'bge-base': 0.5,
        'bge-small': 0.3,
    }

    # Activation memory per batch item (empirical measurements)
    # Based on observation: 1024 texts use <5MB total for small models
    # But larger models have more activation memory due to:
    # - Intermediate layer activations
    # - Attention matrices (scales with sequence length)
    # - Temporary buffers during encoding
    ACTIVATION_MB_PER_ITEM = {
        'stella': 8.0,        # 1024D output, large model
        'jina': 5.0,          # 768D output
        'jina-code': 5.0,
        'bge-large': 8.0,     # 1024D output
        'bge-base': 5.0,      # 768D output
        'bge-small': 3.0,     # 384D output
    }

    # Get model-specific parameters, default to conservative values
    model_weights_gb = MODEL_WEIGHTS_GB.get(model_name, 2.0)
    activation_mb_per_item = ACTIVATION_MB_PER_ITEM.get(model_name, 8.0)

    # Calculate usable memory after accounting for fixed costs
    available_gb = available_gpu_bytes / (1024 ** 3)
    memory_after_fixed = available_gb - model_weights_gb - pipeline_overhead_gb

    if memory_after_fixed <= 0:
        # Not enough memory for model + pipeline
        logger.warning(
            f"Insufficient GPU memory: {available_gb:.1f}GB available, "
            f"but need {model_weights_gb + pipeline_overhead_gb:.1f}GB "
            f"for model + pipeline"
        )
        return 8  # Minimum fallback

    # Apply safety factor to usable memory
    usable_gb = memory_after_fixed * safety_factor
    usable_mb = usable_gb * 1024

    # Calculate batch size
    estimated_batch = int(usable_mb / activation_mb_per_item)

    # Clamp to reasonable range
    batch_size = max(8, min(estimated_batch, 1024))

    logger.debug(
        f"Batch size estimation for {model_name}: "
        f"available={available_gb:.1f}GB, "
        f"model_weights={model_weights_gb}GB, "
        f"pipeline_overhead={pipeline_overhead_gb}GB, "
        f"usable={usable_gb:.1f}GB, "
        f"estimated_batch={batch_size}"
    )

    return batch_size


def get_available_memory_gb() -> float:
    """Get available system memory in GB.

    Returns:
        Available memory in gigabytes
    """
    mem = psutil.virtual_memory()
    return mem.available / (1024 ** 3)


def calculate_safe_workers(
    requested_workers: int,
    estimated_memory_per_worker_mb: int,
    max_memory_gb: Optional[float] = None,
    min_workers: int = 1
) -> tuple[int, str]:
    """Calculate safe number of workers based on available memory.

    Args:
        requested_workers: Number of workers requested
        estimated_memory_per_worker_mb: Estimated memory usage per worker in MB
        max_memory_gb: Maximum memory to use (None = auto-calculate from available)
        min_workers: Minimum number of workers to return

    Returns:
        Tuple of (safe_worker_count, warning_message)
        warning_message is empty string if no warning needed
    """
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)

    # Determine memory limit
    if max_memory_gb is not None:
        memory_limit_gb = max_memory_gb
        limit_source = "user-specified"
    else:
        # Use 80% of available memory to leave headroom
        memory_limit_gb = available_gb * 0.8
        limit_source = "auto-calculated"

    # Calculate maximum workers based on memory
    memory_per_worker_gb = estimated_memory_per_worker_mb / 1024
    max_workers_by_memory = int(memory_limit_gb / memory_per_worker_gb)

    # Ensure at least min_workers
    safe_workers = max(min_workers, min(requested_workers, max_workers_by_memory))

    # Generate warning if we had to reduce workers
    warning = ""
    if safe_workers < requested_workers:
        reduced_by = requested_workers - safe_workers
        warning = (
            f"⚠️  Reduced workers from {requested_workers} to {safe_workers} "
            f"due to memory constraints\n"
            f"   Available: {available_gb:.1f}GB / {total_gb:.1f}GB total, "
            f"Limit ({limit_source}): {memory_limit_gb:.1f}GB, "
            f"Est. per worker: {estimated_memory_per_worker_mb}MB"
        )

    return safe_workers, warning


def log_memory_stats(prefix: str = ""):
    """Log current memory statistics.

    Args:
        prefix: Optional prefix for log message
    """
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    used_gb = mem.used / (1024 ** 3)

    log_msg = f"{prefix}Memory: {used_gb:.1f}GB used / {total_gb:.1f}GB total ({mem.percent:.1f}%), {available_gb:.1f}GB available"

    if mem.percent > 90:
        logger.warning(log_msg)
    else:
        logger.info(log_msg)

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
            # MPS (Apple Silicon) - unified memory architecture
            # Unlike CUDA, MPS shares memory with system RAM and macOS can page/swap.
            # The "allocated" memory is NOT permanently unavailable - OS can reclaim it.
            # Therefore, we use system available memory instead of subtracting allocations.
            mem = psutil.virtual_memory()

            # Use system available memory as proxy for GPU availability
            # MPS can use up to ~70-80% of total RAM, but respect system availability
            estimated_total = int(mem.total * 0.7)  # 70% of RAM theoretical max for GPU

            # For available: use whichever is smaller:
            # 1. System available memory (respects other processes)
            # 2. Estimated total (respects 70% cap)
            system_available = mem.available
            available = min(system_available, estimated_total)

            return (available, estimated_total, "mps")
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
    pipeline_overhead_gb: float = 0.3,
    safety_factor: float = 0.6,
    device_type: str = "cuda"
) -> int:
    """Estimate safe batch size with model-aware memory calculations.

    IMPORTANT: For MPS (Apple Silicon), this uses a simplified heuristic because:
    - Unified memory architecture makes precise calculation unreliable
    - Model weights are allocated from same pool as available memory
    - macOS can page/swap as needed

    For CUDA, uses detailed memory model accounting for model weights and activations.

    Args:
        model_name: Model identifier (stella, jina-code, bge-large, bge-base, bge-small)
        available_gpu_bytes: Available GPU memory in bytes
        pipeline_overhead_gb: Memory reserved for minimal GPU overhead (default: 0.3GB)
        safety_factor: Fraction of memory to use after overhead (default: 0.6 = 60%)
        device_type: "cuda" or "mps" (default: "cuda")

    Returns:
        Recommended safe batch size (8-1024)

    Example:
        >>> # 10GB available, stella model, CUDA
        >>> estimate_safe_batch_size_v2("stella", 10 * 1024**3, device_type="cuda")
        >>> # Returns: ~422 (detailed calculation)

        >>> # 10GB available, stella model, MPS
        >>> estimate_safe_batch_size_v2("stella", 10 * 1024**3, device_type="mps")
        >>> # Returns: 512 (heuristic: use optimal batch size if memory seems adequate)
    """
    # MPS (Apple Silicon): Use simplified heuristic
    # Precise calculation is unreliable due to unified memory architecture
    if device_type == "mps":
        available_gb = available_gpu_bytes / (1024 ** 3)

        # Optimal batch sizes from empirical testing (RDR-013, arcaneum-i7oa)
        # Larger batches = fewer kernel launches = better GPU utilization
        # Values tuned for sustained GPU inference without OOM
        OPTIMAL_BATCH_SIZES = {
            'stella': 512,
            'jina': 512,
            'jina-code': 512,
            'jina-code-0.5b': 256,   # 500M params, moderate batches
            'jina-code-1.5b': 128,   # 1.5B params, smaller batches
            'codesage-large': 256,   # ~400M params
            'nomic-code': 64,        # 7B params, very large model
            'bge-large': 512,
            'bge-base': 512,
            'bge-small': 512,
        }

        # Minimum memory requirements (model weights + reasonable batch headroom)
        MIN_MEMORY_GB = {
            'stella': 4.0,           # 2.5GB model + 1.5GB for batching
            'jina': 2.0,             # 0.5GB model + 1.5GB for batching
            'jina-code': 2.0,
            'jina-code-0.5b': 3.0,   # 1.5GB model + 1.5GB for batching
            'jina-code-1.5b': 6.0,   # 4GB model + 2GB for batching
            'codesage-large': 3.0,   # 1.5GB model + 1.5GB for batching
            'nomic-code': 16.0,      # ~14GB model + 2GB for batching (7B params)
            'bge-large': 2.5,
            'bge-base': 2.0,
            'bge-small': 1.5,
        }

        optimal = OPTIMAL_BATCH_SIZES.get(model_name, 512)
        min_required = MIN_MEMORY_GB.get(model_name, 4.0)

        if available_gb >= min_required:
            # Have enough memory, use optimal batch size
            logger.debug(f"MPS: {available_gb:.1f}GB available >= {min_required}GB required, using optimal batch_size={optimal}")
            return optimal
        else:
            # Low memory, scale down batch size proportionally
            scale_factor = available_gb / min_required
            scaled_batch = max(8, int(optimal * scale_factor))
            logger.warning(f"MPS: Low memory ({available_gb:.1f}GB < {min_required}GB), scaling batch_size to {scaled_batch}")
            return min(scaled_batch, 1024)

    # CUDA: Use detailed memory model
    # Model weights (one-time GPU memory allocation)
    MODEL_WEIGHTS_GB = {
        'stella': 2.5,            # 1.5B parameters
        'jina': 0.5,              # ~110M parameters
        'jina-code': 0.5,
        'jina-code-0.5b': 1.5,    # 500M parameters
        'jina-code-1.5b': 4.0,    # 1.5B parameters
        'codesage-large': 1.5,    # ~400M parameters
        'nomic-code': 14.0,       # 7B parameters
        'bge-large': 0.8,
        'bge-base': 0.5,
        'bge-small': 0.3,
    }

    # Activation memory per batch item (empirical measurements)
    ACTIVATION_MB_PER_ITEM = {
        'stella': 8.0,            # 1024D output, large model
        'jina': 5.0,              # 768D output
        'jina-code': 5.0,
        'jina-code-0.5b': 6.0,    # 896D output
        'jina-code-1.5b': 10.0,   # 1536D output, large model
        'codesage-large': 8.0,    # 1024D output
        'nomic-code': 20.0,       # 3584D output, very large
        'bge-large': 8.0,         # 1024D output
        'bge-base': 5.0,          # 768D output
        'bge-small': 3.0,         # 384D output
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
